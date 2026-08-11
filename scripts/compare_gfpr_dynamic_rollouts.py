#!/usr/bin/env python3
"""Compare two complete GFPR trajectories without aligning their blocks.

An adapted policy changes accepted lengths, hence subsequent r+1 anchors and
cycle counts.  The only valid pairing is at prompt level; zipping cached blocks
would silently evaluate the new head on the released policy's trajectory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch

from train_gfpr_head import load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--minimum-ratio", type=float, default=1.15)
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument(
        "--allow-identical-policy-smoke",
        action="store_true",
        help="Only for comparator self-tests; claim-bearing comparisons reject this.",
    )
    return parser.parse_args()


def _trajectory_metrics(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(record)
    result: dict[str, dict[str, float]] = {}
    for sample_id, prompt_records in grouped.items():
        prompt_records.sort(key=lambda row: int(row["anchor_offset"]))
        for previous, current in zip(prompt_records, prompt_records[1:]):
            if int(previous["next_anchor_offset"]) != int(current["anchor_offset"]):
                raise ValueError(f"broken dynamic chain for {sample_id}")
        accepted = [int(record["accepted_length"]) for record in prompt_records]
        result[sample_id] = {
            "eal": sum(accepted) / len(accepted),
            "cycles": float(len(accepted)),
            "accepted_tokens": float(sum(accepted)),
            "committed_advances": float(sum(value + 1 for value in accepted)),
            "full_block_fraction": sum(value == 16 for value in accepted)
            / len(accepted),
        }
    return result


def _paired(
    prompt_ids: list[str],
    baseline: list[float],
    adapted: list[float],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    base = torch.tensor(baseline, dtype=torch.float64)
    current = torch.tensor(adapted, dtype=torch.float64)
    delta = current - base
    generator = torch.Generator().manual_seed(0)
    draw = torch.randint(
        0,
        len(prompt_ids),
        (bootstrap_samples, len(prompt_ids)),
        generator=generator,
    )
    boot = delta[draw].mean(dim=-1)
    interval = torch.quantile(
        boot, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    gained = float(delta.clamp_min(0).sum())
    lost = float((-delta.clamp_max(0)).sum())
    baseline_mean = float(base.mean())
    adapted_mean = float(current.mean())
    return {
        "prompts": len(prompt_ids),
        "baseline_prompt_mean": baseline_mean,
        "adapted_prompt_mean": adapted_mean,
        "adapted_to_baseline_ratio": adapted_mean / baseline_mean,
        "paired_delta": float(delta.mean()),
        "paired_bootstrap_95_interval": [float(interval[0]), float(interval[1])],
        "gained_prompt_units": gained,
        "lost_prompt_units": lost,
        "lost_to_gained_ratio": lost / gained if gained > 0 else float("inf"),
        "harmful_prompt_fraction": float(delta.lt(0).float().mean()),
        "improved_prompt_fraction": float(delta.gt(0).float().mean()),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1 or args.minimum_ratio <= 1:
        raise ValueError("invalid bootstrap count or minimum ratio")
    baseline_metadata, baseline_records = load_records(args.baseline, args.split)
    adapted_metadata, adapted_records = load_records(args.adapted, args.split)
    for name, metadata in (
        ("baseline", baseline_metadata),
        ("adapted", adapted_metadata),
    ):
        if metadata.get("mode") != "dynamic":
            raise ValueError(f"{name} collection is not a dynamic trajectory")
    for field in ("source_canonical", "target", "domino_draft"):
        if baseline_metadata.get(field) != adapted_metadata.get(field):
            raise ValueError(f"trajectory metadata differs at {field}")
    if not args.allow_identical_policy_smoke:
        if args.baseline.resolve() == args.adapted.resolve():
            raise ValueError("baseline and adapted collections must differ")
        if (
            baseline_metadata.get("policy_version")
            == adapted_metadata.get("policy_version")
        ):
            raise ValueError("baseline and adapted policy versions must differ")
        adapted_checkpoint = adapted_metadata.get("adaptation")
        if adapted_checkpoint is None:
            raise ValueError("adapted trajectory does not name an adaptation")
        if not Path(adapted_checkpoint).is_file():
            raise ValueError(
                f"adapted trajectory checkpoint is missing: {adapted_checkpoint}"
            )
    baseline = _trajectory_metrics(baseline_records)
    adapted = _trajectory_metrics(adapted_records)
    if baseline.keys() != adapted.keys():
        missing = sorted(baseline.keys() ^ adapted.keys())[:10]
        raise ValueError(f"trajectory prompt sets differ: {missing}")
    prompt_ids = sorted(baseline)
    metric_reports = {
        metric: _paired(
            prompt_ids,
            [baseline[key][metric] for key in prompt_ids],
            [adapted[key][metric] for key in prompt_ids],
            bootstrap_samples=args.bootstrap_samples,
        )
        for metric in (
            "eal",
            "cycles",
            "accepted_tokens",
            "committed_advances",
            "full_block_fraction",
        )
    }
    eal = metric_reports["eal"]
    gate = {
        "dynamic_ratio_at_least_target": (
            float(eal["adapted_to_baseline_ratio"]) >= args.minimum_ratio
        ),
        "paired_bootstrap_lower_above_zero": (
            float(eal["paired_bootstrap_95_interval"][0]) > 0
        ),
        "lost_to_gained_at_most_half": (
            float(eal["lost_to_gained_ratio"]) <= 0.5
        ),
        "harmful_prompts_at_most_20pct": (
            float(eal["harmful_prompt_fraction"]) <= 0.2
        ),
    }
    gate["passed"] = all(gate.values())
    report = {
        "format": "gfpr_dynamic_policy_comparison_v1",
        "split": args.split,
        "baseline_collection": str(args.baseline.resolve()),
        "adapted_collection": str(args.adapted.resolve()),
        "baseline_policy_version": baseline_metadata["policy_version"],
        "adapted_policy_version": adapted_metadata["policy_version"],
        "baseline_adaptation": baseline_metadata.get("adaptation"),
        "adapted_adaptation": adapted_metadata.get("adaptation"),
        "metrics": metric_reports,
        "dynamic_success_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.fail_on_gate and not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
