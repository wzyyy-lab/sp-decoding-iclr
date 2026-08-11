#!/usr/bin/env python3
"""Aggregate matched GCLS-v1 scopes without block-level pseudoreplication."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import statistics
from typing import Any


SCOPES = ("local", "causal", "global")
ACCEPTED_FIELD = "mean_accepted_draft_tokens_prompt_balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Run root; repeat when the Slurm matrix spans multiple jobs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--train-prompts",
        type=int,
        help=(
            "Only include runs with this recorded training-prompt count. "
            "Useful when an input root also contains learning-curve runs."
        ),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    parser.add_argument("--minimum-global-base-delta", type=float, default=0.0)
    parser.add_argument(
        "--minimum-global-local-delta", type=float, default=0.15
    )
    parser.add_argument("--maximum-first-token-drop", type=float, default=0.001)
    parser.add_argument("--maximum-domain-drop", type=float, default=0.05)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "values": values,
    }


def accepted_prefix(correct: list[bool]) -> int:
    for index, value in enumerate(correct):
        if not value:
            return index
    return len(correct)


def calibrated_prompt_eal(report: dict[str, Any]) -> dict[str, float]:
    evaluation = report["final_validation"]
    threshold = evaluation["calibrated"]["threshold"]
    if threshold == "base_only":
        threshold_value = float("inf")
    else:
        threshold_value = float(threshold)
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in evaluation["examples"]:
        path = record["candidate_path_indices"]["direct"]
        margins = record["direct_margin_over_base"]
        correct = [
            (
                direct_correct
                if candidate != 0 and margin >= threshold_value
                else base_correct
            )
            for candidate, margin, base_correct, direct_correct in zip(
                path,
                margins,
                record["base_position_correct"],
                record["direct_position_correct"],
                strict=True,
            )
        ]
        grouped[str(record["sample_id"])].append(
            accepted_prefix(correct)
        )
    return {
        sample_id: statistics.fmean(values)
        for sample_id, values in grouped.items()
    }


def prompt_cluster_bootstrap(
    prompt_deltas: dict[str, float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    values = list(prompt_deltas.values())
    generator = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        draws.append(
            statistics.fmean(
                values[generator.randrange(len(values))]
                for _ in values
            )
        )
    draws.sort()
    lower = draws[int(0.025 * (repetitions - 1))]
    upper = draws[int(0.975 * (repetitions - 1))]
    return {
        "estimate": statistics.fmean(values),
        "ci95": [lower, upper],
        "prompts": len(values),
        "repetitions": repetitions,
        "unit": "prompt",
    }


def compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "output",
        "scope",
        "seed",
        "max_train_prompts",
        "train_subset_seed",
    }
    return {
        key: value for key, value in config.items() if key not in ignored
    }


def main() -> None:
    args = parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    metric_paths = sorted(
        {
            path.resolve()
            for root in args.input
            for path in root.glob("*/metrics.json")
        }
    )
    reports = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in metric_paths
    ]
    if args.train_prompts is not None:
        reports = [
            (path, report)
            for path, report in reports
            if int(report["train_prompts"]) == args.train_prompts
        ]
    expected_keys = {
        (scope, seed)
        for scope in SCOPES
        for seed in range(args.seeds)
    }
    by_key: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for path, report in reports:
        key = (
            str(report["scope"]),
            int(report["config"]["seed"]),
        )
        if key in by_key:
            raise RuntimeError(f"duplicate run for {key}: {path}")
        by_key[key] = (path, report)
    if set(by_key) != expected_keys:
        raise RuntimeError(
            "run matrix mismatch: expected "
            f"{sorted(expected_keys)}, found {sorted(by_key)}"
        )

    configs = {
        json.dumps(
            compatible_config(report["config"]), sort_keys=True
        )
        for _, report in by_key.values()
    }
    source_signatures = {
        (
            report["provenance"]["trainer_sha256"],
            report["provenance"]["head_source_sha256"],
            report["provenance"]["data_metadata_sha256"],
        )
        for _, report in by_key.values()
    }
    source_stable = all(
        report["provenance"]["trainer_sha256"]
        == report["provenance"]["trainer_sha256_at_end"]
        and report["provenance"]["head_source_sha256"]
        == report["provenance"]["head_source_sha256_at_end"]
        for _, report in by_key.values()
    )
    if len(configs) != 1:
        raise RuntimeError("matched runs differ beyond scope and seed")
    if len(source_signatures) != 1 or not source_stable:
        raise RuntimeError("source/data hashes differ or changed during runs")

    prompt_predictions = {
        key: calibrated_prompt_eal(report)
        for key, (_, report) in by_key.items()
    }
    aggregate: dict[str, Any] = {
        "inputs": [str(path.resolve()) for path in args.input],
        "runs": {
            f"{scope}/seed{seed}": str(by_key[(scope, seed)][0])
            for scope, seed in sorted(by_key)
        },
        "protocol": {
            "matched_config": True,
            "train_prompts_filter": args.train_prompts,
            "source_and_data_hashes_matched": True,
            "source_stable_during_runs": True,
            "gate_evaluated": any(
                report["final_gate"] is not None
                for _, report in by_key.values()
            ),
            "metric": ACCEPTED_FIELD,
            "bootstrap_unit": "prompt",
        },
        "scopes": {},
        "paired": {},
    }
    for scope in SCOPES:
        scope_reports = [
            by_key[(scope, seed)][1] for seed in range(args.seeds)
        ]
        base = [
            float(report["final_validation"]["base"][ACCEPTED_FIELD])
            for report in scope_reports
        ]
        raw = [
            float(report["final_validation"]["direct"][ACCEPTED_FIELD])
            for report in scope_reports
        ]
        calibrated = [
            float(
                report["final_validation"]["calibrated"][ACCEPTED_FIELD]
            )
            for report in scope_reports
        ]
        base_first = [
            float(
                report["final_validation"]["base"][
                    "first_token_accuracy"
                ]
            )
            for report in scope_reports
        ]
        calibrated_first = [
            float(
                report["final_validation"]["calibrated"][
                    "first_token_accuracy"
                ]
            )
            for report in scope_reports
        ]
        domains = sorted(
            scope_reports[0]["final_validation"]["by_domain"]
        )
        aggregate["scopes"][scope] = {
            "selected_epochs": [
                int(report["selected_epoch"])
                for report in scope_reports
            ],
            "base_eal": mean_std(base),
            "raw_delta_vs_base": mean_std(
                [value - baseline for value, baseline in zip(raw, base)]
            ),
            "calibrated_delta_vs_base": mean_std(
                [
                    value - baseline
                    for value, baseline in zip(calibrated, base)
                ]
            ),
            "calibrated_first_token_delta": mean_std(
                [
                    value - baseline
                    for value, baseline in zip(
                        calibrated_first, base_first
                    )
                ]
            ),
            "calibrated_domain_delta": {
                domain: mean_std(
                    [
                        float(
                            report["final_validation"]["calibrated"][
                                "by_domain"
                            ][domain]["mean_accepted_draft_tokens"]
                        )
                        - float(
                            report["final_validation"]["by_domain"][
                                domain
                            ]["base"]["mean_accepted_draft_tokens"]
                        )
                        for report in scope_reports
                    ]
                )
                for domain in domains
            },
            "calibration_thresholds": [
                report["final_validation"]["calibrated"]["threshold"]
                for report in scope_reports
            ],
        }

    for comparator in ("local", "causal"):
        seed_deltas = []
        per_seed_prompt_delta = []
        for seed in range(args.seeds):
            global_values = prompt_predictions[("global", seed)]
            other_values = prompt_predictions[(comparator, seed)]
            if global_values.keys() != other_values.keys():
                raise RuntimeError("validation prompts differ across runs")
            deltas = {
                prompt: global_values[prompt] - other_values[prompt]
                for prompt in global_values
            }
            per_seed_prompt_delta.append(deltas)
            seed_deltas.append(statistics.fmean(deltas.values()))
        averaged_prompt_delta = {
            prompt: statistics.fmean(
                values[prompt] for values in per_seed_prompt_delta
            )
            for prompt in per_seed_prompt_delta[0]
        }
        aggregate["paired"][f"global_minus_{comparator}"] = {
            "by_seed": mean_std(seed_deltas),
            "prompt_cluster_bootstrap": prompt_cluster_bootstrap(
                averaged_prompt_delta,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed,
            ),
        }

    global_summary = aggregate["scopes"]["global"]
    global_local = aggregate["paired"]["global_minus_local"]["by_seed"]
    checks = {
        "global_vs_base_mean": (
            global_summary["calibrated_delta_vs_base"]["mean"]
            >= args.minimum_global_base_delta
        ),
        "global_vs_base_every_seed": (
            global_summary["calibrated_delta_vs_base"]["minimum"] > 0.0
        ),
        "global_vs_local_mean": (
            global_local["mean"] >= args.minimum_global_local_delta
        ),
        "global_vs_local_every_seed": global_local["minimum"] > 0.0,
        "first_token_safety": (
            global_summary["calibrated_first_token_delta"]["minimum"]
            >= -args.maximum_first_token_drop
        ),
        "domain_safety": min(
            summary["minimum"]
            for summary in global_summary[
                "calibrated_domain_delta"
            ].values()
        )
        >= -args.maximum_domain_drop,
    }
    aggregate["development_gate"] = {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_global_base_delta": args.minimum_global_base_delta,
            "minimum_global_local_delta": args.minimum_global_local_delta,
            "maximum_first_token_drop": args.maximum_first_token_drop,
            "maximum_domain_drop": args.maximum_domain_drop,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(
        args.output.suffix + f".{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
