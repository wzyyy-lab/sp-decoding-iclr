#!/usr/bin/env python3
"""Summarize the Open-PerfectBlend selector curve at the prompt level."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any, Iterable


EAL_FIELD = "mean_accepted_draft_tokens_prompt_balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--width-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_prefix(correct: Iterable[bool]) -> int:
    for index, value in enumerate(correct):
        if not value:
            return index
    return index + 1 if "index" in locals() else 0


def threshold_value(report: dict[str, Any]) -> float:
    threshold = report["final_validation"]["calibrated"]["threshold"]
    return math.inf if threshold == "base_only" else float(threshold)


def prompt_values(
    report: dict[str, Any], mode: str
) -> tuple[dict[str, float], dict[str, str]]:
    if mode not in {"base", "direct", "calibrated"}:
        raise ValueError(f"unknown mode: {mode}")
    threshold = threshold_value(report)
    grouped: dict[str, list[int]] = defaultdict(list)
    domains: dict[str, str] = {}
    for example in report["final_validation"]["examples"]:
        sample_id = str(example["sample_id"])
        domain = str(example["domain"])
        previous = domains.setdefault(sample_id, domain)
        if previous != domain:
            raise RuntimeError(f"domain mismatch for {sample_id}")
        if mode in {"base", "direct"}:
            value = int(example["accepted_draft_tokens"][mode])
        else:
            correct = []
            for candidate, margin, base_correct, direct_correct in zip(
                example["candidate_path_indices"]["direct"],
                example["direct_margin_over_base"],
                example["base_position_correct"],
                example["direct_position_correct"],
                strict=True,
            ):
                use_direct = candidate != 0 and float(margin) >= threshold
                correct.append(
                    bool(direct_correct) if use_direct else bool(base_correct)
                )
            value = accepted_prefix(correct)
        grouped[sample_id].append(value)
    values = {
        sample_id: statistics.fmean(blocks)
        for sample_id, blocks in grouped.items()
    }
    return values, domains


def paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    *,
    repetitions: int,
    seed: int,
    selected_ids: set[str] | None = None,
) -> dict[str, Any]:
    ids = sorted(left if selected_ids is None else selected_ids)
    if set(ids) - left.keys() or set(ids) - right.keys():
        raise RuntimeError("paired comparison has missing prompt IDs")
    deltas = [left[sample_id] - right[sample_id] for sample_id in ids]
    generator = random.Random(seed)
    draws = sorted(
        statistics.fmean(
            deltas[generator.randrange(len(deltas))] for _ in deltas
        )
        for _ in range(repetitions)
    )
    return {
        "estimate": statistics.fmean(deltas),
        "ci95": [
            draws[int(0.025 * (repetitions - 1))],
            draws[int(0.975 * (repetitions - 1))],
        ],
        "prompts": len(ids),
        "positive_prompts": sum(delta > 0 for delta in deltas),
        "negative_prompts": sum(delta < 0 for delta in deltas),
        "zero_prompts": sum(delta == 0 for delta in deltas),
        "unit": "prompt",
        "repetitions": repetitions,
    }


def run_key(report: dict[str, Any]) -> tuple[int, int, int, str]:
    config = report["config"]
    return (
        int(config["model_dim"]),
        int(config["num_layers"]),
        int(report["train_prompts"]),
        str(report["scope"]),
    )


def compact_run(report: dict[str, Any]) -> dict[str, Any]:
    validation = report["final_validation"]
    train = report["final_train_diagnostic"]
    base_eal = float(validation["base"][EAL_FIELD])
    direct_eal = float(validation["direct"][EAL_FIELD])
    calibrated_eal = float(validation["calibrated"][EAL_FIELD])
    domain_deltas = {}
    for domain, metrics in validation["by_domain"].items():
        domain_deltas[domain] = (
            float(validation["calibrated"]["by_domain"][domain][EAL_FIELD])
            - float(metrics["base"][EAL_FIELD])
        )
    return {
        "scope": report["scope"],
        "model_dim": report["config"]["model_dim"],
        "num_layers": report["config"]["num_layers"],
        "parameter_count": report["parameter_count"],
        "train_prompts": report["train_prompts"],
        "train_blocks": report["train_blocks"],
        "train_prompt_set_sha256": report["train_prompt_set_sha256"],
        "epochs": report["config"]["epochs"],
        "selected_epoch": report["selected_epoch"],
        "total_steps": report["total_steps"],
        "base_eal": base_eal,
        "direct_eal": direct_eal,
        "raw_delta_vs_base": direct_eal - base_eal,
        "calibrated_eal": calibrated_eal,
        "calibrated_delta_vs_base": calibrated_eal - base_eal,
        "calibration_threshold": report["calibration_threshold"],
        "calibrated_domain_deltas": domain_deltas,
        "calibrated_first_token_delta": (
            float(validation["calibrated"]["first_token_accuracy"])
            - float(validation["base"]["first_token_accuracy"])
        ),
        "raw_improved_blocks": validation["direct_diagnostics"][
            "improved_blocks"
        ],
        "raw_harmed_blocks": validation["direct_diagnostics"][
            "harmed_blocks"
        ],
        "calibrated_improved_blocks": validation["calibrated"][
            "diagnostics"
        ]["improved_blocks"],
        "calibrated_harmed_blocks": validation["calibrated"][
            "diagnostics"
        ]["harmed_blocks"],
        "validation_non_top1_accuracy": validation[
            "candidate_classification"
        ]["non_top1_accuracy"],
        "train_non_top1_accuracy": train["candidate_classification"][
            "non_top1_accuracy"
        ],
        "validation_oracle_gap_recovered_raw": validation[
            "direct_diagnostics"
        ]["oracle_gap_recovered"],
        "train_oracle_gap_recovered_raw": train["direct_diagnostics"][
            "oracle_gap_recovered"
        ],
        "selected_epoch_is_last": (
            report["selected_epoch"] == report["config"]["epochs"]
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    metric_paths = sorted(
        list(args.curve_root.glob("*/metrics.json"))
        + list(args.width_root.glob("*/metrics.json"))
    )
    if len(metric_paths) != 10:
        raise RuntimeError(f"expected 10 reports, found {len(metric_paths)}")
    reports = {
        run_key(report): (path, report)
        for path in metric_paths
        for report in [json.loads(path.read_text(encoding="utf-8"))]
    }
    if len(reports) != 10:
        raise RuntimeError("duplicate run keys")
    if any(report["final_gate"] is not None for _, report in reports.values()):
        raise RuntimeError("sealed gate was unexpectedly evaluated")
    if any(
        report["provenance"]["trainer_sha256"]
        != report["provenance"]["trainer_sha256_at_end"]
        or report["provenance"]["head_source_sha256"]
        != report["provenance"]["head_source_sha256_at_end"]
        for _, report in reports.values()
    ):
        raise RuntimeError("source changed during a run")

    vectors: dict[
        tuple[int, int, int, str], dict[str, dict[str, float]]
    ] = {}
    domains_by_key: dict[tuple[int, int, int, str], dict[str, str]] = {}
    base_reference: dict[str, float] | None = None
    for key, (_, report) in reports.items():
        vectors[key] = {}
        for mode in ("base", "direct", "calibrated"):
            values, domains = prompt_values(report, mode)
            vectors[key][mode] = values
            domains_by_key[key] = domains
            serialized = report["final_validation"][
                "calibrated" if mode == "calibrated" else mode
            ][EAL_FIELD]
            if abs(statistics.fmean(values.values()) - float(serialized)) > 1e-10:
                raise RuntimeError(f"prompt EAL reconstruction failed: {key} {mode}")
        if base_reference is None:
            base_reference = vectors[key]["base"]
        elif vectors[key]["base"] != base_reference:
            raise RuntimeError("base validation outcomes differ between runs")
    assert base_reference is not None

    comparisons: dict[str, Any] = {}
    seed_offset = 0

    def add_comparison(
        name: str,
        left_key: tuple[int, int, int, str],
        left_mode: str,
        right_key: tuple[int, int, int, str] | None,
        right_mode: str,
    ) -> None:
        nonlocal seed_offset
        left = vectors[left_key][left_mode]
        right = (
            base_reference
            if right_key is None
            else vectors[right_key][right_mode]
        )
        domains = domains_by_key[left_key]
        result = paired_bootstrap(
            left,
            right,
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed + seed_offset,
        )
        result["by_domain"] = {
            domain: paired_bootstrap(
                left,
                right,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed + seed_offset + index + 1,
                selected_ids={
                    sample_id
                    for sample_id, current in domains.items()
                    if current == domain
                },
            )
            for index, domain in enumerate(sorted(set(domains.values())))
        }
        comparisons[name] = result
        seed_offset += 10

    curve_sizes = [10000, 25000, 50000, 99356]
    for prompts in curve_sizes:
        local_key = (64, 1, prompts, "local")
        global_key = (64, 1, prompts, "global")
        add_comparison(
            f"d64_p{prompts}_global_raw_minus_base",
            global_key,
            "direct",
            None,
            "base",
        )
        add_comparison(
            f"d64_p{prompts}_global_raw_minus_local_raw",
            global_key,
            "direct",
            local_key,
            "direct",
        )
        add_comparison(
            f"d64_p{prompts}_global_calibrated_minus_base",
            global_key,
            "calibrated",
            None,
            "base",
        )
        add_comparison(
            f"d64_p{prompts}_global_calibrated_minus_local_calibrated",
            global_key,
            "calibrated",
            local_key,
            "calibrated",
        )
    add_comparison(
        "d64_global_full_minus_50k_calibrated",
        (64, 1, 99356, "global"),
        "calibrated",
        (64, 1, 50000, "global"),
        "calibrated",
    )
    add_comparison(
        "d128_full_global_minus_local_calibrated",
        (128, 2, 99356, "global"),
        "calibrated",
        (128, 2, 99356, "local"),
        "calibrated",
    )
    add_comparison(
        "full_global_d64_minus_d128_calibrated",
        (64, 1, 99356, "global"),
        "calibrated",
        (128, 2, 99356, "global"),
        "calibrated",
    )

    matched_train_hashes = {}
    for prompts in curve_sizes:
        local = reports[(64, 1, prompts, "local")][1]
        global_report = reports[(64, 1, prompts, "global")][1]
        equal = (
            local["train_prompt_set_sha256"]
            == global_report["train_prompt_set_sha256"]
        )
        if not equal:
            raise RuntimeError(f"local/global train prompts differ at {prompts}")
        matched_train_hashes[str(prompts)] = local[
            "train_prompt_set_sha256"
        ]

    output = {
        "schema_version": 1,
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in metric_paths
        ],
        "audit": {
            "reports": len(reports),
            "sealed_gate_evaluated": False,
            "source_stable_during_every_run": True,
            "base_validation_outcomes_identical": True,
            "matched_local_global_train_prompt_hashes": matched_train_hashes,
            "validation_prompts": len(base_reference),
        },
        "runs": {
            path.parent.name: compact_run(report)
            for path, report in reports.values()
        },
        "paired_prompt_bootstrap": comparisons,
        "bootstrap_note": (
            "Development-only intervals conditional on validation-selected "
            "checkpoints and margin thresholds; they are not sealed-test CIs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
