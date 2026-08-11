#!/usr/bin/env python3
"""Summarize the full-data GCLS optimization screen at prompt level."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
from typing import Any

from summarize_global_direct_v1 import (
    accepted_prefix,
    calibrated_prompt_eal,
    prompt_cluster_bootstrap,
)


EAL_FIELD = "mean_accepted_draft_tokens_prompt_balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    return parser.parse_args()


def uncalibrated_prompt_eal(
    report: dict[str, Any], method: str
) -> dict[str, float]:
    if method not in {"base", "direct"}:
        raise ValueError(f"unsupported uncalibrated method: {method}")
    field = (
        "base_position_correct"
        if method == "base"
        else "direct_position_correct"
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in report["final_validation"]["examples"]:
        grouped[str(record["sample_id"])].append(
            accepted_prefix(record[field])
        )
    return {
        sample_id: statistics.fmean(values)
        for sample_id, values in grouped.items()
    }


def paired(
    left: dict[str, float],
    right: dict[str, float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if left.keys() != right.keys():
        raise RuntimeError("paired reports contain different validation prompts")
    return prompt_cluster_bootstrap(
        {key: left[key] - right[key] for key in left},
        repetitions=repetitions,
        seed=seed,
    )


def run_key(report: dict[str, Any]) -> tuple[int, int, int, str]:
    config = report["config"]
    return (
        int(config["model_dim"]),
        int(config["num_layers"]),
        int(config["epochs"]),
        str(report["scope"]),
    )


def label(key: tuple[int, int, int, str]) -> str:
    dim, layers, epochs, scope = key
    return f"d{dim}_l{layers}_e{epochs}_{scope}"


def main() -> None:
    args = parse_args()
    metric_paths = sorted(args.input.glob("*/metrics.json"))
    reports: dict[tuple[int, int, int, str], tuple[Path, dict[str, Any]]] = {}
    for path in metric_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        key = run_key(report)
        if key in reports:
            raise RuntimeError(f"duplicate optimization run {label(key)}")
        reports[key] = (path, report)
    expected = {
        (dim, layers, epochs, scope)
        for dim, layers in ((64, 1), (128, 2))
        for epochs in (6, 9)
        for scope in ("local", "global")
    }
    if set(reports) != expected:
        raise RuntimeError(
            f"optimization matrix mismatch: expected {sorted(expected)}, "
            f"found {sorted(reports)}"
        )

    source_signatures = {
        (
            report["provenance"]["trainer_sha256"],
            report["provenance"]["head_source_sha256"],
            report["provenance"]["data_metadata_sha256"],
        )
        for _, report in reports.values()
    }
    source_stable = all(
        report["provenance"]["trainer_sha256"]
        == report["provenance"]["trainer_sha256_at_end"]
        and report["provenance"]["head_source_sha256"]
        == report["provenance"]["head_source_sha256_at_end"]
        for _, report in reports.values()
    )
    if len(source_signatures) != 1 or not source_stable:
        raise RuntimeError("source/data hashes differ or changed during runs")

    base_predictions: dict[str, float] | None = None
    raw_predictions: dict[tuple[int, int, int, str], dict[str, float]] = {}
    calibrated_predictions: dict[
        tuple[int, int, int, str], dict[str, float]
    ] = {}
    runs: dict[str, Any] = {}
    for key in sorted(reports):
        path, report = reports[key]
        evaluation = report["final_validation"]
        current_base = uncalibrated_prompt_eal(report, "base")
        if base_predictions is None:
            base_predictions = current_base
        elif current_base != base_predictions:
            raise RuntimeError("base validation outcomes differ across runs")
        raw_predictions[key] = uncalibrated_prompt_eal(report, "direct")
        calibrated_predictions[key] = calibrated_prompt_eal(report)
        base_eal = float(evaluation["base"][EAL_FIELD])
        raw_eal = float(evaluation["direct"][EAL_FIELD])
        calibrated_eal = float(evaluation["calibrated"][EAL_FIELD])
        runs[label(key)] = {
            "metrics": str(path.resolve()),
            "parameter_count": int(report["parameter_count"]),
            "selected_epoch": int(report["selected_epoch"]),
            "available_epochs": int(report["config"]["epochs"]),
            "base_eal": base_eal,
            "raw_eal": raw_eal,
            "raw_delta_vs_base": raw_eal - base_eal,
            "calibrated_eal": calibrated_eal,
            "calibrated_delta_vs_base": calibrated_eal - base_eal,
            "calibration_threshold": evaluation["calibrated"]["threshold"],
            "calibrated_first_token_delta": (
                float(evaluation["calibrated"]["first_token_accuracy"])
                - float(evaluation["base"]["first_token_accuracy"])
            ),
            "train_non_top1_accuracy": float(
                report["final_train_diagnostic"]["candidate_classification"]
                ["non_top1_accuracy"]
            ),
            "validation_non_top1_accuracy": float(
                evaluation["candidate_classification"]["non_top1_accuracy"]
            ),
        }
    assert base_predictions is not None

    comparisons: dict[str, Any] = {}
    comparison_index = 0

    def add_comparison(
        name: str,
        left_key: tuple[int, int, int, str],
        right_key: tuple[int, int, int, str],
    ) -> None:
        nonlocal comparison_index
        comparisons[name] = {
            "left": label(left_key),
            "right": label(right_key),
            "raw": paired(
                raw_predictions[left_key],
                raw_predictions[right_key],
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed + 2 * comparison_index,
            ),
            "calibrated": paired(
                calibrated_predictions[left_key],
                calibrated_predictions[right_key],
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed + 2 * comparison_index + 1,
            ),
        }
        comparison_index += 1

    for dim, layers in ((64, 1), (128, 2)):
        for epochs in (6, 9):
            add_comparison(
                f"d{dim}_e{epochs}_global_minus_local",
                (dim, layers, epochs, "global"),
                (dim, layers, epochs, "local"),
            )
        for scope in ("local", "global"):
            add_comparison(
                f"d{dim}_{scope}_e9_minus_e6",
                (dim, layers, 9, scope),
                (dim, layers, 6, scope),
            )
    for epochs in (6, 9):
        for scope in ("local", "global"):
            add_comparison(
                f"e{epochs}_{scope}_d128_minus_d64",
                (128, 2, epochs, scope),
                (64, 1, epochs, scope),
            )

    output = {
        "input": str(args.input.resolve()),
        "protocol": {
            "evidence_tier": "development",
            "validation_prompts": len(base_predictions),
            "bootstrap_unit": "prompt; all anchors kept together",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "source_and_data_hashes_matched": True,
            "source_stable_during_runs": True,
            "warning": (
                "Checkpoints and calibration margins were selected on this "
                "development split; intervals are not sealed-test intervals."
            ),
        },
        "runs": runs,
        "paired_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(
        args.output.suffix + f".{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
