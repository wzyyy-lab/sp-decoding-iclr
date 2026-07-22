#!/usr/bin/env python3
"""Summarize multi-seed development head runs without treating them as formal evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-delta", type=float, default=0.2)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(
        sum((value - center) ** 2 for value in values) / (len(values) - 1)
    )


def metric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": mean(values),
        "sample_std": sample_std(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    args = parse_args()
    metrics_paths = sorted(args.input.glob("*/metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"no */metrics.json files below {args.input}")
    runs = []
    for path in metrics_paths:
        report = json.loads(path.read_text())
        if report.get("evidence_tier") != "development":
            raise ValueError(f"unexpected evidence tier in {path}")
        runs.append(
            {
                "path": str(path.resolve()),
                "head_type": report["head_type"],
                "normalization": report["normalization"],
                "seed": int(report.get("seed", -1)),
                "selected_epoch": int(report["selected_epoch"]),
                "test": report["final_test"],
                "validation": report["final_validation"],
            }
        )
    # Older reports did not expose seed at the top level. The run directory is
    # authoritative for this controlled probe and is validated here.
    for run in runs:
        if run["seed"] < 0:
            name = Path(run["path"]).parent.name
            marker = "_seed"
            if marker not in name:
                raise ValueError(f"cannot recover seed from {name}")
            run["seed"] = int(name.rsplit(marker, 1)[1])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[f"{run['head_type']}/{run['normalization']}"].append(run)

    summaries: dict[str, Any] = {}
    for key, group in sorted(grouped.items()):
        group.sort(key=lambda item: item["seed"])
        decoder_names = [
            "base",
            "local",
            "local_map",
            "local_survival",
            "global_map",
            "global_survival",
        ]
        decoder_metrics = {}
        for decoder in decoder_names:
            decoder_metrics[decoder] = metric_summary(
                [
                    float(run["test"][decoder]["mean_accepted_draft_tokens"])
                    for run in group
                ]
            )
        disagreement_metrics = {}
        for pair_name in group[0]["test"].get("decoder_disagreement", {}):
            disagreement_metrics[pair_name] = {
                "path_disagreement_fraction": metric_summary(
                    [
                        float(
                            run["test"]["decoder_disagreement"][pair_name][
                                "path_disagreement_fraction"
                            ]
                        )
                        for run in group
                    ]
                ),
                "first_token_disagreement_fraction": metric_summary(
                    [
                        float(
                            run["test"]["decoder_disagreement"][pair_name][
                                "first_token_disagreement_fraction"
                            ]
                        )
                        for run in group
                    ]
                ),
            }
        global_vs_map = [
            float(run["test"]["global_survival"]["mean_accepted_draft_tokens"])
            - float(run["test"]["global_map"]["mean_accepted_draft_tokens"])
            for run in group
        ]
        global_vs_local_survival = [
            float(run["test"]["global_survival"]["mean_accepted_draft_tokens"])
            - float(run["test"]["local_survival"]["mean_accepted_draft_tokens"])
            for run in group
        ]
        global_vs_base = [
            float(run["test"]["global_survival"]["mean_accepted_draft_tokens"])
            - float(run["test"]["base"]["mean_accepted_draft_tokens"])
            for run in group
        ]
        summaries[key] = {
            "seeds": [run["seed"] for run in group],
            "selected_epochs": [run["selected_epoch"] for run in group],
            "test_eal": decoder_metrics,
            "decoder_disagreement": disagreement_metrics,
            "deltas": {
                "global_survival_minus_global_map": metric_summary(global_vs_map),
                "global_survival_minus_local_survival": metric_summary(
                    global_vs_local_survival
                ),
                "global_survival_minus_base": metric_summary(global_vs_base),
            },
        }

    primary = summaries.get("no_mixer/absorbing_crf")
    if primary is None:
        raise ValueError("probe is missing no_mixer/absorbing_crf runs")
    primary_delta = primary["deltas"]["global_survival_minus_global_map"]
    probe_pass = bool(
        primary_delta["mean"] >= args.minimum_delta
        and primary_delta["minimum"] > 0.0
    )
    report = {
        "evidence_tier": "development_probe_only",
        "formal_claim_allowed": False,
        "reason": (
            "The source collection has only 96 benchmark prompts and a "
            "12-prompt test split; this probe may gate data scaling but cannot "
            "support a paper claim."
        ),
        "input": str(args.input.resolve()),
        "runs": runs,
        "summaries": summaries,
        "probe_gate": {
            "minimum_delta": args.minimum_delta,
            "criterion": (
                "mean(global_survival - global_map) >= minimum_delta and "
                "the delta is positive for every seed"
            ),
            "pass": probe_pass,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"probe_gate": report["probe_gate"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
