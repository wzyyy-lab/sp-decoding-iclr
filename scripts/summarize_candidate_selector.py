#!/usr/bin/env python3
"""Aggregate matched local/global candidate-selector development runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--minimum-global-base-delta", type=float, default=0.2)
    parser.add_argument("--minimum-global-local-delta", type=float, default=0.15)
    parser.add_argument("--maximum-first-token-drop", type=float, default=0.001)
    parser.add_argument("--maximum-domain-drop", type=float, default=0.05)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def metric(
    report: dict[str, Any],
    split: str,
    method: str,
    field: str,
) -> float:
    section = report[f"final_{split}"]
    if section is None:
        raise RuntimeError(f"{split} was not evaluated")
    return float(section[method][field])


def main() -> None:
    args = parse_args()
    paths = sorted(args.input.glob("*_seed*/metrics.json"))
    reports = [json.loads(path.read_text()) for path in paths]
    expected = 2 * args.seeds
    if len(reports) != expected:
        raise RuntimeError(
            f"expected {expected} metrics files, found {len(reports)}"
        )
    by_key = {
        (str(report["scope"]), int(report["config"]["seed"])): report
        for report in reports
    }
    expected_keys = {
        (scope, seed)
        for scope in ["local", "global"]
        for seed in range(args.seeds)
    }
    if set(by_key) != expected_keys:
        raise RuntimeError(
            f"run matrix differs from expectation: {sorted(by_key)}"
        )
    parameter_counts = {
        int(report["parameter_count"]) for report in reports
    }
    if len(parameter_counts) != 1:
        raise RuntimeError(
            f"local/global parameter counts are not matched: {parameter_counts}"
        )

    field = "mean_accepted_draft_tokens_prompt_balanced"
    aggregate: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "runs": [str(path.resolve()) for path in paths],
        "parameter_count": parameter_counts.pop(),
        "parameter_matched": True,
        "splits": {},
    }
    for split in ["validation", "gate"]:
        aggregate["splits"][split] = {}
        for scope in ["local", "global"]:
            scope_reports = [
                by_key[(scope, seed)] for seed in range(args.seeds)
            ]
            base = [
                metric(report, split, "base", field)
                for report in scope_reports
            ]
            raw = [
                metric(report, split, "survival", field)
                for report in scope_reports
            ]
            keep = [
                metric(report, split, "keep_base", field)
                for report in scope_reports
            ]
            first_base = [
                metric(
                    report,
                    split,
                    "base",
                    "first_token_accuracy",
                )
                for report in scope_reports
            ]
            first_keep = [
                metric(
                    report,
                    split,
                    "keep_base",
                    "first_token_accuracy",
                )
                for report in scope_reports
            ]
            aggregate["splits"][split][scope] = {
                "base_eal": mean_std(base),
                "raw_survival_eal": mean_std(raw),
                "keep_base_eal": mean_std(keep),
                "raw_delta_vs_base": mean_std(
                    [value - baseline for value, baseline in zip(raw, base)]
                ),
                "keep_base_delta_vs_base": mean_std(
                    [value - baseline for value, baseline in zip(keep, base)]
                ),
                "keep_base_first_token_delta": mean_std(
                    [
                        value - baseline
                        for value, baseline in zip(first_keep, first_base)
                    ]
                ),
                "oracle_gap_recovered": mean_std(
                    [
                        float(
                            report[f"final_{split}"]["path_diagnostics"][
                                "survival"
                            ]["oracle_gap_recovered"]
                        )
                        for report in scope_reports
                    ]
                ),
                "first_miss_repair_rate": mean_std(
                    [
                        float(
                            report[f"final_{split}"]["path_diagnostics"][
                                "survival"
                            ]["first_miss_repair_rate_given_k"]
                        )
                        for report in scope_reports
                    ]
                ),
            }
        aggregate["splits"][split]["global_minus_local_keep_base"] = mean_std(
            [
                metric(by_key[("global", seed)], split, "keep_base", field)
                - metric(by_key[("local", seed)], split, "keep_base", field)
                for seed in range(args.seeds)
            ]
        )

    global_gate_reports = [
        by_key[("global", seed)] for seed in range(args.seeds)
    ]
    domain_deltas: dict[str, list[float]] = {}
    for report in global_gate_reports:
        gate = report["final_gate"]
        for domain, keep_metrics in gate["keep_base"]["by_domain"].items():
            domain_deltas.setdefault(domain, []).append(
                float(keep_metrics["mean_accepted_draft_tokens"])
                - float(
                    gate["by_domain"][domain]["base"][
                        "mean_accepted_draft_tokens"
                    ]
                )
            )
    aggregate["gate_global_keep_base_domain_delta"] = {
        domain: mean_std(values)
        for domain, values in sorted(domain_deltas.items())
    }

    gate_global = aggregate["splits"]["gate"]["global"]
    gate_global_local = aggregate["splits"]["gate"][
        "global_minus_local_keep_base"
    ]["mean"]
    minimum_domain_delta = min(
        summary["mean"]
        for summary in aggregate[
            "gate_global_keep_base_domain_delta"
        ].values()
    )
    checks = {
        "global_vs_base": (
            gate_global["keep_base_delta_vs_base"]["mean"]
            >= args.minimum_global_base_delta
        ),
        "global_vs_matched_local": (
            gate_global_local >= args.minimum_global_local_delta
        ),
        "first_token_no_harm": (
            gate_global["keep_base_first_token_delta"]["mean"]
            >= -args.maximum_first_token_drop
        ),
        "domain_no_harm": (
            minimum_domain_delta >= -args.maximum_domain_drop
        ),
    }
    aggregate["development_gate"] = {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "global_vs_base": gate_global[
                "keep_base_delta_vs_base"
            ]["mean"],
            "global_vs_matched_local": gate_global_local,
            "first_token_delta": gate_global[
                "keep_base_first_token_delta"
            ]["mean"],
            "minimum_domain_delta": minimum_domain_delta,
        },
        "thresholds": {
            "global_vs_base": args.minimum_global_base_delta,
            "global_vs_matched_local": args.minimum_global_local_delta,
            "first_token_delta": -args.maximum_first_token_drop,
            "minimum_domain_delta": -args.maximum_domain_drop,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
