#!/usr/bin/env python3
"""Analyze exact-token parity and paired R056 SGLang throughput."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino", type=Path, nargs="+", required=True)
    parser.add_argument("--forest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--required-ratio", type=float, default=1.15)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("format") != "r056_sglang_run_v1":
        raise ValueError(f"unexpected R056 input format: {path}")
    return report


def indexed(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row["sample_id"]): row for row in report["records"]}
    if len(rows) != len(report["records"]):
        raise RuntimeError("duplicate sample IDs in R056 run")
    return rows


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
        if int(lhs) != int(rhs):
            return index
    return None


def parity(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref = indexed(reference)
    cand = indexed(candidate)
    if set(ref) != set(cand):
        raise RuntimeError("R056 parity runs use different prompt IDs")
    mismatches = []
    matching_tokens = 0
    total_tokens = 0
    for sample_id in sorted(ref):
        left = [int(value) for value in ref[sample_id]["output_ids"]]
        right = [int(value) for value in cand[sample_id]["output_ids"]]
        if len(left) != len(right):
            raise RuntimeError("R056 parity output lengths differ")
        mismatch = first_mismatch(left, right)
        prefix = len(left) if mismatch is None else mismatch
        matching_tokens += prefix
        total_tokens += len(left)
        if mismatch is not None:
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "domain": ref[sample_id]["domain"],
                    "first_mismatch": mismatch,
                    "reference_token": left[mismatch],
                    "candidate_token": right[mismatch],
                }
            )
    return {
        "matching_prompts": len(ref) - len(mismatches),
        "total_prompts": len(ref),
        "exact_prompt_fraction": (len(ref) - len(mismatches)) / len(ref),
        "clean_prefix_tokens": matching_tokens,
        "total_tokens": total_tokens,
        "clean_prefix_fraction": matching_tokens / total_tokens,
        "mismatches": mismatches,
    }


def median_rows(reports: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed_runs = [indexed(report) for report in reports]
    ids = set(indexed_runs[0])
    if any(set(run) != ids for run in indexed_runs[1:]):
        raise RuntimeError("R056 repeated runs use different prompt IDs")
    result = {}
    for sample_id in sorted(ids):
        first = indexed_runs[0][sample_id]
        result[sample_id] = {
            "sample_id": sample_id,
            "domain": first["domain"],
            "server_e2e_seconds": statistics.median(
                float(run[sample_id]["server_e2e_seconds"])
                for run in indexed_runs
            ),
            "client_seconds": statistics.median(
                float(run[sample_id]["client_seconds"]) for run in indexed_runs
            ),
            "completion_tokens": int(first["completion_tokens"]),
        }
    return result


def ratio_for_ids(
    ids: list[str],
    domino: dict[str, dict[str, Any]],
    forest: dict[str, dict[str, Any]],
    key: str,
) -> float:
    domino_seconds = sum(float(domino[sample_id][key]) for sample_id in ids)
    forest_seconds = sum(float(forest[sample_id][key]) for sample_id in ids)
    return domino_seconds / forest_seconds


def bootstrap_ratio(
    ids: list[str],
    domino: dict[str, dict[str, Any]],
    forest: dict[str, dict[str, Any]],
    *,
    key: str,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        values.append(ratio_for_ids(sample, domino, forest, key))
    values.sort()
    low = values[int(0.025 * draws)]
    high = values[min(draws - 1, int(0.975 * draws))]
    return {
        "point": ratio_for_ids(ids, domino, forest, key),
        "ci95_low": low,
        "ci95_high": high,
        "draws": draws,
    }


def aggregate_spec(reports: list[dict[str, Any]]) -> dict[str, float | int | None]:
    accepted = sum(
        int(row["spec_accept_token_num"])
        for report in reports
        for row in report["records"]
    )
    verifies = sum(
        int(row["spec_verify_ct"])
        for report in reports
        for row in report["records"]
    )
    output = sum(
        int(row["completion_tokens"])
        for report in reports
        for row in report["records"]
    )
    return {
        "accepted_draft_tokens": accepted,
        "verify_cycles": verifies,
        "output_tokens": output,
        "accepted_draft_eal": None if verifies == 0 else accepted / verifies,
        "output_advance": None if verifies == 0 else output / verifies,
    }


def main() -> None:
    args = parse_args()
    target = load(args.target)
    domino_reports = [load(path) for path in args.domino]
    forest_reports = [load(path) for path in args.forest]
    all_reports = [target, *domino_reports, *forest_reports]
    shape = {
        (report["num_prompts"], report["max_new_tokens"]) for report in all_reports
    }
    if len(shape) != 1:
        raise RuntimeError("R056 runs do not have the same workload shape")

    target_domino = [parity(target, report) for report in domino_reports]
    target_forest = [parity(target, report) for report in forest_reports]
    domino_forest = [
        parity(domino, forest)
        for domino in domino_reports
        for forest in forest_reports
    ]
    domino_rows = median_rows(domino_reports)
    forest_rows = median_rows(forest_reports)
    ids = sorted(domino_rows)
    server_ratio = bootstrap_ratio(
        ids,
        domino_rows,
        forest_rows,
        key="server_e2e_seconds",
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    client_ratio = bootstrap_ratio(
        ids,
        domino_rows,
        forest_rows,
        key="client_seconds",
        draws=args.bootstrap_draws,
        seed=args.seed + 1,
    )
    by_domain = {}
    for domain in sorted({row["domain"] for row in domino_rows.values()}):
        domain_ids = [
            sample_id
            for sample_id in ids
            if domino_rows[sample_id]["domain"] == domain
        ]
        by_domain[domain] = bootstrap_ratio(
            domain_ids,
            domino_rows,
            forest_rows,
            key="server_e2e_seconds",
            draws=args.bootstrap_draws,
            seed=args.seed + sum(ord(char) for char in domain),
        )

    token_parity_passed = all(
        item["matching_prompts"] == item["total_prompts"]
        for item in [*target_domino, *target_forest, *domino_forest]
    )
    report = {
        "format": "r056_sglang_e2e_analysis_v1",
        "workload": {
            "num_prompts": target["num_prompts"],
            "tokens_per_prompt": target["max_new_tokens"],
            "total_tokens_per_run": target["total_completion_tokens"],
        },
        "parity": {
            "target_vs_domino": target_domino,
            "target_vs_forest": target_forest,
            "domino_vs_forest": domino_forest,
            "all_output_tokens_exact": token_parity_passed,
        },
        "acceptance": {
            "domino": aggregate_spec(domino_reports),
            "forest": aggregate_spec(forest_reports),
        },
        "throughput_ratio_forest_over_domino": {
            "server_e2e": server_ratio,
            "client_wall": client_ratio,
            "by_domain_server_e2e": by_domain,
        },
        "required_ratio": args.required_ratio,
        "claim_gate": {
            "token_parity_passed": token_parity_passed,
            "server_ci_low_passed": server_ratio["ci95_low"] >= args.required_ratio,
            "client_ci_low_passed": client_ratio["ci95_low"] >= args.required_ratio,
            "passed": (
                token_parity_passed
                and server_ratio["ci95_low"] >= args.required_ratio
                and client_ratio["ci95_low"] >= args.required_ratio
            ),
        },
        "inputs": {
            "target": str(args.target),
            "domino": [str(path) for path in args.domino],
            "forest": [str(path) for path in args.forest],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
