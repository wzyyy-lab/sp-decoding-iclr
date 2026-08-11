#!/usr/bin/env python3
"""Summarize GFPR identity, all-position oracle, and rollout semantics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch

from sph.gfpr import (
    accepted_lengths,
    oracle_prefix_lengths,
    topk_oracle_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--expected-released-eal", type=float)
    parser.add_argument("--expected-eal-tolerance", type=float, default=1e-5)
    parser.add_argument("--minimum-oracle-eal", type=float, default=8.325)
    parser.add_argument(
        "--oracle-kind",
        choices=("base16", "k16", "k17"),
        default="base16",
        help="Candidate support whose all-position oracle defines the gate.",
    )
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def load_records(root: Path, split: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("collection_complete", False):
        raise RuntimeError(f"incomplete GFPR collection: {root}")
    records: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        for record in torch.load(shard, map_location="cpu", weights_only=False):
            if split is None or str(record["split"]) == split:
                records.append(record)
    if not records:
        raise ValueError("no GFPR records matched the requested split")
    return metadata, records


def prompt_balanced(sample_ids: list[str], values: list[int]) -> float:
    grouped: dict[str, list[int]] = defaultdict(list)
    for sample_id, value in zip(sample_ids, values, strict=True):
        grouped[sample_id].append(int(value))
    return sum(sum(group) / len(group) for group in grouped.values()) / len(grouped)


def _semantic_checks(records: list[dict[str, Any]], mode: str) -> dict[str, int]:
    accepted_mismatches = 0
    next_offset_mismatches = 0
    chain_mismatches = 0
    full_bonus_mismatches = 0
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(record)
        proposal = record["policy_ids"].long().view(1, -1)
        gold = record["gold_ids"].long().view(1, -1)
        recomputed = int(accepted_lengths(proposal, gold).item())
        accepted = int(record["accepted_length"])
        accepted_mismatches += int(recomputed != accepted)
        expected_next = int(record["anchor_offset"]) + accepted + 1
        stored_next = int(record["next_anchor_offset"])
        if stored_next >= 0:
            next_offset_mismatches += int(stored_next != expected_next)

    if mode == "dynamic":
        for sample_records in grouped.values():
            ordered = sorted(sample_records, key=lambda row: int(row["anchor_offset"]))
            for previous, current in zip(ordered, ordered[1:]):
                chain_mismatches += int(
                    int(previous["next_anchor_offset"])
                    != int(current["anchor_offset"])
                )
                if int(previous["accepted_length"]) == 16:
                    full_bonus_mismatches += int(
                        int(previous["bonus_token_id"])
                        != int(current["anchor_token_id"])
                    )
    return {
        "accepted_length_mismatches": accepted_mismatches,
        "next_offset_mismatches": next_offset_mismatches,
        "dynamic_chain_mismatches": chain_mismatches,
        "full_accept_bonus_mismatches": full_bonus_mismatches,
    }


def summarize(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    sample_ids: list[str] = []
    released_lengths: list[int] = []
    frozen_prefix_oracle: list[int] = []
    base16_oracle: list[int] = []
    k17_oracle: list[int] = []
    k16_oracle: list[int] = []
    position_zero_base_covered = 0
    position_zero_released_correct = 0
    position_zero_recoverable_failures = 0
    all_positions = 0
    base_covered = 0
    k16_covered = 0
    k17_covered = 0

    for record in records:
        gold = record["gold_ids"].long().view(1, -1)
        released = record["policy_ids"].long().view(1, -1)
        topk = record["base_topk_ids"].long().view(1, gold.shape[1], -1)
        matches = topk_oracle_matches(
            base_topk_ids=topk,
            released_ids=released,
            gold=gold,
        )
        released_match = released.eq(gold)
        frozen = torch.cat(
            [released_match[:, :1], matches["base16"][:, 1:]], dim=-1
        )
        sample_ids.append(str(record["sample_id"]))
        released_lengths.append(int(accepted_lengths(released, gold).item()))
        frozen_prefix_oracle.append(int(oracle_prefix_lengths(frozen).item()))
        base16_oracle.append(int(oracle_prefix_lengths(matches["base16"]).item()))
        k17_oracle.append(int(oracle_prefix_lengths(matches["k17"]).item()))
        k16_oracle.append(int(oracle_prefix_lengths(matches["k16"]).item()))
        base_zero = bool(matches["base16"][0, 0])
        released_zero = bool(released_match[0, 0])
        position_zero_base_covered += int(base_zero)
        position_zero_released_correct += int(released_zero)
        position_zero_recoverable_failures += int(base_zero and not released_zero)
        all_positions += gold.numel()
        base_covered += int(matches["base16"].sum())
        k16_covered += int(matches["k16"].sum())
        k17_covered += int(matches["k17"].sum())

    released_eal = prompt_balanced(sample_ids, released_lengths)
    return {
        "mode": mode,
        "blocks": len(records),
        "prompts": len(set(sample_ids)),
        "released_eal_prompt_balanced": released_eal,
        "frozen_position_zero_base16_oracle_eal": prompt_balanced(
            sample_ids, frozen_prefix_oracle
        ),
        "all16_base16_oracle_eal": prompt_balanced(sample_ids, base16_oracle),
        "all16_k17_oracle_eal": prompt_balanced(sample_ids, k17_oracle),
        "all16_k16_oracle_eal": prompt_balanced(sample_ids, k16_oracle),
        "all16_base16_oracle_gain_vs_released": prompt_balanced(
            sample_ids, base16_oracle
        )
        - released_eal,
        "position_zero_base16_coverage": position_zero_base_covered / len(records),
        "position_zero_released_accuracy": position_zero_released_correct
        / len(records),
        "position_zero_recoverable_released_failures": position_zero_recoverable_failures,
        "base16_position_coverage": base_covered / all_positions,
        "k16_position_coverage": k16_covered / all_positions,
        "k17_position_coverage": k17_covered / all_positions,
        "semantic_checks": _semantic_checks(records, mode),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    metadata, records = load_records(args.collection, args.split)
    report = summarize(records, str(metadata["mode"]))
    if args.expected_released_eal is not None:
        error = abs(
            float(report["released_eal_prompt_balanced"])
            - args.expected_released_eal
        )
        report["expected_released_eal"] = args.expected_released_eal
        report["released_eal_absolute_error"] = error
        report["released_eal_within_tolerance"] = error <= args.expected_eal_tolerance
    checks = report["semantic_checks"]
    report["semantic_gate_passed"] = all(int(value) == 0 for value in checks.values())
    oracle_key = f"all16_{args.oracle_kind}_oracle_eal"
    report["oracle_gate_metric"] = oracle_key
    report["target_oracle_gate_passed"] = (
        float(report[oracle_key]) >= args.minimum_oracle_eal
    )
    gates = [
        bool(report["semantic_gate_passed"]),
        bool(report["target_oracle_gate_passed"]),
    ]
    if args.expected_released_eal is not None:
        gates.append(bool(report["released_eal_within_tolerance"]))
    report["overall_gate_passed"] = all(gates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.fail_on_gate and not report["overall_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
