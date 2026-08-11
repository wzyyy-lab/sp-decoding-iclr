#!/usr/bin/env python3
"""Measure the deployment-legal Fast-K32 earliest-repair oracle."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
from typing import Any, Sequence

import torch
from transformers import AutoModel

from sph.fast_r048 import (
    candidate_union_with_proposal,
    fast_candidate_domino_decode,
    repair_earliest_frontier,
    sequential_perfect_frontier_repairs,
)
from sph.gfpr import accepted_lengths
from train_domino_cached_head import load_tensor_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--candidate-topk", type=int, default=32)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Claim-bearing fixed-B16 evaluation uses the serving batch shape 1.",
    )
    parser.add_argument("--minimum-one-repair-eal", type=float, default=8.40)
    parser.add_argument("--target-eal", type=float, default=8.325485908649174)
    parser.add_argument(
        "--maximum-required-oracle-recovery", type=float, default=0.90
    )
    parser.add_argument(
        "--minimum-frontier-count-coverage", type=float, default=0.95
    )
    parser.add_argument(
        "--minimum-oracle-reward-coverage", type=float, default=0.95
    )
    parser.add_argument("--expected-released-eal", type=float, default=7.23955296404276)
    parser.add_argument("--expected-eal-tolerance", type=float, default=1e-6)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def load_records(root: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not bool(metadata.get("collection_complete", False)):
        raise RuntimeError(f"incomplete collection: {root}")
    records: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        records.extend(
            record
            for record in torch.load(shard, map_location="cpu", weights_only=False)
            if str(record["split"]) == split
        )
    if not records:
        raise ValueError(f"no records for split={split!r}")
    return metadata, records


def prompt_balanced(sample_ids: Sequence[str], values: Sequence[int]) -> float:
    grouped: dict[str, list[int]] = defaultdict(list)
    for sample_id, value in zip(sample_ids, values, strict=True):
        grouped[str(sample_id)].append(int(value))
    return sum(sum(group) / len(group) for group in grouped.values()) / len(grouped)


def summarize_lengths(
    sample_ids: Sequence[str],
    domains: Sequence[str],
    values: Sequence[int],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "overall": prompt_balanced(sample_ids, values),
        "by_domain": {},
    }
    for domain in sorted(set(domains)):
        indices = [index for index, item in enumerate(domains) if item == domain]
        report["by_domain"][domain] = prompt_balanced(
            [sample_ids[index] for index in indices],
            [values[index] for index in indices],
        )
    return report


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Fast-R048 oracle requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size < 1 or args.candidate_topk < 2:
        raise ValueError("batch size and candidate Top-K must be positive")

    metadata, records = load_records(args.collection, args.split)
    stored_target = Path(str(metadata["target"])).resolve()
    stored_domino = Path(str(metadata["domino_draft"])).resolve()
    if stored_target != args.target.resolve() or stored_domino != args.domino_draft.resolve():
        raise ValueError("collection model provenance differs from requested checkpoints")

    device = torch.device("cuda:0")
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to(device=device, dtype=torch.bfloat16)
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    domino.requires_grad_(False)

    sample_ids: list[str] = []
    domains: list[str] = []
    released_lengths: list[int] = []
    fast_lengths: list[int] = []
    one_lengths: list[int] = []
    two_lengths: list[int] = []
    unrestricted_one_lengths: list[int] = []
    frontier_available = 0
    frontier_incomplete = 0
    released_stored_length_mismatches = 0
    fast_position_zero_mismatches = 0
    fast_released_token_matches = 0
    all_tokens = 0

    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        hidden = torch.stack(
            [record["parallel_hidden"].to(torch.bfloat16) for record in batch_records]
        ).to(device)
        anchors = torch.tensor(
            [int(record["anchor_token_id"]) for record in batch_records],
            dtype=torch.long,
            device=device,
        )
        gold = torch.stack(
            [record["gold_ids"].long() for record in batch_records]
        ).to(device)
        released = torch.stack(
            [record["policy_ids"].long() for record in batch_records]
        ).to(device)

        decoded = fast_candidate_domino_decode(
            domino=domino,
            target_weight=target_weight,
            anchors=anchors,
            hidden=hidden,
            candidate_topk=args.candidate_topk,
        )
        support = candidate_union_with_proposal(
            decoded.candidate_ids,
            decoded.token_ids,
            support_size=args.candidate_topk,
        )
        one = repair_earliest_frontier(
            decoded.token_ids, gold, candidate_ids=support
        )
        two = sequential_perfect_frontier_repairs(
            decoded.token_ids,
            gold,
            candidate_ids=support,
            repairs=2,
        )[-1]
        unrestricted = repair_earliest_frontier(
            decoded.token_ids, gold, candidate_ids=None
        )
        released_batch_lengths = accepted_lengths(released, gold)

        for index, record in enumerate(batch_records):
            sample_ids.append(str(record["sample_id"]))
            domains.append(str(record["domain"]))
            released_value = int(released_batch_lengths[index])
            released_lengths.append(released_value)
            fast_lengths.append(int(one.accepted_before[index]))
            one_lengths.append(int(one.accepted_after[index]))
            two_lengths.append(int(two.accepted_after[index]))
            unrestricted_one_lengths.append(int(unrestricted.accepted_after[index]))
            released_stored_length_mismatches += int(
                released_value != int(record["accepted_length"])
            )
        # Position zero is base-only in both released Domino and Fast-K.  This
        # is a useful end-to-end check that the independently loaded embedding
        # weight reproduces the collection's base argmax.
        fast_position_zero_mismatches += int(
            decoded.token_ids[:, 0].ne(released[:, 0]).sum()
        )
        fast_released_token_matches += int(decoded.token_ids.eq(released).sum())
        all_tokens += int(released.numel())
        frontier_available += int(one.repair_available.sum())
        frontier_incomplete += int(one.accepted_before.lt(gold.shape[1]).sum())

    released_summary = summarize_lengths(sample_ids, domains, released_lengths)
    fast_summary = summarize_lengths(sample_ids, domains, fast_lengths)
    one_summary = summarize_lengths(sample_ids, domains, one_lengths)
    two_summary = summarize_lengths(sample_ids, domains, two_lengths)
    unrestricted_summary = summarize_lengths(
        sample_ids, domains, unrestricted_one_lengths
    )
    released_eal = float(released_summary["overall"])
    fast_eal = float(fast_summary["overall"])
    one_eal = float(one_summary["overall"])
    unrestricted_eal = float(unrestricted_summary["overall"])
    candidate_gain = one_eal - fast_eal
    unrestricted_gain = unrestricted_eal - fast_eal
    required_gain = args.target_eal - fast_eal
    required_oracle_recovery = (
        max(required_gain, 0.0) / candidate_gain
        if candidate_gain > 0
        else float("inf")
    )
    oracle_reward_coverage = (
        candidate_gain / unrestricted_gain if unrestricted_gain > 0 else 1.0
    )
    frontier_count_coverage = frontier_available / max(frontier_incomplete, 1)
    expected_error = abs(released_eal - args.expected_released_eal)
    oracle_gate = one_eal >= args.minimum_one_repair_eal
    recovery_gate = (
        required_oracle_recovery <= args.maximum_required_oracle_recovery
    )
    count_coverage_gate = (
        frontier_count_coverage >= args.minimum_frontier_count_coverage
    )
    reward_coverage_gate = (
        oracle_reward_coverage >= args.minimum_oracle_reward_coverage
    )
    semantic_gate = (
        released_stored_length_mismatches == 0
        and fast_position_zero_mismatches == 0
        and expected_error <= args.expected_eal_tolerance
    )
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "contract": {
            "proposal": "one base-vocabulary GEMM plus causal gathered Domino Top-K correction",
            "repair": "one exact earliest-frontier replacement; suffix unchanged",
            "candidate_support": f"base Top-{args.candidate_topk} retaining current proposal",
            "two_repair_role": "headroom only; not an authorized trained method",
        },
        "source": {
            "collection": str(args.collection.resolve()),
            "split": args.split,
            "blocks": len(records),
            "prompts": len(set(sample_ids)),
            "request_batch_size": args.batch_size,
            "target": str(args.target.resolve()),
            "domino_draft": str(args.domino_draft.resolve()),
        },
        "metrics": {
            "released_domino": released_summary,
            "fast_candidate_proposal": fast_summary,
            "fast_k_one_repair_oracle": one_summary,
            "fast_k_two_repair_oracle": two_summary,
            "fast_unrestricted_one_repair_oracle": unrestricted_summary,
            "fast_delta_vs_released": fast_eal - released_eal,
            "one_repair_delta_vs_fast": candidate_gain,
            "one_repair_delta_vs_released": one_eal - released_eal,
            "ideal_output_ratio_one_repair_vs_released": (one_eal + 1.0)
            / (released_eal + 1.0),
            "frontier_gold_available": frontier_available,
            "frontier_incomplete": frontier_incomplete,
            "frontier_gold_coverage": frontier_count_coverage,
            "oracle_reward_coverage_vs_unrestricted": oracle_reward_coverage,
            "target_eal": args.target_eal,
            "required_oracle_gain_recovery": required_oracle_recovery,
            "fast_vs_released_token_match_fraction": fast_released_token_matches
            / all_tokens,
        },
        "checks": {
            "released_stored_length_mismatches": released_stored_length_mismatches,
            "fast_position_zero_vs_released_mismatches": fast_position_zero_mismatches,
            "expected_released_eal": args.expected_released_eal,
            "released_eal_absolute_error": expected_error,
            "semantic_gate_passed": semantic_gate,
            "minimum_one_repair_eal": args.minimum_one_repair_eal,
            "one_repair_oracle_gate_passed": oracle_gate,
            "maximum_required_oracle_recovery": args.maximum_required_oracle_recovery,
            "required_oracle_recovery_gate_passed": recovery_gate,
            "minimum_frontier_count_coverage": args.minimum_frontier_count_coverage,
            "frontier_count_coverage_gate_passed": count_coverage_gate,
            "minimum_oracle_reward_coverage": args.minimum_oracle_reward_coverage,
            "oracle_reward_coverage_gate_passed": reward_coverage_gate,
            "overall_gate_passed": (
                semantic_gate
                and oracle_gate
                and recovery_gate
                and count_coverage_gate
                and reward_coverage_gate
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.fail_on_gate and not bool(report["checks"]["overall_gate_passed"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
