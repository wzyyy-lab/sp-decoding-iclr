#!/usr/bin/env python3
"""Measure exact runtime Top-16 acceptance headroom over released Domino."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_domino_cached_head import load_tensor_from_checkpoint
from train_plc_imitation import TeacherDataset, collate, load_records, prefix_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def prompt_mean(sample_ids: list[str], values: list[int]) -> float:
    grouped: dict[str, list[int]] = defaultdict(list)
    for sample_id, value in zip(sample_ids, values, strict=True):
        grouped[sample_id].append(value)
    return sum(sum(group) / len(group) for group in grouped.values()) / len(grouped)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    records = load_records(args.canonical, args.split, None)
    loader = DataLoader(
        TeacherDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate,
    )
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to("cuda:0", torch.bfloat16)
    sample_ids: list[str] = []
    teacher_lengths: list[int] = []
    oracle_lengths: list[int] = []
    prefix_failures = 0
    recoverable_first_misses = 0
    corrected_first_misses = 0
    covered_positions = 0
    positions = 0
    teacher_tokens_in_top16 = 0
    for batch in loader:
        hidden = batch["hidden"].to("cuda:0", non_blocking=True)
        prefix = batch["prefix"].to("cuda:0", non_blocking=True)
        gold_full = batch["gold_full"].to("cuda:0", non_blocking=True)
        teacher_full = batch["teacher_full"].to("cuda:0", non_blocking=True)
        base_logits = F.linear(hidden, target_weight)
        top16 = base_logits.topk(16, dim=-1).indices
        corrected_gold = gold_full[:, 1:]
        coverage = top16.eq(corrected_gold.unsqueeze(-1)).any(dim=-1)
        teacher_coverage = top16.eq(
            teacher_full[:, 1:].unsqueeze(-1)
        ).any(dim=-1)
        prefix_match = prefix == gold_full[:, 0]
        oracle_matches = torch.cat([prefix_match[:, None], coverage], dim=-1)
        batch_oracle = oracle_matches.long().cumprod(dim=-1).sum(dim=-1)
        batch_teacher = prefix_lengths(teacher_full, gold_full)

        mismatch = teacher_full != gold_full
        has_miss = mismatch.any(dim=-1)
        first_miss = mismatch.long().argmax(dim=-1)
        prefix_failures += int((has_miss & (first_miss == 0)).sum())
        corrected = has_miss & (first_miss > 0)
        corrected_first_misses += int(corrected.sum())
        corrected_index = (first_miss - 1).clamp_min(0)
        recoverable = coverage.gather(
            1, corrected_index[:, None]
        ).squeeze(1)
        recoverable_first_misses += int((corrected & recoverable).sum())
        covered_positions += int(coverage.sum())
        positions += coverage.numel()
        teacher_tokens_in_top16 += int(teacher_coverage.sum())
        sample_ids.extend(batch["sample_ids"])
        teacher_lengths.extend(batch_teacher.cpu().tolist())
        oracle_lengths.extend(batch_oracle.cpu().tolist())

    teacher_eal = prompt_mean(sample_ids, teacher_lengths)
    oracle_eal = prompt_mean(sample_ids, oracle_lengths)
    report: dict[str, Any] = {
        "split": args.split,
        "blocks": len(records),
        "teacher_eal_prompt_balanced": teacher_eal,
        "base_top16_gold_oracle_eal_prompt_balanced": oracle_eal,
        "oracle_minus_teacher_eal": oracle_eal - teacher_eal,
        "gold_position_coverage_top16": covered_positions / positions,
        "teacher_token_position_coverage_top16": teacher_tokens_in_top16 / positions,
        "teacher_first_miss_after_prefix": corrected_first_misses,
        "teacher_first_miss_recoverable_in_base_top16": recoverable_first_misses,
        "recoverable_fraction_given_correctable_first_miss": (
            recoverable_first_misses / max(1, corrected_first_misses)
        ),
        "unfixable_base_prefix_failures": prefix_failures,
        "interpretation": (
            "The oracle changes only corrected positions and leaves Domino's "
            "base-prefix token fixed, matching the PLC deployment boundary."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

