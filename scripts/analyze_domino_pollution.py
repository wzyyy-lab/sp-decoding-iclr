#!/usr/bin/env python3
"""Quantify which Domino GRU state divergence is acceptance-relevant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from sph.candidate_ceiling import accepted_draft_prefix_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean(tensor: torch.Tensor) -> float:
    return float(tensor.to(torch.float64).mean().item())


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for shard in sorted(args.input.glob("shard-*.pt")):
        records.extend(torch.load(shard, map_location="cpu", weights_only=False))
    if not records:
        raise FileNotFoundError(f"no records in {args.input}")
    gold = torch.stack([record["gold_ids"].long() for record in records])
    base_match = torch.stack([record["base_top1_match"].bool() for record in records])
    onpolicy_match = torch.stack([record["onpolicy_match"].bool() for record in records])
    teacher_match = torch.stack([record["teacher_match"].bool() for record in records])
    onpolicy_ids = torch.stack([record["onpolicy_ids"].long() for record in records])
    teacher_ids = torch.stack([record["teacher_ids"].long() for record in records])
    state_distance = torch.stack([record["state_distance"].float() for record in records])

    base_prefix = accepted_draft_prefix_lengths(base_match)
    onpolicy_prefix = accepted_draft_prefix_lengths(onpolicy_match)
    teacher_counterfactual_prefix = accepted_draft_prefix_lengths(teacher_match)
    if not torch.equal(onpolicy_prefix, teacher_counterfactual_prefix):
        raise AssertionError(
            "teacher forcing changed accepted prefix; states should be identical "
            "through the first on-policy mismatch"
        )

    positions = gold.shape[1]
    divergence = onpolicy_ids != teacher_ids
    post_rejection_mask = (
        torch.arange(positions).unsqueeze(0) > onpolicy_prefix.unsqueeze(1)
    )
    report = {
        "input": str(args.input.resolve()),
        "blocks": len(records),
        "positions": positions,
        "mean_base_top1_accepted_draft_tokens": mean(base_prefix),
        "mean_domino_onpolicy_accepted_draft_tokens": mean(onpolicy_prefix),
        "mean_teacher_counterfactual_accepted_draft_tokens": mean(
            teacher_counterfactual_prefix
        ),
        "accepted_prefix_identical_for_every_block": True,
        "position_accuracy": {
            "base": [float(x) for x in base_match.float().mean(dim=0)],
            "onpolicy": [float(x) for x in onpolicy_match.float().mean(dim=0)],
            "teacher_forced": [float(x) for x in teacher_match.float().mean(dim=0)],
        },
        "mean_state_distance_by_position": [
            float(x) for x in state_distance.mean(dim=0)
        ],
        "onpolicy_teacher_token_divergence_by_position": [
            float(x) for x in divergence.float().mean(dim=0)
        ],
        "post_rejection_positions": int(post_rejection_mask.sum().item()),
        "post_rejection_teacher_correct_onpolicy_wrong": int(
            (post_rejection_mask & teacher_match & ~onpolicy_match).sum().item()
        ),
        "interpretation": (
            "Teacher forcing changes GRU states and later token predictions only "
            "after the first rejected draft token. It therefore cannot increase "
            "the accepted prefix of that same verification round."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
