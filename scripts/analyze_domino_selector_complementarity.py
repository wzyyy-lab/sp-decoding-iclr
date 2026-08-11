#!/usr/bin/env python3
"""Measure whether a pretrained Top-K selector complements released Domino."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from sph.global_direct_selector import GlobalDirectCandidateSelector
from train_domino_cached_head import (
    CachedDominoDataset,
    acceptance_lengths,
    collate,
    load_records,
    load_tensor_from_checkpoint,
    summarize_lengths,
)
from train_domino_global_refiner import (
    direct_selector_logits,
    released_onpolicy_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def prompt_balanced(values: list[int], sample_ids: list[str]) -> float:
    grouped: dict[str, list[int]] = defaultdict(list)
    for sample_id, value in zip(sample_ids, values, strict=True):
        grouped[sample_id].append(value)
    return sum(sum(items) / len(items) for items in grouped.values()) / len(grouped)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("selector complementarity analysis requires CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(args.canonical, args.split, None)
    loader = DataLoader(
        CachedDominoDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate,
    )
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to(device="cuda:0", dtype=torch.bfloat16)
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    payload = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    config = payload.get("args", {})
    selector = GlobalDirectCandidateSelector(
        hidden_size=int(domino.config.hidden_size),
        max_positions=int(records[0]["gold_ids"].numel()),
        max_candidates=args.candidate_topk,
        model_dim=int(config.get("model_dim", 64)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 1)),
        scope=str(config.get("scope", "global")),
        mixer=str(config.get("mixer", "axial")),
        node_encoder=str(config.get("node_encoder", "additive")),
        initialization_seed=int(config.get("seed", 0)),
    ).to("cuda:0").eval()
    selector.load_state_dict(payload["model"], strict=True)

    sample_ids: list[str] = []
    domains: list[str] = []
    domino_lengths: list[int] = []
    selector_lengths: list[int] = []
    path_oracle_lengths: list[int] = []
    token_union_lengths: list[int] = []
    domino_wins = selector_wins = ties = 0
    selector_token_repairs = selector_token_harms = 0
    horizon = 0
    for batch in loader:
        anchors = batch["anchors"].to("cuda:0", non_blocking=True)
        gold = batch["gold"].to("cuda:0", non_blocking=True)
        hidden = batch["hidden"].to("cuda:0", non_blocking=True)
        fixed_logits, released_ids, base_logits = released_onpolicy_logits(
            domino=domino,
            target_weight=target_weight,
            anchors=anchors,
            hidden=hidden,
        )
        selector_logits, _ = direct_selector_logits(
            refiner=selector,
            target_weight=target_weight,
            anchors=anchors,
            hidden=hidden,
            fixed_logits=fixed_logits,
            base_logits=base_logits,
            candidate_topk=args.candidate_topk,
            candidate_source="base_topk_plus_released",
        )
        selector_ids = selector_logits.argmax(dim=-1)
        released_length = acceptance_lengths(released_ids, gold)
        selector_length = acceptance_lengths(selector_ids, gold)
        union_correct = released_ids.eq(gold) | selector_ids.eq(gold)
        union_length = union_correct.long().cumprod(dim=-1).sum(dim=-1)
        for left, right in zip(
            released_length.cpu().tolist(), selector_length.cpu().tolist(), strict=True
        ):
            domino_wins += left > right
            selector_wins += right > left
            ties += left == right
        selector_token_repairs += int((selector_ids.eq(gold) & ~released_ids.eq(gold)).sum())
        selector_token_harms += int((~selector_ids.eq(gold) & released_ids.eq(gold)).sum())
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        domino_lengths.extend(int(value) for value in released_length.cpu())
        selector_lengths.extend(int(value) for value in selector_length.cpu())
        path_oracle_lengths.extend(
            max(int(left), int(right))
            for left, right in zip(released_length.cpu(), selector_length.cpu(), strict=True)
        )
        token_union_lengths.extend(int(value) for value in union_length.cpu())
        horizon = int(gold.shape[1])

    methods = {
        "domino": domino_lengths,
        "selector": selector_lengths,
        "path_oracle": path_oracle_lengths,
        "token_union_oracle": token_union_lengths,
    }
    summaries = {
        name: summarize_lengths(sample_ids, domains, values, horizon)
        for name, values in methods.items()
    }
    baseline = prompt_balanced(domino_lengths, sample_ids)
    report: dict[str, Any] = {
        "status": "completed",
        "blocks": len(sample_ids),
        "prompts": len(set(sample_ids)),
        "summaries": summaries,
        "prompt_balanced_delta_vs_domino": {
            name: prompt_balanced(values, sample_ids) - baseline
            for name, values in methods.items()
        },
        "block_path_comparison": {
            "selector_wins": selector_wins,
            "domino_wins": domino_wins,
            "ties": ties,
        },
        "all_position_token_comparison": {
            "selector_repairs_domino_wrong": selector_token_repairs,
            "selector_harms_domino_correct": selector_token_harms,
        },
        "sample_ids": sample_ids,
        "domains": domains,
        "lengths": methods,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "prompt_balanced_delta_vs_domino": report[
                    "prompt_balanced_delta_vs_domino"
                ],
                "block_path_comparison": report["block_path_comparison"],
                "all_position_token_comparison": report[
                    "all_position_token_comparison"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
