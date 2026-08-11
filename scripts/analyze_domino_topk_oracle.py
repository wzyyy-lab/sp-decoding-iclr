#!/usr/bin/env python3
"""Measure the real released-Domino Top-K prefix oracle on cached anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from train_domino_cached_head import (
    CachedDominoDataset,
    acceptance_lengths,
    collate,
    load_records,
    load_tensor_from_checkpoint,
    summarize_lengths,
)
from train_domino_global_refiner import released_onpolicy_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    # Released BF16 Domino has rare batch-kernel tie differences; batch one is
    # the exact same-anchor contract used by the cached baseline evaluator.
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def oracle_lengths(candidate_ids: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    covered = candidate_ids.eq(gold.unsqueeze(-1)).any(dim=-1)
    return covered.to(torch.int64).cumprod(dim=-1).sum(dim=-1)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    records = load_records(args.canonical, "validation_select", None)
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

    sample_ids: list[str] = []
    domains: list[str] = []
    released_lengths: list[int] = []
    fixed_oracles = {k: [] for k in args.topk}
    base_oracles = {k: [] for k in args.topk}
    union_oracles = {k: [] for k in args.topk}
    base_plus_released_oracles = {k: [] for k in args.topk}
    token_mismatches = 0
    length_mismatches = 0
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
        actual = acceptance_lengths(released_ids, gold)
        cached_ids = batch["cached_released_ids"].to("cuda:0")
        cached_lengths = batch["cached_released_lengths"].to("cuda:0")
        token_mismatches += int(released_ids.ne(cached_ids).sum())
        length_mismatches += int(actual.ne(cached_lengths).sum())
        released_lengths.extend(int(value) for value in actual.cpu())
        for topk in args.topk:
            fixed_ids = fixed_logits.topk(topk, dim=-1).indices
            base_ids = base_logits.topk(topk, dim=-1).indices
            fixed_oracles[topk].extend(
                int(value) for value in oracle_lengths(fixed_ids, gold).cpu()
            )
            base_oracles[topk].extend(
                int(value) for value in oracle_lengths(base_ids, gold).cpu()
            )
            union_ids = torch.cat([fixed_ids, base_ids], dim=-1)
            union_oracles[topk].extend(
                int(value) for value in oracle_lengths(union_ids, gold).cpu()
            )
            hybrid_ids = base_ids.clone()
            released_top1 = fixed_ids[..., 0]
            contains_released = hybrid_ids.eq(released_top1.unsqueeze(-1)).any(dim=-1)
            hybrid_ids[..., -1] = torch.where(
                contains_released, hybrid_ids[..., -1], released_top1
            )
            base_plus_released_oracles[topk].extend(
                int(value) for value in oracle_lengths(hybrid_ids, gold).cpu()
            )
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        horizon = gold.shape[1]

    if token_mismatches or length_mismatches:
        raise RuntimeError(
            f"cache replay mismatch: tokens={token_mismatches}, lengths={length_mismatches}"
        )
    released = summarize_lengths(sample_ids, domains, released_lengths, horizon)
    result = {
        "status": "completed",
        "samples": len(set(sample_ids)),
        "blocks": len(sample_ids),
        "horizon": horizon,
        "released_domino": released,
        "topk": {},
    }
    baseline = released["overall"]["mean_accepted_draft_tokens_prompt_balanced"]
    for topk in args.topk:
        result["topk"][str(topk)] = {}
        for label, values in (
            ("released", fixed_oracles[topk]),
            ("parallel_base", base_oracles[topk]),
            ("base_topk_plus_released_top1", base_plus_released_oracles[topk]),
            ("union", union_oracles[topk]),
        ):
            summary = summarize_lengths(sample_ids, domains, values, horizon)
            eal = summary["overall"]["mean_accepted_draft_tokens_prompt_balanced"]
            result["topk"][str(topk)][label] = {
                "eal": eal,
                "delta_vs_domino": eal - baseline,
                "by_domain": summary["by_domain"],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
