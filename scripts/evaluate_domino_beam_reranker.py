#!/usr/bin/env python3
"""Evaluate a selected Domino candidate-lattice path reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel

from sph.domino_beam_reranker import DominoBeamPathReranker
from train_domino_beam_reranker import eal, evaluate
from train_domino_cached_head import load_records, load_tensor_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_gate")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["args"]
    records = load_records(args.canonical, args.split, None)
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
    horizon = int(records[0]["gold_ids"].numel())
    reranker = DominoBeamPathReranker(
        hidden_size=int(domino.config.hidden_size),
        causal_state_size=int(domino.gru_hidden_dim),
        token_feature_size=int(domino.embed_proj[2].weight.shape[1]) + 256,
        horizon=horizon,
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
    ).to("cuda:0")
    reranker.load_state_dict(payload["model_state_dict"], strict=True)
    summary = evaluate(
        reranker=reranker,
        domino=domino,
        target_weight=target_weight,
        records=records,
        candidate_topk=int(config["candidate_topk"]),
        beam_width=int(config["beam_width"]),
    )
    result = {
        "status": "completed",
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "candidate_source": "parallel_topk_union_released_token",
        "beam_width": int(config["beam_width"]),
        "summary": summary,
        "reranker_eal": eal(summary, "reranker"),
        "delta_vs_released": eal(summary, "reranker")
        - eal(summary, "released"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
