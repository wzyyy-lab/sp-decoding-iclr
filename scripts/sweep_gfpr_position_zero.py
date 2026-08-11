#!/usr/bin/env python3
"""Measure the deployed EAL effect of the shared GFPR position-zero scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel

from sph.gfpr import load_adaptation
from train_domino_cached_head import load_tensor_from_checkpoint
from train_gfpr_head import evaluate, group_prompts, load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--adaptation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=(-2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    )
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("position-zero sweep requires CUDA")
    _, records = load_records(args.rollout, args.split)
    grouped = group_prompts(records, args.max_prompts)
    records = [record for group in grouped.values() for record in group]
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    checkpoint_scale = None
    if args.adaptation is not None:
        checkpoint_scale = float(
            load_adaptation(domino, args.adaptation, map_location="cuda:0")
        )
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to("cuda:0", torch.bfloat16)
    rows = []
    for scale in args.scales:
        metrics = evaluate(
            domino=domino,
            target_weight=target_weight,
            position_zero_scale=torch.tensor(scale, device="cuda:0"),
            records=records,
            batch_size=1,
            bootstrap_samples=args.bootstrap_samples,
            seed=0,
        )
        rows.append({"position_zero_scale": scale, **metrics})
        print(json.dumps(rows[-1], indent=2), flush=True)
    report = {
        "rollout": str(args.rollout.resolve()),
        "adaptation": str(args.adaptation.resolve()) if args.adaptation else None,
        "checkpoint_position_zero_scale": checkpoint_scale,
        "prompts": len(grouped),
        "blocks": len(records),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
