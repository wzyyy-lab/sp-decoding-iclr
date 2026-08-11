#!/usr/bin/env python3
"""Fail-fast production-shape latency gate for PLC-Head v1."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any, Callable

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM

from sph.parallel_lattice_correction import (
    PLCCorrectionGraphRunner,
    ParallelLatticeCorrectionHead,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feed-forward-width", type=int, default=256)
    parser.add_argument("--global-layers", type=int, default=1)
    parser.add_argument("--use-semantic-embedding", action="store_true")
    parser.add_argument("--use-full-hidden", action="store_true")
    return parser.parse_args()


def load_record(root: Path, index: int) -> dict[str, Any]:
    offset = index
    for shard in sorted(root.glob("shard-*.pt")):
        records = torch.load(shard, map_location="cpu", weights_only=False)
        if offset < len(records):
            return records[offset]
        offset -= len(records)
    raise IndexError(f"record-index {index} is outside the canonical dataset")


def cuda_benchmark(
    callback: Callable[[], Any], *, warmup: int, repeats: int
) -> float:
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callback()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


@torch.inference_mode()
def domino_eager_corrected_head(
    *,
    domino: torch.nn.Module,
    target_weight: torch.Tensor,
    prefix_ids: torch.Tensor,
    hidden: torch.Tensor,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    """Released eager loop over the same 15 corrected production positions."""

    _, state = domino.prefix_gru(F.embedding(prefix_ids, target_weight))
    tokens: list[torch.Tensor] = []
    for position in range(hidden.shape[1]):
        state_for_head = state.transpose(0, 1)
        correction = domino.embed_proj(
            torch.cat(
                [hidden[:, position : position + 1], state_for_head], dim=-1
            )
        )
        token = (
            base_logits[:, position : position + 1] + correction
        ).argmax(dim=-1)
        tokens.append(token[:, 0])
        if position + 1 < hidden.shape[1]:
            _, state = domino.prefix_gru(
                F.embedding(token, target_weight), state
            )
    return torch.stack(tokens, dim=1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    record = load_record(args.canonical, args.record_index)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    target.requires_grad_(False)
    domino.requires_grad_(False)

    # The cached tensor has exactly 15 positions.  We use all 15 as corrected
    # positions to benchmark the production B16 correction shape; this Gate 0
    # measures compute and memory, not acceptance quality of random weights.
    hidden = record["parallel_hidden"].to("cuda:0", torch.bfloat16)[None]
    if hidden.shape[1] != 15:
        raise ValueError(f"Gate 0 requires 15 positions, got {hidden.shape[1]}")
    target_weight = target.model.embed_tokens.weight
    base_logits = F.linear(hidden, target_weight).contiguous()
    anchor = torch.tensor(
        [int(record["anchor_token_id"])], device="cuda:0", dtype=torch.long
    )
    prefix = base_logits[:, 0].argmax(dim=-1)

    hidden_width = int(hidden.shape[-1])
    first_projection = domino.embed_proj[0]
    w_h = first_projection.weight[:, :hidden_width].detach().contiguous()
    w_out = domino.embed_proj[2].weight.detach()
    head = ParallelLatticeCorrectionHead(
        w_h=w_h,
        w_out=w_out,
        token_embeddings=(target_weight if args.use_semantic_embedding else None),
        use_full_hidden=args.use_full_hidden,
        max_positions=15,
        candidates=16,
        modes=args.modes,
        width=args.width,
        heads=args.heads,
        feed_forward_width=args.feed_forward_width,
        global_layers=args.global_layers,
    ).to(device="cuda:0", dtype=torch.bfloat16).eval()
    head.requires_grad_(False)
    head.prepare_inference()

    with torch.inference_mode():
        eager_output = head(
            parallel_hiddens=hidden,
            base_logits=base_logits,
            anchor_ids=anchor,
            prefix_ids=prefix,
            return_logits=False,
        ).token_ids
    eager_ms = cuda_benchmark(
        lambda: head(
            parallel_hiddens=hidden,
            base_logits=base_logits,
            anchor_ids=anchor,
            prefix_ids=prefix,
            return_logits=False,
        ).token_ids,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    plc_graph = PLCCorrectionGraphRunner(
        head=head,
        batch_size=1,
        positions=15,
        device=torch.device("cuda:0"),
    )
    plc_prefix = torch.stack([anchor, prefix], dim=-1)
    graph_output = plc_graph(plc_prefix, hidden, base_logits)
    graph_ms = cuda_benchmark(
        lambda: plc_graph(plc_prefix, hidden, base_logits),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    # Exact released optimized comparator at the same 15-correction shape.
    domino_code_root = Path(__file__).resolve().parents[1] / "third_party" / "Domino" / "code"
    sys.path.insert(0, str(domino_code_root))
    from kernel.domino import DraftCorrectionGraphRunner

    domino_graph = DraftCorrectionGraphRunner(
        draft_model=domino,
        target_model=target,
        batch_size=1,
        steps=15,
        hidden_dim=hidden_width,
        gru_hidden_dim=int(domino.gru_hidden_dim),
        vocab_size=int(base_logits.shape[-1]),
        prefix_token_count=2,
        device=torch.device("cuda:0"),
    )
    domino_prefix = torch.stack([anchor, prefix], dim=-1)
    domino_eager_output = domino_eager_corrected_head(
        domino=domino,
        target_weight=target_weight,
        prefix_ids=domino_prefix,
        hidden=hidden,
        base_logits=base_logits,
    )
    domino_eager_ms = cuda_benchmark(
        lambda: domino_eager_corrected_head(
            domino=domino,
            target_weight=target_weight,
            prefix_ids=domino_prefix,
            hidden=hidden,
            base_logits=base_logits,
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    domino_graph_output = domino_graph(domino_prefix, hidden, base_logits)
    domino_graph_ms = cuda_benchmark(
        lambda: domino_graph(domino_prefix, hidden, base_logits),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    limit_ms = 0.8 * domino_graph_ms

    result = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "shape": {
            "batch": 1,
            "corrected_positions": 15,
            "candidates": 16,
            "modes": args.modes,
            "width": args.width,
            "feed_forward_width": args.feed_forward_width,
            "global_layers": args.global_layers,
            "use_semantic_embedding": args.use_semantic_embedding,
            "use_full_hidden": args.use_full_hidden,
            "vocab_size": int(base_logits.shape[-1]),
        },
        "parameters": {
            "plc_trainable": head.trainable_parameter_count,
            "plc_active": head.active_parameter_count,
            "domino_active": 50_823_168,
            "plc_fraction_of_domino": (
                head.active_parameter_count / 50_823_168
            ),
        },
        "runtime_memory_bytes": {
            "projected_lexical_table": head.projected_lexical_table.numel()
            * head.projected_lexical_table.element_size(),
            "domino_gru_input_table": domino_graph._gru_input_proj_table.numel()
            * domino_graph._gru_input_proj_table.element_size(),
        },
        "latency_ms_per_block": {
            "plc_eager": eager_ms,
            "plc_cuda_graph_complete_head": graph_ms,
            "domino_eager_complete_head": domino_eager_ms,
            "domino_cuda_graph_complete_head": domino_graph_ms,
            "plc_over_domino_eager": eager_ms / domino_eager_ms,
            "plc_over_domino": graph_ms / domino_graph_ms,
            "gate_limit_0.8x_domino": limit_ms,
        },
        "checks": {
            "graph_matches_eager_tokens": bool(graph_output.eq(eager_output).all()),
            "domino_graph_matches_eager_tokens": bool(
                domino_graph_output.eq(domino_eager_output).all()
            ),
            "gate0_pass": bool(graph_ms <= limit_ms),
        },
        "interpretation": (
            "Random PLC weights are intentional: Gate 0 tests the complete "
            "production-shape architecture before spending compute on imitation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
