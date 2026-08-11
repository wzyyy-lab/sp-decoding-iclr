#!/usr/bin/env python3
"""Profile GLCS and released Domino on the same B16 correction interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM

from profile_plc_gate0 import cuda_benchmark, domino_eager_corrected_head
from sph.global_lookahead_causal_selector import (
    GLCSGraphRunner,
    GlobalLookaheadCausalSelector,
    topk_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=200)
    return parser.parse_args()


def load_record(root: Path, index: int) -> dict[str, Any]:
    offset = index
    for shard in sorted(root.glob("shard-*.pt")):
        records = torch.load(shard, map_location="cpu", weights_only=False)
        if offset < len(records):
            return records[offset]
        offset -= len(records)
    raise IndexError(f"record index {index} is outside {root}")


def build_head(
    *,
    domino: torch.nn.Module,
    target_weight: torch.Tensor,
    checkpoint: dict[str, Any],
) -> GlobalLookaheadCausalSelector:
    architecture = checkpoint["architecture"]
    first = domino.embed_proj[0].weight.detach()
    hidden_width = int(target_weight.shape[1])
    head = GlobalLookaheadCausalSelector(
        token_embeddings=target_weight,
        candidate_basis=domino.embed_proj[2].weight.detach(),
        gru_weight_ih=domino.prefix_gru.weight_ih_l0.detach(),
        gru_weight_hh=domino.prefix_gru.weight_hh_l0.detach(),
        hidden_projection=first[:, :hidden_width],
        state_projection=first[:, hidden_width:],
        max_positions=int(architecture["positions"]),
        candidates=int(architecture["candidates"]),
        global_width=int(architecture["global_width"]),
        global_heads=int(architecture["global_heads"]),
        global_layers=int(architecture["global_layers"]),
        global_modes=int(architecture["global_modes"]),
        feed_forward_width=int(architecture["feed_forward_width"]),
    ).to("cuda:0")
    head.load_state_dict(checkpoint["model"], strict=True)
    return head


def glcs_eager(
    *,
    head: GlobalLookaheadCausalSelector,
    prefix_ids: torch.Tensor,
    hidden: torch.Tensor,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    ids, logits = topk_candidates(base_logits, head.candidates)
    return head.decode(
        parallel_hiddens=hidden,
        candidate_ids=ids,
        candidate_logits=logits,
        anchor_ids=prefix_ids[:, 0],
        fixed_prefix_ids=prefix_ids[:, 1],
    ).token_ids


def benchmark(callback: Callable[[], Any], args: argparse.Namespace) -> float:
    return cuda_benchmark(callback, warmup=args.warmup, repeats=args.repeats)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.cuda.set_device(0)
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head = build_head(
        domino=domino,
        target_weight=target.model.embed_tokens.weight,
        checkpoint=checkpoint,
    ).to(dtype=torch.bfloat16).eval()
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    head.requires_grad_(False)

    hidden = record["parallel_hidden"].to("cuda:0", torch.bfloat16)[None]
    if hidden.shape[1] != 15:
        raise ValueError("profile requires exactly 15 correction positions")
    target_weight = target.model.embed_tokens.weight
    base_logits = F.linear(hidden, target_weight).contiguous()
    prefix_ids = torch.tensor(
        [
            [
                int(record["anchor_token_id"]),
                int(record["base_prefix_token_id"]),
            ]
        ],
        dtype=torch.long,
        device="cuda:0",
    )

    raw_output = glcs_eager(
        head=head,
        prefix_ids=prefix_ids,
        hidden=hidden,
        base_logits=base_logits,
    )
    raw_eager_ms = benchmark(
        lambda: glcs_eager(
            head=head,
            prefix_ids=prefix_ids,
            hidden=hidden,
            base_logits=base_logits,
        ),
        args,
    )
    head.prepare_inference()
    optimized_output = glcs_eager(
        head=head,
        prefix_ids=prefix_ids,
        hidden=hidden,
        base_logits=base_logits,
    )
    optimized_eager_ms = benchmark(
        lambda: glcs_eager(
            head=head,
            prefix_ids=prefix_ids,
            hidden=hidden,
            base_logits=base_logits,
        ),
        args,
    )
    graph_runner = GLCSGraphRunner(
        head=head,
        batch_size=1,
        positions=15,
        device=torch.device("cuda:0"),
    )
    graph_output = graph_runner(prefix_ids, hidden, base_logits)
    graph_ms = benchmark(
        lambda: graph_runner(prefix_ids, hidden, base_logits), args
    )

    domino_eager_output = domino_eager_corrected_head(
        domino=domino,
        target_weight=target_weight,
        prefix_ids=prefix_ids,
        hidden=hidden,
        base_logits=base_logits,
    )
    domino_eager_ms = benchmark(
        lambda: domino_eager_corrected_head(
            domino=domino,
            target_weight=target_weight,
            prefix_ids=prefix_ids,
            hidden=hidden,
            base_logits=base_logits,
        ),
        args,
    )
    domino_code = Path(__file__).resolve().parents[1] / "third_party" / "Domino" / "code"
    sys.path.insert(0, str(domino_code))
    from kernel.domino import DraftCorrectionGraphRunner

    domino_graph = DraftCorrectionGraphRunner(
        draft_model=domino,
        target_model=target,
        batch_size=1,
        steps=15,
        hidden_dim=int(hidden.shape[-1]),
        gru_hidden_dim=int(domino.gru_hidden_dim),
        vocab_size=int(base_logits.shape[-1]),
        prefix_token_count=2,
        device=torch.device("cuda:0"),
    )
    domino_graph_output = domino_graph(prefix_ids, hidden, base_logits)
    domino_graph_ms = benchmark(
        lambda: domino_graph(prefix_ids, hidden, base_logits), args
    )

    result = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "shape": {
            "batch": 1,
            "positions": 15,
            "candidates": head.candidates,
            "global_width": head.global_width,
            "global_modes": head.global_modes,
        },
        "parameters": {
            "glcs": parameter_count,
            "domino": 50_823_168,
            "ratio": parameter_count / 50_823_168,
            "fraction_of_headless_draft": parameter_count / 537_427_200,
        },
        "runtime_table_bytes": {
            "glcs_lexical": head.projected_lexical_table.numel()
            * head.projected_lexical_table.element_size(),
            "glcs_gru_input": head.projected_gru_input_table.numel()
            * head.projected_gru_input_table.element_size(),
            "domino_gru_input": domino_graph._gru_input_proj_table.numel()
            * domino_graph._gru_input_proj_table.element_size(),
        },
        "latency_ms": {
            "glcs_raw_eager": raw_eager_ms,
            "glcs_preprojected_eager": optimized_eager_ms,
            "glcs_cuda_graph": graph_ms,
            "domino_eager": domino_eager_ms,
            "domino_cuda_graph_triton": domino_graph_ms,
            "glcs_raw_over_domino_eager": raw_eager_ms / domino_eager_ms,
            "glcs_preprojected_over_domino_eager": optimized_eager_ms
            / domino_eager_ms,
            "glcs_graph_over_domino_graph": graph_ms / domino_graph_ms,
        },
        "checks": {
            "preprojection_preserves_tokens": bool(
                raw_output.eq(optimized_output).all()
            ),
            "graph_preserves_tokens": bool(graph_output.eq(optimized_output).all()),
            "domino_graph_preserves_tokens": bool(
                domino_graph_output.eq(domino_eager_output).all()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
