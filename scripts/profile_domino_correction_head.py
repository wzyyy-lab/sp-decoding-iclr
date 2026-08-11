#!/usr/bin/env python3
"""Profile released Domino's sequential correction and its batched-code floor."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=200)
    return parser.parse_args()


def load_record(root: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("record-index must be nonnegative")
    offset = index
    for shard in sorted(root.glob("shard-*.pt")):
        records = torch.load(shard, map_location="cpu", weights_only=False)
        if offset < len(records):
            return records[offset]
        offset -= len(records)
    raise IndexError(f"record-index {index} is outside the canonical dataset")


@torch.inference_mode()
def released_domino_head(
    *,
    domino: torch.nn.Module,
    target_weight: torch.Tensor,
    hidden: torch.Tensor,
    base_logits: torch.Tensor,
    anchor: torch.Tensor,
    return_codes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reproduce the width-one released correction over one cached block."""

    horizon = int(hidden.shape[1])
    first = base_logits[:, 0].argmax(dim=-1)
    ids = [first]
    prefix = torch.stack([anchor, first], dim=-1)
    _, state = domino.prefix_gru(F.embedding(prefix, target_weight))
    projected_codes: list[torch.Tensor] = []
    for position in range(1, horizon):
        z_i = hidden[:, position : position + 1]
        state_i = state.transpose(0, 1)
        if getattr(domino, "use_bias_norm", False):
            state_i = domino.bias_norm(state_i)
        joined = torch.cat([z_i, state_i], dim=-1)
        code = domino.embed_proj[1](domino.embed_proj[0](joined))[:, 0]
        correction = F.linear(code, domino.embed_proj[2].weight)
        token = (base_logits[:, position] + correction).argmax(dim=-1)
        ids.append(token)
        if return_codes:
            projected_codes.append(code)
        if position + 1 < horizon:
            _, state = domino.prefix_gru(
                F.embedding(token[:, None], target_weight), state
            )
    codes = torch.stack(projected_codes, dim=1) if return_codes else None
    return torch.stack(ids, dim=1), codes


@torch.inference_mode()
def batched_vocabulary_projection(
    *,
    domino: torch.nn.Module,
    codes: torch.Tensor,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    """Project already-predicted correction codes in one batched GEMM."""

    first = base_logits[:, :1].argmax(dim=-1)
    correction = F.linear(codes, domino.embed_proj[2].weight)
    later = (base_logits[:, 1:] + correction).argmax(dim=-1)
    return torch.cat([first, later], dim=-1)


def cuda_benchmark(
    callback: Callable[[], Any], *, warmup: int, repeats: int
) -> float:
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    record = load_record(args.canonical, args.record_index)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target), local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    domino = AutoModel.from_pretrained(
        str(args.domino_draft), trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda:0",
    ).eval()
    target.requires_grad_(False)
    domino.requires_grad_(False)
    target_weight = target.model.embed_tokens.weight
    hidden = record["parallel_hidden"].to("cuda:0", torch.bfloat16)[None]
    anchor = torch.tensor(
        [int(record["anchor_token_id"])], device="cuda:0", dtype=torch.long
    )
    base_logits = F.linear(hidden, target_weight)
    released = record["released_onpolicy_ids"].to("cuda:0", torch.long)[None]

    reproduced, codes = released_domino_head(
        domino=domino,
        target_weight=target_weight,
        hidden=hidden,
        base_logits=base_logits,
        anchor=anchor,
        return_codes=True,
    )
    assert codes is not None
    batched = batched_vocabulary_projection(
        domino=domino, codes=codes, base_logits=base_logits
    )
    if not torch.equal(reproduced, batched):
        raise RuntimeError("batched projection changed predictions")

    sequential_ms = cuda_benchmark(
        lambda: released_domino_head(
            domino=domino,
            target_weight=target_weight,
            hidden=hidden,
            base_logits=base_logits,
            anchor=anchor,
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    batched_ms = cuda_benchmark(
        lambda: batched_vocabulary_projection(
            domino=domino, codes=codes, base_logits=base_logits
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    # Measure the released serving implementation, not just the eager reference.
    # The cached record contains one base-prefix token followed by corrected
    # positions, so this runner uses the corresponding horizon - 1 steps.
    domino_code_root = Path(__file__).resolve().parents[1] / "third_party" / "Domino" / "code"
    sys.path.insert(0, str(domino_code_root))
    from kernel.domino import DraftCorrectionGraphRunner

    graph_prefix = torch.stack([anchor, reproduced[:, 0]], dim=-1)
    graph_hidden = hidden[:, 1:].contiguous()
    graph_base = base_logits[:, 1:].contiguous()
    graph_runner = DraftCorrectionGraphRunner(
        draft_model=domino,
        target_model=target,
        batch_size=1,
        steps=int(graph_hidden.shape[1]),
        hidden_dim=int(hidden.shape[-1]),
        gru_hidden_dim=int(domino.gru_hidden_dim),
        vocab_size=int(base_logits.shape[-1]),
        prefix_token_count=2,
        device=torch.device("cuda:0"),
    )
    graph_later = graph_runner(graph_prefix, graph_hidden, graph_base)
    graph_tokens = torch.cat([reproduced[:, :1], graph_later], dim=-1)
    optimized_graph_ms = cuda_benchmark(
        lambda: graph_runner(graph_prefix, graph_hidden, graph_base),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    hidden_size = int(domino.config.hidden_size)
    gru_hidden = int(domino.gru_hidden_dim)
    code_dim = int(domino.emb_dim)
    vocab_size = int(domino.config.vocab_size)
    gru_parameters = 3 * gru_hidden * hidden_size + 3 * gru_hidden * gru_hidden
    projection_parameters = (hidden_size + gru_hidden) * code_dim + code_dim * vocab_size
    horizon = int(hidden.shape[1])
    corrected_positions = horizon - 1
    # The two-token anchor/first-token prefix plus the later one-token updates
    # execute exactly ``horizon`` GRU time steps in total.
    estimated_gru_macs = horizon * gru_parameters
    estimated_projection_macs = corrected_positions * projection_parameters

    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "record": {
            "index": args.record_index,
            "sample_id": str(record["sample_id"]),
            "split": str(record["split"]),
            "domain": str(record["domain"]),
            "horizon": horizon,
        },
        "prediction_replay": {
            "matches_cached_released": bool(reproduced.eq(released).all()),
            "mismatch_tokens": int(reproduced.ne(released).sum()),
            "batched_projection_equal": True,
            "graph_matches_eager": bool(graph_tokens.eq(reproduced).all()),
            "graph_eager_mismatch_tokens": int(graph_tokens.ne(reproduced).sum()),
        },
        "parameters": {
            "gru": gru_parameters,
            "projection": projection_parameters,
            "total_correction_head": gru_parameters + projection_parameters,
        },
        "estimated_macs": {
            "gru_updates": estimated_gru_macs,
            "sequential_projection": estimated_projection_macs,
            "total_sequential_head": estimated_gru_macs + estimated_projection_macs,
            "batched_vocabulary_projection_only": corrected_positions * code_dim * vocab_size,
        },
        "latency_ms_per_block": {
            "released_sequential_head": sequential_ms,
            "released_optimized_graph_head": optimized_graph_ms,
            "batched_projection_with_precomputed_codes": batched_ms,
            "available_budget_at_0.8x_optimized_domino_ms": 0.8 * optimized_graph_ms,
            "remaining_encoder_budget_vs_optimized_ms": (
                0.8 * optimized_graph_ms - batched_ms
            ),
        },
        "benchmark": {"warmup": args.warmup, "repeats": args.repeats},
        "interpretation": (
            "The batched number is a projection floor, not PLC latency: correction "
            "codes are precomputed. PLC must fit its lattice encoder inside the "
            "reported remaining budget against the optimized graph runner and "
            "pass full end-to-end generation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
