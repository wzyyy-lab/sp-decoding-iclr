#!/usr/bin/env python3
"""Fair eager A40 profile of PGCF-16 and released Domino."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from typing import Any, Callable

from safetensors import safe_open
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel

from profile_domino_correction_head import released_domino_head
from sph.parallel_global_candidate_fusion import (
    CANDIDATES,
    DEFAULT_PARAMETER_COUNT,
    ParallelGlobalCandidateFusionHead,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
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
    raise IndexError(f"record {index} is outside {root}")


def load_target_embedding(target: Path) -> Tensor:
    index = json.loads((target / "model.safetensors.index.json").read_text())
    key = "model.embed_tokens.weight"
    with safe_open(
        target / str(index["weight_map"][key]),
        framework="pt",
        device="cpu",
    ) as handle:
        return handle.get_tensor(key)


def build_pgcf(checkpoint_path: Path | None) -> ParallelGlobalCandidateFusionHead:
    if checkpoint_path is None:
        return ParallelGlobalCandidateFusionHead()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint.get("config", {})
    model = ParallelGlobalCandidateFusionHead(
        hidden_size=2560,
        model_dim=int(config.get("model_dim", 256)),
        num_heads=int(config.get("num_heads", 8)),
        num_layers=int(config.get("num_layers", 2)),
        ff_multiplier=int(config.get("ff_multiplier", 2)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def distribution(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def benchmark(
    callback: Callable[[], Any], *, warmup: int, repeats: int
) -> dict[str, float]:
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        callback()
        end.record()
    torch.cuda.synchronize()
    return distribution(
        [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]
    )


def memory_profile(callback: Callable[[], Any]) -> dict[str, int]:
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    callback()
    torch.cuda.synchronize()
    return {
        "baseline_allocated": baseline_allocated,
        "baseline_reserved": baseline_reserved,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "incremental_peak_allocated": (
            torch.cuda.max_memory_allocated() - baseline_allocated
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    record = load_record(args.rollout, args.record_index)
    hidden = record["parallel_hidden"].to(device, torch.bfloat16)[None]
    if hidden.shape != (1, 16, 2560):
        raise RuntimeError("profile requires exact full16 hidden geometry")
    anchor = torch.tensor(
        [int(record["anchor_token_id"])], device=device, dtype=torch.long
    )

    target_weight = load_target_embedding(args.target).to(
        device=device, dtype=torch.bfloat16
    )
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    domino.requires_grad_(False)
    pgcf = build_pgcf(args.checkpoint).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    pgcf.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in pgcf.parameters())
    if args.checkpoint is None and parameter_count != DEFAULT_PARAMETER_COUNT:
        raise RuntimeError(
            f"default PGCF parameter count is {parameter_count}, expected "
            f"{DEFAULT_PARAMETER_COUNT}"
        )

    projected_table = pgcf.project_vocabulary(target_weight).contiguous()
    projected_table_bytes = (
        projected_table.numel() * projected_table.element_size()
    )

    def base_lattice() -> tuple[Tensor, Tensor, Tensor]:
        full_logits = F.linear(hidden, target_weight)
        candidate_ids = full_logits.float().topk(
            CANDIDATES, dim=-1
        ).indices
        candidate_logits = full_logits.gather(-1, candidate_ids).float()
        return full_logits, candidate_ids, candidate_logits

    full_logits, candidate_ids, candidate_logits = base_lattice()
    anchor_embeddings = target_weight[anchor]

    cached_ids = record["base_topk_ids"].to(device).long()[None]
    if not torch.equal(candidate_ids, cached_ids):
        mismatch = int(candidate_ids.ne(cached_ids).sum())
        raise RuntimeError(f"base Top16 differs from cached lattice at {mismatch} rows")

    def pgcf_incremental() -> Tensor:
        projected_candidates = projected_table[candidate_ids]
        output = pgcf(
            hidden,
            candidate_logits,
            anchor_embeddings,
            projected_candidate_embeddings=projected_candidates,
        )
        ranks = output.scores.argmax(dim=-1)
        return candidate_ids.gather(-1, ranks.unsqueeze(-1)).squeeze(-1)

    def domino_incremental() -> Tensor:
        tokens, _ = released_domino_head(
            domino=domino,
            target_weight=target_weight,
            hidden=hidden,
            base_logits=full_logits,
            anchor=anchor,
        )
        return tokens

    def pgcf_complete() -> Tensor:
        _, ids, logits = base_lattice()
        output = pgcf(
            hidden,
            logits,
            anchor_embeddings,
            projected_candidate_embeddings=projected_table[ids],
        )
        ranks = output.scores.argmax(dim=-1)
        return ids.gather(-1, ranks.unsqueeze(-1)).squeeze(-1)

    def domino_complete() -> Tensor:
        logits = F.linear(hidden, target_weight)
        tokens, _ = released_domino_head(
            domino=domino,
            target_weight=target_weight,
            hidden=hidden,
            base_logits=logits,
            anchor=anchor,
        )
        return tokens

    pgcf_tokens = pgcf_incremental()
    domino_tokens = domino_incremental()
    base_tokens = candidate_ids[:, :, 0]
    cached_policy = record["policy_ids"].to(device).long()[None]
    checks = {
        "pgcf_identity_matches_base": bool(pgcf_tokens.eq(base_tokens).all())
        if args.checkpoint is None
        else None,
        "domino_matches_cached_policy": bool(
            domino_tokens.eq(cached_policy).all()
        ),
        "pgcf_complete_matches_incremental": bool(
            pgcf_complete().eq(pgcf_tokens).all()
        ),
        "domino_complete_matches_incremental": bool(
            domino_complete().eq(domino_tokens).all()
        ),
    }
    if not checks["domino_matches_cached_policy"]:
        raise RuntimeError("released Domino eager replay differs from cached policy")
    if checks["pgcf_identity_matches_base"] is False:
        raise RuntimeError("zero-init PGCF does not reproduce base Top1")
    if not checks["pgcf_complete_matches_incremental"]:
        raise RuntimeError("PGCF complete/incremental tokens differ")
    if not checks["domino_complete_matches_incremental"]:
        raise RuntimeError("Domino complete/incremental tokens differ")

    pgcf_incremental_latency = benchmark(
        pgcf_incremental, warmup=args.warmup, repeats=args.repeats
    )
    domino_incremental_latency = benchmark(
        domino_incremental, warmup=args.warmup, repeats=args.repeats
    )
    pgcf_complete_latency = benchmark(
        pgcf_complete, warmup=args.warmup, repeats=args.repeats
    )
    domino_complete_latency = benchmark(
        domino_complete, warmup=args.warmup, repeats=args.repeats
    )
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "record": {
            "index": args.record_index,
            "sample_id": str(record["sample_id"]),
            "domain": str(record["domain"]),
            "split": str(record["split"]),
        },
        "execution": {
            "batch": 1,
            "positions": 16,
            "candidates": 16,
            "dtype": "bfloat16",
            "mode": "eager",
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "parameters": {
            "pgcf_trainable": parameter_count,
            "projected_table_bytes": projected_table_bytes,
        },
        "checks": checks,
        "latency_ms": {
            "pgcf_incremental": pgcf_incremental_latency,
            "domino_incremental": domino_incremental_latency,
            "pgcf_complete": pgcf_complete_latency,
            "domino_complete": domino_complete_latency,
            "complete_p50_ratio": (
                pgcf_complete_latency["p50"]
                / domino_complete_latency["p50"]
            ),
            "incremental_p50_ratio": (
                pgcf_incremental_latency["p50"]
                / domino_incremental_latency["p50"]
            ),
        },
        "memory": {
            "pgcf_incremental": memory_profile(pgcf_incremental),
            "domino_incremental": memory_profile(domino_incremental),
            "pgcf_complete": memory_profile(pgcf_complete),
            "domino_complete": memory_profile(domino_complete),
        },
    }
    report["development_latency_gate_passed"] = (
        report["latency_ms"]["complete_p50_ratio"] <= 1.20
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["development_latency_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
