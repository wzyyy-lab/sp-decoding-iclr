#!/usr/bin/env python3
"""Fair eager A40 profile of PCLD-16R versus released Domino."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel

from profile_domino_correction_head import released_domino_head
from profile_japd16_head import distribution, load_record, memory_profile
from sph.pcld import (
    BLOCK_LENGTH,
    CANDIDATES,
    EXPECTED_PARAMETER_COUNT,
    PCLD16Head,
    assert_frozen_architecture,
)
from train_pcld16 import load_target_lm_head_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--record-index", type=int, default=587)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument(
        "--require-parameter-count", type=int, default=EXPECTED_PARAMETER_COUNT
    )
    return parser.parse_args()


def build_pcld(checkpoint_path: Path | None) -> PCLD16Head:
    model = PCLD16Head(scope="global")
    assert_frozen_architecture(model)
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("format") != "pcld16_checkpoint_v1":
            raise RuntimeError("unsupported PCLD checkpoint format")
        config = checkpoint.get("config", {})
        if config.get("scope") != "global":
            raise RuntimeError("production PCLD profile rejects local checkpoints")
        if int(checkpoint.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError("PCLD checkpoint parameter contract drifted")
        model.load_state_dict(checkpoint["model"], strict=True)
    return model


def benchmark(
    callback: Callable[[], Any], *, warmup: int, repeats: int
) -> dict[str, float]:
    if warmup < 1 or repeats < 1:
        raise ValueError("profile warmup/repeats must be positive")
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
        [
            float(start.elapsed_time(end))
            for start, end in zip(starts, ends, strict=True)
        ]
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PCLD eager profile requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    record = load_record(args.rollout, args.record_index)
    hidden = record["parallel_hidden"].to(device, torch.bfloat16)[None]
    if hidden.shape != (1, BLOCK_LENGTH, 2560):
        raise RuntimeError("PCLD profile requires exact full16 hidden")
    anchor = torch.tensor(
        [int(record["anchor_token_id"])], device=device, dtype=torch.long
    )
    target_weight_cpu, serialized_key = load_target_lm_head_weight(args.target)
    target_weight = target_weight_cpu.to(device=device, dtype=torch.bfloat16)
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    domino.requires_grad_(False)
    pcld = build_pcld(args.checkpoint).to(device).eval()
    pcld.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in pcld.parameters())
    if parameter_count != args.require_parameter_count:
        raise RuntimeError(
            f"PCLD parameter count {parameter_count} != {args.require_parameter_count}"
        )

    def base_lattice() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        full_logits = F.linear(hidden, target_weight)
        full_float = full_logits.float()
        candidate_logits, candidate_ids = full_float.topk(CANDIDATES, dim=-1)
        base_lse = torch.logsumexp(full_float, dim=-1)
        return full_logits, candidate_ids, candidate_logits, base_lse

    full_logits, candidate_ids, candidate_logits, base_lse = base_lattice()
    candidate_rows = target_weight[candidate_ids]
    cached_ids = record["base_topk_ids"].to(device).long()[None]
    cached_logits = record["base_topk_logits"].to(device)[None]
    if not torch.equal(candidate_ids, cached_ids):
        raise RuntimeError("PCLD profile base Top16 IDs differ from canonical rollout")
    if not torch.equal(
        candidate_logits.to(torch.float16), cached_logits.to(torch.float16)
    ):
        raise RuntimeError("PCLD profile base Top16 logits differ from canonical rollout")

    def pcld_core(ids: Tensor, logits: Tensor, lse: Tensor, rows: Tensor) -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = pcld(hidden, rows, logits, lse)
        ranks = output.scores.argmax(dim=-1)
        return ids.gather(-1, ranks.unsqueeze(-1)).squeeze(-1)

    def pcld_incremental() -> Tensor:
        return pcld_core(candidate_ids, candidate_logits, base_lse, candidate_rows)

    def domino_incremental() -> Tensor:
        tokens, _ = released_domino_head(
            domino=domino,
            target_weight=target_weight,
            hidden=hidden,
            base_logits=full_logits,
            anchor=anchor,
        )
        return tokens

    def pcld_complete() -> Tensor:
        _, ids, logits, lse = base_lattice()
        rows = target_weight[ids]
        return pcld_core(ids, logits, lse, rows)

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

    def staged_pcld_complete(repeats: int) -> dict[str, Any]:
        boundaries = [
            [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            for _ in range(repeats)
        ]
        for events in boundaries:
            events[0].record()
            logits = F.linear(hidden, target_weight)
            events[1].record()
            logits_float = logits.float()
            candidate_score, ids = logits_float.topk(CANDIDATES, dim=-1)
            lse = torch.logsumexp(logits_float, dim=-1)
            events[2].record()
            rows = target_weight[ids]
            events[3].record()
            pcld_core(ids, candidate_score, lse, rows)
            events[4].record()
        torch.cuda.synchronize()

        def elapsed(start: int, end: int) -> dict[str, float]:
            return distribution(
                [
                    float(events[start].elapsed_time(events[end]))
                    for events in boundaries
                ]
            )

        result: dict[str, Any] = {
            "base_vocab_gemm": elapsed(0, 1),
            "fp32_top16_plus_logsumexp": elapsed(1, 2),
            "lm_head_candidate_gather": elapsed(2, 3),
            "global_head_residual_dot_argmax": elapsed(3, 4),
            "total": elapsed(0, 4),
        }
        component_mean = sum(
            result[name]["mean"]
            for name in (
                "base_vocab_gemm",
                "fp32_top16_plus_logsumexp",
                "lm_head_candidate_gather",
                "global_head_residual_dot_argmax",
            )
        )
        result["mean_additivity_abs_error_ms"] = abs(
            component_mean - result["total"]["mean"]
        )
        return result

    def staged_domino_complete(repeats: int) -> dict[str, Any]:
        boundaries = [
            [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            for _ in range(repeats)
        ]
        for events in boundaries:
            events[0].record()
            logits = F.linear(hidden, target_weight)
            events[1].record()
            released_domino_head(
                domino=domino,
                target_weight=target_weight,
                hidden=hidden,
                base_logits=logits,
                anchor=anchor,
            )
            events[2].record()
        torch.cuda.synchronize()

        def elapsed(start: int, end: int) -> dict[str, float]:
            return distribution(
                [
                    float(events[start].elapsed_time(events[end]))
                    for events in boundaries
                ]
            )

        result: dict[str, Any] = {
            "base_vocab_gemm": elapsed(0, 1),
            "released_domino_head": elapsed(1, 2),
            "total": elapsed(0, 2),
        }
        result["mean_additivity_abs_error_ms"] = abs(
            result["base_vocab_gemm"]["mean"]
            + result["released_domino_head"]["mean"]
            - result["total"]["mean"]
        )
        return result

    pcld_tokens = pcld_incremental()
    domino_tokens = domino_incremental()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        shape_probe = pcld(hidden, candidate_rows, candidate_logits, base_lse)
    checks = {
        "full16_hidden": list(hidden.shape) == [1, 16, 2560],
        "score_shape_full16_k16": list(shape_probe.scores.shape) == [1, 16, 16],
        "pcld_identity_matches_base": bool(
            pcld_tokens.eq(candidate_ids[..., 0]).all()
        )
        if args.checkpoint is None
        else None,
        "domino_matches_cached_policy": bool(
            domino_tokens.eq(record["policy_ids"].to(device).long()[None]).all()
        ),
        "pcld_complete_matches_incremental": bool(
            pcld_complete().eq(pcld_tokens).all()
        ),
        "domino_complete_matches_incremental": bool(
            domino_complete().eq(domino_tokens).all()
        ),
    }
    failed = [name for name, value in checks.items() if value is False]
    if failed:
        raise RuntimeError(f"PCLD profile parity checks failed: {failed}")

    for _ in range(max(200, args.warmup)):
        pcld_complete()
        domino_complete()
    torch.cuda.synchronize()
    staged_pcld = staged_pcld_complete(args.repeats)
    staged_domino = staged_domino_complete(args.repeats)
    latency = {
        "pcld_incremental": benchmark(
            pcld_incremental, warmup=args.warmup, repeats=args.repeats
        ),
        "domino_incremental": benchmark(
            domino_incremental, warmup=args.warmup, repeats=args.repeats
        ),
        "pcld_complete": benchmark(
            pcld_complete, warmup=args.warmup, repeats=args.repeats
        ),
        "domino_complete": benchmark(
            domino_complete, warmup=args.warmup, repeats=args.repeats
        ),
    }
    staged_ratio = staged_pcld["total"]["p50"] / staged_domino["total"]["p50"]
    standalone_ratio = (
        latency["pcld_complete"]["p50"] / latency["domino_complete"]["p50"]
    )
    complete_ratio = max(staged_ratio, standalone_ratio)
    report = {
        "format": "pcld16_eager_profile_v1",
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "record": {
            "index": args.record_index,
            "sample_id": str(record["sample_id"]),
            "domain": str(record["domain"]),
            "context_length": int(record["context_length"]),
        },
        "execution": {
            "batch": 1,
            "positions": 16,
            "candidates": 16,
            "mode": "eager",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "complete_pcld_components": [
                "base_vocab_gemm",
                "fp32_top16",
                "fp32_logsumexp",
                "lm_head_candidate_gather",
                "global_noncausal_head",
                "residual_lm_head_dot",
                "per_position_argmax",
            ],
        },
        "parameters": {
            "pcld_trainable": parameter_count,
            "dflash_backbone_reference": 537_427_968,
            "fraction_of_dflash_backbone": parameter_count / 537_427_968,
        },
        "architecture": {
            "scope": "global",
            "model_dim": 256,
            "heads": 8,
            "layers": 2,
            "ffn_dim": 1024,
            "dropout": 0.0,
            "serialized_lm_head_key": serialized_key,
        },
        "checks": checks,
        "score_shape": list(shape_probe.scores.shape),
        "latency_ms": latency,
        "staged_complete_latency_ms": {
            "pcld": staged_pcld,
            "domino": staged_domino,
        },
        "staged_complete_p50_ratio": staged_ratio,
        "standalone_complete_p50_ratio": standalone_ratio,
        "complete_p50_ratio": complete_ratio,
        "incremental_p50_ratio": (
            latency["pcld_incremental"]["p50"]
            / latency["domino_incremental"]["p50"]
        ),
        "memory": {
            "pcld_incremental": memory_profile(pcld_incremental),
            "domino_incremental": memory_profile(domino_incremental),
            "pcld_complete": memory_profile(pcld_complete),
            "domino_complete": memory_profile(domino_complete),
        },
        "development_latency_gate_passed": complete_ratio <= 1.20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["development_latency_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
