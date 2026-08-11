#!/usr/bin/env python3
"""Fair eager A40 profile of the frozen JAPD-16 head versus released Domino."""

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
from sph.global_direct_selector import GlobalDirectCandidateSelector
from sph.japd import BLOCK_LENGTH, CANDIDATES


JAPD_D64_PARAMETER_COUNT = 433_852


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-parameter-count", type=int, default=JAPD_D64_PARAMETER_COUNT
    )
    parser.add_argument("--record-index", type=int, default=587)
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
    shard = target / str(index["weight_map"][key])
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key)
    if weight.ndim != 2 or weight.shape[1] != 2560:
        raise RuntimeError(f"unexpected target embedding shape {tuple(weight.shape)}")
    return weight


def build_japd(
    checkpoint_path: Path | None, args: argparse.Namespace
) -> GlobalDirectCandidateSelector:
    config: dict[str, Any] = {
        "scope": "global",
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seed": args.seed,
    }
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_config = checkpoint.get("config", {})
        for key in (
            "scope",
            "model_dim",
            "num_heads",
            "num_layers",
            "dropout",
            "seed",
        ):
            if key in checkpoint_config and checkpoint_config[key] != config[key]:
                raise RuntimeError(
                    f"checkpoint {key}={checkpoint_config[key]!r} does not match "
                    f"profile argument {config[key]!r}"
                )
    scope = str(config.get("scope", "global"))
    if scope != "global":
        raise RuntimeError("J012 profiles only the frozen global JAPD arm")
    model = GlobalDirectCandidateSelector(
        hidden_size=2560,
        max_positions=BLOCK_LENGTH,
        max_candidates=CANDIDATES,
        model_dim=int(config.get("model_dim", 64)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 1)),
        scope=scope,
        mixer="axial",
        node_encoder="additive",
        dropout=float(config.get("dropout", 0.1)),
        initialization_seed=int(config.get("seed", 0)),
    )
    if checkpoint is not None:
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
        [
            float(start.elapsed_time(end))
            for start, end in zip(starts, ends, strict=True)
        ]
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
    if hidden.shape != (1, BLOCK_LENGTH, 2560):
        raise RuntimeError("profile requires exact full16 DFlash hidden geometry")
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
    japd = build_japd(args.checkpoint, args).to(device).eval()
    japd.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in japd.parameters())
    if parameter_count != args.require_parameter_count:
        raise RuntimeError(
            f"JAPD parameter count is {parameter_count}, expected "
            f"{args.require_parameter_count}"
        )

    def base_lattice() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        full_logits = F.linear(hidden, target_weight)
        full_logits_float = full_logits.float()
        candidate_logits, candidate_ids = full_logits_float.topk(
            CANDIDATES, dim=-1
        )
        base_logsumexp = torch.logsumexp(full_logits_float, dim=-1)
        return full_logits, candidate_ids, candidate_logits, base_logsumexp

    full_logits, candidate_ids, candidate_logits, base_logsumexp = base_lattice()
    candidate_embeddings = target_weight[candidate_ids]
    anchor_embeddings = target_weight[anchor]

    cached_ids = record["base_topk_ids"].to(device).long()[None]
    cached_logits = record["base_topk_logits"].to(device)[None]
    if not torch.equal(candidate_ids, cached_ids):
        mismatch = int(candidate_ids.ne(cached_ids).sum().item())
        raise RuntimeError(f"base Top16 differs from cached lattice at {mismatch} cells")
    if not torch.equal(
        candidate_logits.to(torch.float16), cached_logits.to(torch.float16)
    ):
        raise RuntimeError("base Top16 stored-dtype logits differ from cached lattice")

    def japd_core(
        ids: Tensor,
        logits: Tensor,
        lse: Tensor,
        embeddings: Tensor,
        anchor_emb: Tensor,
    ) -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = japd(
                hidden,
                embeddings,
                logits,
                lse,
                anchor_emb,
            )
        ranks = output.scores.argmax(dim=-1)
        return ids.gather(-1, ranks.unsqueeze(-1)).squeeze(-1)

    def japd_incremental() -> Tensor:
        embeddings = target_weight[candidate_ids]
        anchor_emb = target_weight[anchor]
        return japd_core(
            candidate_ids,
            candidate_logits,
            base_logsumexp,
            embeddings,
            anchor_emb,
        )

    def domino_incremental() -> Tensor:
        tokens, _ = released_domino_head(
            domino=domino,
            target_weight=target_weight,
            hidden=hidden,
            base_logits=full_logits,
            anchor=anchor,
        )
        return tokens

    def japd_complete() -> Tensor:
        _, ids, logits, lse = base_lattice()
        embeddings = target_weight[ids]
        anchor_emb = target_weight[anchor]
        return japd_core(ids, logits, lse, embeddings, anchor_emb)

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

    def staged_japd_complete(repeats: int) -> dict[str, dict[str, float]]:
        """Time every component inside the same authoritative complete call."""

        boundaries = [
            [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            for _ in range(repeats)
        ]
        for events in boundaries:
            events[0].record()
            complete_logits = F.linear(hidden, target_weight)
            events[1].record()
            complete_logits_float = complete_logits.float()
            complete_candidate_logits, complete_candidate_ids = (
                complete_logits_float.topk(CANDIDATES, dim=-1)
            )
            complete_lse = torch.logsumexp(complete_logits_float, dim=-1)
            events[2].record()
            complete_embeddings = target_weight[complete_candidate_ids]
            complete_anchor_embeddings = target_weight[anchor]
            events[3].record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                complete_output = japd(
                    hidden,
                    complete_embeddings,
                    complete_candidate_logits,
                    complete_lse,
                    complete_anchor_embeddings,
                )
            complete_ranks = complete_output.scores.argmax(dim=-1)
            complete_candidate_ids.gather(
                -1, complete_ranks.unsqueeze(-1)
            ).squeeze(-1)
            events[4].record()
        torch.cuda.synchronize()

        def elapsed(start: int, end: int) -> dict[str, float]:
            return distribution(
                [float(events[start].elapsed_time(events[end])) for events in boundaries]
            )

        result = {
            "base_vocab_gemm": elapsed(0, 1),
            "fp32_top16_plus_logsumexp": elapsed(1, 2),
            "candidate_plus_anchor_gather": elapsed(2, 3),
            "global_head_plus_argmax": elapsed(3, 4),
            "total": elapsed(0, 4),
        }
        component_mean = sum(
            result[name]["mean"]
            for name in (
                "base_vocab_gemm",
                "fp32_top16_plus_logsumexp",
                "candidate_plus_anchor_gather",
                "global_head_plus_argmax",
            )
        )
        result["mean_additivity_abs_error_ms"] = {
            "value": abs(component_mean - result["total"]["mean"])
        }
        return result

    def staged_domino_complete(repeats: int) -> dict[str, dict[str, float]]:
        boundaries = [
            [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            for _ in range(repeats)
        ]
        for events in boundaries:
            events[0].record()
            complete_logits = F.linear(hidden, target_weight)
            events[1].record()
            released_domino_head(
                domino=domino,
                target_weight=target_weight,
                hidden=hidden,
                base_logits=complete_logits,
                anchor=anchor,
            )
            events[2].record()
        torch.cuda.synchronize()

        def elapsed(start: int, end: int) -> dict[str, float]:
            return distribution(
                [float(events[start].elapsed_time(events[end])) for events in boundaries]
            )

        result = {
            "base_vocab_gemm": elapsed(0, 1),
            "released_domino_head": elapsed(1, 2),
            "total": elapsed(0, 2),
        }
        component_mean = (
            result["base_vocab_gemm"]["mean"]
            + result["released_domino_head"]["mean"]
        )
        result["mean_additivity_abs_error_ms"] = {
            "value": abs(component_mean - result["total"]["mean"])
        }
        return result

    japd_tokens = japd_incremental()
    domino_tokens = domino_incremental()
    base_tokens = candidate_ids[..., 0]
    cached_policy = record["policy_ids"].to(device).long()[None]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        shape_probe = japd(
            hidden,
            candidate_embeddings,
            candidate_logits,
            base_logsumexp,
            anchor_embeddings,
        )
    checks = {
        "full16_hidden": list(hidden.shape) == [1, 16, 2560],
        "score_shape": list(shape_probe.scores.shape),
        "japd_identity_matches_base": bool(japd_tokens.eq(base_tokens).all())
        if args.checkpoint is None
        else None,
        "domino_matches_cached_policy": bool(domino_tokens.eq(cached_policy).all()),
        "japd_complete_matches_incremental": bool(
            japd_complete().eq(japd_tokens).all()
        ),
        "domino_complete_matches_incremental": bool(
            domino_complete().eq(domino_tokens).all()
        ),
    }
    if checks["japd_identity_matches_base"] is False:
        raise RuntimeError("zero-init JAPD does not reproduce DFlash base")
    if checks["score_shape"] != [1, 16, 16]:
        raise RuntimeError(f"JAPD score shape drifted: {checks['score_shape']}")
    if not checks["domino_matches_cached_policy"]:
        raise RuntimeError("released Domino eager replay differs from cached policy")
    if not checks["japd_complete_matches_incremental"]:
        raise RuntimeError("JAPD complete/incremental tokens differ")
    if not checks["domino_complete_matches_incremental"]:
        raise RuntimeError("Domino complete/incremental tokens differ")

    # Stabilize clocks/caches with both complete paths before any authoritative
    # timing.  The earlier standalone component screen is intentionally not
    # used because separately timed kernels need not be additive.
    for _ in range(max(200, args.warmup)):
        japd_complete()
        domino_complete()
    torch.cuda.synchronize()
    staged_japd = staged_japd_complete(args.repeats)
    staged_domino = staged_domino_complete(args.repeats)
    latency = {
        "japd_incremental": benchmark(
            japd_incremental,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "domino_incremental": benchmark(
            domino_incremental,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "japd_complete": benchmark(
            japd_complete,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "domino_complete": benchmark(
            domino_complete,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }
    staged_ratio = (
        staged_japd["total"]["p50"]
        / staged_domino["total"]["p50"]
    )
    standalone_ratio = (
        latency["japd_complete"]["p50"]
        / latency["domino_complete"]["p50"]
    )
    # Use the more conservative of two independently instrumented complete
    # p50 ratios for the development gate.
    complete_ratio = max(staged_ratio, standalone_ratio)
    report = {
        "format": "japd_eager_profile_v2",
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
            "split": str(record["split"]),
            "context_length": int(record["context_length"]),
        },
        "execution": {
            "batch": 1,
            "positions": BLOCK_LENGTH,
            "candidates": CANDIDATES,
            "compute_dtype": "bfloat16_autocast_with_fp32_lse_and_softmax",
            "mode": "eager",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "complete_japd_components": [
                "base_vocab_gemm",
                "fp32_top16",
                "fp32_logsumexp",
                "candidate_and_anchor_gather",
                "global_noncausal_head",
                "per_position_argmax",
            ],
        },
        "parameters": {
            "japd_trainable": parameter_count,
            "dflash_backbone_reference": 537_427_968,
            "fraction_of_dflash_backbone": parameter_count / 537_427_968,
        },
        "architecture": {
            "scope": "global",
            "mixer": "axial",
            "node_encoder": "additive",
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "initialization_seed": args.seed,
        },
        "checks": checks,
        "latency_ms": latency,
        "staged_complete_latency_ms": {
            "japd": staged_japd,
            "domino": staged_domino,
        },
        "staged_complete_p50_ratio": staged_ratio,
        "standalone_complete_p50_ratio": standalone_ratio,
        "complete_p50_ratio": complete_ratio,
        "incremental_p50_ratio": (
            latency["japd_incremental"]["p50"]
            / latency["domino_incremental"]["p50"]
        ),
        "memory": {
            "japd_incremental": memory_profile(japd_incremental),
            "domino_incremental": memory_profile(domino_incremental),
            "japd_complete": memory_profile(japd_complete),
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
