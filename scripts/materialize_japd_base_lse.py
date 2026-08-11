#!/usr/bin/env python3
"""Materialize or replay-verify the JAPD DFlash base-logsumexp sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from safetensors import safe_open
import torch
from torch import Tensor
from torch.nn import functional as F

from sph.global_direct_selector import GlobalDirectCandidateSelector
from sph.japd import BLOCK_LENGTH, CANDIDATES
from sph.japd_data import (
    load_lse_sidecar,
    load_rollout_records,
    record_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--mode", choices=("materialize", "verify"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--lse-atol", type=float, default=1e-5)
    parser.add_argument("--lse-rtol", type=float, default=1e-6)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def load_shared_vocabulary_projection(target: Path) -> Tensor:
    index = json.loads((target / "model.safetensors.index.json").read_text())
    key = "model.embed_tokens.weight"
    if key not in index["weight_map"]:
        raise RuntimeError(f"target checkpoint lacks shared projection key {key}")
    shard = target / str(index["weight_map"][key])
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key)
    if weight.ndim != 2 or weight.shape[1] != 2560:
        raise RuntimeError(f"unexpected vocabulary projection shape {tuple(weight.shape)}")
    return weight


def recompute_one(
    record: dict[str, Any],
    weight: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay the collector's exact batch1, 16-row BF16 GEMM geometry."""

    hidden = record["parallel_hidden"].to(
        device=device, dtype=torch.bfloat16
    ).unsqueeze(0)
    if hidden.shape != (1, BLOCK_LENGTH, 2560):
        raise RuntimeError("sidecar replay requires one full16 record")
    logits = F.linear(hidden, weight)
    top_logits, top_ids = logits.float().topk(CANDIDATES, dim=-1)
    lse = torch.logsumexp(logits.float(), dim=-1)
    return top_ids[0], top_logits[0], lse[0]


def build_replay_audit_head(device: torch.device) -> GlobalDirectCandidateSelector:
    """Deterministic nonzero readout used only to expose replay differences."""

    model = GlobalDirectCandidateSelector(
        hidden_size=2560,
        max_positions=BLOCK_LENGTH,
        max_candidates=CANDIDATES,
        model_dim=64,
        num_heads=4,
        num_layers=1,
        scope="global",
        mixer="axial",
        node_encoder="additive",
        dropout=0.1,
        initialization_seed=0,
    ).to(device).eval()
    with torch.no_grad():
        model.residual_projection.weight.zero_()
        model.residual_projection.weight[0, 0] = 1.0
    return model


def replay_scalar_and_token_parity(
    record: dict[str, Any],
    *,
    replay_top_logits: Tensor,
    replay_lse: Tensor,
    stored_lse: Tensor,
    weight: Tensor,
    audit_head: GlobalDirectCandidateSelector,
    device: torch.device,
    atol: float,
    rtol: float,
) -> tuple[float, float, int]:
    """Compare stored-sidecar and online-replayed complete head inputs."""

    stored_logits = record["base_topk_logits"].to(
        device=device, dtype=torch.float32
    )
    stored_lse = stored_lse.to(device=device, dtype=torch.float32)
    replay_top_logits = replay_top_logits.to(device=device, dtype=torch.float32)
    replay_lse = replay_lse.to(device=device, dtype=torch.float32)
    stored_features, _ = audit_head._scalar_features(
        stored_logits.unsqueeze(0), stored_lse.unsqueeze(0)
    )
    replay_features, _ = audit_head._scalar_features(
        replay_top_logits.unsqueeze(0), replay_lse.unsqueeze(0)
    )
    scalar_error = float(
        (stored_features - replay_features).abs().max().item()
    )
    if not torch.allclose(
        stored_features, replay_features, atol=atol, rtol=rtol
    ):
        raise RuntimeError(
            f"five-scalar replay mismatch for {record_key(record)}: "
            f"max_abs={scalar_error}"
        )

    candidate_ids = record["base_topk_ids"].to(
        device=device, dtype=torch.long
    )
    candidate_embeddings = F.embedding(candidate_ids, weight).unsqueeze(0)
    anchor_id = torch.tensor(
        [int(record["anchor_token_id"])], device=device, dtype=torch.long
    )
    anchor_embeddings = F.embedding(anchor_id, weight)
    hidden = record["parallel_hidden"].to(
        device=device, dtype=torch.bfloat16
    ).unsqueeze(0)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        stored_output = audit_head(
            hidden,
            candidate_embeddings,
            stored_logits.unsqueeze(0),
            stored_lse.unsqueeze(0),
            anchor_embeddings,
        )
        replay_output = audit_head(
            hidden,
            candidate_embeddings,
            replay_top_logits.unsqueeze(0),
            replay_lse.unsqueeze(0),
            anchor_embeddings,
        )
    score_error = float(
        (stored_output.scores - replay_output.scores).abs().max().item()
    )
    if not torch.allclose(
        stored_output.scores,
        replay_output.scores,
        atol=atol,
        rtol=rtol,
    ):
        raise RuntimeError(
            f"complete audit-head score replay mismatch for {record_key(record)}: "
            f"max_abs={score_error}"
        )
    stored_tokens = candidate_ids.gather(
        -1, stored_output.scores.argmax(dim=-1)[0].unsqueeze(-1)
    ).squeeze(-1)
    replay_tokens = candidate_ids.gather(
        -1, replay_output.scores.argmax(dim=-1)[0].unsqueeze(-1)
    ).squeeze(-1)
    token_mismatches = int(stored_tokens.ne(replay_tokens).sum().item())
    if token_mismatches:
        raise RuntimeError(
            f"selected-token replay mismatch for {record_key(record)}: "
            f"{token_mismatches} positions"
        )
    return scalar_error, score_error, token_mismatches


def validate_topk_replay(
    record: dict[str, Any],
    top_ids: Tensor,
    top_logits: Tensor,
) -> None:
    expected_ids = record["base_topk_ids"].to(
        device=top_ids.device, dtype=torch.long
    )
    if not torch.equal(top_ids, expected_ids):
        mismatch = int(top_ids.ne(expected_ids).sum().item())
        raise RuntimeError(
            f"Top-16 ID replay mismatch for {record_key(record)}: {mismatch} cells"
        )
    # The collector writes float32 top-k values to float16.  Comparing in that
    # same storage dtype is an exact, meaningful geometry check rather than a
    # repository hash check.
    replay_stored = top_logits.to(torch.float16).cpu()
    expected_stored = record["base_topk_logits"].to(torch.float16).cpu()
    if not torch.equal(replay_stored, expected_stored):
        max_error = float(
            (replay_stored.float() - expected_stored.float()).abs().max()
        )
        raise RuntimeError(
            f"Top-16 logit replay mismatch for {record_key(record)}: "
            f"max_abs={max_error}"
        )


class SidecarWriter:
    def __init__(self, output: Path, shard_size: int) -> None:
        if shard_size < 1:
            raise ValueError("shard size must be positive")
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []
        self.total = 0

    def add(self, item: dict[str, Any]) -> None:
        self.buffer.append(item)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.output / f"shard-{len(self.shards):05d}.pt"
        torch.save(self.buffer, path)
        count = len(self.buffer)
        self.total += count
        self.shards.append(
            {"path": path.name, "records": count, "bytes": path.stat().st_size}
        )
        self.buffer = []


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_records < 0:
        raise ValueError("max-records must be non-negative")
    if args.mode == "materialize" and (args.output is None or args.sidecar is not None):
        raise ValueError("materialize requires --output and forbids --sidecar")
    if args.mode == "verify" and (args.sidecar is None or args.output is not None):
        raise ValueError("verify requires --sidecar and forbids --output")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for claim-bearing sidecar replay")
    metadata, records = load_rollout_records(
        args.rollout, split=args.split, max_records=args.max_records
    )
    weight = load_shared_vocabulary_projection(args.target).to(
        device=device, dtype=torch.bfloat16
    )
    sidecar_values: dict[tuple[str, int, int], Tensor] | None = None
    sidecar_metadata: dict[str, Any] | None = None
    writer: SidecarWriter | None = None
    audit_head: GlobalDirectCandidateSelector | None = None
    if args.mode == "materialize":
        writer = SidecarWriter(args.output, args.shard_size)
    else:
        sidecar_metadata, sidecar_values = load_lse_sidecar(args.sidecar)
        if int(sidecar_metadata.get("records", -1)) != len(records):
            raise RuntimeError("verify sidecar and selected rollout count differ")
        audit_head = build_replay_audit_head(device)

    started = time.perf_counter()
    max_lse_abs_error = 0.0
    max_scalar_abs_error = 0.0
    max_audit_head_score_abs_error = 0.0
    selected_token_mismatches = 0
    for index, record in enumerate(records):
        top_ids, top_logits, lse = recompute_one(record, weight, device)
        validate_topk_replay(record, top_ids, top_logits)
        key = record_key(record)
        if writer is not None:
            writer.add(
                {
                    "sample_id": key[0],
                    "anchor_offset": key[1],
                    "context_length": key[2],
                    "base_logsumexp": lse.detach().cpu().float(),
                }
            )
        else:
            assert sidecar_values is not None
            if key not in sidecar_values:
                raise RuntimeError(f"verify sidecar lacks {key}")
            expected = sidecar_values[key].to(device=device, dtype=torch.float32)
            error = float((lse.float() - expected).abs().max().item())
            max_lse_abs_error = max(max_lse_abs_error, error)
            if not torch.allclose(
                lse.float(), expected, atol=args.lse_atol, rtol=args.lse_rtol
            ):
                raise RuntimeError(
                    f"base_logsumexp replay mismatch for {key}: max_abs={error}"
                )
            assert audit_head is not None
            scalar_error, score_error, token_mismatches = (
                replay_scalar_and_token_parity(
                    record,
                    replay_top_logits=top_logits,
                    replay_lse=lse,
                    stored_lse=expected,
                    weight=weight,
                    audit_head=audit_head,
                    device=device,
                    atol=args.lse_atol,
                    rtol=args.lse_rtol,
                )
            )
            max_scalar_abs_error = max(max_scalar_abs_error, scalar_error)
            max_audit_head_score_abs_error = max(
                max_audit_head_score_abs_error, score_error
            )
            selected_token_mismatches += token_mismatches
        if (index + 1) % 32 == 0 or index + 1 == len(records):
            print(
                json.dumps(
                    {
                        "progress": index + 1,
                        "records": len(records),
                        "mode": args.mode,
                    }
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    output_root = args.output if writer is not None else args.sidecar
    if writer is not None:
        writer.flush()
        report = {
            "format": "japd_base_lse_v1",
            "collection_complete": True,
            "source_rollout": str(args.rollout.resolve()),
            "source_format": metadata.get("format"),
            "split": args.split,
            "target": str(args.target.resolve()),
            "projection": "model.embed_tokens.weight (shared frozen DFlash vocabulary projection)",
            "geometry": "batch1_full16_bf16_f_linear_then_fp32_logsumexp",
            "records": writer.total,
            "block_length": BLOCK_LENGTH,
            "candidates": CANDIDATES,
            "seconds": elapsed,
            "peak_memory_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda"
                else 0.0
            ),
            "top16_ids_exact": True,
            "stored_dtype_top16_logits_exact": True,
            "shards": writer.shards,
        }
        (args.output / "metadata.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    else:
        report = {
            "format": "japd_base_lse_replay_v1",
            "verified": True,
            "source_rollout": str(args.rollout.resolve()),
            "sidecar": str(args.sidecar.resolve()),
            "split": args.split,
            "records": len(records),
            "geometry": "batch1_full16_bf16_f_linear_then_fp32_logsumexp",
            "top16_ids_exact": True,
            "stored_dtype_top16_logits_exact": True,
            "lse_atol": args.lse_atol,
            "lse_rtol": args.lse_rtol,
            "max_lse_abs_error": max_lse_abs_error,
            "five_scalar_channels_allclose": True,
            "max_scalar_abs_error": max_scalar_abs_error,
            "audit_head_scores_allclose": True,
            "max_audit_head_score_abs_error": max_audit_head_score_abs_error,
            "selected_tokens_exact": True,
            "selected_token_mismatches": selected_token_mismatches,
            "seconds": elapsed,
            "peak_memory_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda"
                else 0.0
            ),
        }
        (args.sidecar / "replay_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    assert output_root is not None
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
