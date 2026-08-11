#!/usr/bin/env python3
"""Materialize or replay the offline PCLD target-hidden supervision sidecar."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from sph.japd_data import load_rollout_records, record_key
from sph.pcld import BLOCK_LENGTH, CANDIDATES, HIDDEN_SIZE
from sph.pcld_data import (
    load_manifest,
    load_pcld_sidecar,
    select_balanced_smoke_records,
    select_manifest_group,
    validate_manifest_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--group",
        choices=("smoke32", "capacity", "fit", "select", "diagnostic", "all"),
        required=True,
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=("materialize", "verify"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--manual-parity-records", type=int, default=0)
    parser.add_argument("--hidden-atol", type=float, default=1e-4)
    parser.add_argument("--hidden-rtol", type=float, default=1e-5)
    parser.add_argument("--score-atol", type=float, default=1e-4)
    parser.add_argument("--score-rtol", type=float, default=1e-5)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def select_records(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata, records = load_rollout_records(args.rollout, split=args.split)
    if args.group == "smoke32":
        count = args.max_records or 32
        records = select_balanced_smoke_records(records, count=count)
    elif args.group == "all":
        if args.max_records:
            records = records[: args.max_records]
    else:
        if args.manifest is None:
            raise ValueError(f"--manifest is required for PCLD group {args.group}")
        manifest = load_manifest(args.manifest)
        validate_manifest_source(manifest, rollout=args.rollout, split=args.split)
        records = select_manifest_group(records, manifest, args.group)
        if args.max_records:
            if args.max_records > len(records):
                raise ValueError("max-records exceeds selected manifest group")
            records = records[: args.max_records]
    if not records:
        raise RuntimeError("PCLD sidecar selected no records")
    return metadata, records


class SidecarWriter:
    def __init__(self, root: Path, *, shard_size: int) -> None:
        if shard_size < 1:
            raise ValueError("shard-size must be positive")
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
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
        path = self.root / f"shard-{len(self.shards):05d}.pt"
        torch.save(self.buffer, path)
        count = len(self.buffer)
        self.total += count
        self.shards.append(
            {"path": path.name, "records": count, "bytes": path.stat().st_size}
        )
        self.buffer = []


def validate_target_checkpoint(target: Path) -> None:
    config = json.loads((target / "config.json").read_text())
    if not bool(config.get("tie_word_embeddings", False)):
        raise RuntimeError(
            "PCLD expects this checkpoint's target.lm_head.weight to share its "
            "serialized storage with model.embed_tokens.weight"
        )
    if int(config.get("hidden_size", -1)) != HIDDEN_SIZE:
        raise RuntimeError("PCLD target hidden size differs from 2560")


@torch.inference_mode()
def target_teacher_hidden(
    target: AutoModelForCausalLM,
    record: dict[str, Any],
    device: torch.device,
) -> Tensor:
    context = record["context_ids_before_anchor"].long()
    context_length = int(context.numel())
    if context_length != int(record["context_length"]):
        raise RuntimeError("rollout context length field is inconsistent")
    anchor = torch.tensor([int(record["anchor_token_id"])], dtype=torch.long)
    gold = record["gold_ids"].long()
    if tuple(gold.shape) != (BLOCK_LENGTH,):
        raise RuntimeError("PCLD teacher requires exactly 16 gold tokens")
    input_ids = torch.cat([context, anchor, gold[: BLOCK_LENGTH - 1]]).to(
        device=device
    )[None]
    outputs = target.model(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state[0, context_length : context_length + BLOCK_LENGTH]
    if hidden.shape != (BLOCK_LENGTH, HIDDEN_SIZE):
        raise RuntimeError("target teacher returned the wrong row geometry")
    return hidden


@torch.inference_mode()
def manual_prefix_hidden(
    target: AutoModelForCausalLM,
    record: dict[str, Any],
    device: torch.device,
) -> Tensor:
    """Independently reconstruct every teacher row from its exact prefix."""

    context = record["context_ids_before_anchor"].long()
    anchor = torch.tensor([int(record["anchor_token_id"])], dtype=torch.long)
    gold = record["gold_ids"].long()
    rows: list[Tensor] = []
    for position in range(BLOCK_LENGTH):
        prefix = torch.cat([context, anchor, gold[:position]]).to(device=device)[None]
        outputs = target.model(
            input_ids=prefix,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        rows.append(outputs.last_hidden_state[0, -1])
    return torch.stack(rows)


def manual_geometry_receipt(
    target: AutoModelForCausalLM,
    batched_hidden: Tensor,
    manual_hidden: Tensor,
) -> dict[str, Tensor]:
    """Compare equal-prefix teacher rows without demanding bitwise SDPA parity.

    Different q-lengths select different BF16/SDPA numerical kernels.  Raw
    hidden RMSE is therefore a reported diagnostic, not a hard method gate.
    Geometry is fail-closed by row identity, and token parity is fail-closed on
    every numerically stable non-tie row.  Ties are reported, never silently
    counted as exact matches.
    """

    if batched_hidden.shape != (BLOCK_LENGTH, HIDDEN_SIZE) or manual_hidden.shape != (
        BLOCK_LENGTH,
        HIDDEN_SIZE,
    ):
        raise ValueError("manual geometry requires two [16,2560] tensors")
    batched_float = batched_hidden.float()
    manual_float = manual_hidden.float()
    difference = batched_float - manual_float
    max_abs = difference.abs().amax(dim=-1)
    rmse = difference.square().mean(dim=-1).sqrt()
    reference_rms = batched_float.square().mean(dim=-1).sqrt().clamp_min(1e-12)
    relative_rmse = rmse / reference_rms
    cosine = F.cosine_similarity(batched_float, manual_float, dim=-1)
    similarity = F.normalize(batched_float, dim=-1) @ F.normalize(
        manual_float, dim=-1
    ).T
    best_manual_rows = similarity.argmax(dim=-1)
    expected_rows = torch.arange(BLOCK_LENGTH, device=batched_hidden.device)
    row_aligned = best_manual_rows.eq(expected_rows)

    batched_logits = target.lm_head(batched_hidden).float()
    manual_logits = target.lm_head(manual_hidden).float()
    batched_top2_values, batched_top2_ids = batched_logits.topk(2, dim=-1)
    manual_top2_values, manual_top2_ids = manual_logits.topk(2, dim=-1)
    top1_equal = batched_top2_ids[:, 0].eq(manual_top2_ids[:, 0])
    batched_centers = batched_logits.gather(-1, batched_top2_ids[:, :1])
    manual_centers = manual_logits.gather(-1, batched_top2_ids[:, :1])
    centered_error = (
        (batched_logits - batched_centers) - (manual_logits - manual_centers)
    ).abs().amax(dim=-1)
    batched_margin = batched_top2_values[:, 0] - batched_top2_values[:, 1]
    manual_margin = manual_top2_values[:, 0] - manual_top2_values[:, 1]
    return {
        "max_abs": max_abs,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "cosine": cosine,
        "best_manual_rows": best_manual_rows,
        "row_aligned": row_aligned,
        "batched_top1": batched_top2_ids[:, 0],
        "manual_top1": manual_top2_ids[:, 0],
        "top1_equal": top1_equal,
        "centered_logit_error": centered_error,
        "batched_margin": batched_margin,
        "manual_margin": manual_margin,
    }


def calibrate_manual_numeric_epsilon(
    receipts: list[dict[str, Tensor]],
) -> float:
    """Freeze a manual-geometry tolerance using agreeing rows only."""

    agreeing_error_rows = [
        receipt["centered_logit_error"][receipt["top1_equal"]]
        for receipt in receipts
        if bool(receipt["top1_equal"].any().item())
    ]
    if not agreeing_error_rows:
        raise RuntimeError("manual geometry has no agreeing rows for calibration")
    epsilon = torch.cat(agreeing_error_rows).max()
    if not bool(torch.isfinite(epsilon).item()) or float(epsilon) < 0:
        raise RuntimeError("manual geometry calibrated a non-finite tolerance")
    return float(epsilon.item())


def stable_manual_mismatch_mask(
    receipt: dict[str, Tensor], epsilon: float
) -> Tensor:
    """Return top1 disagreements separated from the calibrated tie region."""

    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("manual geometry epsilon must be finite and non-negative")
    mismatches = ~receipt["top1_equal"]
    minimum_margin = torch.minimum(
        receipt["batched_margin"], receipt["manual_margin"]
    )
    return mismatches & minimum_margin.gt(2.0 * epsilon)


@torch.inference_mode()
def derive_one(
    target: AutoModelForCausalLM,
    weight_fp32: Tensor,
    record: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    weight = target.lm_head.weight
    hidden = record["parallel_hidden"].to(
        device=device, dtype=torch.bfloat16
    )
    if hidden.shape != (BLOCK_LENGTH, HIDDEN_SIZE):
        raise RuntimeError("PCLD sidecar requires full16 DFlash hidden")

    base_logits = F.linear(hidden.unsqueeze(0), weight)[0]
    base_top_logits, base_top_ids = base_logits.float().topk(CANDIDATES, dim=-1)
    expected_ids = record["base_topk_ids"].to(device=device, dtype=torch.long)
    expected_logits = record["base_topk_logits"].to(device=device, dtype=torch.float16)
    if not torch.equal(base_top_ids, expected_ids):
        raise RuntimeError(
            f"base Top16 ID replay mismatch for {record_key(record)}"
        )
    if not torch.equal(base_top_logits.to(torch.float16), expected_logits):
        raise RuntimeError(
            f"base Top16 stored-logit replay mismatch for {record_key(record)}"
        )
    base_lse = torch.logsumexp(base_logits.float(), dim=-1)

    teacher_hidden = target_teacher_hidden(target, record, device)
    authoritative_logits = target.lm_head(teacher_hidden)
    authoritative_float = authoritative_logits.float()
    top2_logits, top2_ids = authoritative_float.topk(2, dim=-1)
    authoritative_top1 = top2_ids[:, 0]
    target_margins = top2_logits[:, 0] - top2_logits[:, 1]
    teacher_candidate_logits = authoritative_float.gather(-1, expected_ids)

    fp32_logits = F.linear(teacher_hidden.float(), weight_fp32)
    fp32_top1 = fp32_logits.argmax(dim=-1)
    authoritative_centers = authoritative_float.gather(
        -1, authoritative_top1.unsqueeze(-1)
    )
    fp32_centers = fp32_logits.gather(-1, authoritative_top1.unsqueeze(-1))
    centered_errors = (
        (authoritative_float - authoritative_centers)
        - (fp32_logits - fp32_centers)
    ).abs().amax(dim=-1)

    base_fp32 = F.linear(hidden.float(), weight_fp32).gather(-1, expected_ids)
    candidate_rows = weight_fp32[expected_ids]
    residual = teacher_hidden.float() - hidden.float()
    cancellation = base_fp32 + torch.einsum(
        "lkh,lh->lk", candidate_rows, residual
    )
    fp32_teacher_candidates = fp32_logits.gather(-1, expected_ids)
    cancellation_error = float(
        (cancellation - fp32_teacher_candidates).abs().max().item()
    )

    key = record_key(record)
    item = {
        "sample_id": key[0],
        "anchor_offset": key[1],
        "context_length": key[2],
        "base_logsumexp": base_lse.cpu().float(),
        "base_candidate_logits": base_top_logits.cpu().float(),
        "target_hidden": teacher_hidden.cpu().to(torch.bfloat16),
        "target_candidate_logits": teacher_candidate_logits.cpu().float(),
        "authoritative_top1_ids": authoritative_top1.cpu().to(torch.int32),
        "fp32_top1_ids": fp32_top1.cpu().to(torch.int32),
        "target_top1_margins": target_margins.cpu().float(),
        "centered_max_errors": centered_errors.cpu().float(),
        "residual_cancellation_max_error": cancellation_error,
    }
    diagnostics = {
        "numeric_top1_disagreements": int(
            authoritative_top1.ne(fp32_top1).sum().item()
        ),
        "max_centered_error": float(centered_errors.max().item()),
        "max_residual_cancellation_error": cancellation_error,
    }
    return item, diagnostics


def compare_item(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    hidden_atol: float,
    hidden_rtol: float,
    score_atol: float,
    score_rtol: float,
) -> dict[str, float]:
    exact_fields = ("authoritative_top1_ids", "fp32_top1_ids")
    for name in exact_fields:
        if not torch.equal(actual[name].cpu(), expected[name].cpu()):
            raise RuntimeError(f"PCLD replay exact field mismatch: {name}")
    tolerances = {
        "base_logsumexp": (score_atol, score_rtol),
        "base_candidate_logits": (score_atol, score_rtol),
        "target_hidden": (hidden_atol, hidden_rtol),
        "target_candidate_logits": (score_atol, score_rtol),
        "target_top1_margins": (score_atol, score_rtol),
        "centered_max_errors": (score_atol, score_rtol),
    }
    errors: dict[str, float] = {}
    for name, (atol, rtol) in tolerances.items():
        left = actual[name].float().cpu()
        right = expected[name].float().cpu()
        error = float((left - right).abs().max().item())
        errors[name] = error
        if not torch.allclose(left, right, atol=atol, rtol=rtol):
            raise RuntimeError(
                f"PCLD replay mismatch for {name}: max_abs={error}"
            )
    scalar_error = abs(
        float(actual["residual_cancellation_max_error"])
        - float(expected["residual_cancellation_max_error"])
    )
    errors["residual_cancellation_max_error"] = scalar_error
    if scalar_error > score_atol:
        raise RuntimeError("PCLD residual-cancellation replay mismatch")
    return errors


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_records < 0 or args.manual_parity_records < 0:
        raise ValueError("record limits must be non-negative")
    if args.mode == "materialize":
        if args.output is None or args.sidecar is not None:
            raise ValueError("materialize requires --output and forbids --sidecar")
    elif args.sidecar is None or args.output is not None:
        raise ValueError("verify requires --sidecar and forbids --output")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("claim-bearing PCLD sidecars require CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        # The numerical calibration contract is an actual FP32 replay, not a
        # TF32 approximation hidden behind a float32 tensor dtype.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    validate_target_checkpoint(args.target)
    source_metadata, records = select_records(args)

    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0" if device.type == "cuda" else "cpu",
    ).eval()
    target.requires_grad_(False)
    weight_fp32 = target.lm_head.weight.detach().float()

    writer: SidecarWriter | None = None
    expected_values: dict[tuple[str, int, int], dict[str, Any]] | None = None
    sidecar_metadata: dict[str, Any] | None = None
    if args.mode == "materialize":
        assert args.output is not None
        writer = SidecarWriter(args.output, shard_size=args.shard_size)
    else:
        assert args.sidecar is not None
        sidecar_metadata, expected_values = load_pcld_sidecar(args.sidecar)
        if int(sidecar_metadata.get("records", -1)) != len(records):
            raise RuntimeError("PCLD selected records differ from sidecar count")

    manual_count = min(args.manual_parity_records, len(records))
    manual_max_hidden_error = 0.0
    manual_max_hidden_rmse = 0.0
    manual_max_hidden_relative_rmse = 0.0
    manual_min_hidden_cosine = 1.0
    manual_max_centered_logit_error = 0.0
    manual_row0_hidden_error = 0.0
    manual_row15_hidden_error = 0.0
    manual_row_alignment_mismatches = 0
    manual_row0_alignment_mismatches = 0
    manual_row15_alignment_mismatches = 0
    manual_top1_mismatches = 0
    manual_ambiguous_top1_mismatches = 0
    manual_stable_top1_mismatches = 0
    manual_row0_top1_mismatches = 0
    manual_row15_top1_mismatches = 0
    manual_row0_stable_top1_mismatches = 0
    manual_row15_stable_top1_mismatches = 0
    manual_numeric_epsilon: float | None = None
    manual_receipts: list[dict[str, Tensor]] = []
    numeric_disagreements = 0
    max_centered_error = 0.0
    max_cancellation_error = 0.0
    replay_errors: dict[str, float] = {}
    started = time.perf_counter()
    for index, record in enumerate(records):
        item, diagnostic = derive_one(target, weight_fp32, record, device)
        numeric_disagreements += diagnostic["numeric_top1_disagreements"]
        max_centered_error = max(max_centered_error, diagnostic["max_centered_error"])
        max_cancellation_error = max(
            max_cancellation_error,
            diagnostic["max_residual_cancellation_error"],
        )
        if writer is not None:
            writer.add(item)
        else:
            assert expected_values is not None
            expected = expected_values.get(record_key(record))
            if expected is None:
                raise RuntimeError(f"PCLD sidecar lacks {record_key(record)}")
            errors = compare_item(
                item,
                expected,
                hidden_atol=args.hidden_atol,
                hidden_rtol=args.hidden_rtol,
                score_atol=args.score_atol,
                score_rtol=args.score_rtol,
            )
            for name, value in errors.items():
                replay_errors[name] = max(replay_errors.get(name, 0.0), value)

        if index < manual_count:
            manual_hidden = manual_prefix_hidden(target, record, device)
            batched_hidden = item["target_hidden"].to(
                device=device, dtype=torch.bfloat16
            )
            geometry = manual_geometry_receipt(target, batched_hidden, manual_hidden)
            authoritative_top1 = item["authoritative_top1_ids"].to(
                device=device, dtype=torch.long
            )
            if not torch.equal(geometry["batched_top1"], authoritative_top1):
                raise RuntimeError("manual receipt failed to reproduce batched authority")
            error = float(geometry["max_abs"].max().item())
            manual_max_hidden_error = max(manual_max_hidden_error, error)
            manual_max_hidden_rmse = max(
                manual_max_hidden_rmse, float(geometry["rmse"].max().item())
            )
            manual_max_hidden_relative_rmse = max(
                manual_max_hidden_relative_rmse,
                float(geometry["relative_rmse"].max().item()),
            )
            manual_min_hidden_cosine = min(
                manual_min_hidden_cosine, float(geometry["cosine"].min().item())
            )
            manual_max_centered_logit_error = max(
                manual_max_centered_logit_error,
                float(geometry["centered_logit_error"].max().item()),
            )
            manual_row0_hidden_error = max(
                manual_row0_hidden_error,
                float(geometry["max_abs"][0].item()),
            )
            manual_row15_hidden_error = max(
                manual_row15_hidden_error,
                float(geometry["max_abs"][15].item()),
            )
            row_mismatches = ~geometry["row_aligned"]
            row_mismatch_count = int(row_mismatches.sum().item())
            manual_row_alignment_mismatches += row_mismatch_count
            manual_row0_alignment_mismatches += int(row_mismatches[0].item())
            manual_row15_alignment_mismatches += int(row_mismatches[15].item())
            if row_mismatch_count:
                raise RuntimeError(
                    f"manual target-prefix row alignment failed for "
                    f"{record_key(record)} at {row_mismatch_count} rows"
                )
            current_mismatches = ~geometry["top1_equal"]
            mismatch_count = int(current_mismatches.sum().item())
            manual_top1_mismatches += mismatch_count
            manual_row0_top1_mismatches += int(current_mismatches[0].item())
            manual_row15_top1_mismatches += int(current_mismatches[15].item())
            manual_receipts.append(geometry)

        if (index + 1) % 16 == 0 or index + 1 == len(records):
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "progress": index + 1,
                        "records": len(records),
                    }
                ),
                flush=True,
            )

    if manual_receipts:
        manual_numeric_epsilon = calibrate_manual_numeric_epsilon(manual_receipts)
        for receipt in manual_receipts:
            mismatches = ~receipt["top1_equal"]
            stable_mismatches = stable_manual_mismatch_mask(
                receipt, manual_numeric_epsilon
            )
            ambiguous_mismatches = mismatches & ~stable_mismatches
            manual_ambiguous_top1_mismatches += int(
                ambiguous_mismatches.sum().item()
            )
            manual_stable_top1_mismatches += int(stable_mismatches.sum().item())
            manual_row0_stable_top1_mismatches += int(
                stable_mismatches[0].item()
            )
            manual_row15_stable_top1_mismatches += int(
                stable_mismatches[15].item()
            )
        if manual_stable_top1_mismatches:
            raise RuntimeError(
                "manual target-prefix stable-token parity failed at "
                f"{manual_stable_top1_mismatches} rows with "
                f"epsilon={manual_numeric_epsilon}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if writer is not None:
        writer.flush()
        report = {
            "format": "pcld_sidecar_v1",
            "collection_complete": True,
            "source_rollout": str(args.rollout.resolve()),
            "source_format": source_metadata.get("format"),
            "target": str(args.target.resolve()),
            "split": args.split,
            "group": args.group,
            "records": writer.total,
            "block_length": BLOCK_LENGTH,
            "candidates": CANDIDATES,
            "teacher_geometry": "context+anchor+gold[0:15] -> rows anchor..gold14",
            "lexical_authority": "target.lm_head.weight",
            "base_lattice_exact": True,
            "manual_parity_records": manual_count,
            "manual_hidden_max_abs_error": manual_max_hidden_error,
            "manual_hidden_max_rmse": manual_max_hidden_rmse,
            "manual_hidden_max_relative_rmse": manual_max_hidden_relative_rmse,
            "manual_hidden_min_cosine": manual_min_hidden_cosine,
            "manual_centered_logit_max_error": manual_max_centered_logit_error,
            "manual_numeric_epsilon": manual_numeric_epsilon,
            "manual_row0_hidden_max_abs_error": manual_row0_hidden_error,
            "manual_row15_hidden_max_abs_error": manual_row15_hidden_error,
            "manual_row_alignment_mismatches": manual_row_alignment_mismatches,
            "manual_row0_alignment_mismatches": manual_row0_alignment_mismatches,
            "manual_row15_alignment_mismatches": manual_row15_alignment_mismatches,
            "manual_top1_mismatches": manual_top1_mismatches,
            "manual_ambiguous_top1_mismatches": manual_ambiguous_top1_mismatches,
            "manual_stable_top1_mismatches": manual_stable_top1_mismatches,
            "manual_row0_top1_mismatches": manual_row0_top1_mismatches,
            "manual_row15_top1_mismatches": manual_row15_top1_mismatches,
            "manual_row0_stable_top1_mismatches": manual_row0_stable_top1_mismatches,
            "manual_row15_stable_top1_mismatches": manual_row15_stable_top1_mismatches,
            "manual_row_alignment_exact": manual_row_alignment_mismatches == 0,
            "manual_stable_top1_exact": manual_stable_top1_mismatches == 0,
            "manual_row0_alignment_exact": manual_row0_alignment_mismatches == 0,
            "manual_row15_alignment_exact": manual_row15_alignment_mismatches == 0,
            "manual_row0_stable_top1_exact": manual_row0_stable_top1_mismatches == 0,
            "manual_row15_stable_top1_exact": manual_row15_stable_top1_mismatches == 0,
            # Raw equality remains an explicit diagnostic; BF16 ties are not
            # relabeled as exact merely because the stable contract allows them.
            "manual_top1_exact": manual_top1_mismatches == 0,
            "manual_row0_top1_exact": manual_row0_top1_mismatches == 0,
            "manual_row15_top1_exact": manual_row15_top1_mismatches == 0,
            "manual_parity_passed": manual_count > 0
            and manual_row_alignment_mismatches == 0
            and manual_stable_top1_mismatches == 0,
            "numeric_top1_disagreements": numeric_disagreements,
            "max_centered_error": max_centered_error,
            "max_residual_cancellation_error": max_cancellation_error,
            "seconds": elapsed,
            "peak_memory_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda"
                else 0.0
            ),
            "shards": writer.shards,
        }
        assert args.output is not None
        (args.output / "metadata.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    else:
        assert args.sidecar is not None and sidecar_metadata is not None
        manual_fields: dict[str, Any] = {
            "manual_parity_records": manual_count,
            "manual_hidden_max_abs_error": manual_max_hidden_error,
            "manual_hidden_max_rmse": manual_max_hidden_rmse,
            "manual_hidden_max_relative_rmse": manual_max_hidden_relative_rmse,
            "manual_hidden_min_cosine": manual_min_hidden_cosine,
            "manual_centered_logit_max_error": manual_max_centered_logit_error,
            "manual_numeric_epsilon": manual_numeric_epsilon,
            "manual_row0_hidden_max_abs_error": manual_row0_hidden_error,
            "manual_row15_hidden_max_abs_error": manual_row15_hidden_error,
            "manual_row_alignment_mismatches": manual_row_alignment_mismatches,
            "manual_row0_alignment_mismatches": manual_row0_alignment_mismatches,
            "manual_row15_alignment_mismatches": manual_row15_alignment_mismatches,
            "manual_top1_mismatches": manual_top1_mismatches,
            "manual_ambiguous_top1_mismatches": manual_ambiguous_top1_mismatches,
            "manual_stable_top1_mismatches": manual_stable_top1_mismatches,
            "manual_row0_top1_mismatches": manual_row0_top1_mismatches,
            "manual_row15_top1_mismatches": manual_row15_top1_mismatches,
            "manual_row0_stable_top1_mismatches": manual_row0_stable_top1_mismatches,
            "manual_row15_stable_top1_mismatches": manual_row15_stable_top1_mismatches,
            "manual_row_alignment_exact": manual_row_alignment_mismatches == 0,
            "manual_stable_top1_exact": manual_stable_top1_mismatches == 0,
            "manual_row0_alignment_exact": manual_row0_alignment_mismatches == 0,
            "manual_row15_alignment_exact": manual_row15_alignment_mismatches == 0,
            "manual_row0_stable_top1_exact": manual_row0_stable_top1_mismatches == 0,
            "manual_row15_stable_top1_exact": manual_row15_stable_top1_mismatches == 0,
            "manual_top1_exact": manual_top1_mismatches == 0,
            "manual_row0_top1_exact": manual_row0_top1_mismatches == 0,
            "manual_row15_top1_exact": manual_row15_top1_mismatches == 0,
            "manual_parity_passed": manual_count > 0
            and manual_row_alignment_mismatches == 0
            and manual_stable_top1_mismatches == 0,
        }
        if not manual_count:
            manual_fields = {
                name: sidecar_metadata.get(name) for name in manual_fields
            }
        report = {
            "format": "pcld_sidecar_replay_v1",
            "verified": True,
            "source_rollout": str(args.rollout.resolve()),
            "target": str(args.target.resolve()),
            "split": args.split,
            "group": args.group,
            "sidecar": str(args.sidecar.resolve()),
            "records": len(records),
            "base_lattice_exact": True,
            "target_hidden_allclose": True,
            "target_candidate_scores_allclose": True,
            "numeric_authority_exact": True,
            **manual_fields,
            "max_replay_errors": replay_errors,
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
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
