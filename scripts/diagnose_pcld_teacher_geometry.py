#!/usr/bin/env python3
"""Diagnostic-only microscope for batched versus prefix PCLD teacher rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from materialize_pcld_sidecar import manual_prefix_hidden, target_teacher_hidden
from sph.japd_data import load_rollout_records, record_key
from sph.pcld_data import select_balanced_smoke_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--records", type=int, default=6)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "min": float(tensor.min()),
        "median": float(tensor.median()),
        "mean": float(tensor.mean()),
        "max": float(tensor.max()),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.records < 3:
        raise ValueError("geometry diagnostic needs at least three records")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("teacher geometry diagnostic requires CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    _, raw_records = load_rollout_records(args.rollout, split=args.split)
    records = select_balanced_smoke_records(raw_records, count=args.records)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0" if device.type == "cuda" else "cpu",
    ).eval()
    target.requires_grad_(False)

    details: list[dict[str, Any]] = []
    all_max_abs: list[float] = []
    all_rmse: list[float] = []
    all_relative_rmse: list[float] = []
    all_cosine: list[float] = []
    all_centered_errors: list[float] = []
    top1_matches = 0
    rows = 0
    diagonal_best_rows = 0
    for index, record in enumerate(records):
        batched = target_teacher_hidden(target, record, device).float()
        manual = manual_prefix_hidden(target, record, device).float()
        difference = batched - manual
        max_abs = difference.abs().amax(dim=-1)
        rmse = difference.square().mean(dim=-1).sqrt()
        reference_rms = batched.square().mean(dim=-1).sqrt().clamp_min(1e-12)
        relative_rmse = rmse / reference_rms
        cosine = F.cosine_similarity(batched, manual, dim=-1)

        batched_logits = target.lm_head(batched.to(torch.bfloat16)).float()
        manual_logits = target.lm_head(manual.to(torch.bfloat16)).float()
        batched_top2_values, batched_top2_ids = batched_logits.topk(2, dim=-1)
        manual_top2_values, manual_top2_ids = manual_logits.topk(2, dim=-1)
        top1_equal = batched_top2_ids[:, 0].eq(manual_top2_ids[:, 0])
        centers_batched = batched_logits.gather(
            -1, batched_top2_ids[:, :1]
        )
        centers_manual = manual_logits.gather(
            -1, batched_top2_ids[:, :1]
        )
        centered_error = (
            (batched_logits - centers_batched)
            - (manual_logits - centers_manual)
        ).abs().amax(dim=-1)

        normalized_batched = F.normalize(batched, dim=-1)
        normalized_manual = F.normalize(manual, dim=-1)
        similarity = normalized_batched @ normalized_manual.T
        best_manual_rows = similarity.argmax(dim=-1)
        expected_rows = torch.arange(16, device=device)
        diagonal_best = best_manual_rows.eq(expected_rows)

        top1_matches += int(top1_equal.sum().item())
        rows += 16
        diagonal_best_rows += int(diagonal_best.sum().item())
        all_max_abs.extend(max_abs.cpu().tolist())
        all_rmse.extend(rmse.cpu().tolist())
        all_relative_rmse.extend(relative_rmse.cpu().tolist())
        all_cosine.extend(cosine.cpu().tolist())
        all_centered_errors.extend(centered_error.cpu().tolist())
        detail = {
            "key": list(record_key(record)),
            "domain": str(record["domain"]),
            "context_length": int(record["context_length"]),
            "max_abs_by_row": max_abs.cpu().tolist(),
            "rmse_by_row": rmse.cpu().tolist(),
            "relative_rmse_by_row": relative_rmse.cpu().tolist(),
            "cosine_by_row": cosine.cpu().tolist(),
            "centered_logit_max_error_by_row": centered_error.cpu().tolist(),
            "batched_top1_ids": batched_top2_ids[:, 0].cpu().tolist(),
            "manual_top1_ids": manual_top2_ids[:, 0].cpu().tolist(),
            "top1_equal_by_row": top1_equal.cpu().tolist(),
            "batched_margin_by_row": (
                batched_top2_values[:, 0] - batched_top2_values[:, 1]
            ).cpu().tolist(),
            "manual_margin_by_row": (
                manual_top2_values[:, 0] - manual_top2_values[:, 1]
            ).cpu().tolist(),
            "best_manual_row_by_batched_row": best_manual_rows.cpu().tolist(),
            "diagonal_is_best_by_row": diagonal_best.cpu().tolist(),
        }
        details.append(detail)
        print(
            json.dumps(
                {
                    "progress": index + 1,
                    "key": detail["key"],
                    "max_abs": max(detail["max_abs_by_row"]),
                    "max_relative_rmse": max(detail["relative_rmse_by_row"]),
                    "min_cosine": min(detail["cosine_by_row"]),
                    "top1_matches": sum(detail["top1_equal_by_row"]),
                    "diagonal_best": sum(detail["diagonal_is_best_by_row"]),
                }
            ),
            flush=True,
        )

    report = {
        "format": "pcld_teacher_geometry_diagnostic_v1",
        "authority": "diagnostic_only_not_a_claim_gate",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "attn_implementation": args.attn_implementation,
        "records": len(records),
        "rows": rows,
        "top1_matches": top1_matches,
        "top1_match_fraction": top1_matches / rows,
        "diagonal_best_rows": diagonal_best_rows,
        "diagonal_best_fraction": diagonal_best_rows / rows,
        "max_abs": distribution(all_max_abs),
        "rmse": distribution(all_rmse),
        "relative_rmse": distribution(all_relative_rmse),
        "cosine": distribution(all_cosine),
        "centered_logit_max_error": distribution(all_centered_errors),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())
