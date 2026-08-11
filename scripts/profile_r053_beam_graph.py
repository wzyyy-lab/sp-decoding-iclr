#!/usr/bin/env python3
"""CUDA-graph latency probe for the frozen R053 Fast-K64 beam.

The claim-bearing R053 result showed that the target-tree forward is not the
system bottleneck.  This probe isolates the remaining fixed-shape beam at
W={4,8,16}, verifies graph/eager token parity, and reports the exact head budget
implied by the measured N64 and full-W16 acceptance values.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
from typing import Any, Callable

import torch
from torch.nn import functional as F
from transformers import AutoModel

from collect_r048_capacity import load_source
from profile_r052_exact_prefix import event_samples, timing_summary
from sph.fast_r048 import fast_candidate_domino_decode_from_base
from sph.r053_tree import fast_candidate_domino_beam_from_base
from train_domino_cached_head import load_tensor_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--r053-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=200)
    return parser.parse_args()


def median_context_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        range(len(records)),
        key=lambda index: (int(records[index]["context_length"]), index),
    )
    return records[ordered[int(round(0.5 * (len(ordered) - 1)))]]


def capture_graph(callback: Callable[[], Any]) -> tuple[torch.cuda.CUDAGraph, Any]:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            callback()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = callback()
    return graph, output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R053 beam profiling requires CUDA")
    if torch.cuda.get_device_name(0) != "NVIDIA A40":
        raise RuntimeError("official R053 graph profile requires NVIDIA A40")
    if args.beam_widths != [4, 8, 16]:
        raise ValueError("official graph probe is frozen to W={4,8,16}")
    if args.warmup < 1 or args.repeats < 10:
        raise ValueError("profile requires warmup>=1 and repeats>=10")

    metadata, records = load_source(args.source_rollout, args.split)
    record = median_context_record(records)
    report053 = json.loads(args.r053_report.read_text(encoding="utf-8"))
    if str(report053.get("format")) != "r053_tree_budget_pareto_v1":
        raise ValueError("R053 report has the wrong format")
    if str(metadata.get("mode")) != "fixed":
        raise ValueError("beam profile requires the fixed rollout")

    device = torch.device("cuda:0")
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to(device=device, dtype=torch.bfloat16)
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    domino.requires_grad_(False)
    hidden = record["parallel_hidden"].to(device, torch.bfloat16)[None]
    anchor = torch.tensor(
        [int(record["anchor_token_id"])], dtype=torch.long, device=device
    )
    base_logits = F.linear(hidden, target_weight)
    fast_control = fast_candidate_domino_decode_from_base(
        domino=domino,
        target_weight=target_weight,
        anchors=anchor,
        hidden=hidden,
        base_logits=base_logits,
        candidate_topk=64,
    ).token_ids

    base_ms = timing_summary(
        event_samples(
            lambda: F.linear(hidden, target_weight),
            warmup=args.warmup,
            repeats=args.repeats,
        )
    )
    widths: dict[str, Any] = {}
    for width in args.beam_widths:
        def callback() -> Any:
            return fast_candidate_domino_beam_from_base(
                domino=domino,
                target_weight=target_weight,
                anchors=anchor,
                hidden=hidden,
                base_logits=base_logits,
                candidate_pool_topk=64,
                tree_support_size=16,
                beam_width=width,
            )

        eager = callback()
        if not torch.equal(eager.trunk_token_ids, fast_control):
            raise RuntimeError(f"W{width} protected trunk differs from Fast-K64")
        eager_ms = timing_summary(
            event_samples(callback, warmup=args.warmup, repeats=args.repeats)
        )
        graph, graph_output = capture_graph(callback)
        graph.replay()
        torch.cuda.synchronize()
        for field in (
            "token_ids",
            "edge_log_probs",
            "map_scores",
            "trunk_token_ids",
            "candidate_ids",
        ):
            if not torch.equal(getattr(graph_output, field), getattr(eager, field)):
                raise RuntimeError(f"W{width} CUDA graph changed {field}")
        graph_ms = timing_summary(
            event_samples(graph.replay, warmup=args.warmup, repeats=args.repeats)
        )
        widths[str(width)] = {
            "paths": width,
            "horizon": int(eager.token_ids.shape[-1]),
            "padded_forest_rows_with_shared_anchor": 1 + 16 * width,
            "eager_beam_without_base_gemm_ms": eager_ms,
            "cuda_graph_beam_without_base_gemm_ms": graph_ms,
            "graph_eager_token_parity": True,
            "graph_eager_all_output_tensor_parity": True,
            "trunk_matches_fast_k64_control": True,
            "protected_trunk_present": bool(
                eager.token_ids[0].eq(eager.trunk_token_ids).all(dim=-1).any()
            ),
        }

    clean_domino = float(report053["accuracy"]["clean_domino"]["overall"])
    n64_eal = float(
        report053["accuracy"]["budgets"]["64"]
        ["deployable_actual_tree_clean_prefix"]["overall"]
    )
    full_structural_eal = float(
        report053["accuracy"]["full_w16_pool_oracle"]["overall"]
    )
    profile053 = report053["profile"]
    domino_complete = float(
        profile053["latency_ms"]["domino_complete_noncommon_cycle"]["p50"]
    )
    n64_target = float(
        profile053["budgets"]["64"]["latency_ms"]
        ["one_target_tree_forward"]["p50"]
    )
    n64_traversal = float(
        profile053["budgets"]["64"]["latency_ms"]
        ["full_vocab_argmax_and_tree_traversal"]["p50"]
    )
    base_p50 = float(base_ms["p50"])

    def budget(eal: float) -> dict[str, float]:
        maximum_complete = (
            domino_complete * (eal + 1.0) / (clean_domino + 1.0) / 1.15
        )
        return {
            "eal": eal,
            "maximum_complete_cycle_ms_for_1p15": maximum_complete,
            "beam_plus_traversal_budget_using_n64_target_ms": (
                maximum_complete - base_p50 - n64_target
            ),
            "beam_only_budget_with_current_traversal_ms": (
                maximum_complete - base_p50 - n64_target - n64_traversal
            ),
        }

    report = {
        "status": "completed",
        "format": "r053_beam_cuda_graph_profile_v1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "record": {
            "sample_id": str(record["sample_id"]),
            "domain": str(record["domain"]),
            "context_length": int(record["context_length"]),
            "request_batch_size": 1,
        },
        "parameters": {
            "new_trainable_parameters": 0,
            "reuses_released_domino_gru_and_projection": True,
        },
        "latency_ms": {
            "base_vocab_gemm": base_ms,
            "beam_widths": widths,
        },
        "system_budgets": {
            "target_tps_ratio": 1.15,
            "domino_complete_noncommon_p50_ms": domino_complete,
            "r053_n64_target_tree_p50_ms": n64_target,
            "r053_n64_current_traversal_p50_ms": n64_traversal,
            "n64_actual_clean_eal": budget(n64_eal),
            "full_w16_structural_ceiling_optimistic": budget(full_structural_eal),
        },
        "benchmark": {"warmup": args.warmup, "repeats": args.repeats},
        "scope": (
            "Beam-only CUDA-graph feasibility. It does not include forest target "
            "latency, dynamic trie packing, or a lossless SGLang claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
