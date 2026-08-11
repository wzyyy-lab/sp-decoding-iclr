#!/usr/bin/env python3
"""Profile Fast-R048 candidate selection and the 180K tuned lens on A40."""

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

from sph.fast_r048 import (
    R048TunedLens,
    earliest_one_decision,
    fast_candidate_domino_decode,
    fast_candidate_domino_decode_from_base,
)
from train_domino_cached_head import load_tensor_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-topks", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--maximum-k64-delta-ms", type=float, default=0.10)
    return parser.parse_args()


def load_record(root: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("record index must be nonnegative")
    offset = index
    for shard in sorted(root.glob("shard-*.pt")):
        rows = torch.load(shard, map_location="cpu", weights_only=False)
        if offset < len(rows):
            return rows[offset]
        offset -= len(rows)
    raise IndexError("record index lies outside collection")


def event_samples(
    callback: Callable[[], Any], *, warmup: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callback()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p50": statistics.median(ordered),
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "mean": statistics.fmean(ordered),
    }


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
    # The output tensors own graph-pool allocations and must remain alive for
    # every replay even though timing does not inspect their values.
    return graph, output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 1 or args.repeats < 10:
        raise ValueError("insufficient timing repetitions")
    if len(set(args.candidate_topks)) != len(args.candidate_topks):
        raise ValueError("candidate widths must be unique")

    record = load_record(args.collection, args.record_index)
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
        [int(record["anchor_token_id"])], device=device, dtype=torch.long
    )
    base_logits = F.linear(hidden, target_weight)
    stored_anchor = record["target_anchor_early_feature"].to(
        device, torch.bfloat16
    )
    early_states = stored_anchor.view(1, 1, -1).expand(1, 16, -1).contiguous()

    base_timing = summary(
        event_samples(
            lambda: F.linear(hidden, target_weight),
            warmup=args.warmup,
            repeats=args.repeats,
        )
    )
    rows: dict[str, Any] = {}
    for candidate_topk in args.candidate_topks:
        eager_output = fast_candidate_domino_decode_from_base(
            domino=domino,
            target_weight=target_weight,
            anchors=anchor,
            hidden=hidden,
            base_logits=base_logits,
            candidate_topk=candidate_topk,
        )
        full_output = fast_candidate_domino_decode(
            domino=domino,
            target_weight=target_weight,
            anchors=anchor,
            hidden=hidden,
            candidate_topk=candidate_topk,
        )
        if not torch.equal(eager_output.token_ids, full_output.token_ids):
            raise RuntimeError("precomputed-base and complete Fast-K paths differ")
        if not bool(
            eager_output.candidate_ids.eq(
                eager_output.token_ids.unsqueeze(-1)
            ).any(dim=-1).all()
        ):
            raise RuntimeError("Fast-K support dropped its proposal before capture")
        lens = R048TunedLens(
            hidden_width=int(early_states.shape[-1]),
            rank=64,
            candidate_basis=domino.embed_proj[2].weight,
        ).eval()
        if lens.trainable_parameter_count != 180_224:
            raise RuntimeError("R048 tuned lens no longer has 180,224 parameters")
        decision_threshold = torch.tensor(
            0.05, dtype=torch.float32, device=device
        )

        def candidate_callback() -> Any:
            return fast_candidate_domino_decode_from_base(
                domino=domino,
                target_weight=target_weight,
                anchors=anchor,
                hidden=hidden,
                base_logits=base_logits,
                candidate_topk=candidate_topk,
            )

        def lens_callback() -> Any:
            return lens(early_states, eager_output.candidate_ids)

        def combined_callback() -> Any:
            proposal = fast_candidate_domino_decode_from_base(
                domino=domino,
                target_weight=target_weight,
                anchors=anchor,
                hidden=hidden,
                base_logits=base_logits,
                candidate_topk=candidate_topk,
            )
            delta = lens(early_states, proposal.candidate_ids)
            return earliest_one_decision(
                candidate_ids=proposal.candidate_ids,
                candidate_scores=proposal.candidate_scores,
                lens_delta=delta,
                proposal=proposal.token_ids,
                threshold=decision_threshold,
            )

        eager_candidate = summary(
            event_samples(candidate_callback, warmup=args.warmup, repeats=args.repeats)
        )
        eager_lens = summary(
            event_samples(lens_callback, warmup=args.warmup, repeats=args.repeats)
        )
        eager_combined = summary(
            event_samples(combined_callback, warmup=args.warmup, repeats=args.repeats)
        )
        candidate_graph, candidate_graph_output = capture_graph(candidate_callback)
        combined_graph, combined_graph_output = capture_graph(combined_callback)
        graph_candidate = summary(
            event_samples(candidate_graph.replay, warmup=args.warmup, repeats=args.repeats)
        )
        graph_combined = summary(
            event_samples(combined_graph.replay, warmup=args.warmup, repeats=args.repeats)
        )
        if candidate_graph_output is None or combined_graph_output is None:
            raise RuntimeError("CUDA graph did not retain its output")
        rows[str(candidate_topk)] = {
            "eager_candidate_without_base_gemm_ms": eager_candidate,
            "eager_tuned_lens_ms": eager_lens,
            "eager_candidate_plus_lens_ms": eager_combined,
            "cuda_graph_candidate_without_base_gemm_ms": graph_candidate,
            "cuda_graph_candidate_plus_lens_ms": graph_combined,
            "cuda_graph_complete_earliest_one_decision_ms": graph_combined,
            "proposal_matches_complete_path": True,
        }

    k32 = rows.get("32")
    k64 = rows.get("64")
    if k32 is None or k64 is None:
        raise ValueError("latency gate requires both K32 and K64")
    k64_delta = (
        float(k64["cuda_graph_candidate_plus_lens_ms"]["p50"])
        - float(k32["cuda_graph_candidate_plus_lens_ms"]["p50"])
    )
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "record": {
            "index": args.record_index,
            "sample_id": str(record["sample_id"]),
            "horizon": int(hidden.shape[1]),
            "request_batch_size": 1,
        },
        "shared_base_vocab_gemm_ms": base_timing,
        "candidate_widths": rows,
        "gate": {
            "k64_graph_p50_delta_vs_k32_ms": k64_delta,
            "maximum_delta_ms": args.maximum_k64_delta_ms,
            "passed": k64_delta <= args.maximum_k64_delta_ms,
        },
        "benchmark": {"warmup": args.warmup, "repeats": args.repeats},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not bool(report["gate"]["passed"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
