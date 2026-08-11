#!/usr/bin/env python3
"""Profile and validate the real R048 four-layer KV-reuse verifier path."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import statistics
from typing import Any, Callable

import torch
from transformers import AutoModel, AutoModelForCausalLM

from sph.fast_r048 import (
    candidate_union_with_proposal,
    fast_candidate_domino_decode,
    repair_earliest_frontier,
)
from sph.r048_layer_split import (
    clone_dynamic_cache,
    layer_split_verifier_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-index", type=int, default=723)
    parser.add_argument("--max-distribution-records", type=int)
    parser.add_argument("--candidate-topk", type=int, default=64)
    parser.add_argument("--early-layers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--domino-graph-head-ms", type=float, default=2.110945281982422)
    parser.add_argument("--minimum-throughput-ratio", type=float, default=1.20)
    parser.add_argument("--maximum-token-path-mismatches", type=int, default=0)
    return parser.parse_args()


def load_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        rows.extend(torch.load(shard, map_location="cpu", weights_only=False))
    if not rows:
        raise ValueError("empty R048 collection")
    return rows


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


def timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p50": statistics.median(ordered),
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "mean": statistics.fmean(ordered),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    records = load_records(args.collection)
    if not 0 <= args.record_index < len(records):
        raise IndexError("record index lies outside collection")
    record = records[args.record_index]
    distribution_records = records
    if args.max_distribution_records is not None:
        if args.max_distribution_records < 1:
            raise ValueError("max distribution records must be positive")
        distribution_records = records[: args.max_distribution_records]
    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    target.requires_grad_(False)
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    domino.requires_grad_(False)
    target_weight = target.model.embed_tokens.weight

    # Determine the actual Fast-K frontier distribution used to weight suffix
    # recomputation latency.  This is batch-1 and follows the oracle path.
    repair_histogram: Counter[int | str] = Counter()
    for row in distribution_records:
        hidden = row["parallel_hidden"].to(device, torch.bfloat16)[None]
        anchor = torch.tensor(
            [int(row["anchor_token_id"])], dtype=torch.long, device=device
        )
        gold = row["gold_ids"].long().to(device)[None]
        proposal = fast_candidate_domino_decode(
            domino=domino,
            target_weight=target_weight,
            anchors=anchor,
            hidden=hidden,
            candidate_topk=args.candidate_topk,
        )
        support = candidate_union_with_proposal(
            proposal.candidate_ids,
            proposal.token_ids,
            support_size=args.candidate_topk,
        )
        repair = repair_earliest_frontier(
            proposal.token_ids, gold, candidate_ids=support
        )
        if bool(repair.changed[0]):
            repair_histogram[int(repair.frontier[0])] += 1
        else:
            repair_histogram["no_change"] += 1

    hidden = record["parallel_hidden"].to(device, torch.bfloat16)[None]
    anchor = torch.tensor(
        [int(record["anchor_token_id"])], dtype=torch.long, device=device
    )
    proposal = fast_candidate_domino_decode(
        domino=domino,
        target_weight=target_weight,
        anchors=anchor,
        hidden=hidden,
        candidate_topk=args.candidate_topk,
    ).token_ids
    original_inputs = torch.cat([anchor[:, None], proposal], dim=1)
    context = record["context_ids_before_anchor"].long().to(device)[None]
    prefix_length = int(context.shape[1])
    prefix_outputs = target.model(context, use_cache=True, return_dict=True)
    prefix_cache = prefix_outputs.past_key_values

    # Compare split versus unsplit token paths for no-change and three repair
    # locations.  The changed token values are arbitrary but deterministic;
    # only prefix geometry and numerical equivalence matter here.
    correctness: dict[str, Any] = {}
    total_token_mismatches = 0
    for correction_position in (None, 0, 8, 15):
        final_inputs = original_inputs.clone()
        if correction_position is not None:
            column = correction_position + 1
            final_inputs[:, column] = (final_inputs[:, column] + 1) % target.config.vocab_size
        reference_cache = clone_dynamic_cache(prefix_cache, config=target.config)
        split_cache = clone_dynamic_cache(prefix_cache, config=target.config)
        reference = target(
            final_inputs,
            past_key_values=reference_cache,
            use_cache=True,
            return_dict=True,
        ).logits
        split = layer_split_verifier_forward(
            target=target,
            cache=split_cache,
            original_input_ids=original_inputs,
            final_input_ids=final_inputs,
            prefix_length=prefix_length,
            early_layers=args.early_layers,
            correction_position=correction_position,
        )
        mismatch = int(reference.argmax(dim=-1).ne(split.logits.argmax(dim=-1)).sum())
        total_token_mismatches += mismatch
        difference = reference.float() - split.logits.float()
        correctness[str(correction_position)] = {
            "token_path_mismatches": mismatch,
            "logit_relative_rms": float(
                difference.square().mean().sqrt()
                / reference.float().square().mean().sqrt().clamp_min(1e-12)
            ),
            "max_absolute_logit_delta": float(difference.abs().max()),
            "decision_state_shape": list(split.decision_states.shape),
        }

    timing_cache = clone_dynamic_cache(prefix_cache, config=target.config)

    def full_callback() -> Any:
        timing_cache.crop(prefix_length)
        return target(
            original_inputs,
            past_key_values=timing_cache,
            use_cache=True,
            return_dict=True,
        ).logits

    full_timing = timing_summary(
        event_samples(full_callback, warmup=args.warmup, repeats=args.repeats)
    )
    split_timings: dict[str, Any] = {}
    for correction_position in [None, *range(16)]:
        final_inputs = original_inputs.clone()
        if correction_position is not None:
            column = correction_position + 1
            final_inputs[:, column] = (final_inputs[:, column] + 1) % target.config.vocab_size

        def split_callback(
            final_ids: torch.Tensor = final_inputs,
            position: int | None = correction_position,
        ) -> Any:
            timing_cache.crop(prefix_length)
            return layer_split_verifier_forward(
                target=target,
                cache=timing_cache,
                original_input_ids=original_inputs,
                final_input_ids=final_ids,
                prefix_length=prefix_length,
                early_layers=args.early_layers,
                correction_position=position,
            ).logits

        split_timings[str(correction_position)] = timing_summary(
            event_samples(split_callback, warmup=args.warmup, repeats=args.repeats)
        )

    total_blocks = sum(repair_histogram.values())
    weighted_split_p50 = 0.0
    for key, count in repair_histogram.items():
        timing_key = "None" if key == "no_change" else str(key)
        weighted_split_p50 += (
            count / total_blocks * float(split_timings[timing_key]["p50"])
        )
    target_split_overhead = weighted_split_p50 - float(full_timing["p50"])

    candidate_profile = json.loads(args.candidate_profile.read_text(encoding="utf-8"))
    candidate_head_ms = float(
        candidate_profile["candidate_widths"][str(args.candidate_topk)][
            "cuda_graph_candidate_plus_lens_ms"
        ]["p50"]
    )
    oracle = json.loads(args.oracle_report.read_text(encoding="utf-8"))
    output_ratio = float(
        oracle["metrics"]["ideal_output_ratio_one_repair_vs_released"]
    )
    released_noncommon = args.domino_graph_head_ms + float(full_timing["p50"])
    fast_noncommon = candidate_head_ms + weighted_split_p50
    denominator = args.minimum_throughput_ratio - output_ratio
    break_even_common_ms = (
        (output_ratio * released_noncommon - args.minimum_throughput_ratio * fast_noncommon)
        / denominator
        if denominator > 0
        else float("inf")
    )
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "record": {
            "index": args.record_index,
            "sample_id": str(record["sample_id"]),
            "context_length": prefix_length,
            "request_batch_size": 1,
        },
        "correctness": correctness,
        "total_token_path_mismatches": total_token_mismatches,
        "fast_k_repair_histogram": {str(key): value for key, value in repair_histogram.items()},
        "repair_distribution_blocks": len(distribution_records),
        "latency_ms": {
            "unsplit_full_target_verifier": full_timing,
            "split_by_repair_position": split_timings,
            "weighted_split_target_p50": weighted_split_p50,
            "weighted_split_overhead_vs_unsplit_p50": target_split_overhead,
            "released_domino_graph_head": args.domino_graph_head_ms,
            "fast_candidate_plus_lens_graph": candidate_head_ms,
            "released_noncommon_cycle": released_noncommon,
            "fast_noncommon_cycle": fast_noncommon,
        },
        "ideal_throughput_gate": {
            "oracle_output_ratio": output_ratio,
            "minimum_ratio": args.minimum_throughput_ratio,
            "maximum_allowed_shared_dflash_plus_base_ms": break_even_common_ms,
            "requires_common_path_profile": True,
        },
        "checks": {
            "maximum_token_path_mismatches": args.maximum_token_path_mismatches,
            "token_path_gate_passed": total_token_mismatches
            <= args.maximum_token_path_mismatches,
        },
        "benchmark": {"warmup": args.warmup, "repeats": args.repeats},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not bool(report["checks"]["token_path_gate_passed"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
