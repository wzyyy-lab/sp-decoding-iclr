#!/usr/bin/env python3
"""Fair eager A40 profile of Domino versus the selected R051 s=4 cycle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
from typing import Any, Callable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM

from collect_r048_capacity import load_source
from sph.fast_r048 import fast_candidate_domino_decode_from_base
from sph.r048_layer_split import clone_dynamic_cache


SEED_LENGTH = 4
HORIZON = 16
TARGET_THROUGHPUT_RATIO = 1.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--r051-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--candidate-topk", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    return parser.parse_args()


def quantile_record_indices(
    context_lengths: Sequence[int],
    quantiles: Sequence[float] = (0.10, 0.50, 0.90),
) -> list[tuple[float, int]]:
    if not context_lengths:
        raise ValueError("cannot select quantiles from an empty collection")
    if not quantiles or any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("quantiles must lie in [0,1]")
    ordered = sorted(range(len(context_lengths)), key=lambda i: (context_lengths[i], i))
    selected: list[tuple[float, int]] = []
    used: set[int] = set()
    for quantile in quantiles:
        rank = int(round(quantile * (len(ordered) - 1)))
        index = ordered[rank]
        if index in used:
            raise ValueError("context quantiles selected a duplicate record")
        used.add(index)
        selected.append((float(quantile), index))
    return selected


def event_samples(
    callback: Callable[[], Any], *, warmup: int, repeats: int
) -> list[float]:
    if warmup < 1 or repeats < 10:
        raise ValueError("profile requires warmup>=1 and repeats>=10")
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callback()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def timing_summary(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot summarize empty timing samples")
    ordered = sorted(float(value) for value in samples)
    return {
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p50": statistics.median(ordered),
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "mean": statistics.fmean(ordered),
    }


def throughput_analysis(
    *,
    domino_eal: float,
    r051_eal: float,
    domino_ms: float,
    r051_ms: float,
    target_ratio: float = TARGET_THROUGHPUT_RATIO,
) -> dict[str, float | bool]:
    if min(domino_eal, r051_eal, domino_ms, r051_ms) <= 0 or target_ratio <= 0:
        raise ValueError("throughput inputs must be positive")
    output_ratio = (r051_eal + 1.0) / (domino_eal + 1.0)
    measured_time_ratio = r051_ms / domino_ms
    measured_throughput_ratio = output_ratio / measured_time_ratio
    maximum_time_ratio = output_ratio / target_ratio
    denominator = output_ratio - target_ratio
    if denominator <= 0:
        required_common_ms = float("inf")
    else:
        required_common_ms = max(
            0.0,
            (target_ratio * r051_ms - output_ratio * domino_ms) / denominator,
        )
    return {
        "output_advance_ratio": output_ratio,
        "measured_noncommon_time_ratio": measured_time_ratio,
        "measured_noncommon_throughput_ratio": measured_throughput_ratio,
        "maximum_time_ratio_for_target": maximum_time_ratio,
        "noncommon_gate_passed": measured_time_ratio <= maximum_time_ratio,
        "minimum_additional_shared_common_path_ms_for_target": required_common_ms,
    }


@torch.inference_mode()
def released_domino_head(
    *,
    domino: torch.nn.Module,
    target_weight: Tensor,
    hidden: Tensor,
    base_logits: Tensor,
    anchor: Tensor,
) -> Tensor:
    """Reproduce the released eager full-vocabulary correction head."""

    positions = int(hidden.shape[1])
    first = base_logits[:, 0].argmax(dim=-1)
    selected = [first]
    prefix = torch.stack([anchor, first], dim=-1)
    _, state = domino.prefix_gru(F.embedding(prefix, target_weight))
    for position in range(1, positions):
        state_for_head = state.transpose(0, 1)
        if bool(getattr(domino, "use_bias_norm", False)):
            state_for_head = domino.bias_norm(state_for_head)
        joined = torch.cat(
            [hidden[:, position : position + 1], state_for_head], dim=-1
        )
        code = domino.embed_proj[1](domino.embed_proj[0](joined))[:, 0]
        correction = F.linear(code, domino.embed_proj[2].weight)
        token = (base_logits[:, position] + correction).argmax(dim=-1)
        selected.append(token)
        if position + 1 < positions:
            _, state = domino.prefix_gru(
                F.embedding(token[:, None], target_weight), state
            )
    return torch.stack(selected, dim=1)


def _accepted_from_full_logits(proposal: Tensor, logits: Tensor) -> Tensor:
    posterior = logits[:, :HORIZON].float().argmax(dim=-1)
    return proposal.eq(posterior).to(torch.long).cumprod(dim=1).sum(dim=1)


def _accepted_from_seeded_logits(proposal: Tensor, logits: Tensor) -> Tensor:
    suffix = proposal[:, SEED_LENGTH:]
    posterior = logits[:, : HORIZON - SEED_LENGTH].float().argmax(dim=-1)
    accepted_suffix = suffix.eq(posterior).to(torch.long).cumprod(dim=1).sum(dim=1)
    return accepted_suffix + SEED_LENGTH


@torch.inference_mode()
def profile_record(
    *,
    record: dict[str, Any],
    quantile: float,
    record_index: int,
    target: torch.nn.Module,
    domino: torch.nn.Module,
    target_weight: Tensor,
    candidate_topk: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    device = target_weight.device
    context = record["context_ids_before_anchor"].long().to(device)[None]
    prefix_length = int(context.shape[1])
    prefix_output = target.model(context, use_cache=True, return_dict=True)
    prefix_cache = prefix_output.past_key_values
    del prefix_output
    anchor = torch.tensor(
        [int(record["anchor_token_id"])], dtype=torch.long, device=device
    )
    hidden = record["parallel_hidden"].to(device, torch.bfloat16)[None]
    fixed_base_logits = F.linear(hidden, target_weight)

    fixed_domino = released_domino_head(
        domino=domino,
        target_weight=target_weight,
        hidden=hidden,
        base_logits=fixed_base_logits,
        anchor=anchor,
    )
    released = record["policy_ids"].long().to(device)[None]
    if not bool(fixed_domino.eq(released).all()):
        raise RuntimeError("eager Domino head does not reproduce the fixed record")

    seed_cache = clone_dynamic_cache(prefix_cache, config=target.config)
    seed_input = anchor
    fixed_seed_tokens: list[Tensor] = []
    for _ in range(SEED_LENGTH):
        seed_output = target(
            seed_input[:, None],
            past_key_values=seed_cache,
            use_cache=True,
            return_dict=True,
        )
        seed_input = seed_output.logits[:, -1].float().argmax(dim=-1)
        fixed_seed_tokens.append(seed_input)
    fixed_seed = torch.stack(fixed_seed_tokens, dim=1)
    fixed_r051 = fast_candidate_domino_decode_from_base(
        domino=domino,
        target_weight=target_weight,
        anchors=anchor,
        hidden=hidden,
        base_logits=fixed_base_logits,
        candidate_topk=candidate_topk,
        forced_prefix=fixed_seed,
    ).token_ids
    if not bool(fixed_r051[:, :SEED_LENGTH].eq(fixed_seed).all()):
        raise RuntimeError("forced exact prefix was not retained")
    if fixed_domino.shape != (1, HORIZON) or fixed_r051.shape != (1, HORIZON):
        raise RuntimeError("profile proposals must contain exactly 16 draft tokens")
    if int(seed_cache.get_seq_length()) != prefix_length + SEED_LENGTH:
        raise RuntimeError("seed cache must end after input p2 at prefix+4 rows")

    baseline_cache = clone_dynamic_cache(prefix_cache, config=target.config)
    r051_cache = clone_dynamic_cache(prefix_cache, config=target.config)

    def baseline_cycle() -> tuple[Tensor, Tensor]:
        baseline_cache.crop(prefix_length)
        base_logits = F.linear(hidden, target_weight)
        proposal = released_domino_head(
            domino=domino,
            target_weight=target_weight,
            hidden=hidden,
            base_logits=base_logits,
            anchor=anchor,
        )
        output = target(
            torch.cat([anchor[:, None], proposal], dim=1),
            past_key_values=baseline_cache,
            use_cache=True,
            return_dict=True,
        )
        return proposal, _accepted_from_full_logits(proposal, output.logits)

    def r051_cycle() -> tuple[Tensor, Tensor]:
        r051_cache.crop(prefix_length)
        current = anchor
        seeds: list[Tensor] = []
        for _ in range(SEED_LENGTH):
            output = target(
                current[:, None],
                past_key_values=r051_cache,
                use_cache=True,
                return_dict=True,
            )
            current = output.logits[:, -1].float().argmax(dim=-1)
            seeds.append(current)
        forced = torch.stack(seeds, dim=1)
        base_logits = F.linear(hidden, target_weight)
        proposal = fast_candidate_domino_decode_from_base(
            domino=domino,
            target_weight=target_weight,
            anchors=anchor,
            hidden=hidden,
            base_logits=base_logits,
            candidate_topk=candidate_topk,
            forced_prefix=forced,
        ).token_ids
        output = target(
            proposal[:, SEED_LENGTH - 1 :],
            past_key_values=r051_cache,
            use_cache=True,
            return_dict=True,
        )
        return proposal, _accepted_from_seeded_logits(proposal, output.logits)

    baseline_proposal, baseline_accepted = baseline_cycle()
    r051_proposal, r051_accepted = r051_cycle()
    if not bool(baseline_proposal.eq(fixed_domino).all()):
        raise RuntimeError("timed Domino callback changed its proposal")
    if not bool(r051_proposal.eq(fixed_r051).all()):
        raise RuntimeError("timed R051 callback changed its proposal")

    base_gemm = timing_summary(
        event_samples(
            lambda: F.linear(hidden, target_weight), warmup=warmup, repeats=repeats
        )
    )
    domino_head = timing_summary(
        event_samples(
            lambda: released_domino_head(
                domino=domino,
                target_weight=target_weight,
                hidden=hidden,
                base_logits=fixed_base_logits,
                anchor=anchor,
            ),
            warmup=warmup,
            repeats=repeats,
        )
    )
    r051_head = timing_summary(
        event_samples(
            lambda: fast_candidate_domino_decode_from_base(
                domino=domino,
                target_weight=target_weight,
                anchors=anchor,
                hidden=hidden,
                base_logits=fixed_base_logits,
                candidate_topk=candidate_topk,
                forced_prefix=fixed_seed,
            ),
            warmup=warmup,
            repeats=repeats,
        )
    )

    verifier_cache = clone_dynamic_cache(prefix_cache, config=target.config)
    baseline_inputs = torch.cat([anchor[:, None], fixed_domino], dim=1)
    if baseline_inputs.shape[1] != HORIZON + 1:
        raise RuntimeError("Domino verifier must receive anchor plus 16 drafts")

    def baseline_verifier() -> Tensor:
        verifier_cache.crop(prefix_length)
        return target(
            baseline_inputs,
            past_key_values=verifier_cache,
            use_cache=True,
            return_dict=True,
        ).logits

    baseline_verify = timing_summary(
        event_samples(baseline_verifier, warmup=warmup, repeats=repeats)
    )

    chain_cache = clone_dynamic_cache(prefix_cache, config=target.config)

    def seed_chain() -> Tensor:
        chain_cache.crop(prefix_length)
        current = anchor
        for _ in range(SEED_LENGTH):
            output = target(
                current[:, None],
                past_key_values=chain_cache,
                use_cache=True,
                return_dict=True,
            )
            current = output.logits[:, -1].float().argmax(dim=-1)
        return current

    seed_chain_timing = timing_summary(
        event_samples(seed_chain, warmup=warmup, repeats=repeats)
    )

    final_cache = clone_dynamic_cache(seed_cache, config=target.config)
    seeded_cache_length = prefix_length + SEED_LENGTH
    final_inputs = fixed_r051[:, SEED_LENGTH - 1 :]
    if final_inputs.shape[1] != HORIZON + 1 - SEED_LENGTH:
        raise RuntimeError("R051 final verifier must receive p3..p15 (13 rows)")

    def final_verifier() -> Tensor:
        final_cache.crop(seeded_cache_length)
        return target(
            final_inputs,
            past_key_values=final_cache,
            use_cache=True,
            return_dict=True,
        ).logits

    final_verify = timing_summary(
        event_samples(final_verifier, warmup=warmup, repeats=repeats)
    )
    baseline_full = timing_summary(
        event_samples(baseline_cycle, warmup=warmup, repeats=repeats)
    )
    r051_full = timing_summary(
        event_samples(r051_cycle, warmup=warmup, repeats=repeats)
    )

    return {
        "quantile": quantile,
        "record_index": record_index,
        "sample_id": str(record["sample_id"]),
        "domain": str(record["domain"]),
        "context_length": prefix_length,
        "request_batch_size": 1,
        "correctness": {
            "domino_proposal_reproduced": True,
            "r051_seed_retained": True,
            "domino_split_accepted_drafts": int(baseline_accepted[0]),
            "r051_split_accepted_drafts": int(r051_accepted[0]),
        },
        "latency_ms": {
            "shared_base_vocab_gemm": base_gemm,
            "domino_eager_head_without_base_gemm": domino_head,
            "r051_forced_fast_k64_head_without_base_gemm": r051_head,
            "domino_target_verifier_17_rows": baseline_verify,
            "r051_target_seed_chain_4_serial_calls": seed_chain_timing,
            "r051_target_final_verifier_13_rows": final_verify,
            "domino_complete_noncommon_cycle": baseline_full,
            "r051_complete_noncommon_cycle": r051_full,
        },
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R052 requires CUDA")
    if (
        args.split != "validation_select"
        or args.candidate_topk != 64
        or args.warmup < 1
        or args.repeats < 10
    ):
        raise ValueError("official R052 requires validation_select, K64, and stable timing")

    metadata, records = load_source(args.source_rollout, args.split)
    if str(metadata.get("mode")) != "fixed":
        raise ValueError("R052 requires fixed anchors")
    r051 = json.loads(args.r051_report.read_text(encoding="utf-8"))
    if (
        str(r051.get("format")) != "r051_exact_prefix_fixed_v1"
        or int(r051["gates"]["selected_smallest_system_seed"]) != SEED_LENGTH
        or str(r051["gates"]["decision"]) != "GO_SYSTEM_PROFILE"
    ):
        raise ValueError("R051 report does not authorize the s=4 profile")

    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    target.requires_grad_(False)
    domino.requires_grad_(False)
    target_weight = target.model.embed_tokens.weight

    selected = quantile_record_indices(
        [int(record["context_length"]) for record in records]
    )
    torch.cuda.reset_peak_memory_stats(device)
    rows = [
        profile_record(
            record=records[index],
            quantile=quantile,
            record_index=index,
            target=target,
            domino=domino,
            target_weight=target_weight,
            candidate_topk=args.candidate_topk,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for quantile, index in selected
    ]

    domino_median_ms = statistics.median(
        row["latency_ms"]["domino_complete_noncommon_cycle"]["p50"]
        for row in rows
    )
    r051_median_ms = statistics.median(
        row["latency_ms"]["r051_complete_noncommon_cycle"]["p50"]
        for row in rows
    )
    domino_eal = float(r051["controls"]["released_domino_clean_replay"])
    r051_eal = float(r051["seeds"][str(SEED_LENGTH)]["eal_prompt_balanced"][
        "clean_unsplit_authority"
    ])
    throughput = throughput_analysis(
        domino_eal=domino_eal,
        r051_eal=r051_eal,
        domino_ms=domino_median_ms,
        r051_ms=r051_median_ms,
    )
    for row in rows:
        row["throughput"] = throughput_analysis(
            domino_eal=domino_eal,
            r051_eal=r051_eal,
            domino_ms=float(
                row["latency_ms"]["domino_complete_noncommon_cycle"]["p50"]
            ),
            r051_ms=float(
                row["latency_ms"]["r051_complete_noncommon_cycle"]["p50"]
            ),
        )
    report = {
        "status": "completed",
        "format": "r052_exact_prefix_eager_profile_v1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "source_rollout": str(args.source_rollout.resolve()),
        "r051_report": str(args.r051_report.resolve()),
        "profile_scope": {
            "comparison": "fair eager complete non-common cycles",
            "included": (
                "base vocab GEMM, proposal head, target seed chain where applicable, "
                "and target verifier"
            ),
            "excluded_shared_path": (
                "DFlash parallel backbone and serving scheduler; parallel_hidden is "
                "materialized identically for both methods"
            ),
            "request_batch_size": 1,
            "attention": "HF SDPA",
        },
        "records": rows,
        "aggregate": {
            "median_record_p50_domino_complete_noncommon_ms": domino_median_ms,
            "median_record_p50_r051_complete_noncommon_ms": r051_median_ms,
            "domino_clean_eal": domino_eal,
            "r051_clean_unsplit_eal": r051_eal,
            "throughput": throughput,
        },
        "memory": {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "device_total_gib": torch.cuda.get_device_properties(device).total_memory
            / 2**30,
        },
        "known_correctness_limit": {
            "hf_split_emitted_bonus_exact": "300/303",
            "lossless_deployment_claim_allowed": False,
            "latency_measurement_allowed": True,
        },
        "benchmark": {"warmup": args.warmup, "repeats": args.repeats},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
