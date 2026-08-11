#!/usr/bin/env python3
"""Exact fixed-set evaluation of target-seeded Fast-K64 (R050-A)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM

from collect_r048_capacity import load_source, prompt_balanced
from sph.fast_r048 import fast_candidate_domino_decode_from_base
from sph.gfpr import accepted_lengths
from sph.r048_layer_split import clone_dynamic_cache


TARGET_EAL = 8.325485908649174
SYSTEM_EAL_GATE = 9.0
HISTORICAL_DOMINO_EAL = 7.23955296404276


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--candidate-topk", type=int, default=64)
    return parser.parse_args()


def split_verifier_logits(anchor_logits: Tensor, suffix_logits: Tensor) -> Tensor:
    """Align split logits to 16 proposal decisions plus the bonus decision."""

    if anchor_logits.ndim != 2 or suffix_logits.ndim != 3:
        raise ValueError("split verifier logits have invalid rank")
    if (
        suffix_logits.shape[0] != anchor_logits.shape[0]
        or suffix_logits.shape[1] != 16
        or suffix_logits.shape[2] != anchor_logits.shape[1]
    ):
        raise ValueError("split verifier logits have incompatible shape")
    return torch.cat([anchor_logits[:, None], suffix_logits[:, :16]], dim=1)


def split_unsplit_numerical_parity(
    split_logits: Tensor,
    unsplit_logits: Tensor,
) -> dict[str, Tensor]:
    """Classify rows as stable/ambiguous and compare greedy tokens.

    Centering both paths on the unsplit top-1 token removes irrelevant row-wise
    constants before measuring the maximum numerical discrepancy.
    """

    if split_logits.shape != unsplit_logits.shape or split_logits.ndim != 3:
        raise ValueError("split and unsplit logits must share [batch, positions, vocab]")
    split = split_logits.float()
    unsplit = unsplit_logits.float()
    unsplit_top2 = unsplit.topk(2, dim=-1)
    reference_index = unsplit_top2.indices[..., :1]
    split_centered = split - split.gather(-1, reference_index)
    unsplit_centered = unsplit - unsplit.gather(-1, reference_index)
    row_delta = (split_centered - unsplit_centered).abs().max(dim=-1).values
    margin = unsplit_top2.values[..., 0] - unsplit_top2.values[..., 1]
    split_top1 = split.argmax(dim=-1)
    unsplit_top1 = unsplit_top2.indices[..., 0]
    matches = split_top1.eq(unsplit_top1)
    # Calibrate tolerance only on rows whose discrete decision already agrees;
    # letting a mismatching row inflate its own epsilon would make the parity
    # gate tautological.  A real mismatch above this independent band fails.
    calibration = row_delta[matches]
    epsilon = calibration.max() if calibration.numel() else row_delta.new_zeros(())
    stable = margin.gt(2.0 * epsilon)
    return {
        "split_top1": split_top1,
        "unsplit_top1": unsplit_top1,
        "epsilon": epsilon,
        "row_centered_delta": row_delta,
        "reference_margin": margin,
        "stable": stable,
        "matches": matches,
    }


def domain_prompt_balanced(
    sample_ids: Sequence[str],
    domains: Sequence[str],
    values: Sequence[int],
) -> dict[str, float]:
    grouped: dict[str, tuple[list[str], list[int]]] = {}
    for sample_id, domain, value in zip(sample_ids, domains, values, strict=True):
        ids, lengths = grouped.setdefault(str(domain), ([], []))
        ids.append(str(sample_id))
        lengths.append(int(value))
    return {
        domain: prompt_balanced(ids, lengths)
        for domain, (ids, lengths) in sorted(grouped.items())
    }


def paired_summary(reference: Sequence[int], candidate: Sequence[int]) -> dict[str, int]:
    deltas = [int(new) - int(old) for old, new in zip(reference, candidate, strict=True)]
    return {
        "gained_blocks": sum(delta > 0 for delta in deltas),
        "lost_blocks": sum(delta < 0 for delta in deltas),
        "unchanged_blocks": sum(delta == 0 for delta in deltas),
        "gained_tokens": sum(max(delta, 0) for delta in deltas),
        "lost_tokens": sum(max(-delta, 0) for delta in deltas),
        "net_tokens": sum(deltas),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R050-A requires CUDA")
    if args.split != "validation_select" or args.candidate_topk != 64:
        raise ValueError("official R050-A is frozen to validation_select and K64")
    source_metadata, records = load_source(args.source_rollout, args.split)
    if str(source_metadata.get("mode")) != "fixed":
        raise ValueError("R050-A requires fixed anchors")
    if Path(str(source_metadata["target"])).resolve() != args.target.resolve():
        raise ValueError("source target provenance differs")
    if Path(str(source_metadata["domino_draft"])).resolve() != args.domino_draft.resolve():
        raise ValueError("source Domino provenance differs")

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

    sample_ids: list[str] = []
    domains: list[str] = []
    seeded_lengths: list[int] = []
    seeded_unsplit_lengths: list[int] = []
    fast_lengths: list[int] = []
    domino_lengths: list[int] = []
    stored_domino_lengths: list[int] = []
    stable_rows = 0
    stable_matches = 0
    ambiguous_rows = 0
    all_rows = 0
    position_zero_stable_rows = 0
    position_zero_stable_matches = 0
    bonus_stable_rows = 0
    bonus_stable_matches = 0
    bonus_matches = 0
    full_accept_blocks = 0
    emitted_bonus_stable_rows = 0
    emitted_bonus_stable_matches = 0
    emitted_bonus_matches = 0
    maximum_centered_epsilon = 0.0
    ordinary_fast_position_zero_correct = 0
    seeded_suffix_changes_vs_fast = 0
    started = time.perf_counter()

    for index, row in enumerate(records, start=1):
        context = row["context_ids_before_anchor"].long().to(device)[None]
        prefix = target.model(context, use_cache=True, return_dict=True)
        clean_prefix_cache = prefix.past_key_values
        anchor = torch.tensor(
            [int(row["anchor_token_id"])], dtype=torch.long, device=device
        )
        hidden = row["parallel_hidden"].to(device, torch.bfloat16)[None]
        base_logits = F.linear(hidden, target_weight)

        ordinary_fast = fast_candidate_domino_decode_from_base(
            domino=domino,
            target_weight=target_weight,
            anchors=anchor,
            hidden=hidden,
            base_logits=base_logits,
            candidate_topk=args.candidate_topk,
        )

        split_cache = clone_dynamic_cache(clean_prefix_cache, config=target.config)
        anchor_output = target(
            anchor[:, None],
            past_key_values=split_cache,
            use_cache=True,
            return_dict=True,
        )
        anchor_logits = anchor_output.logits[:, -1].float()
        target_first = anchor_logits.argmax(dim=-1)
        seeded = fast_candidate_domino_decode_from_base(
            domino=domino,
            target_weight=target_weight,
            anchors=anchor,
            hidden=hidden,
            base_logits=base_logits,
            candidate_topk=args.candidate_topk,
            forced_first=target_first,
        )
        suffix_output = target(
            seeded.token_ids,
            past_key_values=anchor_output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        split_logits = split_verifier_logits(
            anchor_logits,
            suffix_output.logits.float(),
        )
        split_gold = split_logits[:, :16].argmax(dim=-1)
        seeded_length = int(accepted_lengths(seeded.token_ids, split_gold)[0])
        if int(seeded.token_ids[0, 0]) != int(target_first[0]):
            raise RuntimeError("forced target position zero was not retained")
        del suffix_output, anchor_output, split_cache

        seeded_unsplit_cache = clone_dynamic_cache(
            clean_prefix_cache, config=target.config
        )
        seeded_unsplit_input = torch.cat(
            [anchor[:, None], seeded.token_ids], dim=1
        )
        seeded_unsplit_output = target(
            seeded_unsplit_input,
            past_key_values=seeded_unsplit_cache,
            use_cache=True,
            return_dict=True,
        )
        seeded_unsplit_logits = seeded_unsplit_output.logits[:, :17].float()
        numerical = split_unsplit_numerical_parity(
            split_logits,
            seeded_unsplit_logits,
        )
        stable_rows += int(numerical["stable"].sum())
        stable_matches += int((numerical["stable"] & numerical["matches"]).sum())
        ambiguous_rows += int((~numerical["stable"]).sum())
        all_rows += 17
        position_zero_stable_rows += int(numerical["stable"][0, 0])
        position_zero_stable_matches += int(
            numerical["stable"][0, 0] & numerical["matches"][0, 0]
        )
        bonus_stable_rows += int(numerical["stable"][0, 16])
        bonus_stable_matches += int(
            numerical["stable"][0, 16] & numerical["matches"][0, 16]
        )
        bonus_matches += int(numerical["matches"][0, 16])
        if seeded_length == 16:
            full_accept_blocks += 1
            emitted_bonus_stable_rows += int(numerical["stable"][0, 16])
            emitted_bonus_stable_matches += int(
                numerical["stable"][0, 16] & numerical["matches"][0, 16]
            )
            emitted_bonus_matches += int(numerical["matches"][0, 16])
        maximum_centered_epsilon = max(
            maximum_centered_epsilon,
            float(numerical["epsilon"].max()),
        )
        seeded_unsplit_gold = numerical["unsplit_top1"][:, :16]
        seeded_unsplit_length = int(
            accepted_lengths(seeded.token_ids, seeded_unsplit_gold)[0]
        )
        del seeded_unsplit_output, seeded_unsplit_cache, seeded_unsplit_logits

        fast_cache = clone_dynamic_cache(clean_prefix_cache, config=target.config)
        fast_output = target(
            torch.cat([anchor[:, None], ordinary_fast.token_ids], dim=1),
            past_key_values=fast_cache,
            use_cache=True,
            return_dict=True,
        )
        fast_gold = fast_output.logits[:, :16].float().argmax(dim=-1)
        fast_length = int(accepted_lengths(ordinary_fast.token_ids, fast_gold)[0])
        ordinary_fast_position_zero_correct += int(
            ordinary_fast.token_ids[0, 0].eq(fast_gold[0, 0])
        )
        seeded_suffix_changes_vs_fast += int(
            seeded.token_ids[0, 1:].ne(ordinary_fast.token_ids[0, 1:]).any()
        )
        del fast_output, fast_cache

        released = row["policy_ids"].long().to(device)[None]
        domino_cache = clone_dynamic_cache(clean_prefix_cache, config=target.config)
        domino_output = target(
            torch.cat([anchor[:, None], released], dim=1),
            past_key_values=domino_cache,
            use_cache=True,
            return_dict=True,
        )
        domino_gold = domino_output.logits[:, :16].float().argmax(dim=-1)
        domino_length = int(accepted_lengths(released, domino_gold)[0])

        sample_ids.append(str(row["sample_id"]))
        domains.append(str(row["domain"]))
        seeded_lengths.append(seeded_length)
        seeded_unsplit_lengths.append(seeded_unsplit_length)
        fast_lengths.append(fast_length)
        domino_lengths.append(domino_length)
        stored_domino_lengths.append(int(row["accepted_length"]))
        del domino_output, domino_cache, prefix, clean_prefix_cache, base_logits
        if index % 32 == 0 or index == len(records):
            print(f"evaluated {index}/{len(records)}", flush=True)

    seeded_eal = prompt_balanced(sample_ids, seeded_lengths)
    seeded_unsplit_eal = prompt_balanced(sample_ids, seeded_unsplit_lengths)
    fast_eal = prompt_balanced(sample_ids, fast_lengths)
    domino_eal = prompt_balanced(sample_ids, domino_lengths)
    stored_domino_eal = prompt_balanced(sample_ids, stored_domino_lengths)
    seeded_domains = domain_prompt_balanced(sample_ids, domains, seeded_lengths)
    domino_domains = domain_prompt_balanced(sample_ids, domains, domino_lengths)
    domain_regressions = {
        domain: seeded_domains[domain] - domino_domains[domain]
        for domain in sorted(seeded_domains)
    }
    stable_parity_passed = (
        stable_matches == stable_rows
        and stable_rows > 0
        and emitted_bonus_matches == full_accept_blocks
    )
    domain_gate_passed = all(delta >= 0 for delta in domain_regressions.values())
    if not stable_parity_passed:
        decision = "STOP_SPLIT_NUMERICAL_PARITY"
    elif seeded_eal < TARGET_EAL:
        decision = "CLOSE_R050_ACCURACY_FAIL"
    elif seeded_eal < SYSTEM_EAL_GATE or not domain_gate_passed:
        decision = "ACCURACY_PASS_NO_SYSTEM_INTEGRATION"
    else:
        decision = "GO_EAGER_SPLIT_PROFILE"
    maximum_iteration_ratio_for_1p15x = (
        (seeded_eal + 1.0)
        / (1.15 * (HISTORICAL_DOMINO_EAL + 1.0))
    )
    report: dict[str, Any] = {
        "status": "completed",
        "format": "r050_target_seeded_fixed_v1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "source_rollout": str(args.source_rollout.resolve()),
        "target": str(args.target.resolve()),
        "domino_draft": str(args.domino_draft.resolve()),
        "split": args.split,
        "prompts": len(set(sample_ids)),
        "blocks": len(records),
        "candidate_topk": args.candidate_topk,
        "target_token_geometry": "prefix + [anchor] + [p0..p15] = split 1+16",
        "trainable_parameters": 0,
        "eal_prompt_balanced": {
            "released_domino_clean_replay": domino_eal,
            "released_domino_stored": stored_domino_eal,
            "ordinary_fast_k64_clean": fast_eal,
            "target_seeded_fast_k64_split": seeded_eal,
            "target_seeded_fast_k64_unsplit_control": seeded_unsplit_eal,
            "gain_vs_clean_domino": seeded_eal - domino_eal,
            "gain_vs_ordinary_fast_k64": seeded_eal - fast_eal,
            "ratio_vs_historical_domino": seeded_eal / HISTORICAL_DOMINO_EAL,
        },
        "domain_eal": {
            "released_domino_clean_replay": domino_domains,
            "target_seeded_fast_k64_split": seeded_domains,
            "seeded_minus_domino": domain_regressions,
        },
        "paired_vs_clean_domino": paired_summary(domino_lengths, seeded_lengths),
        "paired_vs_ordinary_fast_k64": paired_summary(fast_lengths, seeded_lengths),
        "split_vs_unsplit": paired_summary(
            seeded_unsplit_lengths, seeded_lengths
        ),
        "clean_vs_stored_domino_length_mismatches": sum(
            clean != stored
            for clean, stored in zip(
                domino_lengths, stored_domino_lengths, strict=True
            )
        ),
        "position_zero": {
            "target_seed_forced_blocks": len(records),
            "forced_token_retained_blocks": len(records),
            "ordinary_fast_correct_blocks": ordinary_fast_position_zero_correct,
            "ordinary_fast_accuracy": ordinary_fast_position_zero_correct / len(records),
            "seeded_suffix_changed_vs_ordinary_fast_blocks": seeded_suffix_changes_vs_fast,
        },
        "numerical_control": {
            "rows": all_rows,
            "stable_rows": stable_rows,
            "ambiguous_rows": ambiguous_rows,
            "stable_matching_argmax_rows": stable_matches,
            "stable_argmax_match_rate": stable_matches / max(stable_rows, 1),
            "position_zero_stable_rows": position_zero_stable_rows,
            "position_zero_stable_matches": position_zero_stable_matches,
            "bonus_stable_rows": bonus_stable_rows,
            "bonus_stable_matches": bonus_stable_matches,
            "bonus_matching_argmax_rows": bonus_matches,
            "full_accept_blocks_emitting_bonus": full_accept_blocks,
            "emitted_bonus_stable_rows": emitted_bonus_stable_rows,
            "emitted_bonus_stable_matches": emitted_bonus_stable_matches,
            "emitted_bonus_matching_argmax_rows": emitted_bonus_matches,
            "emitted_bonus_exact_match_passed": (
                emitted_bonus_matches == full_accept_blocks
            ),
            "maximum_row_centered_logit_epsilon": maximum_centered_epsilon,
            "passed": stable_parity_passed,
        },
        "gates": {
            "target_eal": TARGET_EAL,
            "system_profile_eal": SYSTEM_EAL_GATE,
            "accuracy_passed": seeded_eal >= TARGET_EAL,
            "system_eal_passed": seeded_eal >= SYSTEM_EAL_GATE,
            "domain_no_regression_passed": domain_gate_passed,
            "stable_split_parity_passed": stable_parity_passed,
            "decision": decision,
        },
        "throughput_headroom": {
            "historical_domino_eal": HISTORICAL_DOMINO_EAL,
            "target_throughput_ratio": 1.15,
            "maximum_iteration_time_ratio_vs_domino": maximum_iteration_ratio_for_1p15x,
        },
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not stable_parity_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
