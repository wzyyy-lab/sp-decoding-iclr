#!/usr/bin/env python3
"""R053 clean-B16 accuracy/latency Pareto for one-pass target trees."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM

from collect_r048_capacity import load_source, prompt_balanced
from evaluate_r050_target_seeded import domain_prompt_balanced, paired_summary
from profile_r052_exact_prefix import (
    event_samples,
    released_domino_head,
    timing_summary,
)
from sph.fast_r048 import fast_candidate_domino_decode_from_base
from sph.r048_layer_split import clone_dynamic_cache
from sph.r053_tree import (
    BudgetedTrie,
    build_budgeted_trie,
    fast_candidate_domino_beam_from_base,
    full_pool_oracle_acceptance,
    hindsight_budget_acceptance,
    pack_tree_tensors,
    pack_tree_traversal,
    simulated_tree_acceptance,
    traverse_tree_logits,
    traverse_tree_logits_path,
    traverse_tree_logits_tensor,
)


HORIZON = 16
OFFICIAL_BUDGETS = (17, 24, 32, 48, 64)
TARGET_EAL = 8.325485908649174
TARGET_TPS_RATIO = 1.20
GAMMA = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--r052-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--candidate-pool-topk", type=int, default=64)
    parser.add_argument("--tree-support-size", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(OFFICIAL_BUDGETS))
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Mechanics smoke only; any finite value disables claim-bearing routing.",
    )
    return parser.parse_args()


def _summary(
    sample_ids: Sequence[str], domains: Sequence[str], values: Sequence[int]
) -> dict[str, Any]:
    return {
        "overall": prompt_balanced(sample_ids, values),
        "by_domain": domain_prompt_balanced(sample_ids, domains, values),
    }


def _quantiles(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty values")
    ordered = sorted(int(value) for value in values)
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[int(math.ceil(0.95 * len(ordered))) - 1],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _median_context_index(records: Sequence[dict[str, Any]]) -> int:
    ordered = sorted(
        range(len(records)),
        key=lambda index: (int(records[index]["context_length"]), index),
    )
    return ordered[int(round(0.5 * (len(ordered) - 1)))]


@torch.inference_mode()
def clean_target_continuation_with_logits(
    *,
    target: torch.nn.Module,
    prefix_cache: Any,
    anchor: Tensor,
    stored_gold: Tensor,
) -> tuple[Tensor, Tensor, bool, int, Tensor]:
    """Return unconditional qlen=1 greedy p0..p15, bonus and 17 logits.

    A 17-row teacher-path pass is retained only as a diagnostic.  It never
    chooses the authority because BF16/SDPA kernels can differ by query shape.
    """

    if anchor.shape != (1,) or stored_gold.shape != (1, HORIZON):
        raise ValueError("clean continuation requires batch-1 B16 tensors")
    teacher_cache = clone_dynamic_cache(prefix_cache, config=target.config)
    teacher_input = torch.cat([anchor[:, None], stored_gold], dim=1)
    teacher = target(
        teacher_input,
        past_key_values=teacher_cache,
        use_cache=True,
        return_dict=True,
    )
    teacher_top1 = teacher.logits[:, :HORIZON].float().argmax(dim=-1)
    mismatches = int(teacher_top1.ne(stored_gold).sum())
    teacher_exact = mismatches == 0
    # Always restart from the clean prefix and generate with serving-shape
    # qlen=1.  Even an apparently exact teacher path is not the authority.
    del teacher, teacher_cache, teacher_top1
    cache = clone_dynamic_cache(prefix_cache, config=target.config)
    current = anchor
    tokens: list[Tensor] = []
    decision_logits: list[Tensor] = []
    for step in range(HORIZON + 1):
        output = target(
            current[:, None],
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        row_logits = output.logits[:, -1].float()
        decision_logits.append(row_logits)
        current = row_logits.argmax(dim=-1)
        if step < HORIZON:
            tokens.append(current)
    return (
        torch.stack(tokens, dim=1),
        current,
        teacher_exact,
        mismatches,
        torch.stack(decision_logits, dim=1),
    )


@torch.inference_mode()
def clean_target_continuation(
    *,
    target: torch.nn.Module,
    prefix_cache: Any,
    anchor: Tensor,
    stored_gold: Tensor,
) -> tuple[Tensor, Tensor, bool, int]:
    """Backward-compatible clean authority without retaining decision logits."""

    clean, bonus, teacher_exact, mismatches, _ = clean_target_continuation_with_logits(
        target=target,
        prefix_cache=prefix_cache,
        anchor=anchor,
        stored_gold=stored_gold,
    )
    return clean, bonus, teacher_exact, mismatches


def _accepted(proposal: Tensor, target_tokens: Tensor) -> int:
    return int(
        proposal.eq(target_tokens)
        .to(torch.long)
        .cumprod(dim=-1)
        .sum(dim=-1)[0]
    )


def _path_scores(edge_log_probs: Tensor, gamma: float) -> Tensor:
    if edge_log_probs.ndim != 2:
        raise ValueError("edge log probabilities require [beam,horizon]")
    weights = edge_log_probs.new_tensor(gamma).pow(
        torch.arange(edge_log_probs.shape[1], device=edge_log_probs.device)
    )
    return (edge_log_probs * weights[None]).sum(dim=-1)


def _throughput(
    *, domino_eal: float, tree_eal: float, domino_ms: float, tree_ms: float
) -> dict[str, float | bool]:
    output_ratio = (tree_eal + 1.0) / (domino_eal + 1.0)
    time_ratio = tree_ms / domino_ms
    tps_ratio = output_ratio / time_ratio
    maximum_ms = domino_ms * output_ratio / TARGET_TPS_RATIO
    return {
        "output_advance_ratio": output_ratio,
        "time_ratio": time_ratio,
        "projected_tps_ratio": tps_ratio,
        "maximum_tree_cycle_ms_for_1p20": maximum_ms,
        "latency_gate_passed": tps_ratio >= TARGET_TPS_RATIO,
    }


@torch.inference_mode()
def profile_median_record(
    *,
    record: dict[str, Any],
    target: torch.nn.Module,
    domino: torch.nn.Module,
    target_weight: Tensor,
    budgets: Sequence[int],
    candidate_pool_topk: int,
    tree_support_size: int,
    beam_width: int,
    gamma: float,
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
    fixed_base = F.linear(hidden, target_weight)
    fixed_beam = fast_candidate_domino_beam_from_base(
        domino=domino,
        target_weight=target_weight,
        anchors=anchor,
        hidden=hidden,
        base_logits=fixed_base,
        candidate_pool_topk=candidate_pool_topk,
        tree_support_size=tree_support_size,
        beam_width=beam_width,
    )
    fixed_scores = _path_scores(fixed_beam.edge_log_probs[0], gamma)
    trees = {
        budget: build_budgeted_trie(
            fixed_beam.token_ids[0],
            fixed_scores,
            fixed_beam.trunk_token_ids[0],
            budget=budget,
        )
        for budget in budgets
    }

    stored_gold = record["gold_ids"].long().to(device)[None]
    clean_gold, clean_bonus, teacher_exact, teacher_mismatches = clean_target_continuation(
        target=target,
        prefix_cache=prefix_cache,
        anchor=anchor,
        stored_gold=stored_gold,
    )

    fixed_domino = released_domino_head(
        domino=domino,
        target_weight=target_weight,
        hidden=hidden,
        base_logits=fixed_base,
        anchor=anchor,
    )
    released = record["policy_ids"].long().to(device)[None]
    if not bool(fixed_domino.eq(released).all()):
        raise RuntimeError("profile Domino head does not reproduce fixed record")
    baseline_inputs = torch.cat([anchor[:, None], fixed_domino], dim=1)
    baseline_cache = clone_dynamic_cache(prefix_cache, config=target.config)

    def baseline_cycle() -> Tensor:
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
        ).logits
        posterior = output[:, :HORIZON].float().argmax(dim=-1)
        return proposal.eq(posterior).to(torch.long).cumprod(dim=1).sum(dim=1)

    baseline_logits = baseline_cycle()
    if baseline_logits.shape != (1,):
        raise RuntimeError("Domino complete callback did not return acceptance")
    baseline_timing = timing_summary(
        event_samples(baseline_cycle, warmup=warmup, repeats=repeats)
    )
    base_timing = timing_summary(
        event_samples(
            lambda: F.linear(hidden, target_weight),
            warmup=warmup,
            repeats=repeats,
        )
    )
    beam_timing = timing_summary(
        event_samples(
            lambda: fast_candidate_domino_beam_from_base(
                domino=domino,
                target_weight=target_weight,
                anchors=anchor,
                hidden=hidden,
                base_logits=fixed_base,
                candidate_pool_topk=candidate_pool_topk,
                tree_support_size=tree_support_size,
                beam_width=beam_width,
            ),
            warmup=warmup,
            repeats=repeats,
        )
    )

    budget_reports: dict[str, Any] = {}
    for budget in budgets:
        tree = trees[budget]
        inputs, positions, mask = pack_tree_tensors(
            tree,
            anchor_token_id=int(anchor[0]),
            prefix_length=prefix_length,
            device=device,
            mask_dtype=target_weight.dtype,
        )
        packed_traversal = pack_tree_traversal(tree, device=device)
        tree_cache = clone_dynamic_cache(prefix_cache, config=target.config)

        def verifier() -> Tensor:
            tree_cache.crop(prefix_length)
            return target(
                inputs,
                position_ids=positions,
                attention_mask=mask,
                past_key_values=tree_cache,
                use_cache=True,
                return_dict=True,
            ).logits

        actual_logits = verifier()
        actual_accept, actual_bonus = traverse_tree_logits(tree, actual_logits)
        simulated_accept = simulated_tree_acceptance(tree, clean_gold[0])
        verifier_timing = timing_summary(
            event_samples(verifier, warmup=warmup, repeats=repeats)
        )
        traversal_timing = timing_summary(
            event_samples(
                lambda: traverse_tree_logits_tensor(
                    tree, actual_logits, packed=packed_traversal
                ),
                warmup=warmup,
                repeats=repeats,
            )
        )

        complete_cache = clone_dynamic_cache(prefix_cache, config=target.config)

        def optimistic_complete_cycle() -> Tensor:
            complete_cache.crop(prefix_length)
            base_logits = F.linear(hidden, target_weight)
            beam = fast_candidate_domino_beam_from_base(
                domino=domino,
                target_weight=target_weight,
                anchors=anchor,
                hidden=hidden,
                base_logits=base_logits,
                candidate_pool_topk=candidate_pool_topk,
                tree_support_size=tree_support_size,
                beam_width=beam_width,
            )
            # Packing is held fixed to measure the optimistic lower bound of a
            # fused GPU allocator.  The deterministic beam is still executed.
            if beam.token_ids.shape != fixed_beam.token_ids.shape:
                raise RuntimeError("timed beam changed shape")
            output = target(
                inputs,
                position_ids=positions,
                attention_mask=mask,
                past_key_values=complete_cache,
                use_cache=True,
                return_dict=True,
            ).logits
            return traverse_tree_logits_tensor(
                tree, output, packed=packed_traversal
            )[0]

        complete_timing = timing_summary(
            event_samples(
                optimistic_complete_cycle, warmup=warmup, repeats=repeats
            )
        )
        budget_reports[str(budget)] = {
            "rows_including_anchor": tree.used_nodes_including_anchor,
            "simulated_clean_acceptance": simulated_accept,
            "hf_tree_forward_acceptance": actual_accept,
            "hf_tree_matches_simulation": actual_accept == simulated_accept,
            "hf_bonus_match_if_full_accept": (
                None
                if simulated_accept < HORIZON or actual_accept < HORIZON
                else actual_bonus == int(clean_bonus[0])
            ),
            "temporary_kv_rows_written": tree.used_nodes_including_anchor,
            "selected_branch_rows_to_commit": 1 + actual_accept,
            "commit_operation": (
                "optimistic zero-cost paged-KV pointer commit; HF DynamicCache "
                "cannot commit a non-contiguous tree branch"
            ),
            "latency_ms": {
                "one_target_tree_forward": verifier_timing,
                "full_vocab_argmax_and_tree_traversal": traversal_timing,
                "optimistic_complete_cycle": complete_timing,
            },
        }

    return {
        "record": {
            "sample_id": str(record["sample_id"]),
            "domain": str(record["domain"]),
            "context_length": prefix_length,
            "teacher_path_exact_diagnostic": teacher_exact,
            "clean_authority": "unconditional 17-step qlen=1 autoregressive",
            "teacher_forced_mismatches": teacher_mismatches,
        },
        "scope": {
            "request_batch_size": 1,
            "attention": "HF SDPA eager with one 4-D static tree mask",
            "included": (
                "base vocabulary GEMM, frozen Fast beam, one target tree forward, "
                "full-vocabulary logits, temporary KV writes and traversal"
            ),
            "optimistic_exclusion": (
                "data-dependent trie packing and paged-KV pointer commit; fixed "
                "packed inputs model a fused zero-cost allocator"
            ),
            "excluded_shared_path": (
                "DFlash parallel backbone and serving scheduler; parallel_hidden "
                "is materialized identically for Domino and R053"
            ),
        },
        "latency_ms": {
            "base_vocab_gemm": base_timing,
            "fast_beam_without_base_gemm": beam_timing,
            "domino_complete_noncommon_cycle": baseline_timing,
        },
        "budgets": budget_reports,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R053 requires CUDA")
    if (
        args.split != "validation_select"
        or tuple(args.budgets) != OFFICIAL_BUDGETS
        or args.candidate_pool_topk != 64
        or args.tree_support_size != 16
        or args.beam_width != 16
        or args.gamma != GAMMA
    ):
        raise ValueError("official R053 configuration is frozen")
    if args.warmup < 1 or args.repeats < 10:
        raise ValueError("profile requires warmup>=1 and repeats>=10")

    metadata, records = load_source(args.source_rollout, args.split)
    if str(metadata.get("mode")) != "fixed":
        raise ValueError("R053 requires fixed clean-B16 anchors")
    if Path(str(metadata["target"])).resolve() != args.target.resolve():
        raise ValueError("source target provenance differs")
    if Path(str(metadata["domino_draft"])).resolve() != args.domino_draft.resolve():
        raise ValueError("source Domino provenance differs")
    r052 = json.loads(args.r052_report.read_text(encoding="utf-8"))
    if str(r052.get("format")) != "r052_exact_prefix_eager_profile_v1":
        raise ValueError("R052 report has the wrong format")
    if Path(str(r052.get("source_rollout"))).resolve() != args.source_rollout.resolve():
        raise ValueError("R052 and R053 use different fixed evaluation sources")
    claim_bearing = args.max_records is None
    if args.max_records is not None:
        if not 1 <= args.max_records <= len(records):
            raise ValueError("max-records lies outside the fixed collection")
        records = records[: args.max_records]

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
    torch.cuda.reset_peak_memory_stats(device)

    sample_ids: list[str] = []
    domains: list[str] = []
    clean_domino_lengths: list[int] = []
    trunk_lengths: list[int] = []
    full_pool_lengths: list[int] = []
    deploy_lengths = {budget: [] for budget in OFFICIAL_BUDGETS}
    hindsight_lengths = {budget: [] for budget in OFFICIAL_BUDGETS}
    actual_tree_lengths = {budget: [] for budget in OFFICIAL_BUDGETS}
    actual_clean_prefix_lengths = {budget: [] for budget in OFFICIAL_BUDGETS}
    actual_output_parity = {budget: [] for budget in OFFICIAL_BUDGETS}
    actual_selected_mismatch_positions = {
        budget: [] for budget in OFFICIAL_BUDGETS
    }
    actual_full_accept_bonus_parity = {
        budget: [] for budget in OFFICIAL_BUDGETS
    }
    full_node_counts: list[int] = []
    teacher_exact_blocks = 0
    teacher_mismatch_rows = 0
    trunk_identity_mismatches = 0
    support_mismatches = 0
    tree_below_trunk = 0
    started = time.perf_counter()

    for index, row in enumerate(records, start=1):
        context = row["context_ids_before_anchor"].long().to(device)[None]
        prefix_output = target.model(context, use_cache=True, return_dict=True)
        prefix_cache = prefix_output.past_key_values
        del prefix_output
        anchor = torch.tensor(
            [int(row["anchor_token_id"])], dtype=torch.long, device=device
        )
        hidden = row["parallel_hidden"].to(device, torch.bfloat16)[None]
        base_logits = F.linear(hidden, target_weight)
        beam = fast_candidate_domino_beam_from_base(
            domino=domino,
            target_weight=target_weight,
            anchors=anchor,
            hidden=hidden,
            base_logits=base_logits,
            candidate_pool_topk=args.candidate_pool_topk,
            tree_support_size=args.tree_support_size,
            beam_width=args.beam_width,
        )
        fast_control = fast_candidate_domino_decode_from_base(
            domino=domino,
            target_weight=target_weight,
            anchors=anchor,
            hidden=hidden,
            base_logits=base_logits,
            candidate_topk=args.candidate_pool_topk,
        ).token_ids
        trunk_identity_mismatches += int(
            beam.trunk_token_ids.ne(fast_control).sum()
        )
        base_top15 = base_logits.float().topk(16, dim=-1).indices[:, :, :15]
        support_mismatches += int(
            beam.candidate_ids[:, :, :15].ne(base_top15).sum()
        )
        if not bool(
            beam.candidate_ids.eq(beam.trunk_token_ids[:, :, None]).any(dim=-1).all()
        ):
            raise RuntimeError("K16 tree support omitted a Fast trunk token")

        stored_gold = row["gold_ids"].long().to(device)[None]
        clean_gold, clean_bonus, teacher_exact, mismatches = clean_target_continuation(
            target=target,
            prefix_cache=prefix_cache,
            anchor=anchor,
            stored_gold=stored_gold,
        )
        teacher_exact_blocks += int(teacher_exact)
        teacher_mismatch_rows += mismatches

        released = row["policy_ids"].long().to(device)[None]
        clean_domino_length = _accepted(released, clean_gold)
        trunk_length = _accepted(beam.trunk_token_ids, clean_gold)
        full_pool_length = full_pool_oracle_acceptance(
            beam.token_ids[0], clean_gold[0]
        )
        discounted_scores = _path_scores(beam.edge_log_probs[0], args.gamma)
        full_nodes: int | None = None
        for budget in OFFICIAL_BUDGETS:
            tree = build_budgeted_trie(
                beam.token_ids[0],
                discounted_scores,
                beam.trunk_token_ids[0],
                budget=budget,
            )
            if full_nodes is None:
                full_nodes = tree.full_nodes_including_anchor
            elif full_nodes != tree.full_nodes_including_anchor:
                raise RuntimeError("tree pool changed across node budgets")
            deploy = simulated_tree_acceptance(tree, clean_gold[0])
            hindsight = hindsight_budget_acceptance(
                beam.token_ids[0],
                beam.trunk_token_ids[0],
                clean_gold[0],
                budget=budget,
            )
            tree_inputs, tree_positions, tree_mask = pack_tree_tensors(
                tree,
                anchor_token_id=int(anchor[0]),
                prefix_length=int(context.shape[1]),
                device=device,
                mask_dtype=target_weight.dtype,
            )
            tree_cache = clone_dynamic_cache(prefix_cache, config=target.config)
            tree_output = target(
                tree_inputs,
                position_ids=tree_positions,
                attention_mask=tree_mask,
                past_key_values=tree_cache,
                use_cache=True,
                return_dict=True,
            )
            selected_path, actual_next, _ = traverse_tree_logits_path(
                tree, tree_output.logits
            )
            clean_values = clean_gold[0].detach().cpu().long().tolist()
            common_prefix = 0
            for actual_token, clean_token in zip(selected_path, clean_values):
                if int(actual_token) != int(clean_token):
                    break
                common_prefix += 1
            selected_mismatch = (
                None if common_prefix == len(selected_path) else common_prefix
            )
            expected_next = (
                int(clean_bonus[0])
                if len(selected_path) == HORIZON
                else int(clean_gold[0, len(selected_path)])
            )
            emitted_parity = (
                common_prefix == len(selected_path)
                and int(actual_next) == expected_next
            )
            actual_tree_lengths[budget].append(len(selected_path))
            actual_clean_prefix_lengths[budget].append(common_prefix)
            actual_output_parity[budget].append(emitted_parity)
            actual_selected_mismatch_positions[budget].append(selected_mismatch)
            if len(selected_path) == HORIZON:
                actual_full_accept_bonus_parity[budget].append(
                    int(actual_next) == int(clean_bonus[0])
                )
            del tree_cache, tree_output, tree_inputs, tree_positions, tree_mask
            if deploy < trunk_length:
                tree_below_trunk += 1
            if budget == 17 and deploy != trunk_length:
                raise RuntimeError("N17 tree does not reproduce the Fast trunk")
            deploy_lengths[budget].append(deploy)
            hindsight_lengths[budget].append(hindsight)

        if full_nodes is None:
            raise RuntimeError("no tree budgets evaluated")
        full_node_counts.append(full_nodes)
        sample_ids.append(str(row["sample_id"]))
        domains.append(str(row["domain"]))
        clean_domino_lengths.append(clean_domino_length)
        trunk_lengths.append(trunk_length)
        full_pool_lengths.append(full_pool_length)
        del prefix_cache, context, hidden, base_logits, beam, fast_control
        if index % 32 == 0 or index == len(records):
            print(f"evaluated {index}/{len(records)}", flush=True)

    if trunk_identity_mismatches or support_mismatches or tree_below_trunk:
        raise RuntimeError(
            "R053 proposal/tree semantic controls failed: "
            f"trunk_identity_mismatches={trunk_identity_mismatches}, "
            f"support_mismatches={support_mismatches}, "
            f"simulated_tree_below_trunk={tree_below_trunk}"
        )

    clean_domino = _summary(sample_ids, domains, clean_domino_lengths)
    expected_clean_domino = float(r052["aggregate"]["domino_clean_eal"])
    trunk = _summary(sample_ids, domains, trunk_lengths)
    full_pool = _summary(sample_ids, domains, full_pool_lengths)
    budget_accuracy: dict[str, Any] = {}
    for budget in OFFICIAL_BUDGETS:
        simulated = _summary(sample_ids, domains, deploy_lengths[budget])
        actual_raw = _summary(sample_ids, domains, actual_tree_lengths[budget])
        deploy = _summary(
            sample_ids, domains, actual_clean_prefix_lengths[budget]
        )
        hindsight = _summary(sample_ids, domains, hindsight_lengths[budget])
        mismatch_histogram: dict[str, int] = defaultdict(int)
        for position in actual_selected_mismatch_positions[budget]:
            key = "none" if position is None else str(position)
            mismatch_histogram[key] += 1
        budget_accuracy[str(budget)] = {
            "deployable_actual_tree_clean_prefix": deploy,
            "actual_hf_tree_self_acceptance_diagnostic": actual_raw,
            "draft_only_structure_simulation": simulated,
            "hindsight_structural_upper_bound": hindsight,
            "paired_deployable_vs_clean_domino": paired_summary(
                clean_domino_lengths, actual_clean_prefix_lengths[budget]
            ),
            "paired_deployable_vs_fast_trunk": paired_summary(
                trunk_lengths, actual_clean_prefix_lengths[budget]
            ),
            "actual_selected_token_mismatch_position_histogram": dict(
                sorted(mismatch_histogram.items())
            ),
            "actual_emitted_output_parity_blocks": sum(
                actual_output_parity[budget]
            ),
            "actual_emitted_output_mismatch_blocks": len(records)
            - sum(actual_output_parity[budget]),
            "actual_full_accept_blocks": len(
                actual_full_accept_bonus_parity[budget]
            ),
            "actual_full_accept_bonus_parity_blocks": sum(
                actual_full_accept_bonus_parity[budget]
            ),
            "lossless_deployment_claim_allowed": False,
        }

    median_index = _median_context_index(records)
    profile = profile_median_record(
        record=records[median_index],
        target=target,
        domino=domino,
        target_weight=target_weight,
        budgets=OFFICIAL_BUDGETS[1:],
        candidate_pool_topk=args.candidate_pool_topk,
        tree_support_size=args.tree_support_size,
        beam_width=args.beam_width,
        gamma=args.gamma,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    domino_ms = float(
        profile["latency_ms"]["domino_complete_noncommon_cycle"]["p50"]
    )
    joint_gates: dict[str, Any] = {}
    passing: list[int] = []
    clean_domains = clean_domino["by_domain"]
    for budget in OFFICIAL_BUDGETS[1:]:
        deploy = budget_accuracy[str(budget)][
            "deployable_actual_tree_clean_prefix"
        ]
        eal = float(deploy["overall"])
        tree_ms = float(
            profile["budgets"][str(budget)]["latency_ms"]
            ["optimistic_complete_cycle"]["p50"]
        )
        tps = _throughput(
            domino_eal=float(clean_domino["overall"]),
            tree_eal=eal,
            domino_ms=domino_ms,
            tree_ms=tree_ms,
        )
        domain_deltas = {
            domain: float(deploy["by_domain"][domain])
            - float(clean_domains[domain])
            for domain in sorted(clean_domains)
        }
        accuracy_pass = eal >= TARGET_EAL
        domain_pass = all(value >= 0.0 for value in domain_deltas.values())
        joint = accuracy_pass and bool(tps["latency_gate_passed"]) and domain_pass
        if joint:
            passing.append(budget)
        joint_gates[str(budget)] = {
            "deployable_eal": eal,
            "accuracy_gate_passed": accuracy_pass,
            "domain_delta_vs_clean_domino": domain_deltas,
            "domain_no_regression_gate_passed": domain_pass,
            "throughput": tps,
            "joint_gate_passed": joint,
        }

    selected = min(passing) if passing else None
    if not claim_bearing:
        route = "SMOKE_COMPLETE_NOT_CLAIM_BEARING"
        selected = None
    else:
        route = (
            "GO_IMPLEMENT_PROFILE_ONE_PASS_TREE_BOUNDED_HF"
            if selected is not None
            else "CLOSE_TARGET_MULTIPATH_NO_JOINT_PARETO"
        )
    report = {
        "status": "completed",
        "format": "r053_tree_budget_pareto_v1",
        "claim_bearing": claim_bearing,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "contract": {
            "horizon": HORIZON,
            "candidate_pool_topk_for_fast_trunk": args.candidate_pool_topk,
            "tree_support": "DFlash Top-15 plus protected current Fast trunk; fixed K16",
            "beam_width": args.beam_width,
            "beam_pruning": "ordinary cumulative candidate-logprob",
            "tree_allocator": (
                "trunk-first prefix closure, then max descendant gamma=0.75 "
                "draft-only path score"
            ),
            "budgets_include_anchor": list(OFFICIAL_BUDGETS),
            "target_calls_per_tree_cycle": 1,
        },
        "source": {
            "source_rollout": str(args.source_rollout.resolve()),
            "target": str(args.target.resolve()),
            "domino_draft": str(args.domino_draft.resolve()),
            "r052_report": str(args.r052_report.resolve()),
            "split": args.split,
            "blocks": len(records),
            "prompts": len(set(sample_ids)),
        },
        "clean_authority": {
            "authority": "unconditional 17-step batch1 qlen=1 autoregressive continuation",
            "teacher_path_exact_diagnostic_blocks": teacher_exact_blocks,
            "teacher_path_mismatch_diagnostic_blocks": len(records)
            - teacher_exact_blocks,
            "teacher_forced_mismatch_rows_diagnostic": teacher_mismatch_rows,
        },
        "controls": {
            "fast_k64_trunk_token_mismatches": trunk_identity_mismatches,
            "top15_support_token_mismatches": support_mismatches,
            "deployable_tree_below_trunk_blocks": tree_below_trunk,
            "n17_exactly_reproduces_trunk": deploy_lengths[17] == trunk_lengths,
            "expected_r052_clean_domino_eal": expected_clean_domino,
            "clean_domino_eal_absolute_error": abs(
                float(clean_domino["overall"]) - expected_clean_domino
            ),
            "r052_comparison_note": (
                "R052 uses clean unsplit verifier geometry; R053 clean-output "
                "authority is unconditional qlen=1 AR, so the delta is reported "
                "rather than forced to zero"
            ),
        },
        "accuracy": {
            "clean_domino": clean_domino,
            "fast_k64_trunk": trunk,
            "full_w16_pool_oracle": full_pool,
            "full_w16_unique_nodes_including_anchor": _quantiles(full_node_counts),
            "budgets": budget_accuracy,
        },
        "profile": profile,
        "memory": {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
            / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device)
            / (1024**3),
            "device_total_gib": torch.cuda.get_device_properties(device).total_memory
            / (1024**3),
        },
        "decision": {
            "target_eal": TARGET_EAL,
            "development_tps_ratio": TARGET_TPS_RATIO,
            "joint_gates": joint_gates,
            "selected_smallest_budget": selected,
            "route": route,
            "lossless_deployment_claim_allowed": False,
            "hf_emitted_parity_role": (
                "diagnostic only in R053; conditional SGLang must enforce stable "
                "non-tie selected-branch and full-accept bonus parity"
            ),
        },
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
