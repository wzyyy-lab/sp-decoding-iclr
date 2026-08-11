#!/usr/bin/env python3
"""R055 fixed W4/W8/W16 padded-forest accuracy and A40 graph Pareto."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM

from collect_r048_capacity import load_source, prompt_balanced
from evaluate_r050_target_seeded import (
    domain_prompt_balanced,
    paired_summary,
    split_unsplit_numerical_parity,
)
from evaluate_r053_tree_budget import clean_target_continuation_with_logits
from profile_r052_exact_prefix import (
    event_samples,
    released_domino_head,
    timing_summary,
)
from profile_r053_beam_graph import capture_graph, median_context_record
from sph.fast_r048 import fast_candidate_domino_decode_from_base
from sph.r048_layer_split import clone_dynamic_cache
from sph.r053_tree import fast_candidate_domino_beam_from_base
from sph.r055_forest import (
    pack_padded_forest,
    structural_forest_acceptance,
    traverse_padded_forest,
)


HORIZON = 16
OFFICIAL_WIDTHS = (4, 8, 16)
TARGET_EAL = 8.325485908649174
DEVELOPMENT_TPS_RATIO = 1.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--r053-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--beam-widths", type=int, nargs="+", default=list(OFFICIAL_WIDTHS))
    parser.add_argument(
        "--development-width-sweep",
        action="store_true",
        help=(
            "Allow non-official widths for a non-claim-bearing Pareto diagnostic. "
            "The frozen official R055 widths remain W={4,8,16}."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Mechanics smoke only; finite values disable claim-bearing routing.",
    )
    return parser.parse_args()


def summary(
    sample_ids: Sequence[str], domains: Sequence[str], values: Sequence[int]
) -> dict[str, Any]:
    return {
        "overall": prompt_balanced(sample_ids, values),
        "by_domain": domain_prompt_balanced(sample_ids, domains, values),
    }


def select_balanced_smoke_records(
    records: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select an equal-domain, distinct-prompt, median-context smoke subset."""

    domains = sorted({str(record["domain"]) for record in records})
    if not domains or count < len(domains) or count % len(domains) != 0:
        raise ValueError("smoke record count must be a positive multiple of domains")
    per_domain = count // len(domains)
    canonical_context = int(median_context_record(records)["context_length"])
    selected: list[dict[str, Any]] = []
    for domain in domains:
        prompts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if str(record["domain"]) == domain:
                prompts[str(record["sample_id"])].append(record)
        representatives = [
            min(
                prompt_records,
                key=lambda record: (
                    abs(int(record["context_length"]) - canonical_context),
                    int(record["context_length"]),
                ),
            )
            for prompt_records in prompts.values()
        ]
        representatives.sort(
            key=lambda record: (
                abs(int(record["context_length"]) - canonical_context),
                str(record["sample_id"]),
                int(record["context_length"]),
            )
        )
        if len(representatives) < per_domain:
            raise ValueError(f"domain {domain} lacks distinct smoke prompts")
        selected.extend(representatives[:per_domain])
    return selected


def accepted(proposal: Tensor, target_tokens: Tensor) -> int:
    return int(
        proposal.eq(target_tokens)
        .to(torch.long)
        .cumprod(dim=-1)
        .sum(dim=-1)[0]
    )


def clean_domino_control(
    *,
    released_control: Tensor,
    stored_released: Tensor,
    fast_control: Tensor,
    clean_gold: Tensor,
) -> tuple[int, int]:
    """Validate the real Domino baseline without conflating it with Fast-K64."""

    if not torch.equal(released_control, stored_released):
        raise RuntimeError("same-job Domino head changed released Domino tokens")
    return (
        accepted(released_control, clean_gold),
        int(fast_control.ne(released_control).sum()),
    )


def enforce_forest_controls(
    *, same_job_domino_mismatches: int, trunk_mismatches: dict[int, int]
) -> None:
    if same_job_domino_mismatches != 0:
        raise RuntimeError(
            "same-job Domino head changed released Domino tokens: "
            f"{same_job_domino_mismatches}"
        )
    if any(value != 0 for value in trunk_mismatches.values()):
        raise RuntimeError(f"forest beam changed Fast-K64 trunk: {trunk_mismatches}")


def common_prefix_length(actual: Tensor, clean: Tensor) -> int:
    if actual.ndim != 1 or clean.ndim != 1:
        raise ValueError("common-prefix inputs must be vectors")
    matches = actual.eq(clean[: actual.numel()]).to(torch.long)
    return int(matches.cumprod(dim=-1).sum()) if matches.numel() else 0


def throughput(
    *,
    domino_output_advance: float,
    forest_output_advance: float,
    domino_ms: float,
    forest_ms: float,
) -> dict[str, float | bool]:
    output_ratio = forest_output_advance / domino_output_advance
    time_ratio = forest_ms / domino_ms
    ratio = output_ratio / time_ratio
    return {
        "output_advance_ratio": output_ratio,
        "time_ratio": time_ratio,
        "projected_tps_ratio": ratio,
        "maximum_cycle_ms_for_1p20": domino_ms * output_ratio / DEVELOPMENT_TPS_RATIO,
        "maximum_cycle_ms_for_1p15": domino_ms * output_ratio / 1.15,
        "development_gate_passed": ratio >= DEVELOPMENT_TPS_RATIO,
    }


@torch.inference_mode()
def profile_record(
    *,
    record: dict[str, Any],
    target: torch.nn.Module,
    domino: torch.nn.Module,
    target_weight: Tensor,
    widths: Sequence[int],
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
    base_logits = F.linear(hidden, target_weight)

    released = record["policy_ids"].long().to(device)[None]
    fixed_domino = released_domino_head(
        domino=domino,
        target_weight=target_weight,
        hidden=hidden,
        base_logits=base_logits,
        anchor=anchor,
    )
    if not torch.equal(fixed_domino, released):
        raise RuntimeError("profile Domino head differs from the fixed source")
    domino_cache = clone_dynamic_cache(prefix_cache, config=target.config)

    def domino_cycle() -> Tensor:
        domino_cache.crop(prefix_length)
        current_base = F.linear(hidden, target_weight)
        proposal = released_domino_head(
            domino=domino,
            target_weight=target_weight,
            hidden=hidden,
            base_logits=current_base,
            anchor=anchor,
        )
        logits = target(
            torch.cat([anchor[:, None], proposal], dim=1),
            past_key_values=domino_cache,
            use_cache=True,
            return_dict=True,
        ).logits
        # Materialize all 17 verifier decisions so Domino includes the same
        # full-accept bonus argmax work that the forest traversal includes.
        posterior = logits.float().argmax(dim=-1)
        accepted_length = (
            proposal.eq(posterior[:, :HORIZON])
            .to(torch.long)
            .cumprod(dim=1)
            .sum(dim=1)
        )
        next_token = posterior.gather(1, accepted_length[:, None])[:, 0]
        return torch.stack([accepted_length, next_token], dim=1)

    domino_cycle()
    domino_timing = timing_summary(
        event_samples(domino_cycle, warmup=warmup, repeats=repeats)
    )
    base_timing = timing_summary(
        event_samples(
            lambda: F.linear(hidden, target_weight),
            warmup=warmup,
            repeats=repeats,
        )
    )

    width_reports: dict[str, Any] = {}
    for width in widths:
        def beam_callback() -> Any:
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

        eager_beam = beam_callback()
        beam_graph, graph_beam = capture_graph(beam_callback)
        beam_graph.replay()
        torch.cuda.synchronize()
        for field in (
            "token_ids",
            "edge_log_probs",
            "map_scores",
            "trunk_token_ids",
            "candidate_ids",
        ):
            if not torch.equal(getattr(eager_beam, field), getattr(graph_beam, field)):
                raise RuntimeError(f"W{width} graph changed beam {field}")
        graph_beam_timing = timing_summary(
            event_samples(beam_graph.replay, warmup=warmup, repeats=repeats)
        )

        forest_inputs, forest_positions, forest_mask = pack_padded_forest(
            eager_beam.token_ids[0],
            anchor_token_id=int(anchor[0]),
            prefix_length=prefix_length,
            mask_dtype=target_weight.dtype,
        )
        forest_cache = clone_dynamic_cache(prefix_cache, config=target.config)

        def verifier() -> Tensor:
            forest_cache.crop(prefix_length)
            return target(
                forest_inputs,
                position_ids=forest_positions,
                attention_mask=forest_mask,
                past_key_values=forest_cache,
                use_cache=True,
                return_dict=True,
            ).logits

        fixed_logits = verifier()
        fixed_traversal = traverse_padded_forest(eager_beam.token_ids[0], fixed_logits)
        verifier_timing = timing_summary(
            event_samples(verifier, warmup=warmup, repeats=repeats)
        )
        traversal_timing = timing_summary(
            event_samples(
                lambda: traverse_padded_forest(eager_beam.token_ids[0], fixed_logits),
                warmup=warmup,
                repeats=repeats,
            )
        )
        fill_timing = timing_summary(
            event_samples(
                lambda: forest_inputs[:, 1:].copy_(graph_beam.token_ids[0].reshape(1, -1)),
                warmup=warmup,
                repeats=repeats,
            )
        )
        runtime_forest_mask = torch.empty_like(forest_mask)
        runtime_forest_mask.copy_(forest_mask)
        mask_copy_timing = timing_summary(
            event_samples(
                lambda: runtime_forest_mask.copy_(forest_mask),
                warmup=warmup,
                repeats=repeats,
            )
        )

        # The base GEMM overwrites the exact static tensor read by the captured
        # beam.  Stream ordering then makes the graph output available to the
        # fixed forest fill before the target call begins.
        complete_cache = clone_dynamic_cache(prefix_cache, config=target.config)

        def graph_beam_complete_cycle() -> Tensor:
            complete_cache.crop(prefix_length)
            base_logits.copy_(F.linear(hidden, target_weight))
            beam_graph.replay()
            forest_inputs[:, 1:].copy_(graph_beam.token_ids[0].reshape(1, -1))
            runtime_forest_mask.copy_(forest_mask)
            logits = target(
                forest_inputs,
                position_ids=forest_positions,
                attention_mask=runtime_forest_mask,
                past_key_values=complete_cache,
                use_cache=True,
                return_dict=True,
            ).logits
            return traverse_padded_forest(graph_beam.token_ids[0], logits).accepted

        complete = graph_beam_complete_cycle()
        if complete.shape != (1,):
            raise RuntimeError("forest complete cycle returned wrong acceptance shape")
        complete_timing = timing_summary(
            event_samples(
                graph_beam_complete_cycle, warmup=warmup, repeats=repeats
            )
        )
        width_reports[str(width)] = {
            "rows_including_shared_anchor": 1 + width * HORIZON,
            "graph_eager_all_beam_tensor_parity": True,
            "fixed_record_acceptance": int(fixed_traversal.accepted[0]),
            "latency_ms": {
                "cuda_graph_beam_without_base_gemm": graph_beam_timing,
                "static_forest_fill": fill_timing,
                "static_forest_mask_copy": mask_copy_timing,
                "one_target_forest_forward": verifier_timing,
                "full_vocab_argmax_and_all_path_traversal": traversal_timing,
                "graph_beam_complete_noncommon_cycle": complete_timing,
            },
        }
    return {
        "record": {
            "sample_id": str(record["sample_id"]),
            "domain": str(record["domain"]),
            "context_length": prefix_length,
            "request_batch_size": 1,
        },
        "scope": {
            "included": (
                "base vocabulary GEMM, CUDA-graph frozen beam, fixed forest fill, "
                "one target call, full-vocabulary logits and all-path traversal"
            ),
            "excluded_shared": "DFlash backbone and serving scheduler",
            "optimistic_exclusion": "paged-KV selected-path pointer commit",
        },
        "latency_ms": {
            "base_vocab_gemm": base_timing,
            "domino_complete_noncommon_cycle": domino_timing,
        },
        "widths": width_reports,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R055 requires CUDA")
    if torch.cuda.get_device_name(0) != "NVIDIA A40":
        raise RuntimeError("official R055 requires NVIDIA A40")
    widths = tuple(args.beam_widths)
    if not widths or len(set(widths)) != len(widths) or any(width < 1 for width in widths):
        raise ValueError("beam widths must be distinct positive integers")
    if widths != OFFICIAL_WIDTHS and not args.development_width_sweep:
        raise ValueError("official R055 widths are frozen to W={4,8,16}")
    if args.split != "validation_select":
        raise ValueError("official R055 requires validation_select")
    if args.warmup < 1 or args.repeats < 10:
        raise ValueError("profile requires warmup>=1 and repeats>=10")
    started = time.perf_counter()

    metadata, records = load_source(args.source_rollout, args.split)
    if str(metadata.get("mode")) != "fixed":
        raise ValueError("R055 requires the fixed clean-B16 rollout")
    profile_records = records
    if args.max_records is not None:
        if not 1 <= args.max_records <= len(records):
            raise ValueError("max-records lies outside the source")
        records = select_balanced_smoke_records(records, args.max_records)
    claim_bearing = args.max_records is None and not args.development_width_sweep
    report053 = json.loads(args.r053_report.read_text(encoding="utf-8"))
    if str(report053.get("format")) != "r053_tree_budget_pareto_v1":
        raise ValueError("R053 report has the wrong format")

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
    clean_domino_lengths: list[int] = []
    clean_domino_output_advances: list[int] = []
    teacher_exact_blocks = 0
    teacher_mismatch_rows = 0
    recomputed_domino_released_token_mismatches = 0
    fast_released_token_mismatches = 0
    trunk_mismatches = {width: 0 for width in widths}
    structural_lengths = {width: [] for width in widths}
    actual_lengths = {width: [] for width in widths}
    clean_prefix_lengths = {width: [] for width in widths}
    output_parity = {width: [] for width in widths}
    clean_output_advances = {width: [] for width in widths}
    stable_rows = {width: 0 for width in widths}
    stable_matches = {width: 0 for width in widths}
    ambiguous_rows = {width: 0 for width in widths}
    maximum_centered_epsilon = {width: 0.0 for width in widths}
    unique_clean_winner_blocks = {width: 0 for width in widths}
    unique_clean_winner_selected = {width: 0 for width in widths}
    structural_actual_length_parity = {width: [] for width in widths}
    full_accept_bonus_parity = {width: [] for width in widths}
    selected_mismatch_positions: dict[int, list[int | None]] = {
        width: [] for width in widths
    }

    for index, row in enumerate(records, start=1):
        context = row["context_ids_before_anchor"].long().to(device)[None]
        prefix = target.model(context, use_cache=True, return_dict=True)
        prefix_cache = prefix.past_key_values
        del prefix
        anchor = torch.tensor(
            [int(row["anchor_token_id"])], dtype=torch.long, device=device
        )
        hidden = row["parallel_hidden"].to(device, torch.bfloat16)[None]
        base_logits = F.linear(hidden, target_weight)
        fast_control = fast_candidate_domino_decode_from_base(
            domino=domino,
            target_weight=target_weight,
            anchors=anchor,
            hidden=hidden,
            base_logits=base_logits,
            candidate_topk=64,
        ).token_ids
        same_job_domino = released_domino_head(
            domino=domino,
            target_weight=target_weight,
            hidden=hidden,
            base_logits=base_logits,
            anchor=anchor,
        )
        stored_gold = row["gold_ids"].long().to(device)[None]
        (
            clean_gold,
            clean_bonus,
            teacher_exact,
            mismatches,
            clean_decision_logits,
        ) = clean_target_continuation_with_logits(
            target=target,
            prefix_cache=prefix_cache,
            anchor=anchor,
            stored_gold=stored_gold,
        )
        teacher_exact_blocks += int(teacher_exact)
        teacher_mismatch_rows += mismatches
        released = row["policy_ids"].long().to(device)[None]
        clean_domino_length, fast_mismatches = clean_domino_control(
            released_control=same_job_domino,
            stored_released=released,
            fast_control=fast_control,
            clean_gold=clean_gold,
        )
        fast_released_token_mismatches += fast_mismatches
        clean_domino_lengths.append(clean_domino_length)
        clean_domino_output_advances.append(clean_domino_length + 1)

        for width in widths:
            beam = fast_candidate_domino_beam_from_base(
                domino=domino,
                target_weight=target_weight,
                anchors=anchor,
                hidden=hidden,
                base_logits=base_logits,
                candidate_pool_topk=64,
                tree_support_size=16,
                beam_width=width,
            )
            trunk_mismatches[width] += int(
                beam.trunk_token_ids.ne(fast_control).sum()
            )
            structural = int(
                structural_forest_acceptance(beam.token_ids[0], clean_gold[0])
            )
            canonical_path_lengths = (
                beam.token_ids[0]
                .eq(clean_gold[0][None])
                .to(torch.long)
                .cumprod(dim=-1)
                .sum(dim=-1)
            )
            canonical_winner = int(canonical_path_lengths.argmax())
            unique_clean_winner = int(
                canonical_path_lengths.eq(canonical_path_lengths.max()).sum()
            ) == 1
            forest_inputs, forest_positions, forest_mask = pack_padded_forest(
                beam.token_ids[0],
                anchor_token_id=int(anchor[0]),
                prefix_length=int(context.shape[1]),
                mask_dtype=target_weight.dtype,
            )
            forest_cache = clone_dynamic_cache(prefix_cache, config=target.config)
            forest_output = target(
                forest_inputs,
                position_ids=forest_positions,
                attention_mask=forest_mask,
                past_key_values=forest_cache,
                use_cache=True,
                return_dict=True,
            )
            traversal = traverse_padded_forest(
                beam.token_ids[0], forest_output.logits
            )
            actual = int(traversal.accepted[0])
            path_index = int(traversal.selected_path[0])
            if unique_clean_winner:
                unique_clean_winner_blocks[width] += 1
                unique_clean_winner_selected[width] += int(
                    path_index == canonical_winner
                )
            selected = beam.token_ids[0, path_index, :actual]
            common = common_prefix_length(selected, clean_gold[0])
            actual_next = int(traversal.next_token[0])
            expected_next = (
                int(clean_bonus[0])
                if actual == HORIZON
                else int(clean_gold[0, actual])
            )
            parity = common == actual and actual_next == expected_next
            clean_output_advances[width].append(common + int(parity))
            selected_start = 1 + path_index * HORIZON
            # Keep only canonically reachable decisions: all matching draft
            # decisions plus the first mismatching/emitted-next decision.
            decision_rows = [0] + list(range(selected_start, selected_start + common))
            selected_decision_logits = forest_output.logits[:, decision_rows].float()
            numerical = split_unsplit_numerical_parity(
                selected_decision_logits,
                clean_decision_logits[:, : common + 1],
            )
            stable_rows[width] += int(numerical["stable"].sum())
            stable_matches[width] += int(
                (numerical["stable"] & numerical["matches"]).sum()
            )
            ambiguous_rows[width] += int((~numerical["stable"]).sum())
            maximum_centered_epsilon[width] = max(
                maximum_centered_epsilon[width], float(numerical["epsilon"])
            )
            structural_lengths[width].append(structural)
            actual_lengths[width].append(actual)
            clean_prefix_lengths[width].append(common)
            output_parity[width].append(parity)
            structural_actual_length_parity[width].append(actual == structural)
            selected_mismatch_positions[width].append(
                None if common == actual else common
            )
            if actual == HORIZON:
                full_accept_bonus_parity[width].append(
                    actual_next == int(clean_bonus[0])
                )
            del (
                forest_cache,
                forest_output,
                forest_inputs,
                forest_positions,
                forest_mask,
                beam,
            )
        sample_ids.append(str(row["sample_id"]))
        domains.append(str(row["domain"]))
        del (
            prefix_cache,
            context,
            hidden,
            base_logits,
            fast_control,
            same_job_domino,
            clean_decision_logits,
        )
        if index % 32 == 0 or index == len(records):
            print(f"evaluated {index}/{len(records)}", flush=True)

    enforce_forest_controls(
        same_job_domino_mismatches=recomputed_domino_released_token_mismatches,
        trunk_mismatches=trunk_mismatches,
    )
    clean_domino = summary(sample_ids, domains, clean_domino_lengths)
    clean_domino_advance = summary(
        sample_ids, domains, clean_domino_output_advances
    )
    widths_accuracy: dict[str, Any] = {}
    for width in widths:
        mismatch_histogram: dict[str, int] = defaultdict(int)
        for position in selected_mismatch_positions[width]:
            mismatch_histogram["none" if position is None else str(position)] += 1
        widths_accuracy[str(width)] = {
            "rows_including_shared_anchor": 1 + width * HORIZON,
            "structural_full_pool": summary(
                sample_ids, domains, structural_lengths[width]
            ),
            "actual_hf_self_acceptance_diagnostic": summary(
                sample_ids, domains, actual_lengths[width]
            ),
            "actual_clean_prefix": summary(
                sample_ids, domains, clean_prefix_lengths[width]
            ),
            "canonical_clean_output_advance": summary(
                sample_ids, domains, clean_output_advances[width]
            ),
            "paired_actual_clean_vs_domino": paired_summary(
                clean_domino_lengths, clean_prefix_lengths[width]
            ),
            "actual_equals_structural_blocks": sum(
                structural_actual_length_parity[width]
            ),
            "emitted_output_parity_blocks": sum(output_parity[width]),
            "emitted_output_mismatch_blocks": len(records) - sum(output_parity[width]),
            "stable_non_tie_selected_path": {
                "unique_clean_winner_blocks": unique_clean_winner_blocks[width],
                "unique_clean_winner_selected_blocks": (
                    unique_clean_winner_selected[width]
                ),
                "unique_clean_winner_selection_parity_passed": (
                    unique_clean_winner_selected[width]
                    == unique_clean_winner_blocks[width]
                ),
                "stable_rows": stable_rows[width],
                "stable_matches": stable_matches[width],
                "ambiguous_rows": ambiguous_rows[width],
                "stable_parity_passed": stable_rows[width] == stable_matches[width],
                "maximum_centered_epsilon": maximum_centered_epsilon[width],
            },
            "selected_token_mismatch_position_histogram": dict(
                sorted(mismatch_histogram.items())
            ),
            "actual_full_accept_blocks": len(full_accept_bonus_parity[width]),
            "full_accept_bonus_parity_blocks": sum(
                full_accept_bonus_parity[width]
            ),
            "lossless_deployment_claim_allowed": False,
        }

    profile = profile_record(
        record=median_context_record(profile_records),
        target=target,
        domino=domino,
        target_weight=target_weight,
        widths=widths,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    domino_ms = float(
        profile["latency_ms"]["domino_complete_noncommon_cycle"]["p50"]
    )
    clean_domains = clean_domino["by_domain"]
    gates: dict[str, Any] = {}
    passing: list[int] = []
    for width in widths:
        deploy = widths_accuracy[str(width)]["actual_clean_prefix"]
        eal = float(deploy["overall"])
        forest_advance = float(
            widths_accuracy[str(width)]["canonical_clean_output_advance"]["overall"]
        )
        forest_ms = float(
            profile["widths"][str(width)]["latency_ms"]
            ["graph_beam_complete_noncommon_cycle"]["p50"]
        )
        tps = throughput(
            domino_output_advance=float(clean_domino_advance["overall"]),
            forest_output_advance=forest_advance,
            domino_ms=domino_ms,
            forest_ms=forest_ms,
        )
        domain_delta = {
            domain: float(deploy["by_domain"][domain]) - float(clean_domains[domain])
            for domain in sorted(clean_domains)
        }
        accuracy_pass = eal >= TARGET_EAL
        domain_pass = all(value >= 0.0 for value in domain_delta.values())
        joint = accuracy_pass and domain_pass and bool(tps["development_gate_passed"])
        if joint:
            passing.append(width)
        gates[str(width)] = {
            "actual_clean_eal": eal,
            "canonical_clean_output_advance": forest_advance,
            "nominal_eal_plus_one_output_ratio": (
                (eal + 1.0) / (float(clean_domino["overall"]) + 1.0)
            ),
            "accuracy_gate_passed": accuracy_pass,
            "domain_delta_vs_clean_domino": domain_delta,
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
            "GO_SGLANG_PADDED_FOREST"
            if selected is not None
            else "CLOSE_FIXED_PADDED_FOREST_NO_JOINT_PARETO"
        )

    result = {
        "status": "completed",
        "format": "r055_padded_beam_forest_v1",
        "claim_bearing": claim_bearing,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "contract": {
            "beam_widths": list(widths),
            "official_widths": list(OFFICIAL_WIDTHS),
            "development_width_sweep": args.development_width_sweep,
            "rows": {str(width): 1 + width * HORIZON for width in widths},
            "candidate_pool_topk": 64,
            "branch_support": "DFlash Top-15 plus protected Fast-K64 trunk",
            "one_target_call_per_forest": True,
            "new_trainable_parameters": 0,
            "throughput_output_advance": (
                "canonical clean draft prefix plus one only when the emitted "
                "next token also matches clean qlen=1 authority; clean Domino "
                "uses accepted+1, a conservative convention against the forest"
            ),
        },
        "source": {
            "source_rollout": str(args.source_rollout.resolve()),
            "target": str(args.target.resolve()),
            "domino_draft": str(args.domino_draft.resolve()),
            "r053_report": str(args.r053_report.resolve()),
            "split": args.split,
            "blocks": len(records),
            "prompts": len(set(sample_ids)),
        },
        "clean_authority": {
            "authority": "unconditional 17-step batch1 qlen=1 autoregressive continuation",
            "teacher_path_exact_diagnostic_blocks": teacher_exact_blocks,
            "teacher_path_mismatch_diagnostic_blocks": len(records) - teacher_exact_blocks,
            "teacher_forced_mismatch_rows_diagnostic": teacher_mismatch_rows,
        },
        "controls": {
            "same_job_domino_vs_released_token_mismatches": (
                recomputed_domino_released_token_mismatches
            ),
            "fast_k64_vs_released_domino_token_mismatches": (
                fast_released_token_mismatches
            ),
            "fast_k64_trunk_token_mismatches_by_width": trunk_mismatches,
            "fixed_shape_no_token_deduplication": True,
            "duplicate_first_tokens_evaluated_as_independent_paths": True,
        },
        "accuracy": {
            "clean_domino": clean_domino,
            "clean_domino_output_advance": clean_domino_advance,
            "widths": widths_accuracy,
        },
        "profile": profile,
        "memory": {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            "device_total_gib": torch.cuda.get_device_properties(device).total_memory / (1024**3),
        },
        "decision": {
            "target_eal": TARGET_EAL,
            "development_tps_ratio": DEVELOPMENT_TPS_RATIO,
            "gates": gates,
            "selected_smallest_width": selected,
            "route": route,
            "lossless_deployment_claim_allowed": False,
        },
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
