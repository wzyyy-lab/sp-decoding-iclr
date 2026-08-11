#!/usr/bin/env python3
"""Full fixed-set accuracy falsifier for exact target prefix seeds (R051)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

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
from sph.fast_r048 import fast_candidate_domino_decode_from_base
from sph.gfpr import accepted_lengths
from sph.r048_layer_split import clone_dynamic_cache


TARGET_EAL = 8.325485908649174
SYSTEM_EAL_GATE = 9.0
HISTORICAL_DOMINO_EAL = 7.23955296404276
SEED_LENGTHS = (2, 3, 4)
HORIZON = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--r050-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--candidate-topk", type=int, default=64)
    return parser.parse_args()


def assemble_split_verifier_logits(
    seed_logits: Tensor,
    final_logits: Tensor,
) -> Tensor:
    """Return p0..p15 verification rows plus the bonus row.

    For a seed length ``s``, ``seed_logits`` are produced by sequential inputs
    ``anchor,p0,...,p{s-2}``.  The final pass starts at input ``p{s-1}`` and
    therefore contributes decisions ``p_s,...,p15,bonus``.
    """

    if seed_logits.ndim != 3 or final_logits.ndim != 3:
        raise ValueError("seed/final logits must have shape [batch, rows, vocab]")
    if (
        seed_logits.shape[0] != final_logits.shape[0]
        or seed_logits.shape[2] != final_logits.shape[2]
    ):
        raise ValueError("seed/final logits have incompatible batch or vocabulary")
    seed_length = int(seed_logits.shape[1])
    if seed_length not in SEED_LENGTHS:
        raise ValueError("official split assembly requires seed length 2, 3, or 4")
    if final_logits.shape[1] != HORIZON + 1 - seed_length:
        raise ValueError("final verifier must begin at p{s-1} and end at p15")
    aligned = torch.cat([seed_logits, final_logits], dim=1)
    if aligned.shape[1] != HORIZON + 1:
        raise RuntimeError("split verifier did not produce all 17 decision rows")
    return aligned


def choose_route(
    unsplit_eal: Mapping[int, float],
) -> tuple[str, int | None]:
    """Apply the preregistered accuracy gates without using split self-EAL."""

    if tuple(sorted(unsplit_eal)) != SEED_LENGTHS:
        raise ValueError("R051 decision requires exactly seed lengths 2, 3, and 4")
    best = max(float(value) for value in unsplit_eal.values())
    if best < TARGET_EAL:
        return "CLOSE_EXACT_SEED_FAMILY_ACCURACY_FAIL", None
    eligible = [
        seed
        for seed in SEED_LENGTHS
        if float(unsplit_eal[seed]) >= SYSTEM_EAL_GATE
    ]
    if eligible:
        return "GO_SYSTEM_PROFILE", min(eligible)
    return "ACCURACY_PASS_SYSTEM_NO_GO_WITHOUT_TIMING", None


def _empty_seed_state() -> dict[str, Any]:
    return {
        "split_lengths": [],
        "unsplit_lengths": [],
        "stable_rows": 0,
        "stable_matches": 0,
        "ambiguous_rows": 0,
        "all_rows": 0,
        "seed_prefix_rows": 0,
        "seed_prefix_matches": 0,
        "seed_prefix_stable_rows": 0,
        "seed_prefix_stable_matches": 0,
        "bonus_matches": 0,
        "full_accept_blocks": 0,
        "emitted_bonus_matches": 0,
        "maximum_centered_epsilon": 0.0,
    }


def _validate_r050_control(
    report: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    blocks: int,
) -> None:
    if str(report.get("format")) != "r050_target_seeded_fixed_v1":
        raise ValueError("R050 control report has the wrong format")
    required_paths = {
        "source_rollout": args.source_rollout,
        "target": args.target,
        "domino_draft": args.domino_draft,
    }
    for key, expected in required_paths.items():
        if Path(str(report[key])).resolve() != expected.resolve():
            raise ValueError(f"R050 control {key} differs from R051")
    if (
        str(report.get("split")) != args.split
        or int(report.get("candidate_topk", -1)) != args.candidate_topk
        or int(report.get("blocks", -1)) != blocks
    ):
        raise ValueError("R050 control does not use the same fixed evaluation set")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R051 requires CUDA")
    if args.split != "validation_select" or args.candidate_topk != 64:
        raise ValueError("official R051 is frozen to validation_select and K64")

    source_metadata, records = load_source(args.source_rollout, args.split)
    if str(source_metadata.get("mode")) != "fixed":
        raise ValueError("R051 requires fixed anchors")
    if Path(str(source_metadata["target"])).resolve() != args.target.resolve():
        raise ValueError("source target provenance differs")
    if Path(str(source_metadata["domino_draft"])).resolve() != args.domino_draft.resolve():
        raise ValueError("source Domino provenance differs")
    r050_report = json.loads(args.r050_report.read_text(encoding="utf-8"))
    _validate_r050_control(r050_report, args=args, blocks=len(records))

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
    stored_domino_lengths: list[int] = []
    state = {seed: _empty_seed_state() for seed in SEED_LENGTHS}
    started = time.perf_counter()

    for index, row in enumerate(records, start=1):
        context = row["context_ids_before_anchor"].long().to(device)[None]
        prefix_output = target.model(context, use_cache=True, return_dict=True)
        clean_prefix_cache = prefix_output.past_key_values
        del prefix_output
        anchor = torch.tensor(
            [int(row["anchor_token_id"])], dtype=torch.long, device=device
        )
        hidden = row["parallel_hidden"].to(device, torch.bfloat16)[None]
        base_logits = F.linear(hidden, target_weight)

        released = row["policy_ids"].long().to(device)[None]
        domino_cache = clone_dynamic_cache(clean_prefix_cache, config=target.config)
        domino_output = target(
            torch.cat([anchor[:, None], released], dim=1),
            past_key_values=domino_cache,
            use_cache=True,
            return_dict=True,
        )
        domino_gold = domino_output.logits[:, :HORIZON].float().argmax(dim=-1)
        clean_domino_length = int(accepted_lengths(released, domino_gold)[0])
        del domino_output, domino_cache, domino_gold

        # Generate p0..p3 from the actual sequential split target path.  The
        # live cache is never passed into a final verifier; each final pass gets
        # a clone so advancing it cannot corrupt the next seed step.
        seed_cache = clone_dynamic_cache(clean_prefix_cache, config=target.config)
        seed_input = anchor
        seed_tokens: list[Tensor] = []
        seed_logit_rows: list[Tensor] = []
        for step in range(max(SEED_LENGTHS)):
            seed_output = target(
                seed_input[:, None],
                past_key_values=seed_cache,
                use_cache=True,
                return_dict=True,
            )
            seed_cache = seed_output.past_key_values
            seed_logit = seed_output.logits[:, -1].float()
            seed_token = seed_logit.argmax(dim=-1)
            seed_logit_rows.append(seed_logit)
            seed_tokens.append(seed_token)
            seed_length = step + 1

            if seed_length in SEED_LENGTHS:
                forced_prefix = torch.stack(seed_tokens, dim=1)
                proposal = fast_candidate_domino_decode_from_base(
                    domino=domino,
                    target_weight=target_weight,
                    anchors=anchor,
                    hidden=hidden,
                    base_logits=base_logits,
                    candidate_topk=args.candidate_topk,
                    forced_prefix=forced_prefix,
                )
                if not bool(proposal.token_ids[:, :seed_length].eq(forced_prefix).all()):
                    raise RuntimeError("Fast-K64 failed to retain an exact target seed")
                forced_in_support = proposal.candidate_ids[
                    :, :seed_length
                ].eq(forced_prefix[:, :, None]).any(dim=-1)
                if not bool(forced_in_support.all()):
                    raise RuntimeError("exact target seed fell outside fixed K64 support")

                # Cache currently ends at p{s-2}; input begins at p{s-1} and
                # extends through p15, yielding p_s..p15 plus the bonus row.
                final_cache = clone_dynamic_cache(seed_cache, config=target.config)
                final_input = proposal.token_ids[:, seed_length - 1 :]
                expected_final_rows = HORIZON + 1 - seed_length
                if final_input.shape[1] != expected_final_rows:
                    raise RuntimeError("final verifier input has an off-by-one error")
                final_output = target(
                    final_input,
                    past_key_values=final_cache,
                    use_cache=True,
                    return_dict=True,
                )
                seed_logits = torch.stack(seed_logit_rows, dim=1)
                split_logits = assemble_split_verifier_logits(
                    seed_logits,
                    final_output.logits.float(),
                )
                split_gold = split_logits[:, :HORIZON].argmax(dim=-1)
                split_length = int(
                    accepted_lengths(proposal.token_ids, split_gold)[0]
                )
                del final_output, final_cache, final_input, split_gold

                # Sole accuracy authority: a clean batch-1 unsplit target pass
                # over the exact proposal produced above.
                unsplit_cache = clone_dynamic_cache(
                    clean_prefix_cache, config=target.config
                )
                unsplit_input = torch.cat(
                    [anchor[:, None], proposal.token_ids], dim=1
                )
                unsplit_output = target(
                    unsplit_input,
                    past_key_values=unsplit_cache,
                    use_cache=True,
                    return_dict=True,
                )
                unsplit_logits = unsplit_output.logits[:, : HORIZON + 1].float()
                numerical = split_unsplit_numerical_parity(
                    split_logits, unsplit_logits
                )
                unsplit_gold = numerical["unsplit_top1"][:, :HORIZON]
                unsplit_length = int(
                    accepted_lengths(proposal.token_ids, unsplit_gold)[0]
                )

                seed_state = state[seed_length]
                seed_state["split_lengths"].append(split_length)
                seed_state["unsplit_lengths"].append(unsplit_length)
                seed_state["stable_rows"] += int(numerical["stable"].sum())
                seed_state["stable_matches"] += int(
                    (numerical["stable"] & numerical["matches"]).sum()
                )
                seed_state["ambiguous_rows"] += int((~numerical["stable"]).sum())
                seed_state["all_rows"] += HORIZON + 1
                seed_state["seed_prefix_rows"] += seed_length
                seed_state["seed_prefix_matches"] += int(
                    proposal.token_ids[:, :seed_length]
                    .eq(unsplit_gold[:, :seed_length])
                    .sum()
                )
                seed_state["seed_prefix_stable_rows"] += int(
                    numerical["stable"][:, :seed_length].sum()
                )
                seed_state["seed_prefix_stable_matches"] += int(
                    (
                        numerical["stable"][:, :seed_length]
                        & numerical["matches"][:, :seed_length]
                    ).sum()
                )
                seed_state["bonus_matches"] += int(numerical["matches"][0, 16])
                if split_length == HORIZON:
                    seed_state["full_accept_blocks"] += 1
                    seed_state["emitted_bonus_matches"] += int(
                        numerical["matches"][0, 16]
                    )
                seed_state["maximum_centered_epsilon"] = max(
                    float(seed_state["maximum_centered_epsilon"]),
                    float(numerical["epsilon"].max()),
                )
                del (
                    proposal,
                    split_logits,
                    unsplit_output,
                    unsplit_cache,
                    unsplit_input,
                    unsplit_logits,
                    unsplit_gold,
                    numerical,
                    seed_logits,
                )

            seed_input = seed_token

        sample_ids.append(str(row["sample_id"]))
        domains.append(str(row["domain"]))
        clean_domino_lengths.append(clean_domino_length)
        stored_domino_lengths.append(int(row["accepted_length"]))
        del (
            seed_cache,
            seed_tokens,
            seed_logit_rows,
            clean_prefix_cache,
            base_logits,
            hidden,
            context,
        )
        if index % 32 == 0 or index == len(records):
            print(f"evaluated {index}/{len(records)}", flush=True)

    clean_domino_eal = prompt_balanced(sample_ids, clean_domino_lengths)
    stored_domino_eal = prompt_balanced(sample_ids, stored_domino_lengths)
    clean_domino_domains = domain_prompt_balanced(
        sample_ids, domains, clean_domino_lengths
    )
    unsplit_eal: dict[int, float] = {}
    domain_no_regression: dict[int, bool] = {}
    seed_reports: dict[str, Any] = {}
    for seed in SEED_LENGTHS:
        seed_state = state[seed]
        split_eal = prompt_balanced(sample_ids, seed_state["split_lengths"])
        authority_eal = prompt_balanced(sample_ids, seed_state["unsplit_lengths"])
        split_domains = domain_prompt_balanced(
            sample_ids, domains, seed_state["split_lengths"]
        )
        authority_domains = domain_prompt_balanced(
            sample_ids, domains, seed_state["unsplit_lengths"]
        )
        domain_delta = {
            domain: authority_domains[domain] - clean_domino_domains[domain]
            for domain in sorted(authority_domains)
        }
        domain_pass = all(delta >= 0.0 for delta in domain_delta.values())
        unsplit_eal[seed] = authority_eal
        domain_no_regression[seed] = domain_pass
        stable_pass = (
            int(seed_state["stable_rows"]) > 0
            and int(seed_state["stable_rows"]) == int(seed_state["stable_matches"])
        )
        emitted_bonus_pass = int(seed_state["full_accept_blocks"]) == int(
            seed_state["emitted_bonus_matches"]
        )
        seed_reports[str(seed)] = {
            "target_geometry": (
                f"{seed} sequential seed rows + {HORIZON + 1 - seed} "
                "final verifier rows = 17"
            ),
            "eal_prompt_balanced": {
                "split_self_diagnostic": split_eal,
                "clean_unsplit_authority": authority_eal,
                "gain_vs_clean_domino": authority_eal - clean_domino_eal,
                "ratio_vs_historical_domino": authority_eal
                / HISTORICAL_DOMINO_EAL,
            },
            "domain_eal": {
                "split_self_diagnostic": split_domains,
                "clean_unsplit_authority": authority_domains,
                "authority_minus_clean_domino": domain_delta,
            },
            "paired_vs_clean_domino": paired_summary(
                clean_domino_lengths, seed_state["unsplit_lengths"]
            ),
            "split_vs_unsplit": paired_summary(
                seed_state["unsplit_lengths"], seed_state["split_lengths"]
            ),
            "numerical_control": {
                "rows": seed_state["all_rows"],
                "stable_rows": seed_state["stable_rows"],
                "ambiguous_rows": seed_state["ambiguous_rows"],
                "stable_matching_argmax_rows": seed_state["stable_matches"],
                "stable_non_tie_parity_passed": stable_pass,
                "seed_prefix_rows": seed_state["seed_prefix_rows"],
                "seed_prefix_matching_unsplit_rows": seed_state[
                    "seed_prefix_matches"
                ],
                "seed_prefix_stable_rows": seed_state[
                    "seed_prefix_stable_rows"
                ],
                "seed_prefix_stable_matches": seed_state[
                    "seed_prefix_stable_matches"
                ],
                "bonus_matching_rows": seed_state["bonus_matches"],
                "full_accept_blocks_emitting_bonus": seed_state[
                    "full_accept_blocks"
                ],
                "emitted_bonus_matching_rows": seed_state[
                    "emitted_bonus_matches"
                ],
                "emitted_bonus_exact_match_passed": emitted_bonus_pass,
                "maximum_row_centered_logit_epsilon": seed_state[
                    "maximum_centered_epsilon"
                ],
            },
            "gates": {
                "target_eal_passed": authority_eal >= TARGET_EAL,
                "system_eal_passed": authority_eal >= SYSTEM_EAL_GATE,
                "domain_no_regression_passed": domain_pass,
                "split_numerical_control_passed": stable_pass and emitted_bonus_pass,
            },
            "maximum_iteration_time_ratio_for_1p15x": (
                (authority_eal + 1.0)
                / (1.15 * (HISTORICAL_DOMINO_EAL + 1.0))
            ),
        }

    decision, selected_seed = choose_route(unsplit_eal)
    r050_eal = r050_report["eal_prompt_balanced"]
    # Re-evaluating the same clean Domino control detects evaluator drift while
    # leaving the seed decision independent of the historical stored labels.
    clean_control_delta = clean_domino_eal - float(
        r050_eal["released_domino_clean_replay"]
    )
    if abs(clean_control_delta) > 1e-12:
        raise RuntimeError("R051 clean Domino control drifted from R050")

    report: dict[str, Any] = {
        "status": "completed",
        "format": "r051_exact_prefix_fixed_v1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "source_rollout": str(args.source_rollout.resolve()),
        "target": str(args.target.resolve()),
        "domino_draft": str(args.domino_draft.resolve()),
        "r050_control_report": str(args.r050_report.resolve()),
        "split": args.split,
        "prompts": len(set(sample_ids)),
        "blocks": len(records),
        "candidate_topk": args.candidate_topk,
        "seed_lengths": list(SEED_LENGTHS),
        "trainable_parameters": 0,
        "accuracy_authority": (
            "actual split-sequential seeds -> forced Fast-K64 proposal -> "
            "clean batch-1 unsplit full-target verification"
        ),
        "controls": {
            "released_domino_clean_replay": clean_domino_eal,
            "released_domino_stored": stored_domino_eal,
            "ordinary_fast_k64_clean_r050": r050_eal[
                "ordinary_fast_k64_clean"
            ],
            "seed_1_clean_unsplit_authority_r050": r050_eal[
                "target_seeded_fast_k64_unsplit_control"
            ],
            "clean_domino_delta_vs_r050": clean_control_delta,
            "clean_vs_stored_domino_length_mismatches": sum(
                clean != stored
                for clean, stored in zip(
                    clean_domino_lengths, stored_domino_lengths, strict=True
                )
            ),
            "clean_domino_domain_eal": clean_domino_domains,
        },
        "seeds": seed_reports,
        "gates": {
            "target_eal": TARGET_EAL,
            "system_profile_eal": SYSTEM_EAL_GATE,
            "target_throughput_ratio": 1.15,
            "decision_uses_only_clean_unsplit_eal": True,
            "best_seed_by_unsplit_eal": max(
                SEED_LENGTHS, key=lambda seed: unsplit_eal[seed]
            ),
            "best_clean_unsplit_eal": max(unsplit_eal.values()),
            "selected_smallest_system_seed": selected_seed,
            "decision": decision,
        },
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
