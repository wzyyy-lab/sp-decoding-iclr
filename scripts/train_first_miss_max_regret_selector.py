#!/usr/bin/env python3
"""Capacity-only trainer for tie-safe Cost-Augmented Max-Regret Selection."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import time
from typing import Any

import torch
from torch import Tensor

from sph.first_miss_action_selector import (
    decode_action_indices,
    realized_prefix_lengths,
)
from sph.first_miss_capacity import sha256_file
from sph.first_miss_max_regret_selector import (
    BOUND_TOLERANCE,
    FirstMissMaxRegretSelector,
    first_miss_max_regret_loss,
)
from sph.global_direct_selector import GlobalDirectCandidateSelector

try:
    import train_first_miss_action_selector as fmas
    import train_global_direct_selector as direct
except ModuleNotFoundError:  # Imported as ``scripts.*`` by CPU tests.
    from scripts import train_first_miss_action_selector as fmas
    from scripts import train_global_direct_selector as direct


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_METHOD = "first_miss_cost_augmented_max_regret"
FROZEN_BLOCKS = 512
FROZEN_PROMPTS = 459
FROZEN_BENEFICIAL_ACTIONS = 256
FROZEN_HARMFUL_ACTIONS = 57_765
FROZEN_ORACLE_GAIN_TOKENS = 462
FROZEN_TOTAL_STEPS = 5_120
FROZEN_PROMPT_SHA256 = (
    "1e2be08968b2356f71e9818a5be5b8f3ecdd12ee50299ba6212a035f8a4d2707"
)
FROZEN_MANIFEST_SHA256 = (
    "d60613a00fc8557f4ff227ec302ced42de6a071d030b7ae7eb9eb5120bf5b67f"
)
FROZEN_MEAN_HINGE = 0.0030078125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-manifest", type=Path, required=True)
    parser.add_argument("--scope", choices=["global"], default="global")
    parser.add_argument("--mixer", choices=["axial"], default="axial")
    parser.add_argument("--node-encoder", choices=["additive"], default="additive")
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-subset-seed", type=int, default=20260730)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation_select")
    parser.add_argument("--memorization-blocks", type=int, default=FROZEN_BLOCKS)
    parser.add_argument(
        "--memorization-opportunity-fraction", type=float, default=0.5
    )
    parser.add_argument("--require-capacity-gate", action="store_true")
    parser.add_argument("--max-mean-hinge", type=float, default=FROZEN_MEAN_HINGE)
    parser.add_argument(
        "--min-beneficial-positive-count", type=int, default=254
    )
    parser.add_argument(
        "--min-utility-optimal-count", type=int, default=244
    )
    parser.add_argument(
        "--min-harmful-nonpositive-recall", type=float, default=0.99
    )
    parser.add_argument("--min-prompt-oracle-gap-recovered", type=float, default=0.95)
    parser.add_argument("--max-harmed-blocks", type=int, default=5)
    parser.add_argument("--max-no-benefit-false-edits", type=int, default=2)
    parser.add_argument(
        "--expected-beneficial-actions", type=int, default=FROZEN_BENEFICIAL_ACTIONS
    )
    parser.add_argument(
        "--expected-harmful-actions", type=int, default=FROZEN_HARMFUL_ACTIONS
    )
    parser.add_argument(
        "--expected-oracle-gain-tokens", type=int, default=FROZEN_ORACLE_GAIN_TOKENS
    )
    parser.add_argument("--expected-train-blocks", type=int, default=FROZEN_BLOCKS)
    parser.add_argument("--expected-train-prompts", type=int, default=FROZEN_PROMPTS)
    parser.add_argument("--expected-total-steps", type=int, default=FROZEN_TOTAL_STEPS)
    parser.add_argument(
        "--expected-train-prompt-sha256", default=FROZEN_PROMPT_SHA256
    )
    parser.add_argument(
        "--expected-capacity-manifest-sha256",
        default=FROZEN_MANIFEST_SHA256,
    )
    # Required attributes for the inherited, reviewed capacity loader.
    parser.set_defaults(
        train_data=[],
        max_train_prompts=0,
        evidence_tier="capacity_probe",
        skip_final_train_diagnostic=False,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    exact = {
        "candidate_k": 16,
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_size": 32,
        "epochs": 320,
        "learning_rate": 6e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.04,
        "gradient_clip": 1.0,
        "seed": 0,
        "memorization_blocks": FROZEN_BLOCKS,
        "memorization_opportunity_fraction": 0.5,
        "max_mean_hinge": FROZEN_MEAN_HINGE,
        "expected_beneficial_actions": FROZEN_BENEFICIAL_ACTIONS,
        "expected_harmful_actions": FROZEN_HARMFUL_ACTIONS,
        "expected_oracle_gain_tokens": FROZEN_ORACLE_GAIN_TOKENS,
        "expected_train_blocks": FROZEN_BLOCKS,
        "expected_train_prompts": FROZEN_PROMPTS,
        "expected_total_steps": FROZEN_TOTAL_STEPS,
        "expected_train_prompt_sha256": FROZEN_PROMPT_SHA256,
        "expected_capacity_manifest_sha256": FROZEN_MANIFEST_SHA256,
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            raise ValueError(f"--{name.replace('_', '-')} must equal frozen value {expected}")
    if args.scope != "global" or args.mixer != "axial":
        raise ValueError("CAMRS capacity requires global axial topology")
    if args.node_encoder != "additive":
        raise ValueError("CAMRS capacity requires additive node encoding")
    if args.train_split != "train" or args.validation_split != "validation_select":
        raise ValueError("split names differ from the frozen capacity contract")
    if args.min_beneficial_positive_count != 254:
        raise ValueError("beneficial-positive gate must be 254/256")
    if args.min_utility_optimal_count != 244:
        raise ValueError("utility-optimal gate must be 244/256")
    if args.max_harmed_blocks != 5 or args.max_no_benefit_false_edits != 2:
        raise ValueError("discrete harm/false-edit gates differ from the contract")
    if args.min_harmful_nonpositive_recall != 0.99:
        raise ValueError("harmful nonpositive gate must equal 0.99")
    if args.min_prompt_oracle_gap_recovered != 0.95:
        raise ValueError("prompt-balanced oracle-gap gate must equal 0.95")
    if not args.require_capacity_gate:
        raise ValueError("the frozen CAMRS run requires fail-closed capacity gating")


def _prompt_set_sha256(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(sorted({str(record["sample_id"]) for record in records}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cosine(left: Tensor, right: Tensor) -> float | None:
    left_norm = float(left.norm())
    right_norm = float(right.norm())
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(
        torch.nn.functional.cosine_similarity(
            left.reshape(1, -1), right.reshape(1, -1)
        )
    )


@torch.enable_grad()
def projection_gradient_diagnostics(
    model: FirstMissMaxRegretSelector,
    loader: Any,
    target_embedding: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Split active hinge gradients into oracle-up and competitor-down terms."""

    model.eval()
    projection = model.backbone.residual_projection.weight
    oracle_gradient = torch.zeros_like(projection, dtype=torch.float32)
    competitor_gradient = torch.zeros_like(projection, dtype=torch.float32)
    blocks = len(loader.dataset)
    active_blocks = 0
    active_repairable = 0
    for cpu_batch in loader:
        batch = direct.to_device(cpu_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                target_embedding[batch["candidate_ids"]],
                batch["candidate_logits"],
                batch["base_logsumexp"],
                target_embedding[batch["anchor_ids"]],
            )
            losses = first_miss_max_regret_loss(
                output,
                batch["gold_candidate_indices"],
                batch["gold_in_lattice"],
            )
        scores = output.action_values.float()
        active = losses.raw_max_violations.gt(0).float()
        oracle_scores = scores.gather(
            1, losses.oracle_actions[:, None]
        ).squeeze(1)
        competitor_scores = scores.gather(
            1, losses.competitor_actions[:, None]
        ).squeeze(1)
        active_blocks += int(active.sum())
        oracle_values = losses.target_values.gather(
            1, losses.oracle_actions[:, None]
        ).squeeze(1)
        active_repairable += int((active.bool() & oracle_values.gt(0)).sum())
        oracle_term = (-oracle_scores * active).sum() / float(blocks)
        competitor_term = (competitor_scores * active).sum() / float(blocks)
        oracle_part = torch.autograd.grad(
            oracle_term, projection, retain_graph=True, allow_unused=True
        )[0]
        competitor_part = torch.autograd.grad(
            competitor_term, projection, allow_unused=True
        )[0]
        if oracle_part is not None:
            oracle_gradient.add_(oracle_part.detach().float())
        if competitor_part is not None:
            competitor_gradient.add_(competitor_part.detach().float())
    total = oracle_gradient + competitor_gradient
    oracle_norm = float(oracle_gradient.norm())
    competitor_norm = float(competitor_gradient.norm())
    total_norm = float(total.norm())
    denominator = oracle_norm + competitor_norm
    model.zero_grad(set_to_none=True)
    return {
        "normalization": "sum_active_component/block_count",
        "blocks": blocks,
        "active_blocks": active_blocks,
        "active_repairable_blocks": active_repairable,
        "oracle_upward_projection_gradient_norm": oracle_norm,
        "competitor_downward_projection_gradient_norm": competitor_norm,
        "total_projection_gradient_norm": total_norm,
        "oracle_cosine_with_total": _cosine(oracle_gradient, total),
        "competitor_cosine_with_total": _cosine(competitor_gradient, total),
        "oracle_competitor_cosine": _cosine(
            oracle_gradient, competitor_gradient
        ),
        "cancellation_fraction": (
            1.0 - total_norm / denominator if denominator > 0 else None
        ),
    }


@torch.inference_mode()
def evaluate(
    model: FirstMissMaxRegretSelector,
    loader: Any,
    target_embedding: Tensor,
    device: torch.device,
    *,
    candidate_k: int,
    include_examples: bool = False,
    require_base_identity: bool = False,
) -> dict[str, Any]:
    """Evaluate CAMRS and retain enough records to reconstruct every gate."""

    model.eval()
    records: list[dict[str, Any]] = []
    hinge_sum = 0.0
    hinge_max = 0.0
    zero_loss_blocks = 0
    bound_min_slack = math.inf
    bound_violations = 0
    beneficial_actions = 0
    beneficial_positive = 0
    harmful_actions = 0
    harmful_nonpositive = 0
    repairable_blocks = 0
    utility_optimal_selected = 0
    no_benefit_blocks = 0
    no_benefit_false_edits = 0
    selected_edits = 0
    selected_beneficial = 0
    selected_neutral = 0
    selected_harmful = 0
    decoded_regret_sum = 0.0
    competitor_equals_deployed = 0
    competitor_cost_only = 0
    competitor_ranks: list[int] = []
    competitor_actions_vector: list[int] = []
    competitor_signs: Counter[str] = Counter()
    competitor_regret_tokens: list[int] = []
    oracle_gain_tokens = 0
    block_length: int | None = None

    for cpu_batch in loader:
        batch = direct.to_device(cpu_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                target_embedding[batch["candidate_ids"]],
                batch["candidate_logits"],
                batch["base_logsumexp"],
                target_embedding[batch["anchor_ids"]],
            )
            losses = first_miss_max_regret_loss(
                output,
                batch["gold_candidate_indices"],
                batch["gold_in_lattice"],
            )

        batch_size, length, candidates = batch["candidate_ids"].shape
        block_length = length
        if candidates != candidate_k:
            raise RuntimeError("loader candidate K differs from evaluation K")
        if require_base_identity:
            expected_scores = (
                batch["candidate_logits"].float()
                - batch["base_logsumexp"].float().unsqueeze(-1)
            )
            if not torch.equal(output.direct_output.scores, expected_scores):
                raise RuntimeError("epoch-zero CAMRS does not reproduce DFlash scores")
            if not torch.equal(
                output.direct_output.residual_scores,
                torch.zeros_like(output.direct_output.residual_scores),
            ):
                raise RuntimeError("epoch-zero residual scores are not zero")
            if not torch.equal(
                output.action_values, torch.zeros_like(output.action_values)
            ):
                raise RuntimeError("epoch-zero action scores are not zero")
            if not torch.equal(
                losses.predicted_actions,
                torch.zeros_like(losses.predicted_actions),
            ):
                raise RuntimeError("epoch-zero CAMRS action is not KEEP")

        scores = output.action_values.float()
        targets = losses.target_values
        oracle_values = targets.gather(
            1, losses.oracle_actions[:, None]
        ).squeeze(1)
        selected_values = targets.gather(
            1, losses.predicted_actions[:, None]
        ).squeeze(1)
        competitor_values = targets.gather(
            1, losses.competitor_actions[:, None]
        ).squeeze(1)
        competitor_scores = scores.gather(
            1, losses.competitor_actions[:, None]
        ).squeeze(1)
        action_indices = torch.arange(scores.shape[1], device=device)[None, :]
        raw_ranks = 1 + scores.gt(competitor_scores[:, None]).sum(dim=-1)
        raw_ranks += (
            scores.eq(competitor_scores[:, None])
            & action_indices.lt(losses.competitor_actions[:, None])
        ).sum(dim=-1)
        raw_score_actions = scores.argmax(dim=-1)

        target_edits = targets[:, 1:]
        predicted_edits = scores[:, 1:]
        beneficial_mask = target_edits.gt(0)
        harmful_mask = target_edits.lt(0)
        beneficial_actions += int(beneficial_mask.sum())
        beneficial_positive += int(predicted_edits[beneficial_mask].gt(0).sum())
        harmful_actions += int(harmful_mask.sum())
        harmful_nonpositive += int(
            predicted_edits[harmful_mask].le(0).sum()
        )

        repairable = oracle_values.gt(0)
        optimal = selected_values.eq(oracle_values)
        no_benefit = ~repairable
        edits = losses.predicted_actions.ne(0)
        beneficial_selected = selected_values.gt(0)
        harmful_selected = selected_values.lt(0)
        neutral_selected = edits & selected_values.eq(0)
        repairable_blocks += int(repairable.sum())
        utility_optimal_selected += int((repairable & optimal).sum())
        no_benefit_blocks += int(no_benefit.sum())
        no_benefit_false_edits += int((no_benefit & edits).sum())
        selected_edits += int(edits.sum())
        selected_beneficial += int(beneficial_selected.sum())
        selected_harmful += int(harmful_selected.sum())
        selected_neutral += int(neutral_selected.sum())

        hinge_sum += float(losses.per_block_hinge.detach().sum())
        hinge_max = max(hinge_max, float(losses.per_block_hinge.max()))
        zero_loss_blocks += int(losses.per_block_hinge.eq(0).sum())
        bound_min_slack = min(bound_min_slack, float(losses.bound_slack.min()))
        bound_violations += int(
            losses.bound_slack.lt(-BOUND_TOLERANCE).sum()
        )
        decoded_regret_sum += float(losses.decoded_regret.sum())
        oracle_gain_tokens += int(round(float(oracle_values.sum()) * length))
        competitor_equals_deployed += int(
            losses.competitor_actions.eq(losses.predicted_actions).sum()
        )
        competitor_cost_only += int(
            losses.competitor_actions.ne(raw_score_actions).sum()
        )
        competitor_ranks.extend(raw_ranks.detach().cpu().tolist())
        competitor_actions_vector.extend(
            losses.competitor_actions.detach().cpu().tolist()
        )
        for value in competitor_values.detach().cpu().tolist():
            competitor_signs[
                "beneficial" if value > 0 else "harmful" if value < 0 else "neutral"
            ] += 1
        competitor_regret_tokens.extend(
            ((oracle_values - competitor_values) * length)
            .round()
            .to(torch.int64)
            .detach()
            .cpu()
            .tolist()
        )

        base_paths = torch.zeros(
            batch_size, length, dtype=torch.long, device=device
        )
        camrs_paths = decode_action_indices(
            losses.predicted_actions, length=length, candidates=candidates
        )
        oracle_paths = decode_action_indices(
            losses.oracle_actions, length=length, candidates=candidates
        )
        native_paths = output.direct_output.scores.argmax(dim=-1)
        paths = {
            "base": base_paths,
            "camrs": camrs_paths,
            "single_edit_oracle": oracle_paths,
            "direct_native": native_paths,
        }
        realized = {
            name: realized_prefix_lengths(
                path, batch["candidate_ids"], batch["gold_ids"]
            )
            for name, path in paths.items()
        }
        torch.testing.assert_close(
            selected_values * float(length),
            (realized["camrs"] - realized["base"]).float(),
            rtol=0.0,
            atol=1e-6,
        )

        for item, (sample_id, domain) in enumerate(
            zip(batch["sample_ids"], batch["domains"], strict=True)
        ):
            record: dict[str, Any] = {
                "sample_id": str(sample_id),
                "domain": str(domain),
                "accepted_draft_tokens": {
                    name: int(values[item]) for name, values in realized.items()
                },
                "first_token_correct": {
                    name: bool(values[item] > 0)
                    for name, values in realized.items()
                },
                "predicted_action": int(losses.predicted_actions[item]),
                "oracle_action": int(losses.oracle_actions[item]),
                "competitor_action": int(losses.competitor_actions[item]),
                "selected_true_value": float(selected_values[item]),
                "oracle_true_value": float(oracle_values[item]),
                "competitor_true_value": float(competitor_values[item]),
                "hinge": float(losses.per_block_hinge[item]),
                "raw_max_violation": float(losses.raw_max_violations[item]),
                "decoded_regret": float(losses.decoded_regret[item]),
                "bound_slack": float(losses.bound_slack[item]),
                "competitor_raw_score_rank": int(raw_ranks[item]),
                "competitor_equals_deployed": bool(
                    losses.competitor_actions[item]
                    == losses.predicted_actions[item]
                ),
                "competitor_selected_only_by_cost_augmentation": bool(
                    losses.competitor_actions[item] != raw_score_actions[item]
                ),
            }
            if include_examples:
                record["candidate_path_indices"] = {
                    name: path[item].detach().cpu().tolist()
                    for name, path in paths.items()
                }
                record["predicted_action_scores"] = (
                    scores[item].detach().cpu().tolist()
                )
                record["target_action_values"] = (
                    targets[item].detach().cpu().tolist()
                )
            records.append(record)

    if not records or block_length is None:
        raise RuntimeError("evaluation loader produced no records")
    report: dict[str, Any] = {
        "loss": {
            "objective": hinge_sum / len(records),
            "mean_block_hinge": hinge_sum / len(records),
            "maximum_block_hinge": hinge_max,
            "zero_loss_blocks": zero_loss_blocks,
            "zero_loss_fraction": zero_loss_blocks / len(records),
        },
        "bound": {
            "minimum_slack": bound_min_slack,
            "tolerance": BOUND_TOLERANCE,
            "violations_beyond_tolerance": bound_violations,
            "mean_decoded_regret_normalized": decoded_regret_sum / len(records),
            "mean_decoded_regret_tokens": (
                decoded_regret_sum * block_length / len(records)
            ),
        },
        "signed_score": {
            "beneficial_actions": beneficial_actions,
            "beneficial_predicted_positive": beneficial_positive,
            "beneficial_strict_positive_recall": (
                beneficial_positive / beneficial_actions
                if beneficial_actions else None
            ),
            "harmful_actions": harmful_actions,
            "harmful_predicted_nonpositive": harmful_nonpositive,
            "harmful_nonpositive_recall": (
                harmful_nonpositive / harmful_actions if harmful_actions else None
            ),
        },
        "decision": {
            "repairable_blocks": repairable_blocks,
            "utility_optimal_selected": utility_optimal_selected,
            "utility_optimal_action_accuracy": (
                utility_optimal_selected / repairable_blocks
                if repairable_blocks else None
            ),
            "no_benefit_blocks": no_benefit_blocks,
            "no_benefit_false_edits": no_benefit_false_edits,
            "no_benefit_false_edit_rate": (
                no_benefit_false_edits / no_benefit_blocks
                if no_benefit_blocks else None
            ),
            "selected_edits": selected_edits,
            "selected_beneficial": selected_beneficial,
            "selected_neutral": selected_neutral,
            "selected_harmful": selected_harmful,
            "harmed_fraction": selected_harmful / len(records),
            "edit_precision": (
                selected_beneficial / selected_edits if selected_edits else None
            ),
            "repair_recall": (
                selected_beneficial / repairable_blocks
                if repairable_blocks else None
            ),
            "oracle_gain_tokens_block_weighted": oracle_gain_tokens,
        },
        "competitor": {
            "equals_deployed_fraction": competitor_equals_deployed / len(records),
            "selected_only_by_cost_augmentation_fraction": (
                competitor_cost_only / len(records)
            ),
            "distinct_actions": len(set(competitor_actions_vector)),
            "action_histogram": dict(
                sorted(Counter(competitor_actions_vector).items())
            ),
            "utility_sign_counts": dict(competitor_signs),
            "raw_score_rank": {
                "mean": sum(competitor_ranks) / len(competitor_ranks),
                "minimum": min(competitor_ranks),
                "maximum": max(competitor_ranks),
                "histogram": dict(sorted(Counter(competitor_ranks).items())),
            },
            "regret_tokens": {
                "mean": sum(competitor_regret_tokens)
                / len(competitor_regret_tokens),
                "histogram": dict(
                    sorted(Counter(competitor_regret_tokens).items())
                ),
            },
            "actions_by_evaluation_order": competitor_actions_vector,
            "actions_sha256": hashlib.sha256(
                json.dumps(competitor_actions_vector, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "blocks": len(records),
        "prompts": len({record["sample_id"] for record in records}),
    }
    for method in ("base", "camrs", "single_edit_oracle", "direct_native"):
        report[method] = direct._method_summary(records, method)
    report["by_domain"] = {}
    for domain in sorted({record["domain"] for record in records}):
        subset = [record for record in records if record["domain"] == domain]
        report["by_domain"][domain] = {
            method: direct._method_summary(subset, method)
            for method in ("base", "camrs", "single_edit_oracle", "direct_native")
        }
    base_eal = report["base"]["mean_accepted_draft_tokens_prompt_balanced"]
    camrs_eal = report["camrs"]["mean_accepted_draft_tokens_prompt_balanced"]
    oracle_eal = report["single_edit_oracle"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    denominator = oracle_eal - base_eal
    report["decision"]["prompt_balanced_oracle_gap_recovered"] = (
        (camrs_eal - base_eal) / denominator if denominator > 0 else None
    )
    report["decision"]["block_weighted_oracle_gap_recovered"] = (
        1.0
        - report["bound"]["mean_decoded_regret_tokens"]
        / (oracle_gain_tokens / len(records))
        if oracle_gain_tokens > 0 else None
    )
    if include_examples:
        report["examples"] = records
    return report


def capacity_gate_report(
    evaluation: dict[str, Any],
    args: argparse.Namespace,
    *,
    epoch_zero_identity: bool,
) -> dict[str, Any]:
    signed = evaluation["signed_score"]
    decision = evaluation["decision"]
    bound = evaluation["bound"]
    loss = evaluation["loss"]
    minimum_harmful_nonpositive = math.ceil(
        args.min_harmful_nonpositive_recall * args.expected_harmful_actions
    )
    prompt_gap = decision["prompt_balanced_oracle_gap_recovered"]
    values = {
        "bound_violations": bound["violations_beyond_tolerance"],
        "minimum_bound_slack": bound["minimum_slack"],
        "mean_block_hinge": loss["mean_block_hinge"],
        "beneficial_actions": signed["beneficial_actions"],
        "beneficial_predicted_positive": signed["beneficial_predicted_positive"],
        "repairable_blocks": decision["repairable_blocks"],
        "utility_optimal_selected": decision["utility_optimal_selected"],
        "harmful_actions": signed["harmful_actions"],
        "harmful_predicted_nonpositive": signed["harmful_predicted_nonpositive"],
        "prompt_balanced_oracle_gap_recovered": prompt_gap,
        "selected_harmful": decision["selected_harmful"],
        "no_benefit_blocks": decision["no_benefit_blocks"],
        "no_benefit_false_edits": decision["no_benefit_false_edits"],
        "oracle_gain_tokens": decision["oracle_gain_tokens_block_weighted"],
        "blocks": evaluation["blocks"],
        "prompts": evaluation["prompts"],
        "epoch_zero_identity": epoch_zero_identity,
    }
    thresholds = {
        "bound_violations": 0,
        "minimum_bound_slack": -BOUND_TOLERANCE,
        "mean_block_hinge": args.max_mean_hinge,
        "beneficial_actions": args.expected_beneficial_actions,
        "beneficial_predicted_positive": args.min_beneficial_positive_count,
        "repairable_blocks": FROZEN_BENEFICIAL_ACTIONS,
        "utility_optimal_selected": args.min_utility_optimal_count,
        "harmful_actions": args.expected_harmful_actions,
        "harmful_predicted_nonpositive": minimum_harmful_nonpositive,
        "prompt_balanced_oracle_gap_recovered": (
            args.min_prompt_oracle_gap_recovered
        ),
        "selected_harmful": args.max_harmed_blocks,
        "no_benefit_blocks": FROZEN_BLOCKS - FROZEN_BENEFICIAL_ACTIONS,
        "no_benefit_false_edits": args.max_no_benefit_false_edits,
        "oracle_gain_tokens": args.expected_oracle_gain_tokens,
        "blocks": FROZEN_BLOCKS,
        "prompts": FROZEN_PROMPTS,
        "epoch_zero_identity": True,
    }
    finite_gate_values = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in (
            values["minimum_bound_slack"],
            values["mean_block_hinge"],
            values["prompt_balanced_oracle_gap_recovered"],
        )
    )
    checks = {
        "finite_gate_values": finite_gate_values,
        "bound_violations": values["bound_violations"] == 0,
        "minimum_bound_slack": (
            values["minimum_bound_slack"] >= thresholds["minimum_bound_slack"]
        ),
        "mean_block_hinge": (
            finite_gate_values
            and 0.0 <= values["mean_block_hinge"] <= args.max_mean_hinge
        ),
        "beneficial_actions": (
            values["beneficial_actions"] == args.expected_beneficial_actions
        ),
        "beneficial_predicted_positive": (
            0
            <= values["beneficial_predicted_positive"]
            <= values["beneficial_actions"]
            and values["beneficial_predicted_positive"]
            >= args.min_beneficial_positive_count
        ),
        "repairable_blocks": values["repairable_blocks"] == FROZEN_BENEFICIAL_ACTIONS,
        "utility_optimal_selected": (
            0
            <= values["utility_optimal_selected"]
            <= values["repairable_blocks"]
            and values["utility_optimal_selected"]
            >= args.min_utility_optimal_count
        ),
        "harmful_actions": values["harmful_actions"] == args.expected_harmful_actions,
        "harmful_predicted_nonpositive": (
            0
            <= values["harmful_predicted_nonpositive"]
            <= values["harmful_actions"]
            and values["harmful_predicted_nonpositive"]
            >= minimum_harmful_nonpositive
        ),
        "prompt_balanced_oracle_gap_recovered": (
            finite_gate_values
            and values["prompt_balanced_oracle_gap_recovered"]
            >= args.min_prompt_oracle_gap_recovered
            and values["prompt_balanced_oracle_gap_recovered"] <= 1.0 + 1e-6
        ),
        "selected_harmful": (
            0
            <= values["selected_harmful"]
            <= values["blocks"]
            and values["selected_harmful"] <= args.max_harmed_blocks
        ),
        "no_benefit_blocks": (
            values["no_benefit_blocks"]
            == FROZEN_BLOCKS - FROZEN_BENEFICIAL_ACTIONS
        ),
        "no_benefit_false_edits": (
            0
            <= values["no_benefit_false_edits"]
            <= values["no_benefit_blocks"]
            and values["no_benefit_false_edits"]
            <= args.max_no_benefit_false_edits
        ),
        "oracle_gain_tokens": (
            values["oracle_gain_tokens"] == args.expected_oracle_gain_tokens
        ),
        "blocks": values["blocks"] == FROZEN_BLOCKS,
        "prompts": values["prompts"] == FROZEN_PROMPTS,
        "epoch_zero_identity": bool(epoch_zero_identity),
    }
    return {
        "passed": all(checks.values()),
        "values": values,
        "thresholds": thresholds,
        "checks": checks,
        "aggregation": {
            "hinge": "uniform_block_mean_normalized_tokens",
            "hinge_reduction": "FP32_tensor_then_Python_float_accumulation",
            "oracle_gap": "prompt_balanced_eal",
            "block_oracle_advantage": "462/(512*15)=0.06015625",
        },
        "failure_scope": (
            "exact tie-safe CAMRS objective, D64/H4/L1 axial-additive model, "
            "AdamW/clip/schedule, frozen 512-block manifest, 5120-step budget, "
            "and minimum-hinge checkpoint rule"
        ),
    }


def compact_epoch_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    decision = evaluation["decision"]
    return {
        "mean_block_hinge": evaluation["loss"]["mean_block_hinge"],
        "maximum_block_hinge": evaluation["loss"]["maximum_block_hinge"],
        "zero_loss_fraction": evaluation["loss"]["zero_loss_fraction"],
        "zero_loss_blocks": evaluation["loss"]["zero_loss_blocks"],
        "minimum_bound_slack": evaluation["bound"]["minimum_slack"],
        "bound_violations": evaluation["bound"]["violations_beyond_tolerance"],
        "beneficial_strict_positive_recall": evaluation["signed_score"][
            "beneficial_strict_positive_recall"
        ],
        "harmful_nonpositive_recall": evaluation["signed_score"][
            "harmful_nonpositive_recall"
        ],
        "utility_optimal_action_accuracy": decision[
            "utility_optimal_action_accuracy"
        ],
        "prompt_balanced_oracle_gap_recovered": decision[
            "prompt_balanced_oracle_gap_recovered"
        ],
        "block_weighted_oracle_gap_recovered": decision[
            "block_weighted_oracle_gap_recovered"
        ],
        "harmed_fraction": decision["harmed_fraction"],
        "no_benefit_false_edit_rate": decision["no_benefit_false_edit_rate"],
        "base_eal": evaluation["base"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "camrs_eal": evaluation["camrs"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "single_edit_oracle_eal": evaluation["single_edit_oracle"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "direct_native_eal": evaluation["direct_native"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "competitor": {
            key: evaluation["competitor"][key]
            for key in (
                "equals_deployed_fraction",
                "selected_only_by_cost_augmentation_fraction",
                "distinct_actions",
                "utility_sign_counts",
                "raw_score_rank",
                "regret_tokens",
                "actions_sha256",
            )
        },
    }


def _competitor_churn(current: list[int], previous: list[int] | None) -> float | None:
    if previous is None:
        return None
    if len(current) != len(previous):
        raise RuntimeError("competitor vector length changed across epochs")
    return sum(left != right for left, right in zip(current, previous, strict=True)) / len(current)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CAMRS capacity training requires CUDA")

    direct.seed_everything(args.seed)
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    config_snapshot = direct.serializable_config(args)
    source_paths = {
        "trainer": Path(__file__).resolve(),
        "camrs_head": PROJECT / "src/sph/first_miss_max_regret_selector.py",
        "savs_score_head": PROJECT / "src/sph/first_miss_value_selector.py",
        "fmas_action_semantics": PROJECT / "src/sph/first_miss_action_selector.py",
        "fmas_data_protocol": PROJECT / "scripts/train_first_miss_action_selector.py",
        "capacity_helper": PROJECT / "src/sph/first_miss_capacity.py",
        "direct_head": PROJECT / "src/sph/global_direct_selector.py",
        "direct_trainer_utilities": PROJECT / "scripts/train_global_direct_selector.py",
        "canonical_data": PROJECT / "src/sph/data.py",
        "proposal": PROJECT
        / "refine-logs/first-miss-max-regret/FINAL_PROPOSAL.md",
    }
    source_hashes_start = {
        f"{name}_sha256": sha256_file(path) for name, path in source_paths.items()
    }
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, path in source_paths.items():
        shutil.copy2(path, snapshot / f"{name}_{path.name}")
    start = time.perf_counter()

    metadata_path = args.data / "metadata.json"
    if sha256_file(args.capacity_manifest) != args.expected_capacity_manifest_sha256:
        raise RuntimeError("capacity manifest hash differs from frozen expectation")
    metadata = json.loads(metadata_path.read_text())
    if args.candidate_k > int(metadata.get("top_k", 0)):
        raise ValueError("candidate K exceeds canonical collection top-K")
    verified_target_files = direct.validate_target_embedding_identity(
        metadata, args.target
    )
    (
        train_dataset,
        validation_dataset,
        collection,
        external_collections,
        verified_external,
        capacity_manifest,
    ) = fmas._load_datasets(args)
    if validation_dataset is not train_dataset:
        raise RuntimeError("CAMRS Gate 1 must be an exact same-subset probe")
    train_prompts = len({str(record["sample_id"]) for record in train_dataset.records})
    train_prompt_sha256 = _prompt_set_sha256(train_dataset.records)
    if len(train_dataset) != args.expected_train_blocks:
        raise RuntimeError("training block count differs from frozen expectation")
    if train_prompts != args.expected_train_prompts:
        raise RuntimeError("training prompt count differs from frozen expectation")
    if train_prompt_sha256 != args.expected_train_prompt_sha256:
        raise RuntimeError("training prompt set differs from frozen expectation")

    train_loader = direct.make_loader(
        train_dataset,
        candidate_k=args.candidate_k,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = direct.make_loader(
        validation_dataset,
        candidate_k=args.candidate_k,
        batch_size=args.batch_size,
        shuffle=False,
    )
    target_embedding = (
        direct.load_target_embedding(args.target)
        .to(device=device, dtype=torch.bfloat16)
        .detach()
    )
    target_embedding.requires_grad_(False)
    hidden_size = int(target_embedding.shape[1])
    block_length = int(train_dataset.records[0]["gold_ids"].numel())
    if block_length != 15:
        raise RuntimeError("CAMRS capacity contract requires 15 positions")

    backbone = GlobalDirectCandidateSelector(
        hidden_size=hidden_size,
        max_positions=block_length,
        max_candidates=args.candidate_k,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        scope=args.scope,
        mixer=args.mixer,
        node_encoder=args.node_encoder,
        dropout=args.dropout,
        initialization_seed=args.seed,
    )
    model = FirstMissMaxRegretSelector(backbone).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    named_parameters = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for _, p in named_parameters if p.ndim >= 2],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for _, p in named_parameters if p.ndim < 2],
                "weight_decay": 0.0,
            },
        ],
        lr=args.learning_rate,
    )
    total_steps = args.epochs * len(train_loader)
    if total_steps != args.expected_total_steps:
        raise RuntimeError("optimizer-step budget differs from frozen expectation")
    scheduler, warmup_steps = direct.cosine_warmup_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio
    )

    initial_validation = evaluate(
        model,
        validation_loader,
        target_embedding,
        device,
        candidate_k=args.candidate_k,
        require_base_identity=True,
    )
    epoch_zero_identity = True
    initial_projection = projection_gradient_diagnostics(
        model, validation_loader, target_embedding, device
    )
    initial_gate = capacity_gate_report(
        initial_validation, args, epoch_zero_identity=epoch_zero_identity
    )
    initial_competitors = initial_validation["competitor"][
        "actions_by_evaluation_order"
    ]
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train": None,
            "validation": compact_epoch_metrics(initial_validation),
            "projection_gradient_diagnostics": initial_projection,
            "competitor_churn_fraction": None,
            "capacity_gate": initial_gate,
            "joint_gate_passed": initial_gate["passed"],
            "selection_key": [-initial_validation["loss"]["mean_block_hinge"]],
            "is_selected": False,
        }
    ]
    best_key = (-float(initial_validation["loss"]["mean_block_hinge"]),)
    best_path = args.output / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "args": config_snapshot,
            "epoch": 0,
            "selection_key": best_key,
            "parameter_count": parameter_count,
        },
        best_path,
    )
    print(json.dumps(history[0], ensure_ascii=False), flush=True)

    previous_competitors: list[int] | None = initial_competitors
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        hinge_sum = 0.0
        examples_seen = 0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        clipped_steps = 0
        for cpu_batch in train_loader:
            batch = direct.to_device(cpu_batch, device)
            batch_size = int(batch["hidden"].shape[0])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    batch["hidden"],
                    target_embedding[batch["candidate_ids"]],
                    batch["candidate_logits"],
                    batch["base_logsumexp"],
                    target_embedding[batch["anchor_ids"]],
                )
                losses = first_miss_max_regret_loss(
                    output,
                    batch["gold_candidate_indices"],
                    batch["gold_in_lattice"],
                )
            if not bool(torch.isfinite(losses.loss)):
                raise FloatingPointError("nonfinite CAMRS training loss")
            losses.loss.backward()
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip, error_if_nonfinite=True
            )
            grad_norm = float(grad_norm_tensor.detach())
            clipped_steps += int(grad_norm > args.gradient_clip)
            grad_norm_sum += grad_norm
            grad_norm_max = max(grad_norm_max, grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            examples_seen += batch_size
            hinge_sum += float(losses.per_block_hinge.detach().sum())

        validation = evaluate(
            model,
            validation_loader,
            target_embedding,
            device,
            candidate_k=args.candidate_k,
        )
        projection = projection_gradient_diagnostics(
            model, validation_loader, target_embedding, device
        )
        gate = capacity_gate_report(
            validation, args, epoch_zero_identity=epoch_zero_identity
        )
        current_competitors = validation["competitor"][
            "actions_by_evaluation_order"
        ]
        churn = _competitor_churn(current_competitors, previous_competitors)
        previous_competitors = current_competitors
        key = (-float(validation["loss"]["mean_block_hinge"]),)
        epoch_record = {
            "epoch": epoch,
            "train": {
                "mean_block_hinge": hinge_sum / examples_seen,
                "mean_preclip_grad_norm": grad_norm_sum / len(train_loader),
                "maximum_preclip_grad_norm": grad_norm_max,
                "clipped_steps": clipped_steps,
                "clipped_step_fraction": clipped_steps / len(train_loader),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "blocks": examples_seen,
                "steps": global_step,
            },
            "validation": compact_epoch_metrics(validation),
            "projection_gradient_diagnostics": projection,
            "competitor_churn_fraction": churn,
            "capacity_gate": gate,
            "joint_gate_passed": gate["passed"],
            "selection_key": list(key),
            "is_selected": False,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        if key > best_key:
            best_key = key
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": config_snapshot,
                    "epoch": epoch,
                    "selection_key": best_key,
                    "parameter_count": parameter_count,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    selected_epoch = int(checkpoint["epoch"])
    history[selected_epoch]["is_selected"] = True
    jointly_passing_epochs = [
        int(record["epoch"]) for record in history if record["joint_gate_passed"]
    ]
    model.load_state_dict(checkpoint["model"])
    final_validation = evaluate(
        model,
        validation_loader,
        target_embedding,
        device,
        candidate_k=args.candidate_k,
        include_examples=True,
    )
    capacity_gate = capacity_gate_report(
        final_validation, args, epoch_zero_identity=epoch_zero_identity
    )
    if capacity_gate["passed"] != history[selected_epoch]["joint_gate_passed"]:
        raise RuntimeError("selected checkpoint gate differs from epoch history")

    source_hashes_end = {
        f"{name}_sha256_at_end": sha256_file(path)
        for name, path in source_paths.items()
    }
    for name, path in source_paths.items():
        if source_hashes_start[f"{name}_sha256"] != source_hashes_end[
            f"{name}_sha256_at_end"
        ]:
            raise RuntimeError(f"source changed during run: {path}")

    provenance = {
        "project_commit": direct.git_revision(PROJECT),
        "project_dirty": direct.git_is_dirty(PROJECT),
        "data_metadata_sha256": sha256_file(metadata_path),
        **source_hashes_start,
        **source_hashes_end,
        "capacity_manifest_sha256": sha256_file(args.capacity_manifest),
        "capacity_subset_sha256": capacity_manifest["subset_sha256"],
        "verified_target_embedding_files": verified_target_files,
        "verified_external_target_embedding_files": verified_external,
        "external_train_collections": len(external_collections),
        "base_greedy_witness_status": collection.base_greedy_witness_status,
        "dflash_commit": direct.git_revision(PROJECT / "third_party/dflash"),
        "domino_commit": direct.git_revision(PROJECT / "third_party/Domino"),
        "dpace_commit": direct.git_revision(PROJECT / "third_party/D-PACE"),
    }
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config": config_snapshot,
        "primary_method": PRIMARY_METHOD,
        "evidence_tier": "capacity_probe",
        "split_protocol": "same_subset_adaptive_capacity_probe",
        "train_blocks": len(train_dataset),
        "train_prompts": train_prompts,
        "train_prompt_set_sha256": train_prompt_sha256,
        "validation_blocks": len(validation_dataset),
        "validation_prompts": train_prompts,
        "parameter_count": parameter_count,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "selected_epoch": selected_epoch,
        "selection_key": checkpoint["selection_key"],
        "jointly_passing_epochs": jointly_passing_epochs,
        "selected_checkpoint_passed": capacity_gate["passed"],
        "seconds": time.perf_counter() - start,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "capacity_manifest": capacity_manifest,
        "initial_projection_gradient_diagnostics": initial_projection,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "capacity_gate": capacity_gate,
        "history": history,
        "provenance": provenance,
    }
    metrics_path = args.output / "metrics.json"
    temporary_path = metrics_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(metrics_path)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "selected_epoch": selected_epoch,
                "jointly_passing_epochs": jointly_passing_epochs,
                "capacity_gate": capacity_gate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.require_capacity_gate and not capacity_gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
