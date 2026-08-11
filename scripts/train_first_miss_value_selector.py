#!/usr/bin/env python3
"""Train Signed Action-Value Selection over a frozen DFlash lattice."""

from __future__ import annotations

import argparse
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
from sph.first_miss_value_selector import (
    FirstMissValueSelector,
    decode_strict_positive_actions,
    first_miss_value_loss,
)
from sph.global_direct_selector import GlobalDirectCandidateSelector

try:
    import train_first_miss_action_selector as fmas
    import train_global_direct_selector as direct
except ModuleNotFoundError:  # Imported as ``scripts.*`` in CPU tests.
    from scripts import train_first_miss_action_selector as fmas
    from scripts import train_global_direct_selector as direct


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_METHOD = "first_miss_signed_action_value"
VALUE_CLASSES = ("beneficial", "neutral", "harmful")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, action="append", default=[])
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=["local", "causal", "global"], default="global"
    )
    parser.add_argument("--mixer", choices=["flat", "axial"], default="axial")
    parser.add_argument(
        "--node-encoder",
        choices=["additive", "compatibility"],
        default="additive",
    )
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-subset-seed", type=int, default=20260730)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation_select")
    parser.add_argument("--max-train-prompts", type=int, default=0)
    parser.add_argument("--memorization-blocks", type=int, default=0)
    parser.add_argument(
        "--memorization-opportunity-fraction", type=float, default=0.5
    )
    parser.add_argument("--capacity-manifest", type=Path)
    parser.add_argument("--require-capacity-gate", action="store_true")
    parser.add_argument("--max-value-rmse", type=float, default=0.02)
    parser.add_argument(
        "--min-beneficial-sign-recall", type=float, default=0.99
    )
    parser.add_argument(
        "--min-harmful-nonpositive-recall", type=float, default=0.99
    )
    parser.add_argument("--min-oracle-gap-recovered", type=float, default=0.95)
    parser.add_argument("--max-harmed-fraction", type=float, default=0.01)
    parser.add_argument("--expected-beneficial-actions", type=int, default=0)
    parser.add_argument(
        "--evidence-tier",
        choices=["smoke", "capacity_probe", "development"],
        default="development",
    )
    parser.add_argument("--expected-train-blocks", type=int, default=0)
    parser.add_argument("--expected-train-prompts", type=int, default=0)
    parser.add_argument("--expected-total-steps", type=int, default=0)
    parser.add_argument("--expected-train-prompt-sha256", default="")
    parser.add_argument("--skip-final-train-diagnostic", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.candidate_k < 2:
        raise ValueError("--candidate-k must be at least two")
    if min(args.model_dim, args.num_heads, args.num_layers) < 1:
        raise ValueError("model dimensions and layer count must be positive")
    if args.model_dim % args.num_heads:
        raise ValueError("--model-dim must be divisible by --num-heads")
    if min(args.batch_size, args.epochs) < 1:
        raise ValueError("batch size and epochs must be positive")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning rate and gradient clip must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight decay cannot be negative")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("warmup ratio must be in [0, 1]")
    if not 0.0 <= args.memorization_opportunity_fraction <= 1.0:
        raise ValueError("opportunity fraction must be in [0, 1]")
    if args.max_train_prompts and args.memorization_blocks:
        raise ValueError("prompt subsampling and memorization cannot be combined")
    if args.train_data and args.memorization_blocks:
        raise ValueError("external train data and memorization cannot be combined")
    if bool(args.capacity_manifest) != bool(args.memorization_blocks):
        raise ValueError(
            "--capacity-manifest is required exactly for memorization probes"
        )
    if args.require_capacity_gate and not args.memorization_blocks:
        raise ValueError("capacity gate requires a memorization probe")
    if args.require_capacity_gate and args.expected_beneficial_actions < 1:
        raise ValueError(
            "capacity gate requires --expected-beneficial-actions"
        )
    if args.memorization_blocks and args.evidence_tier not in {
        "smoke",
        "capacity_probe",
    }:
        raise ValueError("same-subset evidence must be smoke/capacity_probe")
    if args.max_value_rmse < 0:
        raise ValueError("value RMSE threshold cannot be negative")
    for value in (
        args.min_beneficial_sign_recall,
        args.min_harmful_nonpositive_recall,
        args.min_oracle_gap_recovered,
        args.max_harmed_fraction,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("capacity rate thresholds must be in [0, 1]")


def _prompt_set_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(
            sorted({str(record["sample_id"]) for record in records})
        ).encode("utf-8")
    ).hexdigest()


def _class_masks(target_edits: Tensor) -> dict[str, Tensor]:
    return {
        "beneficial": target_edits > 0,
        "neutral": target_edits == 0,
        "harmful": target_edits < 0,
    }


def _empty_class_accumulator() -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "count": 0,
            "sse": 0.0,
            "target_sum": 0.0,
            "prediction_sum": 0.0,
            "predicted_positive": 0,
            "predicted_nonpositive": 0,
        }
        for name in VALUE_CLASSES
    }


def _accumulate_value_classes(
    accumulator: dict[str, dict[str, float | int]],
    target_edits: Tensor,
    predicted_edits: Tensor,
    squared_errors: Tensor,
) -> None:
    for name, mask in _class_masks(target_edits).items():
        count = int(mask.sum())
        if not count:
            continue
        entry = accumulator[name]
        entry["count"] = int(entry["count"]) + count
        entry["sse"] = float(entry["sse"]) + float(
            squared_errors[mask].sum()
        )
        entry["target_sum"] = float(entry["target_sum"]) + float(
            target_edits[mask].sum()
        )
        entry["prediction_sum"] = float(entry["prediction_sum"]) + float(
            predicted_edits[mask].sum()
        )
        entry["predicted_positive"] = int(
            entry["predicted_positive"]
        ) + int((predicted_edits[mask] > 0).sum())
        entry["predicted_nonpositive"] = int(
            entry["predicted_nonpositive"]
        ) + int((predicted_edits[mask] <= 0).sum())


def _finalize_value_classes(
    accumulator: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int | None]]:
    total_sse = sum(float(entry["sse"]) for entry in accumulator.values())
    report: dict[str, dict[str, float | int | None]] = {}
    for name in VALUE_CLASSES:
        entry = accumulator[name]
        count = int(entry["count"])
        report[name] = {
            **entry,
            "mean_mse": float(entry["sse"]) / count if count else None,
            "sse_fraction": (
                float(entry["sse"]) / total_sse if total_sse > 0 else None
            ),
            "mean_target": (
                float(entry["target_sum"]) / count if count else None
            ),
            "mean_prediction": (
                float(entry["prediction_sum"]) / count if count else None
            ),
            "predicted_positive_rate": (
                int(entry["predicted_positive"]) / count if count else None
            ),
            "predicted_nonpositive_rate": (
                int(entry["predicted_nonpositive"]) / count
                if count
                else None
            ),
        }
    return report


@torch.enable_grad()
def initial_projection_gradient_diagnostics(
    model: FirstMissValueSelector,
    loader: Any,
    target_embedding: Tensor,
    device: torch.device,
    *,
    candidate_k: int,
) -> dict[str, Any]:
    """Decompose epoch-zero output-projection gradients by utility sign."""

    model.eval()
    projection = model.backbone.residual_projection.weight
    gradient_sums = {
        name: torch.zeros_like(projection, dtype=torch.float32)
        for name in VALUE_CLASSES
    }
    counts = {name: 0 for name in VALUE_CLASSES}
    blocks = len(loader.dataset)
    if blocks < 1:
        raise RuntimeError("gradient diagnostic loader is empty")
    edit_actions: int | None = None

    for cpu_batch in loader:
        batch = direct.to_device(cpu_batch, device)
        candidate_embeddings = target_embedding[batch["candidate_ids"]]
        anchor_embeddings = target_embedding[batch["anchor_ids"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                candidate_embeddings,
                batch["candidate_logits"],
                batch["base_logsumexp"],
                anchor_embeddings,
            )
            losses = first_miss_value_loss(
                output,
                batch["gold_candidate_indices"],
                batch["gold_in_lattice"],
            )
        if not torch.equal(
            output.action_values, torch.zeros_like(output.action_values)
        ):
            raise RuntimeError("gradient decomposition requires epoch-zero values")
        targets = losses.target_values[:, 1:]
        if edit_actions is None:
            edit_actions = int(targets.shape[1])
        elif edit_actions != int(targets.shape[1]):
            raise RuntimeError("inconsistent action count in diagnostic loader")
        denominator = float(blocks * edit_actions)
        masks = _class_masks(targets)
        for index, name in enumerate(VALUE_CLASSES):
            mask = masks[name]
            counts[name] += int(mask.sum())
            component = losses.squared_errors[mask].sum() / denominator
            gradient = torch.autograd.grad(
                component,
                projection,
                retain_graph=index < len(VALUE_CLASSES) - 1,
            )[0]
            gradient_sums[name].add_(gradient.detach().float())

    total_gradient = sum(
        gradient_sums.values(), torch.zeros_like(projection, dtype=torch.float32)
    )
    total_norm = float(total_gradient.norm())
    components: dict[str, Any] = {}
    for name in VALUE_CLASSES:
        gradient = gradient_sums[name]
        norm = float(gradient.norm())
        cosine = (
            float(
                torch.nn.functional.cosine_similarity(
                    gradient.reshape(1, -1), total_gradient.reshape(1, -1)
                )
            )
            if norm > 0 and total_norm > 0
            else None
        )
        components[name] = {
            "actions": counts[name],
            "projection_gradient_norm": norm,
            "cosine_with_total_projection_gradient": cosine,
        }
    model.zero_grad(set_to_none=True)
    return {
        "normalization": "sum_class_squared_error/(blocks*edit_actions)",
        "blocks": blocks,
        "edit_actions_per_block": edit_actions,
        "total_projection_gradient_norm": total_norm,
        "components": components,
    }


@torch.inference_mode()
def evaluate(
    model: FirstMissValueSelector,
    loader: Any,
    target_embedding: Tensor,
    device: torch.device,
    *,
    candidate_k: int,
    include_examples: bool = False,
    require_base_identity: bool = False,
) -> dict[str, Any]:
    """Evaluate SAVS, DFlash, direct-native, and exact one-edit oracle paths."""

    model.eval()
    records: list[dict[str, Any]] = []
    objective_sum = 0.0
    value_classes = _empty_class_accumulator()
    total_squared_error = 0.0
    total_edit_actions = 0
    repair_opportunities = 0
    no_benefit_blocks = 0
    no_benefit_false_edits = 0
    selected_edits = 0
    selected_beneficial = 0
    selected_harmful = 0
    selected_neutral = 0
    selected_regret_sum = 0.0
    max_predicted_edit_values: list[float] = []

    for cpu_batch in loader:
        batch = direct.to_device(cpu_batch, device)
        candidate_embeddings = target_embedding[batch["candidate_ids"]]
        anchor_embeddings = target_embedding[batch["anchor_ids"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                candidate_embeddings,
                batch["candidate_logits"],
                batch["base_logsumexp"],
                anchor_embeddings,
            )
            losses = first_miss_value_loss(
                output,
                batch["gold_candidate_indices"],
                batch["gold_in_lattice"],
            )

        batch_size, length, candidates = batch["candidate_ids"].shape
        if candidates != candidate_k:
            raise RuntimeError("loader candidate K differs from evaluation K")
        if require_base_identity:
            expected_scores = (
                batch["candidate_logits"].float()
                - batch["base_logsumexp"].float().unsqueeze(-1)
            )
            if not torch.equal(output.direct_output.scores, expected_scores):
                maximum_error = float(
                    (output.direct_output.scores - expected_scores).abs().max()
                )
                raise RuntimeError(
                    "epoch-zero SAVS does not reproduce DFlash scores "
                    f"(max error {maximum_error})"
                )
            if not torch.equal(
                output.direct_output.residual_scores,
                torch.zeros_like(output.direct_output.residual_scores),
            ):
                raise RuntimeError("epoch-zero residual scores are not zero")
            if not torch.equal(
                output.action_values, torch.zeros_like(output.action_values)
            ):
                raise RuntimeError("epoch-zero SAVS action values are not zero")
            if not torch.equal(
                losses.predicted_actions,
                torch.zeros_like(losses.predicted_actions),
            ):
                raise RuntimeError("epoch-zero SAVS action is not KEEP")

        target_values = losses.target_values
        predicted_values = output.action_values.float()
        target_edits = target_values[:, 1:]
        predicted_edits = predicted_values[:, 1:]
        predicted_actions = losses.predicted_actions
        oracle_actions = decode_strict_positive_actions(target_values)
        selected_true_values = target_values.gather(
            1, predicted_actions[:, None]
        ).squeeze(1)
        best_true_values = target_values.max(dim=-1).values
        best_predicted_edit_values = predicted_edits.max(dim=-1).values
        regrets = best_true_values - selected_true_values
        if bool((regrets < -1e-7).any()):
            raise RuntimeError("selected-action regret became negative")

        base_paths = torch.zeros(
            batch_size, length, dtype=torch.long, device=device
        )
        savs_paths = decode_action_indices(
            predicted_actions, length=length, candidates=candidates
        )
        oracle_paths = decode_action_indices(
            oracle_actions, length=length, candidates=candidates
        )
        native_paths = output.direct_output.scores.argmax(dim=-1)
        paths = {
            "base": base_paths,
            "savs": savs_paths,
            "single_edit_oracle": oracle_paths,
            "direct_native": native_paths,
        }
        realized = {
            name: realized_prefix_lengths(
                path, batch["candidate_ids"], batch["gold_ids"]
            )
            for name, path in paths.items()
        }
        expected_delta = selected_true_values * float(length)
        actual_delta = realized["savs"] - realized["base"]
        torch.testing.assert_close(
            expected_delta,
            actual_delta.float(),
            rtol=0.0,
            atol=1e-6,
        )

        objective_sum += float(losses.per_block_mse.sum())
        total_squared_error += float(losses.squared_errors.sum())
        total_edit_actions += int(losses.squared_errors.numel())
        _accumulate_value_classes(
            value_classes,
            target_edits,
            predicted_edits,
            losses.squared_errors,
        )
        opportunities = best_true_values > 0
        no_benefit = ~opportunities
        edits = predicted_actions != 0
        beneficial = selected_true_values > 0
        harmful = selected_true_values < 0
        neutral_changed = edits & selected_true_values.eq(0)
        repair_opportunities += int(opportunities.sum())
        no_benefit_blocks += int(no_benefit.sum())
        no_benefit_false_edits += int((no_benefit & edits).sum())
        selected_edits += int(edits.sum())
        selected_beneficial += int(beneficial.sum())
        selected_harmful += int(harmful.sum())
        selected_neutral += int(neutral_changed.sum())
        selected_regret_sum += float(regrets.sum())
        max_predicted_edit_values.extend(
            best_predicted_edit_values.detach().cpu().tolist()
        )

        for item, (sample_id, domain) in enumerate(
            zip(batch["sample_ids"], batch["domains"], strict=True)
        ):
            predicted_action = int(predicted_actions[item])
            oracle_action = int(oracle_actions[item])
            record: dict[str, Any] = {
                "sample_id": str(sample_id),
                "domain": str(domain),
                "accepted_draft_tokens": {
                    name: int(values[item])
                    for name, values in realized.items()
                },
                "first_token_correct": {
                    name: bool(values[item] > 0)
                    for name, values in realized.items()
                },
                "predicted_action": predicted_action,
                "oracle_action": oracle_action,
                "selected_true_value": float(selected_true_values[item]),
                "best_true_value": float(best_true_values[item]),
                "selected_action_regret": float(regrets[item]),
                "max_predicted_edit_value": float(
                    best_predicted_edit_values[item]
                ),
            }
            if include_examples:
                record["candidate_path_indices"] = {
                    name: path[item].detach().cpu().tolist()
                    for name, path in paths.items()
                }
                record["predicted_action_values"] = (
                    predicted_values[item].detach().cpu().tolist()
                )
                record["target_action_values"] = (
                    target_values[item].detach().cpu().tolist()
                )
            records.append(record)

    if not records:
        raise RuntimeError("evaluation loader produced no records")
    class_report = _finalize_value_classes(value_classes)
    beneficial_count = int(class_report["beneficial"]["count"])
    harmful_count = int(class_report["harmful"]["count"])
    report: dict[str, Any] = {
        "loss": {
            "objective": objective_sum / len(records),
            "all_action_rmse": math.sqrt(
                total_squared_error / total_edit_actions
            ),
            "total_squared_error": total_squared_error,
            "edit_action_predictions": total_edit_actions,
        },
        "signed_value": {
            "classes": class_report,
            "beneficial_sign_recall": (
                int(class_report["beneficial"]["predicted_positive"])
                / beneficial_count
                if beneficial_count
                else None
            ),
            "harmful_nonpositive_recall": (
                int(class_report["harmful"]["predicted_nonpositive"])
                / harmful_count
                if harmful_count
                else None
            ),
        },
        "blocks": len(records),
        "prompts": len({record["sample_id"] for record in records}),
    }
    for method in ("base", "savs", "single_edit_oracle", "direct_native"):
        report[method] = direct._method_summary(records, method)
    report["by_domain"] = {}
    for domain in sorted({record["domain"] for record in records}):
        subset = [record for record in records if record["domain"] == domain]
        report["by_domain"][domain] = {
            method: direct._method_summary(subset, method)
            for method in (
                "base",
                "savs",
                "single_edit_oracle",
                "direct_native",
            )
        }

    base_eal = report["base"]["mean_accepted_draft_tokens_prompt_balanced"]
    savs_eal = report["savs"]["mean_accepted_draft_tokens_prompt_balanced"]
    oracle_eal = report["single_edit_oracle"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    oracle_denominator = oracle_eal - base_eal
    sorted_max_predictions = sorted(max_predicted_edit_values)
    report["decision"] = {
        "selected_edits": selected_edits,
        "edit_coverage": selected_edits / len(records),
        "beneficial_selected_actions": selected_beneficial,
        "harmful_selected_actions": selected_harmful,
        "neutral_selected_edits": selected_neutral,
        "harmed_fraction": selected_harmful / len(records),
        "edit_selective_precision": (
            selected_beneficial / selected_edits if selected_edits else None
        ),
        "repair_opportunities": repair_opportunities,
        "repair_recall": (
            selected_beneficial / repair_opportunities
            if repair_opportunities
            else None
        ),
        "no_benefit_blocks": no_benefit_blocks,
        "no_benefit_false_edits": no_benefit_false_edits,
        "no_benefit_false_edit_rate": (
            no_benefit_false_edits / no_benefit_blocks
            if no_benefit_blocks
            else None
        ),
        "mean_selected_action_regret_normalized": (
            selected_regret_sum / len(records)
        ),
        "mean_selected_action_regret_tokens": (
            selected_regret_sum * length / len(records)
        ),
        "single_edit_oracle_gap_recovered": (
            (savs_eal - base_eal) / oracle_denominator
            if oracle_denominator > 0
            else None
        ),
        "max_predicted_edit_value": {
            "mean": sum(sorted_max_predictions) / len(sorted_max_predictions),
            "minimum": sorted_max_predictions[0],
            "median": sorted_max_predictions[
                len(sorted_max_predictions) // 2
            ],
            "maximum": sorted_max_predictions[-1],
        },
    }
    if include_examples:
        report["examples"] = records
    return report


def compact_epoch_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    decision = evaluation["decision"]
    signed = evaluation["signed_value"]
    base_eal = evaluation["base"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    savs_eal = evaluation["savs"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    return {
        "objective": evaluation["loss"]["objective"],
        "all_action_rmse": evaluation["loss"]["all_action_rmse"],
        "beneficial_sign_recall": signed["beneficial_sign_recall"],
        "harmful_nonpositive_recall": signed[
            "harmful_nonpositive_recall"
        ],
        "base_eal": base_eal,
        "savs_eal": savs_eal,
        "raw_delta_vs_dflash": savs_eal - base_eal,
        "direct_native_eal": evaluation["direct_native"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "single_edit_oracle_eal": evaluation["single_edit_oracle"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "single_edit_oracle_gap_recovered": decision[
            "single_edit_oracle_gap_recovered"
        ],
        "harmed_fraction": decision["harmed_fraction"],
        "edit_coverage": decision["edit_coverage"],
        "edit_selective_precision": decision["edit_selective_precision"],
        "no_benefit_false_edit_rate": decision[
            "no_benefit_false_edit_rate"
        ],
        "mean_selected_action_regret_tokens": decision[
            "mean_selected_action_regret_tokens"
        ],
        "first_token_accuracy": evaluation["savs"]["first_token_accuracy"],
    }


def checkpoint_selection_key(
    evaluation: dict[str, Any], *, evidence_tier: str
) -> tuple[float, ...]:
    if evidence_tier in {"smoke", "capacity_probe"}:
        return (-float(evaluation["loss"]["objective"]),)
    return (
        float(
            evaluation["savs"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ]
        ),
        -float(evaluation["decision"]["harmed_fraction"]),
        -float(evaluation["loss"]["objective"]),
    )


def capacity_gate_report(
    evaluation: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    signed = evaluation["signed_value"]
    decision = evaluation["decision"]
    beneficial_actions = int(
        signed["classes"]["beneficial"]["count"]
    )
    values = {
        "all_action_rmse": evaluation["loss"]["all_action_rmse"],
        "beneficial_sign_recall": signed["beneficial_sign_recall"],
        "harmful_nonpositive_recall": signed[
            "harmful_nonpositive_recall"
        ],
        "single_edit_oracle_gap_recovered": decision[
            "single_edit_oracle_gap_recovered"
        ],
        "harmed_fraction": decision["harmed_fraction"],
        "beneficial_actions": beneficial_actions,
    }
    thresholds = {
        "all_action_rmse": args.max_value_rmse,
        "beneficial_sign_recall": args.min_beneficial_sign_recall,
        "harmful_nonpositive_recall": (
            args.min_harmful_nonpositive_recall
        ),
        "single_edit_oracle_gap_recovered": args.min_oracle_gap_recovered,
        "harmed_fraction": args.max_harmed_fraction,
        "beneficial_actions": args.expected_beneficial_actions,
    }
    checks = {
        "all_action_rmse": values["all_action_rmse"]
        <= thresholds["all_action_rmse"],
        "beneficial_sign_recall": (
            values["beneficial_sign_recall"] is not None
            and values["beneficial_sign_recall"]
            >= thresholds["beneficial_sign_recall"]
        ),
        "harmful_nonpositive_recall": (
            values["harmful_nonpositive_recall"] is not None
            and values["harmful_nonpositive_recall"]
            >= thresholds["harmful_nonpositive_recall"]
        ),
        "single_edit_oracle_gap_recovered": (
            values["single_edit_oracle_gap_recovered"] is not None
            and values["single_edit_oracle_gap_recovered"]
            >= thresholds["single_edit_oracle_gap_recovered"]
        ),
        "harmed_fraction": values["harmed_fraction"]
        <= thresholds["harmed_fraction"],
        "beneficial_actions": values["beneficial_actions"]
        == thresholds["beneficial_actions"],
    }
    return {
        "passed": all(checks.values()),
        "values": values,
        "thresholds": thresholds,
        "checks": checks,
        "interpretation": {
            "all_action_rmse": "engineering_fidelity_not_policy_safety",
            "behavior_gates": [
                "single_edit_oracle_gap_recovered",
                "harmed_fraction",
            ],
            "failure_scope": (
                "exact D64/H4/L1 residual-difference parameterization, "
                "unweighted MSE, optimizer/schedule, 512-subset composition, "
                "and minimum-MSE checkpoint rule"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("SAVS training requires CUDA")

    direct.seed_everything(args.seed)
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    config_snapshot = direct.serializable_config(args)
    source_paths = {
        "trainer": Path(__file__).resolve(),
        "savs_head": PROJECT / "src/sph/first_miss_value_selector.py",
        "fmas_action_semantics": PROJECT
        / "src/sph/first_miss_action_selector.py",
        "fmas_data_protocol": PROJECT
        / "scripts/train_first_miss_action_selector.py",
        "capacity_helper": PROJECT / "src/sph/first_miss_capacity.py",
        "direct_head": PROJECT / "src/sph/global_direct_selector.py",
        "direct_trainer_utilities": PROJECT
        / "scripts/train_global_direct_selector.py",
        "canonical_data": PROJECT / "src/sph/data.py",
    }
    source_hashes_start = {
        f"{name}_sha256": sha256_file(path)
        for name, path in source_paths.items()
    }
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, path in source_paths.items():
        shutil.copy2(path, snapshot / f"{name}_{path.name}")
    start = time.perf_counter()

    metadata_path = args.data / "metadata.json"
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
    train_prompts = len(
        {str(record["sample_id"]) for record in train_dataset.records}
    )
    train_prompt_sha256 = _prompt_set_sha256(train_dataset.records)
    if args.expected_train_blocks and len(train_dataset) != args.expected_train_blocks:
        raise RuntimeError("training block count differs from frozen expectation")
    if args.expected_train_prompts and train_prompts != args.expected_train_prompts:
        raise RuntimeError("training prompt count differs from frozen expectation")
    if (
        args.expected_train_prompt_sha256
        and train_prompt_sha256 != args.expected_train_prompt_sha256
    ):
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
    if block_length != int(metadata.get("draft_positions", block_length)):
        raise RuntimeError("block length differs from canonical metadata")
    all_records = train_dataset.records + (
        []
        if validation_dataset is train_dataset
        else validation_dataset.records
    )
    if any(
        int(record["gold_ids"].numel()) != block_length
        for record in all_records
    ):
        raise RuntimeError("records contain inconsistent block lengths")

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
    model = FirstMissValueSelector(backbone).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    named_parameters = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for _, parameter in named_parameters
                    if parameter.ndim >= 2
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    parameter
                    for _, parameter in named_parameters
                    if parameter.ndim < 2
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=args.learning_rate,
    )
    total_steps = args.epochs * len(train_loader)
    if args.expected_total_steps and total_steps != args.expected_total_steps:
        raise RuntimeError("optimizer-step budget differs from frozen expectation")
    scheduler, warmup_steps = direct.cosine_warmup_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    initial_validation = evaluate(
        model,
        validation_loader,
        target_embedding,
        device,
        candidate_k=args.candidate_k,
        require_base_identity=True,
    )
    initial_gradient_diagnostics = (
        initial_projection_gradient_diagnostics(
            model,
            validation_loader,
            target_embedding,
            device,
            candidate_k=args.candidate_k,
        )
        if args.memorization_blocks
        else None
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train": None,
            "validation": compact_epoch_metrics(initial_validation),
        }
    ]
    best_key = checkpoint_selection_key(
        initial_validation, evidence_tier=args.evidence_tier
    )
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

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        squared_error_sum = 0.0
        action_predictions = 0
        examples_seen = 0
        grad_norm_sum = 0.0
        for cpu_batch in train_loader:
            batch = direct.to_device(cpu_batch, device)
            batch_size = int(batch["hidden"].shape[0])
            optimizer.zero_grad(set_to_none=True)
            candidate_embeddings = target_embedding[batch["candidate_ids"]]
            anchor_embeddings = target_embedding[batch["anchor_ids"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    batch["hidden"],
                    candidate_embeddings,
                    batch["candidate_logits"],
                    batch["base_logsumexp"],
                    anchor_embeddings,
                )
                losses = first_miss_value_loss(
                    output,
                    batch["gold_candidate_indices"],
                    batch["gold_in_lattice"],
                )
            if not bool(torch.isfinite(losses.loss)):
                raise FloatingPointError("nonfinite SAVS training loss")
            losses.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            examples_seen += batch_size
            squared_error_sum += float(losses.squared_errors.sum())
            action_predictions += int(losses.squared_errors.numel())
            grad_norm_sum += float(grad_norm)

        validation = evaluate(
            model,
            validation_loader,
            target_embedding,
            device,
            candidate_k=args.candidate_k,
        )
        train_objective = squared_error_sum / action_predictions
        epoch_record = {
            "epoch": epoch,
            "train": {
                "objective": train_objective,
                "all_action_rmse": math.sqrt(train_objective),
                "mean_preclip_grad_norm": grad_norm_sum / len(train_loader),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "blocks": examples_seen,
                "edit_action_predictions": action_predictions,
                "steps": global_step,
            },
            "validation": compact_epoch_metrics(validation),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        key = checkpoint_selection_key(
            validation, evidence_tier=args.evidence_tier
        )
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

    checkpoint = torch.load(
        best_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    final_validation = evaluate(
        model,
        validation_loader,
        target_embedding,
        device,
        candidate_k=args.candidate_k,
        include_examples=True,
    )
    final_train = (
        None
        if args.skip_final_train_diagnostic
        else final_validation
        if validation_dataset is train_dataset
        else evaluate(
            model,
            direct.make_loader(
                train_dataset,
                candidate_k=args.candidate_k,
                batch_size=args.batch_size,
                shuffle=False,
            ),
            target_embedding,
            device,
            candidate_k=args.candidate_k,
        )
    )
    capacity_gate = (
        capacity_gate_report(final_validation, args)
        if args.memorization_blocks
        else None
    )
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
        "external_train_data": [
            {
                "path": str(path.resolve()),
                "metadata_sha256": sha256_file(path / "metadata.json"),
                "base_greedy_witness_status": (
                    dataset.base_greedy_witness_status
                ),
            }
            for path, dataset in zip(
                args.train_data, external_collections, strict=True
            )
        ],
        **source_hashes_start,
        **source_hashes_end,
        "capacity_manifest_sha256": (
            sha256_file(args.capacity_manifest)
            if args.capacity_manifest is not None
            else None
        ),
        "capacity_subset_sha256": (
            capacity_manifest["subset_sha256"]
            if capacity_manifest is not None
            else None
        ),
        "verified_target_embedding_files": verified_target_files,
        "verified_external_target_embedding_files": verified_external,
        "base_greedy_witness_status": collection.base_greedy_witness_status,
        "dflash_commit": direct.git_revision(PROJECT / "third_party/dflash"),
        "domino_commit": direct.git_revision(PROJECT / "third_party/Domino"),
        "dpace_commit": direct.git_revision(PROJECT / "third_party/D-PACE"),
    }
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config": config_snapshot,
        "primary_method": PRIMARY_METHOD,
        "evidence_tier": args.evidence_tier,
        "split_protocol": (
            "same_subset_capacity_probe"
            if args.memorization_blocks
            else "prompt_disjoint_external_train_development"
            if args.train_data
            else "prompt_disjoint_development"
        ),
        "train_blocks": len(train_dataset),
        "train_prompts": train_prompts,
        "train_prompt_set_sha256": train_prompt_sha256,
        "validation_blocks": len(validation_dataset),
        "validation_prompts": len(
            {
                str(record["sample_id"])
                for record in validation_dataset.records
            }
        ),
        "parameter_count": parameter_count,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "selected_epoch": int(checkpoint["epoch"]),
        "selection_key": checkpoint["selection_key"],
        "seconds": time.perf_counter() - start,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated() / 2**30
        ),
        "peak_cuda_reserved_gib": (
            torch.cuda.max_memory_reserved() / 2**30
        ),
        "capacity_manifest": capacity_manifest,
        "initial_projection_gradient_diagnostics": (
            initial_gradient_diagnostics
        ),
        "initial_validation": initial_validation,
        "final_train_diagnostic": final_train,
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
                "selected_epoch": int(checkpoint["epoch"]),
                "parameter_count": parameter_count,
                "validation": compact_epoch_metrics(final_validation),
                "capacity_gate": capacity_gate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if (
        args.require_capacity_gate
        and capacity_gate is not None
        and not capacity_gate["passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
