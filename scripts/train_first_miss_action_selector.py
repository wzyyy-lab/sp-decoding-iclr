#!/usr/bin/env python3
"""Train First-Miss Action Selection over an unchanged DFlash lattice head."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from sph.data import CanonicalBlockDataset
from sph.first_miss_action_selector import (
    FirstMissActionSelector,
    canonical_first_miss_actions,
    decode_action_indices,
    first_miss_action_loss,
    realized_prefix_lengths,
)
from sph.first_miss_capacity import (
    sha256_file,
    verify_capacity_manifest,
)
from sph.global_direct_selector import GlobalDirectCandidateSelector

try:
    import train_global_direct_selector as direct
except ModuleNotFoundError:  # Imported as ``scripts.*`` in CPU tests.
    from scripts import train_global_direct_selector as direct


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_METHOD = "first_miss_action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, action="append", default=[])
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["local", "causal", "global"], default="global")
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
    parser.add_argument("--min-action-accuracy", type=float, default=0.97)
    parser.add_argument("--min-repairable-action-recall", type=float, default=0.95)
    parser.add_argument("--min-oracle-gap-recovered", type=float, default=0.95)
    parser.add_argument("--max-harmed-fraction", type=float, default=0.01)
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
    if args.memorization_blocks and args.evidence_tier not in {
        "smoke",
        "capacity_probe",
    }:
        raise ValueError("same-subset evidence must be smoke/capacity_probe")
    for value in (
        args.min_action_accuracy,
        args.min_repairable_action_recall,
        args.min_oracle_gap_recovered,
        args.max_harmed_fraction,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("capacity thresholds must be in [0, 1]")


def _prompt_set_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(sorted({str(record["sample_id"]) for record in records})).encode(
            "utf-8"
        )
    ).hexdigest()


def _method_summary(
    records: list[dict[str, Any]], method: str
) -> dict[str, float]:
    return direct._method_summary(records, method)


def _decode_edit(action: int, *, candidates: int) -> tuple[int, int] | None:
    if action == 0:
        return None
    flattened = action - 1
    return flattened // (candidates - 1), flattened % (candidates - 1) + 1


@torch.inference_mode()
def evaluate(
    model: FirstMissActionSelector,
    loader: Any,
    target_embedding: Tensor,
    device: torch.device,
    *,
    candidate_k: int,
    include_examples: bool = False,
    require_base_identity: bool = False,
) -> dict[str, Any]:
    """Evaluate native direct, one-edit FMAS, and one-edit oracle paths."""

    model.eval()
    records: list[dict[str, Any]] = []
    objective_sum = 0.0
    action_correct = 0
    repairable_correct = 0
    repairable_total = 0
    gain_correct = 0.0
    gain_total = 0.0
    target_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    target_kind_counts: Counter[str] = Counter()
    prediction_kind_counts: Counter[str] = Counter()
    target_position_counts: Counter[str] = Counter()
    prediction_position_counts: Counter[str] = Counter()
    target_rank_counts: Counter[str] = Counter()
    prediction_rank_counts: Counter[str] = Counter()
    winner_margin_sum = 0.0
    winner_margin_count = 0
    selected_edit_margin_sum = 0.0
    selected_edit_margin_count = 0
    keep_margin_sum = 0.0
    keep_margin_count = 0

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
            losses = first_miss_action_loss(
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
                    "epoch-zero FMAS does not reproduce DFlash scores "
                    f"(max error {maximum_error})"
                )
            if not torch.equal(
                losses.predicted_actions,
                torch.zeros_like(losses.predicted_actions),
            ):
                raise RuntimeError("epoch-zero FMAS action is not KEEP_BASE")
            if bool((output.action_logits[:, 1:] > 0).any()):
                raise RuntimeError("epoch-zero edit logit exceeds KEEP_BASE")

        target_actions = losses.target_actions
        predicted_actions = losses.predicted_actions
        base_paths = torch.zeros(
            batch_size, length, dtype=torch.long, device=device
        )
        fmas_paths = decode_action_indices(
            predicted_actions, length=length, candidates=candidates
        )
        oracle_paths = decode_action_indices(
            target_actions, length=length, candidates=candidates
        )
        native_paths = output.direct_output.scores.argmax(dim=-1)
        paths = {
            "base": base_paths,
            "fmas": fmas_paths,
            "single_edit_oracle": oracle_paths,
            "direct_native": native_paths,
        }
        realized = {
            name: realized_prefix_lengths(
                path, batch["candidate_ids"], batch["gold_ids"]
            )
            for name, path in paths.items()
        }
        first_correct = {
            name: values > 0 for name, values in realized.items()
        }

        objective_sum += float(losses.loss) * batch_size
        action_correct += int(predicted_actions.eq(target_actions).sum())
        top_two = output.action_logits.float().topk(k=2, dim=-1).values
        margins = top_two[:, 0] - top_two[:, 1]
        best_edit_logits = output.action_logits[:, 1:].float().max(dim=-1).values
        winner_margin_sum += float(margins.sum())
        winner_margin_count += batch_size

        for item, (sample_id, domain) in enumerate(
            zip(batch["sample_ids"], batch["domains"], strict=True)
        ):
            target_action = int(target_actions[item])
            predicted_action = int(predicted_actions[item])
            base_prefix = int(realized["base"][item])
            oracle_prefix = int(realized["single_edit_oracle"][item])
            gain = oracle_prefix - base_prefix
            target_edit = _decode_edit(target_action, candidates=candidates)
            prediction_edit = _decode_edit(
                predicted_action, candidates=candidates
            )
            if target_edit is not None:
                repairable_total += 1
                gain_total += gain
                if predicted_action == target_action:
                    repairable_correct += 1
                    gain_correct += gain
                target_kind = "edit"
                target_position_counts[str(target_edit[0])] += 1
                target_rank_counts[str(target_edit[1])] += 1
            elif base_prefix == length:
                target_kind = "keep_full_correct"
            else:
                target_kind = "keep_out_of_k"
            if prediction_edit is None:
                prediction_kind = "keep"
                keep_margin = float(-best_edit_logits[item])
                selected_edit_margin = None
                keep_margin_sum += keep_margin
                keep_margin_count += 1
            else:
                prediction_kind = "edit"
                selected_edit_margin = float(
                    output.action_logits[item, predicted_action]
                )
                keep_margin = None
                selected_edit_margin_sum += selected_edit_margin
                selected_edit_margin_count += 1
                prediction_position_counts[str(prediction_edit[0])] += 1
                prediction_rank_counts[str(prediction_edit[1])] += 1
            target_counts[str(target_action)] += 1
            prediction_counts[str(predicted_action)] += 1
            target_kind_counts[target_kind] += 1
            prediction_kind_counts[prediction_kind] += 1

            record: dict[str, Any] = {
                "sample_id": str(sample_id),
                "domain": str(domain),
                "accepted_draft_tokens": {
                    name: int(values[item])
                    for name, values in realized.items()
                },
                "first_token_correct": {
                    name: bool(values[item])
                    for name, values in first_correct.items()
                },
                "target_action": target_action,
                "predicted_action": predicted_action,
                "target_kind": target_kind,
                "target_gain": gain,
                "action_correct": predicted_action == target_action,
                "winner_margin": float(margins[item]),
                "selected_edit_margin_over_keep": selected_edit_margin,
                "keep_margin_over_best_edit": keep_margin,
            }
            if include_examples:
                record["target_edit"] = (
                    None
                    if target_edit is None
                    else {"position": target_edit[0], "rank": target_edit[1]}
                )
                record["predicted_edit"] = (
                    None
                    if prediction_edit is None
                    else {
                        "position": prediction_edit[0],
                        "rank": prediction_edit[1],
                    }
                )
                record["candidate_path_indices"] = {
                    name: path[item].detach().cpu().tolist()
                    for name, path in paths.items()
                }
                record["action_logits"] = (
                    output.action_logits[item].detach().cpu().tolist()
                )
            records.append(record)

    if not records:
        raise RuntimeError("evaluation loader produced no records")
    report: dict[str, Any] = {
        "loss": {"objective": objective_sum / len(records)},
        "action_classification": {
            "accuracy": action_correct / len(records),
            "correct": action_correct,
            "blocks": len(records),
            "repairable_action_recall": (
                repairable_correct / repairable_total
                if repairable_total
                else None
            ),
            "repairable_correct": repairable_correct,
            "repairable_blocks": repairable_total,
            "gain_weighted_repair_recall": (
                gain_correct / gain_total if gain_total > 0 else None
            ),
            "gain_correct": gain_correct,
            "gain_total": gain_total,
            "target_action_counts": dict(
                sorted(target_counts.items(), key=lambda item: int(item[0]))
            ),
            "predicted_action_counts": dict(
                sorted(prediction_counts.items(), key=lambda item: int(item[0]))
            ),
            "target_kind_counts": dict(sorted(target_kind_counts.items())),
            "prediction_kind_counts": dict(
                sorted(prediction_kind_counts.items())
            ),
            "target_edit_position_counts": dict(
                sorted(target_position_counts.items(), key=lambda item: int(item[0]))
            ),
            "predicted_edit_position_counts": dict(
                sorted(
                    prediction_position_counts.items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "target_edit_rank_counts": dict(
                sorted(target_rank_counts.items(), key=lambda item: int(item[0]))
            ),
            "predicted_edit_rank_counts": dict(
                sorted(
                    prediction_rank_counts.items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "mean_winner_margin": winner_margin_sum / winner_margin_count,
            "mean_selected_edit_margin_over_keep": (
                selected_edit_margin_sum / selected_edit_margin_count
                if selected_edit_margin_count
                else None
            ),
            "mean_keep_margin_over_best_edit": (
                keep_margin_sum / keep_margin_count
                if keep_margin_count
                else None
            ),
        },
        "blocks": len(records),
        "prompts": len({record["sample_id"] for record in records}),
    }
    for method in (
        "base",
        "fmas",
        "single_edit_oracle",
        "direct_native",
    ):
        report[method] = _method_summary(records, method)
    report["by_domain"] = {}
    for domain in sorted({record["domain"] for record in records}):
        subset = [record for record in records if record["domain"] == domain]
        report["by_domain"][domain] = {
            method: _method_summary(subset, method)
            for method in (
                "base",
                "fmas",
                "single_edit_oracle",
                "direct_native",
            )
        }

    improved = sum(
        record["accepted_draft_tokens"]["fmas"]
        > record["accepted_draft_tokens"]["base"]
        for record in records
    )
    harmed = sum(
        record["accepted_draft_tokens"]["fmas"]
        < record["accepted_draft_tokens"]["base"]
        for record in records
    )
    changed = sum(record["predicted_action"] != 0 for record in records)
    neutral = changed - improved - harmed
    base_eal = report["base"]["mean_accepted_draft_tokens_prompt_balanced"]
    fmas_eal = report["fmas"]["mean_accepted_draft_tokens_prompt_balanced"]
    oracle_eal = report["single_edit_oracle"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    denominator = oracle_eal - base_eal
    report["fmas_diagnostics"] = {
        "changed_blocks": changed,
        "improved_blocks": improved,
        "neutral_changed_blocks": neutral,
        "harmed_blocks": harmed,
        "harmed_fraction": harmed / len(records),
        "first_miss_repairs": improved,
        "first_miss_repair_opportunities": repairable_total,
        "first_miss_repair_rate_given_k": (
            improved / repairable_total if repairable_total else None
        ),
        "single_edit_oracle_gap_recovered": (
            (fmas_eal - base_eal) / denominator
            if denominator > 0
            else None
        ),
    }
    if include_examples:
        report["examples"] = records
    return report


def compact_epoch_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    action = evaluation["action_classification"]
    diagnostics = evaluation["fmas_diagnostics"]
    base_eal = evaluation["base"]["mean_accepted_draft_tokens_prompt_balanced"]
    fmas_eal = evaluation["fmas"]["mean_accepted_draft_tokens_prompt_balanced"]
    return {
        "objective": evaluation["loss"]["objective"],
        "action_accuracy": action["accuracy"],
        "repairable_action_recall": action["repairable_action_recall"],
        "gain_weighted_repair_recall": action["gain_weighted_repair_recall"],
        "base_eal": base_eal,
        "fmas_eal": fmas_eal,
        "raw_delta_vs_dflash": fmas_eal - base_eal,
        "direct_native_eal": evaluation["direct_native"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "single_edit_oracle_eal": evaluation["single_edit_oracle"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "single_edit_oracle_gap_recovered": diagnostics[
            "single_edit_oracle_gap_recovered"
        ],
        "harmed_blocks": diagnostics["harmed_blocks"],
        "harmed_fraction": diagnostics["harmed_fraction"],
        "first_token_accuracy": evaluation["fmas"]["first_token_accuracy"],
    }


def checkpoint_selection_key(
    evaluation: dict[str, Any], *, evidence_tier: str
) -> tuple[float, ...]:
    action = evaluation["action_classification"]
    if evidence_tier in {"smoke", "capacity_probe"}:
        return (
            -float(evaluation["loss"]["objective"]),
            float(action["accuracy"]),
        )
    return (
        float(
            evaluation["fmas"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ]
        ),
        -float(evaluation["fmas_diagnostics"]["harmed_fraction"]),
        float(action["accuracy"]),
    )


def capacity_gate_report(
    evaluation: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    action = evaluation["action_classification"]
    diagnostics = evaluation["fmas_diagnostics"]
    values = {
        "action_accuracy": action["accuracy"],
        "repairable_action_recall": action["repairable_action_recall"],
        "single_edit_oracle_gap_recovered": diagnostics[
            "single_edit_oracle_gap_recovered"
        ],
        "harmed_fraction": diagnostics["harmed_fraction"],
    }
    thresholds = {
        "action_accuracy": args.min_action_accuracy,
        "repairable_action_recall": args.min_repairable_action_recall,
        "single_edit_oracle_gap_recovered": args.min_oracle_gap_recovered,
        "harmed_fraction": args.max_harmed_fraction,
    }
    checks = {
        "action_accuracy": values["action_accuracy"]
        >= thresholds["action_accuracy"],
        "repairable_action_recall": (
            values["repairable_action_recall"] is not None
            and values["repairable_action_recall"]
            >= thresholds["repairable_action_recall"]
        ),
        "single_edit_oracle_gap_recovered": (
            values["single_edit_oracle_gap_recovered"] is not None
            and values["single_edit_oracle_gap_recovered"]
            >= thresholds["single_edit_oracle_gap_recovered"]
        ),
        "harmed_fraction": values["harmed_fraction"]
        <= thresholds["harmed_fraction"],
    }
    return {
        "passed": all(checks.values()),
        "values": values,
        "thresholds": thresholds,
        "checks": checks,
    }


def _load_datasets(
    args: argparse.Namespace,
) -> tuple[
    direct.RecordDataset,
    direct.RecordDataset,
    CanonicalBlockDataset,
    list[CanonicalBlockDataset],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    collection = CanonicalBlockDataset(args.data)
    metadata = collection.metadata
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in collection.records:
        by_split[str(record["split"])].append(record)
    if args.train_data and set(by_split) != {args.validation_split}:
        raise RuntimeError(
            "external-training development requires --data to contain only "
            f"the physically isolated {args.validation_split!r} split; found "
            f"{sorted(by_split)}"
        )
    capacity_manifest: dict[str, Any] | None = None
    if args.memorization_blocks:
        if args.train_split not in by_split:
            raise ValueError(f"missing train split {args.train_split!r}")
        selected = direct.deterministic_capacity_subset(
            by_split[args.train_split],
            count=args.memorization_blocks,
            seed=args.seed,
            opportunity_fraction=args.memorization_opportunity_fraction,
            candidate_k=args.candidate_k,
        )
        capacity_manifest = json.loads(args.capacity_manifest.read_text())
        verify_capacity_manifest(
            capacity_manifest,
            selected,
            source_metadata_path=args.data / "metadata.json",
            candidate_k=args.candidate_k,
            seed=args.seed,
            opportunity_fraction=args.memorization_opportunity_fraction,
        )
        train_dataset = direct.RecordDataset(selected, metadata)
        validation_dataset = train_dataset
        external_collections: list[CanonicalBlockDataset] = []
        verified_external: list[dict[str, Any]] = []
    else:
        if args.validation_split not in by_split:
            raise ValueError(f"missing validation split {args.validation_split!r}")
        if not args.train_data and args.train_split not in by_split:
            raise ValueError(f"missing train split {args.train_split!r}")
        external_collections = [
            CanonicalBlockDataset(path) for path in args.train_data
        ]
        verified_external = []
        seen_prompts: set[str] = set()
        for path, dataset in zip(
            args.train_data, external_collections, strict=True
        ):
            direct.assert_canonical_collection_compatible(
                metadata, dataset.metadata, path=path
            )
            if args.candidate_k > int(dataset.metadata.get("top_k", 0)):
                raise RuntimeError(
                    f"candidate K exceeds external collection top-K: {path}"
                )
            prompts = {
                str(record["sample_id"])
                for record in dataset.records
                if str(record["split"]) == args.train_split
            }
            overlap = seen_prompts & prompts
            if overlap:
                raise RuntimeError(
                    f"external train prompt overlap at {path}: {sorted(overlap)[:3]}"
                )
            seen_prompts.update(prompts)
            verified_external.append(
                {
                    "data": str(path.resolve()),
                    "target_fingerprint_matches_base_collection": True,
                    "draft_fingerprint_matches_base_collection": True,
                }
            )
        train_records = (
            [
                record
                for dataset in external_collections
                for record in dataset.records
                if str(record["split"]) == args.train_split
            ]
            if external_collections
            else by_split[args.train_split]
        )
        if not train_records:
            raise ValueError(
                f"training collections contain no {args.train_split!r} records"
            )
        if args.max_train_prompts:
            train_records = direct.deterministic_prompt_subset(
                train_records,
                max_prompts=args.max_train_prompts,
                seed=args.train_subset_seed,
            )
        train_dataset = direct.RecordDataset(train_records, metadata)
        validation_dataset = direct.RecordDataset(
            by_split[args.validation_split], metadata
        )
        direct.assert_prompt_disjoint_splits(
            {"train": train_dataset, "validation": validation_dataset}
        )
    return (
        train_dataset,
        validation_dataset,
        collection,
        external_collections,
        verified_external,
        capacity_manifest,
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("FMAS training requires CUDA")

    direct.seed_everything(args.seed)
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    config_snapshot = direct.serializable_config(args)
    source_paths = {
        "trainer": Path(__file__).resolve(),
        "fmas_head": PROJECT / "src/sph/first_miss_action_selector.py",
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
    ) = _load_datasets(args)
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
        [] if validation_dataset is train_dataset else validation_dataset.records
    )
    if any(int(record["gold_ids"].numel()) != block_length for record in all_records):
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
    model = FirstMissActionSelector(backbone).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
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
        objective_sum = 0.0
        action_correct = 0
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
                losses = first_miss_action_loss(
                    output,
                    batch["gold_candidate_indices"],
                    batch["gold_in_lattice"],
                )
            if not bool(torch.isfinite(losses.loss)):
                raise FloatingPointError("nonfinite FMAS training loss")
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
            objective_sum += float(losses.loss.detach()) * batch_size
            action_correct += int(
                losses.predicted_actions.eq(losses.target_actions).sum()
            )
            grad_norm_sum += float(grad_norm)

        validation = evaluate(
            model,
            validation_loader,
            target_embedding,
            device,
            candidate_k=args.candidate_k,
        )
        epoch_record = {
            "epoch": epoch,
            "train": {
                "objective": objective_sum / examples_seen,
                "action_accuracy": action_correct / examples_seen,
                "mean_preclip_grad_norm": grad_norm_sum / len(train_loader),
                "learning_rate": optimizer.param_groups[0]["lr"],
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

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
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
                "base_greedy_witness_status": dataset.base_greedy_witness_status,
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
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "capacity_manifest": capacity_manifest,
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
