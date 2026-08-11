#!/usr/bin/env python3
"""Physically isolated full-data development trainer for CAMRS."""

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

from sph.first_miss_max_regret_selector import (
    FirstMissMaxRegretSelector,
    first_miss_max_regret_loss,
)
from sph.global_direct_selector import GlobalDirectCandidateSelector

try:
    import evaluate_direct_one_edit as direct_eval
    import train_first_miss_action_selector as fmas
    import train_first_miss_max_regret_selector as capacity
    import train_global_direct_selector as direct
except ModuleNotFoundError:  # Imported as ``scripts.*`` by CPU tests.
    from scripts import evaluate_direct_one_edit as direct_eval
    from scripts import train_first_miss_action_selector as fmas
    from scripts import train_first_miss_max_regret_selector as capacity
    from scripts import train_global_direct_selector as direct


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_METHOD = "first_miss_cost_augmented_max_regret"
EXPECTED_DATA = (
    PROJECT / "artifacts/canonical/qwen3_4b_phase3_validation_select_only_20260805"
)
EXPECTED_TRAIN_ROOT = (
    PROJECT / "artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770"
)
EXPECTED_TRAIN_DATA = [
    str((EXPECTED_TRAIN_ROOT / f"part-{index:03d}").resolve())
    for index in range(8)
]
EXPECTED_TRAIN_METADATA_SHA256 = [
    "d64492233e6112daeee5f54c88cca16dffb3a2b4f98a54ad6c5f11b877935856",
    "a55331a31c9dc6efa4a376896ec5d6f7de828f104e3ffcb10da68546425240de",
    "4d78caf63e0a70c012d7382e20abacd8abd2029cc27de9885de88043628fd7e5",
    "0e447e1a3a40635525128ea36569b3e4a7424d1056aa683a81f0943b32f74d50",
    "d7597e41b7b16f533522f0f9f0d741f8de9ad1e3c8bcc3bf8557a24d8d31b42c",
    "03099d833d1ca9eaa32dcc1f468038ab0d58d8952c4220bcd0ed5e897ba5ec7a",
    "b9dcf53d64b38b1658831e09130cfb9a1a265685ede5a45e562b42e1be016c02",
    "48f015efdb40bc8ed32a41af0e9682d2090dd3dd9bb90e6e0906dfe8c6803585",
]
EXPECTED_DATA_METADATA_SHA256 = (
    "b63be7bbfd56651aadbee57a819bfe0afb39395b1601b5ea4fc1564cc9f933d7"
)
EXPECTED_DATA_MANIFEST_SHA256 = (
    "1496caa3d71ce64de9cd3fc2c29e40be60e9b636a988c9b400a0712e3ee5e811"
)
EXPECTED_TRAIN_BLOCKS = 793_989
EXPECTED_TRAIN_PROMPTS = 99_356
EXPECTED_TRAIN_PROMPT_SHA256 = (
    "45471a62f93a488f3f7653c096bebcddb0ddae3773f6c99744bd070e348a9405"
)
EXPECTED_VALIDATION_BLOCKS = 1_175
EXPECTED_VALIDATION_PROMPTS = 147
EXPECTED_VALIDATION_PROMPT_SHA256 = (
    "278c27e266e50c6b81b94a88bd8dbf5dc2645563add738db7536f2489a01edaa"
)
EXPECTED_TOTAL_STEPS = 37_221
DFLASH_DELTA_THRESHOLD = 0.28499
DIRECT_DELTA_THRESHOLD = 0.05
MAX_HARMED_FRACTION = 0.05
MAX_FIRST_TOKEN_SHORTFALL_BLOCKS = 1
SUMMARY_FIELDS = (
    "mean_accepted_draft_tokens",
    "mean_verification_advance",
    "mean_accepted_draft_tokens_prompt_balanced",
    "mean_verification_advance_prompt_balanced",
    "first_token_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, action="append", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--direct-control", type=Path, required=True)
    parser.add_argument("--expected-direct-metrics-sha256", required=True)
    parser.add_argument("--expected-direct-checkpoint-sha256", required=True)
    parser.add_argument("--expected-direct-control-sha256", required=True)
    parser.add_argument("--scope", choices=["global"], default="global")
    parser.add_argument("--mixer", choices=["axial"], default="axial")
    parser.add_argument("--node-encoder", choices=["additive"], default="additive")
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
    parser.add_argument("--require-development-gate", action="store_true")
    parser.set_defaults(
        max_train_prompts=0,
        memorization_blocks=0,
        memorization_opportunity_fraction=0.5,
        capacity_manifest=None,
        require_capacity_gate=False,
        evidence_tier="development",
        expected_train_blocks=EXPECTED_TRAIN_BLOCKS,
        expected_train_prompts=EXPECTED_TRAIN_PROMPTS,
        expected_total_steps=EXPECTED_TOTAL_STEPS,
        expected_train_prompt_sha256=EXPECTED_TRAIN_PROMPT_SHA256,
        skip_final_train_diagnostic=True,
    )
    return parser.parse_args()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_args(args: argparse.Namespace) -> None:
    exact = {
        "scope": "global",
        "mixer": "axial",
        "node_encoder": "additive",
        "candidate_k": 16,
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_size": 64,
        "epochs": 3,
        "learning_rate": 6e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.04,
        "gradient_clip": 1.0,
        "seed": 0,
        "train_subset_seed": 20260730,
        "train_split": "train",
        "validation_split": "validation_select",
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            raise ValueError(f"--{name.replace('_', '-')} must equal {expected}")
    if args.data.resolve() != EXPECTED_DATA.resolve():
        raise ValueError("--data is not the frozen physically isolated collection")
    observed_train = [str(path.resolve()) for path in args.train_data]
    if observed_train != EXPECTED_TRAIN_DATA:
        raise ValueError("--train-data must list the eight frozen parts in order")
    if not args.require_development_gate:
        raise ValueError("CAMRS development requires fail-closed gating")
    for name in (
        "expected_direct_metrics_sha256",
        "expected_direct_checkpoint_sha256",
        "expected_direct_control_sha256",
    ):
        if not _is_sha256(str(getattr(args, name))):
            raise ValueError(f"--{name.replace('_', '-')} must be a lowercase SHA256")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")


def _prompt_set_sha256(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(sorted({str(record["sample_id"]) for record in records}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_selection_key(evaluation: dict[str, Any]) -> tuple[float, ...]:
    """Frozen development order: EAL, lower harm, lower hinge."""

    return (
        float(evaluation["camrs"]["mean_accepted_draft_tokens_prompt_balanced"]),
        -float(evaluation["decision"]["harmed_fraction"]),
        -float(evaluation["loss"]["mean_block_hinge"]),
    )


def _assert_summary_equal(
    left: dict[str, Any], right: dict[str, Any], *, label: str
) -> None:
    for field in SUMMARY_FIELDS:
        if field not in left or field not in right:
            raise RuntimeError(f"missing {label}.{field}")
        if not math.isclose(
            float(left[field]), float(right[field]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"{label}.{field} differs across artifacts")


def load_and_validate_direct_control(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the external Direct-native/one-edit precondition before training."""

    metrics_path = args.direct_run / "metrics.json"
    checkpoint_path = args.direct_run / "best.pt"
    observed = {
        "metrics": direct.sha256_file(metrics_path),
        "checkpoint": direct.sha256_file(checkpoint_path),
        "control": direct.sha256_file(args.direct_control),
    }
    expected = {
        "metrics": args.expected_direct_metrics_sha256,
        "checkpoint": args.expected_direct_checkpoint_sha256,
        "control": args.expected_direct_control_sha256,
    }
    if observed != expected:
        raise RuntimeError(f"Direct artifact hash mismatch: {observed} != {expected}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    direct_eval.validate_direct_checkpoint_contract(
        metrics, checkpoint, direct_run=args.direct_run
    )
    control = json.loads(args.direct_control.read_text(encoding="utf-8"))
    required = {
        "evidence_tier": "development_control",
        "decoder": "keep_or_global_max_margin_single_edit",
        "direct_run": str(args.direct_run.resolve()),
        "direct_metrics_sha256": expected["metrics"],
        "direct_checkpoint_sha256": expected["checkpoint"],
        "data": str(args.data.resolve()),
        "data_metadata_sha256": EXPECTED_DATA_METADATA_SHA256,
        "validation_split": "validation_select",
        "validation_blocks": EXPECTED_VALIDATION_BLOCKS,
        "validation_prompts": EXPECTED_VALIDATION_PROMPTS,
        "direct_native_identity_check": "exact_method_summary_match",
    }
    for name, value in required.items():
        if control.get(name) != value:
            raise RuntimeError(f"Direct control differs at {name}")
    examples = control.get("examples")
    if not isinstance(examples, list) or len(examples) != EXPECTED_VALIDATION_BLOCKS:
        raise RuntimeError("Direct control examples have wrong cardinality")
    if _prompt_set_sha256(examples) != EXPECTED_VALIDATION_PROMPT_SHA256:
        raise RuntimeError("Direct control prompt set differs from frozen validation")
    for method in ("base", "direct_native", "direct_one_edit"):
        summary = control.get(method)
        if not isinstance(summary, dict):
            raise RuntimeError(f"Direct control lacks {method} summary")
        if not all(math.isfinite(float(summary[field])) for field in SUMMARY_FIELDS):
            raise RuntimeError(f"Direct control {method} summary is nonfinite")
    for report_name, example_name in (
        ("base", "base"),
        ("direct_native", "direct_native"),
        ("direct_one_edit", "fmas"),
        ("single_edit_oracle", "single_edit_oracle"),
    ):
        reconstructed = direct._method_summary(examples, example_name)
        _assert_summary_equal(
            reconstructed, control[report_name], label=f"control.{report_name}"
        )
    _assert_summary_equal(
        control["base"], metrics["final_validation"]["base"], label="direct.base"
    )
    _assert_summary_equal(
        control["direct_native"],
        metrics["final_validation"]["direct"],
        label="direct.native",
    )
    if int(control.get("direct_selected_epoch", -1)) != int(
        metrics.get("selected_epoch", -2)
    ):
        raise RuntimeError("Direct control selected epoch differs from metrics")
    provenance = control.get("provenance", {})
    for name, value in provenance.items():
        if name.endswith("_sha256_at_end"):
            start_name = name[: -len("_at_end")]
            if provenance.get(start_name) != value:
                raise RuntimeError(f"Direct-control source changed: {start_name}")
    control_source_paths = {
        "evaluator": PROJECT / "scripts/evaluate_direct_one_edit.py",
        "fmas_evaluator": PROJECT / "scripts/train_first_miss_action_selector.py",
        "direct_trainer": PROJECT / "scripts/train_global_direct_selector.py",
        "fmas_head": PROJECT / "src/sph/first_miss_action_selector.py",
        "direct_head": PROJECT / "src/sph/global_direct_selector.py",
        "canonical_data": PROJECT / "src/sph/data.py",
    }
    for name, path in control_source_paths.items():
        current = direct.sha256_file(path)
        if provenance.get(f"{name}_sha256") != current:
            raise RuntimeError(f"Direct-control source differs from current {name}")
        if provenance.get(f"{name}_sha256_at_end") != current:
            raise RuntimeError(f"Direct-control end source differs from current {name}")
    return control, metrics


def development_gate_report(
    evaluation: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen, external-control-aware development gate."""

    examples = evaluation.get("examples")
    control_examples = control.get("examples")
    if not isinstance(examples, list) or not isinstance(control_examples, list):
        raise RuntimeError("development gate requires example-level artifacts")
    if len(examples) != len(control_examples):
        raise RuntimeError("CAMRS and Direct control block counts differ")
    if len(examples) != EXPECTED_VALIDATION_BLOCKS:
        raise RuntimeError("CAMRS development examples have wrong cardinality")
    if _prompt_set_sha256(examples) != _prompt_set_sha256(control_examples):
        raise RuntimeError("CAMRS and Direct control prompt sets differ")
    for report_name in ("base", "camrs"):
        reconstructed = direct._method_summary(examples, report_name)
        _assert_summary_equal(
            reconstructed,
            evaluation[report_name],
            label=f"camrs.{report_name}",
        )
    camrs_first_correct = 0
    direct_first_correct = 0
    harmed_blocks = 0
    for camrs_example, direct_example in zip(examples, control_examples, strict=True):
        for key in ("sample_id", "domain"):
            if camrs_example[key] != direct_example[key]:
                raise RuntimeError(f"CAMRS/Direct example order differs at {key}")
        for method in ("base", "single_edit_oracle"):
            if (
                camrs_example["accepted_draft_tokens"][method]
                != direct_example["accepted_draft_tokens"][method]
            ):
                raise RuntimeError(
                    f"CAMRS/Direct {method} realization differs"
                )
            if (
                camrs_example["first_token_correct"][method]
                != direct_example["first_token_correct"][method]
            ):
                raise RuntimeError(
                    f"CAMRS/Direct {method} first-token realization differs"
                )
            if (
                camrs_example["candidate_path_indices"][method]
                != direct_example["candidate_path_indices"][method]
            ):
                raise RuntimeError(f"CAMRS/Direct {method} path differs")
        if camrs_example["oracle_action"] != direct_example["target_action"]:
            raise RuntimeError("CAMRS/Direct single-edit oracle action differs")
        camrs_accepted = camrs_example["accepted_draft_tokens"]["camrs"]
        base_accepted = camrs_example["accepted_draft_tokens"]["base"]
        harmed_blocks += int(camrs_accepted < base_accepted)
        camrs_first_correct += int(camrs_example["first_token_correct"]["camrs"])
        direct_first_correct += int(
            direct_example["first_token_correct"]["direct_native"]
        )
    _assert_summary_equal(evaluation["base"], control["base"], label="base")
    reconstructed_harmed_fraction = harmed_blocks / len(examples)
    reported_harmed_fraction = float(evaluation["decision"]["harmed_fraction"])
    if math.isfinite(reported_harmed_fraction) and not math.isclose(
        reconstructed_harmed_fraction,
        reported_harmed_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("CAMRS harmed fraction differs from examples")

    camrs_eal = float(
        evaluation["camrs"]["mean_accepted_draft_tokens_prompt_balanced"]
    )
    base_eal = float(
        evaluation["base"]["mean_accepted_draft_tokens_prompt_balanced"]
    )
    direct_native_eal = float(
        control["direct_native"]["mean_accepted_draft_tokens_prompt_balanced"]
    )
    direct_one_edit_eal = float(
        control["direct_one_edit"]["mean_accepted_draft_tokens_prompt_balanced"]
    )
    values = {
        "camrs_eal": camrs_eal,
        "dflash_eal": base_eal,
        "direct_native_eal": direct_native_eal,
        "direct_one_edit_eal": direct_one_edit_eal,
        "delta_vs_dflash": camrs_eal - base_eal,
        "delta_vs_direct_native": camrs_eal - direct_native_eal,
        "delta_vs_direct_one_edit": camrs_eal - direct_one_edit_eal,
        "harmed_fraction": reconstructed_harmed_fraction,
        "reported_harmed_fraction": reported_harmed_fraction,
        "harmed_blocks": harmed_blocks,
        "camrs_first_token_correct_blocks": camrs_first_correct,
        "direct_native_first_token_correct_blocks": direct_first_correct,
        "first_token_shortfall_blocks": direct_first_correct - camrs_first_correct,
        "validation_blocks": int(evaluation["blocks"]),
        "validation_prompts": int(evaluation["prompts"]),
    }
    thresholds = {
        "delta_vs_dflash_strictly_greater_than": DFLASH_DELTA_THRESHOLD,
        "minimum_delta_vs_direct_native": DIRECT_DELTA_THRESHOLD,
        "minimum_delta_vs_direct_one_edit": DIRECT_DELTA_THRESHOLD,
        "maximum_harmed_fraction": MAX_HARMED_FRACTION,
        "maximum_first_token_shortfall_blocks": MAX_FIRST_TOKEN_SHORTFALL_BLOCKS,
        "validation_blocks": EXPECTED_VALIDATION_BLOCKS,
        "validation_prompts": EXPECTED_VALIDATION_PROMPTS,
    }
    checks = {
        "finite_values": all(
            math.isfinite(float(value)) for value in values.values()
        ),
        "delta_vs_dflash": values["delta_vs_dflash"]
        > thresholds["delta_vs_dflash_strictly_greater_than"],
        "delta_vs_direct_native": values["delta_vs_direct_native"]
        >= thresholds["minimum_delta_vs_direct_native"],
        "delta_vs_direct_one_edit": values["delta_vs_direct_one_edit"]
        >= thresholds["minimum_delta_vs_direct_one_edit"],
        "harmed_fraction": 0.0
        <= values["harmed_fraction"]
        <= thresholds["maximum_harmed_fraction"],
        "first_token_shortfall": values["first_token_shortfall_blocks"]
        <= thresholds["maximum_first_token_shortfall_blocks"],
        "validation_blocks": values["validation_blocks"]
        == thresholds["validation_blocks"],
        "validation_prompts": values["validation_prompts"]
        == thresholds["validation_prompts"],
    }
    return {
        "passed": all(checks.values()),
        "values": values,
        "thresholds": thresholds,
        "checks": checks,
        "selection": "raw_prompt_balanced_eal_then_lower_harm_then_lower_hinge",
        "failure_scope": (
            "exact CAMRS D64/H4/L1 seed0 full-OPB 37221-step development "
            "procedure with physically isolated validation and frozen Direct controls"
        ),
    }


def development_source_paths() -> dict[str, Path]:
    """Return the frozen runtime/import/document closure for provenance."""

    return {
        "trainer": Path(__file__).resolve(),
        "capacity_evaluator": PROJECT
        / "scripts/train_first_miss_max_regret_selector.py",
        "camrs_head": PROJECT / "src/sph/first_miss_max_regret_selector.py",
        "signed_value_head": PROJECT / "src/sph/first_miss_value_selector.py",
        "fmas_data_protocol": PROJECT
        / "scripts/train_first_miss_action_selector.py",
        "fmas_action_semantics": PROJECT
        / "src/sph/first_miss_action_selector.py",
        "direct_control_evaluator": PROJECT
        / "scripts/evaluate_direct_one_edit.py",
        "direct_trainer_utilities": PROJECT
        / "scripts/train_global_direct_selector.py",
        "direct_head": PROJECT / "src/sph/global_direct_selector.py",
        "canonical_data": PROJECT / "src/sph/data.py",
        "package_init": PROJECT / "src/sph/__init__.py",
        "survival_path_head": PROJECT / "src/sph/survival_path_head.py",
        "candidate_ceiling": PROJECT / "src/sph/candidate_ceiling.py",
        "capacity_helper": PROJECT / "src/sph/first_miss_capacity.py",
        "proposal": PROJECT
        / "refine-logs/first-miss-max-regret/FINAL_PROPOSAL.md",
        "capacity_verdict": PROJECT
        / "refine-logs/first-miss-max-regret/CAPACITY_RESULT_TO_CLAIM.md",
        "prelaunch_control_freeze": PROJECT
        / "refine-logs/first-miss-max-regret/PRELAUNCH_CONTROL_FREEZE.md",
        "launch_wrapper": PROJECT / "scripts/slurm/camrs_development.sbatch",
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    control, direct_metrics = load_and_validate_direct_control(args)
    control_hashes_start = {
        "direct_control_sha256": direct.sha256_file(args.direct_control),
        "direct_metrics_sha256": direct.sha256_file(args.direct_run / "metrics.json"),
        "direct_checkpoint_sha256": direct.sha256_file(args.direct_run / "best.pt"),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CAMRS development training requires CUDA")

    direct.seed_everything(args.seed)
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True)
    config_snapshot = direct.serializable_config(args)
    source_paths = development_source_paths()
    source_hashes_start = {
        f"{name}_sha256": direct.sha256_file(path)
        for name, path in source_paths.items()
    }
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for name, path in source_paths.items():
        shutil.copy2(path, snapshot / f"{name}_{path.name}")
    start = time.perf_counter()

    metadata_path = args.data / "metadata.json"
    manifest_path = args.data / "selected_manifest.jsonl"
    if direct.sha256_file(metadata_path) != EXPECTED_DATA_METADATA_SHA256:
        raise RuntimeError("isolated validation metadata hash mismatch")
    if direct.sha256_file(manifest_path) != EXPECTED_DATA_MANIFEST_SHA256:
        raise RuntimeError("isolated validation manifest hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
    if capacity_manifest is not None or validation_dataset is train_dataset:
        raise RuntimeError("development data unexpectedly use same-subset capacity mode")
    train_prompts = len({str(record["sample_id"]) for record in train_dataset.records})
    validation_prompts = len(
        {str(record["sample_id"]) for record in validation_dataset.records}
    )
    if len(train_dataset) != EXPECTED_TRAIN_BLOCKS:
        raise RuntimeError("training block cardinality mismatch")
    if train_prompts != EXPECTED_TRAIN_PROMPTS:
        raise RuntimeError("training prompt cardinality mismatch")
    if _prompt_set_sha256(train_dataset.records) != EXPECTED_TRAIN_PROMPT_SHA256:
        raise RuntimeError("training prompt set mismatch")
    if len(validation_dataset) != EXPECTED_VALIDATION_BLOCKS:
        raise RuntimeError("validation block cardinality mismatch")
    if validation_prompts != EXPECTED_VALIDATION_PROMPTS:
        raise RuntimeError("validation prompt cardinality mismatch")
    if (
        _prompt_set_sha256(validation_dataset.records)
        != EXPECTED_VALIDATION_PROMPT_SHA256
    ):
        raise RuntimeError("validation prompt set mismatch")
    external_hashes = [
        direct.sha256_file(path / "metadata.json") for path in args.train_data
    ]
    if external_hashes != EXPECTED_TRAIN_METADATA_SHA256:
        raise RuntimeError("external training metadata hashes differ")

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
        raise RuntimeError("CAMRS development requires 15 positions")

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
    if total_steps != EXPECTED_TOTAL_STEPS:
        raise RuntimeError("optimizer-step budget differs from 37,221")
    scheduler, warmup_steps = direct.cosine_warmup_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio
    )

    initial_validation = capacity.evaluate(
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
            "validation": capacity.compact_epoch_metrics(initial_validation),
            "selection_key": list(checkpoint_selection_key(initial_validation)),
            "is_selected": False,
        }
    ]
    best_key = checkpoint_selection_key(initial_validation)
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
                raise FloatingPointError("nonfinite CAMRS development loss")
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

        validation = capacity.evaluate(
            model,
            validation_loader,
            target_embedding,
            device,
            candidate_k=args.candidate_k,
        )
        key = checkpoint_selection_key(validation)
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
            "validation": capacity.compact_epoch_metrics(validation),
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
    model.load_state_dict(checkpoint["model"])
    final_validation = capacity.evaluate(
        model,
        validation_loader,
        target_embedding,
        device,
        candidate_k=args.candidate_k,
        include_examples=True,
        require_base_identity=selected_epoch == 0,
    )
    if checkpoint_selection_key(final_validation) != tuple(checkpoint["selection_key"]):
        raise RuntimeError("selected checkpoint metrics differ from selection key")
    development_gate = development_gate_report(final_validation, control)

    source_hashes_end = {
        f"{name}_sha256_at_end": direct.sha256_file(path)
        for name, path in source_paths.items()
    }
    for name, path in source_paths.items():
        if source_hashes_start[f"{name}_sha256"] != source_hashes_end[
            f"{name}_sha256_at_end"
        ]:
            raise RuntimeError(f"source changed during development run: {path}")
    control_hashes_end = {
        "direct_control_sha256": direct.sha256_file(args.direct_control),
        "direct_metrics_sha256": direct.sha256_file(args.direct_run / "metrics.json"),
        "direct_checkpoint_sha256": direct.sha256_file(args.direct_run / "best.pt"),
    }
    if control_hashes_end != control_hashes_start:
        raise RuntimeError("frozen Direct control artifacts changed during training")
    provenance = {
        "project_commit": direct.git_revision(PROJECT),
        "project_dirty": direct.git_is_dirty(PROJECT),
        "data_metadata_sha256": direct.sha256_file(metadata_path),
        "data_manifest_sha256": direct.sha256_file(manifest_path),
        "external_train_data": [
            {
                "path": str(path.resolve()),
                "metadata_sha256": digest,
                "base_greedy_witness_status": dataset.base_greedy_witness_status,
            }
            for path, digest, dataset in zip(
                args.train_data, external_hashes, external_collections, strict=True
            )
        ],
        "verified_target_embedding_files": verified_target_files,
        "verified_external_target_embedding_files": verified_external,
        "direct_control_path": str(args.direct_control.resolve()),
        **control_hashes_start,
        **{f"{name}_at_start": value for name, value in control_hashes_start.items()},
        **{f"{name}_at_end": value for name, value in control_hashes_end.items()},
        **source_hashes_start,
        **source_hashes_end,
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
        "evidence_tier": "development",
        "split_protocol": "prompt_disjoint_external_train_physical_validation",
        "train_blocks": len(train_dataset),
        "train_prompts": train_prompts,
        "train_prompt_set_sha256": _prompt_set_sha256(train_dataset.records),
        "validation_blocks": len(validation_dataset),
        "validation_prompts": validation_prompts,
        "validation_prompt_set_sha256": _prompt_set_sha256(
            validation_dataset.records
        ),
        "parameter_count": parameter_count,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "selected_epoch": selected_epoch,
        "selection_key": checkpoint["selection_key"],
        "seconds": time.perf_counter() - start,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_cuda_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "history": history,
        "direct_control": {
            "path": str(args.direct_control.resolve()),
            "direct_run": control["direct_run"],
            "direct_selected_epoch": control["direct_selected_epoch"],
            "base": control["base"],
            "direct_native": control["direct_native"],
            "direct_one_edit": control["direct_one_edit"],
            "single_edit_oracle": control["single_edit_oracle"],
            "one_edit_diagnostics": control["one_edit_diagnostics"],
            "direct_metrics_selected_epoch": direct_metrics["selected_epoch"],
        },
        "development_gate": development_gate,
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
                "selection_key": checkpoint["selection_key"],
                "development_gate": development_gate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.require_development_gate and not development_gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
