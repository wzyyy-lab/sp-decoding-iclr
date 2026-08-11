#!/usr/bin/env python3
"""Open and adjudicate the frozen PROS-Gate R083 falsifier exactly once.

This entry point has no training, calibration, threshold, checkpoint-selection,
validation, reserved, or formal-data surface.  It materializes only the frozen
``falsifier`` assignment, evaluates the already published R082 sidecar and
comparators, publishes complete evidence, then exits nonzero on scientific
failure only after READY is durable.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from sph.direct_safety_artifacts import (
    OUTCOME_ARTIFACT_FORMAT_VERSION,
    OUTCOME_ARTIFACT_PROTOCOL,
    _outcome_summary,
    block_key_from_mapping,
    load_outcome_bundle,
    sha256_file,
    split_assignment_map,
    validate_outcome_record,
    verify_phase3_split_manifest,
)
from sph.direct_safety_gate import (
    DirectSafetySidecar,
    scalar_comparator_features,
)
from sph.direct_safety_numeric_policy import numeric_policy_receipt
from sph.direct_safety_protocol import (
    BlockKey,
    SavedGateRecord,
    WeightedRidgeModel,
    assert_stage_splits,
    deterministic_bootstrap_indices,
    reconstruct_saved_gate_evaluation,
)
from sph.direct_safety_publication import (
    FALSIFIER_PURPOSE,
    commit_publication,
    publication_identity,
    reserve_publication_directory,
    verify_published_directory as verify_r083_publication,
)
from sph.source_closure import snapshot_source_closure, verify_source_manifest

try:
    from materialize_direct_safety_artifacts import (
        PHASE3_BLOCKS,
        PHASE3_PROMPTS,
        SOURCE_SPLIT,
        _identity_records,
        load_exclusions,
        load_frozen_direct,
        load_phase3_source,
        materialize_falsifier_outcome_records,
        split_provenance,
    )
    from train_direct_safety_fit import (
        _training_tensors,
        _to_device,
        domain_slice_metrics,
        evaluate_model,
        evaluate_scores,
        verify_published_directory as verify_r082_publication,
    )
    from verify_pros_gate_receipt import (
        load_receipt,
        verify_split_receipt,
    )
except ModuleNotFoundError:  # Imported as ``scripts.*`` in CPU tests.
    from scripts.materialize_direct_safety_artifacts import (
        PHASE3_BLOCKS,
        PHASE3_PROMPTS,
        SOURCE_SPLIT,
        _identity_records,
        load_exclusions,
        load_frozen_direct,
        load_phase3_source,
        materialize_falsifier_outcome_records,
        split_provenance,
    )
    from scripts.train_direct_safety_fit import (
        _training_tensors,
        _to_device,
        domain_slice_metrics,
        evaluate_model,
        evaluate_scores,
        verify_published_directory as verify_r082_publication,
    )
    from scripts.verify_pros_gate_receipt import (
        load_receipt,
        verify_split_receipt,
    )


PROJECT = Path(__file__).resolve().parents[1]
ENTRYPOINT_RELATIVE = "scripts/evaluate_direct_safety_falsifier.py"
FALSIFIER_PROTOCOL = "pros-gate-one-shot-falsifier-v1"
FALSIFIER_SPLIT = "falsifier"
FALSIFIER_PROMPTS = 200
FALSIFIER_DOMAIN_PROMPTS = {"chat": 66, "code": 67, "math": 67}
MATERIALIZATION_BATCH_SIZE = 32
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_806
QUANTILE_LEVELS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
RECOVERY_MINIMUM = 0.90
HARM_MAXIMUM = 0.05
FIRST_TOKEN_MAX_SHORTFALL = 1
COMPARATOR_RECOVERY_MARGIN = 0.05
GATE_CHECK_NAMES = (
    "finite_values_and_selected_gradients",
    "zero_regret_bound_violations",
    "finite_positive_unclipped_recovery",
    "recovery_at_least_0p90",
    "eal_strictly_above_dflash",
    "eal_strictly_above_direct",
    "harmed_fraction_at_most_0p05",
    "first_token_shortfall_at_most_one",
    "all_frozen_comparator_recoveries_valid",
    "recovery_margin_at_least_0p05_over_best_comparator",
    "exact_identity_and_data_boundary",
)

EXPECTED_CANONICAL_METADATA_SHA256 = (
    "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "7a572670867bbf6f811aad58c7ed1365f179083cda8187f5402221d554d1f1c0"
)
EXPECTED_SPLIT_AUDIT_SHA256 = (
    "3df67764ef6dc7e8a827277c34730233bc7a5155451fe42c6b98b99bf7a7ef76"
)
EXPECTED_R079_SOURCE_MANIFEST_SHA256 = (
    "2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4"
)
EXPECTED_DIRECT_CHECKPOINT_SHA256 = (
    "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e"
)
EXPECTED_DIRECT_METRICS_SHA256 = (
    "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef"
)
EXPECTED_R082_PUBLICATION_MANIFEST_SHA256 = (
    "9c17020a95d42725a097e6847b5023d6c6e7971a0647fb4475727e93d2297a5e"
)
EXPECTED_R082_READY_SHA256 = (
    "91c51864339321436ffa560470667bb99d1a22e955e7273fcd7a3d5711f3f508"
)
EXPECTED_R082_METRICS_SHA256 = (
    "ba7fd8264813b7baa4927d94d1acd8d697bad5175d1acdfcc62dea5a103491b0"
)
EXPECTED_SELECTED_BUNDLE_SHA256 = (
    "a0abcfd4e56229647afd1dda5ca2fe861f7dbf21d00c09fe275ba8a66826c142"
)
EXPECTED_SELECTED_CHECKPOINT_SHA256 = (
    "f3e7c68dafd93528c03deda9710e3d23cf5b0e9e51a7b2ef66200f08201066dc"
)
EXPECTED_SELECTED_RECORDS_SHA256 = (
    "1fd6beb846c8a46e874875628b37ac1094e3ac61eaddbfef3bb9d7b7dbd88749"
)
EXPECTED_RIDGE_MODEL_SHA256 = (
    "2c5a76ca96f9f6afb08d47a116cdb17fdce6de11386b6350c5e3e485732f4f16"
)
EXPECTED_RIDGE_RECEIPT_SHA256 = (
    "2a4f6457fc1d85a3ad42ec7430ecb4d28d532d4bf57f06371ac019526c3dc809"
)
EXPECTED_R082_SOURCE_MANIFEST_SHA256 = (
    "f36291a961ea793dbaa888950bc4312d8b53954fcc5ecdb01a5caad4af97e184"
)
EXPECTED_SELECTED_PASS = 5
EXPECTED_SELECTED_UPDATES = 995
EXPECTED_PARAMETER_COUNT = 38_674
EXPECTED_DATA = PROJECT / "artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
EXPECTED_SPLIT_MANIFEST = PROJECT / "artifacts/pros_gate/r079_numeric_v2/split_manifest.json"
EXPECTED_SPLIT_AUDIT = PROJECT / "artifacts/pros_gate/r079_numeric_v2/audits/split.json"
EXPECTED_DIRECT_RUN = PROJECT / (
    "artifacts/training/gcls_v4_feature_100k_10133585/"
    "compact_axial_additive_d64_full_seed0"
)
EXPECTED_R082_OUTPUT = PROJECT / "artifacts/training/pros_gate_fit_10141115/seed0"
EXPECTED_TARGET = Path("/hpc2hdd/home/zwang668/sp decoding/hf_assets/models/Qwen3-4B")
EXPECTED_EXCLUSIONS = {
    "producer_train": PROJECT / "artifacts/manifests/open_perfectblend_100k_v2.jsonl",
    "validation": PROJECT / "artifacts/manifests/phase3_development_v3.jsonl",
    "reserved": PROJECT / "artifacts/manifests/phase3_reserved_test_v3.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-audit-receipt", type=Path, required=True)
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--r082-output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclusion",
        action="append",
        required=True,
        metavar="ROLE=JSONL",
    )
    return parser.parse_args()


def _require_regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} is missing or is a symlink")


def _file_identity(path: Path, name: str) -> dict[str, int | str]:
    _require_regular_file(path, name)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _require_exact_hash(path: Path, expected: str, name: str) -> None:
    observed = _file_identity(path, name)["sha256"]
    if observed != expected:
        raise RuntimeError(f"{name} hash differs from the frozen input")


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError(f"failed to write R083 JSON: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_torch(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _nested_exact_equal(left: Any, right: Any) -> bool:
    """Tensor-aware exact equality for persisted evidence replay."""

    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return (
            isinstance(left, Tensor)
            and isinstance(right, Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _nested_exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(
            right, (list, tuple)
        ):
            return False
        return len(left) == len(right) and all(
            _nested_exact_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return bool(left == right)


def _canonical_order(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in records],
        key=lambda row: block_key_from_mapping(row).serialize(),
    )


def _parsed_exclusions(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise RuntimeError("R083 exclusion must use ROLE=PATH")
        role, raw_path = value.split("=", 1)
        if role in parsed or role not in EXPECTED_EXCLUSIONS or not raw_path:
            raise RuntimeError("R083 exclusion roles differ")
        path = Path(raw_path).resolve()
        if path != EXPECTED_EXCLUSIONS[role].resolve():
            raise RuntimeError(f"R083 exclusion path differs for {role}")
        parsed[role] = path
    if set(parsed) != set(EXPECTED_EXCLUSIONS):
        raise RuntimeError("R083 exclusion role set differs")
    return parsed


def _validate_exact_paths(args: argparse.Namespace, job_id: str) -> None:
    expected = {
        "data": EXPECTED_DATA,
        "split_manifest": EXPECTED_SPLIT_MANIFEST,
        "split_audit_receipt": EXPECTED_SPLIT_AUDIT,
        "direct_run": EXPECTED_DIRECT_RUN,
        "target": EXPECTED_TARGET,
        "r082_output": EXPECTED_R082_OUTPUT,
        "output": PROJECT / f"artifacts/evaluation/pros_gate_falsifier_{job_id}/seed0",
    }
    for name, path in expected.items():
        if getattr(args, name).resolve() != path.resolve():
            raise RuntimeError(f"R083 {name} path differs from the frozen contract")
    _parsed_exclusions(args.exclusion)


def _write_outcome_bundle(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    split_manifest_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    root.mkdir()
    ordered = _canonical_order(records)
    for row in ordered:
        validate_outcome_record(row, expected_split=FALSIFIER_SPLIT)
    records_path = root / "records.pt"
    _save_torch(records_path, ordered)
    metadata = {
        "format_version": OUTCOME_ARTIFACT_FORMAT_VERSION,
        "protocol": OUTCOME_ARTIFACT_PROTOCOL,
        "numeric_policy": numeric_policy_receipt(),
        "split": FALSIFIER_SPLIT,
        "split_manifest_sha256": split_manifest_sha256,
        "records_sha256": sha256_file(records_path),
        "summary": _outcome_summary(ordered, split=FALSIFIER_SPLIT),
        "provenance": dict(provenance),
    }
    _write_json(root / "metadata.json", metadata)
    reloaded, reloaded_metadata = load_outcome_bundle(
        root, expected_split=FALSIFIER_SPLIT
    )
    if reloaded_metadata != metadata or not _nested_exact_equal(reloaded, ordered):
        raise RuntimeError("R083 saved outcome bundle replay differs")
    return metadata


def _load_selected_sidecar(root: Path, device: torch.device) -> tuple[DirectSafetySidecar, dict[str, Any]]:
    checkpoint_path = root / "selected.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("R082 selected checkpoint payload is not a mapping")
    expected_scalars = {
        "protocol": "pros-gate-fit-checkpoint-training-v1",
        "pass": EXPECTED_SELECTED_PASS,
        "completed_updates": EXPECTED_SELECTED_UPDATES,
        "source_manifest_sha256": EXPECTED_R082_SOURCE_MANIFEST_SHA256,
        "ridge_model_sha256": EXPECTED_RIDGE_MODEL_SHA256,
    }
    for name, expected in expected_scalars.items():
        if payload.get(name) != expected:
            raise RuntimeError(f"R082 selected checkpoint differs for {name}")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("R082 selected checkpoint lacks model state")
    model = DirectSafetySidecar(initialization_seed=0)
    model.load_state_dict(state, strict=True)
    if sum(value.numel() for value in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("R082 selected sidecar parameter count differs")
    if not all(bool(torch.isfinite(value).all()) for value in model.parameters()):
        raise RuntimeError("R082 selected sidecar contains nonfinite parameters")
    model.requires_grad_(False).eval().to(device)
    if payload.get("checkpoint", {}).get("gradients_finite") is not True:
        raise RuntimeError("R082 selected checkpoint lacks finite-gradient witness")
    return model, dict(payload)


def _load_frozen_ridge(root: Path) -> tuple[WeightedRidgeModel, dict[str, Any]]:
    payload = torch.load(root / "ridge_model.pt", map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("protocol") != "pros-gate-fit-ridge-freeze-v1":
        raise RuntimeError("R082 ridge payload protocol differs")
    expected = {
        "fit_metadata_sha256": "061069ed644b7fd700d7b65586622c02ef878c611ab6a549968f78bba8425f98",
        "fit_records_sha256": "645007ec2665e141813b09e4bd1e35c33337b4b32e27655ae088c52c89fbcc6b",
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "source_manifest_sha256": EXPECTED_R082_SOURCE_MANIFEST_SHA256,
        "feature_dimension": 21,
        "ridge": 0.001,
        "decision": "apply_iff_float64_score_gt_zero",
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"R082 ridge payload differs for {name}")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise RuntimeError("R082 ridge state is missing")
    model = WeightedRidgeModel(
        feature_mean=state["feature_mean"].double(),
        feature_scale=state["feature_scale"].double(),
        constant_features=state["constant_features"].bool(),
        coefficients=state["coefficients"].double(),
        intercept=state["intercept"].double(),
        ridge=float(payload["ridge"]),
    )
    return model, dict(payload)


def _all_tensors_finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensors_finite(item) for item in value)
    return True


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile requires nonempty values and q in [0,1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _quantile_report(values: Sequence[float]) -> dict[str, float]:
    return {
        f"q{int(round(level * 100)):02d}": _linear_quantile(values, level)
        for level in QUANTILE_LEVELS
    }


def _saved_rows_to_replay(rows: Sequence[Mapping[str, Any]]) -> list[SavedGateRecord]:
    result: list[SavedGateRecord] = []
    for row in rows:
        result.append(
            SavedGateRecord(
                block_key=BlockKey(
                    str(row["sample_id"]),
                    int(row["anchor_offset"]),
                    int(row["context_length"]),
                ),
                base_length=int(row["base_length"]),
                direct_length=int(row["direct_length"]),
                score=float(row["score"]),
                base_first_token_correct=bool(row["base_first_token_correct"]),
                direct_first_token_correct=bool(row["direct_first_token_correct"]),
            )
        )
    return result


def _saved_record_replay(
    rows_by_system: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    return {
        name: reconstruct_saved_gate_evaluation(
            _saved_rows_to_replay(rows), require_valid_recovery=False
        ).metrics
        for name, rows in sorted(rows_by_system.items())
    }


def _bootstrap_report(
    records: Sequence[Mapping[str, Any]],
    scores_by_system: Mapping[str, Tensor],
    *,
    prompt_set_sha256: str,
) -> dict[str, Any]:
    prompt_ids = sorted({str(row["sample_id"]) for row in records})
    if len(prompt_ids) != FALSIFIER_PROMPTS:
        raise RuntimeError("R083 bootstrap prompt cardinality differs")
    prompt_index = {sample_id: index for index, sample_id in enumerate(prompt_ids)}
    counts = torch.zeros(len(prompt_ids), dtype=torch.float64)
    base_sums = torch.zeros_like(counts)
    direct_sums = torch.zeros_like(counts)
    oracle_sums = torch.zeros_like(counts)
    system_sums = {
        name: torch.zeros_like(counts) for name in scores_by_system
    }
    system_harmed = {
        name: torch.zeros_like(counts) for name in scores_by_system
    }
    system_first = {
        name: torch.zeros_like(counts) for name in scores_by_system
    }
    direct_first = torch.zeros_like(counts)
    cpu_scores = {
        name: value.detach().to(device="cpu", dtype=torch.float64)
        for name, value in scores_by_system.items()
    }
    for row_index, row in enumerate(records):
        index = prompt_index[str(row["sample_id"])]
        base = int(row["base_length"])
        direct = int(row["direct_length"])
        counts[index] += 1.0
        base_sums[index] += base
        direct_sums[index] += direct
        oracle_sums[index] += max(base, direct)
        direct_first[index] += int(bool(row["direct_first_token_correct"]))
        for name, scores in cpu_scores.items():
            apply = float(scores[row_index]) > 0.0
            system_sums[name][index] += direct if apply else base
            system_harmed[name][index] += int(apply and direct < base)
            system_first[name][index] += int(
                bool(row["direct_first_token_correct"])
                if apply
                else bool(row["base_first_token_correct"])
            )
    if bool((counts <= 0).any()):
        raise RuntimeError("R083 bootstrap contains an empty prompt")
    indices = deterministic_bootstrap_indices(
        len(prompt_ids),
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        prompt_set_sha256=prompt_set_sha256,
    )
    base_eal = (base_sums / counts)[indices].mean(dim=1)
    direct_eal = (direct_sums / counts)[indices].mean(dim=1)
    oracle_eal = (oracle_sums / counts)[indices].mean(dim=1)
    denominator = oracle_eal - base_eal
    result: dict[str, Any] = {}
    for name in sorted(scores_by_system):
        method_eal = (system_sums[name] / counts)[indices].mean(dim=1)
        recovery = (method_eal - base_eal) / denominator
        harmed = system_harmed[name][indices].sum(dim=1) / counts[indices].sum(dim=1)
        first_delta = (
            system_first[name][indices].sum(dim=1)
            - direct_first[indices].sum(dim=1)
        )
        valid_recovery = torch.isfinite(recovery) & denominator.gt(0.0)
        finite_recovery = recovery[valid_recovery]
        recovery_interval = (
            None
            if finite_recovery.numel() == 0
            else [
                _linear_quantile(finite_recovery.tolist(), 0.025),
                _linear_quantile(finite_recovery.tolist(), 0.975),
            ]
        )
        result[name] = {
            "method_eal_ci95": [
                _linear_quantile(method_eal.tolist(), 0.025),
                _linear_quantile(method_eal.tolist(), 0.975),
            ],
            "delta_vs_dflash_ci95": [
                _linear_quantile((method_eal - base_eal).tolist(), 0.025),
                _linear_quantile((method_eal - base_eal).tolist(), 0.975),
            ],
            "delta_vs_direct_ci95": [
                _linear_quantile((method_eal - direct_eal).tolist(), 0.025),
                _linear_quantile((method_eal - direct_eal).tolist(), 0.975),
            ],
            "oracle_recovery_ci95": recovery_interval,
            "valid_recovery_replicates": int(finite_recovery.numel()),
            "invalid_recovery_replicates": int(
                recovery.numel() - finite_recovery.numel()
            ),
            "harmed_fraction_ci95": [
                _linear_quantile(harmed.tolist(), 0.025),
                _linear_quantile(harmed.tolist(), 0.975),
            ],
            "first_token_delta_vs_direct_ci95": [
                _linear_quantile(first_delta.tolist(), 0.025),
                _linear_quantile(first_delta.tolist(), 0.975),
            ],
        }
    return {
        "protocol": "pros-gate-prompt-cluster-bootstrap-v1",
        "unit": "prompt/sample_id; all blocks retained together",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "prompt_count": len(prompt_ids),
        "prompt_set_sha256": prompt_set_sha256,
        "percentile_interval": [0.025, 0.975],
        "diagnostic_only_point_estimates_bind": True,
        "systems": result,
    }


def _gate_checks(
    pros: Mapping[str, Any],
    comparators: Mapping[str, Mapping[str, Any]],
    *,
    identity_checks_passed: bool,
    gradients_finite: bool,
) -> dict[str, bool]:
    recovery = pros.get("oracle_recovery")
    denominator = pros.get("recovery_denominator")
    recovery_valid = (
        isinstance(recovery, (int, float))
        and not isinstance(recovery, bool)
        and math.isfinite(float(recovery))
        and isinstance(denominator, (int, float))
        and not isinstance(denominator, bool)
        and math.isfinite(float(denominator))
        and float(denominator) > 0.0
        and 0.0 <= float(recovery) <= 1.0 + 1e-6
    )
    comparator_recoveries = []
    comparators_valid = True
    for metrics in comparators.values():
        value = metrics.get("oracle_recovery")
        comparator_denominator = metrics.get("recovery_denominator")
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and isinstance(comparator_denominator, (int, float))
            and not isinstance(comparator_denominator, bool)
            and math.isfinite(float(comparator_denominator))
            and float(comparator_denominator) > 0.0
            and 0.0 <= float(value) <= 1.0 + 1e-6
        )
        comparators_valid = comparators_valid and valid
        if valid:
            comparator_recoveries.append(float(value))
    best_comparator = max(comparator_recoveries) if comparator_recoveries else math.inf
    checks = {
        "finite_values_and_selected_gradients": (
            pros.get("values_finite") is True and gradients_finite
        ),
        "zero_regret_bound_violations": pros.get("regret_bound_violation_count") == 0,
        "finite_positive_unclipped_recovery": recovery_valid,
        "recovery_at_least_0p90": recovery_valid and float(recovery) >= RECOVERY_MINIMUM,
        "eal_strictly_above_dflash": float(pros["method_eal"]) > float(pros["base_eal"]),
        "eal_strictly_above_direct": float(pros["method_eal"]) > float(pros["direct_eal"]),
        "harmed_fraction_at_most_0p05": float(pros["harmed_fraction"]) <= HARM_MAXIMUM,
        "first_token_shortfall_at_most_one": (
            int(pros["method_first_token_count"])
            >= int(pros["direct_first_token_count"]) - FIRST_TOKEN_MAX_SHORTFALL
        ),
        "all_frozen_comparator_recoveries_valid": comparators_valid,
        "recovery_margin_at_least_0p05_over_best_comparator": (
            recovery_valid
            and comparators_valid
            and math.isfinite(best_comparator)
            and float(recovery) >= best_comparator + COMPARATOR_RECOVERY_MARGIN
        ),
        "exact_identity_and_data_boundary": identity_checks_passed,
    }
    if tuple(checks) != GATE_CHECK_NAMES:
        raise RuntimeError("R083 gate-check schema differs from the frozen contract")
    return checks


def _gate_thresholds() -> dict[str, Any]:
    return {
        "recovery_minimum": RECOVERY_MINIMUM,
        "harm_maximum": HARM_MAXIMUM,
        "first_token_max_shortfall": FIRST_TOKEN_MAX_SHORTFALL,
        "comparator_recovery_margin": COMPARATOR_RECOVERY_MARGIN,
        "decision": "apply_iff_raw_score_gt_zero",
    }


def _run_configuration() -> dict[str, Any]:
    return {
        "split": FALSIFIER_SPLIT,
        "prompts": FALSIFIER_PROMPTS,
        "domain_prompts": FALSIFIER_DOMAIN_PROMPTS,
        "materialization_batch_size": MATERIALIZATION_BATCH_SIZE,
        "decision": "apply_iff_raw_score_gt_zero",
        "selected_pass": EXPECTED_SELECTED_PASS,
        "selected_updates": EXPECTED_SELECTED_UPDATES,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _identity_checks_passed(
    frozen_bindings: Mapping[str, Any], native_witness: Mapping[str, Any]
) -> bool:
    return (
        frozen_bindings["r082_publication"]["status"] == "READY"
        and frozen_bindings["split_audit"]["status"] == "BOUND"
        and native_witness["regular_vs_hooked_outputs_bitwise"] is True
        and native_witness["hooked_repeat_outputs_bitwise"] is True
        and native_witness["hooked_repeat_node_states_bitwise"] is True
        and native_witness["state_dict_sha256_before"]
        == native_witness["state_dict_sha256_after"]
    )


def _scientific_evaluation_failure(
    error: BaseException, *, system: str
) -> str | None:
    message = str(error)
    if message in {
        "sidecar scores must be finite",
        "saved gate scores are non-finite",
        "scalar comparator features must be finite",
    }:
        return f"nonfinite_{system}_values"
    if message == "saved-record evaluation violated the regret bound":
        return f"{system}_regret_bound_violation"
    return None


def _score_diagnostics(scores: Mapping[str, Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in sorted(scores.items()):
        cpu = value.detach().to(device="cpu")
        finite = torch.isfinite(cpu)
        result[name] = {
            "dtype": str(cpu.dtype),
            "shape": list(cpu.shape),
            "elements": int(cpu.numel()),
            "finite_elements": int(finite.sum()),
            "nonfinite_elements": int((~finite).sum()),
            "nan_elements": int(torch.isnan(cpu).sum())
            if cpu.is_floating_point()
            else 0,
            "positive_infinity_elements": int(torch.isposinf(cpu).sum())
            if cpu.is_floating_point()
            else 0,
            "negative_infinity_elements": int(torch.isneginf(cpu).sum())
            if cpu.is_floating_point()
            else 0,
        }
    return result


def _scientific_failure_checks(
    pros_metrics: Mapping[str, Any] | None,
    *,
    identity_checks_passed: bool,
    gradients_finite: bool,
) -> dict[str, bool]:
    if pros_metrics is None:
        checks = {name: False for name in GATE_CHECK_NAMES}
        checks["exact_identity_and_data_boundary"] = identity_checks_passed
        return checks
    checks = _gate_checks(
        pros_metrics,
        {},
        identity_checks_passed=identity_checks_passed,
        gradients_finite=gradients_finite,
    )
    checks["all_frozen_comparator_recoveries_valid"] = False
    checks["recovery_margin_at_least_0p05_over_best_comparator"] = False
    return checks


def _load_json_mapping(path: Path, name: str) -> dict[str, Any]:
    _require_regular_file(path, name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


def _safe_input_child(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative or "\0" in relative:
        raise RuntimeError(f"{name} path is invalid")
    component = Path(relative)
    if (
        component.is_absolute()
        or ".." in component.parts
        or component.as_posix() != relative
    ):
        raise RuntimeError(f"{name} path escapes its frozen root")
    path = root / component
    _require_regular_file(path, name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"{name} resolves outside its frozen root")
    return path


def _canonical_shard_identities(data: Path) -> list[dict[str, int | str]]:
    metadata = _load_json_mapping(data / "metadata.json", "canonical metadata")
    rows = metadata.get("shards")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("canonical metadata lacks a shard manifest")
    expected_names: list[str] = []
    identities: list[dict[str, int | str]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not {
            "path",
            "bytes",
            "sha256",
        }.issubset(row):
            raise RuntimeError("canonical shard manifest entry differs")
        relative = row["path"]
        path = _safe_input_child(data, relative, "canonical shard")
        identity = {"path": str(relative), **_file_identity(path, "canonical shard")}
        if (
            identity["bytes"] != row["bytes"]
            or identity["sha256"] != row["sha256"]
        ):
            raise RuntimeError("canonical shard identity differs from metadata")
        expected_names.append(str(relative))
        identities.append(identity)
    if expected_names != sorted(expected_names) or len(set(expected_names)) != len(
        expected_names
    ):
        raise RuntimeError("canonical shard manifest paths are not sorted and unique")
    actual_names = [path.name for path in sorted(data.glob("shard-*.pt"))]
    if actual_names != expected_names:
        raise RuntimeError("canonical shard surface differs from metadata")
    return identities


def _target_embedding_identities(
    data: Path, target: Path
) -> dict[str, Any]:
    metadata = _load_json_mapping(data / "metadata.json", "canonical metadata")
    rows = metadata.get("provenance", {}).get("target_files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("canonical metadata lacks target file identities")
    expected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not {
            "path",
            "bytes",
            "sha256",
        }.issubset(row):
            raise RuntimeError("target identity manifest entry differs")
        relative = str(row["path"])
        if relative in expected:
            raise RuntimeError("target identity manifest repeats a path")
        expected[relative] = row

    index_name = "model.safetensors.index.json"
    required_names = ["config.json", index_name]
    captured: dict[str, dict[str, int | str]] = {}
    for relative in required_names:
        if relative not in expected:
            raise RuntimeError(f"target identity lacks {relative}")
        path = _safe_input_child(target, relative, "target embedding input")
        identity = {"path": relative, **_file_identity(path, "target embedding input")}
        reference = expected[relative]
        if (
            identity["bytes"] != reference["bytes"]
            or identity["sha256"] != reference["sha256"]
        ):
            raise RuntimeError(f"target embedding input differs: {relative}")
        captured[relative] = identity

    index = _load_json_mapping(target / index_name, "target embedding index")
    weight_map = index.get("weight_map")
    embedding_key = "model.embed_tokens.weight"
    if not isinstance(weight_map, Mapping) or embedding_key not in weight_map:
        raise RuntimeError("target index lacks the frozen embedding key")
    embedding_shard = weight_map[embedding_key]
    if not isinstance(embedding_shard, str) or embedding_shard not in expected:
        raise RuntimeError("target embedding shard differs from collection metadata")
    shard_path = _safe_input_child(
        target, embedding_shard, "target embedding shard"
    )
    shard_identity = {
        "path": embedding_shard,
        **_file_identity(shard_path, "target embedding shard"),
    }
    reference = expected[embedding_shard]
    if (
        shard_identity["bytes"] != reference["bytes"]
        or shard_identity["sha256"] != reference["sha256"]
    ):
        raise RuntimeError("target embedding shard identity differs")
    captured[embedding_shard] = shard_identity
    return {
        "embedding_key": embedding_key,
        "embedding_shard": embedding_shard,
        "files": [captured[name] for name in [*required_names, embedding_shard]],
    }


def _capture_input_identities(args: argparse.Namespace) -> dict[str, Any]:
    r082 = args.r082_output
    identities = {
        "canonical_metadata": _file_identity(args.data / "metadata.json", "canonical metadata"),
        "canonical_shards": _canonical_shard_identities(args.data),
        "target_embedding": _target_embedding_identities(args.data, args.target),
        "split_manifest": _file_identity(args.split_manifest, "split manifest"),
        "split_audit_receipt": _file_identity(args.split_audit_receipt, "split audit receipt"),
        "direct_checkpoint": _file_identity(args.direct_run / "best.pt", "Direct checkpoint"),
        "direct_metrics": _file_identity(args.direct_run / "metrics.json", "Direct metrics"),
        "r082_ready": _file_identity(r082 / "READY.json", "R082 READY"),
        "r082_publication_manifest": _file_identity(
            r082 / "PUBLICATION_MANIFEST.json", "R082 publication manifest"
        ),
        "r082_metrics": _file_identity(r082 / "metrics.json", "R082 metrics"),
        "selected_bundle": _file_identity(r082 / "selected_bundle.json", "selected bundle"),
        "selected_checkpoint": _file_identity(r082 / "selected.pt", "selected checkpoint"),
        "selected_checkpoint_records": _file_identity(
            r082 / "selected_checkpoint_records.pt", "selected checkpoint records"
        ),
        "ridge_model": _file_identity(r082 / "ridge_model.pt", "ridge model"),
        "ridge_freeze_receipt": _file_identity(
            r082 / "ridge_freeze_receipt.json", "ridge freeze receipt"
        ),
        "source_manifest": _file_identity(args.source_manifest, "R083 source manifest"),
    }
    identities["exclusions"] = {
        role: _file_identity(path, f"{role} exclusion manifest")
        for role, path in sorted(_parsed_exclusions(args.exclusion).items())
    }
    return identities


def _validate_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    expected_files = (
        (args.data / "metadata.json", EXPECTED_CANONICAL_METADATA_SHA256, "canonical metadata"),
        (args.split_manifest, EXPECTED_SPLIT_MANIFEST_SHA256, "split manifest"),
        (args.split_audit_receipt, EXPECTED_SPLIT_AUDIT_SHA256, "split audit receipt"),
        (args.direct_run / "best.pt", EXPECTED_DIRECT_CHECKPOINT_SHA256, "Direct checkpoint"),
        (args.direct_run / "metrics.json", EXPECTED_DIRECT_METRICS_SHA256, "Direct metrics"),
        (args.r082_output / "READY.json", EXPECTED_R082_READY_SHA256, "R082 READY"),
        (
            args.r082_output / "PUBLICATION_MANIFEST.json",
            EXPECTED_R082_PUBLICATION_MANIFEST_SHA256,
            "R082 publication manifest",
        ),
        (args.r082_output / "metrics.json", EXPECTED_R082_METRICS_SHA256, "R082 metrics"),
        (
            args.r082_output / "selected_bundle.json",
            EXPECTED_SELECTED_BUNDLE_SHA256,
            "selected bundle",
        ),
        (
            args.r082_output / "selected.pt",
            EXPECTED_SELECTED_CHECKPOINT_SHA256,
            "selected checkpoint",
        ),
        (
            args.r082_output / "selected_checkpoint_records.pt",
            EXPECTED_SELECTED_RECORDS_SHA256,
            "selected checkpoint records",
        ),
        (args.r082_output / "ridge_model.pt", EXPECTED_RIDGE_MODEL_SHA256, "ridge model"),
        (
            args.r082_output / "ridge_freeze_receipt.json",
            EXPECTED_RIDGE_RECEIPT_SHA256,
            "ridge freeze receipt",
        ),
    )
    for path, expected, name in expected_files:
        _require_exact_hash(path, expected, name)
    selected_bundle = json.loads(
        (args.r082_output / "selected_bundle.json").read_text(encoding="utf-8")
    )
    expected_bundle = {
        "status": "FROZEN",
        "selected_pass": EXPECTED_SELECTED_PASS,
        "selected_updates": EXPECTED_SELECTED_UPDATES,
        "selected_checkpoint_sha256": EXPECTED_SELECTED_CHECKPOINT_SHA256,
        "selected_records_sha256": EXPECTED_SELECTED_RECORDS_SHA256,
        "ridge_model_sha256": EXPECTED_RIDGE_MODEL_SHA256,
        "source_manifest_sha256": EXPECTED_R082_SOURCE_MANIFEST_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
    }
    for name, expected in expected_bundle.items():
        if selected_bundle.get(name) != expected:
            raise RuntimeError(f"R082 selected bundle differs for {name}")
    r082_summary = verify_r082_publication(args.r082_output)
    if (
        r082_summary.get("publication_manifest_sha256")
        != EXPECTED_R082_PUBLICATION_MANIFEST_SHA256
        or r082_summary.get("ready_sha256") != EXPECTED_R082_READY_SHA256
    ):
        raise RuntimeError("R082 publication verifier differs from frozen hashes")
    receipt = load_receipt(args.split_audit_receipt, EXPECTED_SPLIT_AUDIT_SHA256)
    split_binding = verify_split_receipt(
        receipt,
        project=PROJECT,
        split_manifest_sha256=EXPECTED_SPLIT_MANIFEST_SHA256,
        source_manifest_sha256=EXPECTED_R079_SOURCE_MANIFEST_SHA256,
    )
    return {"r082_publication": r082_summary, "split_audit": split_binding}


def _publish_scientific_evaluation_failure(
    *,
    args: argparse.Namespace,
    output: Path,
    identity: Mapping[str, Any],
    closure_start: Any,
    input_identities_start: Mapping[str, Any],
    job_id: str,
    start: float,
    device: torch.device,
    reason: str,
    system: str,
    error: BaseException | None,
    frozen_bindings: Mapping[str, Any],
    native_witness: Mapping[str, Any],
    outcome_metadata: Mapping[str, Any],
    selected_payload: Mapping[str, Any],
    ridge_payload: Mapping[str, Any],
    selected_gradients_finite: bool,
    scores_by_system: Mapping[str, Tensor] | None = None,
    metrics_by_system: Mapping[str, Mapping[str, Any]] | None = None,
    rows_by_system: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Publish a complete exit-2 packet for a frozen scientific conjunct failure."""

    scores = dict(scores_by_system or {})
    metrics = {name: dict(value) for name, value in (metrics_by_system or {}).items()}
    rows = {
        name: [dict(row) for row in values]
        for name, values in (rows_by_system or {}).items()
    }
    identity_passed = _identity_checks_passed(frozen_bindings, native_witness)
    checks = _scientific_failure_checks(
        metrics.get("pros"),
        identity_checks_passed=identity_passed,
        gradients_finite=selected_gradients_finite,
    )
    failure = {
        "protocol": FALSIFIER_PROTOCOL,
        "scientific_status": "FAIL",
        "reason": reason,
        "system": system,
        "error_type": None if error is None else type(error).__name__,
        "error_message": None if error is None else str(error),
        "score_diagnostics": _score_diagnostics(scores),
        "partial_metric_systems": sorted(metrics),
        "partial_record_systems": sorted(rows),
        "normal_adjudication_completed": False,
        "unavailable_artifacts": [
            "full saved-record replay",
            "complete domain metrics",
            "score quantiles",
            "prompt-cluster bootstrap intervals",
        ],
    }
    _write_json(output / "scientific_failure.json", failure)
    if scores:
        _save_torch(
            output / "failure_scores.pt",
            {
                name: value.detach().to(device="cpu").contiguous()
                for name, value in sorted(scores.items())
            },
        )
    if rows:
        _save_torch(output / "partial_records.pt", rows)
    _write_json(
        output / "gate_receipt.json",
        {
            "protocol": FALSIFIER_PROTOCOL,
            "scientific_status": "FAIL",
            "all_conjunctive": True,
            "checks": checks,
            "thresholds": _gate_thresholds(),
            "early_scientific_failure": failure,
        },
    )

    closure_end = verify_source_manifest(
        PROJECT,
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    if closure_end != closure_start:
        raise RuntimeError("source closure changed during R083 scientific failure")
    input_identities_end = _capture_input_identities(args)
    if input_identities_end != input_identities_start:
        raise RuntimeError("R083 frozen input identities changed")
    relative_artifacts = {
        "falsifier_metadata": "falsifier_outcomes/metadata.json",
        "falsifier_records": "falsifier_outcomes/records.pt",
        "scientific_failure": "scientific_failure.json",
        "gate_receipt": "gate_receipt.json",
    }
    if scores:
        relative_artifacts["failure_scores"] = "failure_scores.pt"
    if rows:
        relative_artifacts["partial_records"] = "partial_records.pt"
    output_artifacts = {
        name: _file_identity(output / relative, name)
        for name, relative in relative_artifacts.items()
    }
    report = {
        "protocol": FALSIFIER_PROTOCOL,
        "evidence_tier": "one_shot_producer_oos_falsifier",
        "scientific_status": "FAIL",
        "job_id": job_id,
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "seconds": time.perf_counter() - start,
        "configuration": _run_configuration(),
        "gate_checks": checks,
        "pros": metrics.get("pros"),
        "comparators": {
            name: value for name, value in metrics.items() if name != "pros"
        },
        "saved_record_replay": None,
        "scientific_failure": failure,
        "frozen_bindings": dict(frozen_bindings),
        "native_direct_witness": dict(native_witness),
        "outcome_metadata": dict(outcome_metadata),
        "selected_checkpoint": {
            "sha256": EXPECTED_SELECTED_CHECKPOINT_SHA256,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "gradients_finite": selected_gradients_finite,
            "payload_protocol": selected_payload["protocol"],
            "pass": selected_payload["pass"],
            "completed_updates": selected_payload["completed_updates"],
        },
        "ridge": {
            "sha256": EXPECTED_RIDGE_MODEL_SHA256,
            "protocol": ridge_payload["protocol"],
            "feature_dimension": ridge_payload["feature_dimension"],
            "ridge": ridge_payload["ridge"],
        },
        "output_artifacts": output_artifacts,
        "input_identities_start": dict(input_identities_start),
        "input_identities_end": input_identities_end,
        "source_closure_start": closure_start.summary(),
        "source_closure_end": closure_end.summary(),
        "limitations": [
            "normal adjudication stopped at a preregistered scientific conjunct failure",
            "the frozen route fails without substituting, clipping, or imputing scores",
            "no threshold, refit, checkpoint selection, seed, calibration, validation, reserved, or formal surface exists",
        ],
    }
    _write_json(output / "metrics.json", report)
    binding = {
        "identity": dict(identity),
        "scientific_status": "FAIL",
        "input_identities_end": input_identities_end,
        "source_closure_end": closure_end.summary(),
    }
    publication = commit_publication(output, binding)
    if verify_r083_publication(output, expected_binding=binding) != publication:
        raise RuntimeError("R083 scientific-failure publication replay differs")
    return {**report, "publication": publication, "output": str(output)}, False


def run_falsifier(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("R083 one-shot materialization requires CUDA")
    assert_stage_splits(FALSIFIER_SPLIT, {FALSIFIER_SPLIT})
    closure_start = verify_source_manifest(
        PROJECT,
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    job_id = os.environ.get("SLURM_JOB_ID")
    if not isinstance(job_id, str) or not job_id.isdecimal():
        raise RuntimeError("R083 requires a decimal SLURM_JOB_ID")
    _validate_exact_paths(args, job_id)
    entrypoint_sha256 = sha256_file(Path(__file__).resolve())
    identity = publication_identity(
        args.output,
        job_id=job_id,
        purpose=FALSIFIER_PURPOSE,
        entrypoint_path=ENTRYPOINT_RELATIVE,
        entrypoint_sha256=entrypoint_sha256,
        wrapper_sha256=args.expected_wrapper_sha256,
        source_manifest_sha256=args.expected_source_manifest_sha256,
    )
    output = reserve_publication_directory(args.output, identity)
    snapshot_source_closure(PROJECT, closure_start, output / "source_snapshot")
    input_identities_start = _capture_input_identities(args)
    frozen_bindings = _validate_frozen_inputs(args)

    device = torch.device("cuda:0")
    selected_model, selected_payload = _load_selected_sidecar(args.r082_output, device)
    ridge_model, ridge_payload = _load_frozen_ridge(args.r082_output)
    selected_gradients_finite = _all_tensors_finite(selected_payload.get("optimizer"))
    if not selected_gradients_finite:
        raise RuntimeError("R082 selected optimizer contains nonfinite tensors")

    # The one-shot data opening starts here.  No score, threshold, model,
    # checkpoint, comparator, seed, or hyperparameter can change after this line.
    start = time.perf_counter()
    observed_split_hash = sha256_file(args.split_manifest)
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    collection, metadata_sha256 = load_phase3_source(args.data)
    prompt_sets, exclusion_hashes, exclusion_sources = load_exclusions(args.exclusion)
    provenance = split_provenance(
        data=args.data,
        metadata_sha256=metadata_sha256,
        exclusion_sources=exclusion_sources,
        source_closure=closure_start.summary(),
    )
    identity_records = _identity_records(collection.records)
    verify_phase3_split_manifest(
        split_manifest,
        identity_records,
        canonical_metadata_sha256=metadata_sha256,
        exclusion_prompt_sets=prompt_sets,
        exclusion_manifest_sha256=exclusion_hashes,
        expected_prompts=PHASE3_PROMPTS,
        expected_blocks=PHASE3_BLOCKS,
        provenance=provenance,
    )
    assignments = split_assignment_map(split_manifest)
    producer, target_embedding, direct_config = load_frozen_direct(
        args.direct_run,
        args.target,
        collection.metadata,
        device=device,
    )
    raw_records, native_witness = materialize_falsifier_outcome_records(
        collection.records,
        assignments,
        split=FALSIFIER_SPLIT,
        producer=producer,
        target_embedding=target_embedding,
        device=device,
        batch_size=MATERIALIZATION_BATCH_SIZE,
        candidate_k=int(direct_config["candidate_k"]),
    )
    expected_block_keys = {
        BlockKey(
            str(row["sample_id"]),
            int(row["anchor_offset"]),
            int(row["context_length"]),
        )
        for row in split_manifest["blocks"]
        if row["split"] == FALSIFIER_SPLIT
    }
    observed_block_keys = {block_key_from_mapping(row) for row in raw_records}
    if observed_block_keys != expected_block_keys:
        raise RuntimeError("R083 outcome block identities differ from split manifest")
    prompt_domains: dict[str, str] = {}
    for row in raw_records:
        sample_id = str(row["sample_id"])
        domain = str(row["domain"])
        previous = prompt_domains.setdefault(sample_id, domain)
        if previous != domain:
            raise RuntimeError("R083 prompt appears under multiple domains")
    if len(prompt_domains) != FALSIFIER_PROMPTS or Counter(prompt_domains.values()) != Counter(
        FALSIFIER_DOMAIN_PROMPTS
    ):
        raise RuntimeError("R083 falsifier prompt/domain cardinality differs")

    closure_after_materialization = verify_source_manifest(
        PROJECT,
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    if closure_after_materialization != closure_start:
        raise RuntimeError("source closure changed during R083 materialization")

    outcome_metadata = _write_outcome_bundle(
        output / "falsifier_outcomes",
        raw_records,
        split_manifest_sha256=observed_split_hash,
        provenance={
            "job_id": job_id,
            "hostname": platform.node(),
            "device": torch.cuda.get_device_name(device),
            "source_data": str(args.data.resolve()),
            "source_metadata_sha256": metadata_sha256,
            "source_split": SOURCE_SPLIT,
            "direct_run": str(args.direct_run.resolve()),
            "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
            "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
            "target": str(args.target.resolve()),
            "target_identity_verified": True,
            "native_witness": native_witness,
            "numeric_policy": numeric_policy_receipt(),
            "source_closure_start": closure_start.summary(),
            "source_closure_end": closure_after_materialization.summary(),
        },
    )
    records = _canonical_order(raw_records)
    cpu_tensors = _training_tensors(
        records,
        torch.device("cpu"),
        expected_records=len(records),
        expected_prompts=FALSIFIER_PROMPTS,
    )
    gpu_tensors = _to_device(cpu_tensors, device)

    def publish_scientific_failure(
        *,
        reason: str,
        system: str,
        error: BaseException | None,
        scores_by_system: Mapping[str, Tensor] | None = None,
        metrics_by_system: Mapping[str, Mapping[str, Any]] | None = None,
        rows_by_system: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        return _publish_scientific_evaluation_failure(
            args=args,
            output=output,
            identity=identity,
            closure_start=closure_start,
            input_identities_start=input_identities_start,
            job_id=job_id,
            start=start,
            device=device,
            reason=reason,
            system=system,
            error=error,
            frozen_bindings=frozen_bindings,
            native_witness=native_witness,
            outcome_metadata=outcome_metadata,
            selected_payload=selected_payload,
            ridge_payload=ridge_payload,
            selected_gradients_finite=selected_gradients_finite,
            scores_by_system=scores_by_system,
            metrics_by_system=metrics_by_system,
            rows_by_system=rows_by_system,
        )

    try:
        pros_metrics, pros_rows, pros_scores = evaluate_model(
            selected_model,
            records,
            gpu_tensors,
            gradients_finite=selected_gradients_finite,
        )
    except (FloatingPointError, RuntimeError, ValueError) as error:
        reason = _scientific_evaluation_failure(error, system="pros")
        if reason is None:
            raise
        return publish_scientific_failure(
            reason=reason,
            system="pros",
            error=error,
        )
    try:
        scalar_features = scalar_comparator_features(
            cpu_tensors["features"], cpu_tensors["changes"]
        ).to(torch.float64)
    except ValueError as error:
        reason = _scientific_evaluation_failure(error, system="ridge")
        if reason is None:
            raise
        return publish_scientific_failure(
            reason=reason,
            system="ridge",
            error=error,
            scores_by_system={"pros": pros_scores.detach().cpu()},
            metrics_by_system={"pros": pros_metrics},
            rows_by_system={"pros": pros_rows},
        )
    scores_by_system: dict[str, Tensor] = {
        "pros": pros_scores.detach().cpu(),
        "ridge": ridge_model.predict(scalar_features),
        "always_keep": torch.zeros(len(records), dtype=torch.float64),
        "always_direct": torch.ones(len(records), dtype=torch.float64),
    }
    rows_by_system: dict[str, list[dict[str, Any]]] = {"pros": pros_rows}
    metrics_by_system: dict[str, dict[str, Any]] = {"pros": pros_metrics}
    for name in ("ridge", "always_keep", "always_direct"):
        if not bool(torch.isfinite(scores_by_system[name]).all()):
            return publish_scientific_failure(
                reason=f"nonfinite_{name}_values",
                system=name,
                error=None,
                scores_by_system=scores_by_system,
                metrics_by_system=metrics_by_system,
                rows_by_system=rows_by_system,
            )
        try:
            metrics, rows = evaluate_scores(
                records,
                cpu_tensors,
                scores_by_system[name],
                values_finite=True,
                gradients_finite=True,
                verify_tensor_loss=True,
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            reason = _scientific_evaluation_failure(error, system=name)
            if reason is None:
                raise
            return publish_scientific_failure(
                reason=reason,
                system=name,
                error=error,
                scores_by_system=scores_by_system,
                metrics_by_system=metrics_by_system,
                rows_by_system=rows_by_system,
            )
        metrics_by_system[name] = metrics
        rows_by_system[name] = rows
    replay = _saved_record_replay(rows_by_system)
    for name in metrics_by_system:
        expected = {
            key: value
            for key, value in metrics_by_system[name].items()
            if key not in {"values_finite", "gradients_finite"}
        }
        if replay[name] != expected:
            raise RuntimeError(f"R083 independent saved-record replay differs for {name}")

    comparator_metrics = {
        name: metrics_by_system[name]
        for name in ("ridge", "always_keep", "always_direct")
    }
    domain_metrics = {
        name: domain_slice_metrics(
            records,
            scores,
            values_finite=bool(torch.isfinite(scores).all()),
            gradients_finite=True if name != "pros" else selected_gradients_finite,
        )
        for name, scores in scores_by_system.items()
    }
    gains = [float(row["normalized_gain"]) for row in records]
    quantiles = {
        "protocol": "pros-gate-exact-linear-quantiles-v1",
        "levels": list(QUANTILE_LEVELS),
        "normalized_gain": _quantile_report(gains),
        "scores": {
            name: _quantile_report(scores.detach().cpu().double().tolist())
            for name, scores in scores_by_system.items()
        },
    }
    bootstrap = _bootstrap_report(
        records,
        scores_by_system,
        prompt_set_sha256=str(outcome_metadata["summary"]["prompt_set_sha256"]),
    )
    identity_checks_passed = _identity_checks_passed(
        frozen_bindings, native_witness
    )
    checks = _gate_checks(
        pros_metrics,
        comparator_metrics,
        identity_checks_passed=identity_checks_passed,
        gradients_finite=selected_gradients_finite,
    )
    scientific_pass = all(checks.values())

    _save_torch(output / "pros_records.pt", pros_rows)
    _save_torch(output / "comparator_records.pt", rows_by_system)
    _write_json(output / "domain_metrics.json", domain_metrics)
    _write_json(output / "quantiles.json", quantiles)
    _write_json(output / "bootstrap.json", bootstrap)
    _write_json(
        output / "gate_receipt.json",
        {
            "protocol": FALSIFIER_PROTOCOL,
            "scientific_status": "PASS" if scientific_pass else "FAIL",
            "all_conjunctive": True,
            "checks": checks,
            "thresholds": _gate_thresholds(),
        },
    )

    closure_end = verify_source_manifest(
        PROJECT,
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    if closure_end != closure_start:
        raise RuntimeError("source closure changed during R083")
    input_identities_end = _capture_input_identities(args)
    if input_identities_end != input_identities_start:
        raise RuntimeError("R083 frozen input identities changed")
    output_artifacts = {
        name: _file_identity(output / relative, name)
        for name, relative in {
            "falsifier_metadata": "falsifier_outcomes/metadata.json",
            "falsifier_records": "falsifier_outcomes/records.pt",
            "pros_records": "pros_records.pt",
            "comparator_records": "comparator_records.pt",
            "domain_metrics": "domain_metrics.json",
            "quantiles": "quantiles.json",
            "bootstrap": "bootstrap.json",
            "gate_receipt": "gate_receipt.json",
        }.items()
    }
    report = {
        "protocol": FALSIFIER_PROTOCOL,
        "evidence_tier": "one_shot_producer_oos_falsifier",
        "scientific_status": "PASS" if scientific_pass else "FAIL",
        "job_id": job_id,
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "seconds": time.perf_counter() - start,
        "configuration": _run_configuration(),
        "gate_checks": checks,
        "pros": pros_metrics,
        "comparators": comparator_metrics,
        "saved_record_replay": replay,
        "frozen_bindings": frozen_bindings,
        "native_direct_witness": native_witness,
        "outcome_metadata": outcome_metadata,
        "selected_checkpoint": {
            "sha256": EXPECTED_SELECTED_CHECKPOINT_SHA256,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "gradients_finite": selected_gradients_finite,
            "payload_protocol": selected_payload["protocol"],
            "pass": selected_payload["pass"],
            "completed_updates": selected_payload["completed_updates"],
        },
        "ridge": {
            "sha256": EXPECTED_RIDGE_MODEL_SHA256,
            "protocol": ridge_payload["protocol"],
            "feature_dimension": ridge_payload["feature_dimension"],
            "ridge": ridge_payload["ridge"],
        },
        "output_artifacts": output_artifacts,
        "input_identities_start": input_identities_start,
        "input_identities_end": input_identities_end,
        "source_closure_start": closure_start.summary(),
        "source_closure_end": closure_end.summary(),
        "limitations": [
            "point estimates, not bootstrap intervals, bind the gate",
            "a falsifier PASS is producer-OOS feasibility evidence, not validation or formal-test evidence",
            "no threshold, refit, checkpoint selection, seed, calibration, validation, reserved, or formal surface exists",
        ],
    }
    _write_json(output / "metrics.json", report)
    binding = {
        "identity": identity,
        "scientific_status": report["scientific_status"],
        "input_identities_end": input_identities_end,
        "source_closure_end": closure_end.summary(),
    }
    publication = commit_publication(output, binding)
    if verify_r083_publication(output, expected_binding=binding) != publication:
        raise RuntimeError("R083 publication replay differs")
    return {**report, "publication": publication, "output": str(output)}, scientific_pass


def main() -> None:
    report, scientific_pass = run_falsifier(parse_args())
    print(
        json.dumps(
            {
                "output": report["output"],
                "scientific_status": report["scientific_status"],
                "gate_checks": report["gate_checks"],
                "pros": report["pros"],
                "comparators": report["comparators"],
                "publication": report["publication"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    raise SystemExit(0 if scientific_pass else 2)


if __name__ == "__main__":
    main()
