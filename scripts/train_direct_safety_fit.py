#!/usr/bin/env python3
"""Run the single reviewed R082 PROS-Gate fit/checkpoint experiment.

Only the physically isolated fit bundle can update parameters.  The scalar
ridge comparator and deterministic fit orders are frozen before the
checkpoint bundle is opened.  The checkpoint split can select one of the 26
saved sidecar states but cannot update the model or comparator.  This entry
point deliberately has no later-stage data surface.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Mapping, Sequence
import uuid

import torch
from torch import Tensor

from sph.direct_safety_artifacts import load_outcome_bundle, sha256_file
from sph.direct_safety_gate import (
    DirectSafetySidecar,
    gain_weighted_unit_hinge,
    prompt_balanced_example_weights,
    scalar_comparator_features,
)
from sph.direct_safety_protocol import (
    BlockKey,
    RIDGE_COEFFICIENT,
    RIDGE_FEATURE_DIMENSION,
    SavedGateRecord,
    WeightedRidgeModel,
    assert_disjoint_prompt_sets,
    complete_pass_schedule,
    fit_checkpoint_selection_key,
    fit_weighted_ridge,
    ordered_block_keys,
    ordered_block_keys_sha256,
    reconstruct_saved_gate_evaluation,
    selected_fit_checkpoint,
)
from sph.source_closure import snapshot_source_closure, verify_source_manifest

try:
    from verify_pros_gate_receipt import (
        load_capacity_adjudication_receipt,
        load_receipt,
        verify_capacity_adjudication_receipt,
        verify_outcomes_receipt,
    )
except ModuleNotFoundError:  # Imported as ``scripts.*`` in CPU tests.
    from scripts.verify_pros_gate_receipt import (
        load_capacity_adjudication_receipt,
        load_receipt,
        verify_capacity_adjudication_receipt,
        verify_outcomes_receipt,
    )


PROJECT = Path(__file__).resolve().parents[1]
FIT_TRAINING_PROTOCOL = "pros-gate-fit-checkpoint-training-v1"
FIT_ORDER_MANIFEST_PROTOCOL = "pros-gate-fit-orders-v1"
RIDGE_FREEZE_PROTOCOL = "pros-gate-fit-ridge-freeze-v1"
PUBLICATION_PROTOCOL = "pros-gate-r082-directory-commit-v1"
PUBLICATION_MANIFEST_NAME = "PUBLICATION_MANIFEST.json"
PUBLICATION_READY_NAME = "READY.json"
PUBLICATION_RESERVATION_NAME = "RESERVATION.json"
FIT_RECORDS = 12_686
FIT_PROMPTS = 1_587
CHECKPOINT_RECORDS = 1_600
CHECKPOINT_PROMPTS = 200
BATCH_SIZE = 64
MAX_UPDATES = 5_120
STEPS_PER_PASS = 199
PASSES = 25
TOTAL_UPDATES = 4_975
WARMUP_UPDATES = 199
FINAL_BATCH_SIZE = 14
PEAK_LEARNING_RATE = 6e-4
BETAS = (0.9, 0.999)
EPSILON = 1e-8
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 1.0
INITIALIZATION_SEED = 0
EXPECTED_CANONICAL_METADATA_SHA256 = (
    "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320"
)
EXPECTED_DIRECT_CHECKPOINT_SHA256 = (
    "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e"
)
EXPECTED_DIRECT_METRICS_SHA256 = (
    "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "7a572670867bbf6f811aad58c7ed1365f179083cda8187f5402221d554d1f1c0"
)
EXPECTED_FIT_METADATA_SHA256 = (
    "061069ed644b7fd700d7b65586622c02ef878c611ab6a549968f78bba8425f98"
)
EXPECTED_FIT_RECORDS_SHA256 = (
    "645007ec2665e141813b09e4bd1e35c33337b4b32e27655ae088c52c89fbcc6b"
)
EXPECTED_CHECKPOINT_METADATA_SHA256 = (
    "cdc879ef861608c6f26e004e4dd27826554f4c5631929e68f8dc57e8ea753047"
)
EXPECTED_CHECKPOINT_RECORDS_SHA256 = (
    "203cadf7141684b91b7c41d70fc0222098898ca2f9200825ccd60fc8ecbb93a2"
)
EXPECTED_OUTCOMES_AUDIT_SHA256 = (
    "29b0c83a26b3f7fa30830e596cb070176b5c2738f6c4d2728b0ebde7bc87f36d"
)
EXPECTED_OUTCOMES_SOURCE_MANIFEST_SHA256 = (
    "2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4"
)
EXPECTED_CAPACITY_ADJUDICATION_SHA256 = (
    "17ac807e5b599c45e414958786fade47d7b2a0e1fd5603c3726d531eb143a352"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint-bundle", type=Path, required=True)
    parser.add_argument("--outcomes-audit-receipt", type=Path, required=True)
    parser.add_argument(
        "--capacity-adjudication-receipt", type=Path, required=True
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _key(row: Mapping[str, Any]) -> BlockKey:
    key = BlockKey(
        row["sample_id"], row["anchor_offset"], row["context_length"]
    )
    key.serialize()
    return key


def _require_regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} is missing or is a symlink")


def _file_identity(path: Path, name: str) -> dict[str, int | str]:
    _require_regular_file(path, name)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _bundle_identities(root: Path, name: str) -> dict[str, dict[str, int | str]]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"{name} bundle is missing or is a symlink")
    return {
        "metadata.json": _file_identity(root / "metadata.json", f"{name} metadata"),
        "records.pt": _file_identity(root / "records.pt", f"{name} records"),
    }


def _assert_physically_isolated(left: Path, right: Path) -> None:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    if (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    ):
        raise RuntimeError("fit and checkpoint bundles are not physically isolated")


def _validate_outcome_metadata(
    metadata: Mapping[str, Any],
    *,
    split: str,
    records: int,
    prompts: int,
    expected_records_sha256: str,
) -> None:
    if metadata.get("split") != split:
        raise RuntimeError(f"{split} metadata has a different split")
    if metadata.get("split_manifest_sha256") != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise RuntimeError(f"{split} metadata names a different split manifest")
    if metadata.get("records_sha256") != expected_records_sha256:
        raise RuntimeError(f"{split} metadata names different records")
    summary = metadata.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError(f"{split} metadata lacks a summary")
    if summary.get("blocks") != records or summary.get("prompts") != prompts:
        raise RuntimeError(f"{split} metadata cardinality differs")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(f"{split} metadata lacks provenance")
    expected = {
        "source_metadata_sha256": EXPECTED_CANONICAL_METADATA_SHA256,
        "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
        "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
    }
    for name, value in expected.items():
        if provenance.get(name) != value:
            raise RuntimeError(f"{split} metadata differs for {name}")
    for boundary in ("source_closure_start", "source_closure_end"):
        closure = provenance.get(boundary)
        if not isinstance(closure, Mapping) or closure.get(
            "source_manifest_sha256"
        ) != EXPECTED_OUTCOMES_SOURCE_MANIFEST_SHA256:
            raise RuntimeError(f"{split} metadata has a different {boundary}")


def _training_tensors(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    expected_records: int,
    expected_prompts: int,
) -> dict[str, Any]:
    if len(records) != expected_records:
        raise ValueError("training tensor record cardinality differs")
    keys = [_key(row) for row in records]
    if len(set(keys)) != expected_records:
        raise ValueError("training tensor block keys must be unique")
    sample_ids = [key.sample_id for key in keys]
    if len(set(sample_ids)) != expected_prompts:
        raise ValueError("training tensor prompt cardinality differs")
    domains = [row.get("domain") for row in records]
    if any(not isinstance(value, str) or not value for value in domains):
        raise ValueError("training tensor domains must be nonempty strings")
    features = torch.stack(
        [row["position_features"].float() for row in records]
    ).to(device)
    paths = torch.stack([row["direct_path"].long() for row in records]).to(
        device
    )
    changes = torch.stack([row["change_mask"].bool() for row in records]).to(
        device
    )
    gains = torch.tensor(
        [float(row["normalized_gain"]) for row in records],
        dtype=torch.float32,
        device=device,
    )
    weights = prompt_balanced_example_weights(sample_ids, device=device)
    if features.shape != (expected_records, 15, 200):
        raise ValueError("position features differ from the Nx15x200 contract")
    if paths.shape != (expected_records, 15) or changes.shape != (
        expected_records,
        15,
    ):
        raise ValueError("Direct path tensors differ from the Nx15 contract")
    if not torch.equal(changes, paths.ne(0)):
        raise ValueError("change masks differ from Direct paths")
    if not bool(torch.isfinite(features).all()) or not bool(
        torch.isfinite(gains).all()
    ):
        raise ValueError("fit/checkpoint tensors must be finite")
    if not math.isclose(
        float(weights.double().mean()), 1.0, rel_tol=0.0, abs_tol=1e-7
    ):
        raise RuntimeError("prompt-balanced weights do not have block mean one")
    return {
        "keys": keys,
        "sample_ids": sample_ids,
        "domains": domains,
        "features": features,
        "paths": paths,
        "changes": changes,
        "gains": gains,
        "weights": weights,
        "by_key": {key: index for index, key in enumerate(keys)},
    }


def _to_device(tensors: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(tensors)
    for name in ("features", "paths", "changes", "gains", "weights"):
        result[name] = tensors[name].to(device)
    return result


def fit_learning_rate(update_index: int) -> float:
    """Binding warmup/cosine LR for the zero-based R082 update index."""

    if isinstance(update_index, bool) or not isinstance(update_index, int):
        raise ValueError("update index must be a non-boolean integer")
    if not 0 <= update_index < TOTAL_UPDATES:
        raise ValueError("update index is outside the 4,975-update schedule")
    completed = update_index + 1
    if completed <= WARMUP_UPDATES:
        return PEAK_LEARNING_RATE * completed / WARMUP_UPDATES
    progress = (completed - WARMUP_UPDATES) / (
        TOTAL_UPDATES - WARMUP_UPDATES
    )
    return PEAK_LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def fit_batch_sizes() -> tuple[int, ...]:
    """Return the exact production partition for one complete fit pass."""

    sizes = tuple(
        min(BATCH_SIZE, FIT_RECORDS - start)
        for start in range(0, FIT_RECORDS, BATCH_SIZE)
    )
    if (
        len(sizes) != STEPS_PER_PASS
        or sizes[:-1] != (BATCH_SIZE,) * (STEPS_PER_PASS - 1)
        or sizes[-1] != FINAL_BATCH_SIZE
        or sum(sizes) != FIT_RECORDS
    ):
        raise RuntimeError("R082 fit batch partition differs from the contract")
    return sizes


def fit_minibatch_objective(
    scores: Tensor,
    gains: Tensor,
    prompt_weights: Tensor,
    *,
    total_records: int,
    steps_per_pass: int,
) -> tuple[Tensor, Tensor]:
    """Return a step-scale loss and its exact full-risk contribution.

    A record has mass ``prompt_weight / total_records``.  Multiplying each
    batch contribution by the number of steps per complete pass makes the
    average frozen-model step gradient equal the declared prompt-balanced
    empirical risk, including the final short batch.
    """

    if (
        scores.ndim != 1
        or gains.shape != scores.shape
        or prompt_weights.shape != scores.shape
        or scores.numel() < 1
    ):
        raise ValueError("fit mini-batch values must have equal shape [B]")
    if total_records < scores.numel() or steps_per_pass < 1:
        raise ValueError("invalid fit risk cardinality")
    weighted_sum = gain_weighted_unit_hinge(
        scores,
        gains,
        example_weights=prompt_weights,
        reduction="sum",
    ).loss
    full_risk_contribution = weighted_sum / float(total_records)
    optimizer_loss = full_risk_contribution * float(steps_per_pass)
    return optimizer_loss, full_risk_contribution


def fit_training_step(
    model: DirectSafetySidecar,
    optimizer: torch.optim.Optimizer,
    tensors: Mapping[str, Any],
    indices: Sequence[int],
    *,
    update_index: int,
) -> dict[str, float | bool]:
    if not 1 <= len(indices) <= BATCH_SIZE:
        raise ValueError("fit batches must contain between 1 and 64 records")
    index = torch.tensor(
        indices, dtype=torch.int64, device=tensors["features"].device
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        tensors["features"][index],
        tensors["changes"][index],
        tensors["paths"][index],
    )
    loss, risk_contribution = fit_minibatch_objective(
        output.scores,
        tensors["gains"][index],
        tensors["weights"][index],
        total_records=FIT_RECORDS,
        steps_per_pass=STEPS_PER_PASS,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("fit optimizer loss is non-finite")
    loss.backward()
    gradients_finite = all(
        value.grad is None or bool(torch.isfinite(value.grad).all())
        for value in model.parameters()
    )
    if not gradients_finite:
        raise FloatingPointError("fit gradient is non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), GRADIENT_CLIP
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("fit gradient norm is non-finite")
    learning_rate = fit_learning_rate(update_index)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.step()
    if not all(bool(torch.isfinite(value).all()) for value in model.parameters()):
        raise FloatingPointError("fit parameter became non-finite")
    return {
        "optimizer_loss": float(loss.detach()),
        "full_risk_contribution": float(risk_contribution.detach()),
        "gradient_norm_before_clip": float(gradient_norm),
        "learning_rate": learning_rate,
        "gradients_finite": True,
    }


def _saved_gate_evaluation(
    records: Sequence[Mapping[str, Any]],
    scores: Tensor,
    *,
    values_finite: bool,
    gradients_finite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if scores.ndim != 1 or scores.numel() != len(records):
        raise ValueError("saved scores must align with records")
    cpu_scores = scores.detach().to(device="cpu")
    if not bool(torch.isfinite(cpu_scores).all()):
        raise FloatingPointError("saved gate scores are non-finite")
    saved_rows: list[dict[str, Any]] = []
    replay_rows: list[SavedGateRecord] = []
    for row, score in zip(records, cpu_scores, strict=True):
        key = _key(row)
        saved = {
            "sample_id": key.sample_id,
            "domain": row["domain"],
            "anchor_offset": int(key.anchor_offset),
            "context_length": int(key.context_length),
            "base_length": int(row["base_length"]),
            "direct_length": int(row["direct_length"]),
            "score": float(score),
            "base_first_token_correct": bool(row["base_first_token_correct"]),
            "direct_first_token_correct": bool(
                row["direct_first_token_correct"]
            ),
        }
        saved_rows.append(saved)
        replay_rows.append(
            SavedGateRecord(
                block_key=key,
                base_length=saved["base_length"],
                direct_length=saved["direct_length"],
                score=saved["score"],
                base_first_token_correct=saved["base_first_token_correct"],
                direct_first_token_correct=saved["direct_first_token_correct"],
            )
        )
    replay = reconstruct_saved_gate_evaluation(
        replay_rows, require_valid_recovery=False
    )
    metrics: dict[str, Any] = {
        **replay.metrics,
        "values_finite": bool(values_finite),
        "gradients_finite": bool(gradients_finite),
    }
    return metrics, saved_rows


def evaluate_scores(
    records: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Any],
    scores: Tensor,
    *,
    values_finite: bool,
    gradients_finite: bool,
    verify_tensor_loss: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics, saved_rows = _saved_gate_evaluation(
        records,
        scores,
        values_finite=values_finite,
        gradients_finite=gradients_finite,
    )
    if verify_tensor_loss:
        exact_loss = gain_weighted_unit_hinge(
            scores,
            tensors["gains"],
            example_weights=tensors["weights"],
            reduction="mean",
        ).loss
        replay_loss = float(metrics["prompt_weighted_gain_hinge"])
        if not math.isclose(
            float(exact_loss), replay_loss, rel_tol=0.0, abs_tol=1e-6
        ):
            raise RuntimeError("tensor and saved-record prompt losses differ")
        tensor_apply = tuple(
            bool(value) for value in scores.detach().gt(0).cpu().tolist()
        )
        replay_apply = tuple(row["score"] > 0.0 for row in saved_rows)
        if tensor_apply != replay_apply:
            raise RuntimeError("tensor and saved-record gate actions differ")
    if metrics["regret_bound_violation_count"] != 0:
        raise RuntimeError("saved-record evaluation violated the regret bound")
    return metrics, saved_rows


def evaluate_model(
    model: DirectSafetySidecar,
    records: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Any],
    *,
    gradients_finite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], Tensor]:
    model.eval()
    with torch.inference_mode():
        output = model(
            tensors["features"], tensors["changes"], tensors["paths"]
        )
    values_finite = bool(torch.isfinite(output.scores).all()) and all(
        bool(torch.isfinite(value).all()) for value in model.parameters()
    )
    metrics, rows = evaluate_scores(
        records,
        tensors,
        output.scores,
        values_finite=values_finite,
        gradients_finite=gradients_finite,
        verify_tensor_loss=True,
    )
    return metrics, rows, output.scores.detach()


def domain_slice_metrics(
    records: Sequence[Mapping[str, Any]],
    scores: Tensor,
    *,
    values_finite: bool,
    gradients_finite: bool,
) -> dict[str, dict[str, Any]]:
    cpu_scores = scores.detach().cpu()
    domains = sorted({str(row["domain"]) for row in records})
    result: dict[str, dict[str, Any]] = {}
    for domain in domains:
        indices = [
            index for index, row in enumerate(records) if row["domain"] == domain
        ]
        subset = [records[index] for index in indices]
        metrics, _ = _saved_gate_evaluation(
            subset,
            cpu_scores[indices],
            values_finite=values_finite,
            gradients_finite=gradients_finite,
        )
        result[domain] = metrics
    return result


def build_fit_order_manifest(
    keys: Sequence[BlockKey],
    training_manifest_sha256: str,
    *,
    passes: int,
) -> dict[str, Any]:
    if not keys or len(set(keys)) != len(keys) or passes < 1:
        raise ValueError("fit order manifest requires unique keys and passes")
    orders: list[dict[str, Any]] = []
    for pass_index in range(passes):
        ordered = ordered_block_keys(
            keys,
            pass_index=pass_index,
            training_manifest_sha256=training_manifest_sha256,
        )
        orders.append(
            {
                "pass_index": pass_index,
                "ordered_block_keys_sha256": ordered_block_keys_sha256(ordered),
                "block_keys": [
                    {
                        "sample_id": key.sample_id,
                        "anchor_offset": int(key.anchor_offset),
                        "context_length": int(key.context_length),
                    }
                    for key in ordered
                ],
            }
        )
    return {
        "protocol": FIT_ORDER_MANIFEST_PROTOCOL,
        "training_manifest_sha256": training_manifest_sha256,
        "records": len(keys),
        "passes": passes,
        "orders": orders,
    }


def load_fit_order_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_training_manifest_sha256: str,
    expected_keys: Sequence[BlockKey],
    expected_passes: int,
) -> list[list[BlockKey]]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("fit order-manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != FIT_ORDER_MANIFEST_PROTOCOL:
        raise RuntimeError("fit order-manifest protocol differs")
    if manifest.get("training_manifest_sha256") != expected_training_manifest_sha256:
        raise RuntimeError("fit order manifest names a different fit artifact")
    expected_records = len(expected_keys)
    if manifest.get("records") != expected_records or manifest.get(
        "passes"
    ) != expected_passes:
        raise RuntimeError("fit order-manifest cardinality differs")
    rows = manifest.get("orders")
    if not isinstance(rows, list) or len(rows) != expected_passes:
        raise RuntimeError("fit order manifest lacks complete pass orders")
    expected_set = set(expected_keys)
    result: list[list[BlockKey]] = []
    for pass_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("pass_index") != pass_index:
            raise RuntimeError("fit order-manifest pass indices differ")
        serialized = row.get("block_keys")
        if not isinstance(serialized, list) or len(serialized) != expected_records:
            raise RuntimeError("fit order-manifest pass cardinality differs")
        ordered = [_key(item) for item in serialized]
        if len(set(ordered)) != expected_records or set(ordered) != expected_set:
            raise RuntimeError("fit order manifest is not a key permutation")
        if ordered_block_keys_sha256(ordered) != row.get(
            "ordered_block_keys_sha256"
        ):
            raise RuntimeError("fit pass-order semantic hash differs")
        result.append(ordered)
    return result


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def freeze_ridge_from_fit(
    records: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Any],
    *,
    output: Path,
    source_manifest_sha256: str,
) -> tuple[WeightedRidgeModel, dict[str, Any]]:
    """Fit, persist, hash, and reload the comparator before checkpoint access."""

    if tensors["features"].device.type != "cpu":
        raise ValueError("ridge fitting requires CPU fit tensors")
    sample_ids = [str(row["sample_id"]) for row in records]
    ridge_features = scalar_comparator_features(
        tensors["features"], tensors["changes"]
    ).to(torch.float64)
    ridge_targets = torch.tensor(
        [float(row["normalized_gain"]) for row in records],
        dtype=torch.float64,
    )
    ridge_weights = prompt_balanced_example_weights(
        sample_ids, dtype=torch.float64
    )
    model = fit_weighted_ridge(ridge_features, ridge_targets, ridge_weights)
    payload = {
        "protocol": RIDGE_FREEZE_PROTOCOL,
        "fit_metadata_sha256": EXPECTED_FIT_METADATA_SHA256,
        "fit_records_sha256": EXPECTED_FIT_RECORDS_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "source_manifest_sha256": source_manifest_sha256,
        "feature_dimension": RIDGE_FEATURE_DIMENSION,
        "ridge": RIDGE_COEFFICIENT,
        "target": "normalized_direct_minus_keep_gain",
        "weights": "float64_prompt_balanced",
        "decision": "apply_iff_float64_score_gt_zero",
        "constant_comparators": {
            "always_keep": {"score": 0.0, "decision": "KEEP"},
            "always_direct": {"score": 1.0, "decision": "APPLY_DIRECT"},
        },
        "state": {
            "feature_mean": model.feature_mean,
            "feature_scale": model.feature_scale,
            "constant_features": model.constant_features,
            "coefficients": model.coefficients,
            "intercept": model.intercept,
        },
    }
    state_path = output / "ridge_model.pt"
    torch.save(payload, state_path)
    _fsync_path(state_path)
    state_sha256 = sha256_file(state_path)
    loaded = torch.load(state_path, map_location="cpu", weights_only=False)
    if loaded.get("protocol") != RIDGE_FREEZE_PROTOCOL:
        raise RuntimeError("reloaded ridge protocol differs")
    state = loaded.get("state")
    if not isinstance(state, Mapping):
        raise RuntimeError("reloaded ridge state is missing")
    reloaded = WeightedRidgeModel(
        feature_mean=state["feature_mean"].double(),
        feature_scale=state["feature_scale"].double(),
        constant_features=state["constant_features"].bool(),
        coefficients=state["coefficients"].double(),
        intercept=state["intercept"].double(),
        ridge=float(loaded["ridge"]),
    )
    torch.testing.assert_close(
        reloaded.predict(ridge_features), model.predict(ridge_features),
        rtol=0.0, atol=0.0,
    )
    receipt = {
        "protocol": RIDGE_FREEZE_PROTOCOL,
        "status": "FROZEN_BEFORE_CHECKPOINT_LOAD",
        "fit_metadata_sha256": EXPECTED_FIT_METADATA_SHA256,
        "fit_records_sha256": EXPECTED_FIT_RECORDS_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "source_manifest_sha256": source_manifest_sha256,
        "feature_dimension": RIDGE_FEATURE_DIMENSION,
        "ridge": RIDGE_COEFFICIENT,
        "ridge_model_sha256": state_sha256,
    }
    receipt_path = output / "ridge_freeze_receipt.json"
    _write_json(receipt_path, receipt)
    return reloaded, {
        "ridge_model_sha256": state_sha256,
        "ridge_freeze_receipt_sha256": sha256_file(receipt_path),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: DirectSafetySidecar,
    optimizer: torch.optim.Optimizer,
    pass_index: int,
    completed_updates: int,
    fit_metrics: Mapping[str, Any],
    checkpoint_metrics: Mapping[str, Any],
    order_manifest_sha256: str,
    source_manifest_sha256: str,
    ridge_model_sha256: str,
) -> None:
    torch.save(
        {
            "protocol": FIT_TRAINING_PROTOCOL,
            "pass": pass_index,
            "completed_updates": completed_updates,
            "model": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "fit": dict(fit_metrics),
            "checkpoint": dict(checkpoint_metrics),
            "fit_metadata_sha256": EXPECTED_FIT_METADATA_SHA256,
            "checkpoint_metadata_sha256": EXPECTED_CHECKPOINT_METADATA_SHA256,
            "order_manifest_sha256": order_manifest_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "ridge_model_sha256": ridge_model_sha256,
        },
        path,
    )
    _fsync_path(path)


def _save_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    torch.save(list(records), path)
    _fsync_path(path)


def _history_row(
    *,
    pass_index: int,
    completed_updates: int,
    fit_metrics: Mapping[str, Any],
    checkpoint_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    key = fit_checkpoint_selection_key(checkpoint_metrics)
    return {
        "pass": pass_index,
        "completed_updates": completed_updates,
        "fit": dict(fit_metrics),
        "checkpoint": dict(checkpoint_metrics),
        "checkpoint_selection_eligible": key is not None,
        "checkpoint_selection_key": None if key is None else list(key),
    }


def _checkpoint_manifest(
    root: Path, history: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in history:
        pass_index = int(row["pass"])
        checkpoint = root / "checkpoints" / f"pass-{pass_index:03d}.pt"
        saved_records = (
            root / "checkpoint_records" / f"pass-{pass_index:03d}.pt"
        )
        result.append(
            {
                "pass": pass_index,
                "completed_updates": int(row["completed_updates"]),
                "checkpoint_path": f"checkpoints/pass-{pass_index:03d}.pt",
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint),
                "records_path": (
                    f"checkpoint_records/pass-{pass_index:03d}.pt"
                ),
                "records_bytes": saved_records.stat().st_size,
                "records_sha256": sha256_file(saved_records),
                "selection_eligible": row["checkpoint_selection_eligible"],
                "selection_key": row["checkpoint_selection_key"],
            }
        )
    return result


def _assert_input_identities_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    if dict(before) != dict(after):
        differing = sorted(
            name
            for name in set(before) | set(after)
            if before.get(name) != after.get(name)
        )
        raise RuntimeError(f"R082 frozen inputs changed: {differing}")


def capture_input_identities(
    *,
    source_manifest: Path,
    outcomes_audit_receipt: Path,
    capacity_adjudication_receipt: Path,
    fit_bundle: Path,
    checkpoint_bundle: Path | None,
) -> dict[str, Any]:
    """Hash every currently authorized input without following symlink files."""

    result: dict[str, Any] = {
        "source_manifest": _file_identity(
            source_manifest, "R082 source manifest"
        ),
        "outcomes_audit_receipt": _file_identity(
            outcomes_audit_receipt, "outcomes audit receipt"
        ),
        "capacity_adjudication_receipt": _file_identity(
            capacity_adjudication_receipt, "capacity adjudication receipt"
        ),
        "fit_bundle": _bundle_identities(fit_bundle, "fit"),
    }
    if checkpoint_bundle is not None:
        result["checkpoint_bundle"] = _bundle_identities(
            checkpoint_bundle, "checkpoint"
        )
    return result


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _fsync_directory(path: Path) -> None:
    descriptor = _open_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_canonical_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise RuntimeError(f"{name} is not a canonical lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(f"{name} is not hexadecimal") from error
    return value


def publication_identity(
    output: Path,
    *,
    job_id: str,
    purpose: str,
    wrapper_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    if not isinstance(job_id, str) or not job_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in job_id
    ):
        raise RuntimeError("R082 publication job identity is invalid")
    if purpose not in {"R082_FIT_CHECKPOINT", "FILESYSTEM_SMOKE"}:
        raise RuntimeError("R082 publication purpose differs")
    identity = {
        "job_id": job_id,
        "seed": 0,
        "purpose": purpose,
        "output": str(output.resolve()),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "wrapper_sha256": _require_canonical_sha256(
            "R082 wrapper hash", wrapper_sha256
        ),
        "source_manifest_sha256": _require_canonical_sha256(
            "R082 source manifest hash", source_manifest_sha256
        ),
    }
    if purpose == "R082_FIT_CHECKPOINT" and (
        not job_id.isdecimal()
        or output.name != "seed0"
        or output.parent.name != f"pros_gate_fit_{job_id}"
    ):
        raise RuntimeError("R082 output path does not match its job and seed")
    return identity


def _reservation_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": PUBLICATION_PROTOCOL,
        "state": "UNCOMMITTED_UNTIL_READY_JSON",
        "identity": dict(identity),
    }


def _reserve_publication_directory(
    output: Path, identity: Mapping[str, Any]
) -> Path:
    """Exclusively reserve a visible but uncommitted output directory."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise RuntimeError("R082 output parent is missing or is a symlink")
    parent_descriptor = _open_directory(output.parent)
    try:
        try:
            os.mkdir(output.name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite R082 run: {output}"
            ) from error
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    try:
        _write_json(
            output / PUBLICATION_RESERVATION_NAME,
            _reservation_payload(identity),
        )
        (output / PUBLICATION_RESERVATION_NAME).chmod(0o400)
        _fsync_path(output / PUBLICATION_RESERVATION_NAME)
        _fsync_directory(output)
    except Exception:
        # A failed reservation is evidence.  It must never be made to look
        # absent or complete by deleting the directory.
        raise
    return output


def _publication_tree(
    output: Path,
) -> tuple[list[str], list[dict[str, int | str]]]:
    """Return the exact symlink-free payload tree, excluding commit controls."""

    if output.is_symlink() or not output.is_dir():
        raise RuntimeError("R082 publication root is missing or is a symlink")
    directories: list[str] = []
    files: list[dict[str, int | str]] = []
    excluded = {PUBLICATION_MANIFEST_NAME, PUBLICATION_READY_NAME}
    for current, directory_names, file_names in os.walk(
        output, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("R082 publication contains a non-directory entry")
            directories.append(path.relative_to(output).as_posix())
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(output).as_posix()
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("R082 publication contains a non-regular file")
            if relative in excluded:
                continue
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    directories.sort()
    files.sort(key=lambda row: str(row["path"]))
    return directories, files


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    _require_regular_file(path, name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


def _publication_manifest(
    output: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    directories, files = _publication_tree(output)
    reservation = _load_json_object(
        output / PUBLICATION_RESERVATION_NAME, "R082 reservation"
    )
    if reservation != _reservation_payload(binding.get("identity", {})):
        raise RuntimeError("R082 reservation schema differs")
    return {
        "protocol": PUBLICATION_PROTOCOL,
        "binding": dict(binding),
        "directories": directories,
        "files": files,
        "payload_directory_count": len(directories),
        "payload_file_count": len(files),
    }


def _fsync_publication_tree(output: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest["files"]
    directories = manifest["directories"]
    if not isinstance(files, list) or not isinstance(directories, list):
        raise RuntimeError("R082 publication manifest tree schema differs")
    for row in files:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise RuntimeError("R082 publication manifest file schema differs")
        _fsync_path(output / row["path"])
    _fsync_path(output / PUBLICATION_MANIFEST_NAME)
    for relative in sorted(
        directories, key=lambda value: (str(value).count("/"), str(value)), reverse=True
    ):
        _fsync_directory(output / str(relative))
    _fsync_directory(output)


def _ready_payload(manifest_sha256: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": PUBLICATION_PROTOCOL,
        "status": "READY",
        "publication_manifest_sha256": manifest_sha256,
        "payload_directory_count": manifest["payload_directory_count"],
        "payload_file_count": manifest["payload_file_count"],
    }


def _ready_pending_path(output: Path) -> Path:
    return output.parent / f".{output.name}.READY.pending"


def _write_ready_pending(output: Path, payload: Mapping[str, Any]) -> Path:
    pending = _ready_pending_path(output)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(pending, flags, 0o400)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("failed to write R082 READY pending receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(output.parent)
    return pending


def _link_ready_no_replace(pending: Path, output: Path) -> None:
    """Commit READY through a GPFS-supported, directory-fd hard link."""

    parent_descriptor = _open_directory(output.parent)
    output_descriptor = _open_directory(output)
    try:
        try:
            os.link(
                pending.name,
                PUBLICATION_READY_NAME,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=output_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite R082 READY receipt: {output}"
            ) from error
        os.fsync(output_descriptor)
    finally:
        os.close(output_descriptor)
        os.close(parent_descriptor)


def verify_published_directory(
    output: Path, *, expected_binding: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Fail closed unless READY commits the exact current payload tree."""

    ready_path = output / PUBLICATION_READY_NAME
    manifest_path = output / PUBLICATION_MANIFEST_NAME
    ready = _load_json_object(ready_path, "R082 READY receipt")
    manifest = _load_json_object(manifest_path, "R082 publication manifest")
    if set(ready) != {
        "protocol",
        "status",
        "publication_manifest_sha256",
        "payload_directory_count",
        "payload_file_count",
    }:
        raise RuntimeError("R082 READY receipt fields differ")
    if set(manifest) != {
        "protocol",
        "binding",
        "directories",
        "files",
        "payload_directory_count",
        "payload_file_count",
    }:
        raise RuntimeError("R082 publication manifest fields differ")
    if ready.get("protocol") != PUBLICATION_PROTOCOL or ready.get("status") != "READY":
        raise RuntimeError("R082 READY receipt protocol or status differs")
    if manifest.get("protocol") != PUBLICATION_PROTOCOL:
        raise RuntimeError("R082 publication manifest protocol differs")
    binding = manifest.get("binding")
    if not isinstance(binding, Mapping) or set(binding) != {
        "identity",
        "scientific_status",
        "input_identities_end",
        "source_closure_end",
    }:
        raise RuntimeError("R082 publication binding schema differs")
    identity = binding.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "job_id",
        "seed",
        "purpose",
        "output",
        "trainer_sha256",
        "wrapper_sha256",
        "source_manifest_sha256",
    }:
        raise RuntimeError("R082 publication identity schema differs")
    if identity.get("output") != str(output.resolve()) or identity.get("seed") != 0:
        raise RuntimeError("R082 publication output or seed identity differs")
    for name in (
        "trainer_sha256",
        "wrapper_sha256",
        "source_manifest_sha256",
    ):
        _require_canonical_sha256(f"R082 publication {name}", identity.get(name))
    purpose = identity.get("purpose")
    job_id = identity.get("job_id")
    if purpose == "R082_FIT_CHECKPOINT":
        if (
            not isinstance(job_id, str)
            or not job_id.isdecimal()
            or output.name != "seed0"
            or output.parent.name != f"pros_gate_fit_{job_id}"
        ):
            raise RuntimeError("R082 committed output does not match job and seed")
    elif purpose != "FILESYSTEM_SMOKE":
        raise RuntimeError("R082 committed publication purpose differs")
    if expected_binding is not None and dict(binding) != dict(expected_binding):
        raise RuntimeError("R082 publication binding differs from the expected input")
    manifest_sha256 = sha256_file(manifest_path)
    if ready.get("publication_manifest_sha256") != manifest_sha256:
        raise RuntimeError("R082 READY receipt names a different manifest")
    observed_directories, observed_files = _publication_tree(output)
    if manifest.get("directories") != observed_directories or manifest.get(
        "files"
    ) != observed_files:
        raise RuntimeError("R082 payload tree differs from its committed manifest")
    if manifest.get("payload_directory_count") != len(observed_directories) or manifest.get(
        "payload_file_count"
    ) != len(observed_files):
        raise RuntimeError("R082 publication manifest cardinality differs")
    if ready.get("payload_directory_count") != len(observed_directories) or ready.get(
        "payload_file_count"
    ) != len(observed_files):
        raise RuntimeError("R082 READY receipt cardinality differs")
    reservation = _load_json_object(
        output / PUBLICATION_RESERVATION_NAME, "R082 reservation"
    )
    if reservation != _reservation_payload(identity):
        raise RuntimeError("R082 reservation changed after commit")
    metrics = _load_json_object(output / "metrics.json", "R082 metrics")
    for name in (
        "scientific_status",
        "input_identities_end",
        "source_closure_end",
    ):
        if metrics.get(name) != binding.get(name):
            raise RuntimeError(f"R082 publication binding differs from metrics: {name}")
    file_rows = {
        str(row["path"]): row
        for row in observed_files
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    source_snapshot_manifest = file_rows.get(
        "source_snapshot/SOURCE_MANIFEST.json"
    )
    snapshotted_trainer = file_rows.get(
        "source_snapshot/scripts/train_direct_safety_fit.py"
    )
    if (
        not isinstance(source_snapshot_manifest, Mapping)
        or source_snapshot_manifest.get("sha256")
        != identity.get("source_manifest_sha256")
        or not isinstance(snapshotted_trainer, Mapping)
        or snapshotted_trainer.get("sha256") != identity.get("trainer_sha256")
    ):
        raise RuntimeError("R082 publication source snapshot identity differs")
    source_manifest = _load_json_object(
        output / "source_snapshot/SOURCE_MANIFEST.json",
        "R082 snapshotted source manifest",
    )
    source_rows = source_manifest.get("files")
    if not isinstance(source_rows, list):
        raise RuntimeError("R082 snapshotted source manifest lacks file entries")
    trainer_rows = [
        row
        for row in source_rows
        if isinstance(row, Mapping)
        and row.get("path") == "scripts/train_direct_safety_fit.py"
    ]
    if len(trainer_rows) != 1 or trainer_rows[0].get("sha256") != identity.get(
        "trainer_sha256"
    ):
        raise RuntimeError("R082 source closure names a different trainer")
    return {
        "protocol": PUBLICATION_PROTOCOL,
        "status": "READY",
        "publication_manifest_sha256": manifest_sha256,
        "ready_sha256": sha256_file(ready_path),
        "payload_directory_count": len(observed_directories),
        "payload_file_count": len(observed_files),
    }


def _commit_publication(
    output: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Fsync, manifest, and atomically commit a reserved directory."""

    for control in (PUBLICATION_MANIFEST_NAME, PUBLICATION_READY_NAME):
        path = output / control
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace R082 control: {path}")
    manifest = _publication_manifest(output, binding)
    manifest_path = output / PUBLICATION_MANIFEST_NAME
    _write_json(manifest_path, manifest)
    if _publication_manifest(output, binding) != manifest:
        raise RuntimeError("R082 payload changed while constructing its manifest")
    _fsync_publication_tree(output, manifest)
    if _publication_manifest(output, binding) != manifest:
        raise RuntimeError("R082 payload changed while fsyncing its manifest")
    manifest_sha256 = sha256_file(manifest_path)
    pending = _write_ready_pending(
        output, _ready_payload(manifest_sha256, manifest)
    )
    if _publication_manifest(output, binding) != manifest:
        raise RuntimeError("R082 payload changed before READY commit")
    _link_ready_no_replace(pending, output)
    # READY is the final mutation inside output.  The parent-side pending hard
    # link is deliberately retained as durable publication evidence.
    return verify_published_directory(output, expected_binding=binding)


def publication_filesystem_smoke(
    parent: Path,
    *,
    job_id: str,
    wrapper_sha256: str,
    source_manifest: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Exercise the exact directory-commit primitives on the target filesystem."""

    parent.mkdir(parents=True, exist_ok=True)
    probe = parent / f".pros-gate-publication-smoke.{uuid.uuid4().hex}"
    _require_regular_file(source_manifest, "publication smoke source manifest")
    if sha256_file(source_manifest) != source_manifest_sha256:
        raise RuntimeError("publication smoke source manifest hash differs")
    identity = publication_identity(
        probe,
        job_id=job_id,
        purpose="FILESYSTEM_SMOKE",
        wrapper_sha256=wrapper_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    source_snapshot = probe / "source_snapshot"
    smoke_metrics = {
        "scientific_status": "NOT_APPLICABLE_FILESYSTEM_SMOKE",
        "input_identities_end": {},
        "source_closure_end": {
            "source_manifest_sha256": source_manifest_sha256
        },
    }
    binding = {
        "identity": identity,
        **smoke_metrics,
    }
    _reserve_publication_directory(probe, identity)
    source_snapshot.mkdir()
    (source_snapshot / "scripts").mkdir()
    shutil.copy2(
        Path(__file__).resolve(),
        source_snapshot / "scripts/train_direct_safety_fit.py",
    )
    _fsync_path(source_snapshot / "scripts/train_direct_safety_fit.py")
    manifest_copy = source_snapshot / "SOURCE_MANIFEST.json"
    shutil.copy2(source_manifest, manifest_copy)
    _fsync_path(manifest_copy)
    _write_json(probe / "metrics.json", binding)
    summary = _commit_publication(probe, binding)
    if verify_published_directory(probe, expected_binding=binding) != summary:
        raise RuntimeError("R082 publication smoke replay differs")
    return {
        **summary,
        "smoke": "PASS",
        "output": str(probe),
        "pending": str(_ready_pending_path(probe)),
    }


def _fit_publication_identity(
    args: argparse.Namespace, source_manifest_sha256: str
) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        raise RuntimeError("R082 fit publication requires SLURM_JOB_ID")
    return publication_identity(
        args.output,
        job_id=job_id,
        purpose="R082_FIT_CHECKPOINT",
        wrapper_sha256=args.expected_wrapper_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )


def _fit_publication_binding(
    report: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    binding = {
        "identity": dict(identity),
        "scientific_status": report.get("scientific_status"),
        "input_identities_end": report.get("input_identities_end"),
        "source_closure_end": report.get("source_closure_end"),
    }
    if (
        binding["scientific_status"] not in {"PASS", "FAIL"}
        or not isinstance(binding["input_identities_end"], Mapping)
        or not isinstance(binding["source_closure_end"], Mapping)
    ):
        raise RuntimeError("R082 fit report lacks publication binding fields")
    return binding


def run_fit(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    _assert_physically_isolated(args.fit_bundle, args.checkpoint_bundle)
    closure_start = verify_source_manifest(
        PROJECT,
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("reviewed R082 training requires CUDA")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("R082 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    schedule = complete_pass_schedule(
        FIT_RECORDS, batch_size=BATCH_SIZE, max_updates=MAX_UPDATES
    )
    if (
        schedule.steps_per_pass != STEPS_PER_PASS
        or schedule.passes != PASSES
        or schedule.total_steps != TOTAL_UPDATES
        or schedule.warmup_steps != WARMUP_UPDATES
    ):
        raise RuntimeError("R082 schedule differs from the frozen contract")

    outcomes_receipt = load_receipt(
        args.outcomes_audit_receipt, EXPECTED_OUTCOMES_AUDIT_SHA256
    )
    outcomes_binding = verify_outcomes_receipt(
        outcomes_receipt,
        split_manifest_sha256=EXPECTED_SPLIT_MANIFEST_SHA256,
        fit_metadata_sha256=EXPECTED_FIT_METADATA_SHA256,
        checkpoint_metadata_sha256=EXPECTED_CHECKPOINT_METADATA_SHA256,
        source_manifest_sha256=EXPECTED_OUTCOMES_SOURCE_MANIFEST_SHA256,
    )
    capacity_receipt = load_capacity_adjudication_receipt(
        args.capacity_adjudication_receipt,
        EXPECTED_CAPACITY_ADJUDICATION_SHA256,
    )
    capacity_binding = verify_capacity_adjudication_receipt(capacity_receipt)

    publication_binding_identity = _fit_publication_identity(
        args, closure_start.manifest_sha256
    )
    run_root = _reserve_publication_directory(
        args.output, publication_binding_identity
    )
    try:
        (run_root / "checkpoints").mkdir()
        (run_root / "checkpoint_records").mkdir()
        source_snapshot = run_root / "source_snapshot"
        snapshot_source_closure(PROJECT, closure_start, source_snapshot)

        input_identities = capture_input_identities(
            source_manifest=args.source_manifest,
            outcomes_audit_receipt=args.outcomes_audit_receipt,
            capacity_adjudication_receipt=args.capacity_adjudication_receipt,
            fit_bundle=args.fit_bundle,
            checkpoint_bundle=None,
        )
        fit_records, fit_metadata = load_outcome_bundle(
            args.fit_bundle,
            expected_split="fit",
            expected_metadata_sha256=EXPECTED_FIT_METADATA_SHA256,
        )
        _validate_outcome_metadata(
            fit_metadata,
            split="fit",
            records=FIT_RECORDS,
            prompts=FIT_PROMPTS,
            expected_records_sha256=EXPECTED_FIT_RECORDS_SHA256,
        )
        cpu = torch.device("cpu")
        fit_cpu = _training_tensors(
            fit_records,
            cpu,
            expected_records=FIT_RECORDS,
            expected_prompts=FIT_PROMPTS,
        )

        ridge_model, ridge_freeze = freeze_ridge_from_fit(
            fit_records,
            fit_cpu,
            output=run_root,
            source_manifest_sha256=closure_start.manifest_sha256,
        )
        ridge_frozen_identities = {
            "ridge_model.pt": _file_identity(
                run_root / "ridge_model.pt", "frozen ridge model"
            ),
            "ridge_freeze_receipt.json": _file_identity(
                run_root / "ridge_freeze_receipt.json",
                "ridge freeze receipt",
            ),
        }

        order_manifest_path = run_root / "order_manifest.json"
        _write_json(
            order_manifest_path,
            build_fit_order_manifest(
                fit_cpu["keys"], EXPECTED_FIT_METADATA_SHA256, passes=PASSES
            ),
        )
        order_manifest_sha256 = sha256_file(order_manifest_path)
        frozen_orders = load_fit_order_manifest(
            order_manifest_path,
            expected_sha256=order_manifest_sha256,
            expected_training_manifest_sha256=EXPECTED_FIT_METADATA_SHA256,
            expected_keys=fit_cpu["keys"],
            expected_passes=PASSES,
        )

        # This is intentionally the first checkpoint-bundle file access.
        input_identities["checkpoint_bundle"] = _bundle_identities(
            args.checkpoint_bundle, "checkpoint"
        )
        checkpoint_records, checkpoint_metadata = load_outcome_bundle(
            args.checkpoint_bundle,
            expected_split="checkpoint",
            expected_metadata_sha256=EXPECTED_CHECKPOINT_METADATA_SHA256,
        )
        _validate_outcome_metadata(
            checkpoint_metadata,
            split="checkpoint",
            records=CHECKPOINT_RECORDS,
            prompts=CHECKPOINT_PROMPTS,
            expected_records_sha256=EXPECTED_CHECKPOINT_RECORDS_SHA256,
        )
        checkpoint_cpu = _training_tensors(
            checkpoint_records,
            cpu,
            expected_records=CHECKPOINT_RECORDS,
            expected_prompts=CHECKPOINT_PROMPTS,
        )
        assert_disjoint_prompt_sets(
            {
                "fit": fit_cpu["sample_ids"],
                "checkpoint": checkpoint_cpu["sample_ids"],
            }
        )
        if set(fit_cpu["domains"]) != {"chat", "code", "math"} or set(
            checkpoint_cpu["domains"]
        ) != {"chat", "code", "math"}:
            raise RuntimeError("fit/checkpoint domain support differs")

        device = torch.device("cuda:0")
        fit_tensors = _to_device(fit_cpu, device)
        checkpoint_tensors = _to_device(checkpoint_cpu, device)
        torch.manual_seed(INITIALIZATION_SEED)
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        model = DirectSafetySidecar(
            initialization_seed=INITIALIZATION_SEED
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=PEAK_LEARNING_RATE,
            betas=BETAS,
            eps=EPSILON,
            weight_decay=WEIGHT_DECAY,
        )
        if any(module.__class__.__name__.startswith("Dropout") for module in model.modules()):
            raise RuntimeError("R082 sidecar unexpectedly contains dropout")

        start = time.perf_counter()
        history: list[dict[str, Any]] = []
        pass_diagnostics: list[dict[str, Any]] = []
        gradients_finite = True
        fit_metrics, _, fit_scores = evaluate_model(
            model, fit_records, fit_tensors, gradients_finite=gradients_finite
        )
        checkpoint_metrics, checkpoint_saved, checkpoint_scores = evaluate_model(
            model,
            checkpoint_records,
            checkpoint_tensors,
            gradients_finite=gradients_finite,
        )
        if bool(fit_scores.ne(0).any()) or bool(checkpoint_scores.ne(0).any()):
            raise RuntimeError("epoch-zero sidecar is not exact KEEP identity")
        history.append(
            _history_row(
                pass_index=0,
                completed_updates=0,
                fit_metrics=fit_metrics,
                checkpoint_metrics=checkpoint_metrics,
            )
        )
        _save_checkpoint(
            run_root / "checkpoints/pass-000.pt",
            model=model,
            optimizer=optimizer,
            pass_index=0,
            completed_updates=0,
            fit_metrics=fit_metrics,
            checkpoint_metrics=checkpoint_metrics,
            order_manifest_sha256=order_manifest_sha256,
            source_manifest_sha256=closure_start.manifest_sha256,
            ridge_model_sha256=ridge_freeze["ridge_model_sha256"],
        )
        _save_records(
            run_root / "checkpoint_records/pass-000.pt", checkpoint_saved
        )

        update_index = 0
        for pass_index in range(PASSES):
            ordered = frozen_orders[pass_index]
            batch_losses: list[float] = []
            risk_contributions: list[float] = []
            maximum_gradient_norm = 0.0
            batch_sizes: list[int] = []
            for start_index in range(0, FIT_RECORDS, BATCH_SIZE):
                batch_keys = ordered[start_index : start_index + BATCH_SIZE]
                indices = [fit_tensors["by_key"][key] for key in batch_keys]
                diagnostics = fit_training_step(
                    model,
                    optimizer,
                    fit_tensors,
                    indices,
                    update_index=update_index,
                )
                gradients_finite = gradients_finite and bool(
                    diagnostics["gradients_finite"]
                )
                batch_losses.append(float(diagnostics["optimizer_loss"]))
                risk_contributions.append(
                    float(diagnostics["full_risk_contribution"])
                )
                maximum_gradient_norm = max(
                    maximum_gradient_norm,
                    float(diagnostics["gradient_norm_before_clip"]),
                )
                batch_sizes.append(len(indices))
                update_index += 1
            if tuple(batch_sizes) != fit_batch_sizes():
                raise RuntimeError("R082 pass batching differs from the contract")
            fit_metrics, _, _ = evaluate_model(
                model,
                fit_records,
                fit_tensors,
                gradients_finite=gradients_finite,
            )
            checkpoint_metrics, checkpoint_saved, _ = evaluate_model(
                model,
                checkpoint_records,
                checkpoint_tensors,
                gradients_finite=gradients_finite,
            )
            row = _history_row(
                pass_index=pass_index + 1,
                completed_updates=update_index,
                fit_metrics=fit_metrics,
                checkpoint_metrics=checkpoint_metrics,
            )
            history.append(row)
            pass_diagnostics.append(
                {
                    "pass": pass_index + 1,
                    "batches": len(batch_sizes),
                    "last_batch_size": batch_sizes[-1],
                    "mean_optimizer_loss": sum(batch_losses) / len(batch_losses),
                    "sum_full_risk_contributions": sum(risk_contributions),
                    "maximum_gradient_norm_before_clip": maximum_gradient_norm,
                    "last_learning_rate": fit_learning_rate(update_index - 1),
                }
            )
            _save_checkpoint(
                run_root / f"checkpoints/pass-{pass_index + 1:03d}.pt",
                model=model,
                optimizer=optimizer,
                pass_index=pass_index + 1,
                completed_updates=update_index,
                fit_metrics=fit_metrics,
                checkpoint_metrics=checkpoint_metrics,
                order_manifest_sha256=order_manifest_sha256,
                source_manifest_sha256=closure_start.manifest_sha256,
                ridge_model_sha256=ridge_freeze["ridge_model_sha256"],
            )
            _save_records(
                run_root
                / f"checkpoint_records/pass-{pass_index + 1:03d}.pt",
                checkpoint_saved,
            )
            print(
                json.dumps(
                    {
                        "pass": pass_index + 1,
                        "updates": update_index,
                        "fit_hinge": fit_metrics[
                            "prompt_weighted_gain_hinge"
                        ],
                        "checkpoint_eal": checkpoint_metrics["method_eal"],
                        "checkpoint_recovery": checkpoint_metrics[
                            "oracle_recovery"
                        ],
                        "checkpoint_harmed": checkpoint_metrics[
                            "harmed_numerator"
                        ],
                        "selection_eligible": row[
                            "checkpoint_selection_eligible"
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if update_index != TOTAL_UPDATES or len(history) != PASSES + 1:
            raise RuntimeError("R082 did not complete exactly 25 whole passes")

        _write_json(run_root / "history.json", history)
        _write_json(run_root / "pass_diagnostics.json", pass_diagnostics)
        checkpoint_manifest = _checkpoint_manifest(run_root, history)
        _write_json(run_root / "checkpoint_manifest.json", checkpoint_manifest)

        try:
            selected_row = selected_fit_checkpoint(history)
        except RuntimeError as error:
            closure_end = verify_source_manifest(
                PROJECT,
                args.source_manifest,
                expected_manifest_sha256=args.expected_source_manifest_sha256,
            )
            if closure_end != closure_start:
                raise RuntimeError("source closure changed during R082") from error
            input_end = capture_input_identities(
                source_manifest=args.source_manifest,
                outcomes_audit_receipt=args.outcomes_audit_receipt,
                capacity_adjudication_receipt=(
                    args.capacity_adjudication_receipt
                ),
                fit_bundle=args.fit_bundle,
                checkpoint_bundle=args.checkpoint_bundle,
            )
            _assert_input_identities_unchanged(input_identities, input_end)
            report = {
                "protocol": FIT_TRAINING_PROTOCOL,
                "evidence_tier": "checkpoint_selection_only_no_final_claim",
                "scientific_status": "FAIL",
                "failure": str(error),
                "history_rows": len(history),
                "eligible_checkpoints": sum(
                    bool(row["checkpoint_selection_eligible"]) for row in history
                ),
                "source_closure_start": closure_start.summary(),
                "source_closure_end": closure_end.summary(),
                "input_identities_start": input_identities,
                "input_identities_end": input_end,
            }
            _write_json(run_root / "metrics.json", report)
            _commit_publication(
                run_root,
                _fit_publication_binding(report, publication_binding_identity),
            )
            return report, False

        selected_pass = int(selected_row["pass"])
        selected_checkpoint = (
            run_root / f"checkpoints/pass-{selected_pass:03d}.pt"
        )
        selected_payload = torch.load(
            selected_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(selected_payload["model"], strict=True)
        selected_fit_metrics, _, _ = evaluate_model(
            model,
            fit_records,
            fit_tensors,
            gradients_finite=gradients_finite,
        )
        selected_checkpoint_metrics, selected_saved, selected_scores = evaluate_model(
            model,
            checkpoint_records,
            checkpoint_tensors,
            gradients_finite=gradients_finite,
        )
        if selected_fit_metrics != selected_row["fit"]:
            raise RuntimeError("selected fit replay differs from history")
        if selected_checkpoint_metrics != selected_row["checkpoint"]:
            raise RuntimeError("selected checkpoint replay differs from history")
        if torch.load(
            run_root / f"checkpoint_records/pass-{selected_pass:03d}.pt",
            map_location="cpu",
            weights_only=False,
        ) != selected_saved:
            raise RuntimeError("selected saved-record replay differs")
        shutil.copy2(selected_checkpoint, run_root / "selected.pt")
        _fsync_path(run_root / "selected.pt")
        _save_records(run_root / "selected_checkpoint_records.pt", selected_saved)

        checkpoint_scalar_features = scalar_comparator_features(
            checkpoint_cpu["features"], checkpoint_cpu["changes"]
        ).to(torch.float64)
        comparator_scores = {
            "ridge": ridge_model.predict(checkpoint_scalar_features),
            "always_keep": torch.zeros(
                CHECKPOINT_RECORDS, dtype=torch.float64
            ),
            "always_direct": torch.ones(
                CHECKPOINT_RECORDS, dtype=torch.float64
            ),
        }
        comparator_metrics: dict[str, Any] = {}
        comparator_records: dict[str, Any] = {}
        comparator_domains: dict[str, Any] = {}
        for name, scores in comparator_scores.items():
            metrics, rows = evaluate_scores(
                checkpoint_records,
                checkpoint_cpu,
                scores,
                values_finite=bool(torch.isfinite(scores).all()),
                gradients_finite=True,
                verify_tensor_loss=False,
            )
            comparator_metrics[name] = metrics
            comparator_records[name] = rows
            comparator_domains[name] = domain_slice_metrics(
                checkpoint_records,
                scores,
                values_finite=True,
                gradients_finite=True,
            )
        _write_json(run_root / "comparator_metrics.json", comparator_metrics)
        _save_records(
            run_root / "comparator_checkpoint_records.pt",
            [
                {"comparator": name, "records": comparator_records[name]}
                for name in ("ridge", "always_keep", "always_direct")
            ],
        )
        selected_domains = domain_slice_metrics(
            checkpoint_records,
            selected_scores,
            values_finite=True,
            gradients_finite=gradients_finite,
        )
        _write_json(
            run_root / "domain_metrics.json",
            {"selected_sidecar": selected_domains, **comparator_domains},
        )

        current_ridge_identities = {
            "ridge_model.pt": _file_identity(
                run_root / "ridge_model.pt", "frozen ridge model"
            ),
            "ridge_freeze_receipt.json": _file_identity(
                run_root / "ridge_freeze_receipt.json",
                "ridge freeze receipt",
            ),
        }
        if current_ridge_identities != ridge_frozen_identities:
            raise RuntimeError("ridge comparator changed after checkpoint load")
        if sha256_file(order_manifest_path) != order_manifest_sha256:
            raise RuntimeError("fit order manifest changed during optimization")
        closure_end = verify_source_manifest(
            PROJECT,
            args.source_manifest,
            expected_manifest_sha256=args.expected_source_manifest_sha256,
        )
        if closure_end != closure_start:
            raise RuntimeError("source closure changed during R082")
        input_end = capture_input_identities(
            source_manifest=args.source_manifest,
            outcomes_audit_receipt=args.outcomes_audit_receipt,
            capacity_adjudication_receipt=args.capacity_adjudication_receipt,
            fit_bundle=args.fit_bundle,
            checkpoint_bundle=args.checkpoint_bundle,
        )
        _assert_input_identities_unchanged(input_identities, input_end)

        selected_key = fit_checkpoint_selection_key(
            selected_checkpoint_metrics
        )
        if selected_key is None:
            raise RuntimeError("selected checkpoint became recovery-ineligible")
        selected_bundle = {
            "protocol": FIT_TRAINING_PROTOCOL,
            "status": "FROZEN",
            "selected_pass": selected_pass,
            "selected_updates": int(selected_row["completed_updates"]),
            "selection_key": list(selected_key),
            "selection_rule": (
                "strict_lexicographic_(checkpoint_eal,-harmed,-hinge);_"
                "exact_ties_keep_earliest"
            ),
            "selected_checkpoint_sha256": sha256_file(
                run_root / "selected.pt"
            ),
            "selected_records_sha256": sha256_file(
                run_root / "selected_checkpoint_records.pt"
            ),
            **ridge_freeze,
            "fit_metadata_sha256": EXPECTED_FIT_METADATA_SHA256,
            "checkpoint_metadata_sha256": EXPECTED_CHECKPOINT_METADATA_SHA256,
            "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
            "source_manifest_sha256": closure_start.manifest_sha256,
            "order_manifest_sha256": order_manifest_sha256,
        }
        _write_json(run_root / "selected_bundle.json", selected_bundle)
        seconds = time.perf_counter() - start
        report = {
            "protocol": FIT_TRAINING_PROTOCOL,
            "evidence_tier": "checkpoint_selection_only_no_final_claim",
            "scientific_status": "PASS",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": platform.node(),
            "device": torch.cuda.get_device_name(device),
            "seconds": seconds,
            "bindings": {
                "outcomes": outcomes_binding,
                "capacity_adjudication": capacity_binding,
            },
            "configuration": {
                "fit_records": FIT_RECORDS,
                "fit_prompts": FIT_PROMPTS,
                "checkpoint_records": CHECKPOINT_RECORDS,
                "checkpoint_prompts": CHECKPOINT_PROMPTS,
                "batch_size": BATCH_SIZE,
                "steps_per_pass": STEPS_PER_PASS,
                "passes": PASSES,
                "maximum_updates": MAX_UPDATES,
                "total_updates": TOTAL_UPDATES,
                "warmup_updates": WARMUP_UPDATES,
                "peak_learning_rate": PEAK_LEARNING_RATE,
                "betas": list(BETAS),
                "epsilon": EPSILON,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip": GRADIENT_CLIP,
                "initialization_seed": INITIALIZATION_SEED,
                "loss": "prompt_balanced_gain_weighted_unit_hinge",
                "decision": "apply_iff_raw_score_gt_zero",
                "selection": (
                    "checkpoint_only_strict_lexicographic_"
                    "(eal,-harmed,-hinge)_earliest_tie"
                ),
                "ridge": {
                    "dtype": "float64",
                    "coefficient": RIDGE_COEFFICIENT,
                    "feature_dimension": RIDGE_FEATURE_DIMENSION,
                    "fit_only": True,
                },
            },
            "history_rows": len(history),
            "eligible_checkpoints": sum(
                bool(row["checkpoint_selection_eligible"]) for row in history
            ),
            "selected_is_epoch_zero": selected_pass == 0,
            "selected": selected_row,
            "selected_checkpoint_replay": selected_checkpoint_metrics,
            "comparators_on_checkpoint": comparator_metrics,
            "artifacts": {
                "selected_bundle_sha256": sha256_file(
                    run_root / "selected_bundle.json"
                ),
                "selected_checkpoint_sha256": sha256_file(
                    run_root / "selected.pt"
                ),
                "selected_records_sha256": sha256_file(
                    run_root / "selected_checkpoint_records.pt"
                ),
                "history_sha256": sha256_file(run_root / "history.json"),
                "pass_diagnostics_sha256": sha256_file(
                    run_root / "pass_diagnostics.json"
                ),
                "checkpoint_manifest_sha256": sha256_file(
                    run_root / "checkpoint_manifest.json"
                ),
                "order_manifest_sha256": order_manifest_sha256,
                "ridge_model_sha256": ridge_freeze[
                    "ridge_model_sha256"
                ],
                "ridge_freeze_receipt_sha256": ridge_freeze[
                    "ridge_freeze_receipt_sha256"
                ],
                "comparator_metrics_sha256": sha256_file(
                    run_root / "comparator_metrics.json"
                ),
                "comparator_records_sha256": sha256_file(
                    run_root / "comparator_checkpoint_records.pt"
                ),
                "domain_metrics_sha256": sha256_file(
                    run_root / "domain_metrics.json"
                ),
                "source_snapshot_manifest_sha256": sha256_file(
                    source_snapshot / "SOURCE_MANIFEST.json"
                ),
            },
            "input_identities_start": input_identities,
            "input_identities_end": input_end,
            "source_closure_start": closure_start.summary(),
            "source_closure_end": closure_end.summary(),
            "limitations": [
                "checkpoint selection is not falsifier or efficacy evidence",
                "fit behavior cannot select the checkpoint",
                "no later-stage data was available to this entry point",
            ],
        }
        _write_json(run_root / "metrics.json", report)
        _commit_publication(
            run_root,
            _fit_publication_binding(report, publication_binding_identity),
        )
        return report, True
    finally:
        # Missing READY means incomplete.  Every failure keeps the reserved
        # directory and all evidence written before the exception.
        pass


def main() -> None:
    args = parse_args()
    report, passed = run_fit(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scientific_status": report["scientific_status"],
                "selected_pass": report.get("selected", {}).get("pass"),
                "selected_checkpoint": report.get(
                    "selected_checkpoint_replay"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
