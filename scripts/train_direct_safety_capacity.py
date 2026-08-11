#!/usr/bin/env python3
"""Run the single preregistered 512-record PROS-Gate capacity probe.

This entry point has one data surface: a reviewed capacity bundle.  It cannot
open canonical, fit, checkpoint, falsifier, validation, target, or Direct
producer paths.  Capacity is a plumbing/memorization gate and is never claim
evidence.
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

from sph.direct_safety_artifacts import load_capacity_bundle, sha256_file
from sph.direct_safety_gate import (
    DirectSafetySidecar,
    gain_weighted_unit_hinge,
)
from sph.direct_safety_protocol import (
    BlockKey,
    SavedGateRecord,
    capacity_gate_passes,
    complete_pass_schedule,
    earliest_exact_minimum,
    ordered_block_keys,
    ordered_block_keys_sha256,
    reconstruct_saved_gate_evaluation,
    selected_capacity_checkpoint,
)
from sph.source_closure import snapshot_source_closure, verify_source_manifest


PROJECT = Path(__file__).resolve().parents[1]
CAPACITY_TRAINING_PROTOCOL = "pros-gate-capacity-training-v1"
CAPACITY_ORDER_MANIFEST_PROTOCOL = "pros-gate-capacity-orders-v1"
RECORDS = 512
BATCH_SIZE = 32
PASSES = 320
TOTAL_UPDATES = 5_120
WARMUP_UPDATES = 205
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-bundle", type=Path, required=True)
    parser.add_argument("--expected-capacity-metadata-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def capacity_learning_rate(update_index: int) -> float:
    """Binding warmup/cosine LR for zero-based optimizer update index."""

    if isinstance(update_index, bool) or not isinstance(update_index, int):
        raise ValueError("update index must be a non-boolean integer")
    if not 0 <= update_index < TOTAL_UPDATES:
        raise ValueError("update index is outside the 5,120-update schedule")
    completed = update_index + 1
    if completed <= WARMUP_UPDATES:
        return PEAK_LEARNING_RATE * completed / WARMUP_UPDATES
    progress = (completed - WARMUP_UPDATES) / (
        TOTAL_UPDATES - WARMUP_UPDATES
    )
    return PEAK_LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def _key(row: Mapping[str, Any]) -> BlockKey:
    key = BlockKey(
        row["sample_id"], row["anchor_offset"], row["context_length"]
    )
    key.serialize()
    return key


def _training_tensors(
    records: Sequence[Mapping[str, Any]], device: torch.device
) -> dict[str, Any]:
    if len(records) != RECORDS:
        raise ValueError("capacity training requires exactly 512 records")
    keys = [_key(row) for row in records]
    if len(set(keys)) != RECORDS:
        raise ValueError("capacity training block keys must be unique")
    if len({key.sample_id for key in keys}) != RECORDS:
        raise ValueError("capacity training records must be prompt unique")
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
    if features.shape != (512, 15, 200):
        raise ValueError("capacity position features differ from 512x15x200")
    if paths.shape != (512, 15) or changes.shape != (512, 15):
        raise ValueError("capacity path tensors differ from 512x15")
    if not torch.equal(changes, paths.ne(0)):
        raise ValueError("capacity change masks differ from Direct paths")
    if not bool(torch.isfinite(features).all()) or not bool(
        torch.isfinite(gains).all()
    ):
        raise ValueError("capacity training tensors must be finite")
    return {
        "keys": keys,
        "features": features,
        "paths": paths,
        "changes": changes,
        "gains": gains,
    }


def capacity_minibatch_objective(
    scores: Tensor, gains: Tensor
) -> tuple[Tensor, Tensor]:
    """Return SGD-scale loss and its exact full-risk mass contribution.

    Every record has frozen mass ``1/512``.  A batch contributes the sum of
    those masses to the full capacity risk; multiplying by ``512/B`` gives
    the standard unbiased batch-mean gradient used for the optimizer.
    """

    if scores.ndim != 1 or gains.shape != scores.shape or scores.numel() < 1:
        raise ValueError("capacity mini-batch scores/gains must have shape [B]")
    masses = torch.full_like(scores.float(), 1.0 / RECORDS)
    contribution = gain_weighted_unit_hinge(
        scores,
        gains,
        example_weights=masses,
        reduction="sum",
    ).loss
    optimizer_loss = contribution * (RECORDS / scores.numel())
    return optimizer_loss, contribution


def evaluate_capacity(
    model: DirectSafetySidecar,
    tensors: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    gradients_finite: bool,
) -> tuple[dict[str, float | int | bool], list[dict[str, Any]]]:
    model.eval()
    with torch.inference_mode():
        output = model(
            tensors["features"], tensors["changes"], tensors["paths"]
        )
        exact_loss = gain_weighted_unit_hinge(
            output.scores,
            tensors["gains"],
            example_weights=torch.full_like(output.scores, 1.0 / RECORDS),
            reduction="sum",
        )
    values_finite = bool(torch.isfinite(output.scores).all()) and bool(
        torch.isfinite(exact_loss.loss)
    ) and all(bool(torch.isfinite(value).all()) for value in model.parameters())
    saved_rows: list[dict[str, Any]] = []
    replay_rows: list[SavedGateRecord] = []
    for row, key, score in zip(
        records, tensors["keys"], output.scores.detach().cpu(), strict=True
    ):
        saved = {
            "sample_id": key.sample_id,
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
    replay = reconstruct_saved_gate_evaluation(replay_rows)
    replay_loss = float(replay.metrics["prompt_weighted_gain_hinge"])
    numeric_loss = float(exact_loss.loss)
    if not math.isclose(replay_loss, numeric_loss, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("tensor and saved-record capacity losses differ")
    metrics: dict[str, float | int | bool] = {
        **{
            name: value
            for name, value in replay.metrics.items()
            if value is not None
        },
        "prompt_weighted_loss": numeric_loss,
        "utility_optimal_count": int(
            replay.metrics["utility_optimal_numerator"]
        ),
        "values_finite": values_finite,
        "gradients_finite": gradients_finite,
    }
    return metrics, saved_rows


def _set_optimizer_lr(
    optimizer: torch.optim.Optimizer, update_index: int
) -> float:
    learning_rate = capacity_learning_rate(update_index)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def capacity_training_step(
    model: DirectSafetySidecar,
    optimizer: torch.optim.Optimizer,
    tensors: Mapping[str, Any],
    indices: Sequence[int],
    *,
    update_index: int,
) -> dict[str, float | bool]:
    if len(indices) != BATCH_SIZE:
        raise ValueError("capacity batches must contain exactly 32 records")
    index = torch.tensor(indices, dtype=torch.int64, device=tensors["features"].device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        tensors["features"][index],
        tensors["changes"][index],
        tensors["paths"][index],
    )
    loss, risk_contribution = capacity_minibatch_objective(
        output.scores, tensors["gains"][index]
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("capacity optimizer loss is non-finite")
    loss.backward()
    gradients_finite = all(
        value.grad is None or bool(torch.isfinite(value.grad).all())
        for value in model.parameters()
    )
    if not gradients_finite:
        raise FloatingPointError("capacity gradient is non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), GRADIENT_CLIP
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("capacity gradient norm is non-finite")
    learning_rate = _set_optimizer_lr(optimizer, update_index)
    optimizer.step()
    if not all(bool(torch.isfinite(value).all()) for value in model.parameters()):
        raise FloatingPointError("capacity parameter became non-finite")
    return {
        "optimizer_loss": float(loss.detach()),
        "full_risk_contribution": float(risk_contribution.detach()),
        "gradient_norm_before_clip": float(gradient_norm),
        "learning_rate": learning_rate,
        "gradients_finite": True,
    }


def build_capacity_order_manifest(
    keys: Sequence[BlockKey], training_manifest_sha256: str
) -> dict[str, Any]:
    if len(keys) != RECORDS or len(set(keys)) != RECORDS:
        raise ValueError("capacity order manifest requires 512 unique block keys")
    passes: list[dict[str, Any]] = []
    for pass_index in range(PASSES):
        ordered = ordered_block_keys(
            keys,
            pass_index=pass_index,
            training_manifest_sha256=training_manifest_sha256,
        )
        passes.append(
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
        "protocol": CAPACITY_ORDER_MANIFEST_PROTOCOL,
        "training_manifest_sha256": training_manifest_sha256,
        "records": RECORDS,
        "passes": PASSES,
        "orders": passes,
    }


def load_capacity_order_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_training_manifest_sha256: str,
    expected_keys: Sequence[BlockKey],
) -> list[list[BlockKey]]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("capacity order-manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != CAPACITY_ORDER_MANIFEST_PROTOCOL:
        raise RuntimeError("capacity order-manifest protocol differs")
    if manifest.get("training_manifest_sha256") != expected_training_manifest_sha256:
        raise RuntimeError("capacity order manifest names a different training artifact")
    if manifest.get("records") != RECORDS or manifest.get("passes") != PASSES:
        raise RuntimeError("capacity order-manifest cardinality differs")
    rows = manifest.get("orders")
    if not isinstance(rows, list) or len(rows) != PASSES:
        raise RuntimeError("capacity order manifest lacks 320 pass orders")
    expected_set = set(expected_keys)
    result: list[list[BlockKey]] = []
    for pass_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("pass_index") != pass_index:
            raise RuntimeError("capacity order-manifest pass indices differ")
        serialized = row.get("block_keys")
        if not isinstance(serialized, list) or len(serialized) != RECORDS:
            raise RuntimeError("capacity order-manifest pass cardinality differs")
        ordered = [_key(item) for item in serialized]
        if len(set(ordered)) != RECORDS or set(ordered) != expected_set:
            raise RuntimeError("capacity order manifest is not a key permutation")
        if ordered_block_keys_sha256(ordered) != row.get(
            "ordered_block_keys_sha256"
        ):
            raise RuntimeError("capacity pass-order semantic hash differs")
        result.append(ordered)
    return result


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
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


def _save_checkpoint(
    path: Path,
    *,
    model: DirectSafetySidecar,
    optimizer: torch.optim.Optimizer,
    pass_index: int,
    update_index: int,
    metrics: Mapping[str, Any],
    capacity_metadata_sha256: str,
    order_manifest_sha256: str,
    source_manifest_sha256: str,
) -> None:
    torch.save(
        {
            "protocol": CAPACITY_TRAINING_PROTOCOL,
            "pass": pass_index,
            "completed_updates": update_index,
            "model": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "metrics": dict(metrics),
            "capacity_metadata_sha256": capacity_metadata_sha256,
            "order_manifest_sha256": order_manifest_sha256,
            "source_manifest_sha256": source_manifest_sha256,
        },
        path,
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def run_capacity(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite capacity run: {args.output}")
    closure_start = verify_source_manifest(
        PROJECT,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("reviewed capacity training requires CUDA")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "capacity training requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    schedule = complete_pass_schedule(RECORDS, batch_size=BATCH_SIZE)
    if (
        schedule.steps_per_pass != 16
        or schedule.passes != PASSES
        or schedule.total_steps != TOTAL_UPDATES
        or schedule.warmup_steps != WARMUP_UPDATES
    ):
        raise RuntimeError("capacity schedule differs from the frozen contract")
    records, capacity_metadata = load_capacity_bundle(
        args.capacity_bundle,
        expected_metadata_sha256=args.expected_capacity_metadata_sha256,
    )
    expected_capacity_fields = {
        "producer_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
        "producer_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
        "canonical_metadata_sha256": EXPECTED_CANONICAL_METADATA_SHA256,
    }
    for name, expected in expected_capacity_fields.items():
        if capacity_metadata.get(name) != expected:
            raise RuntimeError(f"capacity metadata differs from frozen {name}")
    if capacity_metadata.get("source_closure") != closure_start.summary():
        raise RuntimeError("capacity metadata names a different source closure")
    for name in ("parent_fit_metadata_sha256", "split_manifest_sha256"):
        value = capacity_metadata.get(name)
        if not isinstance(value, str) or len(value) != 64 or value != value.lower():
            raise RuntimeError(f"capacity metadata has invalid {name}")
        try:
            int(value, 16)
        except ValueError as error:
            raise RuntimeError(f"capacity metadata has nonhex {name}") from error
    device = torch.device("cuda:0")
    tensors = _training_tensors(records, device)
    training_manifest_sha256 = args.expected_capacity_metadata_sha256
    by_key = {key: index for index, key in enumerate(tensors["keys"])}

    temporary = args.output.with_name(
        f".{args.output.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    try:
        checkpoints = temporary / "checkpoints"
        checkpoints.mkdir()
        source_snapshot = temporary / "source_snapshot"
        snapshot_source_closure(PROJECT, closure_start, source_snapshot)

        order_manifest_path = temporary / "order_manifest.json"
        _write_json(
            order_manifest_path,
            build_capacity_order_manifest(
                tensors["keys"], training_manifest_sha256
            ),
        )
        order_manifest_sha256 = sha256_file(order_manifest_path)
        frozen_orders = load_capacity_order_manifest(
            order_manifest_path,
            expected_sha256=order_manifest_sha256,
            expected_training_manifest_sha256=training_manifest_sha256,
            expected_keys=tensors["keys"],
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    start = time.perf_counter()
    try:
        torch.manual_seed(INITIALIZATION_SEED)
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
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
        history: list[dict[str, Any]] = []
        gradients_finite = True
        initial_metrics, _ = evaluate_capacity(
            model,
            tensors,
            records,
            gradients_finite=gradients_finite,
        )
        initial_record = {"pass": 0, "completed_updates": 0, **initial_metrics}
        history.append(initial_record)
        _save_checkpoint(
            checkpoints / "pass-000.pt",
            model=model,
            optimizer=optimizer,
            pass_index=0,
            update_index=0,
            metrics=initial_metrics,
            capacity_metadata_sha256=training_manifest_sha256,
            order_manifest_sha256=order_manifest_sha256,
            source_manifest_sha256=closure_start.manifest_sha256,
        )
        update_index = 0
        pass_diagnostics: list[dict[str, Any]] = []
        for pass_index in range(PASSES):
            ordered = frozen_orders[pass_index]
            batch_losses: list[float] = []
            maximum_gradient_norm = 0.0
            for start_index in range(0, RECORDS, BATCH_SIZE):
                batch_keys = ordered[start_index : start_index + BATCH_SIZE]
                indices = [by_key[key] for key in batch_keys]
                diagnostics = capacity_training_step(
                    model,
                    optimizer,
                    tensors,
                    indices,
                    update_index=update_index,
                )
                gradients_finite = gradients_finite and bool(
                    diagnostics["gradients_finite"]
                )
                batch_losses.append(float(diagnostics["optimizer_loss"]))
                maximum_gradient_norm = max(
                    maximum_gradient_norm,
                    float(diagnostics["gradient_norm_before_clip"]),
                )
                update_index += 1
            metrics, _ = evaluate_capacity(
                model,
                tensors,
                records,
                gradients_finite=gradients_finite,
            )
            history_row = {
                "pass": pass_index + 1,
                "completed_updates": update_index,
                **metrics,
            }
            history.append(history_row)
            pass_diagnostics.append(
                {
                    "pass": pass_index + 1,
                    "mean_optimizer_loss": sum(batch_losses) / len(batch_losses),
                    "maximum_gradient_norm_before_clip": maximum_gradient_norm,
                    "last_learning_rate": capacity_learning_rate(update_index - 1),
                }
            )
            _save_checkpoint(
                checkpoints / f"pass-{pass_index + 1:03d}.pt",
                model=model,
                optimizer=optimizer,
                pass_index=pass_index + 1,
                update_index=update_index,
                metrics=metrics,
                capacity_metadata_sha256=training_manifest_sha256,
                order_manifest_sha256=order_manifest_sha256,
                source_manifest_sha256=closure_start.manifest_sha256,
            )
            print(
                json.dumps(
                    {
                        "pass": pass_index + 1,
                        "loss": metrics["prompt_weighted_loss"],
                        "recovery": metrics["oracle_recovery"],
                        "beneficial_apply": metrics["beneficial_apply_count"],
                        "harmful_apply": metrics["harmful_apply_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if update_index != TOTAL_UPDATES or len(history) != 321:
            raise RuntimeError("capacity training did not complete 320 whole passes")

        selected_index = earliest_exact_minimum(
            [float(row["prompt_weighted_loss"]) for row in history]
        )
        selected_row = history[selected_index]
        gate_passed = capacity_gate_passes(
            selected_row, float(history[0]["prompt_weighted_loss"])
        )
        if gate_passed:
            protocol_selected = selected_capacity_checkpoint(history)
            if protocol_selected is not selected_row:
                raise RuntimeError("capacity checkpoint selector identity differs")
        selected_checkpoint = checkpoints / f"pass-{selected_index:03d}.pt"
        selected_payload = torch.load(
            selected_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(selected_payload["model"], strict=True)
        selected_metrics, selected_rows = evaluate_capacity(
            model,
            tensors,
            records,
            gradients_finite=gradients_finite,
        )
        for name, value in selected_metrics.items():
            if selected_row.get(name) != value:
                raise RuntimeError(f"selected checkpoint replay differs for {name}")
        shutil.copy2(selected_checkpoint, temporary / "selected.pt")
        torch.save(selected_rows, temporary / "selected_records.pt")
        _fsync_path(temporary / "selected.pt")
        _fsync_path(temporary / "selected_records.pt")
        _write_json(temporary / "history.json", history)
        _write_json(temporary / "pass_diagnostics.json", pass_diagnostics)
        checkpoint_manifest = [
            {
                "pass": pass_index,
                "path": f"checkpoints/pass-{pass_index:03d}.pt",
                "bytes": (
                    checkpoints / f"pass-{pass_index:03d}.pt"
                ).stat().st_size,
                "sha256": sha256_file(
                    checkpoints / f"pass-{pass_index:03d}.pt"
                ),
                "prompt_weighted_loss": history[pass_index][
                    "prompt_weighted_loss"
                ],
            }
            for pass_index in range(321)
        ]
        _write_json(temporary / "checkpoint_manifest.json", checkpoint_manifest)
        closure_end = verify_source_manifest(
            PROJECT,
            expected_manifest_sha256=args.expected_source_manifest_sha256,
        )
        if closure_end != closure_start:
            raise RuntimeError("source closure changed during capacity training")
        report = {
            "protocol": CAPACITY_TRAINING_PROTOCOL,
            "evidence_tier": "capacity_plumbing_only",
            "scientific_status": "PASS" if gate_passed else "FAIL",
            "capacity_gate_passed": gate_passed,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": platform.node(),
            "device": torch.cuda.get_device_name(device),
            "seconds": time.perf_counter() - start,
            "capacity_bundle": str(args.capacity_bundle.resolve()),
            "capacity_metadata": capacity_metadata,
            "capacity_metadata_sha256": training_manifest_sha256,
            "configuration": {
                "records": RECORDS,
                "batch_size": BATCH_SIZE,
                "passes": PASSES,
                "total_updates": TOTAL_UPDATES,
                "warmup_updates": WARMUP_UPDATES,
                "peak_learning_rate": PEAK_LEARNING_RATE,
                "betas": list(BETAS),
                "epsilon": EPSILON,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip": GRADIENT_CLIP,
                "initialization_seed": INITIALIZATION_SEED,
                "loss": "gain_weighted_unit_hinge_exact_1_over_512_mass",
                "decision": "apply_iff_raw_score_gt_zero",
                "checkpoint_selection": "earliest_full_precision_minimum_of_321",
            },
            "history_rows": len(history),
            "selected_pass": selected_index,
            "epoch_zero": history[0],
            "selected": selected_row,
            "artifacts": {
                "selected_checkpoint_sha256": sha256_file(
                    temporary / "selected.pt"
                ),
                "selected_records_sha256": sha256_file(
                    temporary / "selected_records.pt"
                ),
                "history_sha256": sha256_file(temporary / "history.json"),
                "pass_diagnostics_sha256": sha256_file(
                    temporary / "pass_diagnostics.json"
                ),
                "checkpoint_manifest_sha256": sha256_file(
                    temporary / "checkpoint_manifest.json"
                ),
                "order_manifest_sha256": order_manifest_sha256,
                "source_snapshot_manifest_sha256": sha256_file(
                    source_snapshot / "SOURCE_MANIFEST.json"
                ),
            },
            "source_closure_start": closure_start.summary(),
            "source_closure_end": closure_end.summary(),
        }
        _write_json(temporary / "metrics.json", report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, args.output)
        return report, gate_passed
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    args = parse_args()
    report, passed = run_capacity(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scientific_status": report["scientific_status"],
                "selected_pass": report["selected_pass"],
                "selected": report["selected"],
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
