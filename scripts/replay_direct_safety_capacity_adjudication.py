#!/usr/bin/env python3
"""Versioned CPU-only adjudication replay for frozen capacity job 10138104.

This repair utility has one immutable input surface: the already published
capacity run.  It does not load any split manifest, fit/checkpoint/falsifier
bundle, validation data, producer, or GPU state.  It never rewrites the frozen
run and refuses to overwrite its append-only receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import torch

from sph.direct_safety_protocol import (
    CAPACITY_ADJUDICATION_SCHEMA,
    BlockKey,
    SavedGateRecord,
    capacity_gate_passes,
    reconstruct_saved_gate_evaluation,
    selected_capacity_checkpoint,
)
from sph.source_closure import VerifiedSourceClosure, verify_source_manifest


PROJECT = Path(__file__).resolve().parents[1]
RUN = (
    PROJECT / "artifacts/training/pros_gate_capacity_10138104/seed0"
).resolve()
OUTPUT_ROOT = (
    PROJECT / "artifacts/adjudication/pros_gate_capacity_10138104"
).resolve()
REPLAY_PROTOCOL = "pros-gate-capacity-offline-replay-v2"
EXPECTED_HASHES = {
    "metrics.json": "6c5a34c1454a0cc513587e0646615d529b711747da2656d84070e3d84aa707a6",
    "history.json": "3cbab50df060a67880fa8905def457da416925e7050296866bb91124b63e4b16",
    "selected.pt": "8bc70170a67dae1b6e2bac74929a5c6fac83debae16eb7cffffc41658716c684",
    "selected_records.pt": "1f55087ce1854f457050d5a1de5b40f38f73d2f4904ff90a614a556b98a5d0ed",
    "checkpoint_manifest.json": "f8e8f5f579760cd669df3f5c2206420dd9d2c028dcb0145d1a9984caa7e9e3e4",
    "pass_diagnostics.json": "e9904b00b5b2d1991d9e74e97cd1f6e92dbd418345bc89d8b980da9c301b81bf",
    "order_manifest.json": "91427e920d506c68b40d7fee831911cd69d3dae10ddb1d5ef422d056e64eec90",
    "checkpoints/pass-070.pt": "8bc70170a67dae1b6e2bac74929a5c6fac83debae16eb7cffffc41658716c684",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _snapshot_hashes() -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = RUN / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"frozen input is missing or not regular: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"frozen input hash differs: {relative}")
        result[relative] = {"bytes": path.stat().st_size, "sha256": observed}
    return result


def _nonnegative_integer(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"legacy {name} must be a nonnegative integer")
    return value


def repair_legacy_capacity_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Add only the authorized alias after proving every redundant invariant."""

    if "harmful_keep_count" in row:
        raise RuntimeError("legacy row unexpectedly already has harmful_keep_count")
    harmful_count = _nonnegative_integer(row, "harmful_count")
    harmful_apply = _nonnegative_integer(row, "harmful_apply_count")
    numerator = _nonnegative_integer(row, "harm_avoidance_numerator")
    denominator = _nonnegative_integer(row, "harm_avoidance_denominator")
    harmful_keep = harmful_count - harmful_apply
    if harmful_keep < 0:
        raise RuntimeError("legacy harmful APPLY exceeds harmful count")
    if numerator != harmful_keep:
        raise RuntimeError("legacy harm-avoidance numerator violates partition")
    if denominator != harmful_count:
        raise RuntimeError("legacy harm-avoidance denominator differs from count")
    repaired = dict(row)
    repaired["harmful_keep_count"] = harmful_keep
    return repaired


def _saved_records(rows: Any) -> list[SavedGateRecord]:
    if not isinstance(rows, list) or len(rows) != 512:
        raise RuntimeError("selected records must be a 512-row list")
    expected_fields = {
        "sample_id",
        "anchor_offset",
        "context_length",
        "base_length",
        "direct_length",
        "score",
        "base_first_token_correct",
        "direct_first_token_correct",
    }
    result: list[SavedGateRecord] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise RuntimeError("selected record schema differs")
        result.append(
            SavedGateRecord(
                block_key=BlockKey(
                    sample_id=row["sample_id"],
                    anchor_offset=row["anchor_offset"],
                    context_length=row["context_length"],
                ),
                base_length=row["base_length"],
                direct_length=row["direct_length"],
                score=row["score"],
                base_first_token_correct=row["base_first_token_correct"],
                direct_first_token_correct=row["direct_first_token_correct"],
            )
        )
    return result


def _require_same_metric(name: str, observed: Any, expected: Any) -> None:
    if isinstance(observed, bool) or isinstance(expected, bool):
        if observed is not expected:
            raise RuntimeError(f"selected replay differs for {name}")
        return
    if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        if not math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"selected replay differs for {name}")
        return
    if observed != expected:
        raise RuntimeError(f"selected replay differs for {name}")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite replay receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite replay receipt: {path}"
            ) from error
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_unchanged_source_closure(
    initial: VerifiedSourceClosure,
    *,
    project: Path,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
) -> VerifiedSourceClosure:
    """Reverify the exact closure before making any receipt publishable."""

    final = verify_source_manifest(
        project,
        source_manifest,
        expected_manifest_sha256=expected_source_manifest_sha256,
    )
    if final != initial:
        raise RuntimeError("repair source closure changed during offline replay")
    return final


def replay(
    *, source_manifest: Path, expected_source_manifest_sha256: str
) -> dict[str, Any]:
    closure = verify_source_manifest(
        PROJECT,
        source_manifest,
        expected_manifest_sha256=expected_source_manifest_sha256,
    )
    hashes_before = _snapshot_hashes()
    metrics = _load_json(RUN / "metrics.json")
    history = _load_json(RUN / "history.json")
    if not isinstance(metrics, dict) or not isinstance(history, list):
        raise RuntimeError("frozen metrics/history schema differs")
    if metrics.get("job_id") != "10138104":
        raise RuntimeError("frozen metrics job identity differs")
    if metrics.get("protocol") != "pros-gate-capacity-training-v1":
        raise RuntimeError("frozen training protocol differs")
    if metrics.get("scientific_status") != "FAIL":
        raise RuntimeError("original scientific status is not frozen FAIL")
    if metrics.get("capacity_gate_passed") is not False:
        raise RuntimeError("original capacity verdict is not frozen false")
    if metrics.get("selected_pass") != 70 or metrics.get("history_rows") != 321:
        raise RuntimeError("frozen selection identity differs")
    if len(history) != 321:
        raise RuntimeError("frozen history must contain 321 rows")

    epoch_zero_loss = float(history[0]["prompt_weighted_loss"])
    original_selected = history[70]
    if capacity_gate_passes(original_selected, epoch_zero_loss):
        raise RuntimeError("legacy missing-alias row unexpectedly passes")
    repaired_history = [repair_legacy_capacity_row(row) for row in history]
    selected = selected_capacity_checkpoint(repaired_history)
    if selected.get("pass") != 70:
        raise RuntimeError("repaired history changed earliest-minimum selection")

    serialized = torch.load(
        RUN / "selected_records.pt", map_location="cpu", weights_only=True
    )
    saved = _saved_records(serialized)
    replayed = reconstruct_saved_gate_evaluation(saved)
    for name, value in replayed.metrics.items():
        if value is not None:
            _require_same_metric(name, value, selected[name])
    _require_same_metric(
        "prompt_weighted_loss",
        replayed.metrics["prompt_weighted_gain_hinge"],
        selected["prompt_weighted_loss"],
    )
    if selected["utility_optimal_count"] != replayed.metrics[
        "utility_optimal_numerator"
    ]:
        raise RuntimeError("utility-optimal alias differs from saved replay")
    if not capacity_gate_passes(selected, epoch_zero_loss):
        raise RuntimeError("repaired selected row does not pass capacity gate")
    if repaired_history[70] != selected:
        raise RuntimeError("selected repaired row identity differs")
    if sha256_file(RUN / "selected.pt") != sha256_file(
        RUN / "checkpoints/pass-070.pt"
    ):
        raise RuntimeError("selected checkpoint is not byte-identical to pass 70")

    hashes_after = _snapshot_hashes()
    if hashes_after != hashes_before:
        raise RuntimeError("frozen inputs changed during offline replay")
    closure_end = require_unchanged_source_closure(
        closure,
        project=PROJECT,
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    return {
        "protocol": REPLAY_PROTOCOL,
        "adjudication_schema": CAPACITY_ADJUDICATION_SCHEMA,
        "evidence_tier": "same_subset_capacity_plumbing_only",
        "execution": {
            "device": "cpu",
            "training_or_optimizer_steps": 0,
            "frozen_job_id": "10138104",
        },
        "original_machine_verdict": {
            "scientific_status": "FAIL",
            "capacity_gate_passed": False,
            "preserved": True,
        },
        "repair": {
            "only_added_field": "harmful_keep_count",
            "value": int(selected["harmful_keep_count"]),
            "equals_harm_avoidance_numerator": bool(
                selected["harmful_keep_count"]
                == selected["harm_avoidance_numerator"]
            ),
            "equals_harmful_partition": bool(
                selected["harmful_keep_count"]
                == selected["harmful_count"] - selected["harmful_apply_count"]
            ),
            "denominator_equals_harmful_count": bool(
                selected["harm_avoidance_denominator"]
                == selected["harmful_count"]
            ),
        },
        "replay_verdict": {
            "capacity_gate_passed": True,
            "selected_pass": 70,
            "selected_updates": int(selected["completed_updates"]),
            "record_count": int(selected["record_count"]),
            "prompt_count": int(selected["prompt_count"]),
            "beneficial_apply_count": int(selected["beneficial_apply_count"]),
            "beneficial_count": int(selected["beneficial_count"]),
            "harmful_keep_count": int(selected["harmful_keep_count"]),
            "harmful_count": int(selected["harmful_count"]),
            "harmful_apply_count": int(selected["harmful_apply_count"]),
            "neutral_apply_count": int(selected["neutral_apply_count"]),
            "neutral_count": int(selected["neutral_count"]),
            "utility_optimal_count": int(selected["utility_optimal_count"]),
            "prompt_weighted_loss": float(selected["prompt_weighted_loss"]),
            "epoch_zero_loss": epoch_zero_loss,
            "regret_bound_violation_count": int(
                selected["regret_bound_violation_count"]
            ),
            "oracle_recovery": float(selected["oracle_recovery"]),
            "values_finite": selected["values_finite"] is True,
            "gradients_finite_from_frozen_run": selected["gradients_finite"] is True,
        },
        "source_closure_start": closure.summary(),
        "source_closure_end": closure_end.summary(),
        "frozen_input_hashes_before": hashes_before,
        "frozen_input_hashes_after": hashes_after,
        "limitations": [
            "capacity plumbing only; no producer-OOS or downstream claim",
            "neutral APPLY is non-gating and is not calibration evidence",
            "gradients_finite is carried from the immutable frozen run",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.parent != OUTPUT_ROOT or output.name != "capacity_adjudication_v2.json":
        raise RuntimeError("receipt path must be the single versioned append-only target")
    report = replay(
        source_manifest=args.source_manifest.resolve(),
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
    )
    _atomic_write_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "receipt_sha256": sha256_file(output),
                "selected_pass": report["replay_verdict"]["selected_pass"],
                "capacity_gate_passed": report["replay_verdict"][
                    "capacity_gate_passed"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
