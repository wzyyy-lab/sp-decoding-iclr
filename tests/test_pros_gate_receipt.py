from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.verify_pros_gate_receipt import (
    CAPACITY_ADJUDICATION_PROTOCOL,
    CAPACITY_ADJUDICATION_SCHEMA,
    EXPECTED_CANONICAL_METADATA_SHA256,
    EXPECTED_CAPACITY_REPAIR_SOURCE_ENTRIES_SHA256,
    EXPECTED_CAPACITY_REPAIR_SOURCE_MANIFEST_SHA256,
    EXPECTED_DIRECT_CHECKPOINT_SHA256,
    EXPECTED_DIRECT_METRICS_SHA256,
    EXPECTED_EXCLUSIONS,
    load_capacity_adjudication_receipt,
    load_receipt,
    verify_capacity_adjudication_receipt,
    verify_capacity_receipt,
    verify_outcomes_receipt,
    verify_split_receipt,
)


SPLIT = "1" * 64
FIT = "2" * 64
CHECKPOINT = "3" * 64
CAPACITY = "4" * 64
SOURCE = "5" * 64


def _split_receipt(project: Path) -> dict[str, object]:
    return {
        "status": "GO",
        "split_manifest_sha256": SPLIT,
        "canonical_metadata_sha256": EXPECTED_CANONICAL_METADATA_SHA256,
        "source_manifest_sha256": SOURCE,
        "exclusion_manifest_sha256": {
            role: value["aggregate_sha256"]
            for role, value in EXPECTED_EXCLUSIONS.items()
        },
        "exclusion_sources": {
            role: [
                {
                    "path": str(
                        (project.resolve() / value["relative_path"]).resolve()
                    ),
                    "bytes": value["bytes"],
                    "sha256": value["sha256"],
                    "selected_splits": value["selected_splits"],
                    "row_counts_by_split": value["row_counts_by_split"],
                }
            ]
            for role, value in EXPECTED_EXCLUSIONS.items()
        },
    }


def _outcomes_receipt() -> dict[str, object]:
    return {
        "status": "GO",
        "split_manifest_sha256": SPLIT,
        "fit_metadata_sha256": FIT,
        "checkpoint_metadata_sha256": CHECKPOINT,
        "canonical_metadata_sha256": EXPECTED_CANONICAL_METADATA_SHA256,
        "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
        "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
        "source_manifest_sha256": SOURCE,
        "fit": {"metadata_sha256": FIT},
        "checkpoint": {"metadata_sha256": CHECKPOINT},
    }


def _capacity_receipt() -> dict[str, object]:
    return {
        "status": "GO",
        "capacity_metadata_sha256": CAPACITY,
        "fit_metadata_sha256": FIT,
        "split_manifest_sha256": SPLIT,
        "canonical_metadata_sha256": EXPECTED_CANONICAL_METADATA_SHA256,
        "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
        "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
        "source_manifest_sha256": SOURCE,
    }


def _capacity_adjudication_receipt() -> dict[str, object]:
    closure = {
        "protocol": "pros-gate-first-party-source-closure-v1",
        "source_entries_sha256": (
            EXPECTED_CAPACITY_REPAIR_SOURCE_ENTRIES_SHA256
        ),
        "source_file_count": 60,
        "source_manifest_sha256": (
            EXPECTED_CAPACITY_REPAIR_SOURCE_MANIFEST_SHA256
        ),
    }
    identity = {"bytes": 10, "sha256": "a" * 64}
    frozen = {
        "selected.pt": identity,
        "checkpoints/pass-070.pt": identity,
        "history.json": {"bytes": 20, "sha256": "b" * 64},
    }
    return {
        "adjudication_schema": CAPACITY_ADJUDICATION_SCHEMA,
        "protocol": CAPACITY_ADJUDICATION_PROTOCOL,
        "evidence_tier": "same_subset_capacity_plumbing_only",
        "execution": {
            "device": "cpu",
            "frozen_job_id": "10138104",
            "training_or_optimizer_steps": 0,
        },
        "original_machine_verdict": {
            "scientific_status": "FAIL",
            "capacity_gate_passed": False,
            "preserved": True,
        },
        "repair": {
            "only_added_field": "harmful_keep_count",
            "equals_harm_avoidance_numerator": True,
            "denominator_equals_harmful_count": True,
            "equals_harmful_partition": True,
            "value": 128,
        },
        "replay_verdict": {
            "capacity_gate_passed": True,
            "values_finite": True,
            "gradients_finite_from_frozen_run": True,
            "regret_bound_violation_count": 0,
            "selected_pass": 70,
            "selected_updates": 1120,
            "harmful_count": 128,
            "harmful_apply_count": 0,
            "harmful_keep_count": 128,
        },
        "frozen_input_hashes_before": frozen,
        "frozen_input_hashes_after": deepcopy(frozen),
        "source_closure_start": closure,
        "source_closure_end": deepcopy(closure),
    }


def test_split_receipt_binds_exact_exclusion_and_parent_identities(
    tmp_path: Path,
) -> None:
    receipt = _split_receipt(tmp_path)
    result = verify_split_receipt(
        receipt,
        project=tmp_path,
        split_manifest_sha256=SPLIT,
        source_manifest_sha256=SOURCE,
    )
    assert result["status"] == "BOUND"
    tampered = deepcopy(receipt)
    tampered["exclusion_sources"]["reserved"][0]["bytes"] += 1
    with pytest.raises(RuntimeError, match="exclusion identity"):
        verify_split_receipt(
            tampered,
            project=tmp_path,
            split_manifest_sha256=SPLIT,
            source_manifest_sha256=SOURCE,
        )


def test_outcome_and_capacity_receipts_bind_every_downstream_parent() -> None:
    outcomes = verify_outcomes_receipt(
        _outcomes_receipt(),
        split_manifest_sha256=SPLIT,
        fit_metadata_sha256=FIT,
        checkpoint_metadata_sha256=CHECKPOINT,
        source_manifest_sha256=SOURCE,
    )
    capacity = verify_capacity_receipt(
        _capacity_receipt(),
        capacity_metadata_sha256=CAPACITY,
        fit_metadata_sha256=FIT,
        split_manifest_sha256=SPLIT,
        source_manifest_sha256=SOURCE,
    )
    assert outcomes["stage"] == "outcomes"
    assert capacity["stage"] == "capacity"

    stale = _outcomes_receipt()
    stale["fit_metadata_sha256"] = "6" * 64
    with pytest.raises(RuntimeError, match="fit_metadata_sha256"):
        verify_outcomes_receipt(
            stale,
            split_manifest_sha256=SPLIT,
            fit_metadata_sha256=FIT,
            checkpoint_metadata_sha256=CHECKPOINT,
            source_manifest_sha256=SOURCE,
        )


def test_receipt_file_hash_and_go_status_are_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_capacity_receipt()), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_receipt(path, digest)["status"] == "GO"
    with pytest.raises(RuntimeError, match="SHA256 differs"):
        load_receipt(path, "0" * 64)
    value = _capacity_receipt()
    value["status"] = "NO-GO"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="status is not GO"):
        load_receipt(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_capacity_adjudication_binds_no_training_repair_semantics(
    tmp_path: Path,
) -> None:
    receipt = _capacity_adjudication_receipt()
    result = verify_capacity_adjudication_receipt(receipt)
    assert result["status"] == "BOUND"
    assert result["capacity_gate_passed"] is True
    assert result["training_or_optimizer_steps"] == 0

    tampered = deepcopy(receipt)
    tampered["replay_verdict"]["harmful_apply_count"] = 1
    with pytest.raises(RuntimeError, match="partition"):
        verify_capacity_adjudication_receipt(tampered)

    path = tmp_path / "adjudication.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_capacity_adjudication_receipt(path, digest)
    assert loaded["adjudication_schema"] == CAPACITY_ADJUDICATION_SCHEMA


def test_capacity_receipt_cli_matches_the_slurm_argument_surface(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capacity.json"
    path.write_text(json.dumps(_capacity_receipt()), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    script = Path(__file__).resolve().parents[1] / "scripts/verify_pros_gate_receipt.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "capacity",
            "--receipt",
            str(path),
            "--expected-receipt-sha256",
            digest,
            "--capacity-metadata-sha256",
            CAPACITY,
            "--fit-metadata-sha256",
            FIT,
            "--split-manifest-sha256",
            SPLIT,
            "--source-manifest-sha256",
            SOURCE,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "status": "BOUND",
        "stage": "capacity",
        "capacity_metadata_sha256": CAPACITY,
        "fit_metadata_sha256": FIT,
        "split_manifest_sha256": SPLIT,
        "canonical_metadata_sha256": EXPECTED_CANONICAL_METADATA_SHA256,
        "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
        "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
        "source_manifest_sha256": SOURCE,
    }
