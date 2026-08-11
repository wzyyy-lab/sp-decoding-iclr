#!/usr/bin/env python3
"""Stdlib-only semantic verifier for staged PROS-Gate GO receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_CANONICAL_METADATA_SHA256 = (
    "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320"
)
EXPECTED_DIRECT_CHECKPOINT_SHA256 = (
    "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e"
)
EXPECTED_DIRECT_METRICS_SHA256 = (
    "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef"
)
EXPECTED_CAPACITY_REPAIR_SOURCE_MANIFEST_SHA256 = (
    "b072758abf6aabc7f7af39d52db0327d9d749192f1be812c3da7cc5fe735f8f2"
)
EXPECTED_CAPACITY_REPAIR_SOURCE_ENTRIES_SHA256 = (
    "77894a782c151bba34c01dfd89e1482313669e104832905f962afca7a4e46f92"
)
CAPACITY_ADJUDICATION_SCHEMA = "pros-capacity-adjudication-v2"
CAPACITY_ADJUDICATION_PROTOCOL = "pros-gate-capacity-offline-replay-v2"
EXPECTED_EXCLUSIONS = {
    "producer_train": {
        "relative_path": "artifacts/manifests/open_perfectblend_100k_v2.jsonl",
        "bytes": 92_590_866,
        "sha256": "b05087a56e8e717605415026421f7bae23092eb7cb9509361a36932f80260e3a",
        "selected_splits": ["train"],
        "row_counts_by_split": {"train": 100_000},
        "aggregate_sha256": "dcd1decfa63d17b4f4ee180a2d30e774ffb87bc9eed96a956f045a117039b16d",
    },
    "validation": {
        "relative_path": "artifacts/manifests/phase3_development_v3.jsonl",
        "bytes": 1_418_202,
        "sha256": "e16374068e9c8904214fbf282b4adb6187a0b099db5c37e79660fc46a2801d01",
        "selected_splits": ["validation_gate", "validation_select"],
        "row_counts_by_split": {
            "train": 2_000,
            "validation_gate": 150,
            "validation_select": 150,
        },
        "aggregate_sha256": "fc336fa8672140facd82dc6f73be067c02d87e27bb6e276137a290a16cc7ab09",
    },
    "reserved": {
        "relative_path": "artifacts/manifests/phase3_reserved_test_v3.jsonl",
        "bytes": 429_359,
        "sha256": "ae25467fbb52b7091c8d9a5f98776b11ccf76e87e781850b5638734548a53bb4",
        "selected_splits": ["test"],
        "row_counts_by_split": {"test": 600},
        "aggregate_sha256": "94c3e4274af5f310042766f962fcc3bf57b854d9b1f3e99bec950ddb367b4885",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise RuntimeError(f"{name} is not a canonical lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(f"{name} is not hexadecimal") from error
    return value


def _load_json_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = require_sha256("expected receipt hash", expected_sha256)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("GO receipt is missing or is a symlink")
    if sha256_file(path) != expected:
        raise RuntimeError("GO receipt SHA256 differs from reviewed input")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("GO receipt must be a JSON object")
    return value


def load_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    value = _load_json_receipt(path, expected_sha256)
    if value.get("status") != "GO":
        raise RuntimeError("receipt status is not GO")
    return value


def load_capacity_adjudication_receipt(
    path: Path, expected_sha256: str
) -> dict[str, Any]:
    """Load the reviewed capacity replay, which is not a generic GO receipt."""

    return _load_json_receipt(path, expected_sha256)


def require_field(
    receipt: Mapping[str, Any], path: str, expected: Any
) -> None:
    value: Any = receipt
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise RuntimeError(f"GO receipt lacks {path}")
        value = value[component]
    if value != expected:
        raise RuntimeError(f"GO receipt differs for {path}")


def verify_split_receipt(
    receipt: Mapping[str, Any],
    *,
    project: Path,
    split_manifest_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    split_hash = require_sha256("split manifest hash", split_manifest_sha256)
    source_hash = require_sha256("source manifest hash", source_manifest_sha256)
    require_field(receipt, "split_manifest_sha256", split_hash)
    require_field(
        receipt,
        "canonical_metadata_sha256",
        EXPECTED_CANONICAL_METADATA_SHA256,
    )
    require_field(receipt, "source_manifest_sha256", source_hash)
    for role, expected in sorted(EXPECTED_EXCLUSIONS.items()):
        require_field(
            receipt,
            f"exclusion_manifest_sha256.{role}",
            expected["aggregate_sha256"],
        )
        sources = receipt.get("exclusion_sources", {}).get(role)
        if not isinstance(sources, list) or len(sources) != 1:
            raise RuntimeError(f"GO receipt exclusion source count differs for {role}")
        exact_source = {
            "path": str((project.resolve() / expected["relative_path"]).resolve()),
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
            "selected_splits": expected["selected_splits"],
            "row_counts_by_split": expected["row_counts_by_split"],
        }
        if sources[0] != exact_source:
            raise RuntimeError(f"GO receipt exclusion identity differs for {role}")
    return {
        "status": "BOUND",
        "stage": "split",
        "split_manifest_sha256": split_hash,
        "source_manifest_sha256": source_hash,
    }


def verify_outcomes_receipt(
    receipt: Mapping[str, Any],
    *,
    split_manifest_sha256: str,
    fit_metadata_sha256: str,
    checkpoint_metadata_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    expected = {
        "split_manifest_sha256": require_sha256(
            "split manifest hash", split_manifest_sha256
        ),
        "fit_metadata_sha256": require_sha256(
            "fit metadata hash", fit_metadata_sha256
        ),
        "checkpoint_metadata_sha256": require_sha256(
            "checkpoint metadata hash", checkpoint_metadata_sha256
        ),
        "canonical_metadata_sha256": EXPECTED_CANONICAL_METADATA_SHA256,
        "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
        "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
        "source_manifest_sha256": require_sha256(
            "source manifest hash", source_manifest_sha256
        ),
    }
    for name, value in expected.items():
        require_field(receipt, name, value)
    require_field(receipt, "fit.metadata_sha256", expected["fit_metadata_sha256"])
    require_field(
        receipt,
        "checkpoint.metadata_sha256",
        expected["checkpoint_metadata_sha256"],
    )
    return {"status": "BOUND", "stage": "outcomes", **expected}


def verify_capacity_receipt(
    receipt: Mapping[str, Any],
    *,
    capacity_metadata_sha256: str,
    fit_metadata_sha256: str,
    split_manifest_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    expected = {
        "capacity_metadata_sha256": require_sha256(
            "capacity metadata hash", capacity_metadata_sha256
        ),
        "fit_metadata_sha256": require_sha256(
            "fit metadata hash", fit_metadata_sha256
        ),
        "split_manifest_sha256": require_sha256(
            "split manifest hash", split_manifest_sha256
        ),
        "canonical_metadata_sha256": EXPECTED_CANONICAL_METADATA_SHA256,
        "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
        "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
        "source_manifest_sha256": require_sha256(
            "source manifest hash", source_manifest_sha256
        ),
    }
    for name, value in expected.items():
        require_field(receipt, name, value)
    return {"status": "BOUND", "stage": "capacity", **expected}


def verify_capacity_adjudication_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exact no-training repair semantics that authorize R082.

    The original capacity process correctly froze all learned artifacts but
    exited with a legacy reporting-schema failure.  This verifier does not
    rewrite that verdict.  It accepts only the reviewed CPU replay whose sole
    repair is the algebraically redundant ``harmful_keep_count`` alias.
    """

    require_field(receipt, "adjudication_schema", CAPACITY_ADJUDICATION_SCHEMA)
    require_field(receipt, "protocol", CAPACITY_ADJUDICATION_PROTOCOL)
    require_field(
        receipt,
        "evidence_tier",
        "same_subset_capacity_plumbing_only",
    )
    require_field(receipt, "execution.device", "cpu")
    require_field(receipt, "execution.frozen_job_id", "10138104")
    require_field(receipt, "execution.training_or_optimizer_steps", 0)
    require_field(
        receipt, "original_machine_verdict.scientific_status", "FAIL"
    )
    require_field(
        receipt, "original_machine_verdict.capacity_gate_passed", False
    )
    require_field(receipt, "original_machine_verdict.preserved", True)
    require_field(receipt, "repair.only_added_field", "harmful_keep_count")
    require_field(receipt, "repair.equals_harm_avoidance_numerator", True)
    require_field(receipt, "repair.denominator_equals_harmful_count", True)
    require_field(receipt, "repair.equals_harmful_partition", True)
    require_field(receipt, "replay_verdict.capacity_gate_passed", True)
    require_field(receipt, "replay_verdict.values_finite", True)
    require_field(
        receipt, "replay_verdict.gradients_finite_from_frozen_run", True
    )
    require_field(receipt, "replay_verdict.regret_bound_violation_count", 0)
    require_field(receipt, "replay_verdict.selected_pass", 70)
    require_field(receipt, "replay_verdict.selected_updates", 1120)

    harmful_count = receipt.get("replay_verdict", {}).get("harmful_count")
    harmful_apply = receipt.get("replay_verdict", {}).get(
        "harmful_apply_count"
    )
    harmful_keep = receipt.get("replay_verdict", {}).get(
        "harmful_keep_count"
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (harmful_count, harmful_apply, harmful_keep)
    ):
        raise RuntimeError("capacity replay harmful partition is not integral")
    if harmful_keep != harmful_count - harmful_apply:
        raise RuntimeError("capacity replay harmful partition is inconsistent")
    if receipt.get("repair", {}).get("value") != harmful_keep:
        raise RuntimeError("capacity replay repair value differs from alias")

    before = receipt.get("frozen_input_hashes_before")
    after = receipt.get("frozen_input_hashes_after")
    if not isinstance(before, Mapping) or not before or before != after:
        raise RuntimeError("capacity replay frozen inputs changed")
    for relative, identity in before.items():
        if not isinstance(relative, str) or not relative or not isinstance(
            identity, Mapping
        ):
            raise RuntimeError("capacity replay input identity is malformed")
        byte_count = identity.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
        ):
            raise RuntimeError("capacity replay input byte count is invalid")
        require_sha256(
            f"capacity replay input hash for {relative}",
            identity.get("sha256"),
        )
    if before.get("selected.pt") != before.get("checkpoints/pass-070.pt"):
        raise RuntimeError("capacity replay selected checkpoint identity differs")

    expected_closure = {
        "protocol": "pros-gate-first-party-source-closure-v1",
        "source_entries_sha256": EXPECTED_CAPACITY_REPAIR_SOURCE_ENTRIES_SHA256,
        "source_file_count": 60,
        "source_manifest_sha256": (
            EXPECTED_CAPACITY_REPAIR_SOURCE_MANIFEST_SHA256
        ),
    }
    require_field(receipt, "source_closure_start", expected_closure)
    require_field(receipt, "source_closure_end", expected_closure)
    return {
        "status": "BOUND",
        "stage": "capacity-adjudication",
        "adjudication_schema": CAPACITY_ADJUDICATION_SCHEMA,
        "capacity_gate_passed": True,
        "frozen_job_id": "10138104",
        "training_or_optimizer_steps": 0,
        "source_manifest_sha256": (
            EXPECTED_CAPACITY_REPAIR_SOURCE_MANIFEST_SHA256
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="stage", required=True)

    split = commands.add_parser("split")
    split.add_argument("--project", type=Path, required=True)
    split.add_argument("--receipt", type=Path, required=True)
    split.add_argument("--expected-receipt-sha256", required=True)
    split.add_argument("--split-manifest-sha256", required=True)
    split.add_argument("--source-manifest-sha256", required=True)

    outcomes = commands.add_parser("outcomes")
    outcomes.add_argument("--receipt", type=Path, required=True)
    outcomes.add_argument("--expected-receipt-sha256", required=True)
    outcomes.add_argument("--split-manifest-sha256", required=True)
    outcomes.add_argument("--fit-metadata-sha256", required=True)
    outcomes.add_argument("--checkpoint-metadata-sha256", required=True)
    outcomes.add_argument("--source-manifest-sha256", required=True)

    capacity = commands.add_parser("capacity")
    capacity.add_argument("--receipt", type=Path, required=True)
    capacity.add_argument("--expected-receipt-sha256", required=True)
    capacity.add_argument("--capacity-metadata-sha256", required=True)
    capacity.add_argument("--fit-metadata-sha256", required=True)
    capacity.add_argument("--split-manifest-sha256", required=True)
    capacity.add_argument("--source-manifest-sha256", required=True)

    capacity_adjudication = commands.add_parser("capacity-adjudication")
    capacity_adjudication.add_argument("--receipt", type=Path, required=True)
    capacity_adjudication.add_argument(
        "--expected-receipt-sha256", required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "capacity-adjudication":
        receipt = load_capacity_adjudication_receipt(
            args.receipt, args.expected_receipt_sha256
        )
        result = verify_capacity_adjudication_receipt(receipt)
    else:
        receipt = load_receipt(args.receipt, args.expected_receipt_sha256)
    if args.stage == "split":
        result = verify_split_receipt(
            receipt,
            project=args.project,
            split_manifest_sha256=args.split_manifest_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
        )
    elif args.stage == "outcomes":
        result = verify_outcomes_receipt(
            receipt,
            split_manifest_sha256=args.split_manifest_sha256,
            fit_metadata_sha256=args.fit_metadata_sha256,
            checkpoint_metadata_sha256=args.checkpoint_metadata_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
        )
    elif args.stage == "capacity":
        result = verify_capacity_receipt(
            receipt,
            capacity_metadata_sha256=args.capacity_metadata_sha256,
            fit_metadata_sha256=args.fit_metadata_sha256,
            split_manifest_sha256=args.split_manifest_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
        )
    elif args.stage != "capacity-adjudication":  # pragma: no cover
        raise AssertionError(args.stage)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
