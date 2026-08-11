from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch

import scripts.audit_direct_safety_artifacts as audit_module
from scripts.audit_direct_safety_artifacts import (
    EXPECTED_DIRECT_RUN,
    EXPECTED_SOURCE_DATA,
    EXPECTED_TARGET,
    _verify_source_manifest,
    audit_capacity_bundle_values,
    audit_outcome_bundle_values,
    audit_outcome_record,
    audit_split_manifest_values,
)
from sph.direct_safety_artifacts import (
    build_phase3_split_manifest,
    select_capacity_from_fit_bundle,
    sha256_file,
    validate_outcome_record,
    write_capacity_bundle,
    write_outcome_bundle,
)
from sph.direct_safety_gate import direct_safety_position_features
from sph.direct_safety_numeric_policy import (
    NUMERIC_POLICY_ID,
    NUMERIC_POLICY_SHA256,
    numeric_policy_receipt,
)
from sph.global_direct_selector import GlobalDirectOutput


CANONICAL_HASH = "0" * 64
SPLIT_HASH = "1" * 64
PRODUCER_HASH = "2" * 64
METRICS_HASH = "3" * 64
FIT_HASH = "4" * 64
EXCLUSION_HASHES = {
    "producer_train": "5" * 64,
    "validation": "6" * 64,
    "reserved": "7" * 64,
}
EXCLUSIONS = {
    "producer_train": {"producer:a"},
    "validation": {"validation:a"},
    "reserved": {"reserved:a"},
}
SMALL_COUNTS = {
    "chat": {"fit": 1, "checkpoint": 1, "falsifier": 1},
    "code": {"fit": 1, "checkpoint": 1, "falsifier": 1},
}
CURRENT_SOURCE_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "refine-logs/direct-safety-gate/R083_SOURCE_CLOSURE_RESCUE_V2.json"
)


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_small_tensor_checks():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _source_identities() -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"{domain}:{prompt}",
            "domain": domain,
            "split": "train",
            "anchor_offset": block * 15,
            "context_length": 64 + block,
        }
        for domain in ("chat", "code")
        for prompt in range(3)
        for block in range(2)
    ]


def _record(
    sample_id: str,
    outcome: str,
    *,
    anchor_offset: int = 0,
) -> dict[str, object]:
    logits = -torch.arange(16, dtype=torch.float32)[None].expand(15, -1).clone()
    full_lse = torch.logsumexp(logits, dim=-1) + 0.75
    base = logits - full_lse[:, None]
    residual = torch.zeros_like(base)
    residual[0, 1] = 8.0
    residual[0] -= residual[0].mean()
    scores = base + residual
    output = GlobalDirectOutput(
        scores=scores[None],
        log_probs=torch.log_softmax(scores[None], dim=-1),
        residual_scores=residual[None],
        base_log_probs=base[None],
    )
    states = torch.randn(
        1, 15, 16, 64, generator=torch.Generator().manual_seed(20260805)
    )
    features = direct_safety_position_features(
        states, output, logits[None], full_lse[None]
    )
    ids = (
        10_000
        + torch.arange(15, dtype=torch.int64)[:, None] * 100
        + torch.arange(16, dtype=torch.int64)[None]
    )
    gold = ids[:, 0].clone()
    if outcome == "beneficial":
        gold[0] = ids[0, 1]
        lengths = (0, 15)
    elif outcome == "harmful":
        lengths = (15, 0)
    elif outcome == "changed-neutral":
        gold[0] = 999_999
        lengths = (0, 0)
    else:
        raise ValueError(outcome)
    gain = (lengths[1] - lengths[0]) / 15.0
    return {
        "sample_id": sample_id,
        "anchor_offset": anchor_offset,
        "context_length": 64 + anchor_offset,
        "domain": "synthetic",
        "source_split": "train",
        "split": "fit",
        "numeric_policy_id": NUMERIC_POLICY_ID,
        "numeric_policy_sha256": NUMERIC_POLICY_SHA256,
        "position_features": features.position_features[0],
        "direct_path": features.direct_path[0],
        "change_mask": features.change_mask[0],
        "candidate_ids": ids,
        "gold_ids": gold,
        "candidate_logits": logits,
        "base_logsumexp": full_lse,
        "direct_scores": scores,
        "direct_residual_scores": residual,
        "base_log_probs": base,
        "base_length": lengths[0],
        "direct_length": lengths[1],
        "base_first_token_correct": lengths[0] > 0,
        "direct_first_token_correct": lengths[1] > 0,
        "normalized_gain": gain,
    }


def _provenance(records: int) -> dict[str, object]:
    source_closure = _verify_source_manifest()
    return {
        "source_metadata_sha256": (
            "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320"
        ),
        "source_split": "train",
        "direct_checkpoint_sha256": (
            "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e"
        ),
        "direct_metrics_sha256": (
            "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef"
        ),
        "target_identity_verified": True,
        "numeric_policy": numeric_policy_receipt(),
        "source_data": str(EXPECTED_SOURCE_DATA),
        "direct_run": str(EXPECTED_DIRECT_RUN),
        "target": str(EXPECTED_TARGET),
        "native_witness": {
            "batches": 1,
            "records": records,
            "regular_vs_hooked_outputs_bitwise": True,
            "hooked_repeat_outputs_bitwise": True,
            "hooked_repeat_node_states_bitwise": True,
            "state_dict_key_count": 1,
            "state_dict_checks": 3,
            "state_dict_sha256_before": "a" * 64,
            "state_dict_sha256_after": "a" * 64,
            "state_dict_unchanged_after_native": True,
            "state_dict_unchanged_after_hooked": True,
            "state_dict_unchanged_after_repeated_hooked": True,
            "numeric_policy_id": NUMERIC_POLICY_ID,
            "numeric_policy_sha256": NUMERIC_POLICY_SHA256,
            "same_device_numeric_invariants": True,
            "same_device_numeric_batches": 1,
            "same_device_numeric_relation_checks": 15,
        },
        "source_closure_start": source_closure,
        "source_closure_end": source_closure,
    }


def test_independent_split_audit_reconstructs_assignments_and_leakage() -> None:
    records = _source_identities()
    manifest = build_phase3_split_manifest(
        records,
        canonical_metadata_sha256=CANONICAL_HASH,
        exclusion_prompt_sets=EXCLUSIONS,
        exclusion_manifest_sha256=EXCLUSION_HASHES,
        split_counts=SMALL_COUNTS,
        expected_prompts=6,
        expected_blocks=12,
    )
    report = audit_split_manifest_values(
        manifest,
        records,
        metadata_sha256=CANONICAL_HASH,
        exclusion_prompt_sets=EXCLUSIONS,
        exclusion_hashes=EXCLUSION_HASHES,
        expected_prompts=6,
        expected_blocks=12,
        split_counts=SMALL_COUNTS,
    )
    assert report["status"] == "GO"
    assert report["prompt_counts_by_split"] == {
        "checkpoint": 2,
        "falsifier": 2,
        "fit": 2,
    }
    tampered = deepcopy(manifest)
    original = tampered["blocks"][0]["split"]
    tampered["blocks"][0]["split"] = (
        "fit" if original != "fit" else "checkpoint"
    )
    with pytest.raises(RuntimeError, match="block assignments"):
        audit_split_manifest_values(
            tampered,
            records,
            metadata_sha256=CANONICAL_HASH,
            exclusion_prompt_sets=EXCLUSIONS,
            exclusion_hashes=EXCLUSION_HASHES,
            expected_prompts=6,
            expected_blocks=12,
            split_counts=SMALL_COUNTS,
        )


def test_independent_exclusion_reader_filters_combined_manifest_rows(
    tmp_path, monkeypatch
) -> None:
    rows_by_role = {
        "producer_train": [("producer:a", "train")],
        "validation": [
            ("phase3-train:a", "train"),
            ("heldout:gate", "validation_gate"),
            ("heldout:select", "validation_select"),
        ],
        "reserved": [("reserved:a", "test")],
    }
    selected = {
        "producer_train": ["train"],
        "validation": ["validation_gate", "validation_select"],
        "reserved": ["test"],
    }
    expected = {}
    values = []
    for role, rows in rows_by_role.items():
        path = tmp_path / f"{role}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"sample_id": sample_id, "split": split}) + "\n"
                for sample_id, split in rows
            ),
            encoding="utf-8",
        )
        counts = {}
        for _, split in rows:
            counts[split] = counts.get(split, 0) + 1
        expected[role] = (
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "selected_splits": selected[role],
                "row_counts_by_split": dict(sorted(counts.items())),
            },
        )
        values.append(f"{role}={path}")
    monkeypatch.setattr(audit_module, "EXPECTED_EXCLUSION_SOURCES", expected)
    prompt_sets, hashes, provenance = audit_module._read_exclusions(values)
    assert prompt_sets["validation"] == {"heldout:gate", "heldout:select"}
    assert "phase3-train:a" not in prompt_sets["validation"]
    assert set(hashes) == set(rows_by_role)
    assert provenance["validation"][0]["selected_splits"] == selected["validation"]

    bad_expected = deepcopy(expected)
    bad_expected["validation"][0]["row_counts_by_split"]["train"] = 2
    monkeypatch.setattr(
        audit_module, "EXPECTED_EXCLUSION_SOURCES", bad_expected
    )
    with pytest.raises(RuntimeError, match="split census"):
        audit_module._read_exclusions(values)


def test_independent_outcome_audit_does_not_call_writer_validator(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit_module, "SOURCE_MANIFEST", CURRENT_SOURCE_MANIFEST)
    record = _record("fit:0", "beneficial")
    validated = audit_outcome_record(record, expected_split="fit")
    assert validated.normalized_gain == 1.0
    tampered = deepcopy(record)
    tampered["direct_length"] = 14
    with pytest.raises(RuntimeError, match="Direct length"):
        audit_outcome_record(tampered, expected_split="fit")

    bundle = tmp_path / "fit"
    write_outcome_bundle(
        bundle,
        [record],
        split="fit",
        split_manifest_sha256=SPLIT_HASH,
        provenance=_provenance(1),
    )
    split_manifest = {
        "blocks": [
            {
                "sample_id": "fit:0",
                "anchor_offset": 0,
                "context_length": 64,
                "domain": "synthetic",
                "split": "fit",
            }
        ]
    }
    _, _, report = audit_outcome_bundle_values(
        bundle,
        expected_split="fit",
        expected_metadata_sha256=sha256_file(bundle / "metadata.json"),
        split_manifest=split_manifest,
        split_manifest_sha256=SPLIT_HASH,
    )
    assert report["status"] == "GO"
    assert report["base_token_mass"] == 0
    assert report["direct_token_mass"] == 15


def test_independent_outcome_auditor_requires_numeric_policy_binding() -> None:
    record = _record("fit:policy", "beneficial")
    record["numeric_policy_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="policy digest"):
        audit_outcome_record(record, expected_split="fit")


def _audit_move_float32(
    value: torch.Tensor, direction: float, steps: int
) -> torch.Tensor:
    moved = value.clone()
    target = torch.full_like(moved, direction)
    for _ in range(steps):
        moved = torch.nextafter(moved, target)
    return moved


@pytest.mark.parametrize(("column", "index", "message"), [(195, 0, "rank"), (196, 7, "position")])
@pytest.mark.parametrize("direction", [-torch.inf, torch.inf])
def test_independent_auditor_accepts_one_adjacent_normalized_float32(
    column: int,
    index: int,
    message: str,
    direction: float,
) -> None:
    del message
    record = deepcopy(_record("fit:adjacent", "beneficial"))
    value = record["position_features"][index, column]
    record["position_features"][index, column] = _audit_move_float32(
        value, direction, 1
    )
    audit_outcome_record(record, expected_split="fit")


@pytest.mark.parametrize(("column", "index", "message"), [(195, 0, "rank"), (196, 7, "position")])
@pytest.mark.parametrize("steps", [2, 8])
def test_independent_auditor_rejects_multi_ulp_normalized_tampering(
    column: int,
    index: int,
    message: str,
    steps: int,
) -> None:
    record = deepcopy(_record("fit:multi-ulp", "beneficial"))
    value = record["position_features"][index, column]
    record["position_features"][index, column] = _audit_move_float32(
        value, torch.inf, steps
    )
    with pytest.raises(RuntimeError, match=message):
        audit_outcome_record(record, expected_split="fit")


@pytest.mark.parametrize(("column", "index", "message"), [(195, 0, "rank"), (196, 7, "position")])
def test_independent_auditor_rejects_material_normalized_tampering(
    column: int,
    index: int,
    message: str,
) -> None:
    record = deepcopy(_record("fit:material", "beneficial"))
    record["position_features"][index, column] += 1e-4
    with pytest.raises(RuntimeError, match=message):
        audit_outcome_record(record, expected_split="fit")


@pytest.mark.parametrize(
    ("column", "index", "direction", "message"),
    [
        (195, 1, torch.inf, "rank"),
        (196, 0, torch.inf, "position"),
        (196, 14, -torch.inf, "position"),
    ],
)
def test_independent_auditor_keeps_normalized_endpoints_exact(
    column: int,
    index: int,
    direction: float,
    message: str,
) -> None:
    record = deepcopy(_record("fit:endpoint", "beneficial"))
    value = record["position_features"][index, column]
    record["position_features"][index, column] = _audit_move_float32(
        value, direction, 1
    )
    with pytest.raises(RuntimeError, match=message):
        audit_outcome_record(record, expected_split="fit")


def test_actual_cuda_normalized_division_is_accepted_by_both_validators() -> None:
    if not torch.cuda.is_available():
        if os.environ.get("PROS_REQUIRE_CUDA") == "1":
            pytest.fail("PROS numeric smoke requires an allocated CUDA device")
        pytest.skip("CUDA is unavailable")
    record = deepcopy(_record("fit:cuda-portability", "beneficial"))
    path = record["direct_path"].to("cuda")
    record["position_features"][:, 195] = (path.float() / 15.0).cpu()
    record["position_features"][:, 196] = (
        torch.arange(15, device="cuda", dtype=torch.float32) / 14.0
    ).cpu()
    validate_outcome_record(record, expected_split="fit")
    audit_outcome_record(record, expected_split="fit")


def test_independent_capacity_audit_replays_exact_fit_ranking(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit_module, "SOURCE_MANIFEST", CURRENT_SOURCE_MANIFEST)
    fit_records: list[dict[str, object]] = []
    for outcome, count in (
        ("harmful", 128),
        ("changed-neutral", 128),
        ("beneficial", 256),
    ):
        template = _record(f"template:{outcome}", outcome)
        for index in range(count):
            row = dict(template)
            row["sample_id"] = f"{outcome}:{index:04d}"
            row["anchor_offset"] = index
            row["context_length"] = 1000 + index
            fit_records.append(row)
    selected = select_capacity_from_fit_bundle(
        fit_records,
        producer_checkpoint_sha256=PRODUCER_HASH,
        producer_metrics_sha256=METRICS_HASH,
        canonical_metadata_sha256=CANONICAL_HASH,
        split_manifest_sha256=SPLIT_HASH,
    )
    bundle = tmp_path / "capacity"
    write_capacity_bundle(
        bundle,
        selected,
        parent_fit_metadata_sha256=FIT_HASH,
        producer_checkpoint_sha256=PRODUCER_HASH,
        producer_metrics_sha256=METRICS_HASH,
        canonical_metadata_sha256=CANONICAL_HASH,
        split_manifest_sha256=SPLIT_HASH,
        source_closure=_verify_source_manifest(),
    )
    report = audit_capacity_bundle_values(
        bundle,
        expected_metadata_sha256=sha256_file(bundle / "metadata.json"),
        fit_records=fit_records,
        fit_metadata_sha256=FIT_HASH,
        producer_checkpoint_sha256=PRODUCER_HASH,
        producer_metrics_sha256=METRICS_HASH,
        canonical_metadata_sha256=CANONICAL_HASH,
        split_manifest_sha256=SPLIT_HASH,
    )
    assert report["status"] == "GO"
    assert report["composition"] == {
        "beneficial": 256,
        "changed-neutral": 128,
        "harmful": 128,
    }
