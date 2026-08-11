from __future__ import annotations

import json
from pathlib import Path

import pytest

from train_japd16 import (
    capacity_manifest_keys,
    cosine_learning_rate,
    full_fit_manifest_prompts,
    gate_result,
    selection_evaluation_step,
    select_group,
    should_replace_checkpoint,
    validate_m1_train_eval_alignment,
    validate_sidecar_replay_receipt,
    validate_sidecar_source,
)


def test_cosine_schedule_hits_frozen_endpoints() -> None:
    assert cosine_learning_rate(
        1,
        total_steps=1000,
        warmup_steps=200,
        peak=3e-4,
        minimum=3e-5,
    ) == pytest.approx(1.5e-6)
    assert cosine_learning_rate(
        200,
        total_steps=1000,
        warmup_steps=200,
        peak=3e-4,
        minimum=3e-5,
    ) == pytest.approx(3e-4)
    assert cosine_learning_rate(
        1000,
        total_steps=1000,
        warmup_steps=200,
        peak=3e-4,
        minimum=3e-5,
    ) == pytest.approx(3e-5)


def test_capacity_and_full_fit_gates_are_exact_conjunctions() -> None:
    capacity = {
        "j2_prompt_balanced": 0.99,
        "oracle_gap_recovered": 0.95,
        "harmed_fraction": 0.01,
    }
    assert all(gate_result(capacity, "capacity").values())
    capacity["harmed_fraction"] = 0.0100001
    assert not all(gate_result(capacity, "capacity").values())

    full_fit = {"j2_prompt_balanced": 0.90, "oracle_gap_recovered": 0.80}
    assert all(gate_result(full_fit, "full_fit").values())
    full_fit["j2_prompt_balanced"] = 0.899999
    assert not all(gate_result(full_fit, "full_fit").values())


def make_m1_manifest() -> dict:
    capacity = [
        {
            "sample_id": f"capacity-{index}",
            "anchor_offset": 0,
            "context_length": 10 + index,
        }
        for index in range(512)
    ]
    full_fit = [f"full-fit-{index}" for index in range(512)]
    return {
        "capacity": {"records": capacity},
        "full_fit_diagnostic": {"prompts": full_fit},
    }


def test_m1_manifest_rejects_duplicate_missing_and_overlap() -> None:
    manifest = make_m1_manifest()
    assert len(capacity_manifest_keys(manifest)) == 512
    assert len(full_fit_manifest_prompts(manifest)) == 512
    manifest["capacity"]["records"][-1] = dict(
        manifest["capacity"]["records"][0]
    )
    with pytest.raises(RuntimeError, match="unique"):
        capacity_manifest_keys(manifest)

    manifest = make_m1_manifest()
    manifest["full_fit_diagnostic"]["prompts"].pop()
    with pytest.raises(RuntimeError, match="exactly 512"):
        full_fit_manifest_prompts(manifest)

    manifest = make_m1_manifest()
    manifest["full_fit_diagnostic"]["prompts"][0] = "capacity-0"
    with pytest.raises(RuntimeError, match="overlap"):
        full_fit_manifest_prompts(manifest)


def test_m1_train_eval_alignment_rejects_different_record_sets() -> None:
    manifest = make_m1_manifest()
    records = [
        {
            "sample_id": item["sample_id"],
            "anchor_offset": item["anchor_offset"],
            "context_length": item["context_length"],
        }
        for item in manifest["capacity"]["records"]
    ]
    validate_m1_train_eval_alignment(
        "capacity", "capacity", records, list(records), manifest
    )
    with pytest.raises(RuntimeError, match="sets differ"):
        validate_m1_train_eval_alignment(
            "capacity", "capacity", records, records[:-1], manifest
        )


def test_checkpoint_cadence_excludes_step1_and_ties_keep_earlier() -> None:
    assert not selection_evaluation_step(
        1, total_steps=501, eval_every_steps=250
    )
    assert selection_evaluation_step(
        250, total_steps=501, eval_every_steps=250
    )
    assert selection_evaluation_step(
        501, total_steps=501, eval_every_steps=250
    )
    assert not should_replace_checkpoint(
        {"model_eal": 7.0}, {"model_eal": 7.0}
    )
    assert should_replace_checkpoint(
        {"model_eal": 7.0001}, {"model_eal": 7.0}
    )


def test_capacity_group_selects_realistic_512_exact_keys() -> None:
    manifest = make_m1_manifest()
    records = [
        {
            "sample_id": item["sample_id"],
            "anchor_offset": item["anchor_offset"],
            "context_length": item["context_length"],
        }
        for item in manifest["capacity"]["records"]
    ]
    selected = select_group(records, manifest, "capacity")
    assert len(selected) == 512
    assert {row["sample_id"] for row in selected} == {
        f"capacity-{index}" for index in range(512)
    }


def test_sidecar_source_binding_is_semantic_and_fail_closed(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout"
    target = tmp_path / "target"
    rollout.mkdir()
    target.mkdir()
    metadata = {
        "source_rollout": str(rollout.resolve()),
        "target": str(target.resolve()),
        "split": "train",
    }
    validate_sidecar_source(
        metadata, rollout=rollout, target=target, split="train"
    )
    with pytest.raises(RuntimeError, match="target mismatch"):
        validate_sidecar_source(
            {**metadata, "target": str(tmp_path / "wrong")},
            rollout=rollout,
            target=target,
            split="train",
        )


def test_sidecar_replay_receipt_is_required_and_semantic(tmp_path: Path) -> None:
    root = tmp_path / "sidecar"
    root.mkdir()
    metadata = {
        "records": 32,
        "source_rollout": "/rollout",
        "split": "train",
    }
    with pytest.raises(RuntimeError, match="lacks replay receipt"):
        validate_sidecar_replay_receipt(root, metadata)
    receipt = {
        "format": "japd_base_lse_replay_v1",
        "verified": True,
        "top16_ids_exact": True,
        "stored_dtype_top16_logits_exact": True,
        "five_scalar_channels_allclose": True,
        "audit_head_scores_allclose": True,
        "selected_tokens_exact": True,
        "selected_token_mismatches": 0,
        "records": 32,
        "source_rollout": "/rollout",
        "sidecar": str(root.resolve()),
        "split": "train",
    }
    (root / "replay_report.json").write_text(json.dumps(receipt))
    assert validate_sidecar_replay_receipt(root, metadata) == receipt
    receipt["selected_token_mismatches"] = 1
    (root / "replay_report.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="selected-token mismatch"):
        validate_sidecar_replay_receipt(root, metadata)
