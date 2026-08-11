from __future__ import annotations

import json
import random

import pytest
import torch

import sph.pcld_data as pcld_data
from sph.pcld_data import (
    PCLD_FORWARD_FIELDS,
    calibrate_epsilon_from_records,
    capacity_expected_j2_denominator,
    capacity_expected_j2_denominators,
    collate_pcld_records,
    compute_latent_scale,
    filter_effective_records,
    group_record_indices_by_prompt,
    load_manifest,
    pcld_forward_inputs,
    sample_prompt_balanced_records,
    select_balanced_smoke_records,
    validate_capacity_support_receipt,
    validate_sidecar_receipt,
    validate_manifest_source,
)


def make_record(sample_id: str, domain: str, offset: int = 0) -> dict:
    candidate_ids = torch.arange(16).view(1, 16).expand(16, 16).clone()
    gold = torch.zeros(16, dtype=torch.int32)
    hidden = torch.zeros(16, 2560, dtype=torch.bfloat16)
    target_hidden = torch.ones(16, 2560, dtype=torch.bfloat16)
    return {
        "sample_id": sample_id,
        "domain": domain,
        "anchor_offset": offset,
        "context_length": 8 + offset,
        "parallel_hidden": hidden,
        "base_topk_ids": candidate_ids.to(torch.int32),
        "base_topk_logits": torch.full((16, 16), -99.0, dtype=torch.float16),
        "gold_ids": gold,
        "target_top1_ids": gold.clone(),
        "policy_ids": gold.clone(),
        "pcld_base_logsumexp": torch.full((16,), 5.0),
        "pcld_base_candidate_logits": torch.arange(16).float().view(1, 16).expand(16, 16),
        "pcld_target_hidden": target_hidden,
        "pcld_target_candidate_logits": torch.randn(16, 16),
        "pcld_authoritative_top1_ids": gold.clone(),
        "pcld_fp32_top1_ids": gold.clone(),
        "pcld_target_top1_margins": torch.full((16,), 1.0),
        "pcld_centered_max_errors": torch.full((16,), 0.01),
    }


def test_collate_uses_exact_sidecar_logits_and_forward_whitelist() -> None:
    record = make_record("p0", "chat")
    epsilon = calibrate_epsilon_from_records([record])
    batch = collate_pcld_records(
        [record], epsilon_num=epsilon, require_effective=True
    )
    assert torch.equal(batch["candidate_logits"], record["pcld_base_candidate_logits"][None])
    assert not torch.equal(batch["candidate_logits"], record["base_topk_logits"].float()[None])
    weight = torch.randn(64, 2560, dtype=torch.bfloat16)
    online = pcld_forward_inputs(batch, weight)
    assert set(online) == PCLD_FORWARD_FIELDS
    assert all("target" not in name and "gold" not in name for name in online)


def test_collate_keeps_binding_legacy_j2_outside_stable_loss_support() -> None:
    record = make_record("p0", "chat")
    record["pcld_authoritative_top1_ids"][0] = 1
    record["pcld_fp32_top1_ids"][0] = 1
    epsilon = calibrate_epsilon_from_records([record])
    batch = collate_pcld_records(
        [record], epsilon_num=epsilon, require_effective=False
    )
    assert bool(batch["legacy_j2_target_matches_gold"][0, 0])
    assert not bool(batch["authoritative_j2_target_matches_gold"][0, 0])
    assert not bool(batch["authoritative_numeric_j2_target_matches_gold"][0, 0])
    assert not bool(batch["stable_j2_target_matches_gold"][0, 0])
    assert not bool(batch["support_mask"].any())
    weight = torch.randn(64, 2560, dtype=torch.bfloat16)
    assert set(pcld_forward_inputs(batch, weight)) == PCLD_FORWARD_FIELDS


def test_effective_filter_latent_scale_and_prompt_sampling() -> None:
    records = [
        make_record("p0", "chat", 0),
        make_record("p0", "chat", 1),
        make_record("p1", "code", 0),
    ]
    epsilon = calibrate_epsilon_from_records(records)
    effective = filter_effective_records(records, epsilon)
    scale, rows = compute_latent_scale(effective, epsilon)
    assert rows == 48
    assert scale.shape == (2560,)
    assert torch.all(scale >= 1e-3)
    groups = group_record_indices_by_prompt(effective)
    first = sample_prompt_balanced_records(
        effective, groups, batch_size=20, rng=random.Random(7)
    )
    second = sample_prompt_balanced_records(
        effective, groups, batch_size=20, rng=random.Random(7)
    )
    assert [(item["sample_id"], item["anchor_offset"]) for item in first] == [
        (item["sample_id"], item["anchor_offset"]) for item in second
    ]


def test_balanced_smoke_uses_three_domains_and_distinct_prompts() -> None:
    records = []
    for domain in ("chat", "code", "math"):
        for index in range(4):
            records.append(make_record(f"{domain}-{index}", domain))
    selected = select_balanced_smoke_records(records, count=9)
    assert {record["domain"] for record in selected} == {"chat", "code", "math"}
    assert len({record["sample_id"] for record in selected}) == 9


def test_manual_prefix_receipt_is_fail_closed(tmp_path) -> None:
    metadata = {
        "records": 32,
        "source_rollout": "/rollout",
        "target": "/target",
        "split": "train",
        "group": "smoke32",
    }
    report = {
        "format": "pcld_sidecar_replay_v1",
        "verified": True,
        "base_lattice_exact": True,
        "target_hidden_allclose": True,
        "target_candidate_scores_allclose": True,
        "numeric_authority_exact": True,
        "records": 32,
        "sidecar": str(tmp_path.resolve()),
        "source_rollout": "/rollout",
        "target": "/target",
        "split": "train",
        "group": "smoke32",
        "manual_parity_records": 32,
        "manual_parity_passed": True,
        "manual_row_alignment_exact": True,
        "manual_stable_top1_exact": True,
        "manual_row0_alignment_exact": True,
        "manual_row15_alignment_exact": True,
        "manual_row0_stable_top1_exact": True,
        "manual_row15_stable_top1_exact": True,
        # Raw BF16 tie-breaking is reported but is not relabeled as stable.
        "manual_top1_exact": False,
        "manual_top1_mismatches": 2,
    }
    receipt = tmp_path / "replay_report.json"
    receipt.write_text(json.dumps(report))
    assert validate_sidecar_receipt(
        tmp_path, metadata, require_manual_records=32
    ) == report

    for field in (
        "manual_parity_passed",
        "manual_row_alignment_exact",
        "manual_stable_top1_exact",
        "manual_row0_alignment_exact",
        "manual_row15_alignment_exact",
        "manual_row0_stable_top1_exact",
        "manual_row15_stable_top1_exact",
    ):
        broken = dict(report)
        broken[field] = False
        receipt.write_text(json.dumps(broken))
        with pytest.raises(RuntimeError, match="manual-prefix"):
            validate_sidecar_receipt(
                tmp_path, metadata, require_manual_records=32
            )

    short = dict(report)
    short["manual_parity_records"] = 31
    receipt.write_text(json.dumps(short))
    with pytest.raises(RuntimeError, match="coverage"):
        validate_sidecar_receipt(tmp_path, metadata, require_manual_records=32)

    for field in ("target", "split", "group"):
        broken = dict(report)
        broken[field] = "wrong"
        receipt.write_text(json.dumps(broken))
        with pytest.raises(RuntimeError, match=field):
            validate_sidecar_receipt(
                tmp_path, metadata, require_manual_records=32
            )


def test_manifest_source_partitions_and_capacity_denominator_are_bound(tmp_path) -> None:
    manifest = {
        "format": "japd_manifest_v1",
        "complete": True,
        "source_rollout": str((tmp_path / "rollout").resolve()),
        "source_split": "train",
        "label_fields_used_for_selection": [],
        "prompt_splits": {
            "fit": [f"fit-{index}" for index in range(1589)],
            "select": [f"select-{index}" for index in range(199)],
            "diagnostic": [f"diagnostic-{index}" for index in range(199)],
        },
        "capacity": {"strict_multi_repair_blocks_diagnostic_only": 411},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    loaded = load_manifest(path)
    validate_manifest_source(
        loaded, rollout=tmp_path / "rollout", split="train"
    )
    assert capacity_expected_j2_denominator(loaded) == 411
    assert capacity_expected_j2_denominators(loaded) == {
        "legacy": 411,
        "authoritative": 403,
        "authoritative_numeric": 402,
        "stable": 314,
    }

    with pytest.raises(RuntimeError, match="source rollout"):
        validate_manifest_source(
            loaded, rollout=tmp_path / "other", split="train"
        )
    overlapping = dict(manifest)
    overlapping["prompt_splits"] = dict(manifest["prompt_splits"])
    overlapping["prompt_splits"]["select"] = list(
        manifest["prompt_splits"]["select"]
    )
    overlapping["prompt_splits"]["select"][0] = "fit-0"
    path.write_text(json.dumps(overlapping))
    with pytest.raises(RuntimeError, match="overlap"):
        load_manifest(path)


def test_capacity_support_receipt_is_exact_and_fail_closed(
    tmp_path, monkeypatch
) -> None:
    expected = {
        "format": "pcld_capacity_support_receipt_v1",
        "complete": True,
        "source_rollout": str((tmp_path / "rollout").resolve()),
        "source_manifest": str((tmp_path / "manifest").resolve()),
        "target": str((tmp_path / "target").resolve()),
        "sidecar": str((tmp_path / "sidecar").resolve()),
        "split": "train",
        "group": "capacity",
        "sidecar_replay_verified": True,
        "support": {
            "records": 512,
            "semantic_keys": [["p0", 0, 8]],
            "epsilon_num": 0.24676132202148438,
            "margin_threshold": 0.49352264404296875,
            "branches": {
                "legacy": {"j2_denominator": 411, "eligible_keys": []},
                "authoritative": {"j2_denominator": 403, "eligible_keys": []},
                "authoritative_numeric": {
                    "j2_denominator": 402,
                    "eligible_keys": [],
                },
                "stable": {"j2_denominator": 314, "eligible_keys": []},
            },
            "stable_effective_blocks": 503,
            "stable_support_rows": 4754,
            "stable_horizons": [{"key": ["p0", 0, 8], "horizon": 9}],
        },
    }
    monkeypatch.setattr(
        pcld_data, "build_capacity_support_receipt", lambda *args, **kwargs: expected
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(expected))
    kwargs = {
        "rollout": tmp_path / "rollout",
        "manifest": tmp_path / "manifest",
        "target": tmp_path / "target",
        "sidecar": tmp_path / "sidecar",
        "split": "train",
        "group": "capacity",
        "replay_report": {"verified": True},
    }
    assert validate_capacity_support_receipt(
        path, [], 0.24676132202148438, **kwargs
    ) == expected

    corruptions = [
        ("source_rollout", "/wrong"),
        ("target", "/wrong"),
        ("split", "validation"),
        ("group", "fit"),
        ("sidecar_replay_verified", False),
    ]
    for field, value in corruptions:
        broken = json.loads(json.dumps(expected))
        broken[field] = value
        path.write_text(json.dumps(broken))
        with pytest.raises(RuntimeError, match="differs"):
            validate_capacity_support_receipt(
                path, [], 0.24676132202148438, **kwargs
            )
    for field, value in (
        ("records", 511),
        ("epsilon_num", 0.25),
        ("stable_effective_blocks", 502),
        ("stable_support_rows", 4753),
        ("semantic_keys", []),
        ("stable_horizons", []),
    ):
        broken = json.loads(json.dumps(expected))
        broken["support"][field] = value
        path.write_text(json.dumps(broken))
        with pytest.raises(RuntimeError, match="differs"):
            validate_capacity_support_receipt(
                path, [], 0.24676132202148438, **kwargs
            )
    broken = json.loads(json.dumps(expected))
    broken["support"]["branches"]["legacy"]["j2_denominator"] = 314
    path.write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match="differs"):
        validate_capacity_support_receipt(
            path, [], 0.24676132202148438, **kwargs
        )
    path.write_text(json.dumps(expected))
    with pytest.raises(RuntimeError, match="verified"):
        validate_capacity_support_receipt(
            path,
            [],
            0.24676132202148438,
            **{**kwargs, "replay_report": {"verified": False}},
        )
