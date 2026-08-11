from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
import torch

import scripts.audit_direct_safety_artifacts as independent_auditor
from sph.direct_safety_gate import direct_safety_position_features
from sph.direct_safety_numeric_policy import (
    EXPECTED_NUMERIC_POLICY_SHA256,
    NUMERIC_POLICY_ID,
    NUMERIC_POLICY_SHA256,
    assert_add_sub,
    assert_numeric_policy_binding,
    assert_portable_feature_relations,
    assert_retained_mass,
    assert_same_device_feature_invariants,
    numeric_policy_receipt,
)
from sph.global_direct_selector import GlobalDirectOutput


PROJECT = Path(__file__).resolve().parents[1]


def _numeric_fixture(
    device: torch.device | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    device = torch.device("cpu") if device is None else device
    logits = (
        -torch.arange(16, dtype=torch.float32)[None].expand(15, -1).clone()
    ).to(device)
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
        1,
        15,
        16,
        64,
        generator=torch.Generator().manual_seed(20260805),
    ).to(device)
    generated = direct_safety_position_features(
        states, output, logits[None], full_lse[None]
    )
    return (
        generated.position_features,
        generated.direct_path,
        generated.change_mask,
        states,
        logits[None],
        full_lse[None],
        scores[None],
        residual[None],
        base[None],
    )


def test_policy_identity_is_canonical_pinned_and_independently_duplicated() -> None:
    receipt = numeric_policy_receipt()
    assert NUMERIC_POLICY_SHA256 == EXPECTED_NUMERIC_POLICY_SHA256
    assert receipt["id"] == NUMERIC_POLICY_ID
    assert receipt["sha256"] == NUMERIC_POLICY_SHA256
    assert independent_auditor._audit_numeric_policy_receipt() == receipt


def test_independent_auditor_does_not_import_production_policy_or_validator() -> None:
    source = (PROJECT / "scripts/audit_direct_safety_artifacts.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "sph.direct_safety_numeric_policy" not in imports
    assert "sph.direct_safety_artifacts" not in imports


def test_same_device_invariants_cover_fifteen_relations_and_fail_on_tamper() -> None:
    values = _numeric_fixture()
    assert assert_same_device_feature_invariants(*values) == 15
    tampered = list(values)
    tampered[0] = tampered[0].clone()
    tampered[0][0, 0, 198] += 1.0e-4
    with pytest.raises(ValueError, match="same-device position features"):
        assert_same_device_feature_invariants(*tampered)


def test_portable_relations_accept_presealed_retained_scale_but_reject_material() -> None:
    values = _numeric_fixture()
    features, path, changed, _, logits, full_lse, scores, residual, base = values
    assert_portable_feature_relations(
        features[0],
        path[0],
        changed[0],
        logits[0],
        full_lse[0],
        scores[0],
        residual[0],
        base[0],
    )
    shifted_logits = logits[0] + 32.0
    shifted_lse = full_lse[0] + 32.0
    retained = torch.tanh(
        (torch.logsumexp(shifted_logits, dim=-1) - shifted_lse) / 2.0
    )
    portable_retained = retained + torch.tensor(1.9073486328125e-06)
    assert_retained_mass(
        "retained-mass", portable_retained, shifted_logits, shifted_lse
    )
    independent_auditor._audit_retained_mass(
        "retained-mass", portable_retained, shifted_logits, shifted_lse
    )
    with pytest.raises(ValueError, match="retained-mass"):
        assert_retained_mass(
            "retained-mass",
            retained + 1.0e-4,
            shifted_logits,
            shifted_lse,
        )
    with pytest.raises(RuntimeError, match="retained-mass"):
        independent_auditor._audit_retained_mass(
            "retained-mass",
            retained + 1.0e-4,
            shifted_logits,
            shifted_lse,
        )


def test_add_sub_policy_accepts_two_source_ulps_and_rejects_material_delta() -> None:
    left = torch.tensor([1.0], dtype=torch.float32)
    right = torch.tensor([2.0**-20], dtype=torch.float32)
    reference = (left.to(torch.float64) + right.to(torch.float64)).to(torch.float32)
    accepted = reference.clone()
    for _ in range(2):
        accepted = torch.nextafter(accepted, torch.full_like(accepted, torch.inf))
    assert_add_sub("sum", accepted, left, right, operation="add")
    with pytest.raises(ValueError, match="sum"):
        assert_add_sub(
            "sum",
            reference + torch.tensor([1.0e-4], dtype=torch.float32),
            left,
            right,
            operation="add",
        )


def test_policy_binding_fails_closed_on_missing_or_different_identity() -> None:
    assert_numeric_policy_binding(NUMERIC_POLICY_ID, NUMERIC_POLICY_SHA256)
    with pytest.raises(ValueError, match="policy ID"):
        assert_numeric_policy_binding(None, NUMERIC_POLICY_SHA256)
    with pytest.raises(ValueError, match="policy digest"):
        assert_numeric_policy_binding(NUMERIC_POLICY_ID, "0" * 64)


def test_cuda_same_device_and_portable_independent_roundtrip() -> None:
    if not torch.cuda.is_available():
        if os.environ.get("PROS_REQUIRE_CUDA") == "1":
            pytest.fail("PROS numeric-v2 smoke requires an allocated CUDA device")
        pytest.skip("CUDA is unavailable")
    values = _numeric_fixture(torch.device("cuda"))
    assert assert_same_device_feature_invariants(*values) == 15
    features, path, changed, _, logits, full_lse, scores, residual, base = [
        value.detach().cpu() for value in values
    ]
    assert_portable_feature_relations(
        features[0],
        path[0],
        changed[0],
        logits[0],
        full_lse[0],
        scores[0],
        residual[0],
        base[0],
    )
    independent_auditor._audit_portable_feature_relations(
        features[0],
        path[0],
        changed[0],
        logits[0],
        full_lse[0],
        scores[0],
        residual[0],
        base[0],
    )
