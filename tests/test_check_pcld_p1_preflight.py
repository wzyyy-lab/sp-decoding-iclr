from __future__ import annotations

import copy

import pytest

from check_pcld_p1_preflight import (
    REQUIRED_COMPLETE_COMPONENTS,
    REQUIRED_MANUAL_FIELDS,
    REQUIRED_MECHANICS_CHECKS,
    REQUIRED_PROFILE_CHECKS,
    validate_p0_receipts,
)
from sph.pcld import EXPECTED_PARAMETER_COUNT


def receipts() -> tuple[dict, dict]:
    mechanics = {
        "format": "pcld16_mechanics_v1",
        "passed": True,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "records": 4,
        "checks": {name: True for name in REQUIRED_MECHANICS_CHECKS},
        "sidecar_receipt": {
            "verified": True,
            "manual_parity_records": 32,
            **{name: True for name in REQUIRED_MANUAL_FIELDS},
        },
    }
    profile = {
        "format": "pcld16_eager_profile_v1",
        "status": "completed",
        "device": "NVIDIA A40",
        "execution": {
            "batch": 1,
            "positions": 16,
            "candidates": 16,
            "mode": "eager",
            "complete_pcld_components": list(REQUIRED_COMPLETE_COMPONENTS),
        },
        "parameters": {"pcld_trainable": EXPECTED_PARAMETER_COUNT},
        "checks": {name: True for name in REQUIRED_PROFILE_CHECKS},
        "staged_complete_p50_ratio": 0.55,
        "standalone_complete_p50_ratio": 0.53,
        "complete_p50_ratio": 0.55,
        "development_latency_gate_passed": True,
    }
    return mechanics, profile


def test_p1_preflight_requires_both_monotonic_receipts() -> None:
    mechanics, profile = receipts()
    assert validate_p0_receipts(mechanics, profile)["passed"] is True
    corruptions = (
        ("mechanics", "passed", False),
        ("mechanics", "parameter_count", 1),
        ("profile", "complete_p50_ratio", 1.21),
        ("profile", "development_latency_gate_passed", False),
    )
    for side, field, value in corruptions:
        broken_mechanics = copy.deepcopy(mechanics)
        broken_profile = copy.deepcopy(profile)
        target = broken_mechanics if side == "mechanics" else broken_profile
        target[field] = value
        with pytest.raises(RuntimeError):
            validate_p0_receipts(broken_mechanics, broken_profile)

    missing_mechanics = copy.deepcopy(mechanics)
    missing_mechanics["checks"].pop(next(iter(REQUIRED_MECHANICS_CHECKS)))
    with pytest.raises(RuntimeError, match="schema"):
        validate_p0_receipts(missing_mechanics, profile)

    missing_profile = copy.deepcopy(profile)
    missing_profile["checks"].pop(next(iter(REQUIRED_PROFILE_CHECKS)))
    with pytest.raises(RuntimeError, match="schema"):
        validate_p0_receipts(mechanics, missing_profile)

    missing_manual = copy.deepcopy(mechanics)
    missing_manual["sidecar_receipt"].pop(next(iter(REQUIRED_MANUAL_FIELDS)))
    with pytest.raises(RuntimeError, match="manual"):
        validate_p0_receipts(missing_manual, profile)

    missing_component = copy.deepcopy(profile)
    missing_component["execution"]["complete_pcld_components"].pop()
    with pytest.raises(RuntimeError, match="component"):
        validate_p0_receipts(mechanics, missing_component)

    nan_ratio = copy.deepcopy(profile)
    nan_ratio["complete_p50_ratio"] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_p0_receipts(mechanics, nan_ratio)
