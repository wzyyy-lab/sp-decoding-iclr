#!/usr/bin/env python3
"""Monotonic P0 receipt gate before any PCLD P1 sidecar or training work."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from sph.pcld import EXPECTED_PARAMETER_COUNT


REQUIRED_MECHANICS_CHECKS = frozenset(
    {
        "sidecar_manual_parity_32",
        "sidecar_row_geometry_full16",
        "residual_cancellation_within_fp32_tolerance",
        "parameter_count_exact",
        "score_shape_full16_k16",
        "proposal_shape_one_chain",
        "production_encoder_mask_none",
        "production_query_mask_none",
        "causal_scope_rejected",
        "production_forward_has_no_target_fields",
        "zero_score_identity",
        "zero_selected_rank_identity",
        "remote_changes_global_state",
        "zero_u_keeps_scores_fixed",
        "nonzero_u_exposes_remote_score_effect",
        "step0_residual_gradient_finite_nonzero",
        "step0_upstream_gradient_zero",
        "one_update_changes_u",
        "step1_upstream_gradient_finite_nonzero",
    }
)
REQUIRED_PROFILE_CHECKS = frozenset(
    {
        "full16_hidden",
        "score_shape_full16_k16",
        "pcld_identity_matches_base",
        "domino_matches_cached_policy",
        "pcld_complete_matches_incremental",
        "domino_complete_matches_incremental",
    }
)
REQUIRED_MANUAL_FIELDS = frozenset(
    {
        "manual_parity_passed",
        "manual_row_alignment_exact",
        "manual_stable_top1_exact",
        "manual_row0_alignment_exact",
        "manual_row15_alignment_exact",
        "manual_row0_stable_top1_exact",
        "manual_row15_stable_top1_exact",
    }
)
REQUIRED_COMPLETE_COMPONENTS = (
    "base_vocab_gemm",
    "fp32_top16",
    "fp32_logsumexp",
    "lm_head_candidate_gather",
    "global_noncausal_head",
    "residual_lm_head_dot",
    "per_position_argmax",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanics", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    return parser.parse_args()


def validate_p0_receipts(
    mechanics: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    if mechanics.get("format") != "pcld16_mechanics_v1":
        raise RuntimeError("unsupported PCLD mechanics receipt")
    if mechanics.get("passed") is not True:
        raise RuntimeError("PCLD mechanics receipt did not pass")
    if int(mechanics.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("PCLD mechanics parameter count mismatch")
    if int(mechanics.get("records", -1)) != 4:
        raise RuntimeError("PCLD mechanics must cover the frozen four-block probe")
    mechanics_checks = mechanics.get("checks")
    if not isinstance(mechanics_checks, dict) or set(mechanics_checks) != REQUIRED_MECHANICS_CHECKS:
        raise RuntimeError("PCLD mechanics receipt check schema mismatch")
    failed_mechanics = [
        name for name, passed in mechanics_checks.items() if passed is not True
    ]
    if failed_mechanics:
        raise RuntimeError(f"PCLD mechanics checks failed: {failed_mechanics}")
    sidecar_receipt = mechanics.get("sidecar_receipt")
    if not isinstance(sidecar_receipt, dict) or sidecar_receipt.get("verified") is not True:
        raise RuntimeError("PCLD mechanics lacks a verified sidecar receipt")
    if int(sidecar_receipt.get("manual_parity_records", -1)) != 32:
        raise RuntimeError("PCLD mechanics lacks the 32-record manual receipt")
    if sidecar_receipt.get("manual_parity_passed") is not True:
        raise RuntimeError("PCLD manual teacher geometry did not pass")
    failed_manual = [
        name for name in REQUIRED_MANUAL_FIELDS
        if sidecar_receipt.get(name) is not True
    ]
    if failed_manual:
        raise RuntimeError(f"PCLD manual receipt failed: {failed_manual}")

    if profile.get("format") != "pcld16_eager_profile_v1":
        raise RuntimeError("unsupported PCLD eager profile receipt")
    if profile.get("status") != "completed":
        raise RuntimeError("PCLD eager profile is incomplete")
    if "A40" not in str(profile.get("device", "")):
        raise RuntimeError("PCLD development profile did not run on A40")
    execution = profile.get("execution")
    if not isinstance(execution, dict) or (
        int(execution.get("batch", -1)) != 1
        or int(execution.get("positions", -1)) != 16
        or int(execution.get("candidates", -1)) != 16
        or execution.get("mode") != "eager"
    ):
        raise RuntimeError("PCLD eager profile execution contract mismatch")
    if int(profile.get("parameters", {}).get("pcld_trainable", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("PCLD profile parameter count mismatch")
    profile_checks = profile.get("checks")
    if not isinstance(profile_checks, dict) or set(profile_checks) != REQUIRED_PROFILE_CHECKS:
        raise RuntimeError("PCLD profile parity-check schema mismatch")
    failed_profile = [
        name for name, passed in profile_checks.items() if passed is not True
    ]
    if failed_profile:
        raise RuntimeError(f"PCLD profile parity checks failed: {failed_profile}")
    components = execution.get("complete_pcld_components")
    if (
        tuple(components) if isinstance(components, list) else None
    ) != REQUIRED_COMPLETE_COMPONENTS:
        raise RuntimeError("PCLD complete profile component list mismatch")
    ratios = {
        "staged": float(profile.get("staged_complete_p50_ratio", float("nan"))),
        "standalone": float(
            profile.get("standalone_complete_p50_ratio", float("nan"))
        ),
        "complete": float(profile.get("complete_p50_ratio", float("nan"))),
    }
    if any(not math.isfinite(value) or value < 0 for value in ratios.values()):
        raise RuntimeError("PCLD complete eager latency ratio is non-finite")
    ratio = ratios["complete"]
    if not math.isclose(
        ratio, max(ratios["staged"], ratios["standalone"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("PCLD conservative complete ratio is inconsistent")
    if profile.get("development_latency_gate_passed") is not True or ratio > 1.20:
        raise RuntimeError("PCLD complete eager latency gate failed")
    return {
        "format": "pcld16_p1_preflight_v1",
        "passed": True,
        "mechanics_records": int(mechanics["records"]),
        "manual_parity_records": int(sidecar_receipt["manual_parity_records"]),
        "profile_device": str(profile["device"]),
        "complete_p50_ratio": ratio,
    }


def main() -> None:
    args = parse_args()
    mechanics = json.loads(args.mechanics.read_text())
    profile = json.loads(args.profile.read_text())
    print(
        json.dumps(
            validate_p0_receipts(mechanics, profile),
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
