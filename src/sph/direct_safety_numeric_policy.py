"""Versioned numeric contract for portable PROS-Gate outcome artifacts.

The producer first proves exact same-device relations before copying tensors
to CPU.  Persisted tensors are then checked against operation-aware envelopes
whose constants were frozen before the label-blind CUDA diagnostic.  The
independent artifact auditor deliberately carries a separate implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import torch
from torch import Tensor


NUMERIC_POLICY_PROTOCOL = "pros-gate-cross-device-numeric-policy-spec-v2"
NUMERIC_POLICY_ID = "pros-gate-cross-device-numeric-policy-v2"
EPS32 = 2.0**-23
ADD_SUB_ULPS = 2
ADD_SUB_HALF_WIDTH_CAP = 2.0**-14
ENTROPY_ABS_ENVELOPE = 2.0**-17
LSE_ULPS = 8
LSE_ABS_FLOOR = 2.0**-20
RETAINED_OUTER_ULPS = 2
RETAINED_OUTER_FLOOR = 2.0**-20
MAX_LSE_SOURCE_ULP = 2.0**-16
MATERIAL_MUTATION = 1.0e-4
SAME_DEVICE_FEATURE_RELATION_COUNT = 15

NUMERIC_POLICY_SPEC: dict[str, Any] = {
    "protocol": NUMERIC_POLICY_PROTOCOL,
    "id": NUMERIC_POLICY_ID,
    "format": "IEEE-754 binary32 persisted tensors; float64 references",
    "field_classes": {
        "exact": [
            "direct_path",
            "change_mask",
            "selected_state_copy",
            "base_state_copy",
            "change_scalar",
        ],
        "add_sub": [
            "state_difference",
            "base_log_probs",
            "direct_scores",
            "total_margin",
            "residual_margin",
            "dflash_margin",
        ],
        "normalized_neighbor": ["rank", "position"],
        "entropy": ["entropy"],
        "retained_mass": ["retained_mass"],
    },
    "producer_same_device": {
        "relation_count": SAME_DEVICE_FEATURE_RELATION_COUNT,
        "rule": "bitwise exact reconstruction before host copy",
    },
    "relations": {
        "exact": {
            "rule": "bitwise equality",
        },
        "normalized_neighbor": {
            "endpoints": "0 and 1 exact",
            "interior_float32_neighbors": 1,
            "range": "[0,1]",
        },
        "add_sub": {
            "source_scale_floor": 1.0,
            "half_width_source_ulps": ADD_SUB_ULPS,
            "half_width_cap_hex": ADD_SUB_HALF_WIDTH_CAP.hex(),
        },
        "entropy": {
            "absolute_envelope_hex": ENTROPY_ABS_ENVELOPE.hex(),
            "range": "[0,1]",
            "normalizer": "ln(16)",
        },
        "retained_mass": {
            "lse_envelope": "8*ulp32(scale)+2^-20",
            "lse_source_ulp_cap_hex": MAX_LSE_SOURCE_ULP.hex(),
            "outer_float32_neighbors": RETAINED_OUTER_ULPS,
            "outer_floor_hex": RETAINED_OUTER_FLOOR.hex(),
            "analytic_cap": "E_lse/2+2^-20+4*2^-23",
            "analytic_cap_strict_upper_hex": MATERIAL_MUTATION.hex(),
            "subset_invariant": "logsumexp(top16)<=base_logsumexp+E_lse",
            "range": "[-1, propagated_upper]",
        },
    },
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


NUMERIC_POLICY_SHA256 = hashlib.sha256(
    _canonical_json_bytes(NUMERIC_POLICY_SPEC)
).hexdigest()
EXPECTED_NUMERIC_POLICY_SHA256 = (
    "cbd80345e7249707931f71b29c65722ec8910263d51b7d649c5dd5c04fc4d4f0"
)


def numeric_policy_receipt() -> dict[str, Any]:
    """Return the complete canonical policy identity stored in metadata."""

    encoded = _canonical_json_bytes(NUMERIC_POLICY_SPEC)
    observed = hashlib.sha256(encoded).hexdigest()
    if (
        NUMERIC_POLICY_SHA256 != EXPECTED_NUMERIC_POLICY_SHA256
        or observed != EXPECTED_NUMERIC_POLICY_SHA256
    ):
        raise RuntimeError("numeric policy digest differs from the reviewed pin")
    return {
        "protocol": NUMERIC_POLICY_PROTOCOL,
        "id": NUMERIC_POLICY_ID,
        "sha256": NUMERIC_POLICY_SHA256,
        "spec": json.loads(encoded),
    }


def assert_numeric_policy_binding(
    policy_id: Any,
    policy_sha256: Any,
) -> None:
    if policy_id != NUMERIC_POLICY_ID:
        raise ValueError("outcome numeric policy ID differs")
    if policy_sha256 != NUMERIC_POLICY_SHA256:
        raise ValueError("outcome numeric policy digest differs")


def _require_float32(name: str, value: Tensor) -> None:
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite float32")


def _require_same_shape(name: str, actual: Tensor, expected: Tensor) -> None:
    if actual.shape != expected.shape or actual.device != expected.device:
        raise ValueError(f"{name} shape/device differs")


def _broadcast_float64(value: Tensor, reference: Tensor) -> Tensor:
    return torch.broadcast_to(value.to(dtype=torch.float64), reference.shape)


def _source_scale_ulp32(
    operands: Sequence[Tensor], reference64: Tensor
) -> Tensor:
    reference = reference64.to(dtype=torch.float64)
    scale = torch.ones_like(reference)
    for operand in operands:
        scale = torch.maximum(scale, _broadcast_float64(operand, reference).abs())
    scale = torch.maximum(scale, reference.abs())
    scale32 = scale.to(dtype=torch.float32)
    ulp = (
        torch.nextafter(scale32, torch.full_like(scale32, torch.inf)).to(
            dtype=torch.float64
        )
        - scale32.to(dtype=torch.float64)
    )
    if not bool(torch.isfinite(ulp).all()) or not bool(ulp.gt(0.0).all()):
        raise ValueError("numeric-policy source ULP is invalid")
    return ulp


def assert_exact(name: str, actual: Tensor, expected: Tensor) -> None:
    _require_same_shape(name, actual, expected)
    if not torch.equal(actual, expected):
        raise ValueError(f"{name} is inconsistent")


def assert_add_sub(
    name: str,
    actual: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    operation: str,
) -> None:
    _require_float32(name, actual)
    _require_float32(f"{name} left operand", left)
    _require_float32(f"{name} right operand", right)
    left64 = left.to(dtype=torch.float64)
    right64 = right.to(dtype=torch.float64)
    if operation == "add":
        reference = left64 + right64
    elif operation == "subtract":
        reference = left64 - right64
    else:
        raise ValueError("numeric-policy operation must be add/subtract")
    _require_same_shape(name, actual, reference)
    ulp = _source_scale_ulp32((left, right), reference)
    half_width = ADD_SUB_ULPS * ulp
    accepted = (actual.to(dtype=torch.float64) - reference).abs().le(half_width)
    if not bool(half_width.le(ADD_SUB_HALF_WIDTH_CAP).all()):
        raise ValueError(f"{name} numeric-policy cap is exceeded")
    if not bool(accepted.all()):
        raise ValueError(f"{name} is inconsistent")


def assert_normalized_neighbor(
    name: str,
    actual: Tensor,
    expected: Tensor,
) -> None:
    _require_float32(name, actual)
    _require_float32(f"{name} reference", expected)
    _require_same_shape(name, actual, expected)
    endpoint = expected.eq(0.0) | expected.eq(1.0)
    lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
    upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
    accepted = actual.eq(expected) | (
        ~endpoint & (actual.eq(lower) | actual.eq(upper))
    )
    in_range = actual.ge(0.0) & actual.le(1.0)
    if not bool((accepted & in_range).all()):
        raise ValueError(f"{name} is inconsistent")


def _entropy_reference64(logits: Tensor) -> Tensor:
    logits64 = logits.to(dtype=torch.float64)
    log_q = logits64 - torch.logsumexp(logits64, dim=-1, keepdim=True)
    return -(log_q.exp() * log_q).sum(dim=-1) / math.log(16.0)


def assert_entropy(name: str, actual: Tensor, logits: Tensor) -> None:
    _require_float32(name, actual)
    _require_float32(f"{name} logits", logits)
    reference = _entropy_reference64(logits)
    _require_same_shape(name, actual, reference)
    actual64 = actual.to(dtype=torch.float64)
    accepted = (actual64 - reference).abs().le(ENTROPY_ABS_ENVELOPE)
    in_range = actual64.ge(0.0) & actual64.le(1.0)
    if not bool((accepted & in_range).all()):
        raise ValueError(f"{name} is inconsistent")


def _retained_bounds(
    logits: Tensor,
    base_logsumexp: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    logits64 = logits.to(dtype=torch.float64)
    base64 = base_logsumexp.to(dtype=torch.float64)
    lse64 = torch.logsumexp(logits64, dim=-1)
    maximum = logits64.amax(dim=-1).abs()
    scale = torch.maximum(torch.ones_like(lse64), maximum)
    scale = torch.maximum(scale, lse64.abs())
    scale = torch.maximum(scale, base64.abs())
    scale32 = scale.to(dtype=torch.float32)
    lse_source_ulp = (
        torch.nextafter(scale32, torch.full_like(scale32, torch.inf)).to(
            dtype=torch.float64
        )
        - scale32.to(dtype=torch.float64)
    )
    lse_envelope = LSE_ULPS * lse_source_ulp + LSE_ABS_FLOOR
    center = torch.tanh((lse64 - base64) / 2.0)
    lower = torch.tanh(((lse64 - lse_envelope) - base64) / 2.0).to(
        dtype=torch.float32
    )
    upper = torch.tanh(((lse64 + lse_envelope) - base64) / 2.0).to(
        dtype=torch.float32
    )
    for _ in range(RETAINED_OUTER_ULPS):
        lower = torch.nextafter(lower, torch.full_like(lower, -torch.inf))
        upper = torch.nextafter(upper, torch.full_like(upper, torch.inf))
    lower64 = lower.to(dtype=torch.float64) - RETAINED_OUTER_FLOOR
    upper64 = upper.to(dtype=torch.float64) + RETAINED_OUTER_FLOOR
    half_width = torch.maximum((center - lower64).abs(), (upper64 - center).abs())
    analytic_cap = lse_envelope / 2.0 + RETAINED_OUTER_FLOOR + 4.0 * EPS32
    cap_ok = (
        lse_source_ulp.le(MAX_LSE_SOURCE_ULP)
        & analytic_cap.lt(MATERIAL_MUTATION)
        & half_width.le(analytic_cap)
    )
    subset_ok = lse64.le(base64 + lse_envelope)
    return center, lower64, upper64, cap_ok, subset_ok


def assert_retained_mass(
    name: str,
    actual: Tensor,
    logits: Tensor,
    base_logsumexp: Tensor,
) -> None:
    _require_float32(name, actual)
    _require_float32(f"{name} logits", logits)
    _require_float32(f"{name} base logsumexp", base_logsumexp)
    center, lower, upper, cap_ok, subset_ok = _retained_bounds(
        logits, base_logsumexp
    )
    _require_same_shape(name, actual, center)
    actual64 = actual.to(dtype=torch.float64)
    finite = (
        torch.isfinite(center)
        & torch.isfinite(lower)
        & torch.isfinite(upper)
        & torch.isfinite(actual64)
    )
    accepted = actual64.ge(lower) & actual64.le(upper)
    in_range = actual64.ge(-1.0) & actual64.le(upper)
    if not bool((finite & cap_ok & subset_ok & accepted & in_range).all()):
        raise ValueError(f"{name} is inconsistent")


def assert_portable_feature_relations(
    position_features: Tensor,
    direct_path: Tensor,
    change_mask: Tensor,
    candidate_logits: Tensor,
    base_logsumexp: Tensor,
    direct_scores: Tensor,
    residual_scores: Tensor,
    base_log_probs: Tensor,
) -> None:
    """Validate persisted float relations using the versioned policy."""

    assert_exact("saved Direct path", direct_path, direct_scores.argmax(dim=-1))
    assert_exact("saved change mask", change_mask, direct_path.ne(0))
    assert_add_sub(
        "saved DFlash log probabilities",
        base_log_probs,
        candidate_logits,
        base_logsumexp[:, None],
        operation="subtract",
    )
    assert_add_sub(
        "saved Direct scores",
        direct_scores,
        base_log_probs,
        residual_scores,
        operation="add",
    )
    assert_add_sub(
        "saved selected/base state difference",
        position_features[:, 128:192],
        position_features[:, :64],
        position_features[:, 64:128],
        operation="subtract",
    )
    gather = direct_path[:, None]
    chosen_scores = direct_scores.gather(-1, gather).squeeze(-1)
    chosen_residual = residual_scores.gather(-1, gather).squeeze(-1)
    chosen_base = base_log_probs.gather(-1, gather).squeeze(-1)
    for name, actual, left, right in (
        (
            "saved total-margin feature",
            position_features[:, 192],
            chosen_scores,
            direct_scores[:, 0],
        ),
        (
            "saved residual-margin feature",
            position_features[:, 193],
            chosen_residual,
            residual_scores[:, 0],
        ),
        (
            "saved DFlash-margin feature",
            position_features[:, 194],
            chosen_base,
            base_log_probs[:, 0],
        ),
    ):
        assert_add_sub(name, actual, left, right, operation="subtract")
    assert_normalized_neighbor(
        "saved rank feature",
        position_features[:, 195],
        direct_path.float() / 15.0,
    )
    assert_normalized_neighbor(
        "saved position feature",
        position_features[:, 196],
        torch.arange(
            15, device=position_features.device, dtype=torch.float32
        )
        / 14.0,
    )
    assert_exact(
        "saved change scalar",
        position_features[:, 197],
        change_mask.float(),
    )
    assert_entropy(
        "saved entropy feature", position_features[:, 198], candidate_logits
    )
    assert_retained_mass(
        "saved retained-mass feature",
        position_features[:, 199],
        candidate_logits,
        base_logsumexp,
    )


def assert_same_device_feature_invariants(
    position_features: Tensor,
    direct_path: Tensor,
    change_mask: Tensor,
    node_states: Tensor,
    candidate_logits: Tensor,
    base_logsumexp: Tensor,
    direct_scores: Tensor,
    residual_scores: Tensor,
    base_log_probs: Tensor,
) -> int:
    """Recompute every generated feature relation exactly on one device."""

    tensors = (
        position_features,
        direct_path,
        change_mask,
        node_states,
        candidate_logits,
        base_logsumexp,
        direct_scores,
        residual_scores,
        base_log_probs,
    )
    devices = {value.device for value in tensors}
    if len(devices) != 1:
        raise ValueError("same-device invariant tensors span multiple devices")
    states = node_states.detach().float()
    scores = direct_scores.detach().float()
    residual = residual_scores.detach().float()
    base = base_log_probs.detach().float()
    logits = candidate_logits.detach().float()
    full_lse = base_logsumexp.detach().float()
    expected_path = scores.argmax(dim=-1)
    expected_change = expected_path.ne(0)
    gather_state = expected_path[..., None, None].expand(
        -1, -1, 1, states.shape[-1]
    )
    selected_states = states.gather(2, gather_state).squeeze(2)
    base_states = states[:, :, 0]
    gather = expected_path[..., None]
    chosen_scores = scores.gather(-1, gather).squeeze(-1)
    chosen_residual = residual.gather(-1, gather).squeeze(-1)
    chosen_base = base.gather(-1, gather).squeeze(-1)
    log_q = torch.log_softmax(logits, dim=-1)
    expected_features = torch.cat(
        [
            selected_states,
            base_states,
            selected_states - base_states,
            torch.stack(
                [
                    chosen_scores - scores[..., 0],
                    chosen_residual - residual[..., 0],
                    chosen_base - base[..., 0],
                    expected_path.float() / 15.0,
                    (
                        torch.arange(
                            15,
                            device=states.device,
                            dtype=torch.float32,
                        )
                        / 14.0
                    )[None].expand(states.shape[0], -1),
                    expected_change.float(),
                    -(log_q.exp() * log_q).sum(dim=-1) / math.log(16.0),
                    torch.tanh(
                        (torch.logsumexp(logits, dim=-1) - full_lse) / 2.0
                    ),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
    assert_exact("same-device Direct path", direct_path, expected_path)
    assert_exact("same-device change mask", change_mask, expected_change)
    assert_exact(
        "same-device DFlash log probabilities",
        base,
        logits - full_lse[..., None],
    )
    assert_exact("same-device Direct scores", scores, base + residual)
    assert_exact("same-device position features", position_features, expected_features)
    return SAME_DEVICE_FEATURE_RELATION_COUNT
