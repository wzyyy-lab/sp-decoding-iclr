#!/usr/bin/env python3
"""Label-blind CUDA/CPU portability diagnostic for PROS-Gate features.

The canonical shard format is a monolithic pickle.  ``CanonicalBlockDataset``
therefore physically deserializes ``gold_ids`` together with the numeric model
inputs.  This diagnostic never indexes, copies, stacks, hashes, compares,
branches on, passes onward, logs, or computes from that field.  A reviewed
allowlist extractor immediately replaces each selected raw mapping with a
typed numeric-only object; all subsequent code accepts only those objects.

This is a read-only, pre-registered diagnostic.  It emits one aggregate JSON
object to stdout and has no output-path argument.  Constants may not be tuned
from the observations produced here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch
from torch import Tensor, nn

from sph.data import CanonicalBlockDataset
from sph.direct_safety_gate import (
    direct_safety_position_features,
    freeze_direct_producer,
    frozen_direct_forward_with_states,
)
from sph.global_direct_selector import GlobalDirectCandidateSelector


PROTOCOL = "pros-gate-numeric-portability-diagnostic-v1"
NUMERIC_POLICY_ID = "pros-gate-cross-device-numeric-policy-pre-scan-v1"
EPS32 = 2.0**-23
ADD_SUB_HALF_WIDTH_CAP = 2.0**-14
MATERIAL_MUTATION = 1.0e-4
ADD_SUB_ULPS = 2
ENTROPY_ABS_ENVELOPE = 2.0**-17
LSE_ULPS = 8
LSE_ABS_FLOOR = 2.0**-20
RETAINED_OUTER_FLOOR = 2.0**-20
RETAINED_OUTER_ULPS = 2
MAX_LSE_SOURCE_ULP = 2.0**-16
SYNTHETIC_SEEDS = (79_079, 79_080)
SYNTHETIC_SHIFTS = (-64.0, -16.0, 0.0, 16.0, 64.0)
SYNTHETIC_STATE_MAGNITUDES = (0.0, 2.0**-20, 1.0, 32.0)
SYNTHETIC_PATH_RANKS = (0, 1, 15)
EXPECTED_DATA_METADATA_SHA256 = (
    "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320"
)
EXPECTED_DIRECT_CHECKPOINT_SHA256 = (
    "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e"
)
EXPECTED_DIRECT_METRICS_SHA256 = (
    "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef"
)
EXPECTED_SPLIT_PROTOCOL = "pros-gate-phase3-split-manifest-v1"
RAW_INPUT_FIELD_ALLOWLIST = frozenset(
    {
        "sample_id",
        "parallel_hidden",
        "base_topk_ids",
        "base_topk_logits",
        "base_logsumexp",
        "anchor_token_id",
    }
)
SCANNED_SPLITS = ("fit", "checkpoint")
EXPECTED_DIRECT_CONFIG = {
    "scope": "global",
    "mixer": "axial",
    "node_encoder": "additive",
    "candidate_k": 16,
    "model_dim": 64,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.0,
    "seed": 0,
}
EXPECTED_FIELD_NAMES = frozenset(
    {
        "anchor_ids_copy",
        "base_log_probs",
        "base_lse_copy",
        "base_state_copy",
        "candidate_ids_copy",
        "candidate_logits_copy",
        "change_mask",
        "change_scalar",
        "dflash_margin",
        "direct_path",
        "direct_scores",
        "entropy",
        "hidden_copy",
        "position",
        "rank",
        "residual_margin",
        "retained_mass",
        "selected_state_copy",
        "state_difference",
        "total_margin",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class NumericInput:
    """The only raw-record information allowed past the extractor."""

    sample_id: str
    hidden: Tensor
    candidate_ids: Tensor
    candidate_logits: Tensor
    base_logsumexp: Tensor
    anchor_ids: Tensor


@dataclass(frozen=True)
class NumericBatch:
    """Identifier-free numeric tensors accepted by model/diagnostic code."""

    hidden: Tensor
    candidate_ids: Tensor
    candidate_logits: Tensor
    base_logsumexp: Tensor
    anchor_ids: Tensor


def extract_sample_id(record: Mapping[str, Any]) -> str:
    """Read only the selection identity before deciding whether to extract."""

    sample_id = record["sample_id"]
    if not isinstance(sample_id, str) or not sample_id or "\0" in sample_id:
        raise RuntimeError("canonical numeric input has an invalid sample identity")
    return sample_id


def extract_numeric_input(record: Mapping[str, Any]) -> NumericInput:
    """Copy exactly the reviewed numeric allowlist into a typed object."""

    sample_id = record["sample_id"]
    hidden = record["parallel_hidden"]
    candidate_ids = record["base_topk_ids"]
    candidate_logits = record["base_topk_logits"]
    base_logsumexp = record["base_logsumexp"]
    anchor_token_id = record["anchor_token_id"]
    if not isinstance(sample_id, str) or not sample_id or "\0" in sample_id:
        raise RuntimeError("canonical numeric input has an invalid sample identity")
    for name, value in (
        ("hidden", hidden),
        ("candidate IDs", candidate_ids),
        ("candidate logits", candidate_logits),
        ("base logsumexp", base_logsumexp),
    ):
        if not isinstance(value, Tensor):
            raise RuntimeError(f"canonical {name} is not a tensor")
    if tuple(hidden.shape[:1]) != (15,) or hidden.ndim != 2:
        raise RuntimeError("canonical hidden input does not have shape [15,D]")
    candidate_k = int(EXPECTED_DIRECT_CONFIG["candidate_k"])
    if candidate_ids.ndim != 2 or candidate_logits.ndim != 2:
        raise RuntimeError("canonical candidates must have shape [15,K]")
    if candidate_ids.shape != candidate_logits.shape:
        raise RuntimeError("canonical candidate ID/logit shapes differ")
    if candidate_ids.shape[0] != 15 or candidate_ids.shape[1] < candidate_k:
        raise RuntimeError("canonical candidates require [15,K] with K>=16")
    if tuple(base_logsumexp.shape) != (15,):
        raise RuntimeError("canonical base logsumexp does not have shape [15]")
    if isinstance(anchor_token_id, bool) or not isinstance(anchor_token_id, int):
        raise RuntimeError("canonical anchor token is not an integer")
    result = NumericInput(
        sample_id=sample_id,
        hidden=hidden.to(dtype=torch.bfloat16).contiguous().clone(),
        candidate_ids=(
            candidate_ids[:, :candidate_k]
            .to(dtype=torch.int64)
            .contiguous()
            .clone()
        ),
        candidate_logits=(
            candidate_logits[:, :candidate_k]
            .to(dtype=torch.float32)
            .contiguous()
            .clone()
        ),
        base_logsumexp=(
            base_logsumexp.to(dtype=torch.float32).contiguous().clone()
        ),
        anchor_ids=torch.tensor(anchor_token_id, dtype=torch.int64),
    )
    for name, value in (
        ("hidden", result.hidden),
        ("candidate logits", result.candidate_logits),
        ("base logsumexp", result.base_logsumexp),
    ):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"canonical {name} contains a nonfinite value")
    if not bool(
        (result.candidate_logits[:, :-1] >= result.candidate_logits[:, 1:]).all()
    ):
        raise RuntimeError("canonical candidates are not sorted by logit")
    return result


def collate_numeric_inputs(rows: Sequence[NumericInput]) -> NumericBatch:
    if not rows:
        raise RuntimeError("cannot collate an empty numeric batch")
    return NumericBatch(
        hidden=torch.stack([row.hidden for row in rows]),
        candidate_ids=torch.stack([row.candidate_ids for row in rows]),
        candidate_logits=torch.stack([row.candidate_logits for row in rows]),
        base_logsumexp=torch.stack([row.base_logsumexp for row in rows]),
        anchor_ids=torch.stack([row.anchor_ids for row in rows]),
    )


@dataclass
class Assessment:
    actual64: Tensor
    reference64: Tensor
    source_ulp: Tensor
    accepted: Tensor
    half_width: Tensor
    finite: Tensor
    in_range: Tensor
    cap_ok: Tensor


@dataclass
class FieldStats:
    comparisons: int = 0
    exact_mismatches: int = 0
    max_absolute_difference: float = 0.0
    max_source_ulp_ratio: float = 0.0
    envelope_violations: int = 0
    cap_violations: int = 0
    nonfinite_count: int = 0
    range_violations: int = 0

    def update(self, assessment: Assessment) -> None:
        actual = assessment.actual64.reshape(-1)
        reference = assessment.reference64.reshape(-1)
        ulp = assessment.source_ulp.reshape(-1)
        accepted = assessment.accepted.reshape(-1)
        half_width = assessment.half_width.reshape(-1)
        finite = assessment.finite.reshape(-1)
        in_range = assessment.in_range.reshape(-1)
        if not (
            actual.numel()
            == reference.numel()
            == ulp.numel()
            == accepted.numel()
            == half_width.numel()
            == finite.numel()
            == in_range.numel()
        ):
            raise RuntimeError("numeric assessment shapes differ")
        self.comparisons += actual.numel()
        reference32 = reference.to(dtype=torch.float32).to(dtype=torch.float64)
        self.exact_mismatches += int(actual.ne(reference32).sum())
        valid = finite & torch.isfinite(actual) & torch.isfinite(reference)
        if bool(valid.any()):
            difference = (actual[valid] - reference[valid]).abs()
            self.max_absolute_difference = max(
                self.max_absolute_difference, float(difference.max())
            )
            valid_ulp = ulp[valid]
            ratios = difference / valid_ulp
            self.max_source_ulp_ratio = max(
                self.max_source_ulp_ratio, float(ratios.max())
            )
        self.envelope_violations += int((~accepted & finite & in_range).sum())
        self.cap_violations += int((~assessment.cap_ok.reshape(-1)).sum())
        self.nonfinite_count += int((~finite).sum())
        self.range_violations += int((~in_range & finite).sum())

    def as_dict(self) -> dict[str, int | float]:
        return {
            "comparisons": self.comparisons,
            "exact_mismatches": self.exact_mismatches,
            "max_absolute_difference": self.max_absolute_difference,
            "max_source_ulp_ratio": self.max_source_ulp_ratio,
            "envelope_violations": self.envelope_violations,
            "cap_violations": self.cap_violations,
            "nonfinite_count": self.nonfinite_count,
            "range_violations": self.range_violations,
        }


@dataclass
class DiagnosticState:
    per_field: dict[str, FieldStats] = field(default_factory=dict)
    synthetic_case_count: int = 0
    synthetic_negative_case_count: int = 0
    synthetic_negative_rejections: int = 0

    def record(self, name: str, assessment: Assessment) -> None:
        self.per_field.setdefault(name, FieldStats()).update(assessment)

    def require_negative_rejection(self, assessment: Assessment) -> None:
        accepted = assessment.accepted & assessment.cap_ok
        total = accepted.numel()
        rejected = int((~accepted).sum())
        self.synthetic_negative_case_count += total
        self.synthetic_negative_rejections += rejected


def _broadcast_float64(value: Tensor, reference: Tensor) -> Tensor:
    return torch.broadcast_to(value.to(dtype=torch.float64), reference.shape)


def source_scale_ulp32(operands: Sequence[Tensor], reference64: Tensor) -> Tensor:
    """Elementwise ULP at the pre-registered source scale."""

    reference = reference64.to(dtype=torch.float64)
    scale = torch.ones_like(reference)
    for operand in operands:
        scale = torch.maximum(scale, _broadcast_float64(operand, reference).abs())
    scale = torch.maximum(scale, reference.abs())
    scale32 = scale.to(dtype=torch.float32)
    toward_positive = torch.full_like(scale32, torch.inf)
    ulp = (
        torch.nextafter(scale32, toward_positive).to(dtype=torch.float64)
        - scale32.to(dtype=torch.float64)
    )
    if not bool(torch.isfinite(scale32).all()) or not bool(torch.isfinite(ulp).all()):
        raise RuntimeError("source-scale ULP is nonfinite")
    if not bool(ulp.gt(0.0).all()):
        raise RuntimeError("source-scale ULP is not positive")
    return ulp


def assess_add_sub(actual: Tensor, left: Tensor, right: Tensor, op: str) -> Assessment:
    left64 = left.to(dtype=torch.float64)
    right64 = right.to(dtype=torch.float64)
    if op == "add":
        reference = left64 + right64
    elif op == "subtract":
        reference = left64 - right64
    else:
        raise ValueError("add/sub operation is invalid")
    actual64 = actual.to(dtype=torch.float64)
    ulp = source_scale_ulp32((left, right), reference)
    half_width = ADD_SUB_ULPS * ulp
    finite = (
        torch.isfinite(actual64)
        & torch.isfinite(reference)
        & torch.isfinite(half_width)
    )
    accepted = finite & ((actual64 - reference).abs() <= half_width)
    return Assessment(
        actual64=actual64,
        reference64=reference,
        source_ulp=ulp,
        accepted=accepted,
        half_width=half_width,
        finite=finite,
        in_range=torch.ones_like(accepted),
        cap_ok=half_width.le(ADD_SUB_HALF_WIDTH_CAP),
    )


def assess_exact(actual: Tensor, expected: Tensor) -> Assessment:
    if actual.shape != expected.shape:
        raise RuntimeError("exact comparison shapes differ")
    if actual.dtype == torch.bool:
        actual64 = actual.to(dtype=torch.int64).to(dtype=torch.float64)
        reference = expected.to(dtype=torch.int64).to(dtype=torch.float64)
    else:
        actual64 = actual.to(dtype=torch.float64)
        reference = expected.to(dtype=torch.float64)
    accepted = actual.eq(expected)
    finite = torch.isfinite(actual64) & torch.isfinite(reference)
    return Assessment(
        actual64=actual64,
        reference64=reference,
        source_ulp=torch.full_like(reference, EPS32),
        accepted=accepted,
        half_width=torch.zeros_like(reference),
        finite=finite,
        in_range=torch.ones_like(accepted),
        cap_ok=torch.ones_like(accepted),
    )


def assess_normalized_neighbor(actual: Tensor, expected: Tensor) -> Assessment:
    if actual.dtype != torch.float32 or expected.dtype != torch.float32:
        raise RuntimeError("normalized neighbor inputs must be float32")
    if actual.shape != expected.shape:
        raise RuntimeError("normalized neighbor shapes differ")
    endpoint = expected.eq(0.0) | expected.eq(1.0)
    lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
    upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
    in_range = actual.ge(0.0) & actual.le(1.0)
    accepted = in_range & (
        actual.eq(expected)
        | (~endpoint & (actual.eq(lower) | actual.eq(upper)))
    )
    actual64 = actual.to(dtype=torch.float64)
    reference = expected.to(dtype=torch.float64)
    ulp = source_scale_ulp32((expected,), reference)
    finite = torch.isfinite(actual64) & torch.isfinite(reference)
    return Assessment(
        actual64=actual64,
        reference64=reference,
        source_ulp=ulp,
        accepted=accepted & finite,
        half_width=ulp,
        finite=finite,
        in_range=in_range,
        cap_ok=ulp.le(ADD_SUB_HALF_WIDTH_CAP),
    )


def entropy_reference64(logits: Tensor) -> Tensor:
    logits64 = logits.to(dtype=torch.float64)
    log_q = logits64 - torch.logsumexp(logits64, dim=-1, keepdim=True)
    return -(log_q.exp() * log_q).sum(dim=-1) / math.log(16.0)


def assess_entropy(actual: Tensor, logits: Tensor) -> Assessment:
    reference = entropy_reference64(logits)
    actual64 = actual.to(dtype=torch.float64)
    ulp = source_scale_ulp32((logits.abs().amax(dim=-1),), reference)
    half_width = torch.full_like(reference, ENTROPY_ABS_ENVELOPE)
    finite = (
        torch.isfinite(actual64)
        & torch.isfinite(reference)
        & torch.isfinite(half_width)
    )
    in_range = actual64.ge(0.0) & actual64.le(1.0)
    accepted = finite & in_range & ((actual64 - reference).abs() <= half_width)
    return Assessment(
        actual64=actual64,
        reference64=reference,
        source_ulp=ulp,
        accepted=accepted,
        half_width=half_width,
        finite=finite,
        in_range=in_range,
        cap_ok=half_width.le(ADD_SUB_HALF_WIDTH_CAP),
    )


@dataclass(frozen=True)
class RetainedBounds:
    center64: Tensor
    lower64: Tensor
    upper64: Tensor
    source_ulp: Tensor
    lse_source_ulp: Tensor
    half_width: Tensor
    analytic_cap: Tensor
    cap_ok: Tensor
    subset_ok: Tensor
    finite: Tensor


def _retained_lse_envelope(
    logits64: Tensor, lse64: Tensor, base64: Tensor
) -> tuple[Tensor, Tensor]:
    maximum = logits64.amax(dim=-1).abs()
    scale = torch.maximum(torch.ones_like(lse64), maximum)
    scale = torch.maximum(scale, lse64.abs())
    scale = torch.maximum(scale, base64.abs())
    scale32 = scale.to(dtype=torch.float32)
    scale_ulp = (
        torch.nextafter(scale32, torch.full_like(scale32, torch.inf)).to(
            dtype=torch.float64
        )
        - scale32.to(dtype=torch.float64)
    )
    return LSE_ULPS * scale_ulp + LSE_ABS_FLOOR, scale_ulp


def retained_bounds(logits: Tensor, base_logsumexp: Tensor) -> RetainedBounds:
    logits64 = logits.to(dtype=torch.float64)
    base64 = base_logsumexp.to(dtype=torch.float64)
    lse64 = torch.logsumexp(logits64, dim=-1)
    lse_envelope, lse_source_ulp = _retained_lse_envelope(
        logits64, lse64, base64
    )
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
    analytic_cap = (
        lse_envelope / 2.0
        + RETAINED_OUTER_FLOOR
        + 4.0 * EPS32
    )
    cap_ok = (
        lse_source_ulp.le(MAX_LSE_SOURCE_ULP)
        & analytic_cap.lt(MATERIAL_MUTATION)
        & half_width.le(analytic_cap)
    )
    finite = (
        torch.isfinite(center)
        & torch.isfinite(lower64)
        & torch.isfinite(upper64)
        & torch.isfinite(lse_envelope)
        & torch.isfinite(lse_source_ulp)
        & torch.isfinite(analytic_cap)
        & torch.isfinite(half_width)
    )
    return RetainedBounds(
        center64=center,
        lower64=lower64,
        upper64=upper64,
        source_ulp=source_scale_ulp32(
            (logits.abs().amax(dim=-1), base_logsumexp), center
        ),
        lse_source_ulp=lse_source_ulp,
        half_width=half_width,
        analytic_cap=analytic_cap,
        cap_ok=cap_ok,
        subset_ok=lse64.le(base64 + lse_envelope),
        finite=finite,
    )


def assess_retained(actual: Tensor, logits: Tensor, base_logsumexp: Tensor) -> Assessment:
    bounds = retained_bounds(logits, base_logsumexp)
    actual64 = actual.to(dtype=torch.float64)
    in_range = actual64.ge(-1.0) & actual64.le(bounds.upper64)
    interval = actual64.ge(bounds.lower64) & actual64.le(bounds.upper64)
    accepted = bounds.finite & bounds.subset_ok & in_range & interval
    return Assessment(
        actual64=actual64,
        reference64=bounds.center64,
        source_ulp=bounds.source_ulp,
        accepted=accepted,
        half_width=bounds.half_width,
        finite=bounds.finite & torch.isfinite(actual64),
        in_range=in_range,
        cap_ok=bounds.cap_ok,
    )


def _first_float32_above(boundary64: Tensor) -> Tensor:
    candidate = boundary64.to(dtype=torch.float32)
    too_low = candidate.to(dtype=torch.float64).le(boundary64)
    return torch.where(
        too_low,
        torch.nextafter(candidate, torch.full_like(candidate, torch.inf)),
        candidate,
    )


def _first_float32_at_or_above(boundary64: Tensor) -> Tensor:
    candidate = boundary64.to(dtype=torch.float32)
    too_low = candidate.to(dtype=torch.float64).lt(boundary64)
    return torch.where(
        too_low,
        torch.nextafter(candidate, torch.full_like(candidate, torch.inf)),
        candidate,
    )


def _mutate_add_sub_outside(left: Tensor, right: Tensor, op: str) -> Tensor:
    baseline = left.to(dtype=torch.float64) + (
        right.to(dtype=torch.float64) if op == "add" else -right.to(dtype=torch.float64)
    )
    ulp = source_scale_ulp32((left, right), baseline)
    return _first_float32_above(baseline + ADD_SUB_ULPS * ulp)


def _minimum_allowed_base_lse(logits: Tensor) -> Tensor:
    """Return the globally minimal cap-eligible float32 subset boundary.

    Any cap-eligible candidate has ``E_lse <= MAX_E``, hence no value below
    ``lse64 - MAX_E`` can satisfy the subset invariant.  Between that global
    lower bound and the returned candidate, the scale ULP is required to stay
    in one bucket, so E is constant and the predicate is monotone.  The final
    candidate and its immediate predecessor witness the exact transition.
    """

    logits64 = logits.to(dtype=torch.float64)
    lse64 = torch.logsumexp(logits64, dim=-1)
    maximum_envelope = LSE_ULPS * MAX_LSE_SOURCE_ULP + LSE_ABS_FLOOR
    global_lower = _first_float32_at_or_above(lse64 - maximum_envelope)
    probe = lse64.to(dtype=torch.float32)
    envelope, _ = _retained_lse_envelope(
        logits64, lse64, probe.to(dtype=torch.float64)
    )
    candidate = _first_float32_at_or_above(lse64 - envelope)
    lower_envelope, lower_ulp = _retained_lse_envelope(
        logits64, lse64, global_lower.to(dtype=torch.float64)
    )
    candidate_envelope, candidate_ulp = _retained_lse_envelope(
        logits64, lse64, candidate.to(dtype=torch.float64)
    )
    if not bool(lower_ulp.eq(candidate_ulp).all()):
        raise RuntimeError("retained boundary crosses a source-ULP bucket")
    if not bool(lower_envelope.eq(candidate_envelope).all()):
        raise RuntimeError("retained boundary envelope is not constant")
    recomputed_candidate = _first_float32_at_or_above(
        lse64 - candidate_envelope
    )
    if not torch.equal(recomputed_candidate, candidate):
        raise RuntimeError("retained boundary candidate changed after recomputation")
    candidate = recomputed_candidate
    current = retained_bounds(logits, candidate)
    previous = torch.nextafter(candidate, torch.full_like(candidate, -torch.inf))
    previous_envelope, previous_ulp = _retained_lse_envelope(
        logits64, lse64, previous.to(dtype=torch.float64)
    )
    if not bool(
        lower_ulp.eq(candidate_ulp).all()
        and candidate_ulp.eq(previous_ulp).all()
    ):
        raise RuntimeError("retained final witness crosses a source-ULP bucket")
    if not bool(
        lower_envelope.eq(candidate_envelope).all()
        and candidate_envelope.eq(previous_envelope).all()
    ):
        raise RuntimeError("retained final witness envelope is not constant")
    previous_bounds = retained_bounds(logits, previous)
    if not bool((current.cap_ok & current.subset_ok).all()):
        raise RuntimeError("retained global boundary is not policy-eligible")
    if not bool(previous_bounds.cap_ok.all()):
        raise RuntimeError("retained predecessor is not cap-eligible")
    if bool(previous_bounds.subset_ok.any()):
        raise RuntimeError("retained predecessor does not fail the subset invariant")
    return candidate


def synthetic_logit_patterns() -> list[Tensor]:
    patterns: list[Tensor] = []
    for shift in SYNTHETIC_SHIFTS:
        base_patterns = [
            torch.zeros(16, dtype=torch.float32),
            torch.arange(16, dtype=torch.float32) * (2.0**-20),
            -torch.arange(16, dtype=torch.float32) * (2.0**-20),
            torch.linspace(-16.0, 0.0, 16, dtype=torch.float32),
        ]
        for gap in (2.0**-20, 2.0**-10, 1.0, 16.0, 80.0):
            dominant = torch.full((16,), -gap, dtype=torch.float32)
            dominant[0] = 0.0
            base_patterns.append(dominant)
        for seed in SYNTHETIC_SEEDS:
            for scale in (2.0**-10, 1.0, 8.0, 32.0):
                generator = torch.Generator().manual_seed(seed)
                base_patterns.append(
                    torch.randn(16, generator=generator, dtype=torch.float32)
                    * scale
                )
        if len(base_patterns) != 17:
            raise RuntimeError("synthetic pattern census changed")
        for pattern in base_patterns:
            case = (pattern + shift)[None, None, :].expand(1, 15, 16).clone()
            patterns.append(case)
    if len(patterns) != 85:
        raise RuntimeError("synthetic logit grid census changed")
    return patterns


def _positive_base_variants(logits: Tensor) -> list[Tensor]:
    exact = torch.logsumexp(logits, dim=-1).to(dtype=torch.float32)
    variants = [exact]
    neighbor = exact
    for step in range(1, 9):
        neighbor = torch.nextafter(neighbor, torch.full_like(neighbor, torch.inf))
        if step in {1, 2, 8}:
            variants.append(neighbor)
    for offset in (2.0**-20, 2.0**-10, 0.1, 2.0, 16.0):
        variants.append((exact + offset).to(dtype=torch.float32))
    variants.append(torch.nextafter(exact, torch.full_like(exact, -torch.inf)))
    if len(variants) != 10:
        raise RuntimeError("synthetic base-LSE grid census changed")
    return variants


def _record_negative_mutations(state: DiagnosticState) -> None:
    left = torch.tensor([1.0], dtype=torch.float32)
    right = torch.tensor([2.0**-20], dtype=torch.float32)
    for op in ("add", "subtract"):
        outside = _mutate_add_sub_outside(left, right, op)
        material = (
            left + right if op == "add" else left - right
        ) + torch.tensor([MATERIAL_MUTATION], dtype=torch.float32)
        state.require_negative_rejection(assess_add_sub(outside, left, right, op))
        state.require_negative_rejection(assess_add_sub(material, left, right, op))

    expected = torch.tensor([0.5], dtype=torch.float32)
    outside = torch.nextafter(
        torch.nextafter(expected, torch.full_like(expected, torch.inf)),
        torch.full_like(expected, torch.inf),
    )
    material = expected + torch.tensor([MATERIAL_MUTATION], dtype=torch.float32)
    state.require_negative_rejection(
        assess_normalized_neighbor(outside, expected)
    )
    state.require_negative_rejection(
        assess_normalized_neighbor(material, expected)
    )

    logits = torch.zeros(1, 15, 16, dtype=torch.float32)
    entropy = entropy_reference64(logits)
    entropy_outside = _first_float32_above(entropy + ENTROPY_ABS_ENVELOPE)
    entropy_material = entropy.to(dtype=torch.float32) + MATERIAL_MUTATION
    state.require_negative_rejection(
        assess_entropy(entropy_outside, logits)
    )
    state.require_negative_rejection(
        assess_entropy(entropy_material, logits)
    )

    base = torch.logsumexp(logits, dim=-1).to(dtype=torch.float32)
    bounds = retained_bounds(logits, base)
    retained_outside = _first_float32_above(bounds.upper64)
    retained_material = bounds.center64.to(dtype=torch.float32) + MATERIAL_MUTATION
    state.require_negative_rejection(
        assess_retained(retained_outside, logits, base)
    )
    state.require_negative_rejection(
        assess_retained(retained_material, logits, base)
    )

    exact = torch.tensor([1.0], dtype=torch.float32)
    state.require_negative_rejection(
        assess_exact(torch.nextafter(exact, torch.full_like(exact, torch.inf)), exact)
    )
    state.require_negative_rejection(
        assess_exact(exact + torch.tensor([MATERIAL_MUTATION]), exact)
    )


def run_synthetic_scan(device: torch.device, state: DiagnosticState) -> None:
    patterns = synthetic_logit_patterns()
    positive_logits: list[Tensor] = []
    positive_bases: list[Tensor] = []
    for pattern in patterns:
        for base in _positive_base_variants(pattern):
            positive_logits.append(pattern)
            positive_bases.append(base)
    logits_cpu = torch.cat(positive_logits, dim=0)
    bases_cpu = torch.cat(positive_bases, dim=0)
    logits_device = logits_cpu.to(device)
    bases_device = bases_cpu.to(device)
    conditional = torch.log_softmax(logits_device, dim=-1)
    entropy_actual = (
        -(conditional.exp() * conditional).sum(dim=-1) / math.log(16.0)
    ).to(device="cpu")
    retained_actual = torch.tanh(
        (torch.logsumexp(logits_device, dim=-1) - bases_device) / 2.0
    ).to(device="cpu")
    state.record("entropy", assess_entropy(entropy_actual, logits_cpu))
    state.record(
        "retained_mass",
        assess_retained(retained_actual, logits_cpu, bases_cpu),
    )
    state.synthetic_case_count += logits_cpu.shape[0]

    boundary_logits = torch.cat(patterns, dim=0)
    minimum_base = _minimum_allowed_base_lse(boundary_logits)
    minimum_actual = torch.tanh(
        (
            torch.logsumexp(boundary_logits.to(device), dim=-1)
            - minimum_base.to(device)
        )
        / 2.0
    ).to(device="cpu")
    minimum_assessment = assess_retained(
        minimum_actual, boundary_logits, minimum_base
    )
    if not bool(minimum_assessment.accepted.all()):
        raise RuntimeError("retained minimum allowed base-LSE boundary was rejected")
    below = torch.nextafter(minimum_base, torch.full_like(minimum_base, -torch.inf))
    below_actual = torch.tanh(
        (
            torch.logsumexp(boundary_logits.to(device), dim=-1)
            - below.to(device)
        )
        / 2.0
    ).to(device="cpu")
    state.require_negative_rejection(
        assess_retained(below_actual, boundary_logits, below)
    )

    magnitudes: list[float] = []
    ranks: list[int] = []
    for magnitude in SYNTHETIC_STATE_MAGNITUDES:
        for rank in SYNTHETIC_PATH_RANKS:
            magnitudes.append(magnitude)
            ranks.append(rank)
    count = len(magnitudes)
    state.synthetic_case_count += count
    magnitude = torch.tensor(magnitudes, dtype=torch.float32)[:, None]
    path = torch.tensor(ranks, dtype=torch.int64)[:, None].expand(count, 15)
    left = magnitude.expand(count, 15)
    sign = torch.where(
        torch.arange(15)[None].remainder(2).eq(0),
        torch.ones(count, 15),
        -torch.ones(count, 15),
    )
    right = left * sign
    for name, op in (
        ("state_difference", "subtract"),
        ("base_log_probs", "subtract"),
        ("direct_scores", "add"),
        ("total_margin", "subtract"),
        ("residual_margin", "subtract"),
        ("dflash_margin", "subtract"),
    ):
        left_device = left.to(device)
        right_device = right.to(device)
        actual = (
            left_device + right_device
            if op == "add"
            else left_device - right_device
        ).to(device="cpu")
        state.record(name, assess_add_sub(actual, left, right, op))
    rank_actual = (path.to(device).float() / 15.0).to(device="cpu")
    rank_expected = path.float() / 15.0
    state.record("rank", assess_normalized_neighbor(rank_actual, rank_expected))
    position_expected = (
        torch.arange(15, dtype=torch.float32) / 14.0
    )[None].expand(count, -1)
    position_actual = (
        torch.arange(15, device=device, dtype=torch.float32) / 14.0
    )[None].expand(count, -1).to(device="cpu")
    state.record(
        "position", assess_normalized_neighbor(position_actual, position_expected)
    )
    _record_negative_mutations(state)


def load_split_selection(
    path: Path, expected_sha256: str
) -> tuple[dict[str, str], dict[str, int], dict[str, int]]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("split manifest SHA256 differs")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != EXPECTED_SPLIT_PROTOCOL:
        raise RuntimeError("split manifest protocol differs")
    assignments: dict[str, str] = {}
    for row in manifest.get("prompts", []):
        split = row.get("split")
        if split not in SCANNED_SPLITS:
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or "\0" in sample_id:
            raise RuntimeError("split manifest has an invalid selected identity")
        if sample_id in assignments:
            raise RuntimeError("split manifest repeats a selected identity")
        assignments[sample_id] = split
    block_counts = manifest.get("block_counts_by_split")
    prompt_counts = manifest.get("prompt_counts_by_split")
    if not isinstance(block_counts, dict) or not isinstance(prompt_counts, dict):
        raise RuntimeError("split manifest lacks frozen count censuses")
    expected_blocks = {split: int(block_counts[split]) for split in SCANNED_SPLITS}
    expected_prompts = {split: int(prompt_counts[split]) for split in SCANNED_SPLITS}
    observed_prompts = {
        split: sum(assigned == split for assigned in assignments.values())
        for split in SCANNED_SPLITS
    }
    if observed_prompts != expected_prompts:
        raise RuntimeError("selected prompt census differs from split manifest")
    return assignments, expected_blocks, expected_prompts


def load_label_blind_inputs(
    data: Path,
    assignments: Mapping[str, str],
    expected_blocks: Mapping[str, int],
    expected_prompts: Mapping[str, int],
) -> tuple[dict[str, list[NumericInput]], dict[str, Any]]:
    metadata_path = data / "metadata.json"
    if sha256_file(metadata_path) != EXPECTED_DATA_METADATA_SHA256:
        raise RuntimeError("canonical metadata differs from the frozen collection")
    collection = CanonicalBlockDataset(data, split="train")
    selected = {split: [] for split in SCANNED_SPLITS}
    seen_prompts = {split: set() for split in SCANNED_SPLITS}
    for raw_record in collection.records:
        sample_id = extract_sample_id(raw_record)
        split = assignments.get(sample_id)
        if split is None:
            continue
        numeric = extract_numeric_input(raw_record)
        selected[split].append(numeric)
        seen_prompts[split].add(numeric.sample_id)
    metadata = dict(collection.metadata)
    del collection
    gc.collect()
    for split in SCANNED_SPLITS:
        if len(selected[split]) != int(expected_blocks[split]):
            raise RuntimeError(f"{split} numeric-input block census differs")
        if len(seen_prompts[split]) != int(expected_prompts[split]):
            raise RuntimeError(f"{split} numeric-input prompt census differs")
    return selected, metadata


def _verify_target_and_load_embedding(
    metadata: Mapping[str, Any], target: Path
) -> Tensor:
    expected_rows = metadata.get("provenance", {}).get("target_files")
    if not isinstance(expected_rows, list):
        raise RuntimeError("canonical metadata lacks target fingerprints")
    expected = {str(row["path"]): row for row in expected_rows}
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    key = "model.embed_tokens.weight"
    shard_name = str(index["weight_map"][key])
    for name in ("config.json", index_path.name, shard_name):
        path = target / name
        reference = expected.get(name)
        if not isinstance(reference, Mapping):
            raise RuntimeError("target fingerprint is absent from canonical metadata")
        if path.stat().st_size != int(reference["bytes"]):
            raise RuntimeError("target file byte count differs")
        if sha256_file(path) != str(reference["sha256"]):
            raise RuntimeError("target file SHA256 differs")
    with safe_open(target / shard_name, framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor(key)
    if embedding.ndim != 2 or not bool(torch.isfinite(embedding.float()).all()):
        raise RuntimeError("target embedding is invalid")
    return embedding


def load_frozen_producer(
    direct_run: Path,
    target: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
) -> tuple[nn.Module, Tensor]:
    metrics_path = direct_run / "metrics.json"
    checkpoint_path = direct_run / "best.pt"
    if sha256_file(metrics_path) != EXPECTED_DIRECT_METRICS_SHA256:
        raise RuntimeError("Direct metrics differ from the frozen run")
    if sha256_file(checkpoint_path) != EXPECTED_DIRECT_CHECKPOINT_SHA256:
        raise RuntimeError("Direct checkpoint differs from the frozen run")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = metrics.get("config")
    if not isinstance(config, dict) or checkpoint.get("args") != config:
        raise RuntimeError("Direct metrics/checkpoint config binding differs")
    if int(checkpoint.get("epoch", -1)) != int(metrics.get("selected_epoch", -2)):
        raise RuntimeError("Direct selected epoch differs")
    if int(checkpoint.get("parameter_count", -1)) != int(
        metrics.get("parameter_count", -2)
    ):
        raise RuntimeError("Direct parameter census differs")
    for name, expected in EXPECTED_DIRECT_CONFIG.items():
        if config.get(name) != expected:
            raise RuntimeError(f"Direct frozen config differs for {name}")
    embedding = _verify_target_and_load_embedding(metadata, target)
    producer = GlobalDirectCandidateSelector(
        hidden_size=int(embedding.shape[1]),
        max_positions=int(metadata.get("draft_positions", 15)),
        max_candidates=int(config["candidate_k"]),
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        scope=str(config["scope"]),
        mixer=str(config["mixer"]),
        node_encoder=str(config["node_encoder"]),
        dropout=float(config["dropout"]),
        initialization_seed=int(config["seed"]),
    )
    producer.load_state_dict(checkpoint["model"], strict=True)
    if sum(value.numel() for value in producer.parameters()) != int(
        checkpoint["parameter_count"]
    ):
        raise RuntimeError("Direct reconstructed parameter census differs")
    producer = freeze_direct_producer(producer.to(device))
    embedding = embedding.to(device=device, dtype=torch.bfloat16).detach()
    embedding.requires_grad_(False)
    return producer, embedding


def _cpu(value: Tensor, dtype: torch.dtype) -> Tensor:
    return value.detach().to(device="cpu", dtype=dtype).contiguous().clone()


def scan_real_batch(
    batch: NumericBatch,
    producer: nn.Module,
    target_embedding: Tensor,
    device: torch.device,
    state: DiagnosticState,
) -> None:
    hidden = batch.hidden.to(device)
    candidate_ids = batch.candidate_ids.to(device)
    candidate_logits = batch.candidate_logits.to(device)
    base_lse = batch.base_logsumexp.to(device)
    anchor_ids = batch.anchor_ids.to(device)
    output, node_states = frozen_direct_forward_with_states(
        producer,
        hidden,
        target_embedding[candidate_ids],
        candidate_logits,
        base_lse,
        target_embedding[anchor_ids],
    )
    generated = direct_safety_position_features(
        node_states, output, candidate_logits, base_lse
    )
    features = _cpu(generated.position_features, torch.float32)
    path = _cpu(generated.direct_path, torch.int64)
    changed = _cpu(generated.change_mask, torch.bool)
    states = _cpu(node_states, torch.float32)
    logits = _cpu(candidate_logits, torch.float32)
    full_lse = _cpu(base_lse, torch.float32)
    scores = _cpu(output.scores, torch.float32)
    residual = _cpu(output.residual_scores, torch.float32)
    base = _cpu(output.base_log_probs, torch.float32)

    state.record("hidden_copy", assess_exact(_cpu(hidden, torch.bfloat16), batch.hidden))
    state.record(
        "candidate_ids_copy",
        assess_exact(_cpu(candidate_ids, torch.int64), batch.candidate_ids),
    )
    state.record(
        "candidate_logits_copy",
        assess_exact(logits, batch.candidate_logits),
    )
    state.record("base_lse_copy", assess_exact(full_lse, batch.base_logsumexp))
    state.record(
        "anchor_ids_copy", assess_exact(_cpu(anchor_ids, torch.int64), batch.anchor_ids)
    )
    expected_path = scores.argmax(dim=-1)
    state.record("direct_path", assess_exact(path, expected_path))
    state.record("change_mask", assess_exact(changed, path.ne(0)))
    gather_state = path[..., None, None].expand(-1, -1, 1, states.shape[-1])
    selected_states = states.gather(2, gather_state).squeeze(2)
    base_states = states[:, :, 0]
    state.record("selected_state_copy", assess_exact(features[..., :64], selected_states))
    state.record("base_state_copy", assess_exact(features[..., 64:128], base_states))
    state.record(
        "state_difference",
        assess_add_sub(
            features[..., 128:192], selected_states, base_states, "subtract"
        ),
    )
    state.record(
        "base_log_probs",
        assess_add_sub(base, logits, full_lse[..., None], "subtract"),
    )
    state.record(
        "direct_scores", assess_add_sub(scores, base, residual, "add")
    )
    gather = path[..., None]
    chosen_scores = scores.gather(-1, gather).squeeze(-1)
    chosen_residual = residual.gather(-1, gather).squeeze(-1)
    chosen_base = base.gather(-1, gather).squeeze(-1)
    for name, actual, left, right in (
        ("total_margin", features[..., 192], chosen_scores, scores[..., 0]),
        (
            "residual_margin",
            features[..., 193],
            chosen_residual,
            residual[..., 0],
        ),
        ("dflash_margin", features[..., 194], chosen_base, base[..., 0]),
    ):
        state.record(name, assess_add_sub(actual, left, right, "subtract"))
    expected_rank = path.float() / 15.0
    expected_position = (
        torch.arange(15, dtype=torch.float32) / 14.0
    )[None].expand(path.shape[0], -1)
    state.record(
        "rank", assess_normalized_neighbor(features[..., 195], expected_rank)
    )
    state.record(
        "position",
        assess_normalized_neighbor(features[..., 196], expected_position),
    )
    state.record(
        "change_scalar", assess_exact(features[..., 197], changed.float())
    )
    state.record("entropy", assess_entropy(features[..., 198], logits))
    state.record(
        "retained_mass", assess_retained(features[..., 199], logits, full_lse)
    )


def scan_real_inputs(
    selected: Mapping[str, Sequence[NumericInput]],
    producer: nn.Module,
    target_embedding: Tensor,
    device: torch.device,
    batch_size: int,
    state: DiagnosticState,
) -> dict[str, int]:
    if batch_size < 1:
        raise RuntimeError("batch size must be positive")
    counts: dict[str, int] = {}
    with torch.inference_mode():
        for split in SCANNED_SPLITS:
            rows = selected[split]
            for start in range(0, len(rows), batch_size):
                batch = collate_numeric_inputs(rows[start : start + batch_size])
                scan_real_batch(batch, producer, target_embedding, device, state)
            counts[split] = len(rows)
    return counts


def expected_comparison_census(
    input_count: int, hidden_dimension: int
) -> dict[str, int]:
    if input_count < 1 or hidden_dimension < 1:
        raise RuntimeError("numeric comparison census dimensions are invalid")
    positions = input_count * 15
    candidates = positions * 16
    states = positions * 64
    synthetic_scalar = len(SYNTHETIC_STATE_MAGNITUDES) * len(
        SYNTHETIC_PATH_RANKS
    ) * 15
    synthetic_distribution = len(SYNTHETIC_SHIFTS) * 17 * 10 * 15
    result = {
        "anchor_ids_copy": input_count,
        "base_log_probs": candidates + synthetic_scalar,
        "base_lse_copy": positions,
        "base_state_copy": states,
        "candidate_ids_copy": candidates,
        "candidate_logits_copy": candidates,
        "change_mask": positions,
        "change_scalar": positions,
        "dflash_margin": positions + synthetic_scalar,
        "direct_path": positions,
        "direct_scores": candidates + synthetic_scalar,
        "entropy": positions + synthetic_distribution,
        "hidden_copy": positions * hidden_dimension,
        "position": positions + synthetic_scalar,
        "rank": positions + synthetic_scalar,
        "residual_margin": positions + synthetic_scalar,
        "retained_mass": positions + synthetic_distribution,
        "selected_state_copy": states,
        "state_difference": states + synthetic_scalar,
        "total_margin": positions + synthetic_scalar,
    }
    if frozenset(result) != EXPECTED_FIELD_NAMES:
        raise RuntimeError("numeric comparison field contract changed")
    return result


def assert_complete_comparison_census(
    state: DiagnosticState,
    counts: Mapping[str, int],
    hidden_dimension: int,
) -> None:
    total = sum(int(counts[split]) for split in SCANNED_SPLITS)
    expected = expected_comparison_census(total, hidden_dimension)
    observed = {
        name: stats.comparisons for name, stats in state.per_field.items()
    }
    if observed != expected:
        raise RuntimeError("numeric comparison census is incomplete or duplicated")


def numeric_policy_report() -> dict[str, Any]:
    return {
        "id": NUMERIC_POLICY_ID,
        "eps32": EPS32,
        "add_sub_half_width_cap": ADD_SUB_HALF_WIDTH_CAP,
        "material_mutation": MATERIAL_MUTATION,
        "add_sub_ulps": ADD_SUB_ULPS,
        "entropy_abs_envelope": ENTROPY_ABS_ENVELOPE,
        "lse_ulps": LSE_ULPS,
        "lse_abs_floor": LSE_ABS_FLOOR,
        "retained_outer_floor": RETAINED_OUTER_FLOOR,
        "retained_outer_ulps": RETAINED_OUTER_ULPS,
        "max_lse_source_ulp": MAX_LSE_SOURCE_ULP,
        "retained_cap_formula": "E_lse/2+2^-20+4*2^-23",
        "synthetic_seeds": list(SYNTHETIC_SEEDS),
        "synthetic_shifts": list(SYNTHETIC_SHIFTS),
        "state_magnitudes": list(SYNTHETIC_STATE_MAGNITUDES),
        "path_ranks": list(SYNTHETIC_PATH_RANKS),
    }


def build_report(
    state: DiagnosticState,
    counts: Mapping[str, int],
    device: torch.device,
) -> dict[str, Any]:
    fields = {name: stats.as_dict() for name, stats in sorted(state.per_field.items())}
    violations = sum(
        int(values[key])
        for values in fields.values()
        for key in (
            "envelope_violations",
            "cap_violations",
            "nonfinite_count",
            "range_violations",
        )
    )
    negative_failures = (
        state.synthetic_negative_case_count - state.synthetic_negative_rejections
    )
    status = "PASS" if violations == 0 and negative_failures == 0 else "FAIL"
    return {
        "protocol": PROTOCOL,
        "numeric_policy": numeric_policy_report(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "synthetic_case_count": state.synthetic_case_count,
        "synthetic_negative_case_count": state.synthetic_negative_case_count,
        "synthetic_negative_rejections": state.synthetic_negative_rejections,
        "fit_input_count": int(counts["fit"]),
        "checkpoint_input_count": int(counts["checkpoint"]),
        "per_field": fields,
        "forbidden_semantic_operations_executed": 0,
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation is required; diagnostic cannot skip")
    device = torch.device("cuda")
    assignments, expected_blocks, expected_prompts = load_split_selection(
        args.split_manifest, args.expected_split_manifest_sha256
    )
    selected, metadata = load_label_blind_inputs(
        args.data, assignments, expected_blocks, expected_prompts
    )
    producer, target_embedding = load_frozen_producer(
        args.direct_run, args.target, metadata, device
    )
    state = DiagnosticState()
    run_synthetic_scan(device, state)
    counts = scan_real_inputs(
        selected,
        producer,
        target_embedding,
        device,
        args.batch_size,
        state,
    )
    assert_complete_comparison_census(
        state, counts, int(target_embedding.shape[1])
    )
    torch.cuda.synchronize(device)
    report = build_report(state, counts, device)
    print(canonical_json(report), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
