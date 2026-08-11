from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import scripts.diagnose_direct_safety_numeric_portability as diagnostic


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/diagnose_direct_safety_numeric_portability.py"


class TrackingRecord(dict[str, object]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed: list[str] = []

    def __getitem__(self, key: str):
        self.accessed.append(key)
        if key == "gold_ids":
            raise AssertionError("semantic target was accessed")
        return super().__getitem__(key)


def _raw_record(*, stored_k: int = 64) -> TrackingRecord:
    return TrackingRecord(
        {
            "sample_id": "synthetic:0",
            "parallel_hidden": torch.zeros(15, 8, dtype=torch.bfloat16),
            "base_topk_ids": torch.arange(
                15 * stored_k, dtype=torch.int64
            ).reshape(15, stored_k),
            "base_topk_logits": -torch.arange(
                stored_k, dtype=torch.float32
            )[None].expand(15, -1).clone(),
            "base_logsumexp": torch.ones(15, dtype=torch.float32),
            "anchor_token_id": 7,
            "gold_ids": object(),
            "domain": "forbidden-decoy",
            "split": "forbidden-decoy",
        }
    )


def test_frozen_numeric_policy_constants_are_exact() -> None:
    assert diagnostic.EPS32 == 2.0**-23
    assert diagnostic.ADD_SUB_HALF_WIDTH_CAP == 2.0**-14
    assert diagnostic.MATERIAL_MUTATION == 1.0e-4
    assert diagnostic.ADD_SUB_ULPS == 2
    assert diagnostic.ENTROPY_ABS_ENVELOPE == 2.0**-17
    assert diagnostic.LSE_ULPS == 8
    assert diagnostic.LSE_ABS_FLOOR == 2.0**-20
    assert diagnostic.RETAINED_OUTER_FLOOR == 2.0**-20
    assert diagnostic.RETAINED_OUTER_ULPS == 2
    assert diagnostic.MAX_LSE_SOURCE_ULP == 2.0**-16
    assert diagnostic.SYNTHETIC_SEEDS == (79_079, 79_080)
    assert diagnostic.SYNTHETIC_SHIFTS == (-64.0, -16.0, 0.0, 16.0, 64.0)
    assert diagnostic.SYNTHETIC_STATE_MAGNITUDES == (0.0, 2.0**-20, 1.0, 32.0)
    assert diagnostic.SYNTHETIC_PATH_RANKS == (0, 1, 15)


def test_extractor_reads_only_the_reviewed_allowlist_and_drops_mapping() -> None:
    raw = _raw_record()
    expected_ids = raw["base_topk_ids"][:, :16].clone()
    expected_logits = raw["base_topk_logits"][:, :16].clone()
    raw.accessed.clear()
    selected = diagnostic.extract_numeric_input(raw)
    assert set(raw.accessed) == diagnostic.RAW_INPUT_FIELD_ALLOWLIST
    assert selected.sample_id == "synthetic:0"
    assert selected.hidden.dtype == torch.bfloat16
    assert selected.candidate_ids.dtype == torch.int64
    assert selected.candidate_logits.dtype == torch.float32
    assert selected.base_logsumexp.dtype == torch.float32
    assert selected.anchor_ids.dtype == torch.int64
    assert selected.candidate_ids.shape == (15, 16)
    assert selected.candidate_logits.shape == (15, 16)
    assert torch.equal(selected.candidate_ids, expected_ids)
    assert torch.equal(selected.candidate_logits, expected_logits)
    raw["base_topk_ids"][0, 0] = -1
    raw["base_topk_logits"][0, 0] = 1.0
    assert int(selected.candidate_ids[0, 0]) != -1
    assert float(selected.candidate_logits[0, 0]) != 1.0
    batch = diagnostic.collate_numeric_inputs([selected])
    assert not hasattr(batch, "sample_id")
    assert not hasattr(batch, "sample_ids")
    assert batch.candidate_ids.shape == (1, 15, 16)


def test_extractor_ignores_unselected_tail_order_and_finiteness() -> None:
    raw = _raw_record()
    logits = raw["base_topk_logits"]
    logits[:, 16:] = torch.flip(logits[:, 16:], dims=(1,))
    logits[0, 63] = torch.nan
    selected = diagnostic.extract_numeric_input(raw)
    assert selected.candidate_logits.shape == (15, 16)
    assert bool(torch.isfinite(selected.candidate_logits).all())


def test_extractor_rejects_insufficient_or_mismatched_stored_width() -> None:
    with pytest.raises(RuntimeError, match="K>=16"):
        diagnostic.extract_numeric_input(_raw_record(stored_k=15))
    for id_k, logit_k in ((64, 63), (63, 64)):
        raw = _raw_record(stored_k=id_k)
        raw["base_topk_logits"] = -torch.arange(
            logit_k, dtype=torch.float32
        )[None].expand(15, -1).clone()
        with pytest.raises(RuntimeError, match="shapes differ"):
            diagnostic.extract_numeric_input(raw)


@pytest.mark.parametrize(
    ("id_shape", "logit_shape", "message"),
    [
        ((15, 16, 1), (15, 16, 1), "shape \\[15,K\\]"),
        ((14, 64), (14, 64), "K>=16"),
    ],
)
def test_extractor_rejects_nonmatrix_or_wrong_position_count(
    id_shape: tuple[int, ...],
    logit_shape: tuple[int, ...],
    message: str,
) -> None:
    raw = _raw_record()
    raw["base_topk_ids"] = torch.zeros(id_shape, dtype=torch.int64)
    raw["base_topk_logits"] = torch.zeros(logit_shape, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=message):
        diagnostic.extract_numeric_input(raw)


def test_extractor_checks_sorting_only_inside_selected_prefix() -> None:
    raw = _raw_record()
    raw["base_topk_logits"][:, 7], raw["base_topk_logits"][:, 8] = (
        raw["base_topk_logits"][:, 8].clone(),
        raw["base_topk_logits"][:, 7].clone(),
    )
    with pytest.raises(RuntimeError, match="not sorted"):
        diagnostic.extract_numeric_input(raw)


def test_static_noninterference_boundary_and_no_output_argument() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "physically deserializes ``gold_ids``" in source
    prohibited_imports = {
        "collate_canonical_blocks",
        "binary_outcomes_from_tokens",
        "realized_prefix_lengths",
        "audit_outcome_record",
        "validate_outcome_record",
    }
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not (imported & prohibited_imports)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    forbidden_call_fragments = (
        "outcome",
        "accepted_length",
        "gain",
        "capacity",
        "audit",
        "evaluate",
    )
    assert not {
        name
        for name in calls
        if any(fragment in name.lower() for fragment in forbidden_call_fragments)
    }
    parse = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    parse_source = ast.get_source_segment(source, parse)
    assert parse_source is not None
    assert "--output" not in parse_source
    prohibited_writes = {"save", "write", "write_text", "write_bytes", "mkdir"}
    observed_attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (observed_attribute_calls & prohibited_writes)


def test_extractor_ast_has_only_literal_allowlist_subscripts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    extractor = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "extract_numeric_input"
    )
    keys = {
        node.slice.value
        for node in ast.walk(extractor)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "record"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert keys == diagnostic.RAW_INPUT_FIELD_ALLOWLIST
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "extract_sample_id"
    )
    selector_keys = {
        node.slice.value
        for node in ast.walk(selector)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "record"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert selector_keys == {"sample_id"}
    for function in (extractor, selector):
        assert not {
            node.func.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        }


def test_source_scale_ulp_uses_float32_scale_neighbor() -> None:
    left = torch.tensor([32.0], dtype=torch.float32)
    right = torch.tensor([1.0], dtype=torch.float32)
    reference = left.double() - right.double()
    observed = diagnostic.source_scale_ulp32((left, right), reference)
    expected = torch.nextafter(
        torch.tensor([32.0], dtype=torch.float32),
        torch.tensor([torch.inf], dtype=torch.float32),
    ).double() - 32.0
    assert torch.equal(observed, expected)


@pytest.mark.parametrize("op", ["add", "subtract"])
def test_add_sub_envelope_accepts_boundary_and_rejects_next_float(op: str) -> None:
    left = torch.tensor([1.0], dtype=torch.float32)
    right = torch.tensor([2.0**-20], dtype=torch.float32)
    reference = left.double() + (right.double() if op == "add" else -right.double())
    ulp = diagnostic.source_scale_ulp32((left, right), reference)
    boundary = (reference + diagnostic.ADD_SUB_ULPS * ulp).float()
    assert bool(diagnostic.assess_add_sub(boundary, left, right, op).accepted.all())
    outside = torch.nextafter(boundary, torch.full_like(boundary, torch.inf))
    assert not bool(diagnostic.assess_add_sub(outside, left, right, op).accepted.any())
    material = reference.float() + diagnostic.MATERIAL_MUTATION
    assert not bool(diagnostic.assess_add_sub(material, left, right, op).accepted.any())


def test_entropy_fixed_envelope_and_range_are_fail_closed() -> None:
    logits = torch.linspace(-4.0, 0.0, 16, dtype=torch.float32)[None, None].expand(
        1, 15, 16
    )
    reference = diagnostic.entropy_reference64(logits)
    accepted = reference.float()
    assert bool(diagnostic.assess_entropy(accepted, logits).accepted.all())
    outside = diagnostic._first_float32_above(
        reference + diagnostic.ENTROPY_ABS_ENVELOPE
    )
    assert not bool(diagnostic.assess_entropy(outside, logits).accepted.any())
    assert not bool(
        diagnostic.assess_entropy(torch.full_like(accepted, 1.01), logits).accepted.any()
    )


def test_retained_interval_and_exact_float32_subset_boundary() -> None:
    logits = torch.linspace(-16.0, 0.0, 16, dtype=torch.float32)[None, None].expand(
        1, 15, 16
    )
    minimum = diagnostic._minimum_allowed_base_lse(logits)
    logits64 = logits.double()
    lse64 = torch.logsumexp(logits64, dim=-1)
    maximum_envelope = (
        diagnostic.LSE_ULPS * diagnostic.MAX_LSE_SOURCE_ULP
        + diagnostic.LSE_ABS_FLOOR
    )
    global_lower = diagnostic._first_float32_at_or_above(
        lse64 - maximum_envelope
    )
    probe = lse64.float()
    probe_envelope, _ = diagnostic._retained_lse_envelope(
        logits64, lse64, probe.double()
    )
    first_candidate = diagnostic._first_float32_at_or_above(
        lse64 - probe_envelope
    )
    first_envelope, first_ulp = diagnostic._retained_lse_envelope(
        logits64, lse64, first_candidate.double()
    )
    recomputed = diagnostic._first_float32_at_or_above(
        lse64 - first_envelope
    )
    assert torch.equal(recomputed, first_candidate)
    assert torch.equal(recomputed, minimum)
    center = torch.tanh(
        (torch.logsumexp(logits, dim=-1) - minimum) / 2.0
    )
    assert bool(diagnostic.assess_retained(center, logits, minimum).accepted.all())
    below = torch.nextafter(minimum, torch.full_like(minimum, -torch.inf))
    lower_envelope, lower_ulp = diagnostic._retained_lse_envelope(
        logits64, lse64, global_lower.double()
    )
    below_envelope, below_ulp = diagnostic._retained_lse_envelope(
        logits64, lse64, below.double()
    )
    assert torch.equal(lower_ulp, first_ulp)
    assert torch.equal(first_ulp, below_ulp)
    assert torch.equal(lower_envelope, first_envelope)
    assert torch.equal(first_envelope, below_envelope)
    below_actual = torch.tanh((torch.logsumexp(logits, dim=-1) - below) / 2.0)
    below_bounds = diagnostic.retained_bounds(logits, below)
    assert bool(below_bounds.cap_ok.all())
    assert not bool(below_bounds.subset_ok.any())
    assert not bool(
        diagnostic.assess_retained(below_actual, logits, below).accepted.any()
    )
    bounds = diagnostic.retained_bounds(logits, minimum)
    assert bool(bounds.lse_source_ulp.le(diagnostic.MAX_LSE_SOURCE_ULP).all())
    assert bool(bounds.half_width.le(bounds.analytic_cap).all())
    assert bool(bounds.analytic_cap.lt(diagnostic.MATERIAL_MUTATION).all())
    outside = diagnostic._first_float32_above(bounds.upper64)
    assert not bool(
        diagnostic.assess_retained(outside, logits, minimum).accepted.any()
    )


def test_synthetic_grid_has_the_pre_registered_census() -> None:
    patterns = diagnostic.synthetic_logit_patterns()
    assert len(patterns) == 85
    assert all(pattern.shape == (1, 15, 16) for pattern in patterns)
    assert all(pattern.dtype == torch.float32 for pattern in patterns)
    assert all(len(diagnostic._positive_base_variants(pattern)) == 10 for pattern in patterns)


def test_full_synthetic_scan_passes_on_cpu_and_rejects_every_negative() -> None:
    state = diagnostic.DiagnosticState()
    diagnostic.run_synthetic_scan(torch.device("cpu"), state)
    assert state.synthetic_case_count == 862
    assert state.synthetic_negative_case_count > 0
    assert state.synthetic_negative_rejections == state.synthetic_negative_case_count
    for stats in state.per_field.values():
        assert stats.envelope_violations == 0
        assert stats.cap_violations == 0
        assert stats.nonfinite_count == 0
        assert stats.range_violations == 0


def test_report_schema_is_aggregate_only(monkeypatch) -> None:
    state = diagnostic.DiagnosticState()
    state.record(
        "copy",
        diagnostic.assess_exact(
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
        ),
    )
    state.synthetic_case_count = 1
    state.synthetic_negative_case_count = 2
    state.synthetic_negative_rejections = 2
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "synthetic")
    report = diagnostic.build_report(
        state, {"fit": 12_686, "checkpoint": 1_600}, torch.device("cuda")
    )
    assert report["status"] == "PASS"
    assert report["forbidden_semantic_operations_executed"] == 0
    assert report["fit_input_count"] == 12_686
    assert report["checkpoint_input_count"] == 1_600
    encoded = diagnostic.canonical_json(report)
    for forbidden in (
        "sample_id",
        "gold_ids",
        "normalized_gain",
        "base_length",
        "direct_length",
        "/hpc2hdd",
    ):
        assert forbidden not in encoded


def test_complete_comparison_census_is_exact_and_field_closed() -> None:
    total = 12_686 + 1_600
    census = diagnostic.expected_comparison_census(total, 2_560)
    assert frozenset(census) == diagnostic.EXPECTED_FIELD_NAMES
    assert census["anchor_ids_copy"] == total
    assert census["hidden_copy"] == total * 15 * 2_560
    assert census["candidate_logits_copy"] == total * 15 * 16
    assert census["state_difference"] == total * 15 * 64 + 12 * 15
    assert census["entropy"] == total * 15 + 850 * 15
    assert census["retained_mass"] == total * 15 + 850 * 15
    state = diagnostic.DiagnosticState()
    with pytest.raises(RuntimeError, match="incomplete or duplicated"):
        diagnostic.assert_complete_comparison_census(
            state, {"fit": 12_686, "checkpoint": 1_600}, 2_560
        )
