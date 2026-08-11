import math

import pytest
import torch

from sph.fbpf_protocol import (
    DOMAIN_ORDER,
    RESERVE_EXHAUSTED,
    AllocationFailure,
    CandidateRow,
    NearDuplicateComponent,
    ReserveExhausted,
    allocate_components,
    build_near_duplicate_components,
    cluster_bootstrap,
    component_identifier,
    deterministic_complete_offsets,
    exact_jaccard,
    extract_continuation_blocks,
    latency_tost,
    mean_matched_seed_prompt_metric,
    normalize_text,
    overlap_ngrams,
    percentile_interval,
    power_requirements,
    prompt_metrics,
    reserve_quota,
    select_complete_sequence_rows,
    tokenize_for_overlap,
)


def test_four_offset_formula_and_verifier_labels_are_exact() -> None:
    assert deterministic_complete_offsets(19) == (0, 1, 2, 3)
    assert deterministic_complete_offsets(128) == (0, 37, 75, 112)
    with pytest.raises(ValueError, match="at least 19"):
        deterministic_complete_offsets(18)

    continuation = torch.arange(19)
    blocks = extract_continuation_blocks(continuation)
    assert blocks.offsets == (0, 1, 2, 3)
    assert blocks.anchors.tolist() == [0, 1, 2, 3]
    assert blocks.gold.shape == (4, 15)
    assert blocks.gold[3].tolist() == list(range(4, 19))


def test_normalization_tokenization_and_short_ngram_are_frozen() -> None:
    text = "  Ａbc\n\tDEF！  "
    assert normalize_text(text) == "abc def!"
    assert tokenize_for_overlap(text) == ("abc", "def", "!")
    assert overlap_ngrams(text) == frozenset({("abc", "def", "!")})
    assert exact_jaccard(overlap_ngrams(text), overlap_ngrams("ABC def!")) == 1.0


def test_component_builder_excludes_cross_domain_components_deterministically() -> None:
    rows = (
        CandidateRow(0, "math", "same normalized document"),
        CandidateRow(1, "code", "SAME   normalized document"),
        CandidateRow(2, "math", "a separate mathematical prompt"),
        CandidateRow(3, "math", "another unrelated prompt"),
    )
    forward = build_near_duplicate_components(rows)
    reverse = build_near_duplicate_components(tuple(reversed(rows)))
    assert forward == reverse
    assert forward.excluded_cross_domain_ordinals == (0, 1)
    assert {row.source_ordinal for component in forward.components for row in component.rows} == {
        2,
        3,
    }


def _singleton_components(count: int) -> tuple[NearDuplicateComponent, ...]:
    components = []
    for ordinal in range(count):
        row = CandidateRow(ordinal, "math", f"unique synthetic row number {ordinal}")
        components.append(
            NearDuplicateComponent(
                component_id=component_identifier((row,)),
                domain="math",
                rows=(row,),
            )
        )
    return tuple(components)


def test_component_atomic_allocator_has_exact_active_and_reserve_counts() -> None:
    components = _singleton_components(70)
    quotas = {("fit", "math"): 2}
    left = allocate_components(components, quotas)
    right = allocate_components(tuple(reversed(components)), quotas)
    assert left == right
    assert len(left.rows("fit", "math", "active")) == 2
    assert len(left.rows("fit", "math", "reserve")) == reserve_quota(2) == 64
    assert len(set(record.component_id for record in left.records)) == len(left.records)

    with pytest.raises(AllocationFailure):
        allocate_components(components[:60], quotas)


def test_short_attempts_are_consumed_and_reserves_never_reassigned() -> None:
    allocation = allocate_components(_singleton_components(70), {("fit", "math"): 2})
    active = allocation.rows("fit", "math", "active")
    reserves = allocation.rows("fit", "math", "reserve")
    lengths = {record.source_ordinal: 19 for record in allocation.records}
    lengths[active[0].source_ordinal] = 18
    lengths[reserves[0].source_ordinal] = 18
    selected = select_complete_sequence_rows(
        allocation,
        split="fit",
        domain="math",
        continuation_lengths=lengths,
    )
    assert len(selected.selected_ordinals) == 2
    assert selected.consumed_attempt_ordinals[:3] == (
        active[0].source_ordinal,
        reserves[0].source_ordinal,
        reserves[1].source_ordinal,
    )

    for reserve in reserves:
        lengths[reserve.source_ordinal] = 0
    with pytest.raises(ReserveExhausted) as error:
        select_complete_sequence_rows(
            allocation,
            split="fit",
            domain="math",
            continuation_lengths=lengths,
        )
    assert error.value.failure_class == RESERVE_EXHAUSTED


def test_prompt_metrics_use_released_references_and_equal_blocks() -> None:
    released = torch.tensor([[2, 0, 1, 3], [1, 1, 1, 1]])
    model = torch.tensor([[0, 0, 2, 3], [1, 1, 1, 1]])
    metrics = prompt_metrics(released, model)
    torch.testing.assert_close(metrics.eal, torch.tensor([2.25, 2.0], dtype=torch.float64))
    torch.testing.assert_close(
        metrics.harm_rate, torch.tensor([0.25, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        metrics.mean_harm, torch.tensor([0.5, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        metrics.first_token_contrast,
        torch.tensor([-0.25, 0.0], dtype=torch.float64),
    )
    seeds = torch.stack((metrics.eal, metrics.eal + 1, metrics.eal + 2))
    torch.testing.assert_close(mean_matched_seed_prompt_metric(seeds), metrics.eal + 1)


def test_component_cluster_bootstrap_and_linear_percentiles_are_reproducible() -> None:
    values = torch.tensor([1.0, 3.0, 2.0, 4.0], dtype=torch.float64)
    component_ids = ("a", "a", "b", "c")
    domains = ("math", "math", "code", "chat")
    first = cluster_bootstrap(
        values, component_ids, domains, replicates=64, seed=2_026_080_601
    )
    second = cluster_bootstrap(
        values, component_ids, domains, replicates=64, seed=2_026_080_601
    )
    assert torch.equal(first, second)
    lower, upper = percentile_interval(first)
    assert lower <= upper
    assert len(percentile_interval(first, one_sided="lower")) == 1
    assert len(percentile_interval(first, one_sided="upper")) == 1


def test_power_formula_uses_every_registered_requirement() -> None:
    names = {
        "d_vs_released_eal",
        "d_vs_a_eal",
        "d_vs_b_eal",
        "d_vs_c_released_referenced_harm",
        "d_vs_c_eal",
        "first_token_vs_released",
    }
    result = power_requirements(
        sd_upper_by_contrast={name: 0.2 for name in names},
        icc_upper=0.1,
        cv_cluster_size_upper=0.5,
        mean_cluster_size_upper=2.0,
        sd_mean_harm_upper=0.4,
    )
    assert result.design_effect == pytest.approx(1.15)
    assert result.n_power == max(result.requirements.values())
    assert set(names).issubset(result.requirements)
    assert "harm_rate_upper_bound_precision" in result.requirements
    assert "mean_harm_upper_bound_precision" in result.requirements


def test_latency_student_t_ci_and_strict_endpoints() -> None:
    assert latency_tost([0.0] * 20).passed
    assert not latency_tost([0.03] * 20).passed
    at_upper_boundary = latency_tost([math.log(1.02)] * 20)
    assert not at_upper_boundary.passed
    assert at_upper_boundary.sample_standard_deviation == pytest.approx(0.0)
