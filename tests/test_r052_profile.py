from __future__ import annotations

import math

from profile_r052_exact_prefix import (
    quantile_record_indices,
    throughput_analysis,
)


def test_context_quantiles_are_deterministic_and_length_ordered() -> None:
    lengths = [90, 10, 50, 70, 30, 110, 130, 150, 170, 190, 210]
    selected = quantile_record_indices(lengths)
    assert [quantile for quantile, _ in selected] == [0.1, 0.5, 0.9]
    selected_lengths = [lengths[index] for _, index in selected]
    assert selected_lengths == sorted(selected_lengths)


def test_throughput_gate_uses_output_advance_and_complete_cycle_time() -> None:
    result = throughput_analysis(
        domino_eal=7.0,
        r051_eal=9.0,
        domino_ms=40.0,
        r051_ms=42.0,
    )
    assert math.isclose(result["output_advance_ratio"], 1.25)
    assert math.isclose(result["measured_noncommon_time_ratio"], 1.05)
    assert result["noncommon_gate_passed"]
    assert result["minimum_additional_shared_common_path_ms_for_target"] == 0.0


def test_required_common_path_solves_the_target_ratio_exactly() -> None:
    result = throughput_analysis(
        domino_eal=7.0,
        r051_eal=9.0,
        domino_ms=40.0,
        r051_ms=50.0,
    )
    required = float(result["minimum_additional_shared_common_path_ms_for_target"])
    achieved = 1.25 * (required + 40.0) / (required + 50.0)
    assert required > 0.0
    assert math.isclose(achieved, 1.15, rel_tol=1e-12)

