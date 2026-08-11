from __future__ import annotations

import pytest

from compare_gfpr_dynamic_rollouts import _paired, _trajectory_metrics


def _record(sample: str, offset: int, accepted: int, next_offset: int) -> dict:
    return {
        "sample_id": sample,
        "anchor_offset": offset,
        "accepted_length": accepted,
        "next_anchor_offset": next_offset,
    }


def test_dynamic_trajectories_are_paired_by_prompt_not_block() -> None:
    baseline = _trajectory_metrics(
        [_record("p", 0, 2, 3), _record("p", 3, 1, 5)]
    )
    adapted = _trajectory_metrics([_record("p", 0, 4, 5)])
    assert baseline["p"]["cycles"] == 2
    assert baseline["p"]["eal"] == 1.5
    assert adapted["p"]["cycles"] == 1
    assert adapted["p"]["eal"] == 4.0
    paired = _paired(
        ["p"],
        [baseline["p"]["eal"]],
        [adapted["p"]["eal"]],
        bootstrap_samples=20,
    )
    assert paired["paired_delta"] == 2.5
    assert paired["adapted_to_baseline_ratio"] == pytest.approx(4.0 / 1.5)


def test_dynamic_chain_break_fails_closed() -> None:
    with pytest.raises(ValueError, match="broken dynamic chain"):
        _trajectory_metrics(
            [_record("p", 0, 2, 4), _record("p", 3, 1, 5)]
        )
