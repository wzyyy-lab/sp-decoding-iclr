from __future__ import annotations

import pytest
import torch

from materialize_pcld_sidecar import (
    calibrate_manual_numeric_epsilon,
    stable_manual_mismatch_mask,
)


def make_receipt(
    *, equal: list[bool], errors: list[float], batched_margin: list[float], manual_margin: list[float]
) -> dict[str, torch.Tensor]:
    return {
        "top1_equal": torch.tensor(equal, dtype=torch.bool),
        "centered_logit_error": torch.tensor(errors),
        "batched_margin": torch.tensor(batched_margin),
        "manual_margin": torch.tensor(manual_margin),
    }


def test_manual_numeric_parity_uses_agreeing_rows_and_exempts_only_ties() -> None:
    calibration = make_receipt(
        equal=[True, True],
        errors=[0.125, 0.25],
        batched_margin=[3.0, 3.0],
        manual_margin=[3.0, 3.0],
    )
    epsilon = calibrate_manual_numeric_epsilon([calibration])
    assert epsilon == pytest.approx(0.25)

    mismatches = make_receipt(
        equal=[False, False],
        errors=[99.0, 99.0],
        batched_margin=[0.0, 2.0],
        manual_margin=[0.25, 1.5],
    )
    # Per-mismatch errors cannot self-exempt a confident disagreement: only
    # the independently calibrated agreeing-row epsilon is authoritative.
    assert stable_manual_mismatch_mask(mismatches, epsilon).tolist() == [False, True]


def test_manual_numeric_calibration_fails_without_agreeing_rows() -> None:
    receipt = make_receipt(
        equal=[False],
        errors=[0.0],
        batched_margin=[0.0],
        manual_margin=[0.0],
    )
    with pytest.raises(RuntimeError, match="no agreeing rows"):
        calibrate_manual_numeric_epsilon([receipt])
