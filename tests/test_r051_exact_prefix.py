from __future__ import annotations

import pytest
import torch

from evaluate_r051_exact_prefix import (
    TARGET_EAL,
    assemble_split_verifier_logits,
    choose_route,
)


@pytest.mark.parametrize("seed_length", [2, 3, 4])
def test_split_geometry_retains_all_decisions_and_bonus(seed_length: int) -> None:
    vocab = 5
    seed = torch.arange(seed_length * vocab).view(1, seed_length, vocab)
    final_rows = 17 - seed_length
    final = 1000 + torch.arange(final_rows * vocab).view(1, final_rows, vocab)
    aligned = assemble_split_verifier_logits(seed, final)
    assert aligned.shape == (1, 17, vocab)
    assert torch.equal(aligned[:, :seed_length], seed)
    # Final row zero is the p_s decision produced by input p{s-1}.
    assert torch.equal(aligned[:, seed_length], final[:, 0])
    # Row 16 is the bonus decision produced by input p15.
    assert torch.equal(aligned[:, 16], final[:, -1])


def test_split_geometry_rejects_starting_final_verifier_at_p_s() -> None:
    seed = torch.zeros(1, 3, 7)
    # Starting at p_s would produce one fewer row than required.
    with pytest.raises(ValueError, match="p.s-1"):
        assemble_split_verifier_logits(seed, torch.zeros(1, 13, 7))


def test_route_uses_unsplit_values_and_selects_smallest_system_seed() -> None:
    decision, selected = choose_route(
        {2: TARGET_EAL - 0.1, 3: 9.2, 4: 9.5},
    )
    assert decision == "GO_SYSTEM_PROFILE"
    assert selected == 3


def test_route_closes_below_target_and_stops_without_timing_below_nine() -> None:
    assert choose_route(
        {2: TARGET_EAL - 0.3, 3: TARGET_EAL - 0.2, 4: TARGET_EAL - 0.1},
    ) == ("CLOSE_EXACT_SEED_FAMILY_ACCURACY_FAIL", None)
    assert choose_route(
        {2: 8.4, 3: 8.8, 4: 8.9},
    ) == ("ACCURACY_PASS_SYSTEM_NO_GO_WITHOUT_TIMING", None)
