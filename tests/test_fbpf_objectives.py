import pytest
import torch
import torch.nn.functional as F

from sph.fbpf import (
    ConstraintState,
    arm_loss,
    build_constraint_state,
    constraint_batched_vjp,
    dpace_loss,
    flatten_batched_vjp_rows,
    margin_state,
)


def _official_complete_block_dpace(logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Pinned D-PACE algebra for one anchor + 15 predicted positions."""

    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), gold.reshape(-1), reduction="none"
    ).view_as(gold)
    weight_mask = torch.ones_like(per_token)
    weight_mask[:, 0] = 0
    with torch.no_grad():
        q = torch.exp(-per_token)
        smooth = 0.5 * q + 0.5
        smooth = torch.where(weight_mask.bool(), smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)
        suffix = torch.flip(
            torch.cumsum(torch.flip(prefix * weight_mask, dims=(-1,)), dim=-1),
            dims=(-1,),
        )
    return (per_token * weight_mask * suffix).sum() / float(logits.shape[0])


def test_full_dpace_matches_pinned_scalar_and_float64_gradient() -> None:
    generator = torch.Generator().manual_seed(101)
    logits = torch.randn(4, 16, 11, generator=generator, dtype=torch.float64)
    logits.requires_grad_(True)
    gold = torch.randint(0, 11, (4, 16), generator=generator)

    reference = _official_complete_block_dpace(logits, gold)
    ours = dpace_loss(logits[:, 1:, :], gold[:, 1:], reduction_divisor=4)
    reference_gradient = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    ours_gradient = torch.autograd.grad(ours, logits)[0]

    torch.testing.assert_close(ours, reference, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(
        ours_gradient, reference_gradient, atol=1e-10, rtol=1e-9
    )
    assert torch.count_nonzero(ours_gradient[:, 0, :]) == 0


def test_dpace_keeps_suffix_supervision() -> None:
    logits = torch.zeros(4, 15, 5, dtype=torch.float64, requires_grad=True)
    gold = torch.zeros(4, 15, dtype=torch.long)
    loss = dpace_loss(logits, gold, reduction_divisor=4)
    gradient = torch.autograd.grad(loss, logits)[0]
    assert torch.count_nonzero(gradient[:, -1, :]) > 0


def test_margin_decisions_use_lowest_id_ties_and_detached_indices() -> None:
    logits = torch.tensor(
        [
            [
                [3.0, 3.0, 5.0, 0.0],
                [4.0, 4.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 7.0],
            ]
        ],
        requires_grad=True,
    )
    gold = torch.tensor([[2, 1, 3]])
    state = margin_state(logits, gold)

    assert state.predictions.tolist() == [[2, 0, 3]]
    assert state.non_gold_winner.tolist() == [[0, 0, 0]]
    assert state.first_mismatch.tolist() == [1]
    assert not state.predictions.requires_grad
    assert not state.non_gold_winner.requires_grad
    assert not state.first_mismatch.requires_grad
    assert state.gold_margin.requires_grad


def test_block_max_feasibility_is_exact_all_position_feasibility() -> None:
    generator = torch.Generator().manual_seed(103)
    adapted_logits = torch.randn(4, 5, 7, generator=generator, requires_grad=True)
    gold = torch.randint(0, 7, (4, 5), generator=generator)
    base_first_mismatch = torch.tensor([0, 1, 3, 5])
    constraints = build_constraint_state(
        margin_state(adapted_logits, gold), base_first_mismatch
    )
    direct = bool(
        torch.all(
            constraints.per_position.detach()[constraints.protected_mask] <= 1e-5
        )
    )
    assert constraints.feasible(1e-5) == direct
    assert constraints.row_ids.tolist() == [1, 2, 3]


def test_argmax_tie_aware_constraints_allow_only_gold_winning_ties() -> None:
    logits = torch.tensor([[[2.0, 2.0]], [[2.0, 2.0]]], requires_grad=True)
    gold = torch.tensor([[0], [1]])
    constraints = build_constraint_state(
        margin_state(logits, gold),
        torch.tensor([1, 1]),
        epsilon_tie=1e-4,
        argmax_tie_aware=True,
    )
    torch.testing.assert_close(
        constraints.per_position.detach().flatten(),
        torch.tensor([0.0, 1e-4]),
    )
    assert constraints.per_position[0, 0] <= 1e-5
    assert constraints.per_position[1, 0] > 1e-5


def test_reference_margin_constraints_reject_any_released_margin_drop() -> None:
    gold = torch.tensor([[0, 0]])
    base_logits = torch.tensor([[[3.0, 1.0], [2.0, 1.0]]])
    adapted_logits = torch.tensor(
        [[[2.5, 1.0], [2.5, 1.0]]], requires_grad=True
    )
    base = margin_state(base_logits, gold)
    constraints = build_constraint_state(
        margin_state(adapted_logits, gold),
        base.first_mismatch,
        reference_gold_margin=base.gold_margin,
    )
    torch.testing.assert_close(
        constraints.per_position.detach(), torch.tensor([[0.5, -0.5]])
    )
    assert not constraints.feasible()


def test_uniform_exact_tie_batched_vjp_has_one_row_per_block() -> None:
    theta = torch.tensor([0.2, 0.3], requires_grad=True)
    per_position = torch.stack(
        (
            torch.stack((theta[0], theta[0], theta[1])),
            torch.stack((theta.sum(), theta.sum(), theta[0])),
        )
    )
    mask = torch.tensor([[True, True, False], [True, True, False]])
    state = ConstraintState(
        per_position=per_position,
        protected_mask=mask,
        row_ids=torch.tensor([0, 1]),
        block_max=torch.stack((theta[0], theta.sum())),
        tie_mask=torch.tensor([[True, True, False], [True, True, False]]),
    )
    gradients = constraint_batched_vjp(state, (theta,))
    flattened = flatten_batched_vjp_rows(gradients)
    torch.testing.assert_close(
        flattened, torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    )


def test_exact_candidate_check_rejects_an_active_position_switch() -> None:
    old = ConstraintState(
        per_position=torch.tensor([[-0.1, -0.2]], requires_grad=True),
        protected_mask=torch.tensor([[True, True]]),
        row_ids=torch.tensor([0]),
        block_max=torch.tensor([-0.1]),
        tie_mask=torch.tensor([[True, False]]),
    )
    candidate = ConstraintState(
        per_position=torch.tensor([[-0.2, 0.1]], requires_grad=True),
        protected_mask=torch.tensor([[True, True]]),
        row_ids=torch.tensor([0]),
        block_max=torch.tensor([0.1]),
        tie_mask=torch.tensor([[False, True]]),
    )
    assert old.feasible()
    assert not candidate.feasible()


def test_ordinary_arms_do_not_require_or_compute_frozen_constraints() -> None:
    generator = torch.Generator().manual_seed(107)
    logits = torch.randn(4, 15, 9, generator=generator, requires_grad=True)
    gold = torch.randint(0, 9, (4, 15), generator=generator)
    arm_a = arm_loss("A", logits, None, gold)
    arm_c = arm_loss("C", logits, None, gold)
    assert arm_a.base is None and arm_a.constraints is None
    assert arm_c.base is None and arm_c.constraints is None
    torch.testing.assert_close(arm_a.total, arm_a.dpace)
    assert arm_c.total >= arm_c.dpace

    with pytest.raises(ValueError, match="requires frozen base logits"):
        arm_loss("D", logits, None, gold)
    arm_d = arm_loss("D", logits, logits.detach().clone(), gold)
    assert arm_d.base is not None and arm_d.constraints is not None
