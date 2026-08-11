from __future__ import annotations

import pytest
import torch

from sph.first_miss_max_regret_selector import (
    FirstMissMaxRegretSelector,
    first_miss_max_regret_loss,
    tie_safe_cost_augmented_hinge,
    utility_optimal_actions,
)
from sph.first_miss_value_selector import action_values_from_residual_scores
from sph.global_direct_selector import GlobalDirectCandidateSelector


def make_selector() -> FirstMissMaxRegretSelector:
    return FirstMissMaxRegretSelector(
        GlobalDirectCandidateSelector(
            hidden_size=8,
            max_positions=3,
            max_candidates=4,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            scope="global",
            mixer="axial",
            node_encoder="additive",
            initialization_seed=43,
        )
    )


def make_inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(47)
    return (
        torch.randn(3, 3, 8, generator=generator),
        torch.randn(3, 3, 4, 8, generator=generator),
        torch.randn(3, 3, 4, generator=generator),
        torch.randn(3, 3, generator=generator),
        torch.randn(3, 8, generator=generator),
    )


def test_utility_oracle_prefers_keep_and_lowest_positive_tie() -> None:
    targets = torch.tensor(
        [
            [0.0, 0.0, -0.2, 0.0],
            [0.0, 0.2, 0.2, -0.4],
        ]
    )
    assert utility_optimal_actions(targets).tolist() == [0, 1]


def test_true_scores_have_zero_loss_and_zero_gradient() -> None:
    targets = torch.tensor(
        [
            [0.0, 0.2, 0.0, -0.5],
            [0.0, 0.0, -0.4, 0.0],
        ]
    )
    scores = targets.clone().requires_grad_(True)
    result = tie_safe_cost_augmented_hinge(scores, targets)
    assert torch.equal(result.per_block_hinge, torch.zeros(2))
    result.loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_one_over_fifteen_targets_cancel_bit_exactly() -> None:
    targets = torch.tensor(
        [
            [0, 2, 0, -7, -12],
            [0, 0, -1, -8, -15],
            [0, 13, -2, -9, 0],
        ],
        dtype=torch.float32,
    ) / 15.0
    scores = targets.clone().requires_grad_(True)
    result = tie_safe_cost_augmented_hinge(scores, targets)
    assert torch.equal(result.per_block_hinge, torch.zeros(3))
    result.loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_explicit_relu_matches_max_including_oracle_value() -> None:
    scores = torch.tensor(
        [[0.0, -0.1, 0.4, -0.3], [0.0, 0.0, -0.2, 0.1]]
    )
    targets = torch.tensor(
        [[0.0, 0.3, 0.0, -0.4], [0.0, 0.0, -0.5, 0.0]]
    )
    result = tie_safe_cost_augmented_hinge(scores, targets)
    oracle = result.oracle_actions
    oracle_scores = scores.gather(1, oracle[:, None]).squeeze(1)
    oracle_values = targets.gather(1, oracle[:, None]).squeeze(1)
    original = (
        scores + oracle_values[:, None] - targets
    ).max(dim=-1).values - oracle_scores
    torch.testing.assert_close(result.per_block_hinge, original)


@pytest.mark.parametrize(
    ("scores", "targets", "expected_action"),
    [
        ([0.0, -0.2, -0.1, -0.3], [0.0, 0.4, 0.0, -0.5], 0),
        ([0.0, 0.5, 0.1, -0.2], [0.0, 0.4, 0.0, -0.5], 1),
        ([0.0, 0.1, 0.5, -0.2], [0.0, 0.4, 0.0, -0.5], 2),
        ([0.0, 0.1, -0.1, 0.5], [0.0, 0.4, 0.0, -0.5], 3),
    ],
)
def test_hinge_bounds_keep_oracle_neutral_and_harmful_deployment(
    scores: list[float],
    targets: list[float],
    expected_action: int,
) -> None:
    result = tie_safe_cost_augmented_hinge(
        torch.tensor([scores]), torch.tensor([targets])
    )
    assert int(result.predicted_actions[0]) == expected_action
    assert float(result.bound_slack[0]) >= -1e-7


def test_positive_score_tie_uses_lowest_edit_action() -> None:
    result = tie_safe_cost_augmented_hinge(
        torch.tensor([[0.0, 0.3, 0.3, -0.1]]),
        torch.tensor([[0.0, 0.0, 0.2, -0.4]]),
    )
    assert int(result.predicted_actions[0]) == 1


def test_nonoracle_competitor_tie_uses_lowest_action() -> None:
    result = tie_safe_cost_augmented_hinge(
        torch.zeros(1, 4),
        torch.tensor([[0.0, 0.2, -0.5, -0.5]]),
    )
    assert int(result.oracle_actions[0]) == 1
    assert int(result.competitor_actions[0]) == 2


def test_all_neutral_block_has_zero_loss_and_gradient() -> None:
    scores = torch.zeros(1, 4, requires_grad=True)
    targets = torch.zeros(1, 4)
    result = tie_safe_cost_augmented_hinge(scores, targets)
    assert int(result.oracle_actions[0]) == 0
    assert float(result.loss.detach()) == 0.0
    result.loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_zero_score_gradient_is_undiluted_and_residual_coupled() -> None:
    residual = torch.zeros(1, 1, 3, requires_grad=True)
    scores = action_values_from_residual_scores(residual)
    targets = torch.tensor([[0.0, 0.2, -0.5]])
    result = tie_safe_cost_augmented_hinge(scores, targets)
    assert int(result.oracle_actions[0]) == 1
    assert int(result.competitor_actions[0]) == 2
    result.loss.backward()
    torch.testing.assert_close(
        residual.grad, torch.tensor([[[0.0, -1.0, 1.0]]])
    )


def test_randomized_fp32_bound() -> None:
    generator = torch.Generator().manual_seed(20260805)
    for _ in range(32):
        scores = torch.randn(19, 11, generator=generator)
        targets = torch.randn(19, 11, generator=generator).clamp(-1, 1)
        scores[:, 0] = 0
        targets[:, 0] = 0
        result = tie_safe_cost_augmented_hinge(scores, targets)
        assert float(result.bound_slack.min()) >= -1e-6


def test_invalid_keep_or_nonfinite_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="KEEP score"):
        tie_safe_cost_augmented_hinge(
            torch.tensor([[0.1, 0.0]]), torch.tensor([[0.0, 0.0]])
        )
    with pytest.raises(ValueError, match="finite"):
        tie_safe_cost_augmented_hinge(
            torch.tensor([[0.0, torch.inf]]), torch.tensor([[0.0, 0.0]])
        )


def test_identity_then_second_backward_reaches_upstream() -> None:
    model = make_selector()
    inputs = make_inputs()
    ranks = torch.tensor([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    available = torch.ones(3, 3, dtype=torch.bool)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    first_output = model(*inputs)
    assert torch.equal(
        first_output.action_values, torch.zeros_like(first_output.action_values)
    )
    first = first_miss_max_regret_loss(first_output, ranks, available)
    first.loss.backward()
    projection = model.backbone.residual_projection.weight
    assert projection.grad is not None
    assert float(projection.grad.norm()) > 0
    upstream_first = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name != "backbone.residual_projection.weight"
    ]
    assert all(
        gradient is None or int(torch.count_nonzero(gradient)) == 0
        for gradient in upstream_first
    )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = first_miss_max_regret_loss(model(*inputs), ranks, available)
    second.loss.backward()
    upstream_second = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name != "backbone.residual_projection.weight"
    ]
    assert any(
        gradient is not None and int(torch.count_nonzero(gradient)) > 0
        for gradient in upstream_second
    )
    assert all(not value.requires_grad for value in inputs)
