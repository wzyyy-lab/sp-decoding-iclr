import pytest
import torch

from sph.fbpf import (
    BatchLinearization,
    ProjectionBudgetExhausted,
    TransactionState,
    TransactionalAdamW,
    cosine_warmup_learning_rate,
    project_task_direction,
)


def _one_dimensional_evaluation(
    theta: torch.Tensor,
    *,
    boundary: float,
    task_gradient: float,
    need_task: bool,
    need_vjp: bool,
) -> BatchLinearization:
    constraint = theta[:1].double() - boundary
    return BatchLinearization(
        constraint_values=constraint,
        constraint_gradients=(
            torch.ones(1, 1, dtype=torch.float64) if need_vjp else None
        ),
        max_all_position_constraint=float(constraint.item()),
        task_gradient=(
            torch.tensor([task_gradient], dtype=torch.float32) if need_task else None
        ),
        row_ids=(0,),
    )


def test_projection_uses_complete_rows_and_stable_ties() -> None:
    raw = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.2, 0.2], dtype=torch.float64)
    gradients = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    projected = project_task_direction(
        raw, values, gradients, row_ids=(0, 1), max_sweeps=1
    )
    torch.testing.assert_close(
        projected, torch.tensor([-0.2, -0.2], dtype=torch.float64)
    )


def test_projection_budget_exhaustion_is_terminal() -> None:
    # Alternating non-orthogonal halfspaces converge cyclically but do not meet
    # the frozen tolerance in one sweep.  Exact nonlinear backtracking must not
    # turn an exhausted linear projection into a silent commit.
    raw = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.4, 0.4], dtype=torch.float64)
    gradients = torch.tensor([[1.0, 0.0], [-1.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ProjectionBudgetExhausted):
        project_task_direction(
            raw,
            values,
            gradients,
            row_ids=(0, 1),
            max_sweeps=1,
            tau_linear=1e-12,
        )


def test_feasible_task_commit_updates_full_shadow_state_once() -> None:
    optimizer = TransactionalAdamW()
    state = TransactionState.initialize(torch.tensor([0.0]))

    def evaluate(theta: torch.Tensor, need_task: bool, need_vjp: bool) -> BatchLinearization:
        return _one_dimensional_evaluation(
            theta,
            boundary=0.5,
            task_gradient=-1.0,
            need_task=need_task,
            need_vjp=need_vjp,
        )

    result = optimizer.step(state, evaluate, learning_rate=0.1)
    assert result.status == "committed"
    assert not result.restored and not result.aborted
    assert result.state.k_outer == 1
    assert result.state.t_adam == 1
    torch.testing.assert_close(result.state.first_moment, torch.tensor([-0.1]))
    torch.testing.assert_close(result.state.second_moment, torch.tensor([0.05]))
    assert result.state.theta.item() > 0.0


def test_successful_restoration_recomputes_then_commits_same_minibatch() -> None:
    optimizer = TransactionalAdamW()
    state = TransactionState.initialize(torch.tensor([1.0]))
    calls: list[tuple[float, bool, bool]] = []

    def evaluate(theta: torch.Tensor, need_task: bool, need_vjp: bool) -> BatchLinearization:
        calls.append((float(theta.item()), need_task, need_vjp))
        return _one_dimensional_evaluation(
            theta,
            boundary=0.2,
            task_gradient=1.0,
            need_task=need_task,
            need_vjp=need_vjp,
        )

    result = optimizer.step(state, evaluate, learning_rate=0.1)
    assert result.status == "committed"
    assert result.restored and result.restoration_cycles == 1
    assert result.state.k_outer == 1 and result.state.t_adam == 1
    assert result.state.theta.item() < 0.2
    # Initial full linearization, exact restoration candidate, then full refresh.
    assert calls[0][1:] == (True, True)
    assert calls[1][1:] == (False, False)
    assert calls[2][1:] == (True, True)


def test_successful_restoration_then_task_skip_retains_only_restored_theta() -> None:
    optimizer = TransactionalAdamW()
    state = TransactionState.initialize(torch.tensor([1.0]))

    def evaluate(theta: torch.Tensor, need_task: bool, need_vjp: bool) -> BatchLinearization:
        value = float(theta.item())
        if value > 0.9:
            constraint = value - 0.2
            gradient = 1.0
        elif need_task or need_vjp:
            constraint = -1e-4
            gradient = 0.0
        elif value <= 0.2:
            # The exact restoration candidate is feasible.
            constraint = -1e-4
            gradient = 0.0
        else:
            # Every nonlinear task candidate exposes an active-switch violation.
            constraint = 0.5
            gradient = 0.0
        return BatchLinearization(
            constraint_values=torch.tensor([constraint], dtype=torch.float64),
            constraint_gradients=(
                torch.tensor([[gradient]], dtype=torch.float64) if need_vjp else None
            ),
            max_all_position_constraint=constraint,
            task_gradient=(
                torch.tensor([-1.0], dtype=torch.float32) if need_task else None
            ),
            row_ids=(0,),
        )

    result = optimizer.step(state, evaluate, learning_rate=0.1)
    assert result.status == "task_skipped"
    assert result.restored and not result.aborted
    assert result.state.theta.item() < 0.2
    assert result.state.theta.item() != state.theta.item()
    assert torch.equal(result.state.first_moment, state.first_moment)
    assert torch.equal(result.state.second_moment, state.second_moment)
    assert result.state.t_adam == state.t_adam
    assert result.state.k_outer == state.k_outer + 1
    assert result.attempted_alphas == optimizer.candidate_alphas


def test_failed_restoration_rolls_back_parameters_moments_and_t_adam() -> None:
    optimizer = TransactionalAdamW()
    state = TransactionState(
        theta=torch.tensor([1.0]),
        first_moment=torch.tensor([0.3]),
        second_moment=torch.tensor([0.4]),
        k_outer=7,
        t_adam=5,
    )

    def evaluate(theta: torch.Tensor, need_task: bool, need_vjp: bool) -> BatchLinearization:
        return BatchLinearization(
            constraint_values=torch.tensor([0.8], dtype=torch.float64),
            constraint_gradients=(
                torch.zeros(1, 1, dtype=torch.float64) if need_vjp else None
            ),
            max_all_position_constraint=0.8,
            task_gradient=(
                torch.tensor([1.0], dtype=torch.float32) if need_task else None
            ),
            row_ids=(0,),
        )

    result = optimizer.step(state, evaluate, learning_rate=0.1)
    assert result.status == "restoration_failed"
    assert result.aborted and not result.restored
    assert torch.equal(result.state.theta, state.theta)
    assert torch.equal(result.state.first_moment, state.first_moment)
    assert torch.equal(result.state.second_moment, state.second_moment)
    assert result.state.t_adam == state.t_adam
    assert result.state.k_outer == state.k_outer + 1


def test_learning_rate_has_frozen_warmup_and_cosine_endpoints() -> None:
    first = cosine_warmup_learning_rate(0)
    warmup_end = cosine_warmup_learning_rate(319)
    final = cosine_warmup_learning_rate(7_999)
    assert first == 1e-4 / 320
    assert warmup_end == 1e-4
    assert abs(final) < 1e-20
