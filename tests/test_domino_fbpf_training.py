import torch
import pytest

from sph.fbpf import BatchLinearization
from train_domino_fbpf import aggregate_linearizations


def test_macro_linearization_averages_tasks_and_joins_constraints() -> None:
    first = BatchLinearization(
        constraint_values=torch.tensor([-0.2, 0.0], dtype=torch.float64),
        constraint_gradients=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
        ),
        max_all_position_constraint=0.0,
        task_gradient=torch.tensor([2.0, 4.0]),
        row_ids=(0, 3),
    )
    second = BatchLinearization(
        constraint_values=torch.tensor([-0.1], dtype=torch.float64),
        constraint_gradients=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        max_all_position_constraint=0.2,
        task_gradient=torch.tensor([4.0, 8.0]),
        row_ids=(2,),
    )
    result = aggregate_linearizations(
        [first, second], need_task=True, need_vjp=True
    )
    torch.testing.assert_close(result.task_gradient, torch.tensor([3.0, 6.0]))
    torch.testing.assert_close(
        result.constraint_values, torch.tensor([-0.2, 0.0, -0.1], dtype=torch.float64)
    )
    assert result.constraint_gradients.shape == (3, 2)
    assert result.row_ids == (0, 3, 6)
    assert result.max_all_position_constraint == 0.2


def test_macro_linearization_rejects_nan_in_a_later_prompt() -> None:
    safe = BatchLinearization(
        constraint_values=torch.tensor([-0.2], dtype=torch.float64),
        constraint_gradients=None,
        max_all_position_constraint=-0.2,
        task_gradient=None,
        row_ids=(0,),
    )
    invalid = BatchLinearization(
        constraint_values=torch.tensor([float("nan")], dtype=torch.float64),
        constraint_gradients=None,
        max_all_position_constraint=float("nan"),
        task_gradient=None,
        row_ids=(0,),
    )
    with pytest.raises(FloatingPointError, match="non-finite macro"):
        aggregate_linearizations(
            [safe, invalid], need_task=False, need_vjp=False
        )
