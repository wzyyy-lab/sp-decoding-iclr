"""Tie-safe cost-augmented max-regret selection on a DFlash lattice.

The action set and exact-identity residual parameterization are inherited from
Signed Action-Value Selection.  Unlike dense value MSE, CAMRS trains the
single hardest cost-augmented decision constraint in each block.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sph.first_miss_value_selector import (
    FirstMissValueOutput,
    FirstMissValueSelector,
    decode_strict_positive_actions,
    dense_signed_action_values,
)


BOUND_TOLERANCE = 1e-6


@dataclass
class CostAugmentedHingeOutput:
    """Per-block CAMRS quantities in normalized accepted-token units."""

    loss: Tensor
    per_block_hinge: Tensor
    raw_max_violations: Tensor
    target_values: Tensor
    oracle_actions: Tensor
    predicted_actions: Tensor
    competitor_actions: Tensor
    decoded_regret: Tensor
    bound_slack: Tensor


def utility_optimal_actions(target_values: Tensor) -> Tensor:
    """Return KEEP-preferred, lowest-index utility-optimal actions.

    KEEP is column zero and must have exact utility zero.  ``torch.argmax``
    returns the first maximum, so it implements both required tie rules.
    """

    if target_values.ndim != 2 or target_values.shape[1] < 2:
        raise ValueError("target values must have shape [B, A] with A >= 2")
    if not bool(torch.isfinite(target_values).all()):
        raise ValueError("target values must be finite")
    if not torch.equal(
        target_values[:, 0], torch.zeros_like(target_values[:, 0])
    ):
        raise ValueError("KEEP target value must be exactly zero")
    return target_values.argmax(dim=-1)


def tie_safe_cost_augmented_hinge(
    action_scores: Tensor,
    target_values: Tensor,
) -> CostAugmentedHingeOutput:
    """Compute the explicit non-oracle CAMRS hinge in FP32.

    The oracle is excluded before the maximum and a final ReLU provides the
    specified zero derivative at a zero-margin tie.  Non-oracle maximum ties
    use the lowest action index through ``torch.max``.
    """

    if action_scores.shape != target_values.shape:
        raise ValueError("action scores and target values must have equal shape")
    if action_scores.ndim != 2 or action_scores.shape[1] < 2:
        raise ValueError("action tensors must have shape [B, A] with A >= 2")
    scores = action_scores.float()
    targets = target_values.float()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("action scores must be finite")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("target values must be finite")
    if not torch.equal(scores[:, 0], torch.zeros_like(scores[:, 0])):
        raise ValueError("KEEP score must be exactly zero")

    oracle_actions = utility_optimal_actions(targets)
    predicted_actions = decode_strict_positive_actions(scores)
    oracle_scores = scores.gather(1, oracle_actions[:, None]).squeeze(1)
    oracle_values = targets.gather(1, oracle_actions[:, None]).squeeze(1)
    # Compute equal margins through identical subtraction order.  The direct
    # algebra ``s + v* - v - s*`` leaves tiny positive roundoff for real
    # multiples of 1/L and can create a nonzero ReLU subgradient at ``s=v``.
    # Here both margins are bit-identical at ``s=v``, so their difference is
    # exactly zero in FP32.
    target_margins = oracle_values[:, None] - targets
    predicted_margins = oracle_scores[:, None] - scores
    violations = target_margins - predicted_margins
    non_oracle = torch.ones_like(violations, dtype=torch.bool)
    non_oracle.scatter_(1, oracle_actions[:, None], False)
    masked_violations = violations.masked_fill(~non_oracle, -torch.inf)
    raw_max_violations, competitor_actions = masked_violations.max(dim=-1)
    per_block_hinge = torch.relu(raw_max_violations)

    selected_values = targets.gather(
        1, predicted_actions[:, None]
    ).squeeze(1)
    decoded_regret = oracle_values - selected_values
    bound_slack = per_block_hinge - decoded_regret
    if bool((decoded_regret < -BOUND_TOLERANCE).any()):
        raise RuntimeError("decoded regret became negative")

    return CostAugmentedHingeOutput(
        loss=per_block_hinge.mean(),
        per_block_hinge=per_block_hinge,
        raw_max_violations=raw_max_violations,
        target_values=targets,
        oracle_actions=oracle_actions,
        predicted_actions=predicted_actions,
        competitor_actions=competitor_actions,
        decoded_regret=decoded_regret,
        bound_slack=bound_slack,
    )


def first_miss_max_regret_loss(
    output: FirstMissValueOutput,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
) -> CostAugmentedHingeOutput:
    """Construct exact one-edit utilities and apply the CAMRS hinge."""

    if output.direct_output.residual_scores.ndim != 3:
        raise ValueError("direct residual scores must have shape [B, L, K]")
    _, _, candidates = output.direct_output.residual_scores.shape
    targets = dense_signed_action_values(
        gold_candidate_indices,
        gold_in_lattice,
        candidates=candidates,
    )
    return tie_safe_cost_augmented_hinge(output.action_values, targets)


class FirstMissMaxRegretSelector(FirstMissValueSelector):
    """Exact-identity residual selector trained with CAMRS."""
