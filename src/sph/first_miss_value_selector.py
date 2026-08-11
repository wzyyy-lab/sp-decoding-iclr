"""Signed one-edit action values over a frozen DFlash candidate lattice.

The selector keeps the released DFlash rank-zero path or changes exactly one
position to one non-base top-K candidate.  Gold tokens construct dense signed
accepted-prefix advantages only for training and evaluation; model inference
remains gold-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sph.first_miss_action_selector import (
    decode_action_indices,
    num_first_miss_actions,
)
from sph.global_direct_selector import (
    GlobalDirectCandidateSelector,
    GlobalDirectOutput,
)


@dataclass
class FirstMissValueOutput:
    """Gold-free SAVS output for one batch of candidate lattices."""

    action_values: Tensor
    direct_output: GlobalDirectOutput


@dataclass
class FirstMissValueLossOutput:
    """Dense signed-value regression output."""

    loss: Tensor
    target_values: Tensor
    predicted_actions: Tensor
    per_block_mse: Tensor
    squared_errors: Tensor


def dense_signed_action_values(
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
    *,
    candidates: int,
) -> Tensor:
    """Return normalized prefix advantages for KEEP and every one-edit action.

    The returned shape is ``[B, 1 + L * (K - 1)]``.  Column zero is KEEP and
    is exactly zero.  Remaining columns follow ``decode_action_indices`` order.
    """

    if gold_candidate_indices.shape != gold_in_lattice.shape:
        raise ValueError("gold ranks and availability must have equal shape")
    if gold_candidate_indices.ndim != 2:
        raise ValueError("gold ranks must have shape [B, L]")
    if candidates < 2:
        raise ValueError("candidates must be at least two")
    if gold_in_lattice.dtype != torch.bool:
        raise ValueError("gold availability must be boolean")
    if torch.any(
        gold_in_lattice
        & (
            (gold_candidate_indices < 0)
            | (gold_candidate_indices >= candidates)
        )
    ):
        raise ValueError("available gold rank is outside top-K")

    _, length = gold_candidate_indices.shape
    if length < 1:
        raise ValueError("block length must be positive")
    action_count = num_first_miss_actions(length, candidates)
    paths = decode_action_indices(
        torch.arange(action_count, device=gold_candidate_indices.device),
        length=length,
        candidates=candidates,
    )
    correct = gold_in_lattice[:, None, :] & paths[None, :, :].eq(
        gold_candidate_indices[:, None, :]
    )
    accepted = correct.to(torch.int64).cumprod(dim=-1).sum(dim=-1)
    advantages = accepted - accepted[:, :1]
    values = advantages.to(torch.float32) / float(length)
    if not torch.equal(values[:, 0], torch.zeros_like(values[:, 0])):
        raise RuntimeError("KEEP signed value is not exactly zero")
    return values


def action_values_from_residual_scores(residual_scores: Tensor) -> Tensor:
    """Construct KEEP/edit values from candidate residual-score differences."""

    if residual_scores.ndim != 3:
        raise ValueError("residual scores must have shape [B, L, K]")
    batch, length, candidates = residual_scores.shape
    num_first_miss_actions(length, candidates)
    edit_values = residual_scores[..., 1:] - residual_scores[..., :1]
    keep_values = torch.zeros(
        batch,
        1,
        dtype=residual_scores.dtype,
        device=residual_scores.device,
    )
    return torch.cat(
        [keep_values, edit_values.reshape(batch, -1)], dim=-1
    )


def decode_strict_positive_actions(action_values: Tensor) -> Tensor:
    """Choose the largest edit only when its predicted value is positive."""

    if action_values.ndim != 2 or action_values.shape[1] < 2:
        raise ValueError("action values must have shape [B, 1+A] with A >= 1")
    if not bool(torch.isfinite(action_values).all()):
        raise ValueError("action values must be finite")
    if not torch.equal(
        action_values[:, 0], torch.zeros_like(action_values[:, 0])
    ):
        raise ValueError("KEEP action value must be exactly zero")
    best_edit_values, best_edit_indices = action_values[:, 1:].max(dim=-1)
    return torch.where(
        best_edit_values > 0,
        best_edit_indices + 1,
        torch.zeros_like(best_edit_indices),
    )


def first_miss_value_loss(
    output: FirstMissValueOutput,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
) -> FirstMissValueLossOutput:
    """Regress all non-KEEP signed one-edit values with uniform squared loss."""

    if output.direct_output.residual_scores.ndim != 3:
        raise ValueError("direct residual scores must have shape [B, L, K]")
    batch, length, candidates = output.direct_output.residual_scores.shape
    expected_shape = (batch, num_first_miss_actions(length, candidates))
    if output.action_values.shape != expected_shape:
        raise ValueError("action values have an invalid shape")
    targets = dense_signed_action_values(
        gold_candidate_indices,
        gold_in_lattice,
        candidates=candidates,
    )
    squared_errors = (
        output.action_values[:, 1:].float() - targets[:, 1:]
    ).square()
    per_block_mse = squared_errors.mean(dim=-1)
    return FirstMissValueLossOutput(
        loss=per_block_mse.mean(),
        target_values=targets,
        predicted_actions=decode_strict_positive_actions(
            output.action_values.float()
        ),
        per_block_mse=per_block_mse,
        squared_errors=squared_errors,
    )


class FirstMissValueSelector(nn.Module):
    """Gold-free SAVS wrapper around an unchanged direct-selector backbone."""

    def __init__(self, backbone: GlobalDirectCandidateSelector) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        hidden: Tensor,
        candidate_embeddings: Tensor,
        candidate_logits: Tensor,
        base_logsumexp: Tensor,
        anchor_embeddings: Tensor,
    ) -> FirstMissValueOutput:
        direct_output = self.backbone(
            hidden,
            candidate_embeddings,
            candidate_logits,
            base_logsumexp,
            anchor_embeddings,
        )
        return FirstMissValueOutput(
            action_values=action_values_from_residual_scores(
                direct_output.residual_scores
            ),
            direct_output=direct_output,
        )
