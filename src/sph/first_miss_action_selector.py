"""Base-preserving one-edit actions over a frozen DFlash lattice.

The selector makes one block-level decision: keep the released DFlash rank-zero
path, or replace exactly one position with one non-base top-K candidate.  Gold
tokens are used only to construct supervised targets and evaluation rewards;
the model forward path remains gold-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sph.global_direct_selector import (
    GlobalDirectCandidateSelector,
    GlobalDirectOutput,
)


@dataclass
class FirstMissActionOutput:
    """Gold-free FMAS output for one batch of candidate lattices."""

    action_logits: Tensor
    direct_output: GlobalDirectOutput


@dataclass
class FirstMissActionLossOutput:
    """Canonical one-edit oracle-action imitation loss."""

    loss: Tensor
    target_actions: Tensor
    predicted_actions: Tensor
    per_block_nll: Tensor


def num_first_miss_actions(length: int, candidates: int) -> int:
    """Return ``1 + L * (K - 1)`` for KEEP plus all one-edit actions."""

    if length < 1:
        raise ValueError("length must be positive")
    if candidates < 2:
        raise ValueError("candidates must be at least two")
    return 1 + length * (candidates - 1)


def encode_edit_actions(
    positions: Tensor,
    ranks: Tensor,
    *,
    length: int,
    candidates: int,
) -> Tensor:
    """Encode non-base ``(position, rank)`` pairs as action indices."""

    if positions.shape != ranks.shape:
        raise ValueError("positions and ranks must have equal shape")
    if torch.any((positions < 0) | (positions >= length)):
        raise ValueError("edit position is outside the block")
    if torch.any((ranks < 1) | (ranks >= candidates)):
        raise ValueError("edit rank must be non-base and inside top-K")
    return 1 + positions * (candidates - 1) + (ranks - 1)


def decode_action_indices(
    actions: Tensor,
    *,
    length: int,
    candidates: int,
) -> Tensor:
    """Decode KEEP/one-edit actions into candidate-rank paths."""

    if actions.ndim != 1:
        raise ValueError("actions must have shape [B]")
    if actions.dtype == torch.bool or actions.is_floating_point():
        raise ValueError("actions must use an integer dtype")
    action_count = num_first_miss_actions(length, candidates)
    if torch.any((actions < 0) | (actions >= action_count)):
        raise ValueError("action index is outside the declared action space")

    paths = torch.zeros(
        actions.shape[0],
        length,
        dtype=torch.long,
        device=actions.device,
    )
    edit_mask = actions > 0
    if bool(edit_mask.any()):
        flattened = actions[edit_mask] - 1
        positions = torch.div(
            flattened, candidates - 1, rounding_mode="floor"
        )
        ranks = flattened.remainder(candidates - 1) + 1
        paths[edit_mask, positions] = ranks
    return paths


def canonical_first_miss_actions(
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
    *,
    candidates: int,
) -> Tensor:
    """Return the canonical pointwise-optimal action in the one-edit space.

    KEEP (index zero) is the declared tie-break for a fully correct base path
    and for a base first miss whose gold token is outside the retained lattice.
    Otherwise the target repairs exactly the base path's first miss.
    """

    if gold_candidate_indices.shape != gold_in_lattice.shape:
        raise ValueError("gold ranks and availability must have equal shape")
    if gold_candidate_indices.ndim != 2:
        raise ValueError("gold ranks must have shape [B, L]")
    if candidates < 2:
        raise ValueError("candidates must be at least two")
    if torch.any(
        gold_in_lattice
        & (
            (gold_candidate_indices < 0)
            | (gold_candidate_indices >= candidates)
        )
    ):
        raise ValueError("available gold rank is outside top-K")

    base_correct = gold_in_lattice & gold_candidate_indices.eq(0)
    base_wrong = ~base_correct
    has_miss = base_wrong.any(dim=-1)
    first_miss = base_wrong.to(torch.int64).argmax(dim=-1)
    first_available = gold_in_lattice.gather(
        1, first_miss[:, None]
    ).squeeze(1)
    first_rank = gold_candidate_indices.gather(
        1, first_miss[:, None]
    ).squeeze(1)
    repairable = has_miss & first_available
    if bool((repairable & first_rank.eq(0)).any()):
        raise RuntimeError(
            "a base first miss cannot have available gold candidate rank zero"
        )

    actions = torch.zeros_like(first_miss)
    if bool(repairable.any()):
        actions[repairable] = encode_edit_actions(
            first_miss[repairable],
            first_rank[repairable],
            length=gold_candidate_indices.shape[1],
            candidates=candidates,
        )
    return actions


def action_logits_from_scores(scores: Tensor) -> Tensor:
    """Construct KEEP versus one-edit logits from direct candidate scores."""

    if scores.ndim != 3:
        raise ValueError("scores must have shape [B, L, K]")
    batch, length, candidates = scores.shape
    num_first_miss_actions(length, candidates)
    edit_logits = scores[..., 1:] - scores[..., :1]
    keep_logits = torch.zeros(
        batch, 1, dtype=scores.dtype, device=scores.device
    )
    return torch.cat(
        [keep_logits, edit_logits.reshape(batch, -1)], dim=-1
    )


def realized_prefix_lengths(
    candidate_paths: Tensor,
    candidate_ids: Tensor,
    gold_ids: Tensor,
) -> Tensor:
    """Count consecutive correct emitted tokens from position zero."""

    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [B, L, K]")
    batch, length, candidates = candidate_ids.shape
    if candidate_paths.shape != (batch, length):
        raise ValueError("candidate_paths must have shape [B, L]")
    if gold_ids.shape != (batch, length):
        raise ValueError("gold_ids must have shape [B, L]")
    if torch.any((candidate_paths < 0) | (candidate_paths >= candidates)):
        raise ValueError("candidate path rank is outside top-K")
    selected = candidate_ids.gather(
        -1, candidate_paths.unsqueeze(-1)
    ).squeeze(-1)
    correct = selected.eq(gold_ids)
    return correct.to(torch.int64).cumprod(dim=-1).sum(dim=-1)


def first_miss_action_loss(
    output: FirstMissActionOutput,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
) -> FirstMissActionLossOutput:
    """Cross-entropy imitation of the canonical one-edit oracle action."""

    if output.direct_output.scores.ndim != 3:
        raise ValueError("direct scores must have shape [B, L, K]")
    batch, length, candidates = output.direct_output.scores.shape
    expected_shape = (batch, num_first_miss_actions(length, candidates))
    if output.action_logits.shape != expected_shape:
        raise ValueError("action logits have an invalid shape")
    targets = canonical_first_miss_actions(
        gold_candidate_indices,
        gold_in_lattice,
        candidates=candidates,
    )
    per_block_nll = F.cross_entropy(
        output.action_logits.float(), targets, reduction="none"
    )
    return FirstMissActionLossOutput(
        loss=per_block_nll.mean(),
        target_actions=targets,
        predicted_actions=output.action_logits.argmax(dim=-1),
        per_block_nll=per_block_nll,
    )


class FirstMissActionSelector(nn.Module):
    """Gold-free FMAS wrapper around an unchanged direct selector backbone."""

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
    ) -> FirstMissActionOutput:
        direct_output = self.backbone(
            hidden,
            candidate_embeddings,
            candidate_logits,
            base_logsumexp,
            anchor_embeddings,
        )
        return FirstMissActionOutput(
            action_logits=action_logits_from_scores(direct_output.scores),
            direct_output=direct_output,
        )

