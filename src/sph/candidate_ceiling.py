"""Utilities for measuring the hard ceiling of a DFlash candidate lattice."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_candidate_tensors(topk_ids: Tensor, gold_ids: Tensor) -> None:
    if topk_ids.ndim < 2:
        raise ValueError("topk_ids must have shape [..., positions, candidates]")
    if gold_ids.shape != topk_ids.shape[:-1]:
        raise ValueError(
            f"gold_ids shape {tuple(gold_ids.shape)} does not match "
            f"topk_ids prefix {tuple(topk_ids.shape[:-1])}"
        )


def gold_in_candidates(topk_ids: Tensor, gold_ids: Tensor, k: int) -> Tensor:
    """Return whether each gold token occurs in the first ``k`` candidates."""
    validate_candidate_tensors(topk_ids, gold_ids)
    if not 1 <= k <= topk_ids.shape[-1]:
        raise ValueError(f"k must be in [1, {topk_ids.shape[-1]}], got {k}")
    return (topk_ids[..., :k] == gold_ids.unsqueeze(-1)).any(dim=-1)


def accepted_draft_prefix_lengths(position_matches: Tensor) -> Tensor:
    """Count consecutive matching draft positions before the first miss.

    The known anchor token is not included. Add one to obtain the verification
    advance used by the DFlash/Domino generation loops.
    """
    if position_matches.ndim < 1:
        raise ValueError("position_matches needs a positions dimension")
    return position_matches.to(torch.int64).cumprod(dim=-1).sum(dim=-1)


def gold_candidate_ranks(topk_ids: Tensor, gold_ids: Tensor) -> Tensor:
    """Return one-indexed gold ranks, using K+1 when gold is outside top-K."""
    validate_candidate_tensors(topk_ids, gold_ids)
    matches = topk_ids == gold_ids.unsqueeze(-1)
    k = topk_ids.shape[-1]
    ranks = torch.arange(1, k + 1, device=topk_ids.device)
    sentinel = torch.full_like(ranks, k + 1)
    return torch.where(matches, ranks, sentinel).amin(dim=-1)


def first_top1_miss_gold_rank(topk_ids: Tensor, gold_ids: Tensor) -> Tensor:
    """Gold rank at the first position where DFlash top-1 is wrong.

    A fully correct block receives rank zero because it has no first miss.
    """
    validate_candidate_tensors(topk_ids, gold_ids)
    top1_correct = topk_ids[..., 0] == gold_ids
    has_miss = (~top1_correct).any(dim=-1)
    first_miss = (~top1_correct).to(torch.int64).argmax(dim=-1)
    ranks = gold_candidate_ranks(topk_ids, gold_ids)
    selected = ranks.gather(-1, first_miss.unsqueeze(-1)).squeeze(-1)
    return torch.where(has_miss, selected, torch.zeros_like(selected))


def prefix_coverage(position_coverage: Tensor) -> Tensor:
    """Whether all candidate-coverage events up to each position hold."""
    if position_coverage.ndim < 1:
        raise ValueError("position_coverage needs a positions dimension")
    return position_coverage.to(torch.int64).cumprod(dim=-1).to(torch.bool)
