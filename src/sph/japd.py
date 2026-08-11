"""JAPD-16 offline objective and exact evaluation primitives.

The online model remains :class:`GlobalDirectCandidateSelector`: one call
consumes all 16 DFlash positions and returns one ``[B, 16, 16]`` score tensor.
This module contains only offline labels, losses, and ground-truth metrics.  It
does not implement decoding loops, selected-token feedback, target features,
or multiple proposal paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from sph.global_direct_selector import exact_dpace_position_weights


BLOCK_LENGTH = 16
CANDIDATES = 16
FULL_PREFIX_NORMALIZER = BLOCK_LENGTH * (BLOCK_LENGTH + 1) / 2
TEACHER_TEMPERATURE = 2.0
HARD_TARGET_MIXTURE = 0.9


@dataclass(frozen=True)
class JAPDPerBlockOutput:
    """Per-block JAPD terms before dataset-level prompt balancing."""

    per_block_loss: Tensor
    all_prefix_loss: Tensor
    joint_two_frontier_loss: Tensor
    support_mask: Tensor
    horizons: Tensor
    gold_candidate_ranks: Tensor
    base_error_mask: Tensor
    multi_repair_mask: Tensor
    second_error_positions: Tensor
    joint_prefix_mask: Tensor
    target_margins: Tensor


@dataclass(frozen=True)
class JointTwoFrontierMetric:
    """Strict inclusive repair metric for blocks with two base errors."""

    eligible: Tensor
    success: Tensor
    second_error_positions: Tensor

    @property
    def denominator(self) -> int:
        return int(self.eligible.sum().item())

    @property
    def numerator(self) -> int:
        return int(self.success.sum().item())


def validate_full16_lattice(
    candidate_ids: Tensor,
    candidate_logits: Tensor | None = None,
    *,
    require_unique: bool = True,
) -> None:
    """Fail closed unless a batch is the frozen full16-by-K16 lattice."""

    if candidate_ids.ndim != 3 or candidate_ids.shape[1:] != (
        BLOCK_LENGTH,
        CANDIDATES,
    ):
        raise ValueError("candidate_ids must have shape [B,16,16]")
    if candidate_logits is not None and candidate_logits.shape != candidate_ids.shape:
        raise ValueError("candidate_logits must match candidate_ids")
    if require_unique:
        sorted_ids = candidate_ids.sort(dim=-1).values
        if bool(sorted_ids[..., 1:].eq(sorted_ids[..., :-1]).any().item()):
            raise ValueError("every per-position Top-16 candidate set must be unique")


def candidate_gold_ranks(
    candidate_ids: Tensor,
    gold_ids: Tensor,
    *,
    require_unique: bool = True,
) -> Tensor:
    """Return the unique gold rank or ``-1`` when gold is outside K16."""

    validate_full16_lattice(candidate_ids, require_unique=require_unique)
    if gold_ids.shape != candidate_ids.shape[:2]:
        raise ValueError("gold_ids must have shape [B,16]")
    matches = candidate_ids.eq(gold_ids.unsqueeze(-1))
    ranks = matches.to(torch.int64).argmax(dim=-1)
    return torch.where(matches.any(dim=-1), ranks, torch.full_like(ranks, -1))


def clean_support(
    gold_candidate_ranks: Tensor,
    target_matches_gold: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return strict clean-prefix support and its exclusive horizon.

    A row is supervised only while every row through it has both candidate
    coverage and same-geometry target replay agreement with canonical gold.
    The first failing row and its entire suffix are excluded.
    """

    if gold_candidate_ranks.shape != target_matches_gold.shape:
        raise ValueError("rank and target-match tensors must have equal shape")
    if gold_candidate_ranks.ndim != 2 or gold_candidate_ranks.shape[1] != BLOCK_LENGTH:
        raise ValueError("rank tensors must have shape [B,16]")
    row_valid = gold_candidate_ranks.ge(0) & target_matches_gold.bool()
    support = row_valid.to(torch.int64).cumprod(dim=-1).to(torch.bool)
    horizons = support.sum(dim=-1).to(torch.long)
    return support, horizons


def _second_error_geometry(
    gold_candidate_ranks: Tensor,
    support_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return base errors, multi-repair rows, second error, and joint prefix."""

    base_errors = support_mask & gold_candidate_ranks.ne(0)
    error_order = base_errors.to(torch.int64).cumsum(dim=-1)
    second_hits = base_errors & error_order.eq(2)
    multi_repair = base_errors.sum(dim=-1).ge(2)
    sentinel = torch.full(
        (gold_candidate_ranks.shape[0],),
        BLOCK_LENGTH,
        dtype=torch.long,
        device=gold_candidate_ranks.device,
    )
    second_positions = torch.where(
        multi_repair,
        second_hits.to(torch.int64).argmax(dim=-1),
        sentinel,
    )
    positions = torch.arange(
        BLOCK_LENGTH, device=gold_candidate_ranks.device
    ).unsqueeze(0)
    joint_prefix = multi_repair.unsqueeze(-1) & positions.le(
        second_positions.unsqueeze(-1)
    )
    return base_errors, multi_repair, second_positions, joint_prefix


def japd_per_block_loss(
    scores: Tensor,
    candidate_ids: Tensor,
    gold_ids: Tensor,
    target_candidate_logits: Tensor,
    target_matches_gold: Tensor,
    *,
    temperature: float = TEACHER_TEMPERATURE,
    hard_target_mixture: float = HARD_TARGET_MIXTURE,
    full_prefix_normalizer: float = FULL_PREFIX_NORMALIZER,
) -> JAPDPerBlockOutput:
    """Compute the frozen JAPD loss without dataset sampling weights.

    All probability, margin, cumulative, and normalization arithmetic is
    float32.  Target tensors affect only this offline objective and are never
    accepted by the online head's forward signature.
    """

    # Uniqueness is validated once by the CPU data loader.  Repeating that
    # value-dependent check here would synchronize every CUDA training step.
    validate_full16_lattice(candidate_ids, scores, require_unique=False)
    if gold_ids.shape != scores.shape[:2]:
        raise ValueError("gold_ids must have shape [B,16]")
    if target_candidate_logits.shape != scores.shape:
        raise ValueError("target_candidate_logits must have shape [B,16,16]")
    if target_matches_gold.shape != gold_ids.shape:
        raise ValueError("target_matches_gold must have shape [B,16]")
    if not math.isclose(float(temperature), TEACHER_TEMPERATURE):
        raise ValueError("JAPD freezes teacher temperature at 2")
    if not math.isclose(float(hard_target_mixture), HARD_TARGET_MIXTURE):
        raise ValueError("JAPD freezes the hard/soft mixture at 0.9/0.1")
    if not math.isclose(float(full_prefix_normalizer), FULL_PREFIX_NORMALIZER):
        raise ValueError("JAPD freezes the full16 prefix normalizer at 136")

    scores_float = scores.float()
    target_logits_float = target_candidate_logits.detach().float()
    ranks = candidate_gold_ranks(
        candidate_ids, gold_ids, require_unique=False
    )
    support, horizons = clean_support(ranks, target_matches_gold)
    safe_ranks = ranks.clamp(0, CANDIDATES - 1)

    soft_teacher = torch.softmax(
        target_logits_float / TEACHER_TEMPERATURE, dim=-1
    )
    hard_teacher = F.one_hot(
        safe_ranks, num_classes=CANDIDATES
    ).to(torch.float32)
    teacher = (
        HARD_TARGET_MIXTURE * hard_teacher
        + (1.0 - HARD_TARGET_MIXTURE) * soft_teacher
    )
    per_position_ce = -(
        teacher * torch.log_softmax(scores_float, dim=-1)
    ).sum(dim=-1)
    positions = torch.arange(
        BLOCK_LENGTH, device=scores.device, dtype=torch.float32
    ).unsqueeze(0)
    prefix_weights = (
        horizons.to(torch.float32).unsqueeze(-1) - positions
    ).clamp_min(0.0) * support.to(torch.float32)
    all_prefix = (
        per_position_ce * prefix_weights
    ).sum(dim=-1) / FULL_PREFIX_NORMALIZER

    base_errors, multi_repair, second_positions, joint_prefix = (
        _second_error_geometry(ranks, support)
    )
    wrong_scores = scores_float.scatter(
        -1,
        safe_ranks.unsqueeze(-1),
        torch.full_like(safe_ranks.unsqueeze(-1), float("-inf"), dtype=torch.float32),
    )
    target_scores = scores_float.gather(
        -1, safe_ranks.unsqueeze(-1)
    ).squeeze(-1)
    margins = target_scores - wrong_scores.amax(dim=-1)
    negative_margins = (-margins).masked_fill(
        ~joint_prefix, float("-inf")
    )
    joint_energy = torch.logsumexp(negative_margins, dim=-1)
    joint_loss = torch.where(
        multi_repair,
        F.softplus(joint_energy),
        torch.zeros_like(joint_energy),
    )
    per_block = all_prefix + joint_loss
    return JAPDPerBlockOutput(
        per_block_loss=per_block,
        all_prefix_loss=all_prefix,
        joint_two_frontier_loss=joint_loss,
        support_mask=support,
        horizons=horizons,
        gold_candidate_ranks=ranks,
        base_error_mask=base_errors,
        multi_repair_mask=multi_repair,
        second_error_positions=second_positions,
        joint_prefix_mask=joint_prefix,
        target_margins=margins,
    )


def fixed_prompt_balanced_batch_loss(
    per_block_loss: Tensor,
    effective_blocks_per_prompt: Tensor,
    *,
    total_effective_blocks: int,
    total_effective_prompts: int,
) -> Tensor:
    """Unbiased fixed-denominator minibatch estimator of prompt mean."""

    if per_block_loss.ndim != 1:
        raise ValueError("per_block_loss must have shape [B]")
    if effective_blocks_per_prompt.shape != per_block_loss.shape:
        raise ValueError("prompt block counts must have shape [B]")
    if total_effective_blocks < per_block_loss.numel():
        raise ValueError("total_effective_blocks is smaller than the batch")
    if total_effective_prompts < 1:
        raise ValueError("total_effective_prompts must be positive")
    counts = effective_blocks_per_prompt.to(
        device=per_block_loss.device, dtype=torch.float32
    )
    if counts.device.type == "cpu" and bool(counts.le(0).any().item()):
        raise ValueError("every sampled prompt must have an effective block")
    batch_size = int(per_block_loss.numel())
    scale = float(total_effective_blocks) / float(
        batch_size * total_effective_prompts
    )
    return scale * (per_block_loss / counts).sum()


def matched_candidate_dpace_per_block_loss(
    scores: Tensor,
    gold_candidate_ranks: Tensor,
    target_matches_gold: Tensor,
    *,
    alpha: float = 0.5,
) -> Tensor:
    """Candidate-D-PACE control on the same strict clean support.

    This returns one loss per block so the control uses the identical
    prompt-balanced sampling estimator as JAPD.  Only the objective differs.
    """

    if scores.ndim != 3 or scores.shape[1:] != (BLOCK_LENGTH, CANDIDATES):
        raise ValueError("scores must have shape [B,16,16]")
    if gold_candidate_ranks.shape != scores.shape[:2]:
        raise ValueError("gold ranks must have shape [B,16]")
    if target_matches_gold.shape != gold_candidate_ranks.shape:
        raise ValueError("target_matches_gold must have shape [B,16]")
    if not math.isclose(float(alpha), 0.5):
        raise ValueError("the matched Candidate-D-PACE control freezes alpha=0.5")
    support, _ = clean_support(gold_candidate_ranks, target_matches_gold)
    safe_ranks = gold_candidate_ranks.clamp(0, CANDIDATES - 1)
    log_probs = torch.log_softmax(scores.float(), dim=-1)
    gold_log_probs = log_probs.gather(
        -1, safe_ranks.unsqueeze(-1)
    ).squeeze(-1)
    gold_probabilities = gold_log_probs.exp()
    weights = exact_dpace_position_weights(
        gold_probabilities, support, alpha=alpha
    )
    return ((-gold_log_probs) * weights).sum(dim=-1) / float(BLOCK_LENGTH)


def strict_joint_two_frontier_metric(
    predicted_candidate_ranks: Tensor,
    gold_candidate_ranks: Tensor,
    target_matches_gold: Tensor,
) -> JointTwoFrontierMetric:
    """Check the entire inclusive prefix through the second base error."""

    if predicted_candidate_ranks.shape != gold_candidate_ranks.shape:
        raise ValueError("predicted and gold ranks must have equal shape")
    support, _ = clean_support(gold_candidate_ranks, target_matches_gold)
    _, multi_repair, second_positions, joint_prefix = _second_error_geometry(
        gold_candidate_ranks, support
    )
    correct = predicted_candidate_ranks.eq(gold_candidate_ranks)
    success = multi_repair & (correct | ~joint_prefix).all(dim=-1)
    return JointTwoFrontierMetric(
        eligible=multi_repair,
        success=success,
        second_error_positions=second_positions,
    )


def accepted_lengths(proposal_ids: Tensor, gold_ids: Tensor) -> Tensor:
    """Canonical accepted draft-token count in ``[0,16]``."""

    if proposal_ids.shape != gold_ids.shape:
        raise ValueError("proposal and gold IDs must have equal shape")
    if proposal_ids.ndim != 2 or proposal_ids.shape[1] != BLOCK_LENGTH:
        raise ValueError("proposal and gold IDs must have shape [B,16]")
    return proposal_ids.eq(gold_ids).to(torch.int64).cumprod(dim=-1).sum(dim=-1)
