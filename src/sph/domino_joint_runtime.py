"""Shared runtime pieces for joint Domino backbone adaptation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CanonicalBlock:
    sample_id: str
    domain: str
    context_ids: Tensor
    anchor_token_id: int
    gold_ids: Tensor
    anchor_offset: int


def select_even_prompt_blocks(
    records: Sequence[CanonicalBlock], count: int = 4
) -> tuple[CanonicalBlock, ...]:
    if count < 1 or len(records) < count:
        raise ValueError("a prompt does not contain enough canonical blocks")
    ordered = tuple(sorted(records, key=lambda record: record.anchor_offset))
    if count == 1:
        indices = (0,)
    else:
        indices = tuple(
            int(round(index * (len(ordered) - 1) / (count - 1)))
            for index in range(count)
        )
    if len(set(indices)) != count:
        raise RuntimeError("even prompt-block selection produced duplicate indices")
    selected = tuple(ordered[index] for index in indices)
    longest = max(selected, key=lambda record: int(record.context_ids.numel()))
    for record in selected:
        length = int(record.context_ids.numel())
        if not torch.equal(longest.context_ids[:length], record.context_ids):
            raise ValueError("canonical contexts for one prompt are not prefix nested")
    return selected


def dflash_positions_and_mask(
    context_lengths: Tensor,
    *,
    maximum_context: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Build per-row absolute positions and a padding-only DFlash mask."""

    lengths = context_lengths.to(device=device, dtype=torch.long)
    if lengths.ndim != 1 or lengths.numel() == 0:
        raise ValueError("context_lengths must be a nonempty vector")
    if bool(torch.any(lengths < 1)) or bool(torch.any(lengths > maximum_context)):
        raise ValueError("context length lies outside the padded target feature tensor")
    batch = int(lengths.numel())
    total = maximum_context + block_size
    positions = torch.zeros((batch, total), dtype=torch.long, device=device)
    context_axis = torch.arange(maximum_context, device=device)
    noise_axis = torch.arange(block_size, device=device)
    valid_context = context_axis[None, :] < lengths[:, None]
    positions[:, :maximum_context] = torch.where(
        valid_context,
        context_axis[None, :].expand(batch, -1),
        torch.zeros((), dtype=torch.long, device=device),
    )
    positions[:, maximum_context:] = lengths[:, None] + noise_axis[None, :]
    valid_keys = torch.cat(
        [
            valid_context,
            torch.ones((batch, block_size), dtype=torch.bool, device=device),
        ],
        dim=-1,
    )
    minimum = torch.finfo(dtype).min
    attention_mask = torch.where(
        valid_keys[:, None, None, :],
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), minimum, dtype=dtype, device=device),
    ).expand(batch, 1, block_size, total)
    return positions, attention_mask


def domino_prediction_hidden(
    domino: nn.Module, full_hidden: Tensor, *, horizon: int
) -> Tensor:
    """Apply the released Domino shift-label contract to backbone outputs."""

    dflash_config = getattr(domino.config, "dflash_config", {})
    if dflash_config.get("shift_label") is not True:
        raise ValueError("joint adaptation requires released Domino shift_label=true")
    if int(dflash_config.get("pure_draft_prefix_len", -1)) != 1:
        raise ValueError("joint adaptation requires pure_draft_prefix_len=1")
    if horizon not in {int(domino.block_size) - 1, int(domino.block_size)}:
        raise ValueError(
            "prediction horizon must be the legacy B-1 cache horizon or full B"
        )
    if full_hidden.shape[1] < horizon:
        raise ValueError("Domino backbone returned fewer states than the horizon")
    # Released same-anchor Domino consumes indices 0..14.  Unlike pure DFlash,
    # its shift-label checkpoint does not discard index zero.
    return full_hidden[:, :horizon, :]


def domino_teacher_logits(
    *,
    domino: nn.Module,
    target_weight: Tensor,
    anchors: Tensor,
    gold: Tensor,
    hidden: Tensor,
    train_causal_head: bool = False,
) -> Tensor:
    """Teacher-forced released Domino logits with gradient to ``hidden``."""

    final_logits, _ = domino_teacher_and_base_logits(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
        train_causal_head=train_causal_head,
    )
    return final_logits


def domino_teacher_and_base_logits(
    *,
    domino: nn.Module,
    target_weight: Tensor,
    anchors: Tensor,
    gold: Tensor,
    hidden: Tensor,
    train_causal_head: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return teacher-forced final logits and parallel-backbone base logits."""

    horizon = int(gold.shape[1])
    if hidden.shape[1] < horizon:
        raise ValueError("Domino hidden sequence is shorter than teacher horizon")
    # Preserve the released LM-head GEMM geometry when the caller supplies the
    # full 16 states, then supervise only its first 15 shift-label positions.
    base_logits = F.linear(hidden, target_weight.detach())[:, :horizon]
    prediction_hidden = hidden[:, :horizon]
    prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
    prefix_embeddings = F.embedding(prefix_ids, target_weight.detach())
    if train_causal_head:
        gru_out, _ = domino.prefix_gru(prefix_embeddings)
    else:
        with torch.no_grad():
            gru_out, _ = domino.prefix_gru(prefix_embeddings)
    prefix_states = gru_out[:, 1:]
    correction = domino.embed_proj(
        torch.cat([prediction_hidden[:, 1:], prefix_states], dim=-1)
    )
    final_logits = torch.cat(
        [
            base_logits[:, :1],
            released_domino_corrected_logits(base_logits[:, 1:], correction),
        ],
        dim=1,
    )
    return final_logits.float(), base_logits.float()


def greedy_reachable_joint_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    gold: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Accepted-prefix surrogate for joint backbone adaptation.

    Position zero always receives base CE so previously unreachable blocks can
    be repaired.  A later final-logit position receives CE exactly when every
    preceding teacher-forced greedy decision is correct.  The reachability mask
    is detached: it routes gradient but is not itself a soft objective.
    """

    if final_logits.ndim != 3 or final_logits.shape != base_logits.shape:
        raise ValueError("final/base logits must share shape [batch, positions, vocab]")
    if gold.shape != final_logits.shape[:2]:
        raise ValueError("gold must match the logit batch and position axes")
    batch = int(gold.shape[0])
    if batch < 1:
        raise ValueError("joint loss requires a nonempty batch")
    final_ce = F.cross_entropy(
        final_logits.reshape(-1, final_logits.shape[-1]),
        gold.reshape(-1),
        reduction="none",
    ).view_as(gold)
    base_zero = F.cross_entropy(
        base_logits[:, 0], gold[:, 0], reduction="sum"
    ) / float(batch)
    with torch.no_grad():
        correct = final_logits.argmax(dim=-1).eq(gold)
        reachable = torch.zeros_like(correct)
        if gold.shape[1] > 1:
            reachable[:, 1:] = correct[:, :-1].to(torch.long).cumprod(dim=-1).bool()
    reachable_suffix = (
        final_ce * reachable.to(dtype=final_ce.dtype)
    ).sum() / float(batch)
    total = base_zero + reachable_suffix
    return total, {
        "base_zero_ce": base_zero.detach(),
        "reachable_suffix_ce": reachable_suffix.detach(),
        "reachable_suffix_positions_per_block": reachable[:, 1:].sum().detach()
        / float(batch),
    }


def frontier_margin_joint_loss(
    final_logits: Tensor,
    gold: Tensor,
    *,
    repair_margin: float = 1e-4,
    protection_margin: float = 0.05,
    protection_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Repair only the current greedy frontier while protecting its prefix."""

    if final_logits.ndim != 3 or gold.shape != final_logits.shape[:2]:
        raise ValueError("expected logits [batch, positions, vocab] and matching gold")
    if repair_margin < 0 or protection_margin < 0 or protection_weight < 0:
        raise ValueError("frontier margin hyperparameters must be non-negative")
    batch, positions = gold.shape
    if batch < 1:
        raise ValueError("frontier loss requires a nonempty batch")
    logits = final_logits.float()
    gold_ids = gold.to(torch.long)
    with torch.no_grad():
        competitors = logits.detach().clone()
        competitors.scatter_(-1, gold_ids.unsqueeze(-1), -torch.inf)
        competitor_ids = competitors.argmax(dim=-1)
        predictions = logits.detach().argmax(dim=-1)
        axes = torch.arange(positions, device=gold.device).view(1, -1)
        sentinel = torch.full_like(axes, positions).expand_as(gold_ids)
        mismatch_axes = torch.where(
            predictions.ne(gold_ids), axes.expand_as(gold_ids), sentinel
        )
        frontier = mismatch_axes.min(dim=-1).values
        protected = axes < frontier[:, None]
        repair = axes == frontier[:, None]
    gold_scores = logits.gather(-1, gold_ids.unsqueeze(-1)).squeeze(-1)
    competitor_scores = logits.gather(
        -1, competitor_ids.unsqueeze(-1)
    ).squeeze(-1)
    margins = gold_scores - competitor_scores
    repair_loss = (
        torch.relu(repair_margin - margins) * repair.to(margins.dtype)
    ).sum() / float(batch)
    protection_loss = (
        torch.relu(protection_margin - margins) * protected.to(margins.dtype)
    ).sum() / float(batch)
    total = repair_loss + protection_weight * protection_loss
    return total, {
        "frontier_repair_hinge": repair_loss.detach(),
        "prefix_protection_hinge": protection_loss.detach(),
        "accepted_prefix_positions_per_block": protected.sum().detach()
        / float(batch),
        "repairable_blocks_per_batch": repair.any(dim=-1).sum().detach(),
    }


def union_topk_reachable_joint_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    gold: Tensor,
    *,
    topk: int = 16,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Listwise gold loss over the union of base and causal-head candidates.

    Supervision covers the current greedy-reachable prefix including its first
    miss.  The denominator contains each vocabulary item at most once, so
    overlap between the two Top-K lists cannot distort the listwise target.
    """

    if final_logits.ndim != 3 or final_logits.shape != base_logits.shape:
        raise ValueError("final/base logits must share shape [batch, positions, vocab]")
    if gold.shape != final_logits.shape[:2]:
        raise ValueError("gold must match the logit batch and position axes")
    if not 1 <= topk <= final_logits.shape[-1]:
        raise ValueError("topk is outside the vocabulary")
    logits = final_logits.float()
    with torch.no_grad():
        base_ids = base_logits.detach().float().topk(topk, dim=-1).indices
        final_ids = logits.detach().topk(topk, dim=-1).indices
        candidate_mask = torch.zeros_like(logits, dtype=torch.bool)
        candidate_mask.scatter_(-1, base_ids, True)
        candidate_mask.scatter_(-1, final_ids, True)
        covered = candidate_mask.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
        correct = logits.detach().argmax(dim=-1).eq(gold)
        reachable = torch.ones_like(correct)
        if gold.shape[1] > 1:
            reachable[:, 1:] = correct[:, :-1].to(torch.long).cumprod(dim=-1).bool()
        support = covered & reachable
    candidate_logits = logits.masked_fill(~candidate_mask, -torch.inf)
    per_position = torch.logsumexp(candidate_logits, dim=-1) - logits.gather(
        -1, gold.unsqueeze(-1)
    ).squeeze(-1)
    loss = (per_position * support.to(per_position.dtype)).sum() / float(
        gold.shape[0]
    )
    return loss, {
        "union_topk_reachable_loss": loss.detach(),
        "reachable_covered_positions_per_block": support.sum().detach()
        / float(gold.shape[0]),
        "union_coverage_fraction": covered.float().mean().detach(),
        "mean_unique_candidates": candidate_mask.sum(dim=-1).float().mean().detach(),
    }


def union_topk_oracle_prefix_joint_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    gold: Tensor,
    *,
    topk: int = 16,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Train every consecutively selectable token in the Top-K oracle prefix.

    Unlike current-policy reachability, this mask does not wait for an earlier
    greedy error to flip before supervising later positions.  A position is
    active exactly when gold belongs to the base/final Top-K union at that
    position and at every preceding position.  These are precisely the token
    decisions whose joint correction can increase single-path accepted length
    without inventing a token outside the available candidate lattice.
    """

    if final_logits.ndim != 3 or final_logits.shape != base_logits.shape:
        raise ValueError("final/base logits must share shape [batch, positions, vocab]")
    if gold.shape != final_logits.shape[:2]:
        raise ValueError("gold must match the logit batch and position axes")
    if not 1 <= topk <= final_logits.shape[-1]:
        raise ValueError("topk is outside the vocabulary")
    logits = final_logits.float()
    with torch.no_grad():
        base_ids = base_logits.detach().float().topk(topk, dim=-1).indices
        final_ids = logits.detach().topk(topk, dim=-1).indices
        candidate_mask = torch.zeros_like(logits, dtype=torch.bool)
        candidate_mask.scatter_(-1, base_ids, True)
        candidate_mask.scatter_(-1, final_ids, True)
        covered = candidate_mask.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
        oracle_prefix = covered.to(torch.long).cumprod(dim=-1).bool()
    candidate_logits = logits.masked_fill(~candidate_mask, -torch.inf)
    per_position = torch.logsumexp(candidate_logits, dim=-1) - logits.gather(
        -1, gold.unsqueeze(-1)
    ).squeeze(-1)
    loss = (per_position * oracle_prefix.to(per_position.dtype)).sum() / float(
        gold.shape[0]
    )
    return loss, {
        "union_topk_oracle_prefix_loss": loss.detach(),
        "oracle_prefix_positions_per_block": oracle_prefix.sum().detach()
        / float(gold.shape[0]),
        "union_coverage_fraction": covered.float().mean().detach(),
        "mean_unique_candidates": candidate_mask.sum(dim=-1).float().mean().detach(),
    }


def union_topk_frontier_protected_joint_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    gold: Tensor,
    *,
    topk: int = 16,
    protection_margin: float = 0.05,
    protection_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Repair the accepted-length frontier and protect its correct prefix.

    For each block, candidate CE starts at the current first greedy error and
    continues along the consecutively Top-K-covered gold path.  Positions that
    are already in the accepted prefix receive only a margin-preservation
    hinge, avoiding easy-token CE from overwhelming the boundary repair.
    """

    if final_logits.ndim != 3 or final_logits.shape != base_logits.shape:
        raise ValueError("final/base logits must share shape [batch, positions, vocab]")
    if gold.shape != final_logits.shape[:2]:
        raise ValueError("gold must match the logit batch and position axes")
    if not 1 <= topk <= final_logits.shape[-1]:
        raise ValueError("topk is outside the vocabulary")
    if protection_margin < 0 or protection_weight < 0:
        raise ValueError("frontier protection values must be non-negative")
    batch, positions = gold.shape
    logits = final_logits.float()
    with torch.no_grad():
        base_ids = base_logits.detach().float().topk(topk, dim=-1).indices
        final_ids = logits.detach().topk(topk, dim=-1).indices
        candidate_mask = torch.zeros_like(logits, dtype=torch.bool)
        candidate_mask.scatter_(-1, base_ids, True)
        candidate_mask.scatter_(-1, final_ids, True)
        covered = candidate_mask.gather(-1, gold.unsqueeze(-1)).squeeze(-1)

        predictions = logits.detach().argmax(dim=-1)
        axes = torch.arange(positions, device=gold.device).view(1, -1)
        sentinel = torch.full_like(axes, positions).expand_as(gold)
        mismatch_axes = torch.where(
            predictions.ne(gold), axes.expand_as(gold), sentinel
        )
        frontier = mismatch_axes.min(dim=-1).values
        protected = axes < frontier[:, None]
        at_or_after_frontier = axes >= frontier[:, None]
        missing_after_frontier = (
            (~covered) & at_or_after_frontier
        ).to(torch.long).cumsum(dim=-1)
        repair = (
            at_or_after_frontier
            & covered
            & missing_after_frontier.eq(0)
        )

        competitors = logits.detach().clone()
        competitors.scatter_(-1, gold.unsqueeze(-1), -torch.inf)
        competitor_ids = competitors.argmax(dim=-1)

    candidate_logits = logits.masked_fill(~candidate_mask, -torch.inf)
    candidate_ce = torch.logsumexp(candidate_logits, dim=-1) - logits.gather(
        -1, gold.unsqueeze(-1)
    ).squeeze(-1)
    repair_loss = (candidate_ce * repair.to(candidate_ce.dtype)).sum() / float(batch)

    gold_scores = logits.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    competitor_scores = logits.gather(
        -1, competitor_ids.unsqueeze(-1)
    ).squeeze(-1)
    margins = gold_scores - competitor_scores
    protection_loss = (
        torch.relu(protection_margin - margins) * protected.to(margins.dtype)
    ).sum() / float(batch)
    total = repair_loss + protection_weight * protection_loss
    return total, {
        "frontier_topk_repair_loss": repair_loss.detach(),
        "frontier_prefix_protection_loss": protection_loss.detach(),
        "accepted_prefix_positions_per_block": protected.sum().detach()
        / float(batch),
        "repair_positions_per_block": repair.sum().detach() / float(batch),
        "union_coverage_fraction": covered.float().mean().detach(),
        "mean_unique_candidates": candidate_mask.sum(dim=-1).float().mean().detach(),
    }


def target_distilled_union_joint_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    target_logits: Tensor,
    gold: Tensor,
    *,
    topk: int = 16,
    temperature: float = 2.0,
    protect_weight: float = 1.0,
    repair_weight: float = 4.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Distill verifier logits on the current deployable candidate union.

    The lattice is rebuilt from the live LoRA model on every forward pass.  It
    contains the DFlash Top-K list, with its final slot replaced by the current
    Domino action only when that action is absent.  Supervision is restricted
    to the current accepted prefix and its first rejection, where the
    teacher-forced prefix is exactly the online prefix.  Rows on which a fresh
    target replay disagrees with the canonical greedy token are masked instead
    of silently changing the acceptance label.

    Candidate construction is detached, but gradients flow through the live
    Domino scores gathered at the selected IDs.  Thus LoRA changes both the
    lattice on later updates and the scores within the current lattice.
    """

    if (
        final_logits.ndim != 3
        or final_logits.shape != base_logits.shape
        or final_logits.shape != target_logits.shape
    ):
        raise ValueError(
            "final/base/target logits must share shape [batch, positions, vocab]"
        )
    if gold.shape != final_logits.shape[:2]:
        raise ValueError("gold must match the logit batch and position axes")
    if not 2 <= topk <= final_logits.shape[-1]:
        raise ValueError("topk must lie in [2, vocabulary]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if min(protect_weight, repair_weight) < 0:
        raise ValueError("distillation position weights must be non-negative")

    batch, positions = gold.shape
    student = final_logits.float()
    target = target_logits.detach().float()
    with torch.no_grad():
        base_ids = base_logits.detach().float().topk(topk, dim=-1).indices
        current_ids = student.detach().argmax(dim=-1)
        candidate_ids = base_ids.clone()
        missing_current = ~candidate_ids.eq(
            current_ids.unsqueeze(-1)
        ).any(dim=-1)
        candidate_ids[..., -1] = torch.where(
            missing_current, current_ids, candidate_ids[..., -1]
        )

        target_ids = target.argmax(dim=-1)
        target_matches_gold = target_ids.eq(gold)
        gold_available = candidate_ids.eq(gold.unsqueeze(-1)).any(dim=-1)
        current_correct = current_ids.eq(gold)
        axes = torch.arange(positions, device=gold.device).view(1, -1)
        sentinel = torch.full_like(gold, positions)
        mismatch_axes = torch.where(
            current_correct, sentinel, axes.expand(batch, -1)
        )
        frontier = mismatch_axes.min(dim=-1).values
        protected = axes < frontier.unsqueeze(-1)
        repair = axes.eq(frontier.unsqueeze(-1)) & frontier.unsqueeze(-1).lt(
            positions
        )
        active = (protected | repair) & gold_available & target_matches_gold
        position_weights = (
            protect_weight * protected.float() + repair_weight * repair.float()
        ) * active.float()

    student_scores = student.gather(-1, candidate_ids)
    teacher_scores = target.gather(-1, candidate_ids)
    teacher_probabilities = torch.softmax(
        teacher_scores / temperature, dim=-1
    )
    teacher_log_probabilities = torch.log_softmax(
        teacher_scores / temperature, dim=-1
    )
    student_log_probabilities = torch.log_softmax(
        student_scores / temperature, dim=-1
    )
    per_position_kl = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1) * (temperature * temperature)
    per_block_denominator = position_weights.sum(dim=-1).clamp_min(1.0)
    per_block = (
        per_position_kl * position_weights
    ).sum(dim=-1) / per_block_denominator
    loss = per_block.mean()
    active_count = active.sum().clamp_min(1)
    return loss, {
        "target_union_kl": loss.detach(),
        "active_positions_per_block": active.sum().detach().float()
        / float(batch),
        "target_top1_gold_fraction": target_matches_gold.float().mean().detach(),
        "active_target_top1_gold_fraction": (
            (target_matches_gold & (protected | repair)).sum().detach().float()
            / (protected | repair).sum().clamp_min(1).detach().float()
        ),
        "gold_available_active_fraction": (
            (gold_available & (protected | repair)).sum().detach().float()
            / (protected | repair).sum().clamp_min(1).detach().float()
        ),
        "repairable_frontier_fraction": (
            (repair & gold_available & target_matches_gold).sum().detach().float()
            / repair.sum().clamp_min(1).detach().float()
        ),
        "mean_candidate_kl": (
            per_position_kl.detach() * active.float()
        ).sum()
        / active_count.detach().float(),
    }


def target_frontier_distilled_union_joint_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    target_logits: Tensor,
    gold: Tensor,
    *,
    topk: int = 16,
    temperature: float = 2.0,
    protection_margin: float = 0.05,
    protection_weight: float = 1.0,
    repair_weight: float = 4.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Target-KL only at the live frontier, with prefix safety hinges.

    Matching the target's dark distribution on an already-correct prefix can
    perturb harmless Domino decisions without increasing accepted length.
    This variant uses target information only for the first live rejection.
    Earlier positions receive a full-vocabulary gold-margin hinge, which is a
    direct identity/regret anchor and becomes zero once the prefix is safe.
    """

    if (
        final_logits.ndim != 3
        or final_logits.shape != base_logits.shape
        or final_logits.shape != target_logits.shape
    ):
        raise ValueError(
            "final/base/target logits must share shape [batch, positions, vocab]"
        )
    if gold.shape != final_logits.shape[:2]:
        raise ValueError("gold must match the logit batch and position axes")
    if not 2 <= topk <= final_logits.shape[-1]:
        raise ValueError("topk must lie in [2, vocabulary]")
    if temperature <= 0 or protection_margin < 0:
        raise ValueError("invalid target-frontier temperature/margin")
    if min(protection_weight, repair_weight) < 0:
        raise ValueError("target-frontier weights must be non-negative")

    batch, positions = gold.shape
    student = final_logits.float()
    target = target_logits.detach().float()
    with torch.no_grad():
        base_ids = base_logits.detach().float().topk(topk, dim=-1).indices
        current_ids = student.detach().argmax(dim=-1)
        candidate_ids = base_ids.clone()
        missing_current = ~candidate_ids.eq(
            current_ids.unsqueeze(-1)
        ).any(dim=-1)
        candidate_ids[..., -1] = torch.where(
            missing_current, current_ids, candidate_ids[..., -1]
        )
        target_matches_gold = target.argmax(dim=-1).eq(gold)
        gold_available = candidate_ids.eq(gold.unsqueeze(-1)).any(dim=-1)
        axes = torch.arange(positions, device=gold.device).view(1, -1)
        sentinel = torch.full_like(gold, positions)
        mismatch_axes = torch.where(
            current_ids.eq(gold), sentinel, axes.expand(batch, -1)
        )
        frontier = mismatch_axes.min(dim=-1).values
        protected = axes < frontier.unsqueeze(-1)
        repair = axes.eq(frontier.unsqueeze(-1)) & frontier.unsqueeze(-1).lt(
            positions
        )
        repair_active = repair & gold_available & target_matches_gold

        top_values, top_ids = student.detach().topk(2, dim=-1)
        competitor_ids = torch.where(
            top_ids[..., 0].eq(gold), top_ids[..., 1], top_ids[..., 0]
        )

    student_scores = student.gather(-1, candidate_ids)
    teacher_scores = target.gather(-1, candidate_ids)
    teacher_probabilities = torch.softmax(
        teacher_scores / temperature, dim=-1
    )
    teacher_log_probabilities = torch.log_softmax(
        teacher_scores / temperature, dim=-1
    )
    student_log_probabilities = torch.log_softmax(
        student_scores / temperature, dim=-1
    )
    per_position_kl = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1) * (temperature * temperature)
    repair_loss = (
        per_position_kl * repair_active.float()
    ).sum() / float(batch)

    gold_scores = student.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    competitor_scores = student.gather(
        -1, competitor_ids.unsqueeze(-1)
    ).squeeze(-1)
    protection_violations = torch.relu(
        protection_margin - (gold_scores - competitor_scores)
    )
    protection_loss = (
        protection_violations * protected.float()
    ).sum() / float(batch)
    total = repair_weight * repair_loss + protection_weight * protection_loss
    return total, {
        "target_frontier_kl": repair_loss.detach(),
        "prefix_protection_hinge": protection_loss.detach(),
        "accepted_prefix_positions_per_block": protected.sum().detach().float()
        / float(batch),
        "repairable_frontier_fraction": repair_active.sum().detach().float()
        / repair.sum().clamp_min(1).detach().float(),
        "target_top1_gold_fraction": target_matches_gold.float().mean().detach(),
        "frontier_gold_available_fraction": (
            (repair & gold_available).sum().detach().float()
            / repair.sum().clamp_min(1).detach().float()
        ),
        "protected_margin_violation_fraction": (
            (protection_violations.gt(0) & protected).sum().detach().float()
            / protected.sum().clamp_min(1).detach().float()
        ),
    }


def target_full_vocab_distilled_joint_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    target_logits: Tensor,
    gold: Tensor,
    *,
    topk: int = 16,
    temperature: float = 2.0,
    protection_margin: float = 0.05,
    protection_weight: float = 1.0,
    repair_weight: float = 4.0,
    future_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Distill the verifier over the full vocabulary after the live frontier.

    The union objectives cannot train a block whose verifier token is missing
    from the current DFlash Top-K lattice.  This stronger Stage-D falsifier
    removes that restriction while preserving the deployed model graph:

    * already accepted positions receive only a gold-margin safety hinge;
    * the current first rejection receives full-vocabulary target KL;
    * later positions receive a lower-weight full-vocabulary KL under their
      exact gold prefix, which is the online state once preceding repairs
      succeed.

    Target logits are supervision only.  Rows on which the replayed target
    top-1 disagrees with canonical gold are masked rather than relabelled.
    """

    if (
        final_logits.ndim != 3
        or final_logits.shape != base_logits.shape
        or final_logits.shape != target_logits.shape
    ):
        raise ValueError(
            "final/base/target logits must share shape [batch, positions, vocab]"
        )
    if gold.shape != final_logits.shape[:2]:
        raise ValueError("gold must match the logit batch and position axes")
    if not 1 <= topk <= final_logits.shape[-1]:
        raise ValueError("topk must lie in [1, vocabulary]")
    if temperature <= 0 or protection_margin < 0:
        raise ValueError("invalid full-vocabulary distillation temperature/margin")
    if min(protection_weight, repair_weight, future_weight) < 0:
        raise ValueError("full-vocabulary distillation weights must be non-negative")

    batch, positions = gold.shape
    student = final_logits.float()
    target = target_logits.detach().float()
    with torch.no_grad():
        current_ids = student.detach().argmax(dim=-1)
        target_matches_gold = target.argmax(dim=-1).eq(gold)
        axes = torch.arange(positions, device=gold.device).view(1, -1)
        sentinel = torch.full_like(gold, positions)
        mismatch_axes = torch.where(
            current_ids.eq(gold), sentinel, axes.expand(batch, -1)
        )
        frontier = mismatch_axes.min(dim=-1).values
        protected = axes < frontier.unsqueeze(-1)
        repair = axes.eq(frontier.unsqueeze(-1)) & frontier.unsqueeze(-1).lt(
            positions
        )
        future = axes > frontier.unsqueeze(-1)
        protection_active = protected & target_matches_gold
        repair_active = repair & target_matches_gold
        future_active = future & target_matches_gold

        base_ids = base_logits.detach().float().topk(topk, dim=-1).indices
        gold_available = base_ids.eq(gold.unsqueeze(-1)).any(dim=-1)
        _, top_ids = student.detach().topk(2, dim=-1)
        competitor_ids = torch.where(
            top_ids[..., 0].eq(gold), top_ids[..., 1], top_ids[..., 0]
        )

    def active_full_vocab_kl(mask: Tensor) -> Tensor:
        student_rows = student[mask]
        target_rows = target[mask]
        if student_rows.shape[0] == 0:
            return student_rows.sum(dim=-1)
        teacher_probabilities = torch.softmax(
            target_rows / temperature, dim=-1
        )
        teacher_log_probabilities = torch.log_softmax(
            target_rows / temperature, dim=-1
        )
        student_log_probabilities = torch.log_softmax(
            student_rows / temperature, dim=-1
        )
        return (
            teacher_probabilities
            * (teacher_log_probabilities - student_log_probabilities)
        ).sum(dim=-1) * (temperature * temperature)

    repair_kl = active_full_vocab_kl(repair_active)
    repair_loss = repair_kl.sum() / float(batch)

    future_kl = active_full_vocab_kl(future_active)
    future_indices = future_active.nonzero(as_tuple=False)
    if future_indices.shape[0] == 0:
        future_loss = student.sum() * 0.0
    else:
        future_sum = torch.zeros(
            batch, dtype=future_kl.dtype, device=future_kl.device
        ).scatter_add(0, future_indices[:, 0], future_kl)
        future_count = future_active.sum(dim=-1).clamp_min(1).to(future_kl.dtype)
        future_loss = (future_sum / future_count).sum() / float(batch)

    gold_scores = student.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    competitor_scores = student.gather(
        -1, competitor_ids.unsqueeze(-1)
    ).squeeze(-1)
    protection_violations = torch.relu(
        protection_margin - (gold_scores - competitor_scores)
    )
    protection_loss = (
        protection_violations * protection_active.float()
    ).sum() / float(batch)
    total = (
        repair_weight * repair_loss
        + future_weight * future_loss
        + protection_weight * protection_loss
    )
    return total, {
        "target_full_vocab_frontier_kl": repair_loss.detach(),
        "target_full_vocab_future_kl": future_loss.detach(),
        "prefix_protection_hinge": protection_loss.detach(),
        "accepted_prefix_positions_per_block": protected.sum().detach().float()
        / float(batch),
        "protected_target_positions_per_block": protection_active.sum()
        .detach()
        .float()
        / float(batch),
        "repair_target_positions_per_block": repair_active.sum().detach().float()
        / float(batch),
        "future_target_positions_per_block": future_active.sum().detach().float()
        / float(batch),
        "target_top1_gold_fraction": target_matches_gold.float().mean().detach(),
        "frontier_gold_in_base_topk_fraction": (
            (repair & gold_available).sum().detach().float()
            / repair.sum().clamp_min(1).detach().float()
        ),
        "protected_margin_violation_fraction": (
            (protection_violations.gt(0) & protection_active)
            .sum()
            .detach()
            .float()
            / protection_active.sum().clamp_min(1).detach().float()
        ),
    }


def released_domino_corrected_logits(base_logits: Tensor, correction: Tensor) -> Tensor:
    """Reproduce released Domino's pre-argmax arithmetic dtype exactly."""

    if base_logits.dtype != correction.dtype:
        raise ValueError("released Domino base and correction dtypes must match")
    return base_logits + correction


@torch.no_grad()
def domino_onpolicy_ids(
    *,
    domino: nn.Module,
    target_weight: Tensor,
    anchors: Tensor,
    hidden: Tensor,
) -> Tensor:
    base_logits = F.linear(hidden, target_weight)
    batch, positions = hidden.shape[:2]
    proposals = torch.empty((batch, positions), dtype=torch.long, device=hidden.device)
    first = base_logits[:, :1].argmax(dim=-1)
    proposals[:, 0] = first[:, 0]
    prefix_ids = torch.cat([anchors[:, None], first], dim=-1)
    _, state = domino.prefix_gru(F.embedding(prefix_ids, target_weight))
    for position in range(1, positions):
        correction = domino.embed_proj(
            torch.cat(
                [hidden[:, position : position + 1], state.transpose(0, 1)],
                dim=-1,
            )
        )
        token = released_domino_corrected_logits(
            base_logits[:, position : position + 1], correction
        ).argmax(dim=-1)
        proposals[:, position] = token[:, 0]
        if position + 1 < positions:
            _, state = domino.prefix_gru(F.embedding(token, target_weight), state)
    return proposals


def acceptance_lengths(proposals: Tensor, gold: Tensor) -> Tensor:
    return proposals.eq(gold).to(torch.long).cumprod(dim=-1).sum(dim=-1)


def summarize_prompt_balanced_lengths(
    sample_ids: Sequence[str],
    domains: Sequence[str],
    lengths: Sequence[int],
    *,
    horizon: int,
) -> dict[str, Any]:
    if not (len(sample_ids) == len(domains) == len(lengths)):
        raise ValueError("metric inputs differ in length")

    def summarize(indices: Sequence[int]) -> dict[str, float | int]:
        grouped: dict[str, list[int]] = defaultdict(list)
        raw: list[int] = []
        for index in indices:
            value = int(lengths[index])
            grouped[str(sample_ids[index])].append(value)
            raw.append(value)
        prompt_values = [sum(values) / len(values) for values in grouped.values()]
        return {
            "blocks": len(raw),
            "prompts": len(prompt_values),
            "mean_accepted_draft_tokens_prompt_balanced": sum(prompt_values)
            / len(prompt_values),
            "mean_accepted_draft_tokens_round_weighted": sum(raw) / len(raw),
            "first_token_acceptance": sum(value >= 1 for value in raw) / len(raw),
            "full_horizon_rate": sum(value == horizon for value in raw) / len(raw),
        }

    all_indices = list(range(len(lengths)))
    result: dict[str, Any] = {"overall": summarize(all_indices), "by_domain": {}}
    for domain in sorted(set(domains)):
        indices = [index for index, value in enumerate(domains) if value == domain]
        result["by_domain"][domain] = summarize(indices)
    return result


__all__ = [
    "CanonicalBlock",
    "acceptance_lengths",
    "dflash_positions_and_mask",
    "domino_onpolicy_ids",
    "domino_prediction_hidden",
    "domino_teacher_and_base_logits",
    "domino_teacher_logits",
    "frontier_margin_joint_loss",
    "greedy_reachable_joint_loss",
    "released_domino_corrected_logits",
    "select_even_prompt_blocks",
    "summarize_prompt_balanced_lengths",
    "target_distilled_union_joint_loss",
    "target_frontier_distilled_union_joint_loss",
    "target_full_vocab_distilled_joint_loss",
    "union_topk_reachable_joint_loss",
    "union_topk_oracle_prefix_joint_loss",
    "union_topk_frontier_protected_joint_loss",
]
