"""Loss and exact same-set threshold evaluation for R048-B capacity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .gfpr import accepted_lengths


@dataclass(frozen=True)
class R048CapacityLossOutput:
    loss: Tensor
    kl_loss: Tensor
    keep_loss: Tensor
    active_rows: Tensor
    protected_rows: Tensor
    frontier_rows: Tensor


def r048_capacity_loss(
    *,
    base_scores: Tensor,
    lens_delta: Tensor,
    candidate_ids: Tensor,
    proposal: Tensor,
    target_candidate_logits: Tensor,
    valid_teacher_mask: Tensor,
    accepted: Tensor,
    oracle_accepted: Tensor,
    temperature: float = 2.0,
    frontier_weight: float = 4.0,
    keep_weight: float = 1.0,
    keep_margin: float = 0.05,
) -> R048CapacityLossOutput:
    """Centered candidate KL plus potential-loss-weighted KEEP protection."""

    if base_scores.shape != candidate_ids.shape or lens_delta.shape != candidate_ids.shape:
        raise ValueError("candidate score tensors differ in shape")
    if proposal.shape != candidate_ids.shape[:2]:
        raise ValueError("proposal differs from candidate lattice")
    if target_candidate_logits.shape != candidate_ids.shape:
        raise ValueError("target candidate logits differ in shape")
    if valid_teacher_mask.shape != proposal.shape:
        raise ValueError("teacher mask differs from proposal horizon")
    batch, positions = proposal.shape
    if accepted.shape != (batch,) or oracle_accepted.shape != (batch,):
        raise ValueError("accepted lengths must have shape [batch]")
    if temperature <= 0 or min(frontier_weight, keep_weight, keep_margin) < 0:
        raise ValueError("invalid R048 loss hyperparameters")

    adjusted = base_scores.float() + lens_delta.float()
    proposal_matches = candidate_ids.eq(proposal.unsqueeze(-1))
    proposal_index = proposal_matches.to(torch.long).argmax(dim=-1)
    student_proposal = adjusted.gather(-1, proposal_index.unsqueeze(-1))
    target_proposal = target_candidate_logits.float().gather(
        -1, proposal_index.unsqueeze(-1)
    )
    student_centered = adjusted - student_proposal
    target_centered = target_candidate_logits.float() - target_proposal
    teacher_prob = F.softmax(target_centered / temperature, dim=-1)
    student_log_prob = F.log_softmax(student_centered / temperature, dim=-1)
    teacher_log_prob = F.log_softmax(target_centered / temperature, dim=-1)
    per_row_kl = (
        teacher_prob * (teacher_log_prob - student_log_prob)
    ).sum(dim=-1) * (temperature * temperature)

    axes = torch.arange(positions, device=proposal.device).view(1, -1)
    protected = axes.lt(accepted[:, None])
    frontier = axes.eq(accepted[:, None]) & accepted[:, None].lt(positions)
    potential_loss = (accepted[:, None] - axes).clamp_min(0).float()
    frontier_reward = (oracle_accepted - accepted).clamp_min(0).float()[:, None]
    row_weights = (
        protected.float() * potential_loss / float(positions)
        + frontier.float() * frontier_reward * frontier_weight
    )
    active = valid_teacher_mask & (protected | frontier) & row_weights.gt(0)
    weighted_kl = per_row_kl * row_weights * active.float()
    kl_denominator = (row_weights * active.float()).sum().clamp_min(1.0)
    kl_loss = weighted_kl.sum() / kl_denominator

    competitor = adjusted.masked_fill(proposal_matches, -torch.inf).max(dim=-1).values
    proposal_score = student_proposal.squeeze(-1)
    keep_violation = F.relu(keep_margin - (proposal_score - competitor))
    # If the clean verifier's frontier token is outside the fixed support, the
    # one-repair oracle cannot improve this row.  It is therefore a KEEP row,
    # not an unlabelled row on which the head may invent an arbitrary rewrite.
    unrepairable_frontier = frontier & oracle_accepted[:, None].eq(accepted[:, None])
    keep_weights = (
        potential_loss * protected.float() + unrepairable_frontier.float()
    ) * valid_teacher_mask.float()
    keep_denominator = keep_weights.sum().clamp_min(1.0)
    keep_loss = (keep_violation * keep_weights).sum() / keep_denominator
    loss = kl_loss + keep_weight * keep_loss
    return R048CapacityLossOutput(
        loss=loss,
        kl_loss=kl_loss.detach(),
        keep_loss=keep_loss.detach(),
        active_rows=active.sum().detach(),
        protected_rows=protected.sum().detach(),
        frontier_rows=frontier.sum().detach(),
    )


def _prompt_balanced(sample_ids: Sequence[str], values: Sequence[int]) -> float:
    grouped: dict[str, list[int]] = defaultdict(list)
    for sample_id, value in zip(sample_ids, values, strict=True):
        grouped[str(sample_id)].append(int(value))
    return sum(sum(group) / len(group) for group in grouped.values()) / len(grouped)


def decode_earliest_threshold(
    *,
    proposal: Tensor,
    candidate_ids: Tensor,
    adjusted_scores: Tensor,
    threshold: float,
) -> Tensor:
    """CPU/GPU exact earliest-one decode for one scalar threshold."""

    proposal_matches = candidate_ids.eq(proposal.unsqueeze(-1))
    proposal_index = proposal_matches.to(torch.long).argmax(dim=-1)
    score = adjusted_scores.float()
    proposal_score = score.gather(-1, proposal_index.unsqueeze(-1)).squeeze(-1)
    best_score, best_index = score.max(dim=-1)
    best_token = candidate_ids.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    margin = best_score - proposal_score
    eligible = best_token.ne(proposal) & best_score.gt(proposal_score) & margin.gt(threshold)
    batch, positions = proposal.shape
    axes = torch.arange(positions, device=proposal.device).view(1, -1)
    sentinel = torch.full_like(axes.expand(batch, -1), positions)
    selected = torch.where(eligible, axes.expand(batch, -1), sentinel).min(dim=-1).values
    safe = selected.clamp_max(positions - 1)
    token = best_token.gather(1, safe[:, None])[:, 0]
    return torch.where(axes.eq(selected[:, None]), token[:, None], proposal)


def exact_earliest_one_lengths(
    *,
    proposal: Tensor,
    decoded: Tensor,
    verifier_top1: Tensor,
    baseline_lengths: Tensor,
    oracle_lengths: Tensor,
) -> Tensor:
    """Score one earliest-only rewrite against the clean verifier exactly.

    A rewrite before the verifier frontier is immediately harmful.  A rewrite
    after it cannot change the already-observed first rejection.  At the
    frontier, only the clean verifier's top-1 token can extend acceptance, in
    which case ``oracle_lengths`` was measured by rerunning the full unsplit
    verifier on that repaired sequence.
    """

    if proposal.ndim != 2 or decoded.shape != proposal.shape:
        raise ValueError("proposal/decoded must share [batch, positions]")
    if verifier_top1.shape != proposal.shape:
        raise ValueError("verifier top1 differs from proposal horizon")
    batch, positions = proposal.shape
    if baseline_lengths.shape != (batch,) or oracle_lengths.shape != (batch,):
        raise ValueError("accepted lengths must have shape [batch]")
    changed = decoded.ne(proposal)
    if bool(changed.sum(dim=-1).gt(1).any().item()):
        raise ValueError("exact R048 scorer accepts at most one rewrite")
    axes = torch.arange(positions, device=proposal.device).view(1, -1)
    sentinel = torch.full_like(axes.expand(batch, -1), positions)
    selected = torch.where(changed, axes.expand(batch, -1), sentinel).min(dim=-1).values
    safe_selected = selected.clamp_max(positions - 1)
    selected_token = decoded.gather(1, safe_selected[:, None])[:, 0]
    safe_frontier = baseline_lengths.clamp_max(positions - 1)
    frontier_token = verifier_top1.gather(1, safe_frontier[:, None])[:, 0]

    lengths = baseline_lengths.clone()
    lengths = torch.where(selected.lt(baseline_lengths), selected, lengths)
    correct_frontier = (
        baseline_lengths.lt(positions)
        & selected.eq(baseline_lengths)
        & selected_token.eq(frontier_token)
    )
    return torch.where(correct_frontier, oracle_lengths, lengths)


def select_zero_harm_threshold(
    *,
    sample_ids: Sequence[str],
    proposal: Tensor,
    verifier_top1: Tensor,
    candidate_ids: Tensor,
    adjusted_scores: Tensor,
    baseline_lengths: Tensor,
    oracle_lengths: Tensor,
) -> dict[str, float | int | list[int]]:
    """Exactly sweep every decision transition under a zero-harm constraint."""

    if proposal.ndim != 2 or verifier_top1.shape != proposal.shape:
        raise ValueError("proposal/verifier top1 shape mismatch")
    batch, positions = proposal.shape
    if len(sample_ids) != batch:
        raise ValueError("sample IDs differ from block count")
    if baseline_lengths.shape != (batch,) or oracle_lengths.shape != (batch,):
        raise ValueError("accepted lengths must have shape [batch]")
    measured_baseline = accepted_lengths(proposal, verifier_top1)
    if not torch.equal(measured_baseline.cpu(), baseline_lengths.cpu()):
        raise ValueError("stored baseline is not authoritative-verifier exact")

    # Capacity evaluation is small and deterministic on CPU.  A per-block
    # transition sweep avoids quantile-subsampling thresholds while remaining
    # O(blocks * positions^2), rather than repeatedly reducing the K lattice.
    proposal = proposal.detach().cpu()
    verifier_top1 = verifier_top1.detach().cpu()
    candidate_ids = candidate_ids.detach().cpu()
    score = adjusted_scores.detach().float().cpu()
    baseline = baseline_lengths.detach().long().cpu()
    oracle_lengths = oracle_lengths.detach().long().cpu()
    proposal_matches = candidate_ids.eq(proposal.unsqueeze(-1))
    proposal_index = proposal_matches.to(torch.long).argmax(dim=-1)
    current = score.gather(-1, proposal_index.unsqueeze(-1)).squeeze(-1)
    best_score, best_index = score.max(dim=-1)
    best_token = candidate_ids.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    margins = best_score - current
    eligible_candidate = best_token.ne(proposal) & best_score.gt(current)
    all_thresholds = sorted(
        float(value)
        for value in torch.unique(margins[eligible_candidate]).tolist()
        if float(value) > 0.0
    )

    def decision_at(block: int, threshold: float) -> tuple[int, int]:
        eligible = eligible_candidate[block] & margins[block].gt(threshold)
        indices = eligible.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            return positions, int(proposal[block, -1])
        selected = int(indices[0])
        return selected, int(best_token[block, selected])

    def outcome_at(block: int, selected: int, token: int) -> int:
        frontier = int(baseline[block])
        if selected < frontier:
            return selected
        if (
            frontier < positions
            and selected == frontier
            and token == int(verifier_top1[block, frontier])
        ):
            return int(oracle_lengths[block])
        return frontier

    # Events only occur when the currently earliest eligible row disappears.
    # We still visit every unique margin below so the safety-favouring higher
    # threshold tie-break is exact, even across intervals with no event.
    events: dict[float, list[tuple[int, int, int, int]]] = defaultdict(list)
    selected_state: list[int] = []
    token_state: list[int] = []
    length_state: list[int] = []
    for block in range(batch):
        selected, token = decision_at(block, 0.0)
        selected_state.append(selected)
        token_state.append(token)
        length_state.append(outcome_at(block, selected, token))
        previous = selected
        block_thresholds = sorted(
            float(value)
            for value in torch.unique(margins[block][eligible_candidate[block]]).tolist()
            if float(value) > 0.0
        )
        for threshold in block_thresholds:
            new_selected, new_token = decision_at(block, threshold)
            if new_selected != previous:
                events[threshold].append(
                    (
                        block,
                        new_selected,
                        new_token,
                        outcome_at(block, new_selected, new_token),
                    )
                )
                previous = new_selected

    baseline_values = [int(value) for value in baseline]
    baseline_eal = _prompt_balanced(sample_ids, baseline_values)
    oracle_eal = _prompt_balanced(sample_ids, [int(value) for value in oracle_lengths])
    prompt_counts: dict[str, int] = defaultdict(int)
    for sample_id in sample_ids:
        prompt_counts[str(sample_id)] += 1
    prompt_total = len(prompt_counts)
    weights = [1.0 / (prompt_total * prompt_counts[str(sid)]) for sid in sample_ids]

    current_eal = sum(
        weight * length
        for weight, length in zip(weights, length_state, strict=True)
    )
    current_harmful = sum(
        length < base
        for length, base in zip(length_state, baseline_values, strict=True)
    )
    current_changed = sum(selected < positions for selected in selected_state)
    current_gained = sum(
        max(length - base, 0)
        for length, base in zip(length_state, baseline_values, strict=True)
    )
    current_lost = sum(
        max(base - length, 0)
        for length, base in zip(length_state, baseline_values, strict=True)
    )

    best_summary: tuple[float, float, int, int, int, int] | None = None

    def consider(threshold: float) -> None:
        nonlocal best_summary
        if current_harmful != 0:
            return
        if (
            best_summary is None
            or current_eal > best_summary[1] + 1e-12
            or (
                abs(current_eal - best_summary[1]) <= 1e-12
                and threshold > best_summary[0]
            )
        ):
            best_summary = (
                threshold,
                current_eal,
                current_harmful,
                current_changed,
                current_gained,
                current_lost,
            )

    consider(0.0)
    for threshold in all_thresholds:
        for block, selected, token, length in events.get(threshold, []):
            old_selected = selected_state[block]
            old_length = length_state[block]
            base = baseline_values[block]
            current_eal += weights[block] * (length - old_length)
            current_harmful += int(length < base) - int(old_length < base)
            current_changed += int(selected < positions) - int(old_selected < positions)
            current_gained += max(length - base, 0) - max(old_length - base, 0)
            current_lost += max(base - length, 0) - max(base - old_length, 0)
            selected_state[block] = selected
            token_state[block] = token
            length_state[block] = length
        consider(threshold)
    final_threshold = (all_thresholds[-1] + 1.0) if all_thresholds else 1.0
    consider(final_threshold)
    if best_summary is None:
        raise RuntimeError("KEEP threshold unexpectedly failed zero-harm constraint")

    threshold, _, _, _, _, _ = best_summary
    decoded = decode_earliest_threshold(
        proposal=proposal,
        candidate_ids=candidate_ids,
        adjusted_scores=score,
        threshold=threshold,
    )
    exact_lengths = exact_earliest_one_lengths(
        proposal=proposal,
        decoded=decoded,
        verifier_top1=verifier_top1,
        baseline_lengths=baseline,
        oracle_lengths=oracle_lengths,
    )
    values = [int(value) for value in exact_lengths]
    best: dict[str, float | int | list[int]] = {
        "threshold": threshold,
        "eal_prompt_balanced": _prompt_balanced(sample_ids, values),
        "harmful_blocks": int(exact_lengths.lt(baseline).sum()),
        "changed_blocks": int(decoded.ne(proposal).any(dim=-1).sum()),
        "gained_tokens": int((exact_lengths - baseline).clamp_min(0).sum()),
        "lost_tokens": int((baseline - exact_lengths).clamp_min(0).sum()),
        "lengths": values,
    }
    oracle_gain = oracle_eal - baseline_eal
    recovery = (
        (float(best["eal_prompt_balanced"]) - baseline_eal) / oracle_gain
        if oracle_gain > 0
        else 1.0
    )
    best.update(
        {
            "baseline_eal_prompt_balanced": baseline_eal,
            "oracle_eal_prompt_balanced": oracle_eal,
            "oracle_gain": oracle_gain,
            "oracle_gain_recovery": recovery,
            "thresholds_evaluated": len(all_thresholds) + 2,
        }
    )
    return best


__all__ = [
    "R048CapacityLossOutput",
    "decode_earliest_threshold",
    "exact_earliest_one_lengths",
    "r048_capacity_loss",
    "select_zero_harm_threshold",
]
