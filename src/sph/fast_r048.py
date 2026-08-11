"""Fast-K proposal and exact earliest-frontier repair primitives for R048.

The proposal keeps DFlash's parallel vocabulary projection, but restricts the
Domino correction to the base Top-K rows.  This removes the sequential
full-vocabulary projection from the proposal head.  The repair oracle changes
only the first currently rejected token, which is the only decision whose
early-target state is guaranteed to lie on the verified prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .gfpr import accepted_lengths


@dataclass(frozen=True)
class FastCandidateDecodeOutput:
    """One causal candidate-only Domino proposal."""

    token_ids: Tensor
    candidate_ids: Tensor
    candidate_base_logits: Tensor
    candidate_scores: Tensor


@dataclass(frozen=True)
class FrontierRepairOutput:
    """Result of one exact, earliest-only candidate-constrained repair."""

    token_ids: Tensor
    accepted_before: Tensor
    accepted_after: Tensor
    frontier: Tensor
    repair_available: Tensor
    changed: Tensor


@dataclass(frozen=True)
class EarliestOneDecisionOutput:
    """One confidence-gated learned repair, with horizon as KEEP sentinel."""

    token_ids: Tensor
    selected_position: Tensor
    position_margins: Tensor
    adjusted_candidate_scores: Tensor


def _state_for_head(domino: Any, state: Tensor) -> Tensor:
    state_for_head = state.transpose(0, 1)
    if bool(getattr(domino, "use_bias_norm", False)):
        state_for_head = domino.bias_norm(state_for_head)
    return state_for_head


@torch.no_grad()
def fast_candidate_domino_decode(
    *,
    domino: Any,
    target_weight: Tensor,
    anchors: Tensor,
    hidden: Tensor,
    candidate_topk: int = 32,
    forced_first: Tensor | None = None,
    forced_prefix: Tensor | None = None,
) -> FastCandidateDecodeOutput:
    """Decode over base Top-K using gathered frozen Domino correction rows.

    Position zero remains the DFlash argmax, matching released Domino.  Later
    positions consume the selected token through the frozen prefix GRU, form a
    256-wide correction code, and score only the K gathered output rows.  The
    base vocabulary projection is executed once for the full block.
    """

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, positions, width]")
    batch, positions, width = hidden.shape
    if anchors.shape != (batch,):
        raise ValueError("anchors must have shape [batch]")
    if target_weight.ndim != 2 or target_weight.shape[1] != width:
        raise ValueError("target embedding shape is incompatible with hidden")
    if not 2 <= candidate_topk <= target_weight.shape[0]:
        raise ValueError("candidate_topk lies outside the vocabulary")
    if positions < 1:
        raise ValueError("proposal horizon must be positive")

    base_logits = F.linear(hidden, target_weight)
    return fast_candidate_domino_decode_from_base(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        base_logits=base_logits,
        candidate_topk=candidate_topk,
        forced_first=forced_first,
        forced_prefix=forced_prefix,
    )


@torch.no_grad()
def fast_candidate_domino_decode_from_base(
    *,
    domino: Any,
    target_weight: Tensor,
    anchors: Tensor,
    hidden: Tensor,
    base_logits: Tensor,
    candidate_topk: int = 32,
    forced_first: Tensor | None = None,
    forced_prefix: Tensor | None = None,
) -> FastCandidateDecodeOutput:
    """Decode Fast-K when the shared DFlash base GEMM is already available.

    ``forced_prefix`` seeds one or more exact proposal tokens into the frozen
    Domino GRU before candidate-only decoding resumes.  ``forced_first`` is
    retained as the graph-compatible one-token API used by R050.
    """

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, positions, width]")
    batch, positions, width = hidden.shape
    if anchors.shape != (batch,):
        raise ValueError("anchors must have shape [batch]")
    if target_weight.ndim != 2 or target_weight.shape[1] != width:
        raise ValueError("target embedding shape is incompatible with hidden")
    if base_logits.shape != (batch, positions, target_weight.shape[0]):
        raise ValueError("base logits do not match hidden and vocabulary")
    if not 2 <= candidate_topk <= target_weight.shape[0]:
        raise ValueError("candidate_topk lies outside the vocabulary")
    if positions < 1:
        raise ValueError("proposal horizon must be positive")
    if forced_first is not None and forced_prefix is not None:
        raise ValueError("forced_first and forced_prefix are mutually exclusive")
    if forced_first is not None:
        if (
            forced_first.shape != (batch,)
            or forced_first.dtype != torch.long
            or forced_first.device != hidden.device
        ):
            raise ValueError(
                "forced position-zero token must be device-local int64 [batch]"
            )
        forced_prefix = forced_first[:, None]
    if forced_prefix is not None and (
        forced_prefix.ndim != 2
        or forced_prefix.shape[0] != batch
        or not 1 <= forced_prefix.shape[1] <= positions
        or forced_prefix.dtype != torch.long
        or forced_prefix.device != hidden.device
    ):
        raise ValueError(
            "forced prefix must be device-local int64 [batch, 1..positions]"
        )

    # Match the canonical DFlash lattice: rank candidate IDs after promotion
    # to FP32, then gather the original BF16 values so the correction add keeps
    # the serving dtype.  Ranking the BF16 tensor directly can choose a
    # different member of a near-tie at the K boundary.
    candidate_ids = base_logits.float().topk(candidate_topk, dim=-1).indices
    candidate_base_logits = base_logits.gather(-1, candidate_ids)

    # Position zero is the base-vocabulary argmax, not candidate_topk[..., 0].
    # CUDA Top-K is allowed to choose a different member/order for exact BF16
    # ties as K changes, while argmax is the released policy contract.  Exact
    # target seeds are inserted into each corresponding fixed-size support so
    # downstream diagnostics retain the proposal-support invariant.
    if forced_prefix is None:
        forced_prefix = base_logits[:, :1].float().argmax(dim=-1)
    forced_positions = forced_prefix.shape[1]
    for position in range(forced_positions):
        forced = forced_prefix[:, position]
        present = candidate_ids[:, position].eq(forced[:, None]).any(dim=-1)
        forced_score = base_logits[:, position].gather(1, forced[:, None])[:, 0]
        candidate_ids[:, position, -1] = torch.where(
            present, candidate_ids[:, position, -1], forced
        )
        candidate_base_logits[:, position, -1] = torch.where(
            present, candidate_base_logits[:, position, -1], forced_score
        )

    selected = list(forced_prefix.unbind(dim=1))
    score_rows = [
        candidate_base_logits[:, position : position + 1]
        for position in range(forced_positions)
    ]
    prefix_ids = torch.cat([anchors[:, None], forced_prefix], dim=1)
    _, state = domino.prefix_gru(F.embedding(prefix_ids, target_weight))

    output_basis = domino.embed_proj[2].weight
    for position in range(forced_positions, positions):
        joined = torch.cat(
            [hidden[:, position : position + 1], _state_for_head(domino, state)],
            dim=-1,
        )
        code = domino.embed_proj[1](domino.embed_proj[0](joined))[:, 0]
        basis = F.embedding(candidate_ids[:, position], output_basis)
        correction = torch.einsum("bd,bkd->bk", code, basis)
        # Keep the checkpoint dtype for the add, matching released Domino's
        # BF16 score arithmetic before promotion for selection.
        scores = candidate_base_logits[:, position] + correction
        best = scores.float().argmax(dim=-1)
        token = candidate_ids[:, position].gather(1, best[:, None])[:, 0]
        selected.append(token)
        score_rows.append(scores[:, None])
        if position + 1 < positions:
            _, state = domino.prefix_gru(
                F.embedding(token[:, None], target_weight), state
            )

    return FastCandidateDecodeOutput(
        token_ids=torch.stack(selected, dim=-1),
        candidate_ids=candidate_ids,
        candidate_base_logits=candidate_base_logits,
        candidate_scores=torch.cat(score_rows, dim=1),
    )


class R048TunedLens(torch.nn.Module):
    """The fixed 180,224-parameter target-only R048 correction lens."""

    def __init__(
        self,
        *,
        hidden_width: int,
        rank: int,
        candidate_basis: Tensor,
        rms_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_width < 1 or rank < 1 or candidate_basis.ndim != 2:
            raise ValueError("invalid tuned-lens dimensions")
        if rms_epsilon <= 0:
            raise ValueError("rms epsilon must be positive")
        code_width = int(candidate_basis.shape[1])
        self.hidden_width = int(hidden_width)
        self.rank = int(rank)
        self.code_width = code_width
        self.rms_epsilon = float(rms_epsilon)
        self.register_buffer(
            "candidate_basis", candidate_basis.detach(), persistent=False
        )
        self.down = torch.nn.Linear(
            hidden_width,
            rank,
            bias=False,
            device=candidate_basis.device,
            dtype=candidate_basis.dtype,
        )
        self.up = torch.nn.Linear(
            rank,
            code_width,
            bias=False,
            device=candidate_basis.device,
            dtype=candidate_basis.dtype,
        )
        torch.nn.init.zeros_(self.up.weight)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, early_states: Tensor, candidate_ids: Tensor) -> Tensor:
        if early_states.ndim != 3 or early_states.shape[-1] != self.hidden_width:
            raise ValueError("early target states have an incompatible shape")
        if candidate_ids.shape[:2] != early_states.shape[:2]:
            raise ValueError("candidate lattice does not match early states")
        input_dtype = early_states.dtype
        normalized = early_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.rms_epsilon
        )
        code = self.up(F.silu(self.down(normalized.to(input_dtype))))
        basis = F.embedding(candidate_ids, self.candidate_basis)
        return torch.einsum("blc,blkc->blk", code, basis)


def candidate_union_with_proposal(
    base_topk_ids: Tensor,
    proposal: Tensor,
    *,
    support_size: int,
) -> Tensor:
    """Return a fixed-size Top-K support that always retains the proposal.

    If the proposal is outside the first ``support_size`` base candidates, it
    replaces the last row.  Fast-K proposals are selected from this support,
    so this is normally the unchanged base Top-K tensor; the replacement rule
    makes the runtime safety contract explicit.
    """

    if base_topk_ids.ndim != 3 or proposal.shape != base_topk_ids.shape[:2]:
        raise ValueError("candidate/proposal shapes are inconsistent")
    if not 2 <= support_size <= base_topk_ids.shape[-1]:
        raise ValueError("support_size lies outside the supplied Top-K")
    support = base_topk_ids[..., :support_size].clone()
    present = support.eq(proposal.unsqueeze(-1)).any(dim=-1)
    support[..., -1] = torch.where(present, support[..., -1], proposal)
    return support


def earliest_one_decision(
    *,
    candidate_ids: Tensor,
    candidate_scores: Tensor,
    lens_delta: Tensor,
    proposal: Tensor,
    threshold: float | Tensor,
) -> EarliestOneDecisionOutput:
    """Change only the earliest candidate that strictly clears a margin gate."""

    if candidate_ids.ndim != 3 or candidate_scores.shape != candidate_ids.shape:
        raise ValueError("candidate IDs/scores must share shape [batch, positions, K]")
    if lens_delta.shape != candidate_ids.shape or proposal.shape != candidate_ids.shape[:2]:
        raise ValueError("lens/proposal shapes do not match the candidate lattice")
    proposal_matches = candidate_ids.eq(proposal.unsqueeze(-1))

    adjusted = candidate_scores + lens_delta.to(candidate_scores.dtype)
    adjusted_float = adjusted.float()
    proposal_index = proposal_matches.to(torch.long).argmax(dim=-1)
    proposal_score = adjusted_float.gather(
        -1, proposal_index.unsqueeze(-1)
    ).squeeze(-1)
    best_score, best_index = adjusted_float.max(dim=-1)
    best_token = candidate_ids.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    margins = best_score - proposal_score
    if isinstance(threshold, Tensor):
        if threshold.device != margins.device or threshold.dtype != margins.dtype:
            raise ValueError("tensor threshold must already match score device/dtype")
        threshold_tensor = threshold
    else:
        threshold_tensor = torch.as_tensor(
            threshold, dtype=margins.dtype, device=margins.device
        )
    # Strict score improvement is independent of the calibrated threshold, so
    # exact ties can never replace the frozen proposal even if a caller passes
    # a negative threshold.
    eligible = (
        best_token.ne(proposal)
        & best_score.gt(proposal_score)
        & margins.gt(threshold_tensor)
    )
    batch, positions = proposal.shape
    axes = torch.arange(positions, device=proposal.device).view(1, -1)
    sentinel = torch.full_like(axes.expand(batch, -1), positions)
    selected_position = torch.where(
        eligible, axes.expand(batch, -1), sentinel
    ).min(dim=-1).values
    safe_position = selected_position.clamp_max(positions - 1)
    selected_token = best_token.gather(1, safe_position[:, None])[:, 0]
    # Sentinel==positions matches no axis and therefore implements KEEP with a
    # fixed-shape graph-safe where rather than dynamic boolean indexing.
    output = torch.where(
        axes.eq(selected_position[:, None]), selected_token[:, None], proposal
    )
    return EarliestOneDecisionOutput(
        token_ids=output,
        selected_position=selected_position,
        position_margins=margins,
        adjusted_candidate_scores=adjusted,
    )


def repair_earliest_frontier(
    proposal: Tensor,
    gold: Tensor,
    *,
    candidate_ids: Tensor | None,
) -> FrontierRepairOutput:
    """Apply one perfect repair at the current first rejection.

    The suffix is deliberately left untouched.  Consequently, a gain beyond
    one token is counted only when that already-generated suffix happens to
    agree with the target after the repaired frontier.
    """

    if proposal.ndim != 2 or gold.shape != proposal.shape:
        raise ValueError("proposal and gold must share shape [batch, positions]")
    if candidate_ids is not None and candidate_ids.shape[:2] != proposal.shape:
        raise ValueError("candidate lattice does not match proposal horizon")
    batch, positions = proposal.shape
    accepted_before = accepted_lengths(proposal, gold)
    incomplete = accepted_before.lt(positions)
    frontier = accepted_before.clamp_max(positions - 1)
    batch_index = torch.arange(batch, device=proposal.device)
    frontier_gold = gold[batch_index, frontier]
    if candidate_ids is None:
        repair_available = incomplete
    else:
        frontier_candidates = candidate_ids[batch_index, frontier]
        repair_available = incomplete & frontier_candidates.eq(
            frontier_gold[:, None]
        ).any(dim=-1)

    repaired = proposal.clone()
    changed = repair_available & repaired[batch_index, frontier].ne(frontier_gold)
    repaired[batch_index[changed], frontier[changed]] = frontier_gold[changed]
    accepted_after = accepted_lengths(repaired, gold)
    return FrontierRepairOutput(
        token_ids=repaired,
        accepted_before=accepted_before,
        accepted_after=accepted_after,
        frontier=frontier,
        repair_available=repair_available,
        changed=changed,
    )


def sequential_perfect_frontier_repairs(
    proposal: Tensor,
    gold: Tensor,
    *,
    candidate_ids: Tensor | None,
    repairs: int,
) -> list[FrontierRepairOutput]:
    """Apply up to ``repairs`` perfect earliest-only repairs sequentially."""

    if repairs < 1:
        raise ValueError("repairs must be positive")
    current = proposal
    outputs: list[FrontierRepairOutput] = []
    for _ in range(repairs):
        result = repair_earliest_frontier(
            current, gold, candidate_ids=candidate_ids
        )
        outputs.append(result)
        current = result.token_ids
    return outputs


__all__ = [
    "FastCandidateDecodeOutput",
    "FrontierRepairOutput",
    "EarliestOneDecisionOutput",
    "R048TunedLens",
    "candidate_union_with_proposal",
    "earliest_one_decision",
    "fast_candidate_domino_decode",
    "fast_candidate_domino_decode_from_base",
    "repair_earliest_frontier",
    "sequential_perfect_frontier_repairs",
]
