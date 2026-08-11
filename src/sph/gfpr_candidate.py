"""Lightweight Top-K GFPR policy with selected-token causal feedback.

The deployed head reuses Domino's GRU and low-rank correction code but gathers
only the DFlash Top-K rows of the frozen vocabulary basis.  It therefore drops
the expensive 151k-way correction projection while retaining token-specific
causal scoring at every one of the 16 positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CandidateDecodeOutput:
    token_ids: Tensor
    candidate_scores: Tensor


@dataclass(frozen=True)
class CandidateUnionDecodeOutput:
    token_ids: Tensor
    candidate_scores: Tensor
    candidate_ids: Tensor
    released_token_ids: Tensor


@dataclass(frozen=True)
class CandidateUnionTeacherOutput:
    scores: Tensor
    candidate_ids: Tensor
    released_token_ids: Tensor


@dataclass(frozen=True)
class CandidateFrontierLossOutput:
    loss: Tensor
    repair_loss: Tensor
    keep_loss: Tensor
    frontier: Tensor
    repairable_blocks: Tensor
    gold_available_at_frontier: Tensor


@dataclass(frozen=True)
class CandidateDenseLossOutput:
    loss: Tensor
    unweighted_nll: Tensor
    active_positions: Tensor
    position_weights: Tensor


@dataclass(frozen=True)
class CandidateDenseMarginLossOutput:
    loss: Tensor
    active_positions: Tensor
    violations: Tensor


@dataclass(frozen=True)
class CandidateTargetDistillLossOutput:
    kl_loss: Tensor
    advantage_loss: Tensor
    active_positions: Tensor
    raw_teacher_top1_matches_gold: Tensor
    position_weights: Tensor


def select_anchor_early_exit_feature(
    hidden_states: tuple[Tensor, ...],
    *,
    context_length: int,
    early_layers: int,
) -> Tensor:
    """Select the current anchor after exactly ``early_layers`` target layers.

    Hugging Face causal models expose the embedding output as
    ``hidden_states[0]`` and the output of decoder layer ``i`` as
    ``hidden_states[i + 1]``.  The current anchor is the first token after the
    cached context, hence its sequence index is ``context_length``.  Keeping
    this indexing in one tested helper prevents silently substituting the
    previous-token boundary feature used by R042.
    """

    if early_layers < 1 or early_layers >= len(hidden_states):
        raise ValueError("early target layer count lies outside hidden states")
    selected_layer = hidden_states[early_layers]
    if selected_layer.ndim != 3 or selected_layer.shape[0] != 1:
        raise ValueError("target hidden state must have shape [1, sequence, width]")
    if context_length < 0 or context_length >= selected_layer.shape[1]:
        raise ValueError("anchor position lies outside target hidden state")
    return selected_layer[0, context_length].detach()


class GFPRCandidateHead(nn.Module):
    """Domino-initialized causal scorer restricted to a DFlash Top-K lattice."""

    def __init__(
        self,
        *,
        token_embeddings: Tensor,
        candidate_basis: Tensor,
        gru_weight_ih: Tensor,
        gru_weight_hh: Tensor,
        input_projection_weight: Tensor,
        positions: int = 16,
        candidates: int = 16,
        adapter_rank: int = 16,
        boundary_width: int = 0,
    ) -> None:
        super().__init__()
        if token_embeddings.ndim != 2 or candidate_basis.ndim != 2:
            raise ValueError("token embeddings and candidate basis must be matrices")
        vocabulary, hidden_width = token_embeddings.shape
        if candidate_basis.shape[0] != vocabulary:
            raise ValueError("candidate basis vocabulary differs from embeddings")
        if gru_weight_ih.shape[0] % 3:
            raise ValueError("invalid GRU input weight")
        state_width = gru_weight_ih.shape[0] // 3
        code_width = candidate_basis.shape[1]
        if gru_weight_ih.shape != (3 * state_width, hidden_width):
            raise ValueError("GRU input shape is inconsistent")
        if gru_weight_hh.shape != (3 * state_width, state_width):
            raise ValueError("GRU recurrent shape is inconsistent")
        if input_projection_weight.shape != (
            code_width,
            hidden_width + state_width,
        ):
            raise ValueError("correction input projection shape is inconsistent")
        if (
            positions < 1
            or candidates < 2
            or adapter_rank < 1
            or boundary_width < 0
        ):
            raise ValueError("invalid GFPR lattice size")

        self.vocabulary = int(vocabulary)
        self.hidden_width = int(hidden_width)
        self.state_width = int(state_width)
        self.code_width = int(code_width)
        self.positions = int(positions)
        self.candidates = int(candidates)
        self.adapter_rank = min(int(adapter_rank), int(code_width))
        self.boundary_width = int(boundary_width)
        self.register_buffer(
            "token_embeddings", token_embeddings.detach(), persistent=False
        )
        self.register_buffer(
            "candidate_basis", candidate_basis.detach(), persistent=False
        )
        self.prefix_gru = nn.GRU(
            hidden_width,
            state_width,
            num_layers=1,
            batch_first=True,
            bias=False,
            device=gru_weight_ih.device,
            dtype=gru_weight_ih.dtype,
        )
        self.input_projection = nn.Linear(
            hidden_width + state_width,
            code_width,
            bias=False,
            device=input_projection_weight.device,
            dtype=input_projection_weight.dtype,
        )
        self.residual_down = nn.Linear(
            hidden_width + state_width,
            self.adapter_rank,
            bias=False,
            device=input_projection_weight.device,
            dtype=input_projection_weight.dtype,
        )
        self.residual_up = nn.Linear(
            self.adapter_rank,
            code_width,
            bias=False,
            device=input_projection_weight.device,
            dtype=input_projection_weight.dtype,
        )
        self.boundary_down = (
            nn.Linear(
                self.boundary_width,
                self.adapter_rank,
                bias=False,
                device=input_projection_weight.device,
                dtype=input_projection_weight.dtype,
            )
            if self.boundary_width
            else None
        )
        with torch.no_grad():
            self.prefix_gru.weight_ih_l0.copy_(gru_weight_ih)
            self.prefix_gru.weight_hh_l0.copy_(gru_weight_hh)
            self.input_projection.weight.copy_(input_projection_weight)
            self.residual_up.weight.zero_()

        self.base_scale = nn.Parameter(
            torch.ones(
                positions, dtype=torch.float32, device=gru_weight_ih.device
            )
        )
        correction_scale = torch.ones(
            positions, dtype=torch.float32, device=gru_weight_ih.device
        )
        correction_scale[0] = 0.0
        self.correction_scale = nn.Parameter(correction_scale)
        self.rank_bias = nn.Parameter(
            torch.zeros(
                positions,
                candidates,
                dtype=torch.float32,
                device=gru_weight_ih.device,
            )
        )

    @classmethod
    def from_domino(
        cls,
        domino: Any,
        target_weight: Tensor,
        *,
        positions: int = 16,
        candidates: int = 16,
        adapter_rank: int = 16,
        boundary_width: int = 0,
    ) -> "GFPRCandidateHead":
        if bool(getattr(domino, "use_bias_norm", False)) or bool(
            getattr(domino, "use_bias_gate", False)
        ):
            raise ValueError(
                "GFPRCandidateHead needs explicit norm/gate support for this checkpoint"
            )
        return cls(
            token_embeddings=target_weight,
            candidate_basis=domino.embed_proj[2].weight,
            gru_weight_ih=domino.prefix_gru.weight_ih_l0,
            gru_weight_hh=domino.prefix_gru.weight_hh_l0,
            input_projection_weight=domino.embed_proj[0].weight,
            positions=positions,
            candidates=candidates,
            adapter_rank=adapter_rank,
            boundary_width=boundary_width,
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _scores(
        self,
        *,
        hidden: Tensor,
        states: Tensor,
        candidate_ids: Tensor,
        candidate_logits: Tensor,
        target_boundary: Tensor | None = None,
        position_offset: int = 0,
    ) -> Tensor:
        batch, positions, candidates = candidate_ids.shape
        if hidden.shape != (batch, positions, self.hidden_width):
            raise ValueError("parallel hidden shape is inconsistent")
        if states.shape != (batch, positions, self.state_width):
            raise ValueError("prefix state shape is inconsistent")
        if candidate_logits.shape != candidate_ids.shape:
            raise ValueError("candidate IDs/logits differ in shape")
        if candidates != self.candidates:
            raise ValueError("candidate count differs from head capacity")
        if position_offset < 0 or position_offset + positions > self.positions:
            raise ValueError("position slice exceeds head capacity")
        features = torch.cat([hidden, states], dim=-1)
        code = F.silu(self.input_projection(features))
        residual_code = self._residual_code(features, target_boundary)
        basis = F.embedding(candidate_ids, self.candidate_basis)
        correction = torch.einsum("blc,blkc->blk", code, basis).float()
        residual = torch.einsum(
            "blc,blkc->blk", residual_code, basis
        ).float()
        position_slice = slice(position_offset, position_offset + positions)
        base_scale = self.base_scale[position_slice].view(1, positions, 1)
        correction_scale = self.correction_scale[position_slice].view(
            1, positions, 1
        )
        rank_bias = self.rank_bias[position_slice].view(
            1, positions, candidates
        )
        return (
            base_scale * candidate_logits.float()
            + correction_scale * correction
            + residual
            + rank_bias
        )

    def _residual_code(
        self, features: Tensor, target_boundary: Tensor | None
    ) -> Tensor:
        """Return a zero-initialized local/verified-context residual code."""

        batch, positions, _ = features.shape
        local = self.residual_down(features)
        if self.boundary_down is None:
            if target_boundary is not None:
                raise ValueError("head was built without a target boundary input")
        else:
            if target_boundary is None or target_boundary.shape != (
                batch,
                self.boundary_width,
            ):
                raise ValueError("target boundary feature has the wrong shape")
            normalized = target_boundary.float() * torch.rsqrt(
                target_boundary.float().square().mean(dim=-1, keepdim=True)
                + 1e-6
            )
            context = self.boundary_down(
                normalized.to(self.boundary_down.weight.dtype)
            )
            # The context changes a token-specific candidate code after a
            # nonlinear interaction with every position's local causal state;
            # it is not an additive scalar bias shared by candidates.
            local = local + context[:, None, :]
        return self.residual_up(F.silu(local))

    def _released_union_scores(
        self,
        *,
        hidden: Tensor,
        states: Tensor,
        base_candidate_ids: Tensor,
        target_boundary: Tensor | None = None,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Exact Domino scores plus a zero-initialized candidate residual."""

        batch, positions, candidates = base_candidate_ids.shape
        if candidates != self.candidates:
            raise ValueError("candidate count differs from head capacity")
        if hidden.shape != (batch, positions, self.hidden_width):
            raise ValueError("parallel hidden shape is inconsistent")
        if states.shape != (batch, positions, self.state_width):
            raise ValueError("prefix state shape is inconsistent")
        if position_offset < 0 or position_offset + positions > self.positions:
            raise ValueError("position slice exceeds head capacity")

        features = torch.cat([hidden, states], dim=-1)
        code = F.silu(self.input_projection(features))
        residual_code = self._residual_code(features, target_boundary)
        base_logits = F.linear(hidden, self.token_embeddings)
        correction = F.linear(code, self.candidate_basis)
        position_slice = slice(position_offset, position_offset + positions)
        base_scale = self.base_scale[position_slice].to(base_logits.dtype).view(
            1, positions, 1
        )
        correction_scale = self.correction_scale[position_slice].to(
            correction.dtype
        ).view(1, positions, 1)
        # Keep the BF16 multiply/add before float argmax: this is the released
        # Domino numerical contract, not merely an algebraically equal score.
        released_scores = base_scale * base_logits + correction_scale * correction
        released_ids = released_scores.float().argmax(dim=-1)
        union_ids = base_candidate_ids.clone()
        missing = ~union_ids.eq(released_ids.unsqueeze(-1)).any(dim=-1)
        union_ids[..., -1] = torch.where(
            missing, released_ids, union_ids[..., -1]
        )
        scores = released_scores.gather(-1, union_ids).float()
        residual_basis = F.embedding(union_ids, self.candidate_basis)
        residual = torch.einsum(
            "blc,blkc->blk", residual_code, residual_basis
        ).float()
        rank_bias = self.rank_bias[position_slice].view(
            1, positions, candidates
        )
        scores = scores + residual + rank_bias
        # Full-vocabulary argmax resolves exact BF16 ties by vocabulary order,
        # whereas a gathered lattice resolves them by candidate order.  Raise
        # the released action by exactly one detached float32 ULP so the
        # training frontier retains full-vocabulary tie semantics without
        # changing gradients or any strict inequality.
        released_mask = union_ids.eq(released_ids.unsqueeze(-1))
        ulp = torch.nextafter(
            scores.detach(), torch.full_like(scores.detach(), float("inf"))
        ) - scores.detach()
        scores = scores + ulp * released_mask.float()
        return scores, union_ids, released_ids

    def teacher_union_scores(
        self,
        *,
        anchors: Tensor,
        gold: Tensor,
        hidden: Tensor,
        base_candidate_ids: Tensor,
        target_boundary: Tensor | None = None,
    ) -> CandidateUnionTeacherOutput:
        """Teacher-prefix K16 union containing the exact current Domino action."""

        if gold.shape != base_candidate_ids.shape[:2]:
            raise ValueError("gold and candidate lattice shapes differ")
        prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
        states, _ = self.prefix_gru(F.embedding(prefix_ids, self.token_embeddings))
        scores, candidate_ids, released_ids = self._released_union_scores(
            hidden=hidden,
            states=states,
            base_candidate_ids=base_candidate_ids,
            target_boundary=target_boundary,
        )
        return CandidateUnionTeacherOutput(
            scores=scores,
            candidate_ids=candidate_ids,
            released_token_ids=released_ids,
        )

    def teacher_stored_union_scores(
        self,
        *,
        anchors: Tensor,
        gold: Tensor,
        hidden: Tensor,
        base_candidate_ids: Tensor,
        released_token_ids: Tensor,
        target_boundary: Tensor | None = None,
    ) -> CandidateUnionTeacherOutput:
        """Fast exact-frontier teacher using actions stored by released Domino.

        Before the current first rejection, the stored rollout prefix equals
        ``gold``; consequently its released action is the exact current-prefix
        Domino action at every position touched by the frontier/protection
        objective.  Candidate-only dot products avoid two training-time full
        vocabulary GEMMs.  The stored action is made the tie-safe frozen winner
        to absorb harmless GEMM-vs-gather rounding differences.
        """

        batch, positions, candidates = base_candidate_ids.shape
        if gold.shape != (batch, positions) or released_token_ids.shape != gold.shape:
            raise ValueError("stored union tensors have inconsistent shapes")
        if candidates != self.candidates:
            raise ValueError("candidate count differs from head capacity")
        prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
        states, _ = self.prefix_gru(F.embedding(prefix_ids, self.token_embeddings))
        union_ids = base_candidate_ids.clone()
        missing = ~union_ids.eq(released_token_ids.unsqueeze(-1)).any(dim=-1)
        union_ids[..., -1] = torch.where(
            missing, released_token_ids, union_ids[..., -1]
        )

        features = torch.cat([hidden, states], dim=-1)
        code = F.silu(self.input_projection(features))
        residual_code = self._residual_code(features, target_boundary)
        token_basis = F.embedding(union_ids, self.token_embeddings)
        correction_basis = F.embedding(union_ids, self.candidate_basis)
        base_scores = torch.einsum(
            "bld,blkd->blk", hidden, token_basis
        )
        correction = torch.einsum(
            "blc,blkc->blk", code, correction_basis
        )
        base_scale = self.base_scale[:positions].to(base_scores.dtype).view(
            1, positions, 1
        )
        correction_scale = self.correction_scale[:positions].to(
            correction.dtype
        ).view(1, positions, 1)
        frozen_scores = (
            base_scale * base_scores + correction_scale * correction
        ).float()
        released_mask = union_ids.eq(released_token_ids.unsqueeze(-1))
        maximum = frozen_scores.max(dim=-1, keepdim=True).values
        maximum_next = torch.nextafter(
            maximum.detach(),
            torch.full_like(maximum.detach(), float("inf")),
        )
        frozen_scores = torch.where(
            released_mask,
            torch.maximum(frozen_scores, maximum_next),
            frozen_scores,
        )
        residual = torch.einsum(
            "blc,blkc->blk", residual_code, correction_basis
        ).float()
        scores = frozen_scores + residual + self.rank_bias[:positions].view(
            1, positions, candidates
        )
        return CandidateUnionTeacherOutput(
            scores=scores,
            candidate_ids=union_ids,
            released_token_ids=released_token_ids,
        )

    def teacher_scores(
        self,
        *,
        anchors: Tensor,
        gold: Tensor,
        hidden: Tensor,
        candidate_ids: Tensor,
        candidate_logits: Tensor,
        target_boundary: Tensor | None = None,
    ) -> Tensor:
        if gold.shape != candidate_ids.shape[:2]:
            raise ValueError("gold and candidate lattice shapes differ")
        prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
        states, _ = self.prefix_gru(F.embedding(prefix_ids, self.token_embeddings))
        return self._scores(
            hidden=hidden,
            states=states,
            candidate_ids=candidate_ids,
            candidate_logits=candidate_logits,
            target_boundary=target_boundary,
        )

    @torch.no_grad()
    def decode(
        self,
        *,
        anchors: Tensor,
        hidden: Tensor,
        candidate_ids: Tensor,
        candidate_logits: Tensor,
        target_boundary: Tensor | None = None,
    ) -> CandidateDecodeOutput:
        batch, positions, _ = candidate_ids.shape
        if anchors.shape != (batch,):
            raise ValueError("anchors must have shape [batch]")
        _, state = self.prefix_gru(
            F.embedding(anchors[:, None], self.token_embeddings)
        )
        selected: list[Tensor] = []
        score_rows: list[Tensor] = []
        for position in range(positions):
            scores = self._scores(
                hidden=hidden[:, position : position + 1],
                states=state.transpose(0, 1),
                candidate_ids=candidate_ids[:, position : position + 1],
                candidate_logits=candidate_logits[:, position : position + 1],
                target_boundary=target_boundary,
                position_offset=position,
            )
            index = scores.argmax(dim=-1)
            token = candidate_ids[:, position : position + 1].gather(
                -1, index.unsqueeze(-1)
            ).squeeze(-1)
            selected.append(token)
            score_rows.append(scores)
            if position + 1 < positions:
                _, state = self.prefix_gru(
                    F.embedding(token, self.token_embeddings), state
                )
        return CandidateDecodeOutput(
            token_ids=torch.cat(selected, dim=1),
            candidate_scores=torch.cat(score_rows, dim=1),
        )

    @torch.no_grad()
    def decode_with_released_union(
        self,
        *,
        anchors: Tensor,
        hidden: Tensor,
        base_candidate_ids: Tensor,
        target_boundary: Tensor | None = None,
    ) -> CandidateUnionDecodeOutput:
        """Causal decode over Top-15 plus the current-prefix Domino action."""

        batch, positions, _ = base_candidate_ids.shape
        if anchors.shape != (batch,):
            raise ValueError("anchors must have shape [batch]")
        _, state = self.prefix_gru(
            F.embedding(anchors[:, None], self.token_embeddings)
        )
        selected: list[Tensor] = []
        score_rows: list[Tensor] = []
        candidate_rows: list[Tensor] = []
        released_rows: list[Tensor] = []
        for position in range(positions):
            scores, candidate_ids, released_ids = self._released_union_scores(
                hidden=hidden[:, position : position + 1],
                states=state.transpose(0, 1),
                base_candidate_ids=base_candidate_ids[
                    :, position : position + 1
                ],
                target_boundary=target_boundary,
                position_offset=position,
            )
            best_index = scores.argmax(dim=-1)
            released_index = candidate_ids.eq(
                released_ids.unsqueeze(-1)
            ).to(torch.long).argmax(dim=-1)
            best_score = scores.gather(
                -1, best_index.unsqueeze(-1)
            ).squeeze(-1)
            released_score = scores.gather(
                -1, released_index.unsqueeze(-1)
            ).squeeze(-1)
            # Strict override: exact ties always retain the released action.
            index = torch.where(
                best_score > released_score, best_index, released_index
            )
            token = candidate_ids.gather(-1, index.unsqueeze(-1)).squeeze(-1)
            selected.append(token)
            score_rows.append(scores)
            candidate_rows.append(candidate_ids)
            released_rows.append(released_ids)
            if position + 1 < positions:
                _, state = self.prefix_gru(
                    F.embedding(token, self.token_embeddings), state
                )
        return CandidateUnionDecodeOutput(
            token_ids=torch.cat(selected, dim=1),
            candidate_scores=torch.cat(score_rows, dim=1),
            candidate_ids=torch.cat(candidate_rows, dim=1),
            released_token_ids=torch.cat(released_rows, dim=1),
        )


def candidate_frontier_margin_loss(
    scores: Tensor,
    candidate_ids: Tensor,
    gold: Tensor,
    *,
    break_margin: float = 1e-4,
    keep_margin: float = 0.05,
    break_weight: float = 1.0,
    keep_weight: float = 0.1,
    block_weights: Tensor | None = None,
) -> CandidateFrontierLossOutput:
    """Repair only a reachable first rejection in the Top-K lattice."""

    if scores.shape != candidate_ids.shape or gold.shape != scores.shape[:2]:
        raise ValueError("scores/candidates/gold shapes are inconsistent")
    matches = candidate_ids.eq(gold.unsqueeze(-1))
    available = matches.any(dim=-1)
    gold_indices = matches.to(torch.long).argmax(dim=-1)
    predicted_indices = scores.detach().argmax(dim=-1)
    predicted_ids = candidate_ids.gather(
        -1, predicted_indices.unsqueeze(-1)
    ).squeeze(-1)
    batch, positions = gold.shape
    axes = torch.arange(positions, device=gold.device).view(1, -1)
    sentinel = torch.full_like(gold, positions)
    mismatch_axes = torch.where(
        predicted_ids.ne(gold), axes.expand(batch, -1), sentinel
    )
    frontier = mismatch_axes.min(dim=-1).values
    protected = axes < frontier[:, None]
    repair = (axes == frontier[:, None]) & available

    values, ids = scores.float().topk(2, dim=-1)
    gold_scores = scores.float().gather(
        -1, gold_indices.unsqueeze(-1)
    ).squeeze(-1)
    competitor = torch.where(
        ids[..., 0].eq(gold_indices), values[..., 1], values[..., 0]
    )
    margins = gold_scores - competitor
    per_block_keep = (
        torch.relu(keep_margin - margins) * protected.to(margins.dtype)
    ).sum(dim=-1) / frontier.clamp_min(1).to(margins.dtype)
    per_block_repair = (
        torch.relu(break_margin - margins) * repair.to(margins.dtype)
    ).sum(dim=-1)
    if block_weights is None:
        weights = torch.full_like(per_block_keep, 1.0 / batch)
    else:
        if block_weights.shape != (batch,):
            raise ValueError("block weights must have shape [batch]")
        weights = block_weights.float()
        if bool(torch.any(weights < 0)) or float(weights.sum()) <= 0:
            raise ValueError("block weights must be non-negative with positive sum")
        weights = weights / weights.sum()
    repair_loss = (weights * per_block_repair).sum()
    keep_loss = (weights * per_block_keep).sum()
    return CandidateFrontierLossOutput(
        loss=break_weight * repair_loss + keep_weight * keep_loss,
        repair_loss=repair_loss.detach(),
        keep_loss=keep_loss.detach(),
        frontier=frontier.detach(),
        repairable_blocks=repair.any(dim=-1).detach(),
        gold_available_at_frontier=(
            available.gather(
                -1, frontier.clamp_max(positions - 1).unsqueeze(-1)
            ).squeeze(-1)
            & frontier.lt(positions)
        ).detach(),
    )


def candidate_dense_dpace_loss(
    scores: Tensor,
    candidate_ids: Tensor,
    gold: Tensor,
    *,
    alpha: float = 0.5,
    block_weights: Tensor | None = None,
) -> CandidateDenseLossOutput:
    """Dense candidate supervision, censored at the first Top-K miss.

    Frontier repair supplies only one label per block and can memorize a small
    rollout.  This auxiliary objective exposes every reachable candidate label
    while retaining acceptance-aware D-PACE weighting.  Positions at and after
    the first missing gold token are excluded because their teacher prefix is
    no longer a realizable candidate-only path.
    """

    if scores.shape != candidate_ids.shape or gold.shape != scores.shape[:2]:
        raise ValueError("scores/candidates/gold shapes are inconsistent")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    batch, positions, candidates = scores.shape
    matches = candidate_ids.eq(gold.unsqueeze(-1))
    available = matches.any(dim=-1)
    active = torch.cumprod(available.to(torch.long), dim=-1).bool()
    gold_indices = matches.to(torch.long).argmax(dim=-1).clamp(0, candidates - 1)
    log_probs = torch.log_softmax(scores.float(), dim=-1)
    gold_log_probs = log_probs.gather(
        -1, gold_indices.unsqueeze(-1)
    ).squeeze(-1)
    per_position_nll = -gold_log_probs
    with torch.no_grad():
        probabilities = gold_log_probs.detach().exp()
        smoothed = (1.0 - alpha) * probabilities + alpha
        smoothed = torch.where(active, smoothed, torch.ones_like(smoothed))
        inclusive_prefix = torch.cumprod(smoothed, dim=-1)
        position_weights = torch.flip(
            torch.cumsum(
                torch.flip(inclusive_prefix * active.float(), dims=[-1]),
                dim=-1,
            ),
            dims=[-1],
        )
    per_block = (
        per_position_nll * position_weights * active.float()
    ).sum(dim=-1) / float(positions)
    if block_weights is None:
        normalized_block_weights = torch.full_like(per_block, 1.0 / batch)
    else:
        if block_weights.shape != (batch,):
            raise ValueError("block weights must have shape [batch]")
        normalized_block_weights = block_weights.float()
        if bool(torch.any(normalized_block_weights < 0)) or float(
            normalized_block_weights.sum()
        ) <= 0:
            raise ValueError(
                "block weights must be non-negative with positive sum"
            )
        normalized_block_weights = (
            normalized_block_weights / normalized_block_weights.sum()
        )
    active_count = active.sum().clamp_min(1)
    unweighted_nll = (
        per_position_nll * active.float()
    ).sum() / active_count
    return CandidateDenseLossOutput(
        loss=(normalized_block_weights * per_block).sum(),
        unweighted_nll=unweighted_nll.detach(),
        active_positions=active.detach(),
        position_weights=position_weights.detach(),
    )


def candidate_dense_margin_loss(
    scores: Tensor,
    candidate_ids: Tensor,
    gold: Tensor,
    *,
    margin: float = 0.05,
    alpha: float = 0.5,
    block_weights: Tensor | None = None,
) -> CandidateDenseMarginLossOutput:
    """Acceptance-weighted hinge over every reachable Top-K decision.

    Unlike dense cross entropy, this objective is exactly zero once the gold
    candidate has a small safety margin.  It therefore supplies suffix labels
    without continually sharpening already-correct Domino decisions.
    """

    if scores.shape != candidate_ids.shape or gold.shape != scores.shape[:2]:
        raise ValueError("scores/candidates/gold shapes are inconsistent")
    if margin < 0 or not 0.0 <= alpha <= 1.0:
        raise ValueError("invalid dense margin configuration")
    batch, positions, candidates = scores.shape
    matches = candidate_ids.eq(gold.unsqueeze(-1))
    available = matches.any(dim=-1)
    active = torch.cumprod(available.to(torch.long), dim=-1).bool()
    gold_indices = matches.to(torch.long).argmax(dim=-1).clamp(0, candidates - 1)
    scores_float = scores.float()
    gold_scores = scores_float.gather(
        -1, gold_indices.unsqueeze(-1)
    ).squeeze(-1)
    top_values, top_indices = scores_float.topk(2, dim=-1)
    competitors = torch.where(
        top_indices[..., 0].eq(gold_indices),
        top_values[..., 1],
        top_values[..., 0],
    )
    violations = torch.relu(margin - (gold_scores - competitors))
    with torch.no_grad():
        probabilities = torch.softmax(scores_float.detach(), dim=-1).gather(
            -1, gold_indices.unsqueeze(-1)
        ).squeeze(-1)
        smoothed = (1.0 - alpha) * probabilities + alpha
        smoothed = torch.where(active, smoothed, torch.ones_like(smoothed))
        inclusive_prefix = torch.cumprod(smoothed, dim=-1)
        position_weights = torch.flip(
            torch.cumsum(
                torch.flip(inclusive_prefix * active.float(), dims=[-1]),
                dim=-1,
            ),
            dims=[-1],
        )
    per_block = (
        violations * position_weights * active.float()
    ).sum(dim=-1) / float(positions)
    if block_weights is None:
        normalized_block_weights = torch.full_like(per_block, 1.0 / batch)
    else:
        if block_weights.shape != (batch,):
            raise ValueError("block weights must have shape [batch]")
        normalized_block_weights = block_weights.float()
        if bool(torch.any(normalized_block_weights < 0)) or float(
            normalized_block_weights.sum()
        ) <= 0:
            raise ValueError(
                "block weights must be non-negative with positive sum"
            )
        normalized_block_weights = (
            normalized_block_weights / normalized_block_weights.sum()
        )
    return CandidateDenseMarginLossOutput(
        loss=(normalized_block_weights * per_block).sum(),
        active_positions=active.detach(),
        violations=(violations * active.float()).detach(),
    )


def candidate_target_distillation_loss(
    student_scores: Tensor,
    union_candidate_ids: Tensor,
    base_candidate_ids: Tensor,
    target_base_candidate_logits: Tensor,
    released_token_ids: Tensor,
    target_released_logits: Tensor,
    gold: Tensor,
    released_lengths: Tensor,
    *,
    temperature: float = 1.0,
    huber_delta: float = 0.5,
    protect_weight: float = 1.0,
    repair_weight: float = 4.0,
    block_weights: Tensor | None = None,
) -> CandidateTargetDistillLossOutput:
    """Distill target margins on the released policy's reachable frontier.

    The union contains DFlash candidates plus the released Domino action.  At
    accepted positions and the original first rejection that action was
    produced under the gold prefix, so both the candidate set and labels match
    the deployed causal state.  Later suffixes are deliberately excluded: the
    stored action there came from a wrong prefix and is not a valid teacher.

    Canonical gold is the acceptance contract.  A full-sequence BF16 target
    replay can disagree with cached greedy generation on near ties, so the
    gold candidate is raised only to one float32 ULP above the candidate-set
    maximum.  This retains target dark logits while preventing a numerical
    replay artifact from training against the measured acceptance label.
    """

    if (
        student_scores.shape != union_candidate_ids.shape
        or target_base_candidate_logits.shape != base_candidate_ids.shape
        or base_candidate_ids.shape != union_candidate_ids.shape
    ):
        raise ValueError("student/union/base target tensors differ in shape")
    batch, positions, candidates = student_scores.shape
    if (
        released_token_ids.shape != (batch, positions)
        or target_released_logits.shape != (batch, positions)
        or gold.shape != (batch, positions)
        or released_lengths.shape != (batch,)
    ):
        raise ValueError("released/gold tensors have inconsistent shapes")
    if temperature <= 0 or huber_delta <= 0:
        raise ValueError("temperature and Huber delta must be positive")
    if min(protect_weight, repair_weight) < 0:
        raise ValueError("target distillation weights must be non-negative")
    if bool(torch.any((released_lengths < 0) | (released_lengths > positions))):
        raise ValueError("released lengths lie outside the candidate horizon")

    # Align target logits to the possibly replaced final union slot.  Every
    # union action must originate from either the DFlash lattice or the stored
    # released action.
    source_matches = union_candidate_ids.unsqueeze(-1).eq(
        base_candidate_ids.unsqueeze(-2)
    )
    source_available = source_matches.any(dim=-1)
    source_indices = source_matches.to(torch.long).argmax(dim=-1)
    aligned_target = target_base_candidate_logits.float().gather(
        -1, source_indices
    )
    released_slots = union_candidate_ids.eq(released_token_ids.unsqueeze(-1))
    valid_union = source_available | released_slots
    if not bool(valid_union.all()):
        raise ValueError("union contains an action without a target logit")
    aligned_target = torch.where(
        source_available,
        aligned_target,
        target_released_logits.float().unsqueeze(-1),
    )

    gold_slots = union_candidate_ids.eq(gold.unsqueeze(-1))
    gold_available = gold_slots.any(dim=-1)
    raw_teacher_ids = union_candidate_ids.gather(
        -1, aligned_target.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    raw_teacher_top1_matches_gold = raw_teacher_ids.eq(gold)
    maximum = aligned_target.max(dim=-1, keepdim=True).values
    maximum_next = torch.nextafter(
        maximum.detach(), torch.full_like(maximum.detach(), float("inf"))
    )
    teacher_scores = torch.where(
        gold_slots,
        torch.maximum(aligned_target, maximum_next),
        aligned_target,
    )

    axes = torch.arange(positions, device=student_scores.device).view(1, -1)
    protected = axes < released_lengths.unsqueeze(-1)
    frontier = (
        axes.eq(released_lengths.unsqueeze(-1))
        & released_lengths.unsqueeze(-1).lt(positions)
    )
    active = (protected | frontier) & gold_available
    position_weights = (
        protect_weight * protected.float() + repair_weight * frontier.float()
    ) * active.float()

    teacher_probabilities = torch.softmax(teacher_scores / temperature, dim=-1)
    student_log_probabilities = torch.log_softmax(
        student_scores.float() / temperature, dim=-1
    )
    teacher_log_probabilities = torch.log_softmax(
        teacher_scores / temperature, dim=-1
    )
    per_position_kl = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1) * (temperature * temperature)

    released_indices = released_slots.to(torch.long).argmax(dim=-1)
    student_advantages = student_scores.float() - student_scores.float().gather(
        -1, released_indices.unsqueeze(-1)
    )
    target_advantages = teacher_scores - teacher_scores.gather(
        -1, released_indices.unsqueeze(-1)
    )
    per_candidate_huber = F.huber_loss(
        student_advantages,
        target_advantages,
        reduction="none",
        delta=huber_delta,
    )
    per_position_advantage = per_candidate_huber.mean(dim=-1)

    if block_weights is None:
        normalized_block_weights = torch.full(
            (batch,),
            1.0 / batch,
            dtype=torch.float32,
            device=student_scores.device,
        )
    else:
        if block_weights.shape != (batch,):
            raise ValueError("block weights must have shape [batch]")
        normalized_block_weights = block_weights.float()
        if bool(torch.any(normalized_block_weights < 0)) or float(
            normalized_block_weights.sum()
        ) <= 0:
            raise ValueError(
                "block weights must be non-negative with positive sum"
            )
        normalized_block_weights = (
            normalized_block_weights / normalized_block_weights.sum()
        )
    denominator = position_weights.sum(dim=-1).clamp_min(1.0)
    per_block_kl = (per_position_kl * position_weights).sum(dim=-1) / denominator
    per_block_advantage = (
        per_position_advantage * position_weights
    ).sum(dim=-1) / denominator
    return CandidateTargetDistillLossOutput(
        kl_loss=(normalized_block_weights * per_block_kl).sum(),
        advantage_loss=(
            normalized_block_weights * per_block_advantage
        ).sum(),
        active_positions=active.detach(),
        raw_teacher_top1_matches_gold=raw_teacher_top1_matches_gold.detach(),
        position_weights=position_weights.detach(),
    )


__all__ = [
    "CandidateDenseLossOutput",
    "CandidateDenseMarginLossOutput",
    "CandidateTargetDistillLossOutput",
    "CandidateDecodeOutput",
    "CandidateFrontierLossOutput",
    "CandidateUnionDecodeOutput",
    "CandidateUnionTeacherOutput",
    "GFPRCandidateHead",
    "select_anchor_early_exit_feature",
    "candidate_dense_dpace_loss",
    "candidate_dense_margin_loss",
    "candidate_frontier_margin_loss",
    "candidate_target_distillation_loss",
]
