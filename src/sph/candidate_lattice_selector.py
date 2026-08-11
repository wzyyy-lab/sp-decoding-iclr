"""Lightweight selectors over the complete DFlash candidate lattice.

The existing survival head scores one Markov transition at a time.  In
particular, its neural scorer never observes the candidates available at other
draft positions.  This module makes the intended comparison explicit:

* ``scope="local"`` lets a node attend only to the K candidates at its own
  position.
* ``scope="global"`` uses the same parameters but lets every one of the L x K
  candidate nodes attend to the complete block.
* ``scope="causal"`` is a diagnostic control that can see the current and
  preceding candidate sets, but no future candidate sets.

All three scopes therefore have exactly matched parameter counts.  The model
still emits one ordinary token chain, so DFlash verification is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class CandidateLatticeOutput:
    """Scores and auxiliary calibration predictions for one candidate block."""

    edge_scores: Tensor
    log_probs: Tensor
    unary_scores: Tensor
    residual_scores: Tensor
    base_scores: Tensor
    in_lattice_logits: Tensor
    base_correct_logits: Tensor


@dataclass
class CandidateLossOutput:
    """Decomposed candidate-only training objective."""

    loss: Tensor
    candidate_nll: Tensor
    in_lattice_bce: Tensor
    base_correct_bce: Tensor
    active_positions: Tensor
    position_weights: Tensor
    gold_probabilities: Tensor


@dataclass
class PathDecodeOutput:
    """One candidate index per position and its chain score."""

    path: Tensor
    score: Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class CandidateLatticeBlock(nn.Module):
    """Pre-norm transformer block with an explicit candidate-attention scope."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout
        self.attention_norm = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.attention_out = nn.Linear(model_dim, model_dim, bias=False)
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_multiplier * model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ff_multiplier * model_dim, model_dim, bias=False),
        )

    @staticmethod
    def _allowed_attention(
        length: int, candidates: int, scope: str, device: torch.device
    ) -> Tensor | None:
        if scope == "global":
            return None
        positions = torch.arange(length, device=device).repeat_interleave(
            candidates
        )
        query_position = positions[:, None]
        key_position = positions[None, :]
        if scope == "local":
            return query_position == key_position
        if scope == "causal":
            return key_position <= query_position
        raise ValueError(f"unknown candidate attention scope: {scope}")

    def forward(
        self, states: Tensor, *, length: int, candidates: int, scope: str
    ) -> Tensor:
        batch, nodes, _ = states.shape
        if nodes != length * candidates:
            raise ValueError("node count is inconsistent with L x K")
        normalized = self.attention_norm(states)
        query, key, value = (
            self.qkv(normalized)
            .view(batch, nodes, 3, self.num_heads, self.head_dim)
            .unbind(dim=2)
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention_scores = torch.matmul(
            query, key.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)
        allowed = self._allowed_attention(
            length, candidates, scope, states.device
        )
        if allowed is not None:
            attention_scores = attention_scores.masked_fill(
                ~allowed[None, None], torch.finfo(attention_scores.dtype).min
            )
        # Computing the normalization in fp32 avoids avoidable bfloat16
        # differences between the matched local and global controls.
        attention = torch.softmax(attention_scores.float(), dim=-1).to(
            value.dtype
        )
        attention = F.dropout(
            attention, p=self.dropout, training=self.training
        )
        mixed = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch, nodes, self.model_dim
        )
        states = states + F.dropout(
            self.attention_out(mixed),
            p=self.dropout,
            training=self.training,
        )
        states = states + F.dropout(
            self.feed_forward(self.feed_forward_norm(states)),
            p=self.dropout,
            training=self.training,
        )
        return states


class CandidateLatticeSelector(nn.Module):
    """Parameter-matched local/global selector over DFlash top-K candidates.

    Frozen target embeddings provide token semantics while a much smaller
    trainable token table can learn token-specific selection biases.  The
    latter is shared by node and transition features instead of introducing a
    second vocabulary-sized table.
    """

    VALID_SCOPES = {"local", "global", "causal"}

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        max_positions: int = 32,
        max_candidates: int = 16,
        model_dim: int = 128,
        token_dim: int = 64,
        transition_dim: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        scope: str = "global",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if scope not in self.VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(self.VALID_SCOPES)}")
        if max_positions < 1 or max_candidates < 1:
            raise ValueError("max_positions and max_candidates must be positive")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.max_candidates = max_candidates
        self.model_dim = model_dim
        self.token_dim = token_dim
        self.transition_dim = transition_dim
        self.scope = scope

        self.hidden_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.embedding_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False
        )
        self.hidden_projection = nn.Linear(hidden_size, model_dim, bias=False)
        self.frozen_embedding_projection = nn.Linear(
            hidden_size, model_dim, bias=False
        )
        self.token_embedding = nn.Embedding(vocab_size, token_dim)
        self.token_projection = nn.Linear(token_dim, model_dim, bias=False)
        self.position_embedding = nn.Embedding(max_positions, model_dim)
        self.rank_embedding = nn.Embedding(max_candidates, model_dim)
        # Candidate log probability, top-1 gap, retained top-K mass, and
        # conditional top-K entropy.
        self.scalar_projection = nn.Sequential(
            nn.Linear(4, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.input_norm = nn.LayerNorm(model_dim)
        self.blocks = nn.ModuleList(
            [
                CandidateLatticeBlock(
                    model_dim,
                    num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.residual_projection = nn.Linear(model_dim, 1)
        self.in_lattice_projection = nn.Linear(model_dim, 1)
        self.base_correct_projection = nn.Linear(model_dim, 1)

        self.previous_transition_projection = nn.Linear(
            model_dim, transition_dim, bias=False
        )
        self.next_transition_projection = nn.Linear(
            model_dim, transition_dim, bias=False
        )
        self.anchor_transition_projection = nn.Linear(
            token_dim, transition_dim, bias=False
        )
        self.base_scale_unconstrained = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0))
        )
        self.transition_scale_unconstrained = nn.Parameter(
            torch.tensor(_inverse_softplus(0.1))
        )

        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.rank_embedding.weight, mean=0.0, std=0.02)
        # Begin close to the standardized DFlash ranking without creating the
        # zero-gradient bottleneck of a scalar-zeroed residual branch.
        nn.init.normal_(self.residual_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_projection.bias)

    @property
    def base_scale(self) -> Tensor:
        return F.softplus(self.base_scale_unconstrained)

    @property
    def transition_scale(self) -> Tensor:
        return F.softplus(self.transition_scale_unconstrained)

    def _validate_inputs(
        self,
        hidden: Tensor,
        candidate_ids: Tensor,
        candidate_embeddings: Tensor,
        candidate_logits: Tensor,
        base_logsumexp: Tensor,
        anchor_ids: Tensor,
    ) -> tuple[int, int, int]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B, L, D]")
        batch, length, hidden_size = hidden.shape
        if hidden_size != self.hidden_size:
            raise ValueError("hidden size differs from model configuration")
        if candidate_ids.ndim != 3:
            raise ValueError("candidate_ids must have shape [B, L, K]")
        cb, cl, candidates = candidate_ids.shape
        if (cb, cl) != (batch, length):
            raise ValueError("candidate ids are inconsistent with hidden")
        if candidate_embeddings.shape != (
            batch,
            length,
            candidates,
            hidden_size,
        ):
            raise ValueError("candidate embeddings have an invalid shape")
        if candidate_logits.shape != (batch, length, candidates):
            raise ValueError("candidate logits have an invalid shape")
        if base_logsumexp.shape != (batch, length):
            raise ValueError("base_logsumexp has an invalid shape")
        if anchor_ids.shape != (batch,):
            raise ValueError("anchor_ids must have shape [B]")
        if length > self.max_positions:
            raise ValueError("block exceeds max_positions")
        if candidates > self.max_candidates:
            raise ValueError("candidate count exceeds max_candidates")
        return batch, length, candidates

    def forward(
        self,
        hidden: Tensor,
        candidate_ids: Tensor,
        candidate_embeddings: Tensor,
        candidate_logits: Tensor,
        base_logsumexp: Tensor,
        anchor_ids: Tensor,
    ) -> CandidateLatticeOutput:
        batch, length, candidates = self._validate_inputs(
            hidden,
            candidate_ids,
            candidate_embeddings,
            candidate_logits,
            base_logsumexp,
            anchor_ids,
        )

        conditional_log_probs = torch.log_softmax(
            candidate_logits.float(), dim=-1
        )
        top1_gap = (
            candidate_logits[..., :1].float() - candidate_logits.float()
        )
        retained_log_mass = (
            torch.logsumexp(candidate_logits.float(), dim=-1)
            - base_logsumexp.float()
        )
        conditional_probabilities = conditional_log_probs.exp()
        entropy = -(
            conditional_probabilities * conditional_log_probs
        ).sum(dim=-1)
        scalar_features = torch.stack(
            [
                conditional_log_probs,
                top1_gap,
                retained_log_mass[..., None].expand(-1, -1, candidates),
                entropy[..., None].expand(-1, -1, candidates),
            ],
            dim=-1,
        )

        trainable_tokens = self.token_embedding(candidate_ids)
        position_indices = torch.arange(
            length, device=hidden.device
        )[None, :, None]
        rank_indices = torch.arange(
            candidates, device=hidden.device
        )[None, None, :]
        # Canonical shards store these large tensors in bfloat16.  Normalize
        # in fp32 for numerical stability and to keep the module executable in
        # CPU protocol tests where CUDA autocast is unavailable.
        normalized_hidden = self.hidden_norm(hidden.float())
        normalized_candidates = self.embedding_norm(
            candidate_embeddings.float()
        )
        states = (
            self.hidden_projection(normalized_hidden)[:, :, None, :]
            + self.frozen_embedding_projection(
                normalized_candidates
            )
            + self.token_projection(trainable_tokens)
            + self.position_embedding(position_indices)
            + self.rank_embedding(rank_indices)
            + self.scalar_projection(scalar_features)
        )
        states = self.input_norm(states)
        flat_states = states.reshape(
            batch, length * candidates, self.model_dim
        )
        for block in self.blocks:
            flat_states = block(
                flat_states,
                length=length,
                candidates=candidates,
                scope=self.scope,
            )
        states = self.output_norm(
            flat_states.reshape(batch, length, candidates, self.model_dim)
        )

        residual_scores = self.residual_projection(states).squeeze(-1)
        centered = candidate_logits.float() - candidate_logits.float().mean(
            dim=-1, keepdim=True
        )
        scale = candidate_logits.float().std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-3)
        base_scores = centered / scale
        unary_scores = (
            self.base_scale.float() * base_scores
            + residual_scores.float()
        )

        previous_states = torch.cat(
            [
                torch.zeros_like(states[:, :1]),
                states[:, :-1],
            ],
            dim=1,
        )
        previous_transition = self.previous_transition_projection(
            previous_states
        )
        anchor_transition = self.anchor_transition_projection(
            self.token_embedding(anchor_ids)
        )
        previous_transition[:, 0] = anchor_transition[:, None, :]
        next_transition = self.next_transition_projection(states)
        pairwise = torch.einsum(
            "blud,blvd->bluv", previous_transition, next_transition
        ) / math.sqrt(self.transition_dim)
        edge_scores = (
            unary_scores[:, :, None, :]
            + self.transition_scale.float() * pairwise.float()
        )
        log_probs = torch.log_softmax(edge_scores, dim=-1)

        pooled = states.mean(dim=2)
        return CandidateLatticeOutput(
            edge_scores=edge_scores,
            log_probs=log_probs,
            unary_scores=unary_scores,
            residual_scores=residual_scores,
            base_scores=base_scores,
            in_lattice_logits=self.in_lattice_projection(pooled).squeeze(-1),
            base_correct_logits=self.base_correct_projection(pooled).squeeze(
                -1
            ),
        )


def prefix_candidate_mask(gold_in_lattice: Tensor) -> Tensor:
    """Positions with an observable candidate label before the first OTHER."""

    if gold_in_lattice.ndim != 2:
        raise ValueError("gold_in_lattice must have shape [B, L]")
    alive_before = torch.cat(
        [
            torch.ones_like(gold_in_lattice[:, :1]),
            gold_in_lattice[:, :-1].to(torch.int64)
            .cumprod(dim=1)
            .to(torch.bool),
        ],
        dim=1,
    )
    return alive_before & gold_in_lattice


def teacher_forced_logits(
    edge_scores: Tensor,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
) -> Tensor:
    """Gather each edge row using the preceding observed gold candidate."""

    if edge_scores.ndim != 4:
        raise ValueError("edge_scores must have shape [B, L, K, K]")
    batch, length, candidates, next_candidates = edge_scores.shape
    if candidates != next_candidates:
        raise ValueError("candidate dimensions must match")
    if gold_candidate_indices.shape != (batch, length):
        raise ValueError("gold_candidate_indices has an invalid shape")
    if gold_in_lattice.shape != (batch, length):
        raise ValueError("gold_in_lattice has an invalid shape")
    previous = torch.cat(
        [
            torch.zeros_like(gold_candidate_indices[:, :1]),
            gold_candidate_indices[:, :-1],
        ],
        dim=1,
    ).clamp(0, candidates - 1)
    return edge_scores.gather(
        2,
        previous[:, :, None, None].expand(-1, -1, 1, candidates),
    ).squeeze(2)


def dpace_position_weights(
    gold_probabilities: Tensor,
    active_positions: Tensor,
    *,
    minimum_probability: float = 0.05,
) -> Tensor:
    """Detached reach-times-continuation weights for prefix acceptance.

    For position i, the weight is the probability of reaching i through the
    preceding gold choices times the conditional expected continuation from i.
    The weights are normalized to mean one over observed candidate labels.
    """

    if gold_probabilities.shape != active_positions.shape:
        raise ValueError("probabilities and mask must have equal shape")
    probabilities = (
        gold_probabilities.detach()
        .float()
        .clamp(min=minimum_probability, max=1.0)
    )
    # A censored OTHER event terminates the useful suffix.  Setting inactive
    # positions to zero prevents continuation value from leaking through that
    # absorbing boundary.
    probabilities = torch.where(
        active_positions, probabilities, torch.zeros_like(probabilities)
    )
    batch, length = probabilities.shape
    reach_before = torch.ones_like(probabilities)
    if length > 1:
        reach_before[:, 1:] = probabilities[:, :-1].cumprod(dim=1)
    continuation = torch.ones_like(probabilities)
    running = torch.zeros(batch, device=probabilities.device)
    for position in range(length - 1, -1, -1):
        continuation[:, position] = 1.0 + running
        running = probabilities[:, position] * (1.0 + running)
    weights = reach_before * continuation
    weights = torch.where(active_positions, weights, torch.zeros_like(weights))
    denominator = weights.sum().clamp_min(1e-6)
    return weights * active_positions.sum().clamp_min(1) / denominator


def candidate_selector_loss(
    output: CandidateLatticeOutput,
    candidate_ids: Tensor,
    gold_ids: Tensor,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
    *,
    weighting: str = "dpace",
    rank_weight_power: float = 0.0,
    in_lattice_loss_weight: float = 0.1,
    base_correct_loss_weight: float = 0.25,
) -> CandidateLossOutput:
    """Candidate-listwise supervision plus two calibration auxiliaries."""

    if weighting not in {"uniform", "dpace"}:
        raise ValueError("weighting must be 'uniform' or 'dpace'")
    teacher_logits = teacher_forced_logits(
        output.edge_scores, gold_candidate_indices, gold_in_lattice
    )
    active = prefix_candidate_mask(gold_in_lattice)
    safe_gold = gold_candidate_indices.clamp(
        0, teacher_logits.shape[-1] - 1
    )
    per_position_nll = F.cross_entropy(
        teacher_logits.transpose(1, 2),
        safe_gold,
        reduction="none",
    )
    gold_probabilities = torch.softmax(
        teacher_logits.float(), dim=-1
    ).gather(-1, safe_gold.unsqueeze(-1)).squeeze(-1)
    if weighting == "dpace":
        position_weights = dpace_position_weights(
            gold_probabilities, active
        )
    else:
        position_weights = active.float()
    if rank_weight_power:
        rank_weights = (safe_gold.float() + 1.0).pow(rank_weight_power)
        position_weights = position_weights * rank_weights
        position_weights = (
            position_weights
            * active.sum().clamp_min(1)
            / position_weights.sum().clamp_min(1e-6)
        )
    candidate_nll = (
        per_position_nll * position_weights
    ).sum() / active.sum().clamp_min(1)

    # Coverage and base correctness are observable even after the first top-K
    # miss, so unlike the candidate label they can use every stored position.
    in_lattice_bce = F.binary_cross_entropy_with_logits(
        output.in_lattice_logits.float(),
        gold_in_lattice.float(),
    )
    base_correct = candidate_ids[..., 0] == gold_ids
    base_correct_bce = F.binary_cross_entropy_with_logits(
        output.base_correct_logits.float(),
        base_correct.float(),
    )
    loss = (
        candidate_nll
        + in_lattice_loss_weight * in_lattice_bce
        + base_correct_loss_weight * base_correct_bce
    )
    return CandidateLossOutput(
        loss=loss,
        candidate_nll=candidate_nll,
        in_lattice_bce=in_lattice_bce,
        base_correct_bce=base_correct_bce,
        active_positions=active,
        position_weights=position_weights,
        gold_probabilities=gold_probabilities,
    )


def viterbi_decode(edge_scores: Tensor) -> PathDecodeOutput:
    """Exact maximum-score path through a dense L x K candidate lattice."""

    if edge_scores.ndim != 4:
        raise ValueError("edge_scores must have shape [B, L, K, K]")
    batch, length, candidates, next_candidates = edge_scores.shape
    if candidates != next_candidates:
        raise ValueError("candidate dimensions must match")
    score = edge_scores[:, 0, 0]
    backpointers: list[Tensor] = []
    for position in range(1, length):
        joint = score[:, :, None] + edge_scores[:, position]
        score, pointer = joint.max(dim=1)
        backpointers.append(pointer)
    last = score.argmax(dim=-1)
    best_score = score.gather(1, last[:, None]).squeeze(1)
    batch_indices = torch.arange(batch, device=edge_scores.device)
    reversed_path = [last]
    for pointer in reversed(backpointers):
        last = pointer[batch_indices, last]
        reversed_path.append(last)
    return PathDecodeOutput(
        path=torch.stack(list(reversed(reversed_path)), dim=1),
        score=best_score,
    )


def path_scores(edge_scores: Tensor, path: Tensor) -> Tensor:
    """Total unnormalized edge score assigned to each supplied path."""

    if edge_scores.ndim != 4:
        raise ValueError("edge_scores must have shape [B, L, K, K]")
    batch, length, candidates, _ = edge_scores.shape
    if path.shape != (batch, length):
        raise ValueError("path must have shape [B, L]")
    batch_indices = torch.arange(batch, device=path.device)
    previous = torch.zeros(batch, dtype=torch.long, device=path.device)
    selected = []
    for position in range(length):
        current = path[:, position].clamp(0, candidates - 1)
        selected.append(
            edge_scores[batch_indices, position, previous, current]
        )
        previous = current
    return torch.stack(selected, dim=1).sum(dim=1)


def first_divergence_margin(
    log_probs: Tensor, learned_path: Tensor
) -> Tensor:
    """Log-probability advantage at the first non-base path decision.

    Blocks on which the learned path exactly equals DFlash receive ``-inf`` and
    therefore always take KEEP_BASE for every finite threshold.
    """

    if log_probs.ndim != 4:
        raise ValueError("log_probs must have shape [B, L, K, K]")
    batch, length, candidates, _ = log_probs.shape
    if learned_path.shape != (batch, length):
        raise ValueError("learned_path must have shape [B, L]")
    differs = learned_path != 0
    has_divergence = differs.any(dim=1)
    first = differs.to(torch.int64).argmax(dim=1)
    batch_indices = torch.arange(batch, device=log_probs.device)
    previous = torch.zeros(batch, dtype=torch.long, device=log_probs.device)
    selected_margins = torch.full(
        (batch,),
        float("-inf"),
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    for position in range(length):
        at_first = has_divergence & (first == position)
        current = learned_path[:, position].clamp(0, candidates - 1)
        learned_log_prob = log_probs[
            batch_indices, position, previous, current
        ]
        base_log_prob = log_probs[
            batch_indices, position, previous, torch.zeros_like(current)
        ]
        selected_margins = torch.where(
            at_first, learned_log_prob - base_log_prob, selected_margins
        )
        previous = current
    return selected_margins
