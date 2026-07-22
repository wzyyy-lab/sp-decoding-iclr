"""Tensor prototype for a single-chain, acceptance-aligned draft head.

The module deliberately does *not* implement tree verification.  It scores a
small DFlash top-k candidate lattice in parallel and returns exactly one token
sequence.  The target model can verify that sequence with the ordinary DFlash
longest-prefix verifier.

Notation
--------
``log_p[b, i, u, v]`` is the calibrated log probability that candidate ``v``
at position ``i`` is correct, conditional on candidate ``u`` at position
``i-1`` being correct.  At ``i == 0``, row zero denotes the already verified
anchor token.

For a fixed path y, the Markov surrogate predicts

    E[A(y)] = sum_i prod_{j <= i} p_j(y_j | y_{j-1}),

where A is the number of accepted draft tokens.  ``survival_decode`` finds the
exact maximizer of this objective with a backward dynamic program.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass
class TransitionOutput:
    """Candidate-restricted transition probabilities and residual logits."""

    edge_scores: Tensor
    log_probs: Tensor
    residual_logits: Tensor
    outside_log_mass: Tensor
    outside_log_probs: Tensor


@dataclass
class DecodeOutput:
    """One selected candidate index per draft position."""

    path: Tensor
    predicted_utility: Tensor


@dataclass
class ChainCRFOutput:
    """Causal conditionals induced by a candidate-only chain energy.

    This object is retained for the candidate-only CRF ablation.  The proposed
    model uses :class:`PrefixCRFOutput`, which includes an absorbing OTHER
    outcome and therefore represents failure probability explicitly.
    """

    log_conditionals: Tensor
    log_partition: Tensor


@dataclass
class PrefixCRFOutput:
    """Conditionals induced by a globally normalized variable-length prefix CRF.

    ``log_conditionals`` contains candidate continuation probabilities and
    ``outside_log_conditionals`` contains the probability of entering the
    absorbing OTHER state.  Candidate probabilities plus OTHER sum to one for
    every reachable predecessor state.
    """

    log_conditionals: Tensor
    outside_log_conditionals: Tensor
    log_partition: Tensor


def _log_sub_exp(log_x: Tensor, log_y: Tensor, eps: float = 1e-7) -> Tensor:
    """Stable log(exp(log_x) - exp(log_y)) for log_x >= log_y.

    In this prototype ``log_x`` is the full-vocabulary log-partition and
    ``log_y`` is the top-k contribution, so the inequality follows from the
    construction.  Clamping only protects against floating-point roundoff.
    """

    ratio = torch.exp(log_y - log_x).clamp(max=1.0 - eps)
    return log_x + torch.log1p(-ratio)


class SurvivalPathHead(nn.Module):
    """Low-rank context-conditioned transition scorer over DFlash top-k.

    The expensive full-vocabulary LM head remains unchanged and is evaluated
    once by DFlash.  This head only gathers K token embeddings and evaluates
    L x K x K low-rank transition scores in parallel.

    Args:
        hidden_size: DFlash/target hidden and tied-embedding dimension.
        rank: Rank of the pairwise transition potential.
    """

    def __init__(self, hidden_size: int, rank: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank

        self.prev_proj = nn.Linear(hidden_size, rank, bias=False)
        self.next_proj = nn.Linear(hidden_size, rank, bias=False)
        self.context_gate = nn.Linear(hidden_size, rank, bias=True)
        self.hidden_query = nn.Linear(hidden_size, rank, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden: Tensor,
        anchor_embedding: Tensor,
        candidate_embeddings: Tensor,
        candidate_base_logits: Tensor,
        base_logsumexp: Tensor,
    ) -> TransitionOutput:
        """Build calibrated top-k transition probabilities.

        Args:
            hidden: ``[B, L, D]`` DFlash block hidden states.
            anchor_embedding: ``[B, D]`` embedding of the verified anchor.
            candidate_embeddings: ``[B, L, K, D]`` tied embeddings gathered
                for the base-logit top-k candidates.
            candidate_base_logits: ``[B, L, K]`` corresponding base logits.
            base_logsumexp: ``[B, L]`` logsumexp of the *full-vocabulary* base
                logits.  It lets the top-k correction retain exact outside
                probability mass without another full-vocabulary projection.

        Returns:
            ``log_probs`` has shape ``[B, L, K, K]``.  Only row zero is used
            at position zero; later rows index the previous candidate.
        """

        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B, L, D]")
        if candidate_embeddings.ndim != 4:
            raise ValueError("candidate_embeddings must have shape [B, L, K, D]")

        batch, length, dim = hidden.shape
        cb, cl, candidates, embed_dim = candidate_embeddings.shape
        if (cb, cl, embed_dim) != (batch, length, dim):
            raise ValueError("candidate embedding shape is inconsistent with hidden")
        if anchor_embedding.shape != (batch, dim):
            raise ValueError("anchor_embedding must have shape [B, D]")
        if candidate_base_logits.shape != (batch, length, candidates):
            raise ValueError("candidate_base_logits must have shape [B, L, K]")
        if base_logsumexp.shape != (batch, length):
            raise ValueError("base_logsumexp must have shape [B, L]")

        # Position zero has one real predecessor (the verified anchor).  It is
        # repeated K times only to keep a static [B, L, K, K] CUDA-friendly
        # shape; the decoder reads row zero there.
        anchor_rows = anchor_embedding[:, None, None, :].expand(
            batch, 1, candidates, dim
        )
        previous_rows = torch.cat(
            [anchor_rows, candidate_embeddings[:, :-1]], dim=1
        )

        left = self.prev_proj(previous_rows)
        right = self.next_proj(candidate_embeddings)
        gate = 1.0 + torch.tanh(self.context_gate(hidden))
        pairwise = torch.einsum(
            "blkr,bljr->blkj", left * gate[:, :, None, :], right
        ) / math.sqrt(self.rank)

        query = self.hidden_query(hidden)
        unary = torch.einsum("blr,blkr->blk", query, right)
        raw_residual = pairwise + unary[:, :, None, :]
        residual = self.residual_scale * raw_residual

        adjusted = candidate_base_logits[:, :, None, :] + residual

        base_topk_logmass = torch.logsumexp(candidate_base_logits, dim=-1)
        outside_log_mass = _log_sub_exp(base_logsumexp, base_topk_logmass)
        adjusted_topk_logmass = torch.logsumexp(adjusted, dim=-1)
        normalizer = torch.logaddexp(
            adjusted_topk_logmass, outside_log_mass[:, :, None]
        )
        log_probs = adjusted - normalizer[:, :, :, None]
        outside_log_probs = outside_log_mass[:, :, None] - normalizer

        return TransitionOutput(
            edge_scores=adjusted,
            log_probs=log_probs,
            residual_logits=residual,
            outside_log_mass=outside_log_mass,
            outside_log_probs=outside_log_probs,
        )


class BidirectionalBlockMixer(nn.Module):
    """One tiny full-block self-attention layer used only on draft features."""

    def __init__(
        self,
        hidden_size: int,
        model_dim: int = 64,
        num_heads: int = 4,
        max_positions: int = 32,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.max_positions = max_positions
        self.hidden_proj = nn.Linear(hidden_size, model_dim, bias=False)
        self.candidate_summary_proj = nn.Linear(hidden_size, model_dim, bias=False)
        self.position_embedding = nn.Parameter(
            torch.zeros(max_positions, model_dim)
        )
        self.input_norm = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.attention_out = nn.Linear(model_dim, model_dim, bias=False)
        self.output_norm = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 2 * model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * model_dim, model_dim, bias=False),
        )

    def forward(self, hidden: Tensor, candidate_summary: Tensor) -> Tensor:
        batch, length, _ = hidden.shape
        if length > self.max_positions:
            raise ValueError(
                f"block length {length} exceeds max_positions={self.max_positions}"
            )
        states = (
            self.hidden_proj(hidden)
            + self.candidate_summary_proj(candidate_summary)
            + self.position_embedding[:length]
        )
        normalized = self.input_norm(states)
        qkv = self.qkv(normalized).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        weights = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2))
            / math.sqrt(self.head_dim),
            dim=-1,
        )
        mixed = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch, length, self.model_dim
        )
        states = states + self.attention_out(mixed)
        states = states + self.feed_forward(self.output_norm(states))
        return states


class BidirectionalSurvivalPathHead(nn.Module):
    """Bidirectional-mixer capacity ablation for the pairwise scorer.

    The mixer sees the complete DFlash block and its top-K soft token summaries
    in one shot. Pairwise candidate transitions are then scored in parallel and
    locally normalized against the exact base outside mass for the local
    control.  The proposed distribution is obtained by passing ``edge_scores``,
    ``outside_log_mass``, and the full base partition to
    :func:`absorbing_prefix_crf_conditionals`.  DFlash features are already
    block-bidirectional, so this mixer is not the defining proposed component.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 32,
        model_dim: int = 64,
        num_heads: int = 4,
        max_positions: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.mixer = BidirectionalBlockMixer(
            hidden_size,
            model_dim=model_dim,
            num_heads=num_heads,
            max_positions=max_positions,
        )
        self.prev_proj = nn.Linear(hidden_size, rank, bias=False)
        self.next_proj = nn.Linear(hidden_size, rank, bias=False)
        self.context_gate = nn.Linear(model_dim, rank, bias=True)
        self.context_query = nn.Linear(model_dim, rank, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden: Tensor,
        anchor_embedding: Tensor,
        candidate_embeddings: Tensor,
        candidate_base_logits: Tensor,
        base_logsumexp: Tensor,
    ) -> TransitionOutput:
        if hidden.ndim != 3 or candidate_embeddings.ndim != 4:
            raise ValueError("invalid hidden or candidate embedding rank")
        batch, length, dim = hidden.shape
        cb, cl, candidates, embed_dim = candidate_embeddings.shape
        if (cb, cl, embed_dim) != (batch, length, dim):
            raise ValueError("candidate embedding shape is inconsistent with hidden")
        if anchor_embedding.shape != (batch, dim):
            raise ValueError("anchor_embedding must have shape [B, D]")
        if candidate_base_logits.shape != (batch, length, candidates):
            raise ValueError("candidate_base_logits must have shape [B, L, K]")
        if base_logsumexp.shape != (batch, length):
            raise ValueError("base_logsumexp must have shape [B, L]")

        topk_weights = torch.softmax(candidate_base_logits, dim=-1)
        candidate_summary = torch.einsum(
            "blk,blkd->bld", topk_weights, candidate_embeddings
        )
        context = self.mixer(hidden, candidate_summary)

        anchor_rows = anchor_embedding[:, None, None, :].expand(
            batch, 1, candidates, dim
        )
        previous_rows = torch.cat(
            [anchor_rows, candidate_embeddings[:, :-1]], dim=1
        )
        left = self.prev_proj(previous_rows)
        right = self.next_proj(candidate_embeddings)
        gate = 1.0 + torch.tanh(self.context_gate(context))
        pairwise = torch.einsum(
            "blkr,bljr->blkj", left * gate[:, :, None, :], right
        ) / math.sqrt(self.rank)
        query = self.context_query(context)
        unary = torch.einsum("blr,blkr->blk", query, right)
        residual = self.residual_scale * (pairwise + unary[:, :, None, :])
        adjusted = candidate_base_logits[:, :, None, :] + residual

        base_topk_logmass = torch.logsumexp(candidate_base_logits, dim=-1)
        outside_log_mass = _log_sub_exp(base_logsumexp, base_topk_logmass)
        adjusted_topk_logmass = torch.logsumexp(adjusted, dim=-1)
        normalizer = torch.logaddexp(
            adjusted_topk_logmass, outside_log_mass[:, :, None]
        )
        return TransitionOutput(
            edge_scores=adjusted,
            log_probs=adjusted - normalizer[:, :, :, None],
            residual_logits=residual,
            outside_log_mass=outside_log_mass,
            outside_log_probs=outside_log_mass[:, :, None] - normalizer,
        )


def prefix_censored_nll(
    log_probs: Tensor,
    outside_log_probs: Tensor,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
) -> Tensor:
    """NLL of the correct candidate prefix followed by an optional OTHER event.

    Positions after the first out-of-lattice gold token are unreachable under
    longest-prefix verification and are therefore not included in the loss.
    """

    if log_probs.ndim != 4:
        raise ValueError("log_probs must have shape [B, L, K, K]")
    batch, length, candidates, next_candidates = log_probs.shape
    if candidates != next_candidates:
        raise ValueError("candidate dimensions must match")
    if outside_log_probs.shape != (batch, length, candidates):
        raise ValueError("outside_log_probs must have shape [B, L, K]")
    if gold_candidate_indices.shape != (batch, length):
        raise ValueError("gold_candidate_indices must have shape [B, L]")
    if gold_in_lattice.shape != (batch, length):
        raise ValueError("gold_in_lattice must have shape [B, L]")

    batch_index = torch.arange(batch, device=log_probs.device)
    previous = torch.zeros(batch, dtype=torch.long, device=log_probs.device)
    alive = torch.ones(batch, dtype=torch.bool, device=log_probs.device)
    nll = torch.zeros(batch, dtype=log_probs.dtype, device=log_probs.device)
    for position in range(length):
        current = gold_candidate_indices[:, position].clamp(0, candidates - 1)
        candidate_log_prob = log_probs[
            batch_index, position, previous, current
        ]
        failure_log_prob = outside_log_probs[
            batch_index, position, previous
        ]
        in_lattice = gold_in_lattice[:, position]
        observed_log_prob = torch.where(
            in_lattice, candidate_log_prob, failure_log_prob
        )
        nll = nll - torch.where(alive, observed_log_prob, torch.zeros_like(nll))
        previous = torch.where(alive & in_lattice, current, previous)
        alive = alive & in_lattice
    return nll


def gold_prefix_survival_utility(
    log_probs: Tensor,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
) -> Tensor:
    """Predicted expected accepted length of the observed gold prefix.

    The contribution at position ``i`` is the predicted probability that every
    gold candidate through ``i`` is correct.  Once gold enters OTHER, later
    positions are unreachable and contribute zero.  This is the differentiable
    training counterpart of longest-prefix utility, not a replacement for the
    censored likelihood.
    """

    if log_probs.ndim != 4:
        raise ValueError("log_probs must have shape [B, L, K, K]")
    batch, length, candidates, next_candidates = log_probs.shape
    if candidates != next_candidates:
        raise ValueError("candidate dimensions must match")
    if gold_candidate_indices.shape != (batch, length):
        raise ValueError("gold_candidate_indices must have shape [B, L]")
    if gold_in_lattice.shape != (batch, length):
        raise ValueError("gold_in_lattice must have shape [B, L]")

    batch_index = torch.arange(batch, device=log_probs.device)
    previous = torch.zeros(batch, dtype=torch.long, device=log_probs.device)
    alive = torch.ones(batch, dtype=torch.bool, device=log_probs.device)
    log_survival = torch.zeros(
        batch, dtype=log_probs.dtype, device=log_probs.device
    )
    utility = torch.zeros_like(log_survival)
    for position in range(length):
        current = gold_candidate_indices[:, position].clamp(0, candidates - 1)
        selected_log_prob = log_probs[
            batch_index, position, previous, current
        ]
        alive = alive & gold_in_lattice[:, position]
        log_survival = log_survival + torch.where(
            alive, selected_log_prob, torch.zeros_like(selected_log_prob)
        )
        utility = utility + torch.where(
            alive, torch.exp(log_survival), torch.zeros_like(log_survival)
        )
        previous = torch.where(alive, current, previous)
    return utility


def gold_prefix_survival_loss(
    log_probs: Tensor,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
) -> Tensor:
    """Negative gold-prefix utility, returned per example."""

    return -gold_prefix_survival_utility(
        log_probs, gold_candidate_indices, gold_in_lattice
    )


def absorbing_prefix_crf_conditionals(
    edge_scores: Tensor,
    outside_log_mass: Tensor,
    base_logsumexp: Tensor,
) -> PrefixCRFOutput:
    """Globally normalize candidate prefixes with an absorbing OTHER state.

    The sample space contains every full candidate path and every candidate
    prefix followed by OTHER.  At position ``i`` a candidate edge has log
    weight

        edge_scores[i, previous, current] - base_logsumexp[i],

    while terminating in OTHER has log weight

        outside_log_mass[i] - base_logsumexp[i].

    Subtracting the full-vocabulary base partition is essential because paths
    may terminate at different positions.  It also gives the model an exact
    base-recovery invariant: when all learned residuals are zero, every suffix
    partition is one and the induced conditionals equal the original DFlash
    candidate probabilities plus its exact outside mass.

    Args:
        edge_scores: Raw candidate edge scores ``[B, L, K, K]``.  The base
            candidate logits should be included in these scores.
        outside_log_mass: Raw log mass of all non-candidate vocabulary items,
            either ``[B, L]`` (predecessor independent) or ``[B, L, K]``.
        base_logsumexp: Full-vocabulary base log-partition ``[B, L]``.
    """

    if edge_scores.ndim != 4:
        raise ValueError("edge_scores must have shape [B, L, K, K]")
    batch, length, candidates, next_candidates = edge_scores.shape
    if candidates != next_candidates:
        raise ValueError("the two candidate dimensions must match")
    if base_logsumexp.shape != (batch, length):
        raise ValueError("base_logsumexp must have shape [B, L]")
    if outside_log_mass.shape == (batch, length):
        outside_log_mass = outside_log_mass[:, :, None].expand(
            batch, length, candidates
        )
    elif outside_log_mass.shape != (batch, length, candidates):
        raise ValueError(
            "outside_log_mass must have shape [B, L] or [B, L, K]"
        )

    edge_log_weights = edge_scores - base_logsumexp[:, :, None, None]
    outside_log_weights = outside_log_mass - base_logsumexp[:, :, None]

    suffix_log_partition = torch.zeros(
        batch, candidates, dtype=edge_scores.dtype, device=edge_scores.device
    )
    candidate_conditionals: list[Tensor] = [torch.empty(0)] * length
    outside_conditionals: list[Tensor] = [torch.empty(0)] * length
    for position in range(length - 1, -1, -1):
        continued_scores = (
            edge_log_weights[:, position]
            + suffix_log_partition[:, None, :]
        )
        continued_log_mass = torch.logsumexp(continued_scores, dim=-1)
        state_log_partition = torch.logaddexp(
            outside_log_weights[:, position], continued_log_mass
        )
        candidate_conditionals[position] = (
            continued_scores - state_log_partition[:, :, None]
        )
        outside_conditionals[position] = (
            outside_log_weights[:, position] - state_log_partition
        )
        suffix_log_partition = state_log_partition

    return PrefixCRFOutput(
        log_conditionals=torch.stack(candidate_conditionals, dim=1),
        outside_log_conditionals=torch.stack(outside_conditionals, dim=1),
        log_partition=suffix_log_partition[:, 0],
    )


def absorbing_prefix_crf_nll(
    edge_scores: Tensor,
    outside_log_mass: Tensor,
    base_logsumexp: Tensor,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
) -> Tensor:
    """Censored NLL under the absorbing-OTHER globally normalized model."""

    crf = absorbing_prefix_crf_conditionals(
        edge_scores, outside_log_mass, base_logsumexp
    )
    return prefix_censored_nll(
        crf.log_conditionals,
        crf.outside_log_conditionals,
        gold_candidate_indices,
        gold_in_lattice,
    )


def chain_crf_conditionals(edge_scores: Tensor) -> ChainCRFOutput:
    """Convert candidate-only first-order energies into conditionals.

    The globally normalized candidate-lattice distribution is

        q(y | H) = exp(sum_i s_i(y_{i-1}, y_i)) / Z(H).

    A backward sum-product pass produces the exact causal conditionals of this
    distribution.  Although the neural edge scores are evaluated in parallel,
    each early conditional incorporates the partition function of every
    possible suffix.  This is the structured sense in which the head is
    bidirectional.  Because this ablation omits absolute failure probability,
    it is not the proposed prefix model and does not preserve DFlash when its
    learned residual is zero.
    """

    if edge_scores.ndim != 4:
        raise ValueError("edge_scores must have shape [B, L, K, K]")
    batch, length, candidates, next_candidates = edge_scores.shape
    if candidates != next_candidates:
        raise ValueError("the two candidate dimensions must match")

    suffix_log_partition = torch.zeros(
        batch, candidates, dtype=edge_scores.dtype, device=edge_scores.device
    )
    conditionals: list[Tensor] = [torch.empty(0)] * length
    for position in range(length - 1, -1, -1):
        suffix_scores = (
            edge_scores[:, position] + suffix_log_partition[:, None, :]
        )
        prefix_normalizer = torch.logsumexp(suffix_scores, dim=-1)
        conditionals[position] = suffix_scores - prefix_normalizer[:, :, None]
        suffix_log_partition = prefix_normalizer

    return ChainCRFOutput(
        log_conditionals=torch.stack(conditionals, dim=1),
        log_partition=suffix_log_partition[:, 0],
    )


def chain_crf_nll(edge_scores: Tensor, gold_path: Tensor) -> Tensor:
    """Per-example negative log-likelihood for an in-lattice gold path."""

    crf = chain_crf_conditionals(edge_scores)
    gold_score = path_log_probs(edge_scores, gold_path).sum(dim=1)
    return crf.log_partition - gold_score


def global_survival_decode(edge_scores: Tensor) -> DecodeOutput:
    """Bayes-risk decoder for the candidate-only CRF ablation."""

    crf = chain_crf_conditionals(edge_scores)
    return survival_decode(crf.log_conditionals)


def absorbing_prefix_survival_decode(
    edge_scores: Tensor,
    outside_log_mass: Tensor,
    base_logsumexp: Tensor,
) -> DecodeOutput:
    """Bayes-risk decoder for the proposed absorbing-OTHER prefix CRF."""

    crf = absorbing_prefix_crf_conditionals(
        edge_scores, outside_log_mass, base_logsumexp
    )
    return survival_decode(crf.log_conditionals)


def path_log_probs(log_probs: Tensor, path: Tensor) -> Tensor:
    """Return selected edge log probabilities with shape ``[B, L]``."""

    if log_probs.ndim != 4:
        raise ValueError("log_probs must have shape [B, L, K, K]")
    batch, length, candidates, next_candidates = log_probs.shape
    if candidates != next_candidates:
        raise ValueError("the two candidate dimensions must match")
    if path.shape != (batch, length):
        raise ValueError("path must have shape [B, L]")

    selected = []
    previous = torch.zeros(batch, dtype=torch.long, device=path.device)
    batch_index = torch.arange(batch, device=path.device)
    for position in range(length):
        current = path[:, position]
        selected.append(log_probs[batch_index, position, previous, current])
        previous = current
    return torch.stack(selected, dim=1)


def expected_prefix_utility(log_probs: Tensor, path: Tensor) -> Tensor:
    """Predicted expected accepted draft length for a fixed path."""

    selected_log_probs = path_log_probs(log_probs, path)
    prefix_survival = torch.exp(torch.cumsum(selected_log_probs, dim=1))
    return prefix_survival.sum(dim=1)


def survival_decode(log_probs: Tensor) -> DecodeOutput:
    """Exactly maximize predicted expected accepted length.

    Bellman recurrence, with ``V_{L+1}=0``:

        V_i(u) = max_v p_i(v | u) * (1 + V_{i+1}(v)).

    It is acceptance-aligned.  In contrast, ordinary Viterbi maximizes the
    probability that the *entire* block is correct.
    """

    if log_probs.ndim != 4:
        raise ValueError("log_probs must have shape [B, L, K, K]")
    batch, length, candidates, next_candidates = log_probs.shape
    if candidates != next_candidates:
        raise ValueError("the two candidate dimensions must match")

    future_value = torch.zeros(
        batch, candidates, dtype=log_probs.dtype, device=log_probs.device
    )
    backpointers: list[Tensor] = [torch.empty(0)] * length

    for position in range(length - 1, -1, -1):
        probability = torch.exp(log_probs[:, position])
        action_value = probability * (1.0 + future_value[:, None, :])
        future_value, backpointers[position] = action_value.max(dim=-1)

    path = torch.empty(
        batch, length, dtype=torch.long, device=log_probs.device
    )
    batch_index = torch.arange(batch, device=log_probs.device)
    previous = torch.zeros(batch, dtype=torch.long, device=log_probs.device)
    for position in range(length):
        current = backpointers[position][batch_index, previous]
        path[:, position] = current
        previous = current

    predicted_utility = future_value[:, 0]
    return DecodeOutput(path=path, predicted_utility=predicted_utility)


def greedy_markov_decode(log_probs: Tensor) -> DecodeOutput:
    """Left-to-right local argmax control, analogous to a Markov head."""

    batch, length, _, _ = log_probs.shape
    path = torch.empty(batch, length, dtype=torch.long, device=log_probs.device)
    batch_index = torch.arange(batch, device=log_probs.device)
    previous = torch.zeros(batch, dtype=torch.long, device=log_probs.device)
    for position in range(length):
        current = log_probs[batch_index, position, previous].argmax(dim=-1)
        path[:, position] = current
        previous = current
    return DecodeOutput(
        path=path,
        predicted_utility=expected_prefix_utility(log_probs, path),
    )


def viterbi_decode(log_probs: Tensor) -> DecodeOutput:
    """Maximum full-block probability control (not the proposed objective)."""

    batch, length, candidates, _ = log_probs.shape
    score = log_probs[:, 0, 0]
    backpointers: list[Tensor] = []
    for position in range(1, length):
        edge_score = score[:, :, None] + log_probs[:, position]
        score, pointer = edge_score.max(dim=1)
        backpointers.append(pointer)

    last = score.argmax(dim=-1)
    reversed_path = [last]
    batch_index = torch.arange(batch, device=log_probs.device)
    for pointer in reversed(backpointers):
        last = pointer[batch_index, last]
        reversed_path.append(last)
    path = torch.stack(list(reversed(reversed_path)), dim=1)
    return DecodeOutput(
        path=path,
        predicted_utility=expected_prefix_utility(log_probs, path),
    )
