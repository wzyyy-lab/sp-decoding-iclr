"""Full-block non-causal candidate fusion for one parallel DFlash chain.

The production head consumes all 16 draft positions and all 16 candidates per
position at once.  It has no recurrent state, selected-token input, path
decoder, target-model feature, or multi-proposal output.  A single invocation
returns one score tensor with shape ``[B, 16, 16]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


BLOCK_LENGTH = 16
CANDIDATES = 16
DEFAULT_HIDDEN_SIZE = 2560
DEFAULT_MODEL_DIM = 256
DEFAULT_HEADS = 8
DEFAULT_LAYERS = 2
DEFAULT_FF_MULTIPLIER = 2
DEFAULT_PARAMETER_COUNT = 2_438_400


@dataclass
class PGCFOutput:
    """One parallel score tensor and its continuous candidate states."""

    scores: Tensor
    residual_scores: Tensor
    candidate_states: Tensor


@dataclass
class PGCFLossOutput:
    """Verifier-aligned training losses, kept outside the online head."""

    loss: Tensor
    prefix_loss: Tensor
    target_kl_loss: Tensor
    teacher_loss: Tensor
    gold_support: Tensor
    target_kl_positions: Tensor
    teacher_positions: Tensor
    lambda_prefix: float
    lambda_target_kl: float
    lambda_teacher: float


def rms_normalize(value: Tensor, eps: float = 1e-6) -> Tensor:
    """Parameter-free RMS normalization in input precision."""

    scale = value.float().square().mean(dim=-1, keepdim=True)
    normalized = value.float() * torch.rsqrt(scale + eps)
    return normalized.to(value.dtype)


class FullCandidateFusionBlock(nn.Module):
    """One full-node attention block or its parameter-matched local control."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        ff_multiplier: int = DEFAULT_FF_MULTIPLIER,
        local_control: bool = False,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if ff_multiplier < 1:
            raise ValueError("ff_multiplier must be positive")
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = model_dim // num_heads
        self.local_control = bool(local_control)

        self.attention_norm = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.attention_out = nn.Linear(model_dim, model_dim, bias=False)
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.feed_forward_up = nn.Linear(
            model_dim, ff_multiplier * model_dim, bias=False
        )
        self.feed_forward_down = nn.Linear(
            ff_multiplier * model_dim, model_dim, bias=False
        )
        self.relative_position_bias = nn.Embedding(
            2 * BLOCK_LENGTH - 1, num_heads
        )
        self.same_position_bias = nn.Parameter(torch.zeros(num_heads))
        node_positions = torch.arange(BLOCK_LENGTH).repeat_interleave(
            CANDIDATES
        )
        self.register_buffer(
            "relative_position_index",
            node_positions[:, None]
            - node_positions[None, :]
            + BLOCK_LENGTH
            - 1,
            persistent=False,
        )
        self.register_buffer(
            "same_position_mask",
            node_positions[:, None].eq(node_positions[None, :]),
            persistent=False,
        )
        nn.init.zeros_(self.relative_position_bias.weight)

    def _attention_bias(self, dtype: torch.dtype) -> Tensor:
        bias = self.relative_position_bias(
            self.relative_position_index
        ).permute(2, 0, 1)
        bias = bias + (
            self.same_position_bias[:, None, None]
            * self.same_position_mask[None].to(
                self.same_position_bias.dtype
            )
        )
        if self.local_control:
            bias = bias.masked_fill(
                ~self.same_position_mask[None],
                torch.finfo(bias.dtype).min,
            )
        return bias.to(dtype=dtype)

    def forward(self, states: Tensor) -> Tensor:
        batch, nodes, width = states.shape
        if nodes != BLOCK_LENGTH * CANDIDATES or width != self.model_dim:
            raise ValueError(
                "candidate states must have shape "
                f"[B,{BLOCK_LENGTH * CANDIDATES},{self.model_dim}]"
            )
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
        attention_scores = attention_scores + self._attention_bias(
            attention_scores.dtype
        )[None]
        attention = torch.softmax(
            attention_scores.float(), dim=-1
        ).to(value.dtype)
        mixed = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch, nodes, self.model_dim
        )
        states = states + self.attention_out(mixed)
        feed_forward = self.feed_forward_down(
            F.silu(self.feed_forward_up(self.feed_forward_norm(states)))
        )
        return states + feed_forward


class _CandidateFusionHead(nn.Module):
    """Shared implementation for the production global head and local control."""

    NUM_SCALAR_FEATURES = 5

    def __init__(
        self,
        *,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        model_dim: int = DEFAULT_MODEL_DIM,
        num_heads: int = DEFAULT_HEADS,
        num_layers: int = DEFAULT_LAYERS,
        ff_multiplier: int = DEFAULT_FF_MULTIPLIER,
        local_control: bool,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.hidden_size = int(hidden_size)
        self.model_dim = int(model_dim)
        self.local_control = bool(local_control)

        self.hidden_projection = nn.Linear(
            hidden_size, model_dim, bias=False
        )
        self.token_projection = nn.Linear(
            hidden_size, model_dim, bias=False
        )
        self.position_embedding = nn.Embedding(BLOCK_LENGTH, model_dim)
        self.rank_embedding = nn.Embedding(CANDIDATES, model_dim)
        self.scalar_projection = nn.Linear(
            self.NUM_SCALAR_FEATURES, model_dim, bias=True
        )
        self.compatibility_projection = nn.Linear(
            model_dim, model_dim, bias=False
        )
        self.input_norm = nn.LayerNorm(model_dim)
        self.blocks = nn.ModuleList(
            [
                FullCandidateFusionBlock(
                    model_dim,
                    num_heads,
                    ff_multiplier=ff_multiplier,
                    local_control=local_control,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.residual_projection = nn.Linear(model_dim, 1, bias=False)
        self.register_buffer(
            "normalized_candidate_rank",
            torch.linspace(0.0, 1.0, CANDIDATES).view(
                1, 1, CANDIDATES
            ),
            persistent=False,
        )
        nn.init.zeros_(self.residual_projection.weight)

    @staticmethod
    def _validate_online_inputs(
        hidden: Tensor,
        candidate_logits: Tensor,
        anchor_embeddings: Tensor,
        candidate_embeddings: Tensor | None,
        projected_candidate_embeddings: Tensor | None,
        hidden_size: int,
        model_dim: int,
    ) -> int:
        if hidden.ndim != 3 or hidden.shape[1:] != (
            BLOCK_LENGTH,
            hidden_size,
        ):
            raise ValueError(
                f"hidden must have shape [B,{BLOCK_LENGTH},{hidden_size}]"
            )
        batch = hidden.shape[0]
        if candidate_logits.shape != (
            batch,
            BLOCK_LENGTH,
            CANDIDATES,
        ):
            raise ValueError("candidate_logits must have shape [B,16,16]")
        if anchor_embeddings.shape != (batch, hidden_size):
            raise ValueError(
                f"anchor_embeddings must have shape [B,{hidden_size}]"
            )
        if (candidate_embeddings is None) == (
            projected_candidate_embeddings is None
        ):
            raise ValueError(
                "provide exactly one of candidate_embeddings or "
                "projected_candidate_embeddings"
            )
        if candidate_embeddings is not None and candidate_embeddings.shape != (
            batch,
            BLOCK_LENGTH,
            CANDIDATES,
            hidden_size,
        ):
            raise ValueError(
                "candidate_embeddings must have shape [B,16,16,hidden_size]"
            )
        if (
            projected_candidate_embeddings is not None
            and projected_candidate_embeddings.shape
            != (batch, BLOCK_LENGTH, CANDIDATES, model_dim)
        ):
            raise ValueError(
                "projected_candidate_embeddings must have shape "
                "[B,16,16,model_dim]"
            )
        return batch

    def _scalar_features(self, candidate_logits: Tensor) -> Tensor:
        logits = candidate_logits.detach().float()
        conditional_log_probs = torch.log_softmax(logits, dim=-1)
        maximum = logits.amax(dim=-1, keepdim=True)
        centered = logits - logits.mean(dim=-1, keepdim=True)
        gap = maximum - logits
        probabilities = conditional_log_probs.exp()
        entropy = -(
            probabilities * conditional_log_probs
        ).sum(dim=-1, keepdim=True) / math.log(CANDIDATES)
        normalized_rank = self.normalized_candidate_rank.to(
            dtype=logits.dtype
        ).expand_as(logits)
        return torch.stack(
            [
                torch.tanh(centered / 8.0),
                torch.tanh(conditional_log_probs / 8.0),
                torch.tanh(gap / 8.0),
                normalized_rank,
                entropy.expand_as(logits),
            ],
            dim=-1,
        )

    @torch.no_grad()
    def project_vocabulary(
        self, embedding: Tensor, *, chunk_size: int = 4096
    ) -> Tensor:
        """Build the derived frozen inference table after training."""

        if embedding.ndim != 2 or embedding.shape[1] != self.hidden_size:
            raise ValueError("embedding must have shape [V, hidden_size]")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        projected = []
        for start in range(0, embedding.shape[0], chunk_size):
            chunk = embedding[start : start + chunk_size]
            projected.append(self.token_projection(rms_normalize(chunk)))
        return torch.cat(projected, dim=0)

    def encode_candidates(
        self,
        hidden: Tensor,
        candidate_logits: Tensor,
        anchor_embeddings: Tensor,
        *,
        candidate_embeddings: Tensor | None = None,
        projected_candidate_embeddings: Tensor | None = None,
    ) -> Tensor:
        batch = self._validate_online_inputs(
            hidden,
            candidate_logits,
            anchor_embeddings,
            candidate_embeddings,
            projected_candidate_embeddings,
            self.hidden_size,
            self.model_dim,
        )
        hidden_state = self.hidden_projection(rms_normalize(hidden))
        anchor_state = self.token_projection(
            rms_normalize(anchor_embeddings)
        )[:, None, :]
        query_state = hidden_state + anchor_state
        if projected_candidate_embeddings is None:
            assert candidate_embeddings is not None
            candidate_state = self.token_projection(
                rms_normalize(candidate_embeddings)
            )
        else:
            candidate_state = projected_candidate_embeddings

        query_nodes = query_state[:, :, None, :]
        compatibility = self.compatibility_projection(
            query_nodes * candidate_state
        )
        scalar_state = self.scalar_projection(
            self._scalar_features(candidate_logits).to(hidden.dtype)
        )
        positions = self.position_embedding.weight.to(query_state.dtype)[
            None, :, None, :
        ]
        ranks = self.rank_embedding.weight.to(query_state.dtype)[
            None, None, :, :
        ]
        nodes = self.input_norm(
            query_nodes
            + candidate_state
            + compatibility
            + scalar_state
            + positions
            + ranks
        )
        return nodes.reshape(
            batch, BLOCK_LENGTH * CANDIDATES, self.model_dim
        )

    def forward(
        self,
        hidden: Tensor,
        candidate_logits: Tensor,
        anchor_embeddings: Tensor,
        *,
        candidate_embeddings: Tensor | None = None,
        projected_candidate_embeddings: Tensor | None = None,
    ) -> PGCFOutput:
        states = self.encode_candidates(
            hidden,
            candidate_logits,
            anchor_embeddings,
            candidate_embeddings=candidate_embeddings,
            projected_candidate_embeddings=projected_candidate_embeddings,
        )
        for block in self.blocks:
            states = block(states)
        candidate_states = states.reshape(
            hidden.shape[0], BLOCK_LENGTH, CANDIDATES, self.model_dim
        )
        residual = self.residual_projection(
            self.output_norm(candidate_states)
        ).squeeze(-1)
        scores = candidate_logits.float() + residual.float()
        return PGCFOutput(
            scores=scores,
            residual_scores=residual,
            candidate_states=candidate_states,
        )

    @staticmethod
    def proposal_candidate_indices(output: PGCFOutput) -> Tensor:
        """One tensor argmax for all 16 positions."""

        return output.scores.argmax(dim=-1)


class ParallelGlobalCandidateFusionHead(_CandidateFusionHead):
    """Claim-bearing full16 global head; cross-position visibility is fixed."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(local_control=False, **kwargs)


class MatchedLocalCandidateFusionHead(_CandidateFusionHead):
    """Parameter-matched no-cross-position diagnostic control."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(local_control=True, **kwargs)


def pgcf_loss_weights(progress: float) -> tuple[float, float, float]:
    """Return prefix, target-KL, and teacher weights for frozen curriculum."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    if progress < 0.10:
        return 0.0, 0.0, 1.0
    if progress < 0.30:
        transition = (progress - 0.10) / 0.20
        return transition, 0.05 * transition, 1.0 - transition
    return 1.0, 0.05, 0.0


def supported_candidate_cross_entropy(
    scores: Tensor, candidate_ranks: Tensor
) -> tuple[Tensor, Tensor]:
    """Mean CE over supported candidate ranks, with safe pre-gather ranks."""

    support = candidate_ranks.ge(0)
    safe_ranks = torch.where(
        support, candidate_ranks, torch.zeros_like(candidate_ranks)
    )
    log_probabilities = torch.log_softmax(scores.float(), dim=-1)
    selected = log_probabilities.gather(
        -1, safe_ranks.unsqueeze(-1)
    ).squeeze(-1)
    numerator = torch.where(
        support, -selected, torch.zeros_like(selected)
    ).sum()
    denominator = support.sum().clamp_min(1).float()
    return numerator / denominator, support


def parallel_prefix_utility_loss(
    scores: Tensor, gold_candidate_ranks: Tensor
) -> tuple[Tensor, Tensor]:
    """Negative expected accepted-draft fraction with safe out-of-K rows."""

    if scores.shape != (
        gold_candidate_ranks.shape[0],
        BLOCK_LENGTH,
        CANDIDATES,
    ):
        raise ValueError("scores and gold_candidate_ranks have invalid shapes")
    support = gold_candidate_ranks.ge(0)
    safe_ranks = torch.where(
        support,
        gold_candidate_ranks,
        torch.zeros_like(gold_candidate_ranks),
    )
    log_probabilities = torch.log_softmax(scores.float(), dim=-1)
    selected = log_probabilities.gather(
        -1, safe_ranks.unsqueeze(-1)
    ).squeeze(-1)
    log_gold = torch.where(support, selected, torch.zeros_like(selected))
    support_prefix = torch.cumprod(support.float(), dim=-1)
    log_survival = torch.cumsum(log_gold, dim=-1)
    utility = (
        support_prefix * torch.exp(log_survival)
    ).sum(dim=-1)
    return -utility.mean() / BLOCK_LENGTH, support


def target_candidate_kl_loss(
    scores: Tensor,
    target_candidate_logits: Tensor,
    gold_support: Tensor,
    target_matches_gold: Tensor,
) -> tuple[Tensor, Tensor]:
    """KL(p_target || q_head) on the clean teacher-forced prefix only."""

    if target_candidate_logits.shape != scores.shape:
        raise ValueError("target candidate logits must match score shape")
    if gold_support.shape != scores.shape[:2]:
        raise ValueError("gold_support has an invalid shape")
    if target_matches_gold.shape != scores.shape[:2]:
        raise ValueError("target_matches_gold has an invalid shape")
    clean_prefix = torch.cumprod(
        target_matches_gold.float(), dim=-1
    ).bool()
    valid = gold_support & clean_prefix
    target_log_prob = torch.log_softmax(
        target_candidate_logits.float(), dim=-1
    )
    target_probability = target_log_prob.exp()
    student_log_prob = torch.log_softmax(scores.float(), dim=-1)
    row_kl = (
        target_probability * (target_log_prob - student_log_prob)
    ).sum(dim=-1)
    numerator = torch.where(
        valid, row_kl, torch.zeros_like(row_kl)
    ).sum()
    denominator = valid.sum().clamp_min(1).float()
    return numerator / denominator, valid


def pgcf_training_loss(
    output: PGCFOutput,
    gold_candidate_ranks: Tensor,
    *,
    progress: float,
    target_candidate_logits: Tensor | None = None,
    target_matches_gold: Tensor | None = None,
    teacher_candidate_ranks: Tensor | None = None,
) -> PGCFLossOutput:
    """Frozen three-stage objective; labels never enter the online head."""

    prefix_loss, gold_support = parallel_prefix_utility_loss(
        output.scores, gold_candidate_ranks
    )
    zero = output.scores.float().sum() * 0.0
    if target_candidate_logits is None or target_matches_gold is None:
        if not (
            target_candidate_logits is None and target_matches_gold is None
        ):
            raise ValueError(
                "target logits and target_matches_gold must be provided together"
            )
        target_kl = zero
        target_positions = torch.zeros_like(gold_support)
    else:
        target_kl, target_positions = target_candidate_kl_loss(
            output.scores,
            target_candidate_logits,
            gold_support,
            target_matches_gold,
        )
    if teacher_candidate_ranks is None:
        teacher_loss = zero
        teacher_positions = torch.zeros_like(gold_support)
    else:
        teacher_loss, teacher_positions = supported_candidate_cross_entropy(
            output.scores, teacher_candidate_ranks
        )

    lambda_prefix, lambda_kl, lambda_teacher = pgcf_loss_weights(progress)
    total = (
        lambda_prefix * prefix_loss
        + lambda_kl * target_kl
        + lambda_teacher * teacher_loss
    )
    return PGCFLossOutput(
        loss=total,
        prefix_loss=prefix_loss,
        target_kl_loss=target_kl,
        teacher_loss=teacher_loss,
        gold_support=gold_support,
        target_kl_positions=target_positions,
        teacher_positions=teacher_positions,
        lambda_prefix=lambda_prefix,
        lambda_target_kl=lambda_kl,
        lambda_teacher=lambda_teacher,
    )
