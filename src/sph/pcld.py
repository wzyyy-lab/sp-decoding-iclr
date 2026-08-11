"""PCLD-16R: full-block predictive clean-latent distillation.

The production head consumes one complete 16-position DFlash block, performs
global non-causal mixing over the fixed 16x16 candidate lattice, and returns
one simultaneous score tensor.  It contains no selected-token feedback,
recurrent state, decoding loop, target-model feature, or multi-path output.
Target hidden states and teacher logits appear only in the offline loss helpers
below; they are intentionally absent from :meth:`PCLD16Head.forward`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


BLOCK_LENGTH = 16
CANDIDATES = 16
HIDDEN_SIZE = 2560
MODEL_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 2
FFN_DIM = 1024
NUM_SCALAR_FEATURES = 5
EXPECTED_PARAMETER_COUNT = 3_826_688
RMS_EPSILON = 1e-6
LATENT_SCALE_FLOOR = 1e-3
SAFE_MARGIN = 0.1
SAFE_TEMPERATURE = 0.1
KL_TEMPERATURE = 2.0
KL_WEIGHT = 0.1


@dataclass(frozen=True)
class PCLDOutput:
    """One full16, one-chain PCLD prediction."""

    scores: Tensor
    corrections: Tensor
    predicted_residual: Tensor
    global_states: Tensor
    base_scores: Tensor


@dataclass(frozen=True)
class PCLDLossOutput:
    """Per-block losses before prompt-balanced sampling reduction."""

    per_block_loss: Tensor
    safe_loss: Tensor
    latent_loss: Tensor
    candidate_kl: Tensor
    support_mask: Tensor
    horizons: Tensor
    normalized_gold_margins: Tensor
    latent_alpha: float


def parameter_free_rms_norm(value: Tensor, *, epsilon: float = RMS_EPSILON) -> Tensor:
    """RMS-normalize the final dimension without learned parameters."""

    if value.ndim < 1:
        raise ValueError("RMS normalization requires at least one dimension")
    if epsilon <= 0:
        raise ValueError("RMS epsilon must be positive")
    value_float = value.float()
    inverse_rms = torch.rsqrt(value_float.square().mean(dim=-1, keepdim=True) + epsilon)
    return value_float * inverse_rms


def pcld_scalar_features(
    candidate_logits: Tensor, base_logsumexp: Tensor
) -> Tensor:
    """Return the frozen five bounded base-lattice scalar channels."""

    if candidate_logits.ndim != 3:
        raise ValueError("candidate_logits must have shape [B,16,16]")
    if base_logsumexp.shape != candidate_logits.shape[:2]:
        raise ValueError("base_logsumexp must have shape [B,16]")
    logits = candidate_logits.detach().float()
    lse = base_logsumexp.detach().float()
    full_log_probability = logits - lse.unsqueeze(-1)
    conditional_log_probability = torch.log_softmax(logits, dim=-1)
    rank_one_gap = logits.amax(dim=-1, keepdim=True) - logits
    retained_log_mass = torch.logsumexp(logits, dim=-1) - lse
    conditional_probability = conditional_log_probability.exp()
    conditional_entropy = -(
        conditional_probability * conditional_log_probability
    ).sum(dim=-1)
    log_candidates = math.log(CANDIDATES)
    return torch.stack(
        [
            torch.tanh(full_log_probability / 8.0),
            torch.tanh(conditional_log_probability / 8.0),
            torch.tanh(rank_one_gap / 8.0),
            torch.tanh(retained_log_mass.unsqueeze(-1) / 2.0).expand_as(logits),
            (conditional_entropy.unsqueeze(-1) / log_candidates)
            .clamp(0.0, 1.0)
            .expand_as(logits),
        ],
        dim=-1,
    )


class PCLDEncoderBlock(nn.Module):
    """Frozen pre-norm D256/H8/FFN1024 encoder block."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(MODEL_DIM)
        self.attention = nn.MultiheadAttention(
            MODEL_DIM,
            NUM_HEADS,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(MODEL_DIM)
        self.feed_forward = nn.Sequential(
            nn.Linear(MODEL_DIM, FFN_DIM, bias=True),
            nn.GELU(),
            nn.Linear(FFN_DIM, MODEL_DIM, bias=True),
        )

    def forward(self, states: Tensor, *, attention_mask: Tensor | None) -> Tensor:
        normalized = self.attention_norm(states)
        mixed, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        states = states + mixed
        return states + self.feed_forward(self.feed_forward_norm(states))


class PCLD16Head(nn.Module):
    """One-shot global non-causal head that emits exactly one full16 chain.

    ``scope='local'`` exists only as the parameter-matched offline control.
    Production checkpoint loaders must require ``scope='global'``.
    """

    VALID_SCOPES = frozenset({"global", "local"})

    def __init__(self, *, scope: str = "global") -> None:
        super().__init__()
        if scope not in self.VALID_SCOPES:
            raise ValueError("PCLD scope must be 'global' or matched-control 'local'")
        self.scope = scope
        self.hidden_projection = nn.Linear(HIDDEN_SIZE, MODEL_DIM, bias=True)
        self.lexical_projection = nn.Linear(HIDDEN_SIZE, MODEL_DIM, bias=True)
        self.scalar_projection = nn.Linear(
            NUM_SCALAR_FEATURES, MODEL_DIM, bias=True
        )
        self.position_embedding = nn.Embedding(BLOCK_LENGTH, MODEL_DIM)
        self.rank_embedding = nn.Embedding(CANDIDATES, MODEL_DIM)
        self.node_input_norm = nn.LayerNorm(MODEL_DIM)
        self.encoder_blocks = nn.ModuleList(
            [PCLDEncoderBlock() for _ in range(NUM_LAYERS)]
        )
        self.position_queries = nn.Parameter(torch.empty(BLOCK_LENGTH, MODEL_DIM))
        self.query_attention = nn.MultiheadAttention(
            MODEL_DIM,
            NUM_HEADS,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        self.query_output_norm = nn.LayerNorm(MODEL_DIM)
        self.residual_projection = nn.Linear(MODEL_DIM, HIDDEN_SIZE, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.rank_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_queries, mean=0.0, std=0.02)
        # Exact pure-DFlash fallback.  Both terms must be zero.
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    @staticmethod
    def _node_positions(device: torch.device) -> Tensor:
        return torch.arange(BLOCK_LENGTH, device=device).repeat_interleave(CANDIDATES)

    def _encoder_mask(self, device: torch.device) -> Tensor | None:
        if self.scope == "global":
            return None
        positions = self._node_positions(device)
        return positions[:, None].ne(positions[None, :])

    def _query_mask(self, device: torch.device) -> Tensor | None:
        if self.scope == "global":
            return None
        node_positions = self._node_positions(device)
        query_positions = torch.arange(BLOCK_LENGTH, device=device)
        return query_positions[:, None].ne(node_positions[None, :])

    @staticmethod
    def _validate_inputs(
        hidden: Tensor,
        candidate_lm_rows: Tensor,
        candidate_logits: Tensor,
        base_logsumexp: Tensor,
    ) -> int:
        if hidden.ndim != 3 or hidden.shape[1:] != (BLOCK_LENGTH, HIDDEN_SIZE):
            raise ValueError("hidden must have shape [B,16,2560]")
        batch = hidden.shape[0]
        if candidate_lm_rows.shape != (
            batch,
            BLOCK_LENGTH,
            CANDIDATES,
            HIDDEN_SIZE,
        ):
            raise ValueError("candidate_lm_rows must have shape [B,16,16,2560]")
        if candidate_logits.shape != (batch, BLOCK_LENGTH, CANDIDATES):
            raise ValueError("candidate_logits must have shape [B,16,16]")
        if base_logsumexp.shape != (batch, BLOCK_LENGTH):
            raise ValueError("base_logsumexp must have shape [B,16]")
        return batch

    def forward(
        self,
        hidden: Tensor,
        candidate_lm_rows: Tensor,
        candidate_logits: Tensor,
        base_logsumexp: Tensor,
    ) -> PCLDOutput:
        """Score all 16 positions simultaneously from online-available inputs."""

        batch = self._validate_inputs(
            hidden, candidate_lm_rows, candidate_logits, base_logsumexp
        )
        # The frozen DFlash backbone and LM-head rows are never trainable here.
        hidden = hidden.detach()
        candidate_lm_rows = candidate_lm_rows.detach()
        candidate_logits = candidate_logits.detach()
        base_logsumexp = base_logsumexp.detach()

        hidden_nodes = self.hidden_projection(parameter_free_rms_norm(hidden))[
            :, :, None, :
        ]
        lexical_nodes = self.lexical_projection(
            parameter_free_rms_norm(candidate_lm_rows)
        )
        scalars = self.scalar_projection(
            pcld_scalar_features(candidate_logits, base_logsumexp)
        )
        position_ids = torch.arange(BLOCK_LENGTH, device=hidden.device)[
            None, :, None
        ]
        rank_ids = torch.arange(CANDIDATES, device=hidden.device)[None, None, :]
        nodes = self.node_input_norm(
            hidden_nodes
            + lexical_nodes
            + scalars
            + self.position_embedding(position_ids)
            + self.rank_embedding(rank_ids)
        ).reshape(batch, BLOCK_LENGTH * CANDIDATES, MODEL_DIM)

        encoder_mask = self._encoder_mask(hidden.device)
        for block in self.encoder_blocks:
            nodes = block(nodes, attention_mask=encoder_mask)

        queries = self.position_queries.unsqueeze(0).expand(batch, -1, -1)
        query_mixed, _ = self.query_attention(
            queries,
            nodes,
            nodes,
            attn_mask=self._query_mask(hidden.device),
            need_weights=False,
        )
        global_states = self.query_output_norm(queries + query_mixed)
        predicted_residual = self.residual_projection(global_states)
        corrections = torch.einsum(
            "blkh,blh->blk",
            candidate_lm_rows.float(),
            predicted_residual.float(),
        )
        base_scores = candidate_logits.float()
        scores = base_scores + corrections
        return PCLDOutput(
            scores=scores,
            corrections=corrections,
            predicted_residual=predicted_residual,
            global_states=global_states,
            base_scores=base_scores,
        )

    def proposal_ids(self, candidate_ids: Tensor, output: PCLDOutput) -> Tensor:
        """Materialize the unique one-chain proposal with one tensor argmax."""

        if candidate_ids.shape != output.scores.shape:
            raise ValueError("candidate_ids must match the PCLD score tensor")
        ranks = output.scores.argmax(dim=-1)
        return candidate_ids.gather(-1, ranks.unsqueeze(-1)).squeeze(-1)


def candidate_gold_ranks(candidate_ids: Tensor, gold_ids: Tensor) -> Tensor:
    """Unique candidate rank for every gold token, or -1 outside Top16."""

    if candidate_ids.ndim != 3 or candidate_ids.shape[1:] != (
        BLOCK_LENGTH,
        CANDIDATES,
    ):
        raise ValueError("candidate_ids must have shape [B,16,16]")
    if gold_ids.shape != candidate_ids.shape[:2]:
        raise ValueError("gold_ids must have shape [B,16]")
    sorted_ids = candidate_ids.sort(dim=-1).values
    if bool(sorted_ids[..., 1:].eq(sorted_ids[..., :-1]).any().item()):
        raise ValueError("each Top16 candidate row must contain unique IDs")
    matches = candidate_ids.eq(gold_ids.unsqueeze(-1))
    ranks = matches.to(torch.int64).argmax(dim=-1)
    return torch.where(matches.any(dim=-1), ranks, torch.full_like(ranks, -1))


def calibrate_numeric_epsilon(
    authoritative_top1_ids: Tensor,
    fp32_top1_ids: Tensor,
    centered_max_errors: Tensor,
) -> Tensor:
    """Calibrate epsilon only from train rows with agreeing full-vocab top1."""

    if authoritative_top1_ids.shape != fp32_top1_ids.shape:
        raise ValueError("numeric top1 tensors must have equal shape")
    if centered_max_errors.shape != authoritative_top1_ids.shape:
        raise ValueError("centered errors must match numeric top1 tensors")
    agreeing = authoritative_top1_ids.eq(fp32_top1_ids)
    finite = torch.isfinite(centered_max_errors)
    eligible = agreeing & finite
    if not bool(eligible.any().item()):
        raise RuntimeError("no agreeing finite rows exist for epsilon calibration")
    return centered_max_errors.float()[eligible].max()


def stable_teacher_rows(
    authoritative_top1_ids: Tensor,
    fp32_top1_ids: Tensor,
    target_top1_margins: Tensor,
    epsilon_num: Tensor | float,
) -> Tensor:
    """Frozen numerical support: agreeing top1 and margin greater than 2 epsilon."""

    if authoritative_top1_ids.shape != fp32_top1_ids.shape:
        raise ValueError("numeric top1 tensors must have equal shape")
    if target_top1_margins.shape != authoritative_top1_ids.shape:
        raise ValueError("target margins must match numeric top1 tensors")
    epsilon = torch.as_tensor(
        epsilon_num,
        device=target_top1_margins.device,
        dtype=torch.float32,
    )
    if epsilon.ndim != 0 or not bool(torch.isfinite(epsilon).item()) or epsilon < 0:
        raise ValueError("epsilon_num must be one finite non-negative scalar")
    return authoritative_top1_ids.eq(fp32_top1_ids) & target_top1_margins.float().gt(
        2.0 * epsilon
    )


def continuous_clean_support(
    gold_candidate_ranks: Tensor,
    target_top1_ids: Tensor,
    gold_ids: Tensor,
    stable_rows: Tensor,
) -> tuple[Tensor, Tensor]:
    """One shared continuous clean-prefix support for every PCLD loss."""

    expected = gold_candidate_ranks.shape
    if expected != target_top1_ids.shape or expected != gold_ids.shape:
        raise ValueError("rank, target top1 and gold tensors must have equal shape")
    if stable_rows.shape != expected:
        raise ValueError("stable_rows must match the full16 label tensors")
    if gold_candidate_ranks.ndim != 2 or gold_candidate_ranks.shape[1] != BLOCK_LENGTH:
        raise ValueError("PCLD support tensors must have shape [B,16]")
    row_valid = (
        gold_candidate_ranks.ge(0)
        & target_top1_ids.eq(gold_ids)
        & stable_rows.bool()
    )
    support = row_valid.to(torch.int64).cumprod(dim=-1).to(torch.bool)
    return support, support.sum(dim=-1).to(torch.long)


def latent_alpha(step: int, total_steps: int) -> float:
    """Frozen 1.0 -> 0.1 schedule over the first 30% of updates."""

    if total_steps < 1 or step < 0 or step > total_steps:
        raise ValueError("latent schedule step must lie in [0,total_steps]")
    transition_steps = max(1, math.ceil(0.30 * total_steps))
    progress = min(1.0, step / transition_steps)
    return 1.0 - 0.9 * progress


def pcld_per_block_loss(
    output: PCLDOutput,
    candidate_ids: Tensor,
    gold_ids: Tensor,
    target_residual: Tensor,
    target_candidate_logits: Tensor,
    target_top1_ids: Tensor,
    stable_rows: Tensor,
    latent_scale: Tensor,
    *,
    alpha: float,
    safe_margin: float = SAFE_MARGIN,
    safe_temperature: float = SAFE_TEMPERATURE,
    kl_temperature: float = KL_TEMPERATURE,
) -> PCLDLossOutput:
    """Frozen PCLD objective with one shared strict prefix support."""

    scores = output.scores.float()
    batch = scores.shape[0]
    if scores.shape != (batch, BLOCK_LENGTH, CANDIDATES):
        raise ValueError("PCLD scores must have shape [B,16,16]")
    if target_residual.shape != (batch, BLOCK_LENGTH, HIDDEN_SIZE):
        raise ValueError("target_residual must have shape [B,16,2560]")
    if target_candidate_logits.shape != scores.shape:
        raise ValueError("target candidate logits must match PCLD scores")
    if latent_scale.shape != (HIDDEN_SIZE,):
        raise ValueError("latent_scale must have shape [2560]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("latent alpha must lie in [0,1]")
    if safe_margin < 0 or safe_temperature <= 0 or kl_temperature <= 0:
        raise ValueError("PCLD margin/temperature values are invalid")
    if bool((latent_scale <= 0).any().item()) or not bool(
        torch.isfinite(latent_scale).all().item()
    ):
        raise ValueError("latent_scale must be finite and positive")

    gold_ranks = candidate_gold_ranks(candidate_ids, gold_ids)
    support, horizons = continuous_clean_support(
        gold_ranks, target_top1_ids, gold_ids, stable_rows
    )
    support_count = support.sum(dim=-1).clamp_min(1).float()
    safe_ranks = gold_ranks.clamp(0, CANDIDATES - 1)

    gold_scores = scores.gather(-1, safe_ranks.unsqueeze(-1)).squeeze(-1)
    gold_slots = F.one_hot(safe_ranks, num_classes=CANDIDATES).bool()
    strongest_other = scores.masked_fill(gold_slots, -torch.inf).amax(dim=-1)
    base_scale = output.base_scores.float().std(
        dim=-1, correction=0
    ).clamp_min(0.25)
    normalized_margin = (gold_scores - strongest_other) / base_scale
    position_weights = (
        (BLOCK_LENGTH - torch.arange(BLOCK_LENGTH, device=scores.device)).float()
        / BLOCK_LENGTH
    )
    risk_terms = (
        position_weights.log().unsqueeze(0)
        + (safe_margin - normalized_margin) / safe_temperature
    ).masked_fill(~support, -torch.inf)
    safe_loss = safe_temperature * torch.logsumexp(
        torch.cat(
            [torch.zeros(batch, 1, device=scores.device), risk_terms], dim=-1
        ).float(),
        dim=-1,
    )

    scale = latent_scale.float().view(1, 1, HIDDEN_SIZE)
    latent_rows = F.smooth_l1_loss(
        output.predicted_residual.float() / scale,
        target_residual.float() / scale,
        reduction="none",
    ).mean(dim=-1)
    latent_loss = (latent_rows * support.float()).sum(dim=-1) / support_count

    teacher_log_probs = torch.log_softmax(
        target_candidate_logits.float() / kl_temperature, dim=-1
    )
    student_log_probs = torch.log_softmax(scores / kl_temperature, dim=-1)
    teacher_probabilities = teacher_log_probs.exp()
    kl_rows = (
        teacher_probabilities * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1) * (kl_temperature**2)
    candidate_kl = (kl_rows * support.float()).sum(dim=-1) / support_count

    per_block = safe_loss + float(alpha) * latent_loss + KL_WEIGHT * candidate_kl
    return PCLDLossOutput(
        per_block_loss=per_block,
        safe_loss=safe_loss,
        latent_loss=latent_loss,
        candidate_kl=candidate_kl,
        support_mask=support,
        horizons=horizons,
        normalized_gold_margins=normalized_margin,
        latent_alpha=float(alpha),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_frozen_architecture(model: PCLD16Head) -> None:
    """Fail closed on any accidental architecture expansion."""

    count = count_trainable_parameters(model)
    if count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"PCLD trainable parameter count {count} != {EXPECTED_PARAMETER_COUNT}"
        )
    if len(model.encoder_blocks) != NUM_LAYERS:
        raise RuntimeError("PCLD encoder depth drifted")
    if model.query_attention.embed_dim != MODEL_DIM:
        raise RuntimeError("PCLD query dimension drifted")
