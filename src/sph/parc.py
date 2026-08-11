"""PARC-16: one-call global correction of one full DFlash chain."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sph.parallel_global_candidate_fusion import (
    BLOCK_LENGTH,
    CANDIDATES,
    DEFAULT_FF_MULTIPLIER,
    DEFAULT_HEADS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LAYERS,
    DEFAULT_MODEL_DIM,
    DEFAULT_PARAMETER_COUNT,
    FullCandidateFusionBlock,
    rms_normalize,
)


EXPECTED_PARAMETER_COUNT = DEFAULT_PARAMETER_COUNT
PURE_DFLASH_INPUT_LENGTH = BLOCK_LENGTH + 1


def nonshift_full16_prediction_hidden(raw_hidden: Tensor) -> Tensor:
    """Drop the anchor-carrier row from extended non-shift DFlash raw17 output."""

    if raw_hidden.ndim != 3 or raw_hidden.shape[1] != PURE_DFLASH_INPUT_LENGTH:
        raise ValueError(
            "extended non-shift DFlash must return [B,17,H] before full16 slicing"
        )
    prediction_hidden = raw_hidden[:, 1 : 1 + BLOCK_LENGTH]
    if prediction_hidden.shape[1] != BLOCK_LENGTH:
        raise AssertionError("non-shift full16 slicing lost a prediction row")
    return prediction_hidden


@dataclass(frozen=True)
class PARCOutput:
    """One simultaneous candidate score tensor for one full16 chain."""

    scores: Tensor
    residual_advantages: Tensor
    candidate_states: Tensor


@dataclass(frozen=True)
class PARCLossOutput:
    """Fixed-reference gain and deterministic-harm quantities."""

    gain_loss: Tensor
    conditional_gain: Tensor
    harm_upper_bound: Tensor
    actual_harm: Tensor
    support_drop: Tensor
    ambiguous: Tensor
    gain_positions: Tensor
    gold_live_ranks: Tensor


class PARC16Head(nn.Module):
    """Full 256-action, non-causal, single-call, single-chain head."""

    NUM_SCALAR_FEATURES = 5

    def __init__(
        self,
        *,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        model_dim: int = DEFAULT_MODEL_DIM,
        num_heads: int = DEFAULT_HEADS,
        num_layers: int = DEFAULT_LAYERS,
        ff_multiplier: int = DEFAULT_FF_MULTIPLIER,
        local_control: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.hidden_size = int(hidden_size)
        self.model_dim = int(model_dim)
        self.local_control = bool(local_control)

        self.hidden_projection = nn.Linear(hidden_size, model_dim, bias=False)
        self.token_projection = nn.Linear(hidden_size, model_dim, bias=False)
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
            torch.linspace(0.0, 1.0, CANDIDATES).view(1, 1, CANDIDATES),
            persistent=False,
        )
        nn.init.zeros_(self.residual_projection.weight)

    @staticmethod
    def _validate_inputs(
        hidden: Tensor,
        candidate_logits: Tensor,
        anchor_embeddings: Tensor,
        candidate_embeddings: Tensor,
        *,
        hidden_size: int,
    ) -> int:
        if hidden.ndim != 3 or hidden.shape[1:] != (BLOCK_LENGTH, hidden_size):
            raise ValueError(f"hidden must have shape [B,16,{hidden_size}]")
        batch = int(hidden.shape[0])
        if candidate_logits.shape != (batch, BLOCK_LENGTH, CANDIDATES):
            raise ValueError("candidate_logits must have shape [B,16,16]")
        if candidate_embeddings.shape != (
            batch,
            BLOCK_LENGTH,
            CANDIDATES,
            hidden_size,
        ):
            raise ValueError(
                "candidate_embeddings must have shape [B,16,16,hidden_size]"
            )
        if anchor_embeddings.shape != (batch, hidden_size):
            raise ValueError(f"anchor_embeddings must have shape [B,{hidden_size}]")
        return batch

    def _scalar_features(self, candidate_logits: Tensor) -> Tensor:
        logits = candidate_logits.detach().float()
        log_probabilities = torch.log_softmax(logits, dim=-1)
        centered = logits - logits.mean(dim=-1, keepdim=True)
        gap = logits.amax(dim=-1, keepdim=True) - logits
        probabilities = log_probabilities.exp()
        entropy = -(
            probabilities * log_probabilities
        ).sum(dim=-1, keepdim=True) / torch.log(
            torch.tensor(float(CANDIDATES), device=logits.device)
        )
        ranks = self.normalized_candidate_rank.to(logits.dtype).expand_as(logits)
        return torch.stack(
            [
                torch.tanh(centered / 8.0),
                torch.tanh(log_probabilities / 8.0),
                torch.tanh(gap / 8.0),
                ranks,
                entropy.expand_as(logits),
            ],
            dim=-1,
        )

    def encode_candidates(
        self,
        hidden: Tensor,
        candidate_logits: Tensor,
        anchor_embeddings: Tensor,
        candidate_embeddings: Tensor,
    ) -> Tensor:
        batch = self._validate_inputs(
            hidden,
            candidate_logits,
            anchor_embeddings,
            candidate_embeddings,
            hidden_size=self.hidden_size,
        )
        hidden_state = self.hidden_projection(rms_normalize(hidden))
        anchor_state = self.token_projection(
            rms_normalize(anchor_embeddings)
        )[:, None, :]
        position_query = hidden_state + anchor_state
        candidate_state = self.token_projection(
            rms_normalize(candidate_embeddings)
        )
        candidate_delta = candidate_state - candidate_state[:, :, :1, :]
        query_nodes = position_query[:, :, None, :]
        compatibility = self.compatibility_projection(
            query_nodes * candidate_delta
        )
        scalar_state = self.scalar_projection(
            self._scalar_features(candidate_logits).to(hidden.dtype)
        )
        positions = self.position_embedding.weight.to(hidden.dtype)[
            None, :, None, :
        ]
        ranks = self.rank_embedding.weight.to(hidden.dtype)[None, None, :, :]
        nodes = self.input_norm(
            query_nodes
            + candidate_delta
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
        candidate_embeddings: Tensor,
    ) -> PARCOutput:
        states = self.encode_candidates(
            hidden,
            candidate_logits,
            anchor_embeddings,
            candidate_embeddings,
        )
        for block in self.blocks:
            states = block(states)
        candidate_states = states.reshape(
            hidden.shape[0], BLOCK_LENGTH, CANDIDATES, self.model_dim
        )
        raw_residual = self.residual_projection(
            self.output_norm(candidate_states)
        ).squeeze(-1)
        residual_advantages = raw_residual - raw_residual[:, :, :1]
        base_advantages = (
            candidate_logits.float() - candidate_logits[:, :, :1].float()
        )
        scores = base_advantages + residual_advantages.float()
        return PARCOutput(
            scores=scores,
            residual_advantages=residual_advantages,
            candidate_states=candidate_states,
        )

    @staticmethod
    def proposal_ids(candidate_ids: Tensor, output: PARCOutput) -> Tensor:
        if candidate_ids.shape != output.scores.shape:
            raise ValueError("candidate IDs must match PARC scores")
        return candidate_ids.gather(
            -1, output.scores.argmax(dim=-1, keepdim=True)
        ).squeeze(-1)

    @torch.no_grad()
    def project_vocabulary(self, embedding: Tensor, chunk_size: int = 4096) -> Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != self.hidden_size:
            raise ValueError("embedding must have shape [vocabulary,hidden_size]")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        return torch.cat(
            [
                self.token_projection(
                    rms_normalize(embedding[start : start + chunk_size])
                )
                for start in range(0, embedding.shape[0], chunk_size)
            ],
            dim=0,
        )


def live_gold_support(
    candidate_ids: Tensor, gold_ids: Tensor
) -> tuple[Tensor, Tensor]:
    if candidate_ids.ndim != 3 or candidate_ids.shape[-2:] != (
        BLOCK_LENGTH,
        CANDIDATES,
    ):
        raise ValueError("candidate_ids must have shape [B,16,16]")
    if gold_ids.shape != candidate_ids.shape[:2]:
        raise ValueError("gold_ids must have shape [B,16]")
    matches = candidate_ids.eq(gold_ids.unsqueeze(-1))
    support = matches.any(dim=-1)
    if bool((matches.sum(dim=-1) > 1).any()):
        raise ValueError("a Top16 row contains duplicate gold IDs")
    safe_ranks = matches.to(torch.long).argmax(dim=-1)
    ranks = torch.where(
        support, safe_ranks, torch.full_like(safe_ranks, -1)
    )
    return support, ranks


def conditional_gain_loss(
    scores: Tensor,
    gold_live_ranks: Tensor,
    reference_accepted: Tensor,
    *,
    block_enabled: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Detached-product loss with exact local gradient of ``-G/16``."""

    if scores.ndim != 3 or scores.shape[-2:] != (BLOCK_LENGTH, CANDIDATES):
        raise ValueError("scores must have shape [B,16,16]")
    batch = int(scores.shape[0])
    if gold_live_ranks.shape != (batch, BLOCK_LENGTH):
        raise ValueError("gold_live_ranks must have shape [B,16]")
    if reference_accepted.shape != (batch,):
        raise ValueError("reference_accepted must have shape [B]")
    if bool(((reference_accepted < 0) | (reference_accepted > BLOCK_LENGTH)).any()):
        raise ValueError("reference accepted lengths lie outside [0,16]")
    if block_enabled is None:
        block_enabled = torch.ones(batch, dtype=torch.bool, device=scores.device)
    if block_enabled.shape != (batch,):
        raise ValueError("block_enabled must have shape [B]")

    log_probabilities = torch.log_softmax(scores.float(), dim=-1)
    support = gold_live_ranks.ge(0)
    safe_ranks = torch.where(
        support, gold_live_ranks, torch.zeros_like(gold_live_ranks)
    )
    selected_log_probability = log_probabilities.gather(
        -1, safe_ranks.unsqueeze(-1)
    ).squeeze(-1)
    total_loss = scores.float().sum() * 0.0
    gains: list[Tensor] = []
    masks: list[Tensor] = []
    axis = torch.arange(BLOCK_LENGTH, device=scores.device)
    for row in range(batch):
        start = int(reference_accepted[row].detach().item())
        row_mask = torch.zeros(BLOCK_LENGTH, dtype=torch.bool, device=scores.device)
        if not bool(block_enabled[row].detach().item()) or start == BLOCK_LENGTH:
            gains.append(scores.new_zeros((), dtype=torch.float32))
            masks.append(row_mask)
            continue
        suffix_support = support[row, start:]
        unsupported = (~suffix_support).nonzero(as_tuple=False)
        stop = (
            BLOCK_LENGTH
            if unsupported.numel() == 0
            else start + int(unsupported[0, 0].item())
        )
        if stop == start:
            gains.append(scores.new_zeros((), dtype=torch.float32))
            masks.append(row_mask)
            continue
        row_mask[start:stop] = True
        log_gold = selected_log_probability[row, start:stop]
        prefix_products = torch.cumprod(log_gold.exp(), dim=0)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix_products, dims=(0,)), dim=0),
            dims=(0,),
        ).detach()
        total_loss = total_loss + (weights * -log_gold).sum()
        gains.append(prefix_products.sum())
        masks.append(row_mask)
    gain = torch.stack(gains)
    return total_loss / float(BLOCK_LENGTH * batch), gain, torch.stack(masks)


def parc_fixed_reference_loss(
    output: PARCOutput,
    candidate_ids: Tensor,
    gold_ids: Tensor,
    reference_accepted: Tensor,
    reference_delta: Tensor,
    *,
    delta_min: float,
) -> PARCLossOutput:
    if delta_min <= 0:
        raise ValueError("delta_min must be positive")
    batch = int(output.scores.shape[0])
    if reference_delta.shape != (batch,):
        raise ValueError("reference_delta must have shape [B]")
    support, ranks = live_gold_support(candidate_ids, gold_ids)
    positions = torch.arange(BLOCK_LENGTH, device=gold_ids.device)[None]
    protected = positions < reference_accepted[:, None]
    support_drop = (protected & ~support).any(dim=-1)
    ambiguous = (reference_accepted > 0) & reference_delta.le(delta_min)

    gain_loss, gain, gain_positions = conditional_gain_loss(
        output.scores,
        ranks,
        reference_accepted,
        block_enabled=~support_drop,
    )

    safe_ranks = torch.where(support, ranks, torch.zeros_like(ranks))
    gold_scores = output.scores.gather(
        -1, safe_ranks.unsqueeze(-1)
    ).squeeze(-1)
    competitors = output.scores.masked_fill(
        candidate_ids.eq(gold_ids.unsqueeze(-1)), -torch.inf
    )
    competitor_scores = competitors.amax(dim=-1)
    margins = competitor_scores - gold_scores
    negative_infinity = torch.full_like(margins, -torch.inf)
    block_margin = torch.where(protected, margins, negative_infinity).amax(dim=-1)
    empty_prefix = reference_accepted.eq(0)
    block_margin = torch.where(empty_prefix, torch.full_like(block_margin, -torch.inf), block_margin)
    gamma = reference_delta.float() / 2.0
    stable_bound = torch.relu(1.0 + block_margin / gamma.clamp_min(delta_min / 2.0))
    harm_upper_bound = torch.where(
        empty_prefix,
        torch.zeros_like(stable_bound),
        torch.where(
            ambiguous | support_drop,
            torch.ones_like(stable_bound),
            stable_bound,
        ),
    )
    proposals = PARC16Head.proposal_ids(candidate_ids, output)
    actual_harm = (protected & proposals.ne(gold_ids)).any(dim=-1).float()
    return PARCLossOutput(
        gain_loss=gain_loss,
        conditional_gain=gain,
        harm_upper_bound=harm_upper_bound,
        actual_harm=actual_harm,
        support_drop=support_drop,
        ambiguous=ambiguous,
        gain_positions=gain_positions,
        gold_live_ranks=ranks,
    )


def assert_frozen_architecture(model: PARC16Head) -> None:
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"PARC trainable parameter count {count} != {EXPECTED_PARAMETER_COUNT}"
        )
    if model.hidden_size != DEFAULT_HIDDEN_SIZE or model.model_dim != DEFAULT_MODEL_DIM:
        raise RuntimeError("PARC hidden/model width drifted")
    if len(model.blocks) != DEFAULT_LAYERS:
        raise RuntimeError("PARC encoder depth drifted")
    if model.local_control:
        raise RuntimeError("claim-bearing PARC head cannot use local control")
