"""Direct globally informed reranking of a frozen DFlash candidate lattice.

This module intentionally contains no recurrent state, pairwise transition,
teacher-forced input, path decoder, or trainable vocabulary table.  Every
candidate score is produced directly from the complete, deterministic DFlash
candidate lattice available before draft tokens are selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class GlobalDirectOutput:
    """Direct candidate scores for one DFlash block."""

    scores: Tensor
    log_probs: Tensor
    residual_scores: Tensor
    base_log_probs: Tensor


@dataclass
class GlobalDirectLossOutput:
    """Acceptance-oriented direct candidate-classification objective."""

    loss: Tensor
    unweighted_nll: Tensor
    active_positions: Tensor
    training_positions: Tensor
    post_break_positions: Tensor
    position_weights: Tensor
    gold_probabilities: Tensor
    components: dict[str, Tensor]


def prefix_candidate_mask(gold_in_lattice: Tensor) -> Tensor:
    """Candidate labels in the prefix strictly before the first top-K miss."""

    if gold_in_lattice.ndim != 2:
        raise ValueError("gold_in_lattice must have shape [B, L]")
    alive_before = torch.cat(
        [
            torch.ones_like(gold_in_lattice[:, :1]),
            gold_in_lattice[:, :-1]
            .to(torch.int64)
            .cumprod(dim=1)
            .to(torch.bool),
        ],
        dim=1,
    )
    return alive_before & gold_in_lattice


def prediction_conditioned_prefix_mask(
    predicted_candidate_indices: Tensor,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
) -> Tensor:
    """Coverage positions reachable before the current greedy breaker.

    Reach at position ``i`` depends only on predictions at positions before
    ``i``.  Consequently an in-lattice breaker remains supervised, while its
    suffix does not.  Candidate coverage is kept as a separate fixed mask so
    evaluation denominators cannot be selected by the current model.
    """

    if predicted_candidate_indices.shape != gold_candidate_indices.shape:
        raise ValueError("predicted and gold candidate indices must match")
    if gold_candidate_indices.shape != gold_in_lattice.shape:
        raise ValueError("gold indices and lattice mask must match")
    if gold_candidate_indices.ndim != 2:
        raise ValueError("candidate indices must have shape [B, L]")
    correct = gold_in_lattice & predicted_candidate_indices.eq(
        gold_candidate_indices
    )
    reachable_before = torch.cat(
        [
            torch.ones_like(correct[:, :1]),
            correct[:, :-1]
            .to(torch.int64)
            .cumprod(dim=1)
            .to(torch.bool),
        ],
        dim=1,
    )
    coverage = prefix_candidate_mask(gold_in_lattice)
    reachable = coverage & reachable_before
    # ``reachable`` and ``coverage & ~reachable`` form the required disjoint
    # partition by construction.  Keep the invariants in unit tests rather
    # than synchronizing CUDA to assert them on every training batch.
    return reachable


def base_accepted_prefix_mask(
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
) -> Tensor:
    """Positions in the contiguous prefix accepted by DFlash rank one."""

    if gold_candidate_indices.shape != gold_in_lattice.shape:
        raise ValueError("gold indices and lattice mask must have equal shape")
    if gold_candidate_indices.ndim != 2:
        raise ValueError("gold indices must have shape [B, L]")
    base_correct = gold_in_lattice & gold_candidate_indices.eq(0)
    return base_correct.to(torch.int64).cumprod(dim=1).to(torch.bool)


def accepted_reach_survival(
    gold_probabilities: Tensor,
    gold_in_lattice: Tensor,
) -> Tensor:
    """Soft probability of accepting every prefix through each position.

    The selector distributions are interpreted as independent categorical
    decisions conditioned on the frozen DFlash lattice.  A missing gold token
    has probability zero and therefore censors that position and its suffix.
    """

    if gold_probabilities.shape != gold_in_lattice.shape:
        raise ValueError("probabilities and lattice mask must have equal shape")
    if gold_probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [B, L]")
    probabilities = torch.where(
        gold_in_lattice,
        gold_probabilities.float(),
        torch.zeros_like(gold_probabilities, dtype=torch.float32),
    )
    return torch.cumprod(probabilities, dim=-1)


def exact_dpace_position_weights(
    gold_probabilities: Tensor,
    active_positions: Tensor,
    *,
    alpha: float = 0.5,
) -> Tensor:
    """Official D-PACE inclusive-prefix, suffix-sum weights.

    This is the candidate-support adaptation of the official implementation in
    ``third_party/D-PACE/specforge/core/dflash.py`` at commit
    ``f36bad6e6b0f9f5b59e1e6cf405c705b46d2b43f``.  Invalid/censored positions
    are multiplicative no-ops in the prefix product and are excluded from the
    suffix sum.  The returned weights are detached and are deliberately not
    normalized.
    """

    if gold_probabilities.shape != active_positions.shape:
        raise ValueError("probabilities and mask must have equal shape")
    if gold_probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [B, L]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    with torch.no_grad():
        probabilities = gold_probabilities.detach().float()
        smoothed = (1.0 - alpha) * probabilities + alpha
        smoothed = torch.where(
            active_positions, smoothed, torch.ones_like(smoothed)
        )
        inclusive_prefix = torch.cumprod(smoothed, dim=-1)
        suffix_sum = torch.flip(
            torch.cumsum(
                torch.flip(
                    inclusive_prefix * active_positions.float(),
                    dims=[-1],
                ),
                dim=-1,
            ),
            dims=[-1],
        )
    return suffix_sum


class GlobalDirectBlock(nn.Module):
    """Pre-norm transformer block with a matched receptive-field control."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        max_positions: int,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.max_positions = max_positions
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
        self.relative_position_bias = nn.Embedding(
            2 * max_positions - 1, num_heads
        )
        # In a full L x K lattice only 1/L of the keys share the query's
        # position.  A log(L) prior gives local and cross-position evidence
        # comparable total mass at initialization instead of drowning the
        # candidate-comparison signal in 14x as many remote nodes.  The value
        # is a common offset under the local mask, so local remains an exactly
        # parameter-matched control.
        self.same_position_bias = nn.Parameter(
            torch.full((num_heads,), math.log(max_positions))
        )
        nn.init.zeros_(self.relative_position_bias.weight)

    @staticmethod
    def _node_positions(
        length: int, candidates: int, device: torch.device
    ) -> Tensor:
        return torch.arange(length, device=device).repeat_interleave(
            candidates
        )

    @classmethod
    def _allowed_attention(
        cls,
        length: int,
        candidates: int,
        scope: str,
        device: torch.device,
    ) -> Tensor | None:
        if scope == "global":
            return None
        positions = cls._node_positions(length, candidates, device)
        query_position = positions[:, None]
        key_position = positions[None, :]
        if scope == "local":
            return query_position == key_position
        if scope == "causal":
            return key_position <= query_position
        raise ValueError(f"unknown direct attention scope: {scope}")

    def _attention_bias(
        self,
        *,
        length: int,
        candidates: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        positions = self._node_positions(length, candidates, device)
        relative = (
            positions[:, None]
            - positions[None, :]
            + self.max_positions
            - 1
        )
        bias = self.relative_position_bias(relative).permute(2, 0, 1)
        same_position = positions[:, None] == positions[None, :]
        bias = bias + (
            self.same_position_bias[:, None, None]
            * same_position[None].to(self.same_position_bias.dtype)
        )
        return bias.to(dtype=dtype)

    def forward(
        self,
        states: Tensor,
        *,
        length: int,
        candidates: int,
        scope: str,
    ) -> Tensor:
        batch, nodes, _ = states.shape
        if nodes != length * candidates:
            raise ValueError("node count is inconsistent with L x K")
        if length > self.max_positions:
            raise ValueError("block exceeds relative-position capacity")

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
            length=length,
            candidates=candidates,
            device=states.device,
            dtype=attention_scores.dtype,
        )[None]
        allowed = self._allowed_attention(
            length, candidates, scope, states.device
        )
        if allowed is not None:
            attention_scores = attention_scores.masked_fill(
                ~allowed[None, None],
                torch.finfo(attention_scores.dtype).min,
            )
        attention = torch.softmax(
            attention_scores.float(), dim=-1
        ).to(value.dtype)
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


class AxialGlobalDirectBlock(nn.Module):
    """Candidate-local encoding followed by candidate-to-position attention.

    The flat lattice block makes a candidate compete for attention with all
    ``L * K`` nodes at once.  This axial alternative first preserves the
    within-position candidate comparison, then compresses every position's
    complete candidate distribution into one soft summary.  Each candidate
    subsequently queries all allowed position summaries, so its final score
    remains candidate-specific and globally informed without unstructured
    240-node mixing.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        max_positions: int,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.max_positions = max_positions
        self.dropout = dropout

        self.local_block = GlobalDirectBlock(
            model_dim,
            num_heads,
            max_positions=max_positions,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
        )
        self.pool_norm = nn.LayerNorm(model_dim)
        self.pool_score = nn.Linear(model_dim, 1, bias=False)
        self.cross_norm = nn.LayerNorm(model_dim)
        self.cross_query = nn.Linear(model_dim, model_dim, bias=False)
        self.cross_key_value = nn.Linear(
            model_dim, 2 * model_dim, bias=False
        )
        self.cross_out = nn.Linear(model_dim, model_dim, bias=False)
        self.cross_relative_position_bias = nn.Embedding(
            2 * max_positions - 1, num_heads
        )
        self.cross_feed_forward_norm = nn.LayerNorm(model_dim)
        self.cross_feed_forward = nn.Sequential(
            nn.Linear(
                model_dim, ff_multiplier * model_dim, bias=False
            ),
            nn.SiLU(),
            nn.Linear(
                ff_multiplier * model_dim, model_dim, bias=False
            ),
        )
        # Initial pooling is exactly the frozen DFlash conditional
        # distribution; training may learn which candidate evidence should
        # dominate a position summary.
        nn.init.zeros_(self.pool_score.weight)
        nn.init.zeros_(self.cross_relative_position_bias.weight)

    @staticmethod
    def _allowed_positions(
        length: int, scope: str, device: torch.device
    ) -> Tensor | None:
        if scope == "global":
            return None
        positions = torch.arange(length, device=device)
        query = positions[:, None]
        key = positions[None, :]
        if scope == "local":
            return query == key
        if scope == "causal":
            return key <= query
        raise ValueError(f"unknown direct attention scope: {scope}")

    def _relative_bias(
        self,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        positions = torch.arange(length, device=device)
        relative = (
            positions[:, None]
            - positions[None, :]
            + self.max_positions
            - 1
        )
        return self.cross_relative_position_bias(relative).permute(
            2, 0, 1
        ).to(dtype=dtype)

    def forward(
        self,
        states: Tensor,
        conditional_log_probs: Tensor,
        *,
        scope: str,
    ) -> Tensor:
        if states.ndim != 4:
            raise ValueError("axial states must have shape [B, L, K, D]")
        batch, length, candidates, model_dim = states.shape
        if model_dim != self.model_dim:
            raise ValueError("axial state dimension is inconsistent")
        if conditional_log_probs.shape != (
            batch,
            length,
            candidates,
        ):
            raise ValueError(
                "conditional_log_probs must have shape [B, L, K]"
            )
        if length > self.max_positions:
            raise ValueError("block exceeds relative-position capacity")

        local_states = self.local_block(
            states.reshape(batch, length * candidates, model_dim),
            length=length,
            candidates=candidates,
            scope="local",
        ).reshape(batch, length, candidates, model_dim)

        pool_logits = (
            conditional_log_probs.detach().float()
            + self.pool_score(
                self.pool_norm(local_states)
            ).squeeze(-1).float()
        )
        pool_weights = torch.softmax(pool_logits, dim=-1).to(
            local_states.dtype
        )
        position_summaries = (
            pool_weights[..., None] * local_states
        ).sum(dim=2)

        normalized_candidates = self.cross_norm(local_states)
        query = self.cross_query(normalized_candidates).view(
            batch,
            length,
            candidates,
            self.num_heads,
            self.head_dim,
        ).permute(0, 3, 1, 2, 4)
        key, value = (
            self.cross_key_value(position_summaries)
            .view(
                batch,
                length,
                2,
                self.num_heads,
                self.head_dim,
            )
            .unbind(dim=2)
        )
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attention_scores = torch.einsum(
            "bhlkd,bhmd->bhlkm", query, key
        ) / math.sqrt(self.head_dim)
        attention_scores = attention_scores + self._relative_bias(
            length,
            device=states.device,
            dtype=attention_scores.dtype,
        )[None, :, :, None, :]
        allowed = self._allowed_positions(
            length, scope, states.device
        )
        if allowed is not None:
            attention_scores = attention_scores.masked_fill(
                ~allowed[None, None, :, None, :],
                torch.finfo(attention_scores.dtype).min,
            )
        attention = torch.softmax(
            attention_scores.float(), dim=-1
        ).to(value.dtype)
        attention = F.dropout(
            attention, p=self.dropout, training=self.training
        )
        mixed = torch.einsum(
            "bhlkm,bhmd->bhlkd", attention, value
        ).permute(0, 2, 3, 1, 4).reshape(
            batch, length, candidates, model_dim
        )
        states = local_states + F.dropout(
            self.cross_out(mixed),
            p=self.dropout,
            training=self.training,
        )
        states = states + F.dropout(
            self.cross_feed_forward(
                self.cross_feed_forward_norm(states)
            ),
            p=self.dropout,
            training=self.training,
        )
        return states


class GlobalDirectCandidateSelector(nn.Module):
    """Matched local/causal/global direct selector over DFlash top-K nodes."""

    VALID_SCOPES = {"local", "causal", "global"}
    VALID_MIXERS = {"flat", "axial"}
    VALID_NODE_ENCODERS = {"additive", "compatibility"}
    NUM_SCALAR_FEATURES = 5

    def __init__(
        self,
        *,
        hidden_size: int,
        max_positions: int = 32,
        max_candidates: int = 16,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        scope: str = "global",
        mixer: str = "flat",
        node_encoder: str = "additive",
        dropout: float = 0.0,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        # Module constructors consume the process-wide CPU RNG in an order
        # that depends on which treatment modules exist.  Save it so model
        # construction cannot silently change either shared initialization or
        # the subsequent DataLoader shuffle stream across ablation cells.
        construction_rng_state = torch.random.get_rng_state()
        if scope not in self.VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(self.VALID_SCOPES)}")
        if mixer not in self.VALID_MIXERS:
            raise ValueError(f"mixer must be one of {sorted(self.VALID_MIXERS)}")
        if node_encoder not in self.VALID_NODE_ENCODERS:
            raise ValueError(
                "node_encoder must be one of "
                f"{sorted(self.VALID_NODE_ENCODERS)}"
            )
        if max_positions < 1 or max_candidates < 1:
            raise ValueError("position and candidate capacities must be positive")
        self.hidden_size = hidden_size
        self.max_positions = max_positions
        self.max_candidates = max_candidates
        self.model_dim = model_dim
        self.scope = scope
        self.mixer = mixer
        self.node_encoder = node_encoder
        self.initialization_seed = int(initialization_seed)

        self.hidden_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False
        )
        self.embedding_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False
        )
        self.hidden_projection = nn.Linear(
            hidden_size, model_dim, bias=False
        )
        # This projection is shared by candidate and anchor token embeddings.
        self.token_projection = nn.Linear(
            hidden_size, model_dim, bias=False
        )
        self.position_embedding = nn.Embedding(
            max_positions, model_dim
        )
        self.rank_embedding = nn.Embedding(
            max_candidates, model_dim
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(self.NUM_SCALAR_FEATURES, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.compatibility_projection = (
            nn.Sequential(
                nn.Linear(4 * model_dim, 2 * model_dim, bias=False),
                nn.SiLU(),
                nn.Linear(2 * model_dim, model_dim, bias=False),
            )
            if node_encoder == "compatibility"
            else None
        )
        self.input_norm = nn.LayerNorm(model_dim)
        self.blocks = nn.ModuleList(
            [
                (
                    GlobalDirectBlock(
                        model_dim,
                        num_heads,
                        max_positions=max_positions,
                        dropout=dropout,
                    )
                    if mixer == "flat"
                    else AxialGlobalDirectBlock(
                        model_dim,
                        num_heads,
                        max_positions=max_positions,
                        dropout=dropout,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.residual_projection = nn.Linear(model_dim, 1, bias=False)

        self._reset_parameters_deterministically()
        torch.random.set_rng_state(construction_rng_state)

    def _named_initialization_seed(self, name: str) -> int:
        payload = f"{self.initialization_seed}:{name}".encode("utf-8")
        value = int.from_bytes(
            hashlib.sha256(payload).digest()[:8], "big"
        )
        return value % (2**63 - 1)

    def _with_named_cpu_rng(
        self, name: str, callback: Callable[[], Any]
    ) -> Any:
        """Run one initializer without perturbing the process RNG stream."""

        state = torch.random.get_rng_state()
        torch.default_generator.manual_seed(
            self._named_initialization_seed(name)
        )
        try:
            return callback()
        finally:
            torch.random.set_rng_state(state)

    def _reset_parameters_deterministically(self) -> None:
        """Initialize shared names identically across architecture cells."""

        for name, module in self.named_modules():
            if not name:
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                self._with_named_cpu_rng(name, reset)

        # Preserve the frozen custom initialization contract after the generic
        # module resets.  Position/rank tables still use N(0, .02), but their
        # streams now depend only on name and the declared run seed.
        self._with_named_cpu_rng(
            "position_embedding.custom_normal",
            lambda: nn.init.normal_(
                self.position_embedding.weight, mean=0.0, std=0.02
            ),
        )
        self._with_named_cpu_rng(
            "rank_embedding.custom_normal",
            lambda: nn.init.normal_(
                self.rank_embedding.weight, mean=0.0, std=0.02
            ),
        )
        for module in self.modules():
            if isinstance(module, GlobalDirectBlock):
                nn.init.zeros_(module.relative_position_bias.weight)
                nn.init.constant_(
                    module.same_position_bias,
                    math.log(module.max_positions),
                )
            elif isinstance(module, AxialGlobalDirectBlock):
                nn.init.zeros_(module.pool_score.weight)
                nn.init.zeros_(
                    module.cross_relative_position_bias.weight
                )
        # Exact identity initialization: scores equal frozen DFlash log-probs.
        nn.init.zeros_(self.residual_projection.weight)

    def _validate_inputs(
        self,
        hidden: Tensor,
        candidate_embeddings: Tensor,
        candidate_logits: Tensor,
        base_logsumexp: Tensor,
        anchor_embeddings: Tensor,
    ) -> tuple[int, int, int]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B, L, D]")
        batch, length, hidden_size = hidden.shape
        if hidden_size != self.hidden_size:
            raise ValueError("hidden size differs from model configuration")
        if candidate_embeddings.ndim != 4:
            raise ValueError(
                "candidate_embeddings must have shape [B, L, K, D]"
            )
        cb, cl, candidates, embedding_size = candidate_embeddings.shape
        if (cb, cl, embedding_size) != (
            batch,
            length,
            self.hidden_size,
        ):
            raise ValueError("candidate embeddings have an invalid shape")
        if candidate_logits.shape != (batch, length, candidates):
            raise ValueError("candidate logits have an invalid shape")
        if base_logsumexp.shape != (batch, length):
            raise ValueError("base_logsumexp has an invalid shape")
        if anchor_embeddings.shape != (batch, self.hidden_size):
            raise ValueError("anchor_embeddings must have shape [B, D]")
        if length > self.max_positions:
            raise ValueError("block exceeds max_positions")
        if candidates > self.max_candidates:
            raise ValueError("candidate count exceeds max_candidates")
        return batch, length, candidates

    @staticmethod
    def _scalar_features(
        candidate_logits: Tensor, base_logsumexp: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the five fixed, bounded confidence features.

        Channels are, in order: full-vocabulary log probability, top-K
        conditional log probability, non-negative rank-1 logit gap, retained
        full-vocabulary log mass, and entropy of the renormalized top-K
        distribution.  Log quantities are smoothly clipped with ``tanh``;
        entropy is normalized by ``log(K)`` and clipped to ``[0, 1]``.
        All feature arithmetic is float32 even under autocast.
        """
        logits = candidate_logits.detach().float()
        full_lse = base_logsumexp.detach().float()
        base_log_probs = logits - full_lse[..., None]
        conditional_log_probs = torch.log_softmax(logits, dim=-1)
        # Use the actual maximum rather than assuming slot zero is also rank
        # one for every auxiliary feature source.  This matters when a caller
        # deploys one score distribution while exposing a second (for example
        # the parallel-base lattice) as selector evidence.
        top1_gap = logits.amax(dim=-1, keepdim=True) - logits
        retained_log_mass = torch.logsumexp(logits, dim=-1) - full_lse
        conditional_probabilities = conditional_log_probs.exp()
        entropy = -(
            conditional_probabilities * conditional_log_probs
        ).sum(dim=-1)
        log_candidates = math.log(max(2, logits.shape[-1]))

        # Smooth fixed scaling keeps heterogeneous scalar channels bounded
        # without altering the raw DFlash log-probs used in the final scores.
        features = torch.stack(
            [
                torch.tanh(base_log_probs / 8.0),
                torch.tanh(conditional_log_probs / 8.0),
                torch.tanh(top1_gap / 8.0),
                torch.tanh(retained_log_mass[..., None] / 2.0).expand_as(
                    logits
                ),
                (
                    entropy[..., None] / log_candidates
                ).clamp(0.0, 1.0).expand_as(logits),
            ],
            dim=-1,
        )
        return features, base_log_probs

    def forward(
        self,
        hidden: Tensor,
        candidate_embeddings: Tensor,
        candidate_logits: Tensor,
        base_logsumexp: Tensor,
        anchor_embeddings: Tensor,
        score_candidate_logits: Tensor | None = None,
        score_logsumexp: Tensor | None = None,
    ) -> GlobalDirectOutput:
        batch, length, candidates = self._validate_inputs(
            hidden,
            candidate_embeddings,
            candidate_logits,
            base_logsumexp,
            anchor_embeddings,
        )
        # Explicitly enforce the frozen target/DFlash boundary even if this
        # module is later attached to an online model rather than offline data.
        hidden = hidden.detach()
        candidate_embeddings = candidate_embeddings.detach()
        anchor_embeddings = anchor_embeddings.detach()
        scalar_features, feature_log_probs = self._scalar_features(
            candidate_logits, base_logsumexp
        )
        if (score_candidate_logits is None) != (score_logsumexp is None):
            raise ValueError(
                "score_candidate_logits and score_logsumexp must be provided together"
            )
        if score_candidate_logits is None:
            base_log_probs = feature_log_probs
        else:
            if score_candidate_logits.shape != candidate_logits.shape:
                raise ValueError("score_candidate_logits has an invalid shape")
            if score_logsumexp.shape != base_logsumexp.shape:
                raise ValueError("score_logsumexp has an invalid shape")
            base_log_probs = (
                score_candidate_logits.detach().float()
                - score_logsumexp.detach().float()[..., None]
            )

        normalized_hidden = self.hidden_norm(hidden.float())
        normalized_candidates = self.embedding_norm(
            candidate_embeddings.float()
        )
        normalized_anchor = self.embedding_norm(
            anchor_embeddings.float()
        )
        position_indices = torch.arange(
            length, device=hidden.device
        )[None, :, None]
        rank_indices = torch.arange(
            candidates, device=hidden.device
        )[None, None, :]
        hidden_states = self.hidden_projection(normalized_hidden)[
            :, :, None, :
        ]
        candidate_states = self.token_projection(normalized_candidates)
        anchor_states = self.token_projection(normalized_anchor)[
            :, None, None, :
        ]
        if self.compatibility_projection is None:
            node_states = hidden_states + candidate_states + anchor_states
        else:
            node_states = self.compatibility_projection(
                torch.cat(
                    [
                        hidden_states.expand_as(candidate_states),
                        candidate_states,
                        hidden_states * candidate_states,
                        anchor_states * candidate_states,
                    ],
                    dim=-1,
                )
            )
        states = (
            node_states
            + self.position_embedding(position_indices)
            + self.rank_embedding(rank_indices)
            + self.scalar_projection(scalar_features)
        )
        states = self.input_norm(states)
        if self.mixer == "flat":
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
            states = flat_states.reshape(
                batch, length, candidates, self.model_dim
            )
        else:
            conditional_log_probs = torch.log_softmax(
                candidate_logits.detach().float(), dim=-1
            )
            for block in self.blocks:
                states = block(
                    states,
                    conditional_log_probs,
                    scope=self.scope,
                )
        states = self.output_norm(
            states
        )
        residual_scores = self.residual_projection(states).squeeze(-1)
        # A per-position common offset cannot change a candidate distribution;
        # removing it makes residual magnitudes directly interpretable.
        residual_scores = residual_scores - residual_scores.mean(
            dim=-1, keepdim=True
        )
        scores = base_log_probs + residual_scores.float()
        # Keep the categorical normalization used by accepted-reach training
        # in float32.  In particular, do not let autocast turn a long prefix
        # product into a bf16-underflow diagnostic.
        log_probs = torch.log_softmax(scores.float(), dim=-1)
        return GlobalDirectOutput(
            scores=scores,
            log_probs=log_probs,
            residual_scores=residual_scores.float(),
            base_log_probs=base_log_probs,
        )


def global_direct_candidate_loss(
    output: GlobalDirectOutput,
    gold_candidate_indices: Tensor,
    gold_in_lattice: Tensor,
    *,
    weighting: str = "dpace",
    dpace_alpha: float = 0.5,
    exponential_gamma: float = 7.0,
    post_break_weight: float = 1.0,
    base_safety_weight: float = 0.0,
    base_safety_margin: float = 0.1,
) -> GlobalDirectLossOutput:
    """Train the exact direct logits used by greedy deployment."""

    if weighting not in {
        "uniform",
        "exponential",
        "dpace",
        "candidate_dpace",
        "reachable_dpace",
        "accepted_reach",
    }:
        raise ValueError(
            "weighting must be 'uniform', 'exponential', 'dpace', "
            "'candidate_dpace', 'reachable_dpace', or 'accepted_reach'"
        )
    if output.scores.ndim != 3:
        raise ValueError("scores must have shape [B, L, K]")
    batch, length, candidates = output.scores.shape
    if gold_candidate_indices.shape != (batch, length):
        raise ValueError("gold_candidate_indices has an invalid shape")
    if gold_in_lattice.shape != (batch, length):
        raise ValueError("gold_in_lattice has an invalid shape")
    if exponential_gamma <= 0:
        raise ValueError("exponential_gamma must be positive")
    if not 0.0 <= post_break_weight <= 1.0:
        raise ValueError("post_break_weight must be in [0, 1]")
    if base_safety_weight < 0:
        raise ValueError("base_safety_weight cannot be negative")
    if base_safety_margin < 0:
        raise ValueError("base_safety_margin cannot be negative")

    coverage_positions = prefix_candidate_mask(gold_in_lattice)
    training_positions = coverage_positions
    post_break_positions = torch.zeros_like(coverage_positions)
    safe_gold = gold_candidate_indices.clamp(0, candidates - 1)
    # Recompute explicitly from float32 scores rather than trusting a caller's
    # cached dtype.  This makes reach arithmetic stable for hand-built outputs
    # in tests as well as for mixed-precision training.
    score_log_probs = torch.log_softmax(output.scores.float(), dim=-1)
    gold_log_probs = score_log_probs.gather(
        -1, safe_gold.unsqueeze(-1)
    ).squeeze(-1)
    per_position_nll = -gold_log_probs
    gold_probabilities = gold_log_probs.exp()

    if weighting in {"dpace", "candidate_dpace", "reachable_dpace"}:
        position_weights = exact_dpace_position_weights(
            gold_probabilities,
            coverage_positions,
            alpha=dpace_alpha,
        )
        if weighting == "reachable_dpace":
            predicted = output.scores.detach().argmax(dim=-1)
            training_positions = prediction_conditioned_prefix_mask(
                predicted,
                gold_candidate_indices,
                gold_in_lattice,
            ).detach()
            post_break_positions = (
                coverage_positions & ~training_positions
            ).detach()
            if post_break_weight < 1.0:
                support_coefficients = (
                    training_positions.float()
                    + post_break_weight * post_break_positions.float()
                )
                position_weights = (
                    position_weights * support_coefficients
                )
    elif weighting == "exponential":
        positions = torch.arange(
            length,
            device=output.scores.device,
            dtype=torch.float32,
        )
        decay = torch.exp(-positions / exponential_gamma)
        position_weights = coverage_positions.float() * decay[None]
    else:
        position_weights = coverage_positions.float()

    base_prefix = base_accepted_prefix_mask(
        gold_candidate_indices, gold_in_lattice
    )
    scores_float = output.scores.float()
    alternative_scores = scores_float[..., 1:].max(dim=-1).values
    safety_violations = F.relu(
        base_safety_margin
        + alternative_scores
        - scores_float[..., 0]
    )
    base_prefix_counts = base_prefix.sum(dim=-1)
    per_block_safety = (
        safety_violations * base_prefix.float()
    ).sum(dim=-1) / base_prefix_counts.clamp_min(1)
    per_block_safety = torch.where(
        base_prefix_counts > 0,
        per_block_safety,
        torch.zeros_like(per_block_safety),
    )
    # Block-balanced averaging matches the deployment harm statistic: a long
    # DFlash prefix must not make one block dominate the safety regularizer.
    safety_loss = per_block_safety.mean()

    if weighting == "accepted_reach":
        survival = accepted_reach_survival(
            gold_probabilities, gold_in_lattice
        )
        expected_accepted = survival.sum(dim=-1).mean()
        reach_risk = 1.0 - expected_accepted / float(length)
        loss = reach_risk + base_safety_weight * safety_loss
        # This detached quantity is the continuation-value coefficient of
        # -log(q_i).  At alpha=0 it matches Candidate-D-PACE's gradient up to
        # ARR's 1/L normalization.
        position_weights = torch.flip(
            torch.cumsum(torch.flip(survival.detach(), dims=[-1]), dim=-1),
            dims=[-1],
        )
    else:
        loss = (
            per_position_nll
            * position_weights
        ).sum() / float(batch)
        if weighting in {"candidate_dpace", "reachable_dpace"}:
            # The explicitly named top-K adaptation is length-normalized so
            # alpha=0 has the same gradient as ARR.  The historical "dpace"
            # alias retains official per-block scaling for reproducibility.
            loss = loss / float(length)
        expected_accepted = accepted_reach_survival(
            gold_probabilities, gold_in_lattice
        ).sum(dim=-1).mean()
        reach_risk = 1.0 - expected_accepted / float(length)
        loss = loss + base_safety_weight * safety_loss
    unweighted_nll = (
        per_position_nll * coverage_positions.float()
    ).sum() / coverage_positions.sum().clamp_min(1)
    dpace_weights = exact_dpace_position_weights(
        gold_probabilities,
        coverage_positions,
        alpha=dpace_alpha,
    )
    reachable_component = (
        per_position_nll
        * dpace_weights
        * training_positions.float()
    ).sum() / float(batch * length)
    post_break_component = (
        per_position_nll
        * dpace_weights
        * post_break_positions.float()
    ).sum() / float(batch * length)
    coverage_count = coverage_positions.sum().float()
    reachable_fraction = (
        training_positions.sum().float()
        / coverage_count.clamp_min(1.0)
    )
    return GlobalDirectLossOutput(
        loss=loss,
        unweighted_nll=unweighted_nll,
        active_positions=coverage_positions,
        training_positions=training_positions,
        post_break_positions=post_break_positions,
        position_weights=position_weights,
        gold_probabilities=gold_probabilities,
        components={
            "reach_risk": reach_risk,
            "soft_expected_accepted_tokens": expected_accepted,
            "base_safety": safety_loss,
            "weighted_base_safety": base_safety_weight * safety_loss,
            "reachable_prefix_loss": reachable_component,
            "post_break_suffix_loss": post_break_component,
            "weighted_post_break_suffix_loss": (
                post_break_weight * post_break_component
            ),
            "reachable_fraction_of_coverage": reachable_fraction,
            "coverage_positions_per_block": (
                coverage_count / float(batch)
            ),
            "reachable_positions_per_block": (
                training_positions.sum().float() / float(batch)
            ),
            "post_break_positions_per_block": (
                post_break_positions.sum().float() / float(batch)
            ),
        },
    )


__all__ = [
    "AxialGlobalDirectBlock",
    "GlobalDirectBlock",
    "GlobalDirectCandidateSelector",
    "GlobalDirectLossOutput",
    "GlobalDirectOutput",
    "accepted_reach_survival",
    "base_accepted_prefix_mask",
    "exact_dpace_position_weights",
    "global_direct_candidate_loss",
    "prediction_conditioned_prefix_mask",
    "prefix_candidate_mask",
]
