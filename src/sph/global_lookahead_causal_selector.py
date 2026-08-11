"""Candidate-only Domino replacement with global lookahead and causal feedback.

The head keeps Domino's useful causal state and token-specific correction
basis, but never projects a correction vector over the full vocabulary.  It
first encodes the complete DFlash Top-K lattice, then rolls out one token at a
time and scores only the K available candidates.  Consequently an actually
selected token affects all later decisions while future lattice evidence can
affect the current decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class CausalSelectorOutput:
    token_ids: Tensor
    candidate_scores: Tensor
    route_weights: Tensor


class _LookaheadBlock(nn.Module):
    """Small pre-norm Transformer block over position/mode tokens."""

    def __init__(self, width: int, heads: int, feed_forward_width: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, bias=False, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, feed_forward_width, bias=False),
            nn.SiLU(),
            nn.Linear(feed_forward_width, width, bias=False),
        )

    def forward(self, states: Tensor) -> Tensor:
        normalized = self.attention_norm(states)
        mixed, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        states = states + mixed
        return states + self.feed_forward(self.feed_forward_norm(states))


class GlobalLookaheadCausalSelector(nn.Module):
    """Global Top-K lattice encoder plus a Domino-initialized causal selector.

    ``token_embeddings`` is the frozen target embedding table.  ``candidate_basis``
    is initialized from Domino's final correction projection and remains
    trainable by default.  The lookahead contribution is zero-initialized, so
    construction exactly reproduces Domino's correction formula restricted to
    the supplied candidate set.
    """

    def __init__(
        self,
        *,
        token_embeddings: Tensor,
        candidate_basis: Tensor,
        gru_weight_ih: Tensor,
        gru_weight_hh: Tensor,
        hidden_projection: Tensor,
        state_projection: Tensor,
        max_positions: int = 15,
        candidates: int = 16,
        global_width: int = 512,
        global_heads: int = 8,
        global_layers: int = 2,
        global_modes: int = 4,
        feed_forward_width: int = 1536,
    ) -> None:
        super().__init__()
        if token_embeddings.ndim != 2 or candidate_basis.ndim != 2:
            raise ValueError("token_embeddings and candidate_basis must be matrices")
        vocabulary, hidden_width = token_embeddings.shape
        if candidate_basis.shape[0] != vocabulary:
            raise ValueError("embedding and candidate-basis vocabularies differ")
        code_width = int(candidate_basis.shape[1])
        if gru_weight_ih.shape[0] % 3:
            raise ValueError("GRU input projection has invalid gate dimension")
        state_width = int(gru_weight_ih.shape[0] // 3)
        if gru_weight_ih.shape != (3 * state_width, hidden_width):
            raise ValueError("GRU input projection shape is inconsistent")
        if gru_weight_hh.shape != (3 * state_width, state_width):
            raise ValueError("GRU recurrent projection shape is inconsistent")
        if hidden_projection.shape != (code_width, hidden_width):
            raise ValueError("hidden correction projection has the wrong shape")
        if state_projection.shape != (code_width, state_width):
            raise ValueError("state correction projection has the wrong shape")
        if max_positions < 1 or candidates < 2 or global_modes < 1:
            raise ValueError("invalid selector lattice shape")
        if global_width % global_heads:
            raise ValueError("global_width must be divisible by global_heads")

        self.vocabulary = int(vocabulary)
        self.hidden_width = int(hidden_width)
        self.state_width = state_width
        self.code_width = code_width
        self.max_positions = int(max_positions)
        self.candidates = int(candidates)
        self.global_width = int(global_width)
        self.global_modes = int(global_modes)

        self.register_buffer(
            "token_embeddings", token_embeddings.detach(), persistent=False
        )
        self.register_buffer(
            "projected_lexical_table",
            torch.empty(
                0,
                global_width,
                dtype=token_embeddings.dtype,
                device=token_embeddings.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "projected_gru_input_table",
            torch.empty(
                0,
                3 * state_width,
                dtype=token_embeddings.dtype,
                device=token_embeddings.device,
            ),
            persistent=False,
        )
        # Keep trainable lexical rows in fp32.  The released checkpoint is
        # bf16, but the low learning rates needed for safe fine-tuning would
        # otherwise fall below a bf16 parameter ULP and silently do nothing.
        self.candidate_basis = nn.Parameter(
            candidate_basis.detach().float().clone()
        )
        self.prefix_gru = nn.GRU(
            hidden_width,
            state_width,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        with torch.no_grad():
            self.prefix_gru.weight_ih_l0.copy_(gru_weight_ih)
            self.prefix_gru.weight_hh_l0.copy_(gru_weight_hh)
        self.hidden_projection = nn.Linear(hidden_width, code_width, bias=False)
        self.state_projection = nn.Linear(state_width, code_width, bias=False)
        with torch.no_grad():
            self.hidden_projection.weight.copy_(hidden_projection)
            self.state_projection.weight.copy_(state_projection)

        self.semantic_projection = nn.Linear(
            hidden_width, global_width, bias=False
        )
        self.basis_projection = nn.Linear(code_width, global_width, bias=False)
        self.hidden_global_projection = nn.Linear(
            hidden_width, global_width, bias=False
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(5, global_width, bias=False),
            nn.SiLU(),
            nn.Linear(global_width, global_width, bias=False),
        )
        self.rank_embedding = nn.Embedding(candidates, global_width)
        self.position_embedding = nn.Embedding(max_positions, global_width)
        self.node_norm = nn.LayerNorm(global_width)

        self.mode_queries = nn.Parameter(torch.empty(global_modes, global_width))
        self.local_attention = nn.MultiheadAttention(
            global_width, global_heads, bias=False, batch_first=True
        )
        self.local_norm = nn.LayerNorm(global_width)
        self.global_blocks = nn.ModuleList(
            [
                _LookaheadBlock(
                    global_width, global_heads, feed_forward_width
                )
                for _ in range(global_layers)
            ]
        )
        self.global_norm = nn.LayerNorm(global_width)
        self.state_route_projection = nn.Linear(
            state_width, global_width, bias=False
        )
        self.route_key_projection = nn.Linear(
            global_width, global_width, bias=False
        )
        self.global_to_code = nn.Linear(global_width, code_width, bias=False)

        # Flexible rank calibration helps remote candidates cross a large base
        # margin without discarding the released score at initialization.
        self.base_logit_scale = nn.Parameter(torch.ones(max_positions))
        self.rank_bias = nn.Parameter(torch.zeros(max_positions, candidates))

        nn.init.normal_(self.mode_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.rank_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        # This is the exact baseline gate.  The global encoder can be trained
        # immediately because gradients reach this matrix on the first step.
        nn.init.zeros_(self.global_to_code.weight)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @torch.no_grad()
    def prepare_inference(self) -> None:
        """Preproject vocabulary tables used by the fixed-weight runtime."""

        self.projected_lexical_table = (
            F.linear(self.token_embeddings, self.semantic_projection.weight)
            + F.linear(self.candidate_basis, self.basis_projection.weight)
        ).contiguous()
        self.projected_gru_input_table = F.linear(
            self.token_embeddings, self.prefix_gru.weight_ih_l0
        ).contiguous()

    def clear_inference_table(self) -> None:
        self.projected_lexical_table = torch.empty(
            0,
            self.global_width,
            dtype=self.token_embeddings.dtype,
            device=self.token_embeddings.device,
        )
        self.projected_gru_input_table = torch.empty(
            0,
            3 * self.state_width,
            dtype=self.token_embeddings.dtype,
            device=self.token_embeddings.device,
        )

    def _projected_gru_cell(self, token_ids: Tensor, state: Tensor) -> Tensor:
        """One bias-free PyTorch GRU cell using a preprojected token table."""

        if self.projected_gru_input_table.shape != (
            self.vocabulary,
            3 * self.state_width,
        ):
            raise RuntimeError("call prepare_inference before projected GRU rollout")
        input_gates = F.embedding(token_ids, self.projected_gru_input_table)
        hidden_gates = F.linear(state, self.prefix_gru.weight_hh_l0)
        input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        new = torch.tanh(input_new + reset * hidden_new)
        return (1.0 - update) * new + update * state

    @staticmethod
    def _scalar_features(candidate_logits: Tensor) -> Tensor:
        logits = candidate_logits.float()
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        gap = logits[..., :1] - logits
        scale = gap[..., -1:].abs().clamp_min(1e-3)
        standardized_gap = gap / scale
        entropy = -(probabilities * log_probabilities).sum(
            dim=-1, keepdim=True
        )
        entropy = entropy.expand_as(logits)
        cumulative_mass = probabilities.cumsum(dim=-1)
        rank = torch.linspace(
            0.0,
            1.0,
            logits.shape[-1],
            dtype=logits.dtype,
            device=logits.device,
        ).view(1, 1, -1).expand_as(logits)
        return torch.stack(
            [
                torch.tanh(log_probabilities / 8.0),
                torch.tanh(standardized_gap),
                entropy,
                cumulative_mass,
                rank,
            ],
            dim=-1,
        )

    def _lexical_nodes(self, candidate_ids: Tensor) -> Tensor:
        if self.projected_lexical_table.shape == (
            self.vocabulary,
            self.global_width,
        ):
            return F.embedding(candidate_ids, self.projected_lexical_table)
        semantics = self.semantic_projection(
            F.embedding(candidate_ids, self.token_embeddings)
        )
        basis = self.basis_projection(
            F.embedding(candidate_ids, self.candidate_basis)
        )
        return semantics + basis

    def encode_lattice(
        self,
        *,
        parallel_hiddens: Tensor,
        candidate_ids: Tensor,
        candidate_logits: Tensor,
    ) -> Tensor:
        """Return globally mixed modes with shape ``[B,L,M,D]``."""

        batch, positions, candidates = candidate_ids.shape
        if candidates != self.candidates:
            raise ValueError("candidate count differs from selector capacity")
        if positions > self.max_positions:
            raise ValueError("position count differs from selector capacity")
        if candidate_logits.shape != candidate_ids.shape:
            raise ValueError("candidate ids/logits shapes differ")
        if parallel_hiddens.shape != (batch, positions, self.hidden_width):
            raise ValueError("parallel hidden shape is inconsistent")

        position_ids = torch.arange(positions, device=candidate_ids.device)
        rank_ids = torch.arange(candidates, device=candidate_ids.device)
        nodes = self._lexical_nodes(candidate_ids)
        nodes = nodes + self.hidden_global_projection(
            parallel_hiddens
        ).unsqueeze(-2)
        nodes = nodes + self.scalar_projection(
            self._scalar_features(candidate_logits).to(nodes.dtype)
        )
        nodes = nodes + self.position_embedding(position_ids)[None, :, None]
        nodes = nodes + self.rank_embedding(rank_ids)[None, None]
        nodes = self.node_norm(nodes)

        flat_nodes = nodes.reshape(
            batch * positions, candidates, self.global_width
        )
        queries = self.mode_queries[None].expand(batch * positions, -1, -1)
        local, _ = self.local_attention(
            queries, flat_nodes, flat_nodes, need_weights=False
        )
        modes = self.local_norm(queries + local).view(
            batch, positions * self.global_modes, self.global_width
        )
        for block in self.global_blocks:
            modes = block(modes)
        return self.global_norm(modes).view(
            batch, positions, self.global_modes, self.global_width
        )

    def prefix_states(self, *, anchor_ids: Tensor, previous_ids: Tensor) -> Tensor:
        """Compute the clean-prefix GRU state used at every scored position.

        ``previous_ids[:, 0]`` is the fixed DFlash prefix token.  Later entries
        are the tokens selected at preceding correction positions.
        """

        if anchor_ids.ndim != 1 or previous_ids.ndim != 2:
            raise ValueError("anchor_ids/previous_ids must have rank one/two")
        if previous_ids.shape[0] != anchor_ids.shape[0]:
            raise ValueError("anchor and previous-token batches differ")
        sequence = torch.cat([anchor_ids[:, None], previous_ids], dim=-1)
        outputs, _ = self.prefix_gru(F.embedding(sequence, self.token_embeddings))
        return outputs[:, 1:]

    def score_candidates(
        self,
        *,
        parallel_hiddens: Tensor,
        candidate_ids: Tensor,
        candidate_logits: Tensor,
        prefix_states: Tensor,
        global_modes: Tensor,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        batch, positions, candidates = candidate_ids.shape
        if position_offset < 0 or position_offset + positions > self.max_positions:
            raise ValueError("score position range exceeds selector capacity")
        if prefix_states.shape != (batch, positions, self.state_width):
            raise ValueError("prefix-state shape is inconsistent")
        if global_modes.shape != (
            batch,
            positions,
            self.global_modes,
            self.global_width,
        ):
            raise ValueError("global-mode shape is inconsistent")
        route_query = self.state_route_projection(prefix_states)
        route_keys = self.route_key_projection(global_modes)
        route_logits = torch.einsum(
            "bld,blmd->blm", route_query, route_keys
        ) / math.sqrt(self.global_width)
        route_weights = torch.softmax(route_logits.float(), dim=-1).to(
            global_modes.dtype
        )
        global_context = torch.einsum(
            "blm,blmd->bld", route_weights, global_modes
        )

        correction_code = (
            self.hidden_projection(parallel_hiddens)
            + self.state_projection(prefix_states)
            + self.global_to_code(global_context)
        )
        correction_code = F.silu(correction_code)
        basis = F.embedding(candidate_ids, self.candidate_basis)
        correction_scores = torch.einsum(
            "blc,blkc->blk", correction_code, basis
        )
        position_slice = slice(position_offset, position_offset + positions)
        scale = self.base_logit_scale[position_slice].view(1, positions, 1)
        rank_bias = self.rank_bias[position_slice, :candidates].view(
            1, positions, candidates
        )
        scores = (
            scale.float() * candidate_logits.float()
            + correction_scores.float()
            + rank_bias.float()
        )
        return scores, route_weights

    def teacher_forward(
        self,
        *,
        parallel_hiddens: Tensor,
        candidate_ids: Tensor,
        candidate_logits: Tensor,
        anchor_ids: Tensor,
        previous_ids: Tensor,
    ) -> CausalSelectorOutput:
        modes = self.encode_lattice(
            parallel_hiddens=parallel_hiddens,
            candidate_ids=candidate_ids,
            candidate_logits=candidate_logits,
        )
        states = self.prefix_states(
            anchor_ids=anchor_ids, previous_ids=previous_ids
        )
        scores, route = self.score_candidates(
            parallel_hiddens=parallel_hiddens,
            candidate_ids=candidate_ids,
            candidate_logits=candidate_logits,
            prefix_states=states,
            global_modes=modes,
        )
        tokens = candidate_ids.gather(
            -1, scores.argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
        return CausalSelectorOutput(tokens, scores, route)

    @torch.inference_mode()
    def decode(
        self,
        *,
        parallel_hiddens: Tensor,
        candidate_ids: Tensor,
        candidate_logits: Tensor,
        anchor_ids: Tensor,
        fixed_prefix_ids: Tensor,
    ) -> CausalSelectorOutput:
        batch, positions, _ = candidate_ids.shape
        if anchor_ids.shape != (batch,) or fixed_prefix_ids.shape != (batch,):
            raise ValueError("anchor/fixed-prefix shapes are inconsistent")
        modes = self.encode_lattice(
            parallel_hiddens=parallel_hiddens,
            candidate_ids=candidate_ids,
            candidate_logits=candidate_logits,
        )
        projected_rollout = self.projected_gru_input_table.shape == (
            self.vocabulary,
            3 * self.state_width,
        )
        if projected_rollout:
            flat_state = torch.zeros(
                batch,
                self.state_width,
                dtype=self.prefix_gru.weight_hh_l0.dtype,
                device=anchor_ids.device,
            )
            flat_state = self._projected_gru_cell(anchor_ids, flat_state)
            flat_state = self._projected_gru_cell(fixed_prefix_ids, flat_state)
            state = flat_state.unsqueeze(0)
        else:
            initial = torch.stack([anchor_ids, fixed_prefix_ids], dim=-1)
            _, state = self.prefix_gru(F.embedding(initial, self.token_embeddings))
        tokens: list[Tensor] = []
        scores_by_position: list[Tensor] = []
        routes: list[Tensor] = []
        for position in range(positions):
            current_state = state.transpose(0, 1)
            scores, route = self.score_candidates(
                parallel_hiddens=parallel_hiddens[:, position : position + 1],
                candidate_ids=candidate_ids[:, position : position + 1],
                candidate_logits=candidate_logits[:, position : position + 1],
                prefix_states=current_state,
                global_modes=modes[:, position : position + 1],
                position_offset=position,
            )
            token = candidate_ids[:, position].gather(
                -1, scores[:, 0].argmax(dim=-1, keepdim=True)
            )
            tokens.append(token)
            scores_by_position.append(scores)
            routes.append(route)
            if position + 1 < positions:
                if projected_rollout:
                    state = self._projected_gru_cell(
                        token[:, 0], state[0]
                    ).unsqueeze(0)
                else:
                    _, state = self.prefix_gru(
                        F.embedding(token, self.token_embeddings), state
                    )
        return CausalSelectorOutput(
            torch.cat(tokens, dim=-1),
            torch.cat(scores_by_position, dim=1),
            torch.cat(routes, dim=1),
        )


def topk_candidates(base_logits: Tensor, candidates: int) -> tuple[Tensor, Tensor]:
    """Return sorted Top-K logits and IDs with a stable shared convention."""

    if base_logits.ndim != 3:
        raise ValueError("base_logits must have shape [batch, positions, vocab]")
    values, ids = torch.topk(base_logits, candidates, dim=-1, sorted=True)
    return ids, values


class GLCSGraphRunner:
    """Fixed-shape CUDA-graph runner matching Domino's correction interface."""

    def __init__(
        self,
        *,
        head: GlobalLookaheadCausalSelector,
        batch_size: int,
        positions: int,
        device: torch.device,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GLCSGraphRunner requires CUDA")
        if positions > head.max_positions:
            raise ValueError("position count exceeds head capacity")
        if head.projected_lexical_table.shape != (
            head.vocabulary,
            head.global_width,
        ) or head.projected_gru_input_table.shape != (
            head.vocabulary,
            3 * head.state_width,
        ):
            raise RuntimeError("call head.prepare_inference() before graph capture")
        self.head = head
        self.batch_size = int(batch_size)
        self.positions = int(positions)
        dtype = head.token_embeddings.dtype
        self.static_prefix_ids = torch.zeros(
            batch_size, 2, dtype=torch.long, device=device
        )
        self.static_parallel_hiddens = torch.zeros(
            batch_size, positions, head.hidden_width, dtype=dtype, device=device
        )
        self.static_base_logits = torch.zeros(
            batch_size, positions, head.vocabulary, dtype=dtype, device=device
        )

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(side):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.static_output = self._forward()

    def _forward(self) -> Tensor:
        candidate_ids, candidate_logits = topk_candidates(
            self.static_base_logits, self.head.candidates
        )
        return self.head.decode(
            parallel_hiddens=self.static_parallel_hiddens,
            candidate_ids=candidate_ids,
            candidate_logits=candidate_logits,
            anchor_ids=self.static_prefix_ids[:, 0],
            fixed_prefix_ids=self.static_prefix_ids[:, 1],
        ).token_ids

    @torch.inference_mode()
    def __call__(
        self,
        prefix_ids: Tensor,
        parallel_hiddens: Tensor,
        base_logits: Tensor,
    ) -> Tensor:
        if prefix_ids.shape != self.static_prefix_ids.shape:
            raise ValueError("prefix IDs differ from captured shape")
        if parallel_hiddens.shape != self.static_parallel_hiddens.shape:
            raise ValueError("parallel hiddens differ from captured shape")
        if base_logits.shape != self.static_base_logits.shape:
            raise ValueError("base logits differ from captured shape")
        self.static_prefix_ids.copy_(prefix_ids)
        self.static_parallel_hiddens.copy_(parallel_hiddens)
        self.static_base_logits.copy_(base_logits)
        self.graph.replay()
        return self.static_output.clone()
