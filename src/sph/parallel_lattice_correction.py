"""Lightweight block-parallel replacement for Domino's sequential GRU head.

The module consumes only draft-side tensors.  It predicts all correction codes
in parallel and emits one ordinary token chain; target verification is outside
this module and remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ParallelCorrectionOutput:
    """Outputs needed by distillation, diagnostics, and greedy deployment."""

    token_ids: Tensor
    correction_codes: Tensor
    route_weights: Tensor
    candidate_ids: Tensor
    candidate_logits: Tensor
    corrected_logits: Tensor | None


class _FusedAttention(nn.Module):
    """Small SDPA layer used for both local slots and global mixing."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        batch, query_length, _ = query.shape
        key_length = key_value.shape[1]
        q = self.query(query).view(
            batch, query_length, self.heads, self.head_width
        ).transpose(1, 2)
        k = self.key(key_value).view(
            batch, key_length, self.heads, self.head_width
        ).transpose(1, 2)
        v = self.value(key_value).view(
            batch, key_length, self.heads, self.head_width
        ).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(q, k, v)
        mixed = mixed.transpose(1, 2).reshape(
            batch, query_length, self.width
        )
        return self.output(mixed)


class _GlobalBlock(nn.Module):
    """Exactly one pre-norm global block with a width-2x feed-forward."""

    def __init__(self, width: int, heads: int, feed_forward_width: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = _FusedAttention(width, heads)
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, feed_forward_width, bias=False),
            nn.SiLU(),
            nn.Linear(feed_forward_width, width, bias=False),
        )

    def forward(self, states: Tensor) -> Tensor:
        normalized = self.attention_norm(states)
        states = states + self.attention(normalized, normalized)
        return states + self.feed_forward(self.feed_forward_norm(states))


class ParallelLatticeCorrectionHead(nn.Module):
    """Four-mode, one-block PLC head for fixed B16 Domino deployment.

    ``w_h`` and ``w_out`` are references to the frozen released correction
    basis.  They are registered as non-persistent buffers so a student
    checkpoint contains only the new sub-million-parameter encoder.
    """

    def __init__(
        self,
        *,
        w_h: Tensor,
        w_out: Tensor,
        token_embeddings: Tensor | None = None,
        use_full_hidden: bool = False,
        max_positions: int = 15,
        candidates: int = 16,
        modes: int = 4,
        width: int = 128,
        heads: int = 4,
        feed_forward_width: int = 256,
        global_layers: int = 1,
    ) -> None:
        super().__init__()
        if w_h.ndim != 2 or w_out.ndim != 2:
            raise ValueError("w_h and w_out must be matrices")
        code_width, hidden_width = w_h.shape
        if w_out.shape[1] != code_width:
            raise ValueError("w_out input dimension must match w_h output")
        if token_embeddings is not None and token_embeddings.shape != (
            w_out.shape[0],
            hidden_width,
        ):
            raise ValueError(
                "token_embeddings must have shape [vocab_size, hidden_width]"
            )
        if (
            max_positions < 1
            or candidates < 2
            or modes < 1
            or global_layers < 1
        ):
            raise ValueError("invalid lattice shape")
        self.hidden_width = int(hidden_width)
        self.code_width = int(code_width)
        self.vocab_size = int(w_out.shape[0])
        self.max_positions = max_positions
        self.candidates = candidates
        self.modes = modes
        self.width = width
        self.global_layers = global_layers
        self.use_full_hidden = use_full_hidden

        # These tensors share the released head storage and are deliberately
        # excluded from the student checkpoint.
        self.register_buffer("w_h", w_h.detach(), persistent=False)
        self.register_buffer("w_out", w_out.detach(), persistent=False)
        self.register_buffer(
            "token_embeddings",
            (
                token_embeddings.detach()
                if token_embeddings is not None
                else torch.empty(
                    0,
                    hidden_width,
                    device=w_out.device,
                    dtype=w_out.dtype,
                )
            ),
            persistent=False,
        )
        self.register_buffer(
            "projected_lexical_table",
            torch.empty(0, width, device=w_out.device, dtype=w_out.dtype),
            persistent=False,
        )

        self.lexical_projection = nn.Linear(code_width, width, bias=False)
        self.semantic_projection = (
            nn.Linear(hidden_width, width, bias=False)
            if token_embeddings is not None
            else None
        )
        self.hidden_projection = nn.Linear(code_width, width, bias=False)
        self.context_projection = (
            nn.Linear(hidden_width, width, bias=False)
            if use_full_hidden
            else None
        )
        self.scalar_projection = nn.Linear(6, width, bias=False)
        self.rank_embedding = nn.Embedding(candidates, width)
        self.position_embedding = nn.Embedding(max_positions, width)
        self.input_norm = nn.LayerNorm(width)

        self.mode_queries = nn.Parameter(torch.empty(modes, width))
        self.local_attention = _FusedAttention(width, heads)
        self.local_norm = nn.LayerNorm(width)

        self.route_token = nn.Parameter(torch.empty(1, 1, width))
        self.global_blocks = nn.ModuleList(
            [
                _GlobalBlock(width, heads, feed_forward_width)
                for _ in range(global_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(width)
        self.mode_route = nn.Linear(width, width, bias=False)
        self.code_projection = nn.Linear(width, code_width, bias=False)

        nn.init.normal_(self.mode_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.route_token, mean=0.0, std=0.02)
        nn.init.normal_(self.rank_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def active_parameter_count(self) -> int:
        return self.w_h.numel() + self.w_out.numel() + self.trainable_parameter_count

    @torch.no_grad()
    def prepare_inference(self) -> None:
        """Preproject the frozen lexical table after training is frozen."""

        table = F.linear(self.w_out, self.lexical_projection.weight)
        if self.semantic_projection is not None:
            table = table + F.linear(
                self.token_embeddings, self.semantic_projection.weight
            )
        self.projected_lexical_table = table.contiguous()

    def clear_inference_table(self) -> None:
        """Return to the differentiable training path."""

        self.projected_lexical_table = torch.empty(
            0,
            self.width,
            device=self.w_out.device,
            dtype=self.w_out.dtype,
        )

    def _candidate_features(self, top_logits: Tensor) -> Tensor:
        local_log_probabilities = F.log_softmax(top_logits, dim=-1)
        local_probabilities = local_log_probabilities.exp()
        centered = top_logits - top_logits.mean(dim=-1, keepdim=True)
        standardized = centered / centered.square().mean(
            dim=-1, keepdim=True
        ).add(1e-6).sqrt()
        gap = top_logits[..., :1] - top_logits
        gap = gap / gap[..., -1:].clamp_min(1e-3)
        rank = torch.linspace(
            0.0,
            1.0,
            self.candidates,
            device=top_logits.device,
            dtype=top_logits.dtype,
        ).view(1, 1, self.candidates).expand_as(top_logits)
        entropy = -(
            local_probabilities * local_log_probabilities
        ).sum(dim=-1, keepdim=True).expand_as(top_logits)
        cumulative_mass = local_probabilities.cumsum(dim=-1)
        return torch.stack(
            [
                standardized,
                local_log_probabilities,
                gap,
                rank,
                entropy,
                cumulative_mass,
            ],
            dim=-1,
        )

    def _lexical_nodes(self, candidate_ids: Tensor) -> Tensor:
        if self.projected_lexical_table.shape[0] == self.vocab_size:
            return F.embedding(candidate_ids, self.projected_lexical_table)
        lexical_codes = F.embedding(candidate_ids, self.w_out)
        lexical = self.lexical_projection(lexical_codes)
        if self.semantic_projection is not None:
            semantic = F.embedding(candidate_ids, self.token_embeddings)
            lexical = lexical + self.semantic_projection(semantic)
        return lexical

    def forward(
        self,
        *,
        parallel_hiddens: Tensor,
        base_logits: Tensor,
        anchor_ids: Tensor,
        prefix_ids: Tensor,
        return_logits: bool = True,
    ) -> ParallelCorrectionOutput:
        batch, positions, hidden_width = parallel_hiddens.shape
        if hidden_width != self.hidden_width:
            raise ValueError("parallel hidden width does not match w_h")
        if positions > self.max_positions:
            raise ValueError("position count exceeds configured maximum")
        if base_logits.shape != (batch, positions, self.vocab_size):
            raise ValueError("base logits have the wrong shape")
        if anchor_ids.shape != (batch,) or prefix_ids.shape != (batch,):
            raise ValueError("anchor_ids and prefix_ids must have shape [batch]")

        top_logits, candidate_ids = torch.topk(
            base_logits, self.candidates, dim=-1
        )
        z = F.linear(parallel_hiddens, self.w_h)
        scalar_features = self._candidate_features(top_logits).to(z.dtype)

        lexical = self._lexical_nodes(candidate_ids)
        hidden_state = self.hidden_projection(z)
        if self.context_projection is not None:
            normalized_hidden = parallel_hiddens * torch.rsqrt(
                parallel_hiddens.float().square().mean(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
            ).to(parallel_hiddens.dtype)
            hidden_state = hidden_state + self.context_projection(
                normalized_hidden
            )
        hidden_nodes = hidden_state.unsqueeze(-2)
        scalar_nodes = self.scalar_projection(scalar_features)
        rank = self.rank_embedding(
            torch.arange(self.candidates, device=base_logits.device)
        ).view(1, 1, self.candidates, self.width)
        position = self.position_embedding(
            torch.arange(positions, device=base_logits.device)
        ).view(1, positions, 1, self.width)
        nodes = self.input_norm(
            lexical + hidden_nodes + scalar_nodes + rank + position
        )

        # Treat all B*L local lattices as one attention batch.  There is no
        # Python loop over positions and no selected-token feedback.
        local_batch = batch * positions
        flat_nodes = nodes.reshape(
            local_batch, self.candidates, self.width
        )
        queries = self.mode_queries.view(1, self.modes, self.width).expand(
            local_batch, -1, -1
        )
        modes = self.local_norm(
            queries + self.local_attention(queries, flat_nodes)
        ).view(batch, positions, self.modes, self.width)

        if self.projected_lexical_table.shape[0] == self.vocab_size:
            boundary = 0.5 * (
                F.embedding(anchor_ids, self.projected_lexical_table)
                + F.embedding(prefix_ids, self.projected_lexical_table)
            )
        else:
            boundary_codes = 0.5 * (
                F.embedding(anchor_ids, self.w_out)
                + F.embedding(prefix_ids, self.w_out)
            )
            boundary = self.lexical_projection(boundary_codes)
            if self.semantic_projection is not None:
                boundary_semantics = 0.5 * (
                    F.embedding(anchor_ids, self.token_embeddings)
                    + F.embedding(prefix_ids, self.token_embeddings)
                )
                boundary = boundary + self.semantic_projection(
                    boundary_semantics
                )
        route = self.route_token.expand(batch, -1, -1) + boundary.unsqueeze(1)
        global_input = torch.cat(
            [route, modes.reshape(batch, positions * self.modes, self.width)],
            dim=1,
        )
        global_output = global_input
        for block in self.global_blocks:
            global_output = block(global_output)
        global_output = self.output_norm(global_output)
        route = global_output[:, 0]
        modes = global_output[:, 1:].view(
            batch, positions, self.modes, self.width
        )

        route_scores = torch.einsum(
            "bd,blqd->blq",
            route,
            self.mode_route(modes),
        ) / math.sqrt(self.width)
        route_weights = torch.softmax(route_scores, dim=-1)
        mode_codes = self.code_projection(modes)
        correction_codes = torch.einsum(
            "blq,blqc->blc", route_weights, mode_codes
        )

        correction_hidden = F.silu(z + correction_codes)
        correction_logits = F.linear(correction_hidden, self.w_out)
        corrected_logits = base_logits + correction_logits
        token_ids = corrected_logits.argmax(dim=-1)
        return ParallelCorrectionOutput(
            token_ids=token_ids,
            correction_codes=correction_codes,
            route_weights=route_weights,
            candidate_ids=candidate_ids,
            candidate_logits=top_logits,
            corrected_logits=corrected_logits if return_logits else None,
        )


class PLCCorrectionGraphRunner:
    """Drop-in fixed-shape graph runner for Domino ``spec_generate``.

    Its call signature matches the released ``DraftCorrectionGraphRunner``:
    the two prefix IDs are ``[accepted_anchor, base_prefix]`` and the returned
    tensor contains all 15 corrected draft tokens.
    """

    def __init__(
        self,
        *,
        head: ParallelLatticeCorrectionHead,
        batch_size: int,
        positions: int,
        device: torch.device,
    ) -> None:
        if not torch.cuda.is_available() or device.type != "cuda":
            raise RuntimeError("PLCCorrectionGraphRunner requires CUDA")
        if positions > head.max_positions:
            raise ValueError("positions exceed the PLC head capacity")
        if head.projected_lexical_table.shape[0] != head.vocab_size:
            raise RuntimeError("call head.prepare_inference() before graph capture")
        self.head = head
        self.batch_size = batch_size
        self.positions = positions
        dtype = head.w_out.dtype
        self.static_prefix_ids = torch.zeros(
            batch_size, 2, dtype=torch.long, device=device
        )
        self.static_parallel_hiddens = torch.zeros(
            batch_size,
            positions,
            head.hidden_width,
            dtype=dtype,
            device=device,
        )
        self.static_base_logits = torch.zeros(
            batch_size,
            positions,
            head.vocab_size,
            dtype=dtype,
            device=device,
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
        return self.head(
            parallel_hiddens=self.static_parallel_hiddens,
            base_logits=self.static_base_logits,
            anchor_ids=self.static_prefix_ids[:, 0],
            prefix_ids=self.static_prefix_ids[:, 1],
            return_logits=False,
        ).token_ids

    @torch.inference_mode()
    def __call__(
        self,
        prefix_ids: Tensor,
        parallel_hiddens: Tensor,
        base_logits: Tensor,
    ) -> Tensor:
        if prefix_ids.shape != self.static_prefix_ids.shape:
            raise ValueError("prefix_ids differ from captured shape")
        if parallel_hiddens.shape != self.static_parallel_hiddens.shape:
            raise ValueError("parallel_hiddens differ from captured shape")
        if base_logits.shape != self.static_base_logits.shape:
            raise ValueError("base_logits differ from captured shape")
        self.static_prefix_ids.copy_(prefix_ids)
        self.static_parallel_hiddens.copy_(parallel_hiddens)
        self.static_base_logits.copy_(base_logits)
        self.graph.replay()
        return self.static_output.clone()
