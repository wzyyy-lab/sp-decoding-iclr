"""A lightweight tuned lens for candidate scoring at an intermediate target layer."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TargetTunedExitHead(nn.Module):
    """Map an intermediate target residual stream into the final embedding space."""

    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        final_norm_weight: Tensor,
        rms_epsilon: float,
    ) -> None:
        super().__init__()
        if rank < 1 or rank > hidden_size:
            raise ValueError("rank must be within [1, hidden_size]")
        if final_norm_weight.shape != (hidden_size,):
            raise ValueError("final norm weight has the wrong shape")
        self.hidden_size = int(hidden_size)
        self.rms_epsilon = float(rms_epsilon)
        self.channel_scale = nn.Parameter(torch.ones(hidden_size))
        self.channel_bias = nn.Parameter(torch.zeros(hidden_size))
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "final_norm_weight", final_norm_weight.detach().float().clone()
        )

    def transform(self, hidden: Tensor) -> Tensor:
        hidden = hidden.float()
        residual = self.up(F.silu(self.down(hidden)))
        adapted = hidden * self.channel_scale + self.channel_bias + residual
        variance = adapted.square().mean(dim=-1, keepdim=True)
        normalized = adapted * torch.rsqrt(variance + self.rms_epsilon)
        return normalized * self.final_norm_weight

    def score(self, hidden: Tensor, candidate_embeddings: Tensor) -> Tensor:
        transformed = self.transform(hidden)
        scale = self.logit_scale.clamp(-2.0, 2.0).exp()
        return F.linear(transformed, candidate_embeddings.float()) * scale


__all__ = ["TargetTunedExitHead"]
