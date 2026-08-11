"""Acceptance-aware global reranker for single-chain Domino beam paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class BeamRerankerOutput:
    match_logits: Tensor
    predicted_utility: Tensor
    selection_scores: Tensor


class DominoBeamPathReranker(nn.Module):
    """Predict prefix survival from a complete draft-only candidate path."""

    def __init__(
        self,
        *,
        hidden_size: int,
        causal_state_size: int,
        token_feature_size: int,
        horizon: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.horizon = int(horizon)
        self.hidden_proj = nn.Linear(hidden_size, model_dim, bias=False)
        self.causal_proj = nn.Linear(causal_state_size, model_dim, bias=False)
        self.token_proj = nn.Linear(token_feature_size, model_dim, bias=False)
        self.scalar_proj = nn.Linear(2, model_dim, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(horizon, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.match_residual = nn.Linear(model_dim, 1)
        self.path_residual = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.match_residual.weight)
        nn.init.zeros_(self.match_residual.bias)
        nn.init.zeros_(self.path_residual.weight)
        nn.init.zeros_(self.path_residual.bias)

    def forward(
        self,
        *,
        hidden: Tensor,
        causal_states: Tensor,
        token_features: Tensor,
        edge_log_probs: Tensor,
        released_indicator: Tensor,
    ) -> BeamRerankerOutput:
        if hidden.ndim != 2 or hidden.shape[0] != self.horizon:
            raise ValueError("hidden must have shape [horizon, hidden_size]")
        if token_features.ndim != 3 or token_features.shape[1] != self.horizon:
            raise ValueError(
                "token_features must have shape [paths, horizon, token_feature_size]"
            )
        if causal_states.shape[:2] != token_features.shape[:2]:
            raise ValueError("causal states must match the path and position axes")
        if edge_log_probs.shape != token_features.shape[:2]:
            raise ValueError("edge log probabilities must match path and position axes")
        if released_indicator.shape != edge_log_probs.shape:
            raise ValueError("released indicator must match edge log probabilities")

        paths = int(token_features.shape[0])
        shared = self.hidden_proj(hidden.float())[None].expand(paths, -1, -1)
        causal = self.causal_proj(causal_states.float())
        lexical = self.token_proj(token_features.float())
        scalars = torch.stack(
            [edge_log_probs.float(), released_indicator.float()], dim=-1
        )
        x = (
            shared
            + causal
            + lexical
            + self.scalar_proj(scalars)
            + self.position_embedding[None]
        )
        encoded = self.final_norm(self.encoder(x))

        base_probability = edge_log_probs.float().exp().clamp(1e-5, 1.0 - 1e-5)
        base_match_logit = torch.logit(base_probability)
        match_logits = base_match_logit + self.match_residual(encoded).squeeze(-1)
        match_probability = match_logits.sigmoid()
        predicted_utility = match_probability.cumprod(dim=-1).sum(dim=-1)
        path_delta = self.path_residual(encoded.mean(dim=1)).squeeze(-1)
        selection_scores = predicted_utility + path_delta
        return BeamRerankerOutput(
            match_logits=match_logits,
            predicted_utility=predicted_utility,
            selection_scores=selection_scores,
        )


class DominoBeamSetReranker(nn.Module):
    """Jointly compare all surviving paths before selecting one draft chain.

    The proposal paths remain fixed and draft-only.  Unlike
    :class:`DominoBeamPathReranker`, this module treats the path axis as a set,
    so every path score can depend on its alternatives.  Its zero-initialized
    residual exactly preserves the full-vocabulary gamma baseline at step 0.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        causal_state_size: int,
        token_feature_size: int,
        horizon: int,
        model_dim: int = 192,
        num_heads: int = 6,
        position_layers: int = 2,
        set_layers: int = 2,
        base_gamma: float = 0.75,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if position_layers < 1 or set_layers < 1:
            raise ValueError("position_layers and set_layers must be positive")
        if not 0.0 < base_gamma <= 1.0:
            raise ValueError("base_gamma must be in (0, 1]")
        self.horizon = int(horizon)
        self.base_gamma = float(base_gamma)
        self.hidden_proj = nn.Linear(hidden_size, model_dim, bias=False)
        self.causal_proj = nn.Linear(causal_state_size, model_dim, bias=False)
        self.token_proj = nn.Linear(token_feature_size, model_dim, bias=False)
        self.edge_proj = nn.Linear(2, model_dim, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(horizon, model_dim))

        position_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.position_encoder = nn.TransformerEncoder(
            position_layer,
            num_layers=position_layers,
            enable_nested_tensor=False,
        )
        self.position_norm = nn.LayerNorm(model_dim)
        self.pool_logits = nn.Parameter(
            torch.linspace(0.0, -2.0, steps=horizon)
        )
        self.path_stat_proj = nn.Linear(4, model_dim, bias=False)
        set_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=set_layers,
            enable_nested_tensor=False,
        )
        self.set_norm = nn.LayerNorm(model_dim)
        self.selection_residual = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.selection_residual.weight)
        nn.init.zeros_(self.selection_residual.bias)

    def forward(
        self,
        *,
        hidden: Tensor,
        causal_states: Tensor,
        token_features: Tensor,
        edge_log_probs: Tensor,
        released_indicator: Tensor,
    ) -> BeamRerankerOutput:
        if hidden.ndim != 2 or hidden.shape[0] != self.horizon:
            raise ValueError("hidden must have shape [horizon, hidden_size]")
        if token_features.ndim != 3 or token_features.shape[1] != self.horizon:
            raise ValueError(
                "token_features must have shape [paths, horizon, token_feature_size]"
            )
        if causal_states.shape[:2] != token_features.shape[:2]:
            raise ValueError("causal states must match the path and position axes")
        if edge_log_probs.shape != token_features.shape[:2]:
            raise ValueError("edge log probabilities must match path and position axes")
        if released_indicator.shape != edge_log_probs.shape:
            raise ValueError("released indicator must match edge log probabilities")

        paths = int(token_features.shape[0])
        shared = self.hidden_proj(hidden.float())[None].expand(paths, -1, -1)
        scalars = torch.stack(
            [edge_log_probs.float(), released_indicator.float()], dim=-1
        )
        x = (
            shared
            + self.causal_proj(causal_states.float())
            + self.token_proj(token_features.float())
            + self.edge_proj(scalars)
            + self.position_embedding[None]
        )
        encoded = self.position_norm(self.position_encoder(x))
        pool_weights = self.pool_logits.softmax(dim=0)
        pooled = (encoded * pool_weights[None, :, None]).sum(dim=1)

        axis = torch.arange(self.horizon, device=edge_log_probs.device)
        gamma_weights = edge_log_probs.new_tensor(self.base_gamma).pow(axis)
        gamma_score = (edge_log_probs.float() * gamma_weights[None]).sum(dim=-1)
        map_score = edge_log_probs.float().sum(dim=-1)
        survival_score = edge_log_probs.float().cumsum(dim=-1).exp().sum(dim=-1)
        released_fraction = released_indicator.float().mean(dim=-1)
        path_stats = torch.stack(
            [gamma_score, map_score, survival_score, released_fraction], dim=-1
        )
        # Only relative path statistics are useful to a within-block chooser.
        path_stats = (path_stats - path_stats.mean(dim=0, keepdim=True)) / (
            path_stats.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
        )
        set_input = pooled + self.path_stat_proj(path_stats)
        compared = self.set_norm(self.set_encoder(set_input[None])[0])
        residual = self.selection_residual(compared).squeeze(-1)
        selection_scores = gamma_score + residual

        base_probability = edge_log_probs.float().exp().clamp(1e-5, 1.0 - 1e-5)
        match_logits = torch.logit(base_probability)
        predicted_utility = base_probability.cumprod(dim=-1).sum(dim=-1)
        return BeamRerankerOutput(
            match_logits=match_logits,
            predicted_utility=predicted_utility,
            selection_scores=selection_scores,
        )


def acceptance_aware_reranker_loss(
    output: BeamRerankerOutput,
    paths: Tensor,
    gold: Tensor,
    *,
    list_temperature: float = 0.5,
    list_weight: float = 1.0,
    frontier_weight: float = 0.25,
) -> tuple[Tensor, dict[str, Tensor]]:
    if paths.ndim != 2 or gold.ndim != 1 or paths.shape[1] != gold.shape[0]:
        raise ValueError("expected paths [count, horizon] and gold [horizon]")
    if list_temperature <= 0 or list_weight < 0 or frontier_weight < 0:
        raise ValueError("invalid reranker loss hyperparameters")
    matches = paths.eq(gold[None])
    true_lengths = matches.long().cumprod(dim=-1).sum(dim=-1)
    target_distribution = torch.softmax(
        true_lengths.float() / list_temperature, dim=0
    )
    list_loss = -(
        target_distribution
        * torch.log_softmax(output.selection_scores / list_temperature, dim=0)
    ).sum()

    positions = paths.shape[1]
    axis = torch.arange(positions, device=paths.device)[None]
    frontier_mask = axis <= true_lengths[:, None].clamp(max=positions - 1)
    full_correct = true_lengths.eq(positions)
    frontier_mask = torch.where(
        full_correct[:, None], torch.ones_like(frontier_mask), frontier_mask
    )
    frontier_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        output.match_logits,
        matches.float(),
        reduction="none",
    )
    frontier_loss = (
        frontier_bce * frontier_mask.to(frontier_bce.dtype)
    ).sum() / frontier_mask.sum().clamp_min(1)
    loss = list_weight * list_loss + frontier_weight * frontier_loss
    return loss, {
        "list_loss": list_loss.detach(),
        "frontier_loss": frontier_loss.detach(),
        "beam_oracle_length": true_lengths.max().detach().float(),
        "selected_true_length": true_lengths[
            output.selection_scores.detach().argmax()
        ].detach().float(),
        "mean_true_length": true_lengths.float().mean().detach(),
    }


def oracle_path_ranking_loss(
    output: BeamRerankerOutput,
    paths: Tensor,
    gold: Tensor,
    *,
    temperature: float = 0.5,
    regret_weight: float = 0.25,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Put probability mass on any maximum-acceptance path in the set."""

    if paths.ndim != 2 or gold.ndim != 1 or paths.shape[1] != gold.shape[0]:
        raise ValueError("expected paths [count, horizon] and gold [horizon]")
    if temperature <= 0 or regret_weight < 0:
        raise ValueError("invalid oracle-ranking loss hyperparameters")
    matches = paths.eq(gold[None])
    true_lengths = matches.long().cumprod(dim=-1).sum(dim=-1)
    best_length = true_lengths.max()
    best_mask = true_lengths.eq(best_length)
    scaled_scores = output.selection_scores / temperature
    best_mass_loss = torch.logsumexp(scaled_scores, dim=0) - torch.logsumexp(
        scaled_scores[best_mask], dim=0
    )
    probabilities = torch.softmax(scaled_scores, dim=0)
    normalized_regret = (best_length - true_lengths).float() / float(paths.shape[1])
    expected_regret = (probabilities * normalized_regret).sum()
    loss = best_mass_loss + regret_weight * expected_regret
    return loss, {
        "best_mass_loss": best_mass_loss.detach(),
        "expected_regret": expected_regret.detach(),
        "beam_oracle_length": best_length.detach().float(),
        "selected_true_length": true_lengths[
            output.selection_scores.detach().argmax()
        ].detach().float(),
        "mean_true_length": true_lengths.float().mean().detach(),
    }


__all__ = [
    "BeamRerankerOutput",
    "DominoBeamPathReranker",
    "DominoBeamSetReranker",
    "acceptance_aware_reranker_loss",
    "oracle_path_ranking_loss",
]
