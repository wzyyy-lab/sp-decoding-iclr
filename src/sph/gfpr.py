"""Greedy-frontier policy replay primitives for all-position Domino.

The functions in this module deliberately separate three contracts:

* Stages A--C decode over the full vocabulary with the adapted Domino head.
* Top-K sets are used only for oracle and optional contraction diagnostics.
* Training touches the accepted prefix and the current first rejection only.

Keeping these contracts explicit prevents a candidate-restricted training loss
from silently replacing the deployed full-vocabulary policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class GFPRDecodeOutput:
    """One full-vocabulary causal rollout plus its DFlash Top-K lattice."""

    token_ids: Tensor
    base_topk_ids: Tensor
    base_topk_logits: Tensor


@dataclass(frozen=True)
class FrontierLossOutput:
    """Scalar loss and detached diagnostics for a current frontier."""

    loss: Tensor
    repair_loss: Tensor
    keep_loss: Tensor
    frontier: Tensor
    accepted_positions: Tensor
    repairable_blocks: Tensor
    mean_gold_margin: Tensor


def accepted_lengths(proposals: Tensor, gold: Tensor) -> Tensor:
    """Return the exact greedy accepted-prefix length for every row."""

    if proposals.shape != gold.shape or proposals.ndim != 2:
        raise ValueError("proposals and gold must share shape [batch, positions]")
    return proposals.eq(gold).to(torch.long).cumprod(dim=-1).sum(dim=-1)


def next_anchor_offsets(offsets: Tensor, accepted: Tensor) -> Tensor:
    """Advance by accepted draft tokens plus the target correction/bonus."""

    if offsets.shape != accepted.shape:
        raise ValueError("offsets and accepted lengths must share shape")
    if bool(torch.any(accepted < 0)):
        raise ValueError("accepted lengths must be non-negative")
    return offsets.to(torch.long) + accepted.to(torch.long) + 1


def _state_for_head(domino: Any, state: Tensor) -> Tensor:
    state_for_head = state.transpose(0, 1)
    if bool(getattr(domino, "use_bias_norm", False)):
        state_for_head = domino.bias_norm(state_for_head)
    return state_for_head


def _correction(domino: Any, hidden: Tensor, state_for_head: Tensor) -> Tensor:
    bias = domino.embed_proj(torch.cat([hidden, state_for_head], dim=-1))
    if bool(getattr(domino, "use_bias_gate", False)):
        bias = torch.sigmoid(domino.bias_gate(hidden)) * bias
    return bias


def all_position_teacher_logits(
    *,
    domino: Any,
    target_weight: Tensor,
    anchors: Tensor,
    gold: Tensor,
    hidden: Tensor,
    position_zero_scale: Tensor | float,
    return_base_logits: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Return all-16 GFPR logits under a clean teacher prefix.

    The GRU is reset for each batch item, consumes the anchor first, and then
    consumes ``gold[:, i]`` after decision ``i``.  Up to the current first
    mismatch this is exactly the deployed current-policy state, because all
    earlier selected tokens equal gold there.
    """

    if hidden.ndim != 3 or gold.shape != hidden.shape[:2]:
        raise ValueError("hidden/gold must have shapes [B,L,D] and [B,L]")
    if anchors.shape != (hidden.shape[0],):
        raise ValueError("anchors must have shape [batch]")
    prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
    prefix_embeddings = F.embedding(prefix_ids, target_weight.detach())
    states, _ = domino.prefix_gru(prefix_embeddings)
    state_for_head = states
    if bool(getattr(domino, "use_bias_norm", False)):
        state_for_head = domino.bias_norm(state_for_head)
    correction = _correction(domino, hidden, state_for_head)
    base_logits = F.linear(hidden, target_weight.detach())
    scale = torch.as_tensor(
        position_zero_scale, dtype=correction.dtype, device=correction.device
    )
    first = base_logits[:, :1] + scale * correction[:, :1]
    suffix = base_logits[:, 1:] + correction[:, 1:]
    combined = torch.cat([first, suffix], dim=1).float()
    if return_base_logits:
        return combined, base_logits.float()
    return combined


@torch.no_grad()
def all_position_onpolicy_decode(
    *,
    domino: Any,
    target_weight: Tensor,
    anchors: Tensor,
    hidden: Tensor,
    position_zero_scale: Tensor | float,
    topk: int = 16,
) -> GFPRDecodeOutput:
    """Decode all positions with selected-token feedback and a reset GRU."""

    if hidden.ndim != 3 or anchors.shape != (hidden.shape[0],):
        raise ValueError("invalid hidden or anchor shape")
    if not 1 <= topk <= target_weight.shape[0]:
        raise ValueError("topk lies outside the vocabulary")
    base_logits = F.linear(hidden, target_weight)
    top_logits, top_ids = base_logits.float().topk(topk, dim=-1)
    _, state = domino.prefix_gru(
        F.embedding(anchors[:, None], target_weight)
    )
    scale = torch.as_tensor(
        position_zero_scale, dtype=base_logits.dtype, device=base_logits.device
    )
    selected: list[Tensor] = []
    for position in range(hidden.shape[1]):
        state_for_head = _state_for_head(domino, state)
        bias = _correction(
            domino,
            hidden[:, position : position + 1],
            state_for_head,
        )
        if position == 0:
            scores = base_logits[:, :1] + scale * bias
        else:
            scores = base_logits[:, position : position + 1] + bias
        token = scores.float().argmax(dim=-1)
        selected.append(token)
        if position + 1 < hidden.shape[1]:
            _, state = domino.prefix_gru(
                F.embedding(token, target_weight), state
            )
    return GFPRDecodeOutput(
        token_ids=torch.cat(selected, dim=1),
        base_topk_ids=top_ids,
        base_topk_logits=top_logits,
    )


def frontier_masks(logits: Tensor, gold: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return current frontier index, protected prefix, and repair mask."""

    if logits.ndim != 3 or gold.shape != logits.shape[:2]:
        raise ValueError("expected logits [B,L,V] and gold [B,L]")
    predictions = logits.detach().argmax(dim=-1)
    batch, positions = gold.shape
    axes = torch.arange(positions, device=gold.device).view(1, -1)
    sentinel = torch.full_like(gold, positions)
    mismatch_axes = torch.where(
        predictions.ne(gold), axes.expand(batch, -1), sentinel
    )
    frontier = mismatch_axes.min(dim=-1).values
    protected = axes < frontier[:, None]
    repair = axes == frontier[:, None]
    return frontier, protected, repair


def normalized_frontier_margin_loss(
    logits: Tensor,
    gold: Tensor,
    *,
    break_margin: float = 1e-4,
    keep_margin: float = 0.05,
    break_weight: float = 1.0,
    keep_weight: float = 0.1,
    block_weights: Tensor | None = None,
) -> FrontierLossOutput:
    """Repair one current rejection with a capped accepted-prefix budget."""

    if min(break_margin, keep_margin, break_weight, keep_weight) < 0:
        raise ValueError("margins and weights must be non-negative")
    frontier, protected, repair = frontier_masks(logits, gold)
    values, ids = logits.float().topk(2, dim=-1)
    gold_scores = logits.float().gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    competitor_scores = torch.where(ids[..., 0].eq(gold), values[..., 1], values[..., 0])
    margins = gold_scores - competitor_scores
    per_position_keep = torch.relu(keep_margin - margins) * protected.to(margins.dtype)
    keep_denominator = frontier.clamp_min(1).to(margins.dtype)
    per_block_keep = per_position_keep.sum(dim=-1) / keep_denominator
    per_block_repair = (
        torch.relu(break_margin - margins) * repair.to(margins.dtype)
    ).sum(dim=-1)
    if block_weights is None:
        weights = torch.full_like(per_block_keep, 1.0 / per_block_keep.numel())
    else:
        if block_weights.shape != per_block_keep.shape:
            raise ValueError("block_weights must have shape [batch]")
        weights = block_weights.float()
        if bool(torch.any(weights < 0)) or float(weights.sum()) <= 0:
            raise ValueError("block_weights must be non-negative with positive sum")
        weights = weights / weights.sum()
    repair_loss = (weights * per_block_repair).sum()
    keep_loss = (weights * per_block_keep).sum()
    loss = break_weight * repair_loss + keep_weight * keep_loss
    return FrontierLossOutput(
        loss=loss,
        repair_loss=repair_loss.detach(),
        keep_loss=keep_loss.detach(),
        frontier=frontier.detach(),
        accepted_positions=protected.sum(dim=-1).detach(),
        repairable_blocks=repair.any(dim=-1).detach(),
        mean_gold_margin=margins.detach().mean(),
    )


def topk_oracle_matches(
    *,
    base_topk_ids: Tensor,
    released_ids: Tensor,
    gold: Tensor,
) -> dict[str, Tensor]:
    """Return per-position coverage for DFlash K16 and Domino unions."""

    if base_topk_ids.ndim != 3 or base_topk_ids.shape[-1] < 16:
        raise ValueError("base_topk_ids must contain at least 16 candidates")
    if released_ids.shape != gold.shape or gold.shape != base_topk_ids.shape[:2]:
        raise ValueError("released/gold shapes do not match the candidate lattice")
    base16 = base_topk_ids[..., :16]
    base_match = base16.eq(gold.unsqueeze(-1)).any(dim=-1)
    released_match = released_ids.eq(gold)
    action_in_base = base16.eq(released_ids.unsqueeze(-1)).any(dim=-1)
    # K17 is a set union.  K16 keeps all base candidates when the released
    # action is already present; otherwise it replaces DFlash rank 16.
    k17_match = base_match | released_match
    base15_match = base16[..., :15].eq(gold.unsqueeze(-1)).any(dim=-1)
    k16_match = torch.where(action_in_base, base_match, base15_match | released_match)
    return {"base16": base_match, "k17": k17_match, "k16": k16_match}


def oracle_prefix_lengths(matches: Tensor) -> Tensor:
    """Convert per-position oracle coverage to greedy prefix lengths."""

    if matches.ndim != 2 or matches.dtype != torch.bool:
        raise ValueError("matches must be a boolean [batch, positions] tensor")
    return matches.to(torch.long).cumprod(dim=-1).sum(dim=-1)


def paired_prompt_summary(
    sample_ids: Sequence[str],
    baseline_lengths: Sequence[int],
    current_lengths: Sequence[int],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int | list[float]]:
    """Prompt-cluster paired EAL, bootstrap interval, and explicit harm."""

    if not (
        len(sample_ids) == len(baseline_lengths) == len(current_lengths)
    ) or not sample_ids:
        raise ValueError("paired metric inputs must be non-empty and equal length")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    grouped_base: dict[str, list[int]] = defaultdict(list)
    grouped_current: dict[str, list[int]] = defaultdict(list)
    for sample_id, baseline, current in zip(
        sample_ids, baseline_lengths, current_lengths, strict=True
    ):
        grouped_base[str(sample_id)].append(int(baseline))
        grouped_current[str(sample_id)].append(int(current))
    if grouped_base.keys() != grouped_current.keys():
        raise RuntimeError("paired prompt groups differ")
    prompt_ids = sorted(grouped_base)
    prompt_base = torch.tensor(
        [sum(grouped_base[key]) / len(grouped_base[key]) for key in prompt_ids],
        dtype=torch.float64,
    )
    prompt_current = torch.tensor(
        [
            sum(grouped_current[key]) / len(grouped_current[key])
            for key in prompt_ids
        ],
        dtype=torch.float64,
    )
    deltas = prompt_current - prompt_base
    generator = torch.Generator().manual_seed(seed)
    draw = torch.randint(
        0,
        len(prompt_ids),
        (bootstrap_samples, len(prompt_ids)),
        generator=generator,
    )
    bootstrap = deltas[draw].mean(dim=-1)
    interval = torch.quantile(
        bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    raw_delta = torch.tensor(current_lengths, dtype=torch.float64) - torch.tensor(
        baseline_lengths, dtype=torch.float64
    )
    gained = float(raw_delta.clamp_min(0).sum())
    lost = float((-raw_delta.clamp_max(0)).sum())
    return {
        "prompts": len(prompt_ids),
        "blocks": len(sample_ids),
        "baseline_eal_prompt_balanced": float(prompt_base.mean()),
        "current_eal_prompt_balanced": float(prompt_current.mean()),
        "paired_delta": float(deltas.mean()),
        "paired_bootstrap_95_interval": [float(interval[0]), float(interval[1])],
        "gained_accepted_tokens": gained,
        "lost_accepted_tokens": lost,
        "lost_to_gained_ratio": lost / gained if gained > 0 else float("inf"),
        "harmful_prompt_fraction": float(deltas.lt(0).float().mean()),
        "improved_prompt_fraction": float(deltas.gt(0).float().mean()),
    }


def adaptation_state_dict(
    domino: Any, position_zero_scale: Tensor | float
) -> dict[str, Any]:
    """Build the portable, head-only GFPR checkpoint payload."""

    payload: dict[str, Any] = {
        "format": "gfpr_head_v1",
        "prefix_gru": domino.prefix_gru.state_dict(),
        "embed_proj": domino.embed_proj.state_dict(),
        "position_zero_scale": torch.as_tensor(position_zero_scale).detach().cpu(),
    }
    if bool(getattr(domino, "use_bias_norm", False)):
        payload["bias_norm"] = domino.bias_norm.state_dict()
    if bool(getattr(domino, "use_bias_gate", False)):
        payload["bias_gate"] = domino.bias_gate.state_dict()
    return payload


def load_adaptation(
    domino: Any,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_target: str | Path | None = None,
    expected_base_domino: str | Path | None = None,
) -> Tensor:
    """Load a GFPR head checkpoint and return its position-zero scale."""

    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if payload.get("format") != "gfpr_head_v1":
        raise ValueError(f"unsupported GFPR checkpoint format: {payload.get('format')!r}")
    expected = {
        "target": expected_target,
        "base_domino": expected_base_domino,
    }
    if any(value is not None for value in expected.values()):
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                "GFPR checkpoint lacks provenance required for a claim-bearing load"
            )
        for field, value in expected.items():
            if value is None:
                continue
            stored = provenance.get(field)
            if stored is None or Path(stored).resolve() != Path(value).resolve():
                raise ValueError(
                    f"GFPR checkpoint provenance {field}={stored!r} does not "
                    f"match {Path(value).resolve()}"
                )
    domino.prefix_gru.load_state_dict(payload["prefix_gru"], strict=True)
    domino.embed_proj.load_state_dict(payload["embed_proj"], strict=True)
    if bool(getattr(domino, "use_bias_norm", False)):
        if "bias_norm" not in payload:
            raise ValueError("GFPR checkpoint omits required bias_norm")
        domino.bias_norm.load_state_dict(payload["bias_norm"], strict=True)
    if bool(getattr(domino, "use_bias_gate", False)):
        if "bias_gate" not in payload:
            raise ValueError("GFPR checkpoint omits required bias_gate")
        domino.bias_gate.load_state_dict(payload["bias_gate"], strict=True)
    return torch.as_tensor(payload["position_zero_scale"]).float()


__all__ = [
    "FrontierLossOutput",
    "GFPRDecodeOutput",
    "accepted_lengths",
    "adaptation_state_dict",
    "all_position_onpolicy_decode",
    "all_position_teacher_logits",
    "frontier_masks",
    "load_adaptation",
    "next_anchor_offsets",
    "normalized_frontier_margin_loss",
    "oracle_prefix_lengths",
    "paired_prompt_summary",
    "topk_oracle_matches",
]
