"""Fixed-shape padded beam-forest primitives for R055.

Every beam path keeps an independent 16-token chain behind one shared anchor.
The representation intentionally duplicates common token prefixes: doing so
removes data-dependent trie construction and makes inputs, masks, positions and
acceptance traversal static for CUDA graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ForestTraversalOutput:
    """Device-local result of independently checking every beam path."""

    accepted: Tensor
    selected_path: Tensor
    next_token: Tensor
    per_path_accepted: Tensor
    posterior_tokens: Tensor


def pack_padded_forest(
    paths: Tensor,
    *,
    anchor_token_id: int,
    prefix_length: int,
    mask_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack one shared anchor plus W independent fixed-horizon chains.

    ``paths`` must already be on the target device.  The returned tensors stay
    on that device and have fixed shapes for a fixed ``[W,H]`` input.
    """

    if paths.ndim != 2 or paths.dtype != torch.long:
        raise ValueError("forest paths must be int64 [width,horizon]")
    width, horizon = paths.shape
    if width < 1 or horizon < 1 or prefix_length < 1:
        raise ValueError("forest width, horizon and prefix must be positive")
    device = paths.device
    anchor = torch.full(
        (1,), int(anchor_token_id), dtype=torch.long, device=device
    )
    input_ids = torch.cat([anchor, paths.reshape(-1)], dim=0)[None]
    rows = 1 + width * horizon
    if input_ids.shape != (1, rows):
        raise RuntimeError("forest input packing changed row count")

    depths = torch.arange(1, horizon + 1, dtype=torch.long, device=device)
    position_ids = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            depths.repeat(width),
        ],
        dim=0,
    )[None] + prefix_length

    # Static block-diagonal forest mask.  Every row sees the cached prefix.
    # The shared anchor sees itself; path rows see the anchor and only their
    # own chain up through the current input token.
    mask = torch.full(
        (1, 1, rows, prefix_length + rows),
        float("-inf"),
        dtype=mask_dtype,
        device=device,
    )
    mask[:, :, :, :prefix_length] = 0.0
    mask[0, 0, 0, prefix_length] = 0.0
    for path_index in range(width):
        start = 1 + path_index * horizon
        for depth in range(horizon):
            row = start + depth
            mask[0, 0, row, prefix_length] = 0.0
            mask[0, 0, row, prefix_length + start : prefix_length + row + 1] = 0.0
    return input_ids, position_ids, mask


def structural_forest_acceptance(paths: Tensor, target_tokens: Tensor) -> Tensor:
    """Longest target prefix covered by any complete path."""

    if paths.ndim != 2 or target_tokens.shape != (paths.shape[1],):
        raise ValueError("forest paths and target continuation are incompatible")
    lengths = (
        paths.eq(target_tokens[None])
        .to(torch.long)
        .cumprod(dim=-1)
        .sum(dim=-1)
    )
    return lengths.max()


def traverse_padded_forest(paths: Tensor, logits: Tensor) -> ForestTraversalOutput:
    """Independently verify all W paths and select the longest accepted one.

    Duplicate sibling tokens are deliberately not collapsed.  All paths first
    compare token zero with the single anchor posterior, then compare later
    tokens with the preceding row posterior on their own chain.  ``argmax`` on
    the per-path lengths supplies a deterministic lowest-index tie break.
    """

    if paths.ndim != 2 or paths.dtype != torch.long:
        raise ValueError("forest paths must be int64 [width,horizon]")
    width, horizon = paths.shape
    rows = 1 + width * horizon
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != rows:
        raise ValueError("forest logits have incompatible rows")
    posterior = logits.float().argmax(dim=-1)[0]
    anchor_posterior = posterior[0]
    path_posterior = posterior[1:].view(width, horizon)
    first_match = paths[:, :1].eq(anchor_posterior)
    if horizon == 1:
        matches = first_match
    else:
        matches = torch.cat(
            [first_match, paths[:, 1:].eq(path_posterior[:, :-1])], dim=-1
        )
    per_path = matches.to(torch.long).cumprod(dim=-1).sum(dim=-1)
    selected = per_path.argmax(dim=0)
    accepted = per_path.gather(0, selected.view(1))[0]
    selected_tokens = paths.index_select(0, selected.view(1))[0]

    # If no draft token is accepted, the anchor posterior is emitted.  After
    # accepting a>0 drafts, the row holding token a-1 predicts the next token;
    # for a==H this is precisely the full-accept bonus row.
    posterior_index = accepted.sub(1).clamp_min(0).clamp_max(horizon - 1)
    path_next = path_posterior.index_select(0, selected.view(1))[0].gather(
        0, posterior_index.view(1)
    )[0]
    next_token = torch.where(accepted.eq(0), anchor_posterior, path_next)
    return ForestTraversalOutput(
        accepted=accepted.view(1),
        selected_path=selected.view(1),
        next_token=next_token.view(1),
        per_path_accepted=per_path,
        posterior_tokens=posterior,
    )
