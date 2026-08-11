"""Real layer-split target verification path used by Fast-R048.

The first four verifier layers run over the frozen proposal before the repair
decision.  Their KV and outputs are reused up to the changed token; only the
invalid suffix is recomputed.  Remaining verifier layers run exactly once on
the final 17-token verification block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class LayerSplitVerifierOutput:
    logits: Tensor
    decision_states: Tensor
    final_early_hidden: Tensor


def clone_dynamic_cache(cache: DynamicCache, *, config: Any) -> DynamicCache:
    """Clone a populated prefix cache without replaying the prompt."""

    data = []
    for layer in cache.layers:
        if not bool(getattr(layer, "is_initialized", False)):
            raise ValueError("cannot clone an uninitialized prefix-cache layer")
        data.append((layer.keys.clone(), layer.values.clone()))
    return DynamicCache(ddp_cache_data=data, config=config)


def causal_additive_mask(
    *,
    batch_size: int,
    query_length: int,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Build a lower-right causal mask for a cached multi-token query."""

    if batch_size < 1 or query_length < 1 or past_length < 0:
        raise ValueError("invalid causal-mask dimensions")
    query_positions = past_length + torch.arange(query_length, device=device)
    key_positions = torch.arange(past_length + query_length, device=device)
    allowed = key_positions.view(1, -1) <= query_positions.view(-1, 1)
    mask = torch.zeros((query_length, past_length + query_length), dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.view(1, 1, query_length, -1).expand(batch_size, 1, -1, -1)


def _run_layers(
    *,
    model: Any,
    layers: Sequence[Any],
    hidden: Tensor,
    cache: DynamicCache,
    past_length: int,
) -> Tensor:
    query_length = int(hidden.shape[1])
    positions = torch.arange(
        past_length,
        past_length + query_length,
        dtype=torch.long,
        device=hidden.device,
    )
    position_ids = positions.unsqueeze(0).expand(hidden.shape[0], -1)
    position_embeddings = model.rotary_emb(hidden, position_ids)
    attention_mask = causal_additive_mask(
        batch_size=int(hidden.shape[0]),
        query_length=query_length,
        past_length=past_length,
        dtype=hidden.dtype,
        device=hidden.device,
    )
    for layer in layers:
        hidden = layer(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=positions,
            position_embeddings=position_embeddings,
        )
    return hidden


@torch.no_grad()
def early_decision_prepass(
    *,
    target: Any,
    cache: DynamicCache,
    input_ids: Tensor,
    prefix_length: int,
    early_layers: int = 4,
) -> Tensor:
    """Run only the disposable early-layer proposal decision prepass."""

    layers = list(target.model.layers)
    if not 1 <= early_layers < len(layers):
        raise ValueError("early layer split lies outside target depth")
    if input_ids.ndim != 2 or input_ids.shape[1] != 16:
        raise ValueError("decision prepass requires anchor + proposal[:15]")
    if any(cache.get_seq_length(index) != prefix_length for index in range(len(layers))):
        raise ValueError("decision cache is not a clean full-model prefix fork")
    return _run_layers(
        model=target.model,
        layers=layers[:early_layers],
        hidden=target.model.embed_tokens(input_ids),
        cache=cache,
        past_length=prefix_length,
    )


@torch.no_grad()
def layer_split_verifier_forward(
    *,
    target: Any,
    cache: DynamicCache,
    original_input_ids: Tensor,
    final_input_ids: Tensor,
    prefix_length: int,
    early_layers: int = 4,
    correction_position: int | None,
) -> LayerSplitVerifierOutput:
    """Run the deployment-shaped R048 verifier with prefix-KV reuse.

    Both inputs contain ``anchor + 16 proposal tokens``.  The early decision
    prepass consumes only ``anchor + proposal[:15]``.  A correction at draft
    position ``j`` changes input row ``j+1``; prepass rows through ``j`` and
    their early-layer KV remain valid.
    """

    if original_input_ids.shape != final_input_ids.shape:
        raise ValueError("original/final verifier inputs differ in shape")
    if original_input_ids.ndim != 2 or original_input_ids.shape[1] != 17:
        raise ValueError("R048 verifier input must have shape [batch, 17]")
    layers = list(target.model.layers)
    if not 1 <= early_layers < len(layers):
        raise ValueError("early layer split lies outside target depth")
    if correction_position is not None and not 0 <= correction_position < 16:
        raise ValueError("correction position lies outside the B16 proposal")
    if any(cache.get_seq_length(index) != prefix_length for index in range(len(layers))):
        raise ValueError("target cache is not reset to one shared prefix length")

    early = layers[:early_layers]
    late = layers[early_layers:]
    prepass_ids = original_input_ids[:, :16]
    prepass_hidden = early_decision_prepass(
        target=target,
        cache=cache,
        input_ids=prepass_ids,
        prefix_length=prefix_length,
        early_layers=early_layers,
    )
    decision_states = prepass_hidden

    if correction_position is None:
        tail_start = 16
        valid_prepass_rows = 16
    else:
        tail_start = correction_position + 1
        valid_prepass_rows = correction_position + 1
        # Later layers still have only prefix_length rows and are unchanged by
        # this crop.  Early layers discard the original changed token onward.
        cache.crop(prefix_length + valid_prepass_rows)

    tail_ids = final_input_ids[:, tail_start:]
    tail_hidden = _run_layers(
        model=target.model,
        layers=early,
        hidden=target.model.embed_tokens(tail_ids),
        cache=cache,
        past_length=prefix_length + valid_prepass_rows,
    )
    final_early_hidden = torch.cat(
        [prepass_hidden[:, :valid_prepass_rows], tail_hidden], dim=1
    )
    if final_early_hidden.shape[1] != 17:
        raise RuntimeError("early layer split did not reconstruct 17 verifier rows")

    final_hidden = _run_layers(
        model=target.model,
        layers=late,
        hidden=final_early_hidden,
        cache=cache,
        past_length=prefix_length,
    )
    final_hidden = target.model.norm(final_hidden)
    logits = target.lm_head(final_hidden)
    return LayerSplitVerifierOutput(
        logits=logits,
        decision_states=decision_states,
        final_early_hidden=final_early_hidden,
    )


__all__ = [
    "LayerSplitVerifierOutput",
    "causal_additive_mask",
    "clone_dynamic_cache",
    "early_decision_prepass",
    "layer_split_verifier_forward",
]
