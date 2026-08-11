#!/usr/bin/env python3
"""Train a causal prefix decoder that cross-attends the full DFlash lattice."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel

from sph.global_direct_selector import GlobalDirectBlock
from train_domino_cached_head import (
    CachedDominoDataset,
    acceptance_lengths,
    auf_reach_mask,
    best_competitor_margin_loss,
    collate,
    cosine_schedule,
    load_records,
    load_tensor_from_checkpoint,
    prompt_bootstrap_difference,
    summarize_lengths,
)
from train_domino_global_refiner import released_onpolicy_logits


def causal_collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate(records)
    batch["prompt_balance_weights"] = torch.tensor(
        [float(record.get("_prompt_balance_weight", 1.0)) for record in records],
        dtype=torch.float32,
    )
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument(
        "--additional-train-canonical", nargs="*", type=Path, default=[]
    )
    parser.add_argument("--eval-canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lattice-layers", type=int, default=1)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument(
        "--init-selector-checkpoint",
        type=Path,
        help=(
            "Optional GCLS checkpoint used to initialize compatible hidden/token/"
            "position/rank projections before causal training."
        ),
    )
    parser.add_argument(
        "--init-decoder-checkpoint",
        type=Path,
        help=(
            "Optional full causal-decoder pretraining checkpoint. All feature "
            "layers and the learned candidate direction are restored behind a "
            "zero transfer gate, so the Domino stage starts from exact released "
            "behavior without discarding the pretrained classifier."
        ),
    )
    parser.add_argument(
        "--reset-transfer-projection",
        action="store_true",
        help=(
            "Keep the pretrained causal/lattice representation but reset its "
            "DFlash-specific candidate projection before Domino adaptation."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=["decay_ce", "breaker_margin"],
        default="decay_ce",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-decay-gamma", type=float, default=7.0)
    parser.add_argument("--prefix-weight", type=float, default=0.5)
    parser.add_argument("--margin-temperature", type=float, default=1.0)
    parser.add_argument("--margin-offset", type=float, default=0.0)
    parser.add_argument("--residual-penalty-weight", type=float, default=1e-4)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--max-train-blocks", type=int)
    parser.add_argument("--max-eval-blocks", type=int)
    return parser.parse_args()


class CausalCrossBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(model_dim)
        self.self_attention = nn.MultiheadAttention(
            model_dim, num_heads, bias=False, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(
            model_dim, num_heads, bias=False, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(4 * model_dim, model_dim, bias=False),
        )

    def forward(self, prefix: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        length = prefix.shape[1]
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=prefix.device),
            diagonal=1,
        )
        normalized = self.self_norm(prefix)
        mixed, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        prefix = prefix + mixed
        normalized = self.cross_norm(prefix)
        mixed, _ = self.cross_attention(
            normalized, memory, memory, need_weights=False
        )
        prefix = prefix + mixed
        return prefix + self.feed_forward(self.ff_norm(prefix))


class CausalLatticeDecoder(nn.Module):
    """Autoregressive Top-K reranker with lossless full-lattice memory."""

    def __init__(
        self,
        *,
        hidden_size: int,
        positions: int,
        candidates: int,
        model_dim: int,
        num_heads: int,
        lattice_layers: int,
        decoder_layers: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.positions = positions
        self.candidates = candidates
        self.model_dim = model_dim
        self.hidden_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.token_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.hidden_projection = nn.Linear(hidden_size, model_dim, bias=False)
        self.token_projection = nn.Linear(hidden_size, model_dim, bias=False)
        self.position_embedding = nn.Embedding(positions, model_dim)
        self.prefix_position_embedding = nn.Embedding(positions, model_dim)
        self.rank_embedding = nn.Embedding(candidates, model_dim)
        self.scalar_projection = nn.Sequential(
            nn.Linear(3, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        # The deployed candidate set is mostly the parallel base Top-K, but its
        # final slot can be replaced by Domino's current on-policy top-1.  A
        # slot-only rank embedding aliases those two very different actions.
        # Keep the exact candidate set while exposing whether an action is the
        # released fallback and whether it lies outside the base Top-K.
        self.action_projection = nn.Linear(2, model_dim, bias=False)
        self.lattice_blocks = nn.ModuleList(
            [
                GlobalDirectBlock(
                    model_dim,
                    num_heads,
                    max_positions=positions,
                    dropout=0.0,
                )
                for _ in range(lattice_layers)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [CausalCrossBlock(model_dim, num_heads) for _ in range(decoder_layers)]
        )
        self.compatibility = nn.Sequential(
            nn.Linear(4 * model_dim, 2 * model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * model_dim, model_dim, bias=False),
            nn.SiLU(),
        )
        self.residual_projection = nn.Linear(model_dim, 1, bias=False)
        self.residual_scale = nn.Parameter(torch.ones(()))
        nn.init.zeros_(self.residual_projection.weight)

    @staticmethod
    def scalar_features(
        candidate_logits: torch.Tensor, full_logsumexp: torch.Tensor
    ) -> torch.Tensor:
        logits = candidate_logits.detach().float()
        full_log_probs = logits - full_logsumexp.detach().float()[..., None]
        conditional = torch.log_softmax(logits, dim=-1)
        gap = logits.amax(dim=-1, keepdim=True) - logits
        return torch.stack(
            [
                torch.tanh(full_log_probs / 8.0),
                torch.tanh(conditional / 8.0),
                torch.tanh(gap / 8.0),
            ],
            dim=-1,
        )

    def encode_lattice(
        self,
        *,
        hidden: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_logits: torch.Tensor,
        full_logsumexp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, candidates = candidate_logits.shape
        if (length, candidates) != (self.positions, self.candidates):
            raise ValueError("lattice shape differs from decoder capacity")
        normalized_hidden = self.hidden_norm(hidden.detach().float())
        normalized_candidates = self.token_norm(candidate_embeddings.detach().float())
        local_hidden = self.hidden_projection(normalized_hidden)
        token_states = self.token_projection(normalized_candidates)
        positions = torch.arange(length, device=hidden.device)[None, :, None]
        ranks = torch.arange(candidates, device=hidden.device)[None, None, :]
        states = (
            local_hidden[:, :, None, :]
            + token_states
            + self.position_embedding(positions)
            + self.rank_embedding(ranks)
            + self.scalar_projection(
                self.scalar_features(candidate_logits, full_logsumexp)
            )
        )
        memory = states.reshape(batch, length * candidates, self.model_dim)
        for block in self.lattice_blocks:
            memory = block(
                memory,
                length=length,
                candidates=candidates,
                scope="global",
            )
        return memory, local_hidden

    def encode_prefix(
        self,
        *,
        prefix_embeddings: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        length = prefix_embeddings.shape[1]
        positions = torch.arange(length, device=prefix_embeddings.device)
        states = self.token_projection(
            self.token_norm(prefix_embeddings.detach().float())
        ) + self.prefix_position_embedding(positions)[None]
        for block in self.decoder_blocks:
            states = block(states, memory)
        return states

    def score_candidates(
        self,
        *,
        prefix_states: torch.Tensor,
        local_hidden: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        fixed_candidate_logits: torch.Tensor,
        fixed_logsumexp: torch.Tensor,
        action_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, candidates = fixed_candidate_logits.shape
        if prefix_states.shape[:2] != (batch, length):
            raise ValueError("prefix state length must match scored positions")
        candidate_states = self.token_projection(
            self.token_norm(candidate_embeddings.detach().float())
        )
        ranks = torch.arange(candidates, device=candidate_states.device)[
            None, None, :
        ]
        candidate_states = (
            candidate_states
            + self.rank_embedding(ranks)
            + self.scalar_projection(
                self.scalar_features(fixed_candidate_logits, fixed_logsumexp)
            )
        )
        if action_features is not None:
            if action_features.shape != (*fixed_candidate_logits.shape, 2):
                raise ValueError("candidate action feature shape differs from scores")
            candidate_states = candidate_states + self.action_projection(
                action_features.detach().float()
            )
        query = prefix_states[:, :, None, :].expand_as(candidate_states)
        local = local_hidden[:, :, None, :].expand_as(candidate_states)
        compatibility = self.compatibility(
            torch.cat(
                [query, candidate_states, query * candidate_states, local * candidate_states],
                dim=-1,
            )
        )
        raw_residual = self.residual_projection(compatibility).squeeze(-1)
        raw_residual = raw_residual - raw_residual.mean(dim=-1, keepdim=True)
        # A pretrained feature stack can encounter candidate/logit combinations
        # outside its base-DFlash training distribution.  Sanitise before the
        # transfer gate: mathematically a zero gate must produce an exact zero,
        # whereas IEEE ``0 * NaN`` would otherwise corrupt the released action.
        raw_residual = torch.nan_to_num(
            raw_residual.float(), nan=0.0, posinf=1.0e4, neginf=-1.0e4
        )
        residual = self.residual_scale.float() * raw_residual.float()
        fixed_log_probs = (
            fixed_candidate_logits.detach().float()
            - fixed_logsumexp.detach().float()[..., None]
        )
        if action_features is not None:
            # ``fixed_logits.argmax`` defines the released Domino action over
            # the full vocabulary.  BF16 can leave that action exactly tied
            # with another candidate; candidate-subset argmax would then use
            # subset order and occasionally change accepted length at a zero
            # residual.  This float32-only epsilon resolves exact ties in
            # favour of the already-defined released action and is far below
            # one BF16 logit ULP, so no non-tied decision can change.
            fixed_log_probs = fixed_log_probs + 1.0e-6 * action_features[
                ..., 0
            ].detach().float()
        return fixed_log_probs + residual.float(), residual.float()


def candidate_set(
    *, base_logits: torch.Tensor, fixed_logits: torch.Tensor, topk: int
) -> torch.Tensor:
    ids = base_logits.topk(topk, dim=-1).indices
    released = fixed_logits.argmax(dim=-1)
    contains = ids.eq(released.unsqueeze(-1)).any(dim=-1)
    ids = ids.clone()
    ids[..., -1] = torch.where(contains, ids[..., -1], released)
    return ids


def candidate_action_features(
    *, base_logits: torch.Tensor, fixed_logits: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    """Mark Domino's released action and a released action outside base Top-K."""

    base_ids = base_logits.topk(ids.shape[-1], dim=-1).indices
    released = fixed_logits.argmax(dim=-1, keepdim=True)
    is_released = ids.eq(released)
    in_base = ids.unsqueeze(-1).eq(base_ids.unsqueeze(-2)).any(dim=-1)
    return torch.stack([is_released, ~in_base], dim=-1).float()


@torch.no_grad()
def teacher_domino_logits(
    *,
    domino: nn.Module,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    gold: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = F.linear(hidden, target_weight)
    prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
    gru_out, _ = domino.prefix_gru(F.embedding(prefix_ids, target_weight))
    correction = domino.embed_proj(
        torch.cat([hidden[:, 1:], gru_out[:, 1:]], dim=-1)
    )
    fixed = torch.cat([base[:, :1], base[:, 1:] + correction], dim=1)
    return fixed, base


def training_loss(
    *,
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    gold: torch.Tensor,
    objective: str,
    gamma: float,
    prefix_weight: float,
    margin_temperature: float,
    margin_offset: float,
    block_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    matches = candidate_ids.eq(gold.unsqueeze(-1))
    in_candidates = matches.any(dim=-1)
    gold_indices = matches.to(torch.int64).argmax(dim=-1)
    coverage = in_candidates.to(torch.int64).cumprod(dim=-1).to(torch.bool)
    predicted_indices = scores.detach().argmax(dim=-1)
    predicted_ids = candidate_ids.gather(
        -1, predicted_indices.unsqueeze(-1)
    ).squeeze(-1)
    if objective == "decay_ce":
        decay = torch.exp(
            -torch.arange(gold.shape[1], device=gold.device, dtype=torch.float32)
            / gamma
        )
        weights = coverage.float() * decay[None]
        losses = F.cross_entropy(
            scores.float().reshape(-1, scores.shape[-1]),
            gold_indices.reshape(-1),
            reduction="none",
        ).reshape_as(gold)
    elif objective == "breaker_margin":
        reach = auf_reach_mask(predicted_ids, gold)
        correct = predicted_ids.eq(gold).float()
        weights = coverage.float() * reach * (
            (1.0 - correct) + prefix_weight * correct
        )
        losses = best_competitor_margin_loss(
            scores,
            gold_indices,
            temperature=margin_temperature,
            offset=margin_offset,
        )
    else:
        raise ValueError(f"unknown objective {objective!r}")
    # Selection is prompt-balanced EAL.  First normalize inside each block so
    # long/high-coverage blocks cannot dominate, then inverse-weight blocks by
    # their prompt's block count in the caller.  This makes the training
    # measure match the model-selection measure instead of optimizing token CE.
    block_weight_sum = weights.sum(dim=-1)
    active_blocks = block_weight_sum.gt(0)
    block_losses = (losses * weights).sum(dim=-1) / block_weight_sum.clamp_min(1.0)
    if block_weights is None:
        block_weights = torch.ones_like(block_losses)
    if block_weights.shape != block_losses.shape:
        raise ValueError("prompt-balance weights must have one value per block")
    effective_block_weights = block_weights.float() * active_blocks.float()
    loss = (block_losses * effective_block_weights).sum() / effective_block_weights.sum().clamp_min(1.0)
    lengths = acceptance_lengths(predicted_ids, gold)
    return loss, {
        "weight_sum": float(weights.sum()),
        "teacher_eal": float(lengths.float().mean()),
        "teacher_full_horizon": float((lengths == gold.shape[1]).float().mean()),
        "candidate_coverage_positions": float(coverage.sum()),
        "active_blocks": float(active_blocks.sum()),
        "prompt_weight_sum": float(effective_block_weights.sum()),
    }


def teacher_forward(
    *,
    domino: nn.Module,
    decoder: CausalLatticeDecoder,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    gold: torch.Tensor,
    hidden: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fixed, base = teacher_domino_logits(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
    )
    base_ids = base.topk(topk, dim=-1).indices
    base_candidate_logits = base.gather(-1, base_ids)
    base_embeddings = F.embedding(base_ids, target_weight)
    memory, local_hidden = decoder.encode_lattice(
        hidden=hidden,
        candidate_embeddings=base_embeddings,
        candidate_logits=base_candidate_logits,
        full_logsumexp=torch.logsumexp(base.float(), dim=-1),
    )
    prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
    prefix_states = decoder.encode_prefix(
        prefix_embeddings=F.embedding(prefix_ids, target_weight),
        memory=memory,
    )
    ids = candidate_set(base_logits=base, fixed_logits=fixed, topk=topk)
    candidate_embeddings = F.embedding(ids, target_weight)
    fixed_candidate_logits = fixed.gather(-1, ids)
    scores, residual = decoder.score_candidates(
        prefix_states=prefix_states,
        local_hidden=local_hidden,
        candidate_embeddings=candidate_embeddings,
        fixed_candidate_logits=fixed_candidate_logits,
        fixed_logsumexp=torch.logsumexp(fixed.float(), dim=-1),
        action_features=candidate_action_features(
            base_logits=base, fixed_logits=fixed, ids=ids
        ),
    )
    return scores, residual, ids


@torch.inference_mode()
def decode_ids(
    *,
    domino: nn.Module,
    decoder: CausalLatticeDecoder,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    hidden: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    base = F.linear(hidden, target_weight)
    base_ids = base.topk(topk, dim=-1).indices
    memory, local_hidden = decoder.encode_lattice(
        hidden=hidden,
        candidate_embeddings=F.embedding(base_ids, target_weight),
        candidate_logits=base.gather(-1, base_ids),
        full_logsumexp=torch.logsumexp(base.float(), dim=-1),
    )
    batch, positions = hidden.shape[:2]
    proposals = torch.empty((batch, positions), dtype=torch.long, device=hidden.device)
    prefix_ids = anchors[:, None]
    state: torch.Tensor | None = None
    for position in range(positions):
        if position == 0:
            fixed = base[:, :1]
        else:
            if state is None:
                raise RuntimeError("Domino prefix state was not initialized")
            correction = domino.embed_proj(
                torch.cat(
                    [hidden[:, position : position + 1], state.transpose(0, 1)],
                    dim=-1,
                )
            )
            fixed = base[:, position : position + 1] + correction
        ids = candidate_set(
            base_logits=base[:, position : position + 1],
            fixed_logits=fixed,
            topk=topk,
        )
        prefix_states = decoder.encode_prefix(
            prefix_embeddings=F.embedding(prefix_ids, target_weight),
            memory=memory,
        )
        scores, _ = decoder.score_candidates(
            prefix_states=prefix_states[:, -1:],
            local_hidden=local_hidden[:, position : position + 1],
            candidate_embeddings=F.embedding(ids, target_weight),
            fixed_candidate_logits=fixed.gather(-1, ids),
            fixed_logsumexp=torch.logsumexp(fixed.float(), dim=-1),
            action_features=candidate_action_features(
                base_logits=base[:, position : position + 1],
                fixed_logits=fixed,
                ids=ids,
            ),
        )
        token = ids.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
        proposals[:, position] = token[:, 0]
        prefix_ids = torch.cat([prefix_ids, token], dim=-1)
        if position == 0:
            _, state = domino.prefix_gru(F.embedding(prefix_ids, target_weight))
        elif position + 1 < positions:
            _, state = domino.prefix_gru(F.embedding(token, target_weight), state)
    return proposals


@torch.inference_mode()
def evaluate(
    *,
    domino: nn.Module,
    decoder: CausalLatticeDecoder,
    target_weight: torch.Tensor,
    loader: DataLoader,
    topk: int,
) -> dict[str, Any]:
    decoder.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    lengths: list[int] = []
    released_lengths: list[int] = []
    baseline_token_mismatches = 0
    horizon = 0
    for batch in loader:
        anchors = batch["anchors"].to("cuda:0", non_blocking=True)
        gold = batch["gold"].to("cuda:0", non_blocking=True)
        hidden = batch["hidden"].to("cuda:0", non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            proposals = decode_ids(
                domino=domino,
                decoder=decoder,
                target_weight=target_weight,
                anchors=anchors,
                hidden=hidden,
                topk=topk,
            )
        cached_ids = batch["cached_released_ids"].to("cuda:0")
        baseline_token_mismatches += int((proposals != cached_ids).sum())
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        lengths.extend(int(value) for value in acceptance_lengths(proposals, gold).cpu())
        released_lengths.extend(
            int(value) for value in batch["cached_released_lengths"].tolist()
        )
        horizon = int(gold.shape[1])
    result = summarize_lengths(sample_ids, domains, lengths, horizon)
    result.update(
        {
            "sample_ids": sample_ids,
            "domains": domains,
            "lengths": lengths,
            "released_lengths": released_lengths,
            "baseline_token_mismatches": baseline_token_mismatches,
            "baseline_length_mismatches": sum(
                left != right
                for left, right in zip(lengths, released_lengths, strict=True)
            ),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("causal lattice training requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    train_records = load_records(args.canonical, "train", args.max_train_blocks)
    for root in args.additional_train_canonical:
        train_records.extend(load_records(root, "train", None))
    prompt_block_counts = Counter(str(record["sample_id"]) for record in train_records)
    train_records = [
        {
            **record,
            "_prompt_balance_weight": 1.0
            / prompt_block_counts[str(record["sample_id"])],
        }
        for record in train_records
    ]
    eval_records = load_records(
        args.eval_canonical, "validation_select", args.max_eval_blocks
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        CachedDominoDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=True,
        collate_fn=causal_collate,
    )
    eval_loader = DataLoader(
        CachedDominoDataset(eval_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=causal_collate,
    )
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to(device="cuda:0", dtype=torch.bfloat16)
    target_weight.requires_grad_(False)
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    for parameter in domino.parameters():
        parameter.requires_grad_(False)
    horizon = int(train_records[0]["gold_ids"].numel())
    decoder = CausalLatticeDecoder(
        hidden_size=int(domino.config.hidden_size),
        positions=horizon,
        candidates=args.candidate_topk,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        lattice_layers=args.lattice_layers,
        decoder_layers=args.decoder_layers,
    ).to("cuda:0")
    initialized_from_selector: list[str] = []
    initialized_from_decoder: list[str] = []
    if (
        args.init_selector_checkpoint is not None
        and args.init_decoder_checkpoint is not None
    ):
        raise ValueError("choose at most one decoder initialization source")
    if args.init_decoder_checkpoint is not None:
        payload = torch.load(
            args.init_decoder_checkpoint, map_location="cpu", weights_only=False
        )
        source = payload.get("decoder_state_dict", payload.get("model"))
        if not isinstance(source, dict):
            raise ValueError("decoder checkpoint has no decoder state dictionary")
        target_state = decoder.state_dict()
        for name, target_value in target_state.items():
            if name == "residual_scale":
                continue
            source_value = source.get(name)
            if source_value is None or source_value.shape != target_value.shape:
                raise ValueError(
                    f"decoder initialization is incompatible at {name}: "
                    f"source={getattr(source_value, 'shape', None)}, "
                    f"target={target_value.shape}"
                )
            target_value.copy_(source_value)
            initialized_from_decoder.append(name)
        decoder.load_state_dict(target_state, strict=True)
        if args.reset_transfer_projection:
            # DFlash pretraining can learn useful prefix/lattice features while
            # its final decision direction conflicts with an already-stronger
            # Domino base policy.  Retain the representation and relearn only
            # that final direction from an exact identity start.
            nn.init.zeros_(decoder.residual_projection.weight)
            decoder.residual_scale.data.fill_(1.0)
        else:
            # Preserve the pretrained candidate decision direction behind a
            # zero gate.  Its first gradient decides whether that direction
            # transfers without changing the released starting policy.
            decoder.residual_scale.data.zero_()
        print(
            json.dumps(
                {"initialized_from_decoder": initialized_from_decoder}, indent=2
            ),
            flush=True,
        )
    if args.init_selector_checkpoint is not None:
        payload = torch.load(
            args.init_selector_checkpoint, map_location="cpu", weights_only=False
        )
        source = payload.get("model")
        if not isinstance(source, dict):
            raise ValueError("selector checkpoint has no model state dictionary")
        target_state = decoder.state_dict()
        for name in [
            "hidden_projection.weight",
            "token_projection.weight",
            "position_embedding.weight",
        ]:
            if name not in source or source[name].shape != target_state[name].shape:
                raise ValueError(
                    f"selector initialization is incompatible at {name}: "
                    f"source={getattr(source.get(name), 'shape', None)}, "
                    f"target={target_state[name].shape}"
                )
            target_state[name].copy_(source[name])
            initialized_from_selector.append(name)
        target_state["prefix_position_embedding.weight"].copy_(
            source["position_embedding.weight"]
        )
        initialized_from_selector.append("prefix_position_embedding.weight")
        source_rank = source.get("rank_embedding.weight")
        if (
            source_rank is None
            or source_rank.shape[1] != target_state["rank_embedding.weight"].shape[1]
            or source_rank.shape[0] < args.candidate_topk
        ):
            raise ValueError("selector rank embedding is incompatible")
        target_state["rank_embedding.weight"].copy_(
            source_rank[: args.candidate_topk]
        )
        initialized_from_selector.append("rank_embedding.weight")
        decoder.load_state_dict(target_state, strict=True)
        nn.init.zeros_(decoder.residual_projection.weight)
        decoder.residual_scale.data.fill_(1.0)
        print(
            json.dumps(
                {"initialized_from_selector": initialized_from_selector}, indent=2
            ),
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_schedule(step, total_steps, warmup_steps)
    )

    baseline = evaluate(
        domino=domino,
        decoder=decoder,
        target_weight=target_weight,
        loader=eval_loader,
        topk=args.candidate_topk,
    )
    if baseline["baseline_length_mismatches"]:
        raise RuntimeError(
            "zero decoder changed released-Domino accepted length: "
            f"length={baseline['baseline_length_mismatches']}"
        )
    if baseline["baseline_token_mismatches"]:
        print(
            json.dumps(
                {
                    "non_acceptance_relevant_suffix_token_mismatches": baseline[
                        "baseline_token_mismatches"
                    ],
                    "accepted_length_mismatches": 0,
                },
                indent=2,
            ),
            flush=True,
        )
    baseline_eal = baseline["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    print(json.dumps({"baseline_eal": baseline_eal}, indent=2), flush=True)
    raw_transfer: dict[str, Any] | None = None
    if (
        args.init_decoder_checkpoint is not None
        and not args.reset_transfer_projection
    ):
        decoder.residual_scale.data.fill_(1.0)
        raw_transfer_eval = evaluate(
            domino=domino,
            decoder=decoder,
            target_weight=target_weight,
            loader=eval_loader,
            topk=args.candidate_topk,
        )
        raw_transfer_eal = raw_transfer_eval["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]
        raw_transfer = {
            "overall": raw_transfer_eval["overall"],
            "by_domain": raw_transfer_eval["by_domain"],
            "delta_vs_released": raw_transfer_eal - baseline_eal,
        }
        print(json.dumps({"raw_pretrained_transfer": raw_transfer}, indent=2), flush=True)
        decoder.residual_scale.data.zero_()
    best_state = {
        name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()
    }
    best_step = 0
    best_eval = baseline
    step_history: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in train_loader:
            anchors = batch["anchors"].to("cuda:0", non_blocking=True)
            gold = batch["gold"].to("cuda:0", non_blocking=True)
            hidden = batch["hidden"].to("cuda:0", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores, residual, ids = teacher_forward(
                    domino=domino,
                    decoder=decoder,
                    target_weight=target_weight,
                    anchors=anchors,
                    gold=gold,
                    hidden=hidden,
                    topk=args.candidate_topk,
                )
            task_loss, diagnostics = training_loss(
                scores=scores,
                candidate_ids=ids,
                gold=gold,
                objective=args.objective,
                gamma=args.loss_decay_gamma,
                prefix_weight=args.prefix_weight,
                margin_temperature=args.margin_temperature,
                margin_offset=args.margin_offset,
                block_weights=batch["prompt_balance_weights"].to(
                    "cuda:0", non_blocking=True
                ),
            )
            residual_penalty = residual.square().mean()
            loss = task_loss + args.residual_penalty_weight * residual_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
            )
            optimizer.step()
            scheduler.step()
            totals["loss"] += float(loss.detach())
            totals["task_loss"] += float(task_loss.detach())
            totals["residual_penalty"] += float(residual_penalty.detach())
            totals["residual_rms"] += float(residual.detach().square().mean().sqrt())
            totals["grad_norm"] += grad_norm
            for key, value in diagnostics.items():
                totals[key] += value
            batches += 1
            global_step += 1
            if global_step % 100 == 0:
                print(
                    f"step={global_step}/{total_steps} epoch={epoch} "
                    f"loss={float(loss.detach()):.6f} "
                    f"residual_rms={float(residual.detach().square().mean().sqrt()):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )
            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                decoder.eval()
                current = evaluate(
                    domino=domino,
                    decoder=decoder,
                    target_weight=target_weight,
                    loader=eval_loader,
                    topk=args.candidate_topk,
                )
                eal = current["overall"][
                    "mean_accepted_draft_tokens_prompt_balanced"
                ]
                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "overall": current["overall"],
                    "by_domain": current["by_domain"],
                    "delta_vs_released": eal - baseline_eal,
                }
                step_history.append(record)
                print(json.dumps({"step_validation": record}, indent=2), flush=True)
                best_eal = best_eval["overall"][
                    "mean_accepted_draft_tokens_prompt_balanced"
                ]
                if eal > best_eal:
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in decoder.state_dict().items()
                    }
                    best_step = global_step
                    best_eval = current
                decoder.train()

        current = evaluate(
            domino=domino,
            decoder=decoder,
            target_weight=target_weight,
            loader=eval_loader,
            topk=args.candidate_topk,
        )
        eal = current["overall"]["mean_accepted_draft_tokens_prompt_balanced"]
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": {key: value / batches for key, value in totals.items()},
            "validation_select": {
                "overall": current["overall"],
                "by_domain": current["by_domain"],
                "delta_vs_released": eal - baseline_eal,
            },
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)
        best_eal = best_eval["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]
        if eal > best_eal:
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in decoder.state_dict().items()
            }
            best_step = global_step
            best_eval = current

    decoder.load_state_dict(best_state)
    selected = evaluate(
        domino=domino,
        decoder=decoder,
        target_weight=target_weight,
        loader=eval_loader,
        topk=args.candidate_topk,
    )
    selected_eal = selected["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    paired = prompt_bootstrap_difference(
        selected["sample_ids"],
        selected["lengths"],
        selected["released_lengths"],
        args.bootstrap_samples,
        args.seed + 6337,
    )
    checkpoint = args.output / "best_decoder.pt"
    torch.save(
        {
            "decoder_state_dict": best_state,
            "best_step": best_step,
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "lattice_layers": args.lattice_layers,
            "decoder_layers": args.decoder_layers,
            "candidate_topk": args.candidate_topk,
        },
        checkpoint,
    )
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "objective": args.objective,
        "train_blocks": len(train_records),
        "validation_blocks": len(eval_records),
        "trainable_parameters": sum(p.numel() for p in decoder.parameters()),
        "init_selector_checkpoint": (
            str(args.init_selector_checkpoint.resolve())
            if args.init_selector_checkpoint is not None
            else None
        ),
        "initialized_from_selector": initialized_from_selector,
        "init_decoder_checkpoint": (
            str(args.init_decoder_checkpoint.resolve())
            if args.init_decoder_checkpoint is not None
            else None
        ),
        "initialized_from_decoder": initialized_from_decoder,
        "raw_pretrained_transfer": raw_transfer,
        "reset_transfer_projection": args.reset_transfer_projection,
        "baseline_eal": baseline_eal,
        "best_step": best_step,
        "step_history": step_history,
        "history": history,
        "selected": {"overall": selected["overall"], "by_domain": selected["by_domain"]},
        "selected_delta_vs_released": selected_eal - baseline_eal,
        "paired_vs_released": paired,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
