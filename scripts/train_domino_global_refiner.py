#!/usr/bin/env python3
"""Train a low-rank whole-block causal residual on released Domino logits."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel

from sph.global_direct_selector import (
    GlobalDirectCandidateSelector,
    exact_dpace_position_weights,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument(
        "--additional-train-canonical", nargs="*", type=Path, default=[]
    )
    parser.add_argument(
        "--eval-canonical",
        type=Path,
        help="Optional separate cache carrying validation_select.",
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument(
        "--architecture",
        choices=["causal_refiner", "direct_selector"],
        default="causal_refiner",
        help=(
            "causal_refiner predicts a vocabulary residual from a Jacobi path; "
            "direct_selector explicitly reranks the full Domino Top-K lattice."
        ),
    )
    parser.add_argument("--selector-dim", type=int, default=256)
    parser.add_argument("--selector-heads", type=int, default=8)
    parser.add_argument("--selector-layers", type=int, default=3)
    parser.add_argument(
        "--selector-mixer", choices=["flat", "axial"], default="axial"
    )
    parser.add_argument(
        "--selector-node-encoder",
        choices=["additive", "compatibility"],
        default="compatibility",
    )
    parser.add_argument(
        "--selector-candidates",
        choices=["released_topk", "base_topk_plus_released"],
        default="base_topk_plus_released",
        help=(
            "Candidate source for the direct selector. The default takes the "
            "parallel-base Top-K and replaces its last item with released "
            "Domino's current token only when needed, preserving exact identity."
        ),
    )
    parser.add_argument(
        "--selector-init-checkpoint",
        type=Path,
        help="Optional pretrained GlobalDirectCandidateSelector checkpoint.",
    )
    parser.add_argument(
        "--selector-reset-output",
        action="store_true",
        help="After loading selector features, reset its residual readout to identity.",
    )
    parser.add_argument(
        "--selector-freeze-encoder",
        action="store_true",
        help="Train only the selector residual readout after checkpoint loading.",
    )
    parser.add_argument(
        "--candidate-topk",
        type=int,
        default=16,
        help="Rerank the union of base and released top-k candidates; 0 uses full vocab.",
    )
    parser.add_argument(
        "--objective",
        choices=["decay_ce", "candidate_dpace", "breaker", "breaker_margin"],
        default="decay_ce",
    )
    parser.add_argument("--dpace-alpha", type=float, default=0.5)
    parser.add_argument("--loss-decay-gamma", type=float, default=7.0)
    parser.add_argument("--prefix-weight", type=float, default=0.1)
    parser.add_argument("--consistency-weight", type=float, default=0.3)
    parser.add_argument("--margin-temperature", type=float, default=1.0)
    parser.add_argument("--margin-offset", type=float, default=0.0)
    parser.add_argument("--iterations", nargs="+", type=int, default=[1, 2, 4, 6])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--max-train-blocks", type=int)
    parser.add_argument("--max-eval-blocks", type=int)
    return parser.parse_args()


class GlobalCausalRefiner(nn.Module):
    """xPress-style low-rank fuse/mix/transform with a zero residual start."""

    def __init__(self, *, hidden_size: int, rank: int, positions: int) -> None:
        super().__init__()
        self.positions = positions
        self.rank = rank
        self.hidden_proj = nn.Linear(hidden_size, rank, bias=False)
        self.global_proj = nn.Linear(hidden_size, rank, bias=False)
        self.token_proj = nn.Linear(hidden_size, rank, bias=False)
        self.fuse = nn.Linear(3 * rank, rank, bias=False)
        self.causal_mixer = nn.Parameter(torch.zeros(rank, positions, positions))
        self.mlp_up = nn.Linear(rank, 2 * rank, bias=False)
        self.mlp_down = nn.Linear(2 * rank, rank, bias=False)
        self.residual_out = nn.Linear(rank, rank, bias=False)
        nn.init.zeros_(self.residual_out.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        prefix_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        hidden_float = hidden.float()
        prefix_float = prefix_embeddings.float()
        hidden_float = hidden_float * torch.rsqrt(
            hidden_float.square().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        prefix_float = prefix_float * torch.rsqrt(
            prefix_float.square().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        global_hidden = hidden_float.mean(dim=1, keepdim=True).expand_as(hidden_float)
        fused = self.fuse(
            torch.cat(
                [
                    self.hidden_proj(hidden_float),
                    self.global_proj(global_hidden),
                    self.token_proj(prefix_float),
                ],
                dim=-1,
            )
        )
        lower = torch.tril(self.causal_mixer, diagonal=-1)
        mixed = fused + torch.einsum("dkj,bjd->bkd", lower, fused)
        transformed = mixed + self.mlp_down(F.silu(self.mlp_up(mixed)))
        return self.residual_out(transformed)


@torch.no_grad()
def released_onpolicy_logits(
    *,
    domino: nn.Module,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Released sequential chain and the logits that generated each token."""

    base = F.linear(hidden, target_weight)
    batch, positions = hidden.shape[:2]
    proposals = torch.empty((batch, positions), dtype=torch.long, device=hidden.device)
    logits: list[torch.Tensor] = [base[:, :1]]
    first = base[:, :1].argmax(dim=-1)
    proposals[:, 0] = first[:, 0]
    _, state = domino.prefix_gru(
        F.embedding(torch.cat([anchors[:, None], first], dim=-1), target_weight)
    )
    for position in range(1, positions):
        correction = domino.embed_proj(
            torch.cat(
                [hidden[:, position : position + 1], state.transpose(0, 1)], dim=-1
            )
        )
        current_logits = base[:, position : position + 1] + correction
        logits.append(current_logits)
        token = current_logits.argmax(dim=-1)
        proposals[:, position] = token[:, 0]
        if position + 1 < positions:
            _, state = domino.prefix_gru(F.embedding(token, target_weight), state)
    return torch.cat(logits, dim=1), proposals, base


def apply_candidate_residual(
    *,
    fixed_logits: torch.Tensor,
    base_logits: torch.Tensor,
    delta: torch.Tensor,
    candidate_topk: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Restrict the learned change to deployable base/released candidates."""

    if candidate_topk == 0:
        return fixed_logits + delta, None
    if candidate_topk < 1 or candidate_topk > fixed_logits.shape[-1]:
        raise ValueError("candidate-topk is outside the vocabulary range")
    fixed_ids = fixed_logits.topk(candidate_topk, dim=-1).indices
    base_ids = base_logits.topk(candidate_topk, dim=-1).indices
    candidate_ids = torch.cat([fixed_ids, base_ids], dim=-1)
    candidate_mask = torch.zeros_like(fixed_logits, dtype=torch.bool)
    candidate_mask.scatter_(-1, candidate_ids, True)
    # Keep the candidate logits above a finite outside floor even under an
    # aggressive update.  This makes argmax genuinely candidate-only while
    # avoiding inf*0 in losses for gold tokens outside the deployable union.
    candidate_logits = fixed_logits + delta.clamp(min=-50.0, max=50.0)
    outside_floor = fixed_logits.amin(dim=-1, keepdim=True) - 100.0
    corrected = torch.where(candidate_mask, candidate_logits, outside_floor)
    return corrected, candidate_mask


def residual_logits(
    *,
    refiner: GlobalCausalRefiner,
    readout_weight: torch.Tensor,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    guesses: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    prior_ids = torch.cat([anchors[:, None], guesses[:, :-1]], dim=-1)
    prefix_embeddings = F.embedding(prior_ids, target_weight).detach()
    rank_residual = refiner(hidden, prefix_embeddings)
    return F.linear(rank_residual.to(readout_weight.dtype), readout_weight)


def direct_selector_logits(
    *,
    refiner: GlobalDirectCandidateSelector,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    hidden: torch.Tensor,
    fixed_logits: torch.Tensor,
    base_logits: torch.Tensor,
    candidate_topk: int,
    candidate_source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rerank released Domino's explicit Top-K lattice with a global head."""

    if candidate_topk < 1 or candidate_topk > fixed_logits.shape[-1]:
        raise ValueError("direct_selector requires a valid positive candidate-topk")
    if candidate_source == "released_topk":
        candidate_ids = fixed_logits.topk(candidate_topk, dim=-1).indices
    elif candidate_source == "base_topk_plus_released":
        candidate_ids = base_logits.topk(candidate_topk, dim=-1).indices
        released_top1 = fixed_logits.argmax(dim=-1)
        contains_released = candidate_ids.eq(released_top1.unsqueeze(-1)).any(dim=-1)
        candidate_ids = candidate_ids.clone()
        candidate_ids[..., -1] = torch.where(
            contains_released, candidate_ids[..., -1], released_top1
        )
    else:
        raise ValueError(f"unknown selector candidate source: {candidate_source!r}")
    # Preserve the feature distribution's rank order.  For the default source
    # this is parallel-base rank, which is exactly what the selector's rank
    # embedding was designed and pretrained to encode.  A released top-1 not
    # present in base Top-K occupies the final fallback slot.  The selector's
    # confidence gap uses an order-independent maximum, while deployment base
    # scores below remain released-Domino logits and can peak at any slot.
    candidate_logits = fixed_logits.gather(-1, candidate_ids)
    base_candidate_logits = base_logits.gather(-1, candidate_ids)
    candidate_embeddings = F.embedding(candidate_ids, target_weight).detach()
    anchor_embeddings = F.embedding(anchors, target_weight).detach()
    base_logsumexp = torch.logsumexp(base_logits.float(), dim=-1)
    fixed_logsumexp = torch.logsumexp(fixed_logits.float(), dim=-1)
    output = refiner(
        hidden=hidden,
        candidate_embeddings=candidate_embeddings,
        candidate_logits=base_candidate_logits,
        base_logsumexp=base_logsumexp,
        anchor_embeddings=anchor_embeddings,
        score_candidate_logits=candidate_logits,
        score_logsumexp=fixed_logsumexp,
    )
    outside_floor = fixed_logits.float().amin(dim=-1, keepdim=True) - 100.0
    corrected = outside_floor.expand_as(fixed_logits).clone().scatter(
        -1, candidate_ids, output.scores.to(torch.float32)
    )
    candidate_mask = torch.zeros_like(fixed_logits, dtype=torch.bool)
    candidate_mask.scatter_(-1, candidate_ids, True)
    return corrected, candidate_mask


def all_position_breaker_loss(
    *,
    logits: torch.Tensor,
    gold: torch.Tensor,
    objective: str,
    prefix_weight: float,
    margin_temperature: float,
    margin_offset: float,
    loss_decay_gamma: float = 7.0,
    dpace_alpha: float = 0.5,
    trainable_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not 0.0 <= prefix_weight <= 1.0:
        raise ValueError("prefix weight must be in [0,1]")
    with torch.no_grad():
        predicted = logits.argmax(dim=-1)
        if objective == "candidate_dpace":
            if trainable_mask is None:
                raise ValueError("candidate_dpace requires a candidate coverage mask")
            coverage = trainable_mask.to(torch.int64).cumprod(dim=-1).to(torch.bool)
            gold_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
                -1, gold.unsqueeze(-1)
            ).squeeze(-1)
            weights = exact_dpace_position_weights(
                gold_log_probs.exp(), coverage, alpha=dpace_alpha
            )
        elif objective == "decay_ce":
            weights = torch.exp(
                -torch.arange(
                    gold.shape[1], device=gold.device, dtype=torch.float32
                )
                / loss_decay_gamma
            ).view(1, -1).expand_as(gold)
        else:
            reach = auf_reach_mask(predicted, gold)
            matches = (predicted == gold).float()
            weights = reach * ((1.0 - matches) + prefix_weight * matches)
        if trainable_mask is not None:
            weights = weights * trainable_mask.to(weights.dtype)
    if objective == "breaker_margin":
        losses = best_competitor_margin_loss(
            logits,
            gold,
            temperature=margin_temperature,
            offset=margin_offset,
        )
    elif objective in {"breaker", "decay_ce", "candidate_dpace"}:
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            gold.reshape(-1),
            reduction="none",
        ).reshape_as(gold)
    else:
        raise ValueError(f"unknown objective {objective!r}")
    loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
    accepted = acceptance_lengths(predicted, gold)
    return loss, {
        "weight_sum": float(weights.sum()),
        "teacher_eal": float(accepted.float().mean()),
        "teacher_full_horizon": float((accepted == gold.shape[1]).float().mean()),
    }


def corrected_teacher_logits(
    *,
    domino: nn.Module,
    refiner: nn.Module,
    readout_weight: torch.Tensor,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    gold: torch.Tensor,
    hidden: torch.Tensor,
    consistency_weight: float,
    candidate_topk: int,
    selector_candidates: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    fixed_logits, released_ids, base_logits = released_onpolicy_logits(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
    )
    if isinstance(refiner, GlobalDirectCandidateSelector):
        teacher, candidate_vocab_mask = direct_selector_logits(
            refiner=refiner,
            target_weight=target_weight,
            anchors=anchors,
            hidden=hidden,
            fixed_logits=fixed_logits,
            base_logits=base_logits,
            candidate_topk=candidate_topk,
            candidate_source=selector_candidates,
        )
        return teacher, None, released_ids, candidate_vocab_mask
    teacher_residual = residual_logits(
        refiner=refiner,
        readout_weight=readout_weight,
        target_weight=target_weight,
        anchors=anchors,
        guesses=gold,
        hidden=hidden,
    )
    teacher, candidate_vocab_mask = apply_candidate_residual(
        fixed_logits=fixed_logits,
        base_logits=base_logits,
        delta=teacher_residual,
        candidate_topk=candidate_topk,
    )
    seeded: torch.Tensor | None = None
    if consistency_weight > 0.0:
        seeded_residual = residual_logits(
            refiner=refiner,
            readout_weight=readout_weight,
            target_weight=target_weight,
            anchors=anchors,
            guesses=released_ids,
            hidden=hidden,
        )
        seeded, _ = apply_candidate_residual(
            fixed_logits=fixed_logits,
            base_logits=base_logits,
            delta=seeded_residual,
            candidate_topk=candidate_topk,
        )
    return teacher, seeded, released_ids, candidate_vocab_mask


@torch.inference_mode()
def jacobi_ids(
    *,
    domino: nn.Module,
    refiner: nn.Module,
    readout_weight: torch.Tensor,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    hidden: torch.Tensor,
    iterations: int,
    candidate_topk: int,
    selector_candidates: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    fixed_logits, released_ids, base_logits = released_onpolicy_logits(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
    )
    if isinstance(refiner, GlobalDirectCandidateSelector):
        corrected, _ = direct_selector_logits(
            refiner=refiner,
            target_weight=target_weight,
            anchors=anchors,
            hidden=hidden,
            fixed_logits=fixed_logits,
            base_logits=base_logits,
            candidate_topk=candidate_topk,
            candidate_source=selector_candidates,
        )
        return corrected.argmax(dim=-1), released_ids
    guesses = released_ids
    for _ in range(iterations):
        delta = residual_logits(
            refiner=refiner,
            readout_weight=readout_weight,
            target_weight=target_weight,
            anchors=anchors,
            guesses=guesses,
            hidden=hidden,
        )
        corrected, _ = apply_candidate_residual(
            fixed_logits=fixed_logits,
            base_logits=base_logits,
            delta=delta,
            candidate_topk=candidate_topk,
        )
        updated = corrected.argmax(dim=-1)
        if torch.equal(updated, guesses):
            guesses = updated
            break
        guesses = updated
    return guesses, released_ids


@torch.inference_mode()
def evaluate(
    *,
    domino: nn.Module,
    refiner: nn.Module,
    readout_weight: torch.Tensor,
    target_weight: torch.Tensor,
    loader: DataLoader,
    iterations: int,
    candidate_topk: int,
    selector_candidates: str,
) -> dict[str, Any]:
    refiner.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    lengths: list[int] = []
    released_lengths: list[int] = []
    cached_lengths: list[int] = []
    baseline_token_mismatches = 0
    horizon = 0
    for batch in loader:
        anchors = batch["anchors"].to(target_weight.device, non_blocking=True)
        gold = batch["gold"].to(target_weight.device, non_blocking=True)
        hidden = batch["hidden"].to(target_weight.device, non_blocking=True)
        proposals, released_ids = jacobi_ids(
            domino=domino,
            refiner=refiner,
            readout_weight=readout_weight,
            target_weight=target_weight,
            anchors=anchors,
            hidden=hidden,
            iterations=iterations,
            candidate_topk=candidate_topk,
            selector_candidates=selector_candidates,
        )
        lengths.extend(int(x) for x in acceptance_lengths(proposals, gold).cpu())
        released_lengths.extend(
            int(x) for x in acceptance_lengths(released_ids, gold).cpu()
        )
        cached_ids = batch["cached_released_ids"].to(target_weight.device)
        baseline_token_mismatches += int((released_ids != cached_ids).sum())
        cached_lengths.extend(int(x) for x in batch["cached_released_lengths"])
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        horizon = int(gold.shape[1])
    summary = summarize_lengths(sample_ids, domains, lengths, horizon)
    summary.update(
        {
            "sample_ids": sample_ids,
            "domains": domains,
            "lengths": lengths,
            "released_lengths": released_lengths,
            "baseline_length_mismatches": sum(
                left != right
                for left, right in zip(released_lengths, cached_lengths, strict=True)
            ),
            "baseline_token_mismatches": baseline_token_mismatches,
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("global refiner training requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    if any(value < 1 for value in args.iterations):
        raise ValueError("all Jacobi iteration counts must be positive")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    train_records = load_records(args.canonical, "train", args.max_train_blocks)
    for additional_root in args.additional_train_canonical:
        train_records.extend(load_records(additional_root, "train", None))
    eval_root = args.eval_canonical or args.canonical
    eval_records = load_records(
        eval_root, "validation_select", args.max_eval_blocks
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        CachedDominoDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=True,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        CachedDominoDataset(eval_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate,
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
    readout_weight = domino.embed_proj[2].weight.detach()
    if (
        args.architecture == "causal_refiner"
        and args.rank != int(readout_weight.shape[1])
    ):
        raise ValueError(
            f"rank must match released Domino readout width {readout_weight.shape[1]}"
        )
    horizon = int(train_records[0]["gold_ids"].numel())
    if args.architecture == "causal_refiner":
        refiner: nn.Module = GlobalCausalRefiner(
            hidden_size=int(domino.config.hidden_size),
            rank=args.rank,
            positions=horizon,
        ).to("cuda:0")
    else:
        if args.candidate_topk < 1:
            raise ValueError("direct_selector requires candidate-topk >= 1")
        refiner = GlobalDirectCandidateSelector(
            hidden_size=int(domino.config.hidden_size),
            max_positions=horizon,
            max_candidates=args.candidate_topk,
            model_dim=args.selector_dim,
            num_heads=args.selector_heads,
            num_layers=args.selector_layers,
            scope="global",
            mixer=args.selector_mixer,
            node_encoder=args.selector_node_encoder,
            initialization_seed=args.seed,
        ).to("cuda:0")
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
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
        refiner=refiner,
        readout_weight=readout_weight,
        target_weight=target_weight,
        loader=eval_loader,
        iterations=1,
        candidate_topk=args.candidate_topk,
        selector_candidates=args.selector_candidates,
    )
    if baseline["baseline_length_mismatches"] or baseline["baseline_token_mismatches"]:
        raise RuntimeError(
            "released logits failed cache replay: "
            f"length={baseline['baseline_length_mismatches']}, "
            f"token={baseline['baseline_token_mismatches']}"
        )
    if baseline["lengths"] != baseline["released_lengths"]:
        raise RuntimeError("zero-initialized refiner does not reproduce released Domino")
    baseline_eal = summarize_lengths(
        baseline["sample_ids"], baseline["domains"], baseline["released_lengths"], horizon
    )["overall"]["mean_accepted_draft_tokens_prompt_balanced"]
    print(json.dumps({"baseline_eal": baseline_eal}, indent=2), flush=True)

    best_state = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
    best_epoch = 0
    best_iterations = 1
    best_eval = baseline
    warmstart_eval: dict[str, Any] | None = None
    if args.selector_init_checkpoint is not None:
        if not isinstance(refiner, GlobalDirectCandidateSelector):
            raise ValueError("selector-init-checkpoint requires direct_selector")
        payload = torch.load(
            args.selector_init_checkpoint, map_location="cpu", weights_only=False
        )
        state = payload.get("model", payload.get("refiner_state_dict"))
        if not isinstance(state, dict):
            raise ValueError("selector init checkpoint has no model state dictionary")
        refiner.load_state_dict(state, strict=True)
        if args.selector_reset_output:
            nn.init.zeros_(refiner.residual_projection.weight)
        if args.selector_freeze_encoder:
            for name, parameter in refiner.named_parameters():
                parameter.requires_grad_(name == "residual_projection.weight")
        current = evaluate(
            domino=domino,
            refiner=refiner,
            readout_weight=readout_weight,
            target_weight=target_weight,
            loader=eval_loader,
            iterations=1,
            candidate_topk=args.candidate_topk,
            selector_candidates=args.selector_candidates,
        )
        current_eal = current["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]
        warmstart_eval = {
            "overall": current["overall"],
            "by_domain": current["by_domain"],
            "delta_vs_released": current_eal - baseline_eal,
        }
        print(json.dumps({"warmstart": warmstart_eval}, indent=2), flush=True)
        if current_eal > baseline_eal:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in refiner.state_dict().items()
            }
            best_epoch = -1
            best_eval = current
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        refiner.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in train_loader:
            anchors = batch["anchors"].to(target_weight.device, non_blocking=True)
            gold = batch["gold"].to(target_weight.device, non_blocking=True)
            hidden = batch["hidden"].to(target_weight.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            teacher_logits, seeded_logits, _, candidate_vocab_mask = corrected_teacher_logits(
                domino=domino,
                refiner=refiner,
                readout_weight=readout_weight,
                target_weight=target_weight,
                anchors=anchors,
                gold=gold,
                hidden=hidden,
                consistency_weight=args.consistency_weight,
                candidate_topk=args.candidate_topk,
                selector_candidates=args.selector_candidates,
            )
            candidate_mask = None
            if candidate_vocab_mask is not None:
                candidate_mask = candidate_vocab_mask.gather(
                    -1, gold.unsqueeze(-1)
                ).squeeze(-1)
            teacher_loss, diagnostics = all_position_breaker_loss(
                logits=teacher_logits,
                gold=gold,
                objective=args.objective,
                prefix_weight=args.prefix_weight,
                margin_temperature=args.margin_temperature,
                margin_offset=args.margin_offset,
                loss_decay_gamma=args.loss_decay_gamma,
                dpace_alpha=args.dpace_alpha,
                trainable_mask=candidate_mask,
            )
            consistency_loss = torch.zeros((), device=teacher_loss.device)
            if seeded_logits is not None:
                consistency_loss, _ = all_position_breaker_loss(
                    logits=seeded_logits,
                    gold=gold,
                    objective=args.objective,
                    prefix_weight=args.prefix_weight,
                    margin_temperature=args.margin_temperature,
                    margin_offset=args.margin_offset,
                    loss_decay_gamma=args.loss_decay_gamma,
                    dpace_alpha=args.dpace_alpha,
                    trainable_mask=candidate_mask,
                )
            loss = teacher_loss + args.consistency_weight * consistency_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.max_grad_norm)
            )
            optimizer.step()
            scheduler.step()
            totals["loss"] += float(loss.detach())
            totals["teacher_loss"] += float(teacher_loss.detach())
            totals["consistency_loss"] += float(consistency_loss.detach())
            totals["grad_norm"] += grad_norm
            for key, value in diagnostics.items():
                totals[key] += value
            batches += 1
            global_step += 1

        evaluations: dict[str, Any] = {}
        for iterations in sorted(set(args.iterations)):
            current = evaluate(
                domino=domino,
                refiner=refiner,
                readout_weight=readout_weight,
                target_weight=target_weight,
                loader=eval_loader,
                iterations=iterations,
                candidate_topk=args.candidate_topk,
                selector_candidates=args.selector_candidates,
            )
            eal = current["overall"]["mean_accepted_draft_tokens_prompt_balanced"]
            evaluations[str(iterations)] = {
                "overall": current["overall"],
                "by_domain": current["by_domain"],
                "delta_vs_released": eal - baseline_eal,
            }
            best_eal = best_eval["overall"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ]
            if eal > best_eal:
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in refiner.state_dict().items()
                }
                best_epoch = epoch
                best_iterations = iterations
                best_eval = current
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": {key: value / batches for key, value in totals.items()},
            "validation_select_by_iterations": evaluations,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)

    refiner.load_state_dict(best_state)
    selected = evaluate(
        domino=domino,
        refiner=refiner,
        readout_weight=readout_weight,
        target_weight=target_weight,
        loader=eval_loader,
        iterations=best_iterations,
        candidate_topk=args.candidate_topk,
        selector_candidates=args.selector_candidates,
    )
    selected_eal = selected["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    paired = prompt_bootstrap_difference(
        selected["sample_ids"],
        selected["lengths"],
        selected["released_lengths"],
        args.bootstrap_samples,
        args.seed + 2903,
    )
    torch.save(
        {
            "refiner_state_dict": best_state,
            "architecture": args.architecture,
            "rank": args.rank,
            "positions": horizon,
            "objective": args.objective,
            "best_epoch": best_epoch,
            "best_iterations": best_iterations,
            "candidate_topk": args.candidate_topk,
            "selector_candidates": args.selector_candidates,
            "selector_init_checkpoint": (
                str(args.selector_init_checkpoint.resolve())
                if args.selector_init_checkpoint is not None
                else None
            ),
            "selector_reset_output": args.selector_reset_output,
            "selector_freeze_encoder": args.selector_freeze_encoder,
        },
        args.output / "best_refiner.pt",
    )
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "architecture": args.architecture,
        "objective": args.objective,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "consistency_weight": args.consistency_weight,
        "dpace_alpha": args.dpace_alpha,
        "candidate_topk": args.candidate_topk,
        "selector_candidates": args.selector_candidates,
        "trainable_parameters": sum(
            p.numel() for p in refiner.parameters() if p.requires_grad
        ),
        "train_blocks": len(train_records),
        "validation_blocks": len(eval_records),
        "seconds": time.perf_counter() - started,
        "baseline_eal": baseline_eal,
        "warmstart": warmstart_eval,
        "history": history,
        "best_epoch": best_epoch,
        "best_iterations": best_iterations,
        "selected": {"overall": selected["overall"], "by_domain": selected["by_domain"]},
        "selected_delta_vs_released": selected_eal - baseline_eal,
        "paired_vs_released": paired,
        "checkpoint": str((args.output / "best_refiner.pt").resolve()),
        "inputs": {
            "train_canonical": str(args.canonical.resolve()),
            "additional_train_canonical": [
                str(path.resolve()) for path in args.additional_train_canonical
            ],
            "eval_canonical": str(eval_root.resolve()),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
