#!/usr/bin/env python3
"""Learn a small state/position gate over the released Domino correction."""

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

from train_domino_cached_head import (
    CachedDominoDataset,
    acceptance_lengths,
    collate,
    cosine_schedule,
    load_records,
    load_tensor_from_checkpoint,
    objective_loss,
    prompt_bootstrap_difference,
    summarize_lengths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gate-type", choices=["position", "state_position"], required=True
    )
    parser.add_argument(
        "--objective", choices=["breaker", "breaker_margin"], required=True
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--breaker-prefix-weight", type=float, default=0.1)
    parser.add_argument("--margin-temperature", type=float, default=1.0)
    parser.add_argument("--margin-offset", type=float, default=0.0)
    parser.add_argument("--gate-anchor-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--max-train-blocks", type=int)
    parser.add_argument("--max-eval-blocks", type=int)
    return parser.parse_args()


class AdaptiveCorrectionGate(nn.Module):
    """A scale in (0, 2), initialized to reproduce released Domino exactly."""

    def __init__(
        self,
        *,
        gate_type: str,
        positions: int,
        hidden_size: int,
        state_size: int,
        width: int,
    ) -> None:
        super().__init__()
        self.gate_type = gate_type
        self.positions = positions
        if gate_type == "position":
            self.position_logits = nn.Parameter(torch.zeros(positions))
        elif gate_type == "state_position":
            self.hidden_proj = nn.Linear(hidden_size, width, bias=False)
            self.state_proj = nn.Linear(state_size, width, bias=False)
            self.position_embedding = nn.Embedding(positions, width)
            self.output = nn.Linear(width, 1, bias=True)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            raise ValueError(f"unknown gate type {gate_type!r}")

    def forward(
        self,
        hidden: torch.Tensor,
        state: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.gate_type == "position":
            raw = self.position_logits[position_ids]
            while raw.ndim < hidden.ndim - 1:
                raw = raw.unsqueeze(0)
            raw = raw.expand(hidden.shape[:-1])
        else:
            hidden_float = hidden.float()
            state_float = state.float()
            hidden_float = hidden_float * torch.rsqrt(
                hidden_float.square().mean(dim=-1, keepdim=True).clamp_min(1e-6)
            )
            state_float = state_float * torch.rsqrt(
                state_float.square().mean(dim=-1, keepdim=True).clamp_min(1e-6)
            )
            position = self.position_embedding(position_ids)
            while position.ndim < hidden_float.ndim:
                position = position.unsqueeze(0)
            features = self.hidden_proj(hidden_float) + self.state_proj(state_float)
            raw = self.output(F.silu(features + position)).squeeze(-1)
        return 2.0 * torch.sigmoid(raw)


@torch.no_grad()
def frozen_teacher_components(
    *,
    domino: nn.Module,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    gold: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = F.linear(hidden, target_weight)
    prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
    gru_out, _ = domino.prefix_gru(F.embedding(prefix_ids, target_weight))
    states = gru_out[:, 1:]
    correction = domino.embed_proj(
        torch.cat([hidden[:, 1:], states], dim=-1)
    )
    return base, states, correction


def gated_teacher_logits(
    *,
    domino: nn.Module,
    gate: AdaptiveCorrectionGate,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    gold: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base, states, correction = frozen_teacher_components(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
    )
    position_ids = torch.arange(hidden.shape[1] - 1, device=hidden.device)
    scales = gate(hidden[:, 1:], states, position_ids)
    suffix = (
        base[:, 1:].float() + scales.unsqueeze(-1) * correction.float()
    ).to(base.dtype)
    return torch.cat([base[:, :1], suffix], dim=1), scales


@torch.inference_mode()
def gated_onpolicy_ids(
    *,
    domino: nn.Module,
    gate: AdaptiveCorrectionGate,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    base = F.linear(hidden, target_weight)
    batch, positions = hidden.shape[:2]
    proposals = torch.empty((batch, positions), dtype=torch.long, device=hidden.device)
    first = base[:, :1].argmax(dim=-1)
    proposals[:, 0] = first[:, 0]
    _, state = domino.prefix_gru(
        F.embedding(torch.cat([anchors[:, None], first], dim=-1), target_weight)
    )
    for position in range(1, positions):
        current_state = state.transpose(0, 1)
        correction = domino.embed_proj(
            torch.cat([hidden[:, position : position + 1], current_state], dim=-1)
        )
        gate_position = torch.tensor(position - 1, device=hidden.device)
        scale = gate(
            hidden[:, position : position + 1], current_state, gate_position
        )
        logits = (
            base[:, position : position + 1].float()
            + scale.unsqueeze(-1) * correction.float()
        ).to(base.dtype)
        token = logits.argmax(dim=-1)
        proposals[:, position] = token[:, 0]
        if position + 1 < positions:
            _, state = domino.prefix_gru(F.embedding(token, target_weight), state)
    return proposals


@torch.inference_mode()
def evaluate(
    *,
    domino: nn.Module,
    gate: AdaptiveCorrectionGate,
    target_weight: torch.Tensor,
    loader: DataLoader,
) -> dict[str, Any]:
    gate.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    lengths: list[int] = []
    cached_lengths: list[int] = []
    token_mismatches = 0
    horizon = 0
    for batch in loader:
        anchors = batch["anchors"].to(target_weight.device, non_blocking=True)
        gold = batch["gold"].to(target_weight.device, non_blocking=True)
        hidden = batch["hidden"].to(target_weight.device, non_blocking=True)
        proposals = gated_onpolicy_ids(
            domino=domino,
            gate=gate,
            target_weight=target_weight,
            anchors=anchors,
            hidden=hidden,
        )
        batch_lengths = acceptance_lengths(proposals, gold).cpu().tolist()
        cached_ids = batch["cached_released_ids"].to(target_weight.device)
        token_mismatches += int((proposals != cached_ids).sum())
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        lengths.extend(int(value) for value in batch_lengths)
        cached_lengths.extend(
            int(value) for value in batch["cached_released_lengths"].tolist()
        )
        horizon = int(gold.shape[1])
    summary = summarize_lengths(sample_ids, domains, lengths, horizon)
    summary.update(
        {
            "sample_ids": sample_ids,
            "domains": domains,
            "lengths": lengths,
            "cached_released_lengths": cached_lengths,
            "cached_length_mismatches": sum(
                left != right
                for left, right in zip(lengths, cached_lengths, strict=True)
            ),
            "cached_token_mismatches": token_mismatches,
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("adaptive gate training requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    train_records = load_records(args.canonical, "train", args.max_train_blocks)
    eval_records = load_records(
        args.canonical, "validation_select", args.max_eval_blocks
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

    horizon = int(train_records[0]["gold_ids"].numel())
    gate = AdaptiveCorrectionGate(
        gate_type=args.gate_type,
        positions=horizon - 1,
        hidden_size=int(domino.config.hidden_size),
        state_size=int(domino.gru_hidden_dim),
        width=args.width,
    ).to("cuda:0")
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, total_steps, warmup_steps),
    )

    baseline = evaluate(
        domino=domino, gate=gate, target_weight=target_weight, loader=eval_loader
    )
    if baseline["cached_length_mismatches"] or baseline["cached_token_mismatches"]:
        raise RuntimeError(
            "unit-scale gate did not reproduce released Domino: "
            f"length={baseline['cached_length_mismatches']}, "
            f"token={baseline['cached_token_mismatches']}"
        )
    baseline_eal = baseline["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    print(json.dumps({"baseline_eal": baseline_eal}, indent=2), flush=True)

    best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    best_epoch = 0
    best_eval = baseline
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        gate.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in train_loader:
            anchors = batch["anchors"].to(target_weight.device, non_blocking=True)
            gold = batch["gold"].to(target_weight.device, non_blocking=True)
            hidden = batch["hidden"].to(target_weight.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, scales = gated_teacher_logits(
                domino=domino,
                gate=gate,
                target_weight=target_weight,
                anchors=anchors,
                gold=gold,
                hidden=hidden,
            )
            task_loss, diagnostics = objective_loss(
                all_logits=logits,
                gold=gold,
                objective=args.objective,
                gamma=7.0,
                dpace_smoothing=0.5,
                breaker_prefix_weight=args.breaker_prefix_weight,
                margin_temperature=args.margin_temperature,
                margin_offset=args.margin_offset,
            )
            anchor_penalty = (scales - 1.0).square().mean()
            loss = task_loss + args.gate_anchor_weight * anchor_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(gate.parameters(), args.max_grad_norm)
            )
            optimizer.step()
            scheduler.step()
            totals["loss"] += float(loss.detach())
            totals["task_loss"] += float(task_loss.detach())
            totals["anchor_penalty"] += float(anchor_penalty.detach())
            totals["scale_mean"] += float(scales.detach().mean())
            totals["scale_std"] += float(scales.detach().std())
            totals["grad_norm"] += grad_norm
            for key, value in diagnostics.items():
                totals[key] += value
            batches += 1
            global_step += 1

        epoch_eval = evaluate(
            domino=domino,
            gate=gate,
            target_weight=target_weight,
            loader=eval_loader,
        )
        epoch_eal = epoch_eval["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": {key: value / batches for key, value in totals.items()},
            "validation_select": {
                "overall": epoch_eval["overall"],
                "by_domain": epoch_eval["by_domain"],
                "delta_vs_released": epoch_eal - baseline_eal,
            },
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        if epoch_eal > best_eval["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in gate.state_dict().items()
            }
            best_epoch = epoch
            best_eval = epoch_eval

    gate.load_state_dict(best_state)
    selected = evaluate(
        domino=domino, gate=gate, target_weight=target_weight, loader=eval_loader
    )
    selected_eal = selected["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    paired = prompt_bootstrap_difference(
        selected["sample_ids"],
        selected["lengths"],
        baseline["lengths"],
        args.bootstrap_samples,
        args.seed + 1907,
    )
    torch.save(
        {
            "gate_state_dict": best_state,
            "gate_type": args.gate_type,
            "width": args.width,
            "positions": horizon - 1,
            "objective": args.objective,
            "best_epoch": best_epoch,
        },
        args.output / "best_gate.pt",
    )
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gate_type": args.gate_type,
        "objective": args.objective,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gate_anchor_weight": args.gate_anchor_weight,
        "trainable_parameters": sum(p.numel() for p in gate.parameters()),
        "train_blocks": len(train_records),
        "validation_blocks": len(eval_records),
        "seconds": time.perf_counter() - started,
        "baseline": {"overall": baseline["overall"], "by_domain": baseline["by_domain"]},
        "history": history,
        "best_epoch": best_epoch,
        "selected": {"overall": selected["overall"], "by_domain": selected["by_domain"]},
        "selected_delta_vs_released": selected_eal - baseline_eal,
        "paired_vs_released": paired,
        "checkpoint": str((args.output / "best_gate.pt").resolve()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
