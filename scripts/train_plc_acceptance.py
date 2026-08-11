#!/usr/bin/env python3
"""Stage-2 acceptance-frontier training for PLC-Head."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sph.parallel_lattice_correction import ParallelLatticeCorrectionHead
from train_domino_cached_head import load_tensor_from_checkpoint
from train_plc_imitation import (
    TeacherDataset,
    collate,
    evaluate,
    load_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--preserve-weight", type=float, default=0.5)
    parser.add_argument("--code-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--objective", choices=["ce", "margin", "ce_margin"], default="ce"
    )
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--margin-offset", type=float, default=1.0)
    parser.add_argument("--margin-temperature", type=float, default=1.0)
    parser.add_argument("--survival-floor", type=float, default=0.5)
    parser.add_argument("--unreachable-weight", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-blocks", type=int)
    parser.add_argument("--max-eval-blocks", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def survival_continuation_weights(
    gold_probabilities: torch.Tensor,
    prefix_reachable: torch.Tensor,
    *,
    survival_floor: float,
    unreachable_weight: float,
) -> torch.Tensor:
    """Detached soft clean-prefix reach times continuation utility."""

    if not 0.0 <= survival_floor < 1.0:
        raise ValueError("survival_floor must be in [0,1)")
    if not 0.0 <= unreachable_weight <= 1.0:
        raise ValueError("unreachable_weight must be in [0,1]")
    probabilities = gold_probabilities.detach().float()
    softened = survival_floor + (1.0 - survival_floor) * probabilities
    batch, positions = softened.shape
    reach = torch.ones_like(softened)
    if positions > 1:
        reach[:, 1:] = softened[:, :-1].cumprod(dim=-1)
    prefix_scale = torch.where(
        prefix_reachable[:, None],
        torch.ones((batch, 1), device=softened.device),
        torch.full(
            (batch, 1), unreachable_weight, device=softened.device
        ),
    )
    reach = reach * prefix_scale

    continuation = torch.ones_like(softened)
    for position in range(positions - 2, -1, -1):
        continuation[:, position] = 1.0 + (
            softened[:, position + 1] * continuation[:, position + 1]
        )
    weights = reach * continuation
    return weights / weights.mean().clamp_min(1e-6)


def cosine_multiplier(step: int, total: int, warmup: int) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def gold_competitor_margin_loss(
    logits: torch.Tensor,
    gold: torch.Tensor,
    *,
    offset: float,
    temperature: float,
) -> torch.Tensor:
    """Smoothly enforce gold above the best non-gold vocabulary token."""

    if temperature <= 0.0:
        raise ValueError("margin_temperature must be positive")
    top_values, top_ids = logits.topk(2, dim=-1)
    gold_values = logits.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    competitor = torch.where(
        top_ids[..., 0] == gold, top_values[..., 1], top_values[..., 0]
    )
    gap = (competitor.float() - gold_values.float() + offset) / temperature
    return F.softplus(gap) * temperature


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_records = load_records(args.canonical, "train", args.max_train_blocks)
    eval_records = load_records(
        args.canonical, "validation_select", args.max_eval_blocks
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TeacherDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        TeacherDataset(eval_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to("cuda:0", torch.bfloat16)
    first_projection = load_tensor_from_checkpoint(
        args.domino_draft, "embed_proj.0.weight"
    ).to("cuda:0", torch.bfloat16)
    w_out = load_tensor_from_checkpoint(
        args.domino_draft, "embed_proj.2.weight"
    ).to("cuda:0", torch.bfloat16)
    hidden_width = int(train_records[0]["parallel_hidden"].shape[-1])
    w_h = first_projection[:, :hidden_width].contiguous()
    del first_projection
    checkpoint = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    architecture = checkpoint["architecture"]
    head = ParallelLatticeCorrectionHead(
        w_h=w_h,
        w_out=w_out,
        token_embeddings=(
            target_weight
            if bool(architecture.get("use_semantic_embedding", False))
            else None
        ),
        use_full_hidden=bool(architecture.get("use_full_hidden", False)),
        max_positions=int(architecture["positions"]),
        candidates=int(architecture["candidates"]),
        modes=int(architecture["modes"]),
        width=int(architecture["width"]),
        heads=int(architecture["heads"]),
        feed_forward_width=int(architecture["feed_forward_width"]),
        global_layers=int(architecture.get("global_layers", 1)),
    ).to("cuda:0")
    head.load_state_dict(checkpoint["model"], strict=True)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(step, total_steps, warmup_steps),
    )
    initial = evaluate(head=head, target_weight=target_weight, loader=eval_loader)
    print(json.dumps({"initial": initial}, indent=2), flush=True)
    best_state = {
        name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()
    }
    best_epoch = 0
    best_eval = initial
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in train_loader:
            hidden = batch["hidden"].to("cuda:0", non_blocking=True)
            anchors = batch["anchors"].to("cuda:0", non_blocking=True)
            prefix = batch["prefix"].to("cuda:0", non_blocking=True)
            teacher_ids = batch["teacher_ids"].to("cuda:0", non_blocking=True)
            gold_full = batch["gold_full"].to("cuda:0", non_blocking=True)
            gold = gold_full[:, 1:]
            teacher_delta = batch["teacher_delta"].to(
                "cuda:0", non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                base_logits = F.linear(hidden, target_weight)
                output = head(
                    parallel_hiddens=hidden,
                    base_logits=base_logits,
                    anchor_ids=anchors,
                    prefix_ids=prefix,
                    return_logits=True,
                )
                assert output.corrected_logits is not None
                flat_logits = output.corrected_logits.reshape(
                    -1, output.corrected_logits.shape[-1]
                )
                ce_losses = F.cross_entropy(
                    flat_logits, gold.reshape(-1), reduction="none"
                ).view_as(gold)
                margin_losses = gold_competitor_margin_loss(
                    output.corrected_logits,
                    gold,
                    offset=args.margin_offset,
                    temperature=args.margin_temperature,
                )
                if args.objective == "ce":
                    token_losses = ce_losses
                elif args.objective == "margin":
                    token_losses = margin_losses
                else:
                    token_losses = ce_losses + args.margin_weight * margin_losses
                with torch.no_grad():
                    gold_probabilities = torch.exp(-ce_losses.float())
                    weights = survival_continuation_weights(
                        gold_probabilities,
                        prefix == gold_full[:, 0],
                        survival_floor=args.survival_floor,
                        unreachable_weight=args.unreachable_weight,
                    )
                    preserve = (teacher_ids == gold).float()
                target_loss = (token_losses * weights).mean()
                preserve_loss = (
                    token_losses * weights * preserve
                ).sum() / (weights * preserve).sum().clamp_min(1.0)
                scale = teacher_delta.float().square().mean(
                    dim=(-1, -2), keepdim=True
                ).sqrt().clamp_min(1e-3)
                code_loss = F.smooth_l1_loss(
                    output.correction_codes.float() / scale,
                    teacher_delta.float() / scale,
                )
                code_decay = 1.0 - 0.8 * (
                    global_step / max(1, total_steps - 1)
                )
                loss = (
                    target_loss
                    + args.preserve_weight * preserve_loss
                    + args.code_loss_weight * code_decay * code_loss
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            totals["loss"] += float(loss.detach())
            totals["target_loss"] += float(target_loss.detach())
            totals["preserve_loss"] += float(preserve_loss.detach())
            totals["code_loss"] += float(code_loss.detach())
            totals["weight_first"] += float(weights[:, 0].mean())
            totals["weight_last"] += float(weights[:, -1].mean())
            batches += 1

        epoch_eval = evaluate(
            head=head, target_weight=target_weight, loader=eval_loader
        )
        record = {
            "epoch": epoch,
            "train": {key: value / batches for key, value in totals.items()},
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": epoch_eval,
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)
        if epoch_eval["student_eal_prompt_balanced"] > best_eval[
            "student_eal_prompt_balanced"
        ]:
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in head.state_dict().items()
            }
            best_epoch = epoch
            best_eval = epoch_eval

    head.load_state_dict(best_state)
    selected = evaluate(head=head, target_weight=target_weight, loader=eval_loader)
    teacher_eal = selected["teacher_eal_prompt_balanced"]
    acceptance_gate = selected["student_eal_prompt_balanced"] >= 1.15 * teacher_eal
    torch.save(
        {
            "model": best_state,
            "architecture": architecture,
            "best_epoch": best_epoch,
        },
        args.output / "checkpoint.pt",
    )
    report = {
        "status": "completed",
        "stage": "plc_acceptance_frontier",
        "config": vars(args) | {
            "canonical": str(args.canonical),
            "target": str(args.target),
            "domino_draft": str(args.domino_draft),
            "init_checkpoint": str(args.init_checkpoint),
            "output": str(args.output),
        },
        "architecture": architecture,
        "parameters": {
            "trainable": head.trainable_parameter_count,
            "active": head.active_parameter_count,
        },
        "initial": initial,
        "best_epoch": best_epoch,
        "selected_validation": selected,
        "eal_ratio_over_domino": (
            selected["student_eal_prompt_balanced"] / teacher_eal
        ),
        "acceptance_1.15x_gate_pass": acceptance_gate,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
