#!/usr/bin/env python3
"""Train a global final-hidden adapter jointly with frozen Domino correction."""

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

from sph.global_direct_selector import GlobalDirectBlock
from train_domino_cached_head import (
    CachedDominoDataset,
    acceptance_lengths,
    collate,
    cosine_schedule,
    load_records,
    load_tensor_from_checkpoint,
    prompt_bootstrap_difference,
    summarize_lengths,
)
from train_domino_global_refiner import all_position_breaker_loss


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
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--adapter-heads", type=int, default=8)
    parser.add_argument("--adapter-layers", type=int, default=2)
    parser.add_argument(
        "--adapter-application",
        choices=["base_only", "joint"],
        default="base_only",
        help=(
            "base_only changes the target-tied parallel logits while keeping "
            "Domino's released correction features fixed; joint also feeds the "
            "adapted hidden state into the correction MLP."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=["decay_ce", "breaker", "breaker_margin"],
        default="decay_ce",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-decay-gamma", type=float, default=7.0)
    parser.add_argument("--prefix-weight", type=float, default=0.25)
    parser.add_argument("--margin-temperature", type=float, default=1.0)
    parser.add_argument("--margin-offset", type=float, default=0.0)
    parser.add_argument("--adapter-output-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help="Run exact on-policy validation inside long epochs; 0 disables it.",
    )
    parser.add_argument("--max-train-blocks", type=int)
    parser.add_argument("--max-eval-blocks", type=int)
    return parser.parse_args()


class GlobalFinalHiddenAdapter(nn.Module):
    """A globally mixed residual on the final parallel-draft representation."""

    def __init__(
        self,
        *,
        hidden_size: int,
        positions: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.positions = positions
        self.input_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.local_projection = nn.Linear(hidden_size, model_dim, bias=False)
        self.global_projection = nn.Linear(hidden_size, model_dim, bias=False)
        self.position_embedding = nn.Embedding(positions, model_dim)
        self.blocks = nn.ModuleList(
            [
                GlobalDirectBlock(
                    model_dim,
                    num_heads,
                    max_positions=positions,
                    dropout=0.0,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, hidden_size, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden must have shape [B, L, hidden_size]")
        if hidden.shape[1] != self.positions:
            raise ValueError("hidden block length differs from adapter capacity")
        normalized = self.input_norm(hidden.detach().float())
        global_summary = normalized.mean(dim=1, keepdim=True)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        states = (
            self.local_projection(normalized)
            + self.global_projection(global_summary)
            + self.position_embedding(positions)[None]
        )
        for block in self.blocks:
            states = block(
                states,
                length=hidden.shape[1],
                candidates=1,
                scope="global",
            )
        delta = self.output_projection(self.output_norm(states))
        adapted = hidden.detach().float() + delta
        return adapted, delta


def teacher_logits(
    *,
    domino: nn.Module,
    adapter: GlobalFinalHiddenAdapter,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    gold: torch.Tensor,
    hidden: torch.Tensor,
    application: str = "base_only",
) -> tuple[torch.Tensor, torch.Tensor]:
    adapted_float, delta = adapter(hidden)
    adapted = adapted_float.to(target_weight.dtype)
    base = F.linear(adapted, target_weight)
    prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
    gru_out, _ = domino.prefix_gru(F.embedding(prefix_ids, target_weight))
    if application == "base_only":
        correction_hidden = hidden.detach().to(target_weight.dtype)
    elif application == "joint":
        correction_hidden = adapted
    else:
        raise ValueError(f"unknown adapter application {application!r}")
    correction = domino.embed_proj(
        torch.cat([correction_hidden[:, 1:], gru_out[:, 1:]], dim=-1)
    )
    # Preserve released Domino's BF16 addition at the identity point.  The
    # subsequent float conversion is only for stable loss arithmetic; moving
    # the addition itself to FP32 can change suffix argmax ties.
    logits = torch.cat([base[:, :1], base[:, 1:] + correction], dim=1)
    return logits.float(), delta


@torch.inference_mode()
def onpolicy_ids(
    *,
    domino: nn.Module,
    adapter: GlobalFinalHiddenAdapter,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    hidden: torch.Tensor,
    application: str = "base_only",
) -> torch.Tensor:
    adapted_float, _ = adapter(hidden)
    adapted = adapted_float.to(target_weight.dtype)
    base = F.linear(adapted, target_weight)
    if application == "base_only":
        correction_hidden = hidden.detach().to(target_weight.dtype)
    elif application == "joint":
        correction_hidden = adapted
    else:
        raise ValueError(f"unknown adapter application {application!r}")
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
            torch.cat(
                [correction_hidden[:, position : position + 1], current_state], dim=-1
            )
        )
        logits = base[:, position : position + 1] + correction
        token = logits.argmax(dim=-1)
        proposals[:, position] = token[:, 0]
        if position + 1 < positions:
            _, state = domino.prefix_gru(F.embedding(token, target_weight), state)
    return proposals


@torch.inference_mode()
def evaluate(
    *,
    domino: nn.Module,
    adapter: GlobalFinalHiddenAdapter,
    target_weight: torch.Tensor,
    loader: DataLoader,
    application: str = "base_only",
) -> dict[str, Any]:
    adapter.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    lengths: list[int] = []
    released_lengths: list[int] = []
    token_mismatches = 0
    horizon = 0
    for batch in loader:
        anchors = batch["anchors"].to(target_weight.device, non_blocking=True)
        gold = batch["gold"].to(target_weight.device, non_blocking=True)
        hidden = batch["hidden"].to(target_weight.device, non_blocking=True)
        proposals = onpolicy_ids(
            domino=domino,
            adapter=adapter,
            target_weight=target_weight,
            anchors=anchors,
            hidden=hidden,
            application=application,
        )
        cached_ids = batch["cached_released_ids"].to(target_weight.device)
        token_mismatches += int((proposals != cached_ids).sum())
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        lengths.extend(int(x) for x in acceptance_lengths(proposals, gold).cpu())
        released_lengths.extend(
            int(x) for x in batch["cached_released_lengths"].tolist()
        )
        horizon = int(gold.shape[1])
    result = summarize_lengths(sample_ids, domains, lengths, horizon)
    result.update(
        {
            "sample_ids": sample_ids,
            "domains": domains,
            "lengths": lengths,
            "released_lengths": released_lengths,
            "baseline_length_mismatches": sum(
                left != right
                for left, right in zip(lengths, released_lengths, strict=True)
            ),
            "baseline_token_mismatches": token_mismatches,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("global hidden-adapter training requires CUDA")
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
    adapter = GlobalFinalHiddenAdapter(
        hidden_size=int(domino.config.hidden_size),
        positions=horizon,
        model_dim=args.adapter_dim,
        num_heads=args.adapter_heads,
        num_layers=args.adapter_layers,
    ).to("cuda:0")
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
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
        adapter=adapter,
        target_weight=target_weight,
        loader=eval_loader,
        application=args.adapter_application,
    )
    if baseline["baseline_length_mismatches"] or baseline["baseline_token_mismatches"]:
        raise RuntimeError(
            "zero adapter failed released-Domino replay: "
            f"length={baseline['baseline_length_mismatches']}, "
            f"tokens={baseline['baseline_token_mismatches']}"
        )
    baseline_eal = baseline["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    print(json.dumps({"baseline_eal": baseline_eal}, indent=2), flush=True)
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
    }
    best_epoch = 0
    best_step = 0
    best_eval = baseline
    history: list[dict[str, Any]] = []
    step_history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in train_loader:
            anchors = batch["anchors"].to(target_weight.device, non_blocking=True)
            gold = batch["gold"].to(target_weight.device, non_blocking=True)
            hidden = batch["hidden"].to(target_weight.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, delta = teacher_logits(
                domino=domino,
                adapter=adapter,
                target_weight=target_weight,
                anchors=anchors,
                gold=gold,
                hidden=hidden,
                application=args.adapter_application,
            )
            task_loss, diagnostics = all_position_breaker_loss(
                logits=logits,
                gold=gold,
                objective=args.objective,
                prefix_weight=args.prefix_weight,
                margin_temperature=args.margin_temperature,
                margin_offset=args.margin_offset,
                loss_decay_gamma=args.loss_decay_gamma,
            )
            output_penalty = delta.float().square().mean()
            loss = task_loss + args.adapter_output_weight * output_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
            )
            optimizer.step()
            scheduler.step()
            totals["loss"] += float(loss.detach())
            totals["task_loss"] += float(task_loss.detach())
            totals["output_penalty"] += float(output_penalty.detach())
            totals["delta_rms"] += float(delta.detach().float().square().mean().sqrt())
            totals["grad_norm"] += grad_norm
            for key, value in diagnostics.items():
                totals[key] += value
            batches += 1
            global_step += 1
            if global_step % 200 == 0:
                print(
                    f"step={global_step}/{total_steps} epoch={epoch} "
                    f"loss={float(loss.detach()):.6f} "
                    f"delta_rms={float(delta.detach().float().square().mean().sqrt()):.5f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )
            if (
                args.eval_every_steps > 0
                and global_step % args.eval_every_steps == 0
            ):
                adapter.eval()
                step_eval = evaluate(
                    domino=domino,
                    adapter=adapter,
                    target_weight=target_weight,
                    loader=eval_loader,
                    application=args.adapter_application,
                )
                step_eal = step_eval["overall"][
                    "mean_accepted_draft_tokens_prompt_balanced"
                ]
                step_record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "overall": step_eval["overall"],
                    "by_domain": step_eval["by_domain"],
                    "delta_vs_released": step_eal - baseline_eal,
                }
                step_history.append(step_record)
                print(
                    json.dumps(
                        {"step_validation": step_record},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    flush=True,
                )
                best_eal = best_eval["overall"][
                    "mean_accepted_draft_tokens_prompt_balanced"
                ]
                if step_eal > best_eal:
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in adapter.state_dict().items()
                    }
                    best_epoch = epoch
                    best_step = global_step
                    best_eval = step_eval
                adapter.train()

        current = evaluate(
            domino=domino,
            adapter=adapter,
            target_weight=target_weight,
            loader=eval_loader,
            application=args.adapter_application,
        )
        current_eal = current["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": {key: value / batches for key, value in totals.items()},
            "validation_select": {
                "overall": current["overall"],
                "by_domain": current["by_domain"],
                "delta_vs_released": current_eal - baseline_eal,
            },
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        best_eal = best_eval["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]
        if current_eal > best_eal:
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in adapter.state_dict().items()
            }
            best_epoch = epoch
            best_step = global_step
            best_eval = current

    adapter.load_state_dict(best_state)
    selected = evaluate(
        domino=domino,
        adapter=adapter,
        target_weight=target_weight,
        loader=eval_loader,
        application=args.adapter_application,
    )
    selected_eal = selected["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    paired = prompt_bootstrap_difference(
        selected["sample_ids"],
        selected["lengths"],
        selected["released_lengths"],
        args.bootstrap_samples,
        args.seed + 4253,
    )
    checkpoint = args.output / "best_adapter.pt"
    torch.save(
        {
            "adapter_state_dict": best_state,
            "adapter_dim": args.adapter_dim,
            "adapter_heads": args.adapter_heads,
            "adapter_layers": args.adapter_layers,
            "adapter_application": args.adapter_application,
            "positions": horizon,
            "best_epoch": best_epoch,
            "best_step": best_step,
        },
        checkpoint,
    )
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "objective": args.objective,
        "adapter_dim": args.adapter_dim,
        "adapter_heads": args.adapter_heads,
        "adapter_layers": args.adapter_layers,
        "adapter_application": args.adapter_application,
        "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
        "train_blocks": len(train_records),
        "validation_blocks": len(eval_records),
        "baseline_eal": baseline_eal,
        "history": history,
        "step_history": step_history,
        "eval_every_steps": args.eval_every_steps,
        "best_epoch": best_epoch,
        "best_step": best_step,
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
