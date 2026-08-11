#!/usr/bin/env python3
"""Stage-1 on-policy Domino imitation for PLC-Head v1."""

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
from torch.utils.data import DataLoader, Dataset

from sph.parallel_lattice_correction import ParallelLatticeCorrectionHead
from train_domino_cached_head import load_tensor_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--code-loss-weight", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-blocks", type=int, default=1024)
    parser.add_argument("--max-eval-blocks", type=int)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feed-forward-width", type=int, default=256)
    parser.add_argument("--global-layers", type=int, default=1)
    parser.add_argument("--use-semantic-embedding", action="store_true")
    parser.add_argument("--use-full-hidden", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_records(
    root: Path, split: str, maximum: int | None
) -> list[dict[str, Any]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("collection_complete", False):
        raise RuntimeError(f"incomplete PLC teacher cache: {root}")
    if metadata.get("format") != "plc_runtime_teacher_v1":
        raise ValueError(f"unexpected PLC cache format: {metadata.get('format')}")
    records: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        records.extend(
            record
            for record in torch.load(shard, map_location="cpu", weights_only=False)
            if str(record["split"]) == split
        )
        if maximum is not None and len(records) >= maximum:
            break
    if maximum is not None:
        records = records[:maximum]
    if not records:
        raise ValueError(f"no records for split={split}")
    return records


class TeacherDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_ids": [str(record["sample_id"]) for record in records],
        "domains": [str(record["domain"]) for record in records],
        "anchors": torch.tensor(
            [int(record["anchor_token_id"]) for record in records],
            dtype=torch.long,
        ),
        "prefix": torch.tensor(
            [int(record["base_prefix_token_id"]) for record in records],
            dtype=torch.long,
        ),
        "hidden": torch.stack(
            [record["parallel_hidden"].to(torch.bfloat16) for record in records]
        ),
        "teacher_delta": torch.stack(
            [record["teacher_delta"].to(torch.bfloat16) for record in records]
        ),
        "teacher_ids": torch.stack(
            [record["teacher_ids"].long() for record in records]
        ),
        "teacher_full": torch.stack(
            [record["teacher_full_ids"].long() for record in records]
        ),
        "gold_full": torch.stack(
            [record["gold_full_ids"].long() for record in records]
        ),
        "cached_teacher_lengths": torch.tensor(
            [int(record["teacher_accepted_length"]) for record in records],
            dtype=torch.long,
        ),
    }


def prefix_lengths(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left == right).to(torch.long).cumprod(dim=-1).sum(dim=-1)


def prompt_balanced_mean(sample_ids: list[str], values: list[float]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample_id, value in zip(sample_ids, values, strict=True):
        grouped[sample_id].append(float(value))
    return sum(sum(group) / len(group) for group in grouped.values()) / len(grouped)


@torch.inference_mode()
def evaluate(
    *,
    head: ParallelLatticeCorrectionHead,
    target_weight: torch.Tensor,
    loader: DataLoader,
) -> dict[str, Any]:
    head.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    student_lengths: list[int] = []
    teacher_lengths: list[int] = []
    teacher_lcp: list[int] = []
    total_agree = 0
    total_tokens = 0
    first_agree = 0
    blocks = 0
    cached_length_mismatches = 0
    for batch in loader:
        hidden = batch["hidden"].to("cuda:0", non_blocking=True)
        anchors = batch["anchors"].to("cuda:0", non_blocking=True)
        prefix = batch["prefix"].to("cuda:0", non_blocking=True)
        teacher_ids = batch["teacher_ids"].to("cuda:0", non_blocking=True)
        teacher_full = batch["teacher_full"].to("cuda:0", non_blocking=True)
        gold_full = batch["gold_full"].to("cuda:0", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base_logits = F.linear(hidden, target_weight)
            output = head(
                parallel_hiddens=hidden,
                base_logits=base_logits,
                anchor_ids=anchors,
                prefix_ids=prefix,
                return_logits=False,
            )
        student_full = torch.cat([prefix[:, None], output.token_ids], dim=-1)
        batch_student_lengths = prefix_lengths(student_full, gold_full)
        batch_teacher_lengths = prefix_lengths(teacher_full, gold_full)
        batch_lcp = prefix_lengths(student_full, teacher_full)
        cached = batch["cached_teacher_lengths"].to("cuda:0")
        cached_length_mismatches += int((batch_teacher_lengths != cached).sum())
        agreement = output.token_ids == teacher_ids
        total_agree += int(agreement.sum())
        total_tokens += agreement.numel()
        first_agree += int(agreement[:, 0].sum())
        blocks += agreement.shape[0]
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        student_lengths.extend(batch_student_lengths.cpu().tolist())
        teacher_lengths.extend(batch_teacher_lengths.cpu().tolist())
        teacher_lcp.extend(batch_lcp.cpu().tolist())

    return {
        "blocks": blocks,
        "student_eal_prompt_balanced": prompt_balanced_mean(
            sample_ids, student_lengths
        ),
        "teacher_eal_prompt_balanced": prompt_balanced_mean(
            sample_ids, teacher_lengths
        ),
        "student_eal_round_weighted": sum(student_lengths) / len(student_lengths),
        "teacher_eal_round_weighted": sum(teacher_lengths) / len(teacher_lengths),
        "teacher_path_lcp_prompt_balanced": prompt_balanced_mean(
            sample_ids, teacher_lcp
        ),
        "all_corrected_token_agreement": total_agree / total_tokens,
        "first_corrected_token_agreement": first_agree / blocks,
        "cached_teacher_length_mismatches": cached_length_mismatches,
    }


def cosine_multiplier(step: int, total: int, warmup: int) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


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
    target_weight.requires_grad_(False)
    w_h.requires_grad_(False)
    w_out.requires_grad_(False)
    head = ParallelLatticeCorrectionHead(
        w_h=w_h,
        w_out=w_out,
        token_embeddings=(target_weight if args.use_semantic_embedding else None),
        use_full_hidden=args.use_full_hidden,
        max_positions=15,
        candidates=16,
        modes=args.modes,
        width=args.width,
        heads=args.heads,
        feed_forward_width=args.feed_forward_width,
        global_layers=args.global_layers,
    ).to("cuda:0")

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(step, total_steps, warmup_steps),
    )
    baseline = evaluate(head=head, target_weight=target_weight, loader=eval_loader)
    if baseline["cached_teacher_length_mismatches"]:
        raise RuntimeError(f"teacher replay mismatch: {baseline}")
    print(json.dumps({"untrained": baseline}, indent=2), flush=True)

    best_state = {
        name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()
    }
    best_epoch = 0
    best_eval = baseline
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    position_weights = (
        torch.arange(15, 0, -1, device="cuda:0", dtype=torch.float32)
    )
    position_weights = position_weights / position_weights.mean()
    for epoch in range(1, args.epochs + 1):
        head.train()
        epoch_ce = 0.0
        epoch_code = 0.0
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            hidden = batch["hidden"].to("cuda:0", non_blocking=True)
            anchors = batch["anchors"].to("cuda:0", non_blocking=True)
            prefix = batch["prefix"].to("cuda:0", non_blocking=True)
            teacher_ids = batch["teacher_ids"].to("cuda:0", non_blocking=True)
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
                token_losses = F.cross_entropy(
                    output.corrected_logits.reshape(-1, output.corrected_logits.shape[-1]),
                    teacher_ids.reshape(-1),
                    reduction="none",
                ).view_as(teacher_ids)
                teacher_ce = (
                    token_losses * position_weights[None]
                ).mean()
                scale = teacher_delta.float().square().mean(
                    dim=(-1, -2), keepdim=True
                ).sqrt().clamp_min(1e-3)
                code_loss = F.smooth_l1_loss(
                    output.correction_codes.float() / scale,
                    teacher_delta.float() / scale,
                )
                decay = 1.0 - 0.8 * (global_step / max(1, total_steps - 1))
                loss = teacher_ce + args.code_loss_weight * decay * code_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_ce += float(teacher_ce.detach())
            epoch_code += float(code_loss.detach())
            epoch_loss += float(loss.detach())
            batches += 1

        epoch_eval = evaluate(
            head=head, target_weight=target_weight, loader=eval_loader
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss / batches,
            "train_teacher_ce": epoch_ce / batches,
            "train_code_loss": epoch_code / batches,
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": epoch_eval,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2), flush=True)
        current_key = (
            epoch_eval["teacher_path_lcp_prompt_balanced"],
            epoch_eval["all_corrected_token_agreement"],
        )
        best_key = (
            best_eval["teacher_path_lcp_prompt_balanced"],
            best_eval["all_corrected_token_agreement"],
        )
        if current_key > best_key:
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in head.state_dict().items()
            }
            best_epoch = epoch
            best_eval = epoch_eval

    head.load_state_dict(best_state)
    train_eval_loader = DataLoader(
        TeacherDataset(train_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    selected_train = evaluate(
        head=head, target_weight=target_weight, loader=train_eval_loader
    )
    selected_validation = evaluate(
        head=head, target_weight=target_weight, loader=eval_loader
    )
    gate_pass = (
        selected_validation["student_eal_prompt_balanced"]
        >= selected_validation["teacher_eal_prompt_balanced"] - 0.10
    )
    torch.save(
        {
            "model": best_state,
            "architecture": {
                "positions": 15,
                "candidates": 16,
                "modes": args.modes,
                "width": args.width,
                "heads": args.heads,
                "feed_forward_width": args.feed_forward_width,
                "global_layers": args.global_layers,
                "use_semantic_embedding": args.use_semantic_embedding,
                "use_full_hidden": args.use_full_hidden,
            },
            "best_epoch": best_epoch,
        },
        args.output / "checkpoint.pt",
    )
    report = {
        "status": "completed",
        "stage": "plc_onpolicy_imitation",
        "config": vars(args) | {
            "canonical": str(args.canonical),
            "target": str(args.target),
            "domino_draft": str(args.domino_draft),
            "output": str(args.output),
        },
        "parameters": {
            "trainable": head.trainable_parameter_count,
            "active": head.active_parameter_count,
        },
        "untrained_validation": baseline,
        "best_epoch": best_epoch,
        "selected_train": selected_train,
        "selected_validation": selected_validation,
        "imitation_gate_pass": gate_pass,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
