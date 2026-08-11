#!/usr/bin/env python3
"""Train the global-lookahead causal Top-K selector on aligned Domino data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sph.global_lookahead_causal_selector import (
    GlobalLookaheadCausalSelector,
    topk_candidates,
)
from train_domino_cached_head import load_tensor_from_checkpoint
from train_plc_imitation import prefix_lengths, prompt_balanced_mean


DOMINO_FORMAT = "domino_same_anchor_hidden_v1"
RUNTIME_FORMAT = "plc_runtime_teacher_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-canonical", nargs="+", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--eval-split", default="validation_select")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument("--global-width", type=int, default=512)
    parser.add_argument("--global-heads", type=int, default=8)
    parser.add_argument("--global-layers", type=int, default=2)
    parser.add_argument("--global-modes", type=int, default=4)
    parser.add_argument("--feed-forward-width", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--adapter-learning-rate", type=float, default=2e-4)
    parser.add_argument("--base-learning-rate", type=float, default=2e-5)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--margin-offset", type=float, default=1.0)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--max-train-blocks", type=int)
    parser.add_argument("--max-eval-blocks", type=int)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_records(
    roots: list[Path], split: str, maximum: int | None
) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    expected_format: str | None = None
    for root in roots:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if not metadata.get("collection_complete", False):
            raise RuntimeError(f"incomplete canonical collection: {root}")
        current_format = str(metadata.get("format"))
        if current_format not in {DOMINO_FORMAT, RUNTIME_FORMAT}:
            raise ValueError(f"unsupported canonical format {current_format!r}")
        if expected_format is None:
            expected_format = current_format
        elif current_format != expected_format:
            raise ValueError("one training run cannot mix differently aligned formats")
        for shard in sorted(root.glob("shard-*.pt")):
            records.extend(
                record
                for record in torch.load(
                    shard, map_location="cpu", weights_only=False
                )
                if str(record["split"]) == split
            )
            if maximum is not None and len(records) >= maximum:
                break
        if maximum is not None and len(records) >= maximum:
            break
    if expected_format is None or not records:
        raise ValueError(f"no records found for split={split!r}")
    if maximum is not None:
        records = records[:maximum]
    return expected_format, records


class RecordDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate_records(
    records: list[dict[str, Any]], *, record_format: str
) -> dict[str, Any]:
    common = {
        "sample_ids": [str(record["sample_id"]) for record in records],
        "domains": [str(record["domain"]) for record in records],
        "anchors": torch.tensor(
            [int(record["anchor_token_id"]) for record in records],
            dtype=torch.long,
        ),
        "hidden": torch.stack(
            [record["parallel_hidden"].to(torch.bfloat16) for record in records]
        ),
    }
    if record_format == DOMINO_FORMAT:
        common.update(
            {
                "gold": torch.stack([record["gold_ids"].long() for record in records]),
                "released_full": torch.stack(
                    [record["released_onpolicy_ids"].long() for record in records]
                ),
                "released_lengths": torch.tensor(
                    [int(record["released_accepted_length"]) for record in records],
                    dtype=torch.long,
                ),
            }
        )
    elif record_format == RUNTIME_FORMAT:
        common.update(
            {
                "fixed_prefix": torch.tensor(
                    [int(record["base_prefix_token_id"]) for record in records],
                    dtype=torch.long,
                ),
                "gold_full": torch.stack(
                    [record["gold_full_ids"].long() for record in records]
                ),
                "gold": torch.stack([record["gold_ids"].long() for record in records]),
                "released_full": torch.stack(
                    [record["teacher_full_ids"].long() for record in records]
                ),
                "released_lengths": torch.tensor(
                    [int(record["teacher_accepted_length"]) for record in records],
                    dtype=torch.long,
                ),
            }
        )
    else:
        raise ValueError(f"unknown record format {record_format!r}")
    return common


def aligned_batch(
    *,
    batch: dict[str, Any],
    record_format: str,
    target_weight: Tensor,
    candidate_topk: int,
) -> dict[str, Any]:
    hidden = batch["hidden"].to("cuda:0", non_blocking=True)
    anchors = batch["anchors"].to("cuda:0", non_blocking=True)
    gold = batch["gold"].to("cuda:0", non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        base_logits = F.linear(hidden, target_weight)
    if record_format == DOMINO_FORMAT:
        fixed_prefix = base_logits[:, 0].argmax(dim=-1)
        correction_hiddens = hidden[:, 1:]
        correction_logits = base_logits[:, 1:]
        labels = gold[:, 1:]
        gold_full = gold
        released_full = batch["released_full"].to("cuda:0", non_blocking=True)
    elif record_format == RUNTIME_FORMAT:
        fixed_prefix = batch["fixed_prefix"].to("cuda:0", non_blocking=True)
        correction_hiddens = hidden
        correction_logits = base_logits
        labels = gold
        gold_full = batch["gold_full"].to("cuda:0", non_blocking=True)
        released_full = batch["released_full"].to("cuda:0", non_blocking=True)
    else:
        raise ValueError(f"unknown record format {record_format!r}")
    candidate_ids, candidate_logits = topk_candidates(
        correction_logits, candidate_topk
    )
    previous_ids = torch.cat(
        [fixed_prefix[:, None], labels[:, :-1]], dim=-1
    )
    return {
        "sample_ids": batch["sample_ids"],
        "domains": batch["domains"],
        "anchors": anchors,
        "fixed_prefix": fixed_prefix,
        "parallel_hiddens": correction_hiddens,
        "candidate_ids": candidate_ids,
        "candidate_logits": candidate_logits,
        "previous_ids": previous_ids,
        "labels": labels,
        "gold_full": gold_full,
        "released_full": released_full,
        "released_lengths": batch["released_lengths"].to("cuda:0"),
    }


def gold_candidate_targets(
    candidate_ids: Tensor, labels: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    matches = candidate_ids.eq(labels.unsqueeze(-1))
    available = matches.any(dim=-1)
    indices = matches.to(torch.int64).argmax(dim=-1)
    prefix_coverage = available.to(torch.int64).cumprod(dim=-1).to(torch.bool)
    return indices, available, prefix_coverage


def acceptance_loss(
    *,
    scores: Tensor,
    candidate_ids: Tensor,
    labels: Tensor,
    fixed_prefix: Tensor,
    gold_full: Tensor,
    margin_weight: float,
    margin_offset: float,
    auxiliary_weight: float,
    prompt_weights: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    gold_indices, available, prefix_coverage = gold_candidate_targets(
        candidate_ids, labels
    )
    prefix_reachable = fixed_prefix.eq(gold_full[:, 0])
    positions = labels.shape[1]
    continuation = torch.arange(
        positions, 0, -1, device=scores.device, dtype=torch.float32
    )
    primary_weights = (
        prefix_coverage.float()
        * prefix_reachable[:, None].float()
        * continuation[None]
    )
    auxiliary_weights = available.float() * auxiliary_weight
    weights = primary_weights + auxiliary_weights

    flat_ce = F.cross_entropy(
        scores.reshape(-1, scores.shape[-1]),
        gold_indices.reshape(-1),
        reduction="none",
    ).view_as(labels)
    gold_scores = scores.gather(-1, gold_indices.unsqueeze(-1)).squeeze(-1)
    masked = scores.masked_fill(
        F.one_hot(gold_indices, scores.shape[-1]).bool(), float("-inf")
    )
    competitor = masked.amax(dim=-1)
    margin = F.softplus(competitor - gold_scores + margin_offset)
    losses = flat_ce + margin_weight * margin

    block_denominator = weights.sum(dim=-1)
    active = block_denominator.gt(0)
    block_losses = (losses * weights).sum(dim=-1) / block_denominator.clamp_min(1.0)
    effective = prompt_weights.float() * active.float()
    loss = (block_losses * effective).sum() / effective.sum().clamp_min(1.0)
    predictions = scores.detach().argmax(dim=-1)
    covered_correct = predictions.eq(gold_indices) & available
    return loss, {
        "active_blocks": float(active.sum()),
        "primary_weight": float(primary_weights.sum()),
        "auxiliary_weight": float(auxiliary_weights.sum()),
        "covered_candidate_accuracy": float(
            covered_correct.sum() / available.sum().clamp_min(1)
        ),
        "fixed_prefix_reach": float(prefix_reachable.float().mean()),
    }


def cosine_multiplier(step: int, total: int, warmup: int) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


@torch.inference_mode()
def evaluate(
    *,
    head: GlobalLookaheadCausalSelector,
    target_weight: Tensor,
    loader: DataLoader,
    record_format: str,
    candidate_topk: int,
) -> dict[str, Any]:
    head.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    student_lengths: list[int] = []
    released_lengths: list[int] = []
    oracle_lengths: list[int] = []
    covered_correct = 0
    covered_positions = 0
    first_correct = 0
    first_available = 0
    improved = 0
    harmed = 0
    blocks = 0
    for raw_batch in loader:
        batch = aligned_batch(
            batch=raw_batch,
            record_format=record_format,
            target_weight=target_weight,
            candidate_topk=candidate_topk,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = head.decode(
                parallel_hiddens=batch["parallel_hiddens"],
                candidate_ids=batch["candidate_ids"],
                candidate_logits=batch["candidate_logits"],
                anchor_ids=batch["anchors"],
                fixed_prefix_ids=batch["fixed_prefix"],
            )
        student_full = torch.cat(
            [batch["fixed_prefix"][:, None], output.token_ids], dim=-1
        )
        if record_format == DOMINO_FORMAT:
            # The correction view omits the final matched-horizon position.
            gold_for_student = batch["gold_full"][:, : student_full.shape[1]]
            released_for_student = batch["released_full"][:, : student_full.shape[1]]
        else:
            gold_for_student = batch["gold_full"]
            released_for_student = batch["released_full"]
        lengths = prefix_lengths(student_full, gold_for_student)
        baseline = prefix_lengths(released_for_student, gold_for_student)
        _, available, coverage = gold_candidate_targets(
            batch["candidate_ids"], batch["labels"]
        )
        prefix_ok = batch["fixed_prefix"].eq(gold_for_student[:, 0])
        oracle = prefix_ok.long() * (1 + coverage.long().sum(dim=-1))
        predicted_ranks = output.candidate_scores.argmax(dim=-1)
        gold_ranks, _, _ = gold_candidate_targets(
            batch["candidate_ids"], batch["labels"]
        )
        correct = predicted_ranks.eq(gold_ranks) & available
        covered_correct += int(correct.sum())
        covered_positions += int(available.sum())
        first_correct += int(correct[:, 0].sum())
        first_available += int(available[:, 0].sum())
        improved += int((lengths > baseline).sum())
        harmed += int((lengths < baseline).sum())
        blocks += int(lengths.numel())
        sample_ids.extend(raw_batch["sample_ids"])
        domains.extend(raw_batch["domains"])
        student_lengths.extend(lengths.cpu().tolist())
        released_lengths.extend(baseline.cpu().tolist())
        oracle_lengths.extend(oracle.cpu().tolist())

    result = {
        "blocks": blocks,
        "student_eal_prompt_balanced": prompt_balanced_mean(
            sample_ids, student_lengths
        ),
        "released_eal_prompt_balanced": prompt_balanced_mean(
            sample_ids, released_lengths
        ),
        "oracle_eal_prompt_balanced": prompt_balanced_mean(sample_ids, oracle_lengths),
        "student_eal_round_weighted": sum(student_lengths) / len(student_lengths),
        "released_eal_round_weighted": sum(released_lengths) / len(released_lengths),
        "oracle_eal_round_weighted": sum(oracle_lengths) / len(oracle_lengths),
        "covered_candidate_accuracy": covered_correct / max(1, covered_positions),
        "first_correction_accuracy_when_available": first_correct
        / max(1, first_available),
        "improved_blocks": improved,
        "harmed_blocks": harmed,
        "unchanged_blocks": blocks - improved - harmed,
    }
    result["ratio_vs_released"] = (
        result["student_eal_prompt_balanced"]
        / result["released_eal_prompt_balanced"]
    )
    result["oracle_gap_recovery"] = (
        (result["student_eal_prompt_balanced"] - result["released_eal_prompt_balanced"])
        / max(
            1e-9,
            result["oracle_eal_prompt_balanced"]
            - result["released_eal_prompt_balanced"],
        )
    )
    return result


def build_head(
    *, args: argparse.Namespace, target_weight: Tensor
) -> GlobalLookaheadCausalSelector:
    first = load_tensor_from_checkpoint(
        args.domino_draft, "embed_proj.0.weight"
    )
    basis = load_tensor_from_checkpoint(
        args.domino_draft, "embed_proj.2.weight"
    )
    gru_ih = load_tensor_from_checkpoint(
        args.domino_draft, "prefix_gru.weight_ih_l0"
    )
    gru_hh = load_tensor_from_checkpoint(
        args.domino_draft, "prefix_gru.weight_hh_l0"
    )
    hidden_width = int(target_weight.shape[1])
    head = GlobalLookaheadCausalSelector(
        token_embeddings=target_weight,
        candidate_basis=basis,
        gru_weight_ih=gru_ih,
        gru_weight_hh=gru_hh,
        hidden_projection=first[:, :hidden_width],
        state_projection=first[:, hidden_width:],
        max_positions=15,
        candidates=args.candidate_topk,
        global_width=args.global_width,
        global_heads=args.global_heads,
        global_layers=args.global_layers,
        global_modes=args.global_modes,
        feed_forward_width=args.feed_forward_width,
    ).to("cuda:0")
    if args.init_checkpoint is not None:
        checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        head.load_state_dict(checkpoint["model"], strict=True)
    return head


def checkpoint_payload(
    *, head: GlobalLookaheadCausalSelector, args: argparse.Namespace, step: int
) -> dict[str, Any]:
    return {
        "model": {
            name: value.detach().cpu().clone()
            for name, value in head.state_dict().items()
        },
        "architecture": {
            "positions": head.max_positions,
            "candidates": head.candidates,
            "global_width": head.global_width,
            "global_heads": args.global_heads,
            "global_layers": args.global_layers,
            "global_modes": head.global_modes,
            "feed_forward_width": args.feed_forward_width,
        },
        "step": step,
    }


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
    torch.cuda.set_device(0)

    train_format, train_records = load_records(
        args.train_canonical, "train", args.max_train_blocks
    )
    eval_format, eval_records = load_records(
        [args.eval_runtime], args.eval_split, args.max_eval_blocks
    )
    if eval_format != RUNTIME_FORMAT:
        raise ValueError("the binding evaluator must use the aligned runtime cache")
    prompt_counts = Counter(str(record["sample_id"]) for record in train_records)
    prompt_weights = {
        sample_id: 1.0 / count for sample_id, count in prompt_counts.items()
    }

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        RecordDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=lambda records: collate_records(
            records, record_format=train_format
        ),
    )
    eval_loader = DataLoader(
        RecordDataset(eval_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=lambda records: collate_records(
            records, record_format=eval_format
        ),
    )
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to("cuda:0", torch.bfloat16)
    target_weight.requires_grad_(False)
    head = build_head(args=args, target_weight=target_weight)

    base_names = {
        "candidate_basis",
        "prefix_gru.weight_ih_l0",
        "prefix_gru.weight_hh_l0",
        "hidden_projection.weight",
        "state_projection.weight",
    }
    base_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if name in base_names
    ]
    adapter_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if name not in base_names
    ]
    if args.freeze_base:
        for parameter in base_parameters:
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": args.adapter_learning_rate},
            {"params": base_parameters, "lr": args.base_learning_rate},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(step, total_steps, warmup_steps),
    )

    initial = evaluate(
        head=head,
        target_weight=target_weight,
        loader=eval_loader,
        record_format=eval_format,
        candidate_topk=args.candidate_topk,
    )
    print(json.dumps({"initial": initial}, indent=2), flush=True)
    best_eval = initial
    best_step = 0
    best_state = checkpoint_payload(head=head, args=args, step=0)
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        head.train()
        head.clear_inference_table()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for raw_batch in train_loader:
            batch = aligned_batch(
                batch=raw_batch,
                record_format=train_format,
                target_weight=target_weight,
                candidate_topk=args.candidate_topk,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = head.teacher_forward(
                    parallel_hiddens=batch["parallel_hiddens"],
                    candidate_ids=batch["candidate_ids"],
                    candidate_logits=batch["candidate_logits"],
                    anchor_ids=batch["anchors"],
                    previous_ids=batch["previous_ids"],
                )
                weights = torch.tensor(
                    [prompt_weights[item] for item in raw_batch["sample_ids"]],
                    dtype=torch.float32,
                    device="cuda:0",
                )
                loss, diagnostics = acceptance_loss(
                    scores=output.candidate_scores,
                    candidate_ids=batch["candidate_ids"],
                    labels=batch["labels"],
                    fixed_prefix=batch["fixed_prefix"],
                    gold_full=batch["gold_full"],
                    margin_weight=args.margin_weight,
                    margin_offset=args.margin_offset,
                    auxiliary_weight=args.auxiliary_weight,
                    prompt_weights=weights,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batches += 1
            totals["loss"] += float(loss.detach())
            totals["grad_norm"] += grad_norm
            for key, value in diagnostics.items():
                totals[key] += value
            if global_step % 50 == 0:
                print(
                    f"step={global_step}/{total_steps} epoch={epoch} "
                    f"loss={float(loss.detach()):.6f} grad={grad_norm:.3f} "
                    f"adapter_lr={scheduler.get_last_lr()[0]:.3e} "
                    f"base_lr={scheduler.get_last_lr()[1]:.3e}",
                    flush=True,
                )
            if args.eval_every_steps and global_step % args.eval_every_steps == 0:
                current = evaluate(
                    head=head,
                    target_weight=target_weight,
                    loader=eval_loader,
                    record_format=eval_format,
                    candidate_topk=args.candidate_topk,
                )
                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "validation": current,
                }
                history.append(record)
                print(json.dumps(record, indent=2), flush=True)
                if current["student_eal_prompt_balanced"] > best_eval[
                    "student_eal_prompt_balanced"
                ]:
                    best_eval = current
                    best_step = global_step
                    best_state = checkpoint_payload(
                        head=head, args=args, step=global_step
                    )
                    torch.save(best_state, args.output / "best.pt")
                head.train()

        current = evaluate(
            head=head,
            target_weight=target_weight,
            loader=eval_loader,
            record_format=eval_format,
            candidate_topk=args.candidate_topk,
        )
        record = {
            "epoch": epoch,
            "step": global_step,
            "train": {key: value / batches for key, value in totals.items()},
            "validation": current,
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)
        if current["student_eal_prompt_balanced"] > best_eval[
            "student_eal_prompt_balanced"
        ]:
            best_eval = current
            best_step = global_step
            best_state = checkpoint_payload(head=head, args=args, step=global_step)
            torch.save(best_state, args.output / "best.pt")

    torch.save(best_state, args.output / "best.pt")
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "train_format": train_format,
        "train_blocks": len(train_records),
        "train_prompts": len(prompt_counts),
        "validation_blocks": len(eval_records),
        "trainable_parameters": head.trainable_parameter_count,
        "initial": initial,
        "best_step": best_step,
        "best_validation": best_eval,
        "history": history,
        "seconds": time.perf_counter() - started,
        "checkpoint": str((args.output / "best.pt").resolve()),
        "config": vars(args),
    }
    serializable = json.loads(json.dumps(report, default=str))
    (args.output / "report.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(serializable, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
