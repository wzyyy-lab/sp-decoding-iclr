#!/usr/bin/env python3
"""Train/evaluate the full16 parallel global candidate fusion head."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

from safetensors import safe_open
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from sph.parallel_global_candidate_fusion import (
    BLOCK_LENGTH,
    CANDIDATES,
    DEFAULT_PARAMETER_COUNT,
    MatchedLocalCandidateFusionHead,
    ParallelGlobalCandidateFusionHead,
    PGCFLossOutput,
    pgcf_training_loss,
    supported_candidate_cross_entropy,
)


CAPACITY_GATE_KEYS = {
    "combined": (
        "candidate_accuracy",
        "hard_candidate_accuracy",
        "oracle_gap_recovered",
        "harmed_fraction",
        "teacher_action_accuracy",
    ),
    "target": (
        "candidate_accuracy",
        "hard_candidate_accuracy",
        "oracle_gap_recovered",
        "harmed_fraction",
    ),
    "teacher": ("teacher_action_accuracy",),
}


def effective_loss_progress(loss_mode: str, progress: float) -> float:
    if loss_mode == "teacher_only":
        return 0.0
    if loss_mode == "gold_ce":
        return 0.0
    if loss_mode != "curriculum":
        raise ValueError(f"unsupported loss mode: {loss_mode}")
    return progress


def validate_capacity_mode_pair(loss_mode: str, gate_mode: str) -> None:
    allowed_loss_modes = {
        "target": {"curriculum", "gold_ce"},
        "teacher": {"teacher_only"},
    }.get(gate_mode)
    if allowed_loss_modes is not None and loss_mode not in allowed_loss_modes:
        raise ValueError(
            f"capacity gate {gate_mode!r} allows loss modes "
            f"{sorted(allowed_loss_modes)!r}, got {loss_mode!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rollout", type=Path, required=True)
    parser.add_argument("--eval-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", choices=("global", "local"), default="global")
    parser.add_argument(
        "--loss-mode",
        choices=("curriculum", "teacher_only", "gold_ce"),
        default="curriculum",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation_select")
    parser.add_argument("--same-subset", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-eval-records", type=int, default=0)
    parser.add_argument("--train-diagnostic-records", type=int, default=0)
    parser.add_argument("--train-diagnostic-stride", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-multiplier", type=int, default=2)
    parser.add_argument("--require-default-parameter-count", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-capacity-gate", action="store_true")
    parser.add_argument(
        "--capacity-gate-mode",
        choices=("combined", "target", "teacher"),
        default="combined",
    )
    parser.add_argument("--min-candidate-accuracy", type=float, default=0.99)
    parser.add_argument("--min-hard-accuracy", type=float, default=0.97)
    parser.add_argument("--min-oracle-gap-recovered", type=float, default=0.95)
    parser.add_argument("--max-harmed-fraction", type=float, default=0.01)
    parser.add_argument("--min-teacher-accuracy", type=float, default=0.99)
    return parser.parse_args()


class RolloutDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("rollout dataset is empty")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rollout(
    root: Path,
    *,
    split: str,
    max_records: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("format") != "gfpr_rollout_v1":
        raise RuntimeError(f"unsupported full16 rollout format: {root}")
    if not metadata.get("collection_complete", False):
        raise RuntimeError(f"incomplete rollout: {root}")
    records: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        for record in torch.load(shard, map_location="cpu", weights_only=False):
            if str(record["split"]) != split:
                continue
            validate_record(record)
            records.append(record)
            if max_records and len(records) >= max_records:
                return metadata, records
    if not records:
        raise ValueError(f"no records for split {split!r} in {root}")
    return metadata, records


def fixed_stride_diagnostic_records(
    records: list[dict[str, Any]],
    *,
    count: int,
    stride: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select a label-independent diagnostic manifest in canonical order."""

    if count < 0:
        raise ValueError("train diagnostic record count must be non-negative")
    if stride < 1:
        raise ValueError("train diagnostic stride must be positive")
    indices = [stride * index for index in range(count)]
    if indices and indices[-1] >= len(records):
        raise ValueError(
            "train diagnostic manifest exceeds canonical training records: "
            f"last index {indices[-1]}, records {len(records)}"
        )
    return indices, [records[index] for index in indices]


def record_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Return label-free fields that uniquely identify one rollout block."""

    return {
        "sample_id": str(record["sample_id"]),
        "domain": str(record["domain"]),
        "anchor_offset": int(record["anchor_offset"]),
        "context_length": int(record["context_length"]),
    }


def validate_record(record: dict[str, Any]) -> None:
    expected = {
        "parallel_hidden": (BLOCK_LENGTH, 2560),
        "base_topk_ids": (BLOCK_LENGTH, CANDIDATES),
        "base_topk_logits": (BLOCK_LENGTH, CANDIDATES),
        "gold_ids": (BLOCK_LENGTH,),
        "policy_ids": (BLOCK_LENGTH,),
        "target_candidate_logits": (BLOCK_LENGTH, CANDIDATES),
        "target_top1_ids": (BLOCK_LENGTH,),
    }
    for key, shape in expected.items():
        value = record.get(key)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(
                f"record field {key} must have shape {shape}, got "
                f"{None if value is None else tuple(value.shape)}"
            )
def candidate_ranks(candidate_ids: Tensor, labels: Tensor) -> Tensor:
    matches = candidate_ids.eq(labels.unsqueeze(-1))
    ranks = matches.to(torch.int64).argmax(dim=-1)
    return torch.where(matches.any(dim=-1), ranks, torch.full_like(ranks, -1))


def collate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = torch.stack(
        [record["base_topk_ids"].long() for record in records]
    )
    gold_ids = torch.stack([record["gold_ids"].long() for record in records])
    policy_ids = torch.stack(
        [record["policy_ids"].long() for record in records]
    )
    target_top1_ids = torch.stack(
        [record["target_top1_ids"].long() for record in records]
    )
    return {
        "hidden": torch.stack(
            [record["parallel_hidden"] for record in records]
        ),
        "candidate_ids": candidate_ids,
        "candidate_logits": torch.stack(
            [record["base_topk_logits"].float() for record in records]
        ),
        "anchor_ids": torch.tensor(
            [int(record["anchor_token_id"]) for record in records],
            dtype=torch.long,
        ),
        "gold_ids": gold_ids,
        "gold_candidate_ranks": candidate_ranks(candidate_ids, gold_ids),
        "policy_ids": policy_ids,
        "teacher_candidate_ranks": candidate_ranks(candidate_ids, policy_ids),
        "target_candidate_logits": torch.stack(
            [record["target_candidate_logits"].float() for record in records]
        ),
        "target_matches_gold": target_top1_ids.eq(gold_ids),
        "sample_ids": [str(record["sample_id"]) for record in records],
        "domains": [str(record["domain"]) for record in records],
    }


def make_loader(
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        RolloutDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_records,
    )


def load_target_embedding(target: Path) -> Tensor:
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "model.embed_tokens.weight"
    shard = target / str(index["weight_map"][key])
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
        )
        for key, value in batch.items()
    }


def accepted_lengths(proposal: Tensor, gold: Tensor) -> Tensor:
    return torch.cumprod(proposal.eq(gold).to(torch.int64), dim=-1).sum(dim=-1)


def capacity_gold_ce_loss(
    output: Any, gold_candidate_ranks: Tensor
) -> PGCFLossOutput:
    """Capacity-only dense gold objective; never used by claim training."""

    dense_loss, support = supported_candidate_cross_entropy(
        output.scores, gold_candidate_ranks
    )
    zero = output.scores.float().sum() * 0.0
    empty = torch.zeros_like(support)
    return PGCFLossOutput(
        loss=dense_loss,
        prefix_loss=zero,
        target_kl_loss=zero,
        teacher_loss=zero,
        gold_support=support,
        target_kl_positions=empty,
        teacher_positions=empty,
        lambda_prefix=0.0,
        lambda_target_kl=0.0,
        lambda_teacher=0.0,
    )


def prompt_balanced(
    sample_ids: list[str], values: list[float]
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample_id, value in zip(sample_ids, values, strict=True):
        grouped[sample_id].append(float(value))
    return sum(sum(group) / len(group) for group in grouped.values()) / len(grouped)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    target_embedding: Tensor,
    device: torch.device,
    *,
    require_identity: bool = False,
) -> dict[str, Any]:
    model.eval()
    all_sample_ids: list[str] = []
    all_domains: list[str] = []
    model_lengths: list[float] = []
    base_lengths: list[float] = []
    released_lengths: list[float] = []
    oracle_lengths: list[float] = []
    harms = 0
    active_correct = active_total = 0
    hard_correct = hard_total = 0
    prefix_correct = prefix_total = 0
    teacher_correct = teacher_total = 0
    residual_abs_max = 0.0

    for cpu_batch in loader:
        batch = move_tensors(cpu_batch, device)
        candidate_embeddings = target_embedding[batch["candidate_ids"]]
        anchor_embeddings = target_embedding[batch["anchor_ids"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                batch["candidate_logits"],
                anchor_embeddings,
                candidate_embeddings=candidate_embeddings,
            )
        if require_identity:
            if not torch.equal(
                output.scores, batch["candidate_logits"].float()
            ):
                raise RuntimeError("zero-init PGCF scores do not equal base logits")
        selected_ranks = output.scores.argmax(dim=-1)
        proposal = batch["candidate_ids"].gather(
            -1, selected_ranks.unsqueeze(-1)
        ).squeeze(-1)
        base = batch["candidate_ids"][:, :, 0]
        gold = batch["gold_ids"]
        support = batch["gold_candidate_ranks"].ge(0)
        oracle = torch.where(support, gold, base)

        model_batch = accepted_lengths(proposal, gold)
        base_batch = accepted_lengths(base, gold)
        released_batch = accepted_lengths(batch["policy_ids"], gold)
        oracle_batch = accepted_lengths(oracle, gold)
        harms += int(model_batch.lt(base_batch).sum())

        prefix_support = torch.cumprod(
            support.to(torch.int64), dim=-1
        ).bool()
        correct = selected_ranks.eq(batch["gold_candidate_ranks"])
        hard = support & batch["gold_candidate_ranks"].gt(0)
        active_correct += int((correct & support).sum())
        active_total += int(support.sum())
        hard_correct += int((correct & hard).sum())
        hard_total += int(hard.sum())
        prefix_correct += int((correct & prefix_support).sum())
        prefix_total += int(prefix_support.sum())
        teacher_support = batch["teacher_candidate_ranks"].ge(0)
        teacher_correct += int(
            (
                selected_ranks.eq(batch["teacher_candidate_ranks"])
                & teacher_support
            ).sum()
        )
        teacher_total += int(teacher_support.sum())
        residual_abs_max = max(
            residual_abs_max, float(output.residual_scores.abs().max())
        )

        all_sample_ids.extend(batch["sample_ids"])
        all_domains.extend(batch["domains"])
        model_lengths.extend(model_batch.float().cpu().tolist())
        base_lengths.extend(base_batch.float().cpu().tolist())
        released_lengths.extend(released_batch.float().cpu().tolist())
        oracle_lengths.extend(oracle_batch.float().cpu().tolist())

    model_eal = prompt_balanced(all_sample_ids, model_lengths)
    base_eal = prompt_balanced(all_sample_ids, base_lengths)
    released_eal = prompt_balanced(all_sample_ids, released_lengths)
    oracle_eal = prompt_balanced(all_sample_ids, oracle_lengths)
    denominator = oracle_eal - base_eal
    domain_metrics: dict[str, dict[str, float]] = {}
    for domain in sorted(set(all_domains)):
        indices = [index for index, value in enumerate(all_domains) if value == domain]
        ids = [all_sample_ids[index] for index in indices]
        domain_metrics[domain] = {
            "model_eal": prompt_balanced(ids, [model_lengths[index] for index in indices]),
            "base_eal": prompt_balanced(ids, [base_lengths[index] for index in indices]),
            "released_eal": prompt_balanced(
                ids, [released_lengths[index] for index in indices]
            ),
        }
    return {
        "blocks": len(model_lengths),
        "prompts": len(set(all_sample_ids)),
        "model_eal": model_eal,
        "base_eal": base_eal,
        "released_eal": released_eal,
        "base16_oracle_eal": oracle_eal,
        "oracle_gap_recovered": (
            (model_eal - base_eal) / denominator if denominator > 0 else 0.0
        ),
        "candidate_accuracy": active_correct / max(1, active_total),
        "prefix_candidate_accuracy": prefix_correct / max(1, prefix_total),
        "hard_candidate_accuracy": hard_correct / max(1, hard_total),
        "supported_positions": active_total,
        "prefix_positions": prefix_total,
        "hard_positions": hard_total,
        "teacher_action_accuracy": teacher_correct / max(1, teacher_total),
        "teacher_positions": teacher_total,
        "harmed_fraction": harms / len(model_lengths),
        "residual_abs_max": residual_abs_max,
        "domains": domain_metrics,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    step: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "step": step,
            "metrics": metrics,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    validate_capacity_mode_pair(args.loss_mode, args.capacity_gate_mode)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("this claim-bearing PGCF run requires CUDA")

    train_metadata, train_records = load_rollout(
        args.train_rollout,
        split=args.train_split,
        max_records=args.max_train_records,
    )
    if args.same_subset:
        eval_metadata = train_metadata
        eval_records = list(train_records)
    else:
        eval_metadata, eval_records = load_rollout(
            args.eval_rollout,
            split=args.eval_split,
            max_records=args.max_eval_records,
        )
        overlap = {
            str(record["sample_id"]) for record in train_records
        } & {str(record["sample_id"]) for record in eval_records}
        if overlap:
            raise RuntimeError(f"train/eval prompt overlap: {sorted(overlap)[:3]}")

    model_type = (
        ParallelGlobalCandidateFusionHead
        if args.head == "global"
        else MatchedLocalCandidateFusionHead
    )
    model = model_type(
        hidden_size=2560,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_multiplier=args.ff_multiplier,
    ).to(device)
    if device.type == "cpu":
        model = model.to(dtype=torch.bfloat16)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if args.require_default_parameter_count and parameter_count != DEFAULT_PARAMETER_COUNT:
        raise RuntimeError(
            f"expected {DEFAULT_PARAMETER_COUNT} parameters, got {parameter_count}"
        )

    target_embedding = load_target_embedding(args.target).to(
        device=device, dtype=torch.bfloat16
    )
    train_loader = make_loader(
        train_records, batch_size=args.batch_size, shuffle=True
    )
    eval_loader = make_loader(
        eval_records, batch_size=args.eval_batch_size, shuffle=False
    )
    diagnostic_indices, diagnostic_records = fixed_stride_diagnostic_records(
        train_records,
        count=args.train_diagnostic_records,
        stride=args.train_diagnostic_stride,
    )
    diagnostic_loader = (
        make_loader(
            diagnostic_records,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )
        if diagnostic_records
        else None
    )
    identity = evaluate(
        model,
        eval_loader,
        target_embedding,
        device,
        require_identity=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_metrics: dict[str, Any] | None = None
    best_step = -1
    best_teacher_metrics: dict[str, Any] | None = None
    best_teacher_step = -1
    history: list[dict[str, Any]] = []
    iterator = iter(train_loader)
    model.train()
    for step in range(args.max_steps):
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            cpu_batch = next(iterator)
        batch = move_tensors(cpu_batch, device)
        progress = step / max(1, args.max_steps - 1)
        loss_progress = effective_loss_progress(args.loss_mode, progress)
        candidate_embeddings = target_embedding[batch["candidate_ids"]]
        anchor_embeddings = target_embedding[batch["anchor_ids"]]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                batch["candidate_logits"],
                anchor_embeddings,
                candidate_embeddings=candidate_embeddings,
            )
            if args.loss_mode == "gold_ce":
                loss_output = capacity_gold_ce_loss(
                    output, batch["gold_candidate_ranks"]
                )
            else:
                loss_output = pgcf_training_loss(
                    output,
                    batch["gold_candidate_ranks"],
                    progress=loss_progress,
                    target_candidate_logits=batch["target_candidate_logits"],
                    target_matches_gold=batch["target_matches_gold"],
                    teacher_candidate_ranks=batch["teacher_candidate_ranks"],
                )
        if not bool(torch.isfinite(loss_output.loss).detach().cpu()):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss_output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm).detach().cpu()):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()

        should_evaluate = (
            step == 0
            or (step + 1) % args.eval_every_steps == 0
            or step + 1 == args.max_steps
        )
        if should_evaluate:
            metrics = evaluate(
                model, eval_loader, target_embedding, device
            )
            train_diagnostic = (
                evaluate(model, diagnostic_loader, target_embedding, device)
                if diagnostic_loader is not None
                else None
            )
            record = {
                "step": step + 1,
                "progress": progress,
                "loss_progress": loss_progress,
                "train_loss": float(loss_output.loss.detach()),
                "prefix_loss": float(loss_output.prefix_loss.detach()),
                "target_kl_loss": float(loss_output.target_kl_loss.detach()),
                "teacher_loss": float(loss_output.teacher_loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "lambda_prefix": loss_output.lambda_prefix,
                "lambda_target_kl": loss_output.lambda_target_kl,
                "lambda_teacher": loss_output.lambda_teacher,
                "eval": metrics,
                "train_diagnostic": train_diagnostic,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if best_metrics is None or metrics["model_eal"] > best_metrics["model_eal"]:
                best_metrics = metrics
                best_step = step + 1
                save_checkpoint(
                    args.output / "best.pt",
                    model,
                    step=best_step,
                    metrics=metrics,
                    args=args,
                )
            if (args.loss_mode == "teacher_only" or progress <= 0.10) and (
                best_teacher_metrics is None
                or metrics["teacher_action_accuracy"]
                > best_teacher_metrics["teacher_action_accuracy"]
                or (
                    metrics["teacher_action_accuracy"]
                    == best_teacher_metrics["teacher_action_accuracy"]
                    and metrics["model_eal"]
                    > best_teacher_metrics["model_eal"]
                )
            ):
                best_teacher_metrics = metrics
                best_teacher_step = step + 1
                save_checkpoint(
                    args.output / "best_teacher.pt",
                    model,
                    step=best_teacher_step,
                    metrics=metrics,
                    args=args,
                )
            model.train()

    if best_metrics is None or best_teacher_metrics is None:
        raise RuntimeError("training produced no selectable checkpoints")
    checkpoint = torch.load(
        args.output / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    selected_metrics = evaluate(model, eval_loader, target_embedding, device)
    selected_train_diagnostic = (
        evaluate(model, diagnostic_loader, target_embedding, device)
        if diagnostic_loader is not None
        else None
    )
    save_checkpoint(
        args.output / "last_selected.pt",
        model,
        step=best_step,
        metrics=selected_metrics,
        args=args,
    )
    capacity_checks = {
        "candidate_accuracy": selected_metrics["candidate_accuracy"]
        >= args.min_candidate_accuracy,
        "hard_candidate_accuracy": selected_metrics["hard_candidate_accuracy"]
        >= args.min_hard_accuracy,
        "oracle_gap_recovered": selected_metrics["oracle_gap_recovered"]
        >= args.min_oracle_gap_recovered,
        "harmed_fraction": selected_metrics["harmed_fraction"]
        <= args.max_harmed_fraction,
        "teacher_action_accuracy": best_teacher_metrics["teacher_action_accuracy"]
        >= args.min_teacher_accuracy,
    }
    capacity_gate_keys = CAPACITY_GATE_KEYS[args.capacity_gate_mode]
    capacity_gate_passed = all(
        capacity_checks[key] for key in capacity_gate_keys
    )
    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": str(device),
        "parameter_count": parameter_count,
        "train_metadata": train_metadata,
        "eval_metadata": eval_metadata,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "train_diagnostic_manifest": {
            "selection": "canonical_index=stride*j",
            "count": len(diagnostic_records),
            "stride": args.train_diagnostic_stride,
            "indices": diagnostic_indices,
            "records": [record_identity(record) for record in diagnostic_records],
            "used_for_checkpoint_selection": False,
        },
        "selected_train_diagnostic": selected_train_diagnostic,
        "identity": identity,
        "best_step": best_step,
        "best_metrics": selected_metrics,
        "best_teacher_step": best_teacher_step,
        "best_teacher_metrics": best_teacher_metrics,
        "capacity_checks": capacity_checks,
        "capacity_gate_mode": args.capacity_gate_mode,
        "capacity_gate_keys": capacity_gate_keys,
        "capacity_gate_passed": capacity_gate_passed,
        "history": history,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({key: report[key] for key in (
        "parameter_count",
        "best_step",
        "best_metrics",
        "best_teacher_step",
        "best_teacher_metrics",
        "capacity_checks",
        "capacity_gate_mode",
        "capacity_gate_keys",
        "capacity_gate_passed",
    )}, indent=2), flush=True)
    if args.require_capacity_gate and not report["capacity_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
