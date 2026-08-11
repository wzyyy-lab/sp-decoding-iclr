#!/usr/bin/env python3
"""Train the released Domino causal head on GFPR current frontiers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from transformers import AutoModel

from sph.gfpr import (
    accepted_lengths,
    adaptation_state_dict,
    all_position_onpolicy_decode,
    all_position_teacher_logits,
    normalized_frontier_margin_loss,
    paired_prompt_summary,
    load_adaptation,
)
from train_domino_cached_head import MasterAdamW, load_tensor_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-rollout",
        type=Path,
        action="append",
        required=True,
        help="Repeat for prompt-balanced v0/v1 replay.",
    )
    parser.add_argument("--eval-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation_select")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--prompts-per-batch", type=int, default=2)
    parser.add_argument("--gradient-accumulation-prompts", type=int, default=16)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Use 1 for exact agreement with the released single-request runtime.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--position-zero-learning-rate",
        type=float,
        default=2e-2,
        help="Separate LR for alpha0; zero initialization otherwise moves too slowly.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--delta-l2",
        type=float,
        default=1e-3,
        help="L2-SP coefficient on the summed drift from the initial head.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--break-margin", type=float, default=1e-4)
    parser.add_argument("--keep-margin", type=float, default=0.05)
    parser.add_argument("--break-weight", type=float, default=1.0)
    parser.add_argument("--keep-weight", type=float, default=0.1)
    parser.add_argument("--correct-position-zero", action="store_true")
    parser.add_argument(
        "--trainable-scope",
        choices=("gru_rank", "full_head"),
        default="gru_rank",
        help="gru_rank freezes the 38.9M vocabulary projection for safer transfer.",
    )
    parser.add_argument(
        "--initial-adaptation",
        type=Path,
        help="Initialize from the policy that generated refreshed rollouts.",
    )
    parser.add_argument(
        "--capacity-only-allow-overlap",
        action="store_true",
        help="Allow train/eval prompt overlap but suppress every claim-bearing gate.",
    )
    parser.add_argument("--max-train-prompts", type=int)
    parser.add_argument("--max-eval-prompts", type=int)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--save-every-eval", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_records(root: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "gfpr_rollout_v1" or not metadata.get(
        "collection_complete", False
    ):
        raise RuntimeError(f"not a complete GFPR rollout collection: {root}")
    records: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        records.extend(
            record
            for record in torch.load(shard, map_location="cpu", weights_only=False)
            if str(record["split"]) == split
        )
    if not records:
        raise ValueError(f"no records in split={split!r}: {root}")
    return metadata, records


def validate_rollout_metadata(
    metadata: dict[str, Any],
    *,
    rollout: Path,
    target: Path,
    domino_draft: Path,
) -> None:
    expected = {
        "target": target.resolve(),
        "domino_draft": domino_draft.resolve(),
    }
    for field, expected_path in expected.items():
        stored = metadata.get(field)
        if stored is None or Path(stored).resolve() != expected_path:
            raise ValueError(
                f"{rollout} {field}={stored!r} is incompatible with {expected_path}"
            )
    if int(metadata.get("blocks", 0)) < 1:
        raise ValueError(f"{rollout} records no blocks")
    if metadata.get("policy_version") is None:
        raise ValueError(f"{rollout} omits policy_version")


def group_prompts(
    records: Iterable[dict[str, Any]], maximum: int | None
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(record)
    prompt_ids = sorted(grouped)
    if maximum is not None:
        if maximum < 1:
            raise ValueError("maximum prompts must be positive")
        prompt_ids = prompt_ids[:maximum]
    return {
        prompt_id: sorted(
            grouped[prompt_id], key=lambda row: int(row["anchor_offset"])
        )
        for prompt_id in prompt_ids
    }


def prompt_batches(
    grouped: dict[str, list[dict[str, Any]]],
    prompts_per_batch: int,
    rng: random.Random,
) -> list[list[dict[str, Any]]]:
    if prompts_per_batch < 1:
        raise ValueError("prompts_per_batch must be positive")
    prompt_ids = list(grouped)
    rng.shuffle(prompt_ids)
    batches: list[list[dict[str, Any]]] = []
    for start in range(0, len(prompt_ids), prompts_per_batch):
        batch: list[dict[str, Any]] = []
        for prompt_id in prompt_ids[start : start + prompts_per_batch]:
            prompt_records = grouped[prompt_id]
            prompt_weight = 1.0 / len(prompt_records)
            for record in prompt_records:
                copied = dict(record)
                copied["prompt_block_weight"] = prompt_weight
                batch.append(copied)
        batches.append(batch)
    return batches


def collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_ids": [str(record["sample_id"]) for record in records],
        "training_prompt_keys": [
            str(record.get("_training_prompt_key", record["sample_id"]))
            for record in records
        ],
        "domains": [str(record["domain"]) for record in records],
        "anchors": torch.tensor(
            [int(record["anchor_token_id"]) for record in records],
            dtype=torch.long,
        ),
        "hidden": torch.stack(
            [record["parallel_hidden"].to(torch.bfloat16) for record in records]
        ),
        "gold": torch.stack([record["gold_ids"].long() for record in records]),
        "released_ids": torch.stack(
            [record["policy_ids"].long() for record in records]
        ),
        "released_lengths": torch.tensor(
            [int(record["accepted_length"]) for record in records],
            dtype=torch.long,
        ),
        "block_weights": torch.tensor(
            [float(record.get("prompt_block_weight", 1.0)) for record in records],
            dtype=torch.float32,
        ),
    }


def _trainable_modules(domino: Any) -> list[nn.Module]:
    modules = [domino.prefix_gru, domino.embed_proj]
    if bool(getattr(domino, "use_bias_norm", False)):
        modules.append(domino.bias_norm)
    if bool(getattr(domino, "use_bias_gate", False)):
        modules.append(domino.bias_gate)
    return modules


def _set_module_mode(modules: Iterable[nn.Module], training: bool) -> None:
    for module in modules:
        module.train(training)


def _delta_l2(parameters: list[nn.Parameter], initial: list[Tensor]) -> Tensor:
    numerator = torch.zeros((), dtype=torch.float32, device=parameters[0].device)
    for parameter, reference in zip(parameters, initial, strict=True):
        difference = parameter.float() - reference
        numerator = numerator + difference.square().sum()
    return numerator


def _save_checkpoint(
    path: Path,
    *,
    domino: Any,
    position_zero_scale: Tensor,
    step: int,
    validation: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    payload = adaptation_state_dict(domino, position_zero_scale)
    payload.update(
        {"step": step, "validation": validation, "provenance": provenance}
    )
    torch.save(payload, path)


@torch.inference_mode()
def evaluate(
    *,
    domino: Any,
    target_weight: Tensor,
    position_zero_scale: Tensor,
    records: list[dict[str, Any]],
    batch_size: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    modules = _trainable_modules(domino)
    _set_module_mode(modules, False)
    sample_ids: list[str] = []
    domains: list[str] = []
    baseline_lengths: list[int] = []
    current_lengths: list[int] = []
    position_zero_changed = 0
    position_zero_baseline_correct = 0
    position_zero_current_correct = 0
    position_zero_repaired = 0
    position_zero_harmed = 0
    released_token_mismatches = 0
    released_length_mismatches = 0
    for start in range(0, len(records), batch_size):
        batch = collate(records[start : start + batch_size])
        hidden = batch["hidden"].to("cuda:0", non_blocking=True)
        anchors = batch["anchors"].to("cuda:0", non_blocking=True)
        gold = batch["gold"].to("cuda:0", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            decoded = all_position_onpolicy_decode(
                domino=domino,
                target_weight=target_weight,
                anchors=anchors,
                hidden=hidden,
                position_zero_scale=position_zero_scale,
                topk=1,
            )
        current = accepted_lengths(decoded.token_ids, gold)
        baseline = batch["released_lengths"]
        released_token_mismatches += int(
            decoded.token_ids.cpu().ne(batch["released_ids"]).sum()
        )
        released_length_mismatches += int(current.cpu().ne(baseline).sum())
        released_zero_correct = batch["released_ids"][:, 0].eq(batch["gold"][:, 0])
        current_zero_correct = decoded.token_ids[:, 0].cpu().eq(batch["gold"][:, 0])
        position_zero_baseline_correct += int(released_zero_correct.sum())
        position_zero_current_correct += int(current_zero_correct.sum())
        position_zero_repaired += int((~released_zero_correct & current_zero_correct).sum())
        position_zero_harmed += int((released_zero_correct & ~current_zero_correct).sum())
        position_zero_changed += int(
            decoded.token_ids[:, 0]
            .cpu()
            .ne(batch["released_ids"][:, 0])
            .sum()
        )
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        baseline_lengths.extend(baseline.tolist())
        current_lengths.extend(current.cpu().tolist())
    report = paired_prompt_summary(
        sample_ids,
        baseline_lengths,
        current_lengths,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    report["position_zero_changed_blocks"] = position_zero_changed
    report["position_zero_changed_fraction"] = position_zero_changed / len(records)
    report["released_token_mismatches"] = released_token_mismatches
    report["released_length_mismatches"] = released_length_mismatches
    report["position_zero_baseline_accuracy"] = (
        position_zero_baseline_correct / len(records)
    )
    report["position_zero_current_accuracy"] = (
        position_zero_current_correct / len(records)
    )
    report["position_zero_repaired_blocks"] = position_zero_repaired
    report["position_zero_harmed_blocks"] = position_zero_harmed
    by_domain: dict[str, Any] = {}
    for domain in sorted(set(domains)):
        indices = [index for index, value in enumerate(domains) if value == domain]
        by_domain[domain] = paired_prompt_summary(
            [sample_ids[index] for index in indices],
            [baseline_lengths[index] for index in indices],
            [current_lengths[index] for index in indices],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    report["by_domain"] = by_domain
    return report


def _learning_rate_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GFPR head training requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    if args.epochs < 1 or args.eval_every_steps < 1:
        raise ValueError("epochs and eval-every-steps must be positive")
    if args.learning_rate <= 0 or args.position_zero_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.gradient_accumulation_prompts < args.prompts_per_batch:
        raise ValueError("gradient accumulation must cover at least one prompt batch")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_sources = [
        load_records(path, args.train_split) for path in args.train_rollout
    ]
    eval_metadata, eval_records = load_records(args.eval_rollout, args.eval_split)
    for path, (metadata, _) in zip(
        args.train_rollout, train_sources, strict=True
    ):
        validate_rollout_metadata(
            metadata,
            rollout=path,
            target=args.target,
            domino_draft=args.domino_draft,
        )
        adaptation = metadata.get("adaptation")
        if adaptation is not None and args.initial_adaptation is None:
            raise ValueError(
                f"{path} was generated by an adapted policy; supply "
                "--initial-adaptation before replaying its frontiers"
            )
        if adaptation is not None and Path(adaptation).resolve() != Path(
            args.initial_adaptation
        ).resolve():
            raise ValueError(
                f"{path} was generated by {adaptation}, not the requested "
                f"initial policy {args.initial_adaptation}"
            )
    validate_rollout_metadata(
        eval_metadata,
        rollout=args.eval_rollout,
        target=args.target,
        domino_draft=args.domino_draft,
    )
    eval_adaptation = eval_metadata.get("adaptation")
    if args.initial_adaptation is None and eval_adaptation is not None:
        raise ValueError(
            "released initialization requires an evaluation rollout from the "
            "released policy"
        )
    if args.initial_adaptation is not None and (
        eval_adaptation is None
        or Path(eval_adaptation).resolve() != args.initial_adaptation.resolve()
    ):
        raise ValueError(
            "refreshed training requires a fixed evaluation rollout generated "
            "by the same --initial-adaptation"
        )
    train_metadata = [metadata for metadata, _ in train_sources]
    train_grouped: dict[str, list[dict[str, Any]]] = {}
    train_prompt_ids: set[str] = set()
    source_prompt_counts: list[int] = []
    for source_index, (_, source_records) in enumerate(train_sources):
        source_grouped = group_prompts(source_records, args.max_train_prompts)
        source_prompt_counts.append(len(source_grouped))
        train_prompt_ids.update(source_grouped)
        for prompt_id, prompt_records in source_grouped.items():
            training_key = f"{source_index}:{prompt_id}"
            train_grouped[training_key] = [
                {**record, "_training_prompt_key": training_key}
                for record in prompt_records
            ]
    if len(source_prompt_counts) > 1 and len(set(source_prompt_counts)) != 1:
        raise ValueError(
            "repeatable --train-rollout inputs must expose equal prompt counts "
            "for exact prompt-level source balancing"
        )
    eval_grouped = group_prompts(eval_records, args.max_eval_prompts)
    overlap = sorted(train_prompt_ids & set(eval_grouped))
    if overlap and not args.capacity_only_allow_overlap:
        raise RuntimeError(
            f"train/eval share {len(overlap)} prompts; use disjoint splits for Gate B "
            "or explicitly mark a same-set diagnostic with "
            "--capacity-only-allow-overlap"
        )
    eval_records = [record for group in eval_grouped.values() for record in group]

    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    )
    initial_position_zero_scale = torch.zeros(
        (), dtype=torch.float32, device="cuda:0"
    )
    if args.initial_adaptation is not None:
        initial_position_zero_scale = load_adaptation(
            domino,
            args.initial_adaptation,
            map_location="cuda:0",
            expected_target=args.target,
            expected_base_domino=args.domino_draft,
        ).to("cuda:0")
    domino.requires_grad_(False)
    modules = _trainable_modules(domino)
    for module in modules:
        # Keep the deployed BF16 forward path exact.  FP32 optimizer masters
        # below preserve small updates without changing inference arithmetic.
        module.requires_grad_(True)
    position_zero_scale = nn.Parameter(
        initial_position_zero_scale.float(),
        requires_grad=args.correct_position_zero,
    )
    named_parameters: list[tuple[str, nn.Parameter]] = []
    for module_name in ("prefix_gru", "embed_proj", "bias_norm", "bias_gate"):
        if not hasattr(domino, module_name):
            continue
        for name, parameter in getattr(domino, module_name).named_parameters():
            trainable = not (
                args.trainable_scope == "gru_rank"
                and module_name == "embed_proj"
                and name.startswith("2.")
            )
            parameter.requires_grad_(trainable)
            if trainable:
                named_parameters.append((f"{module_name}.{name}", parameter))
    if position_zero_scale.requires_grad:
        named_parameters.append(("position_zero_scale", position_zero_scale))
    parameters = [parameter for _, parameter in named_parameters]
    initial_parameters = [parameter.detach().float().clone() for parameter in parameters]
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to("cuda:0", torch.bfloat16)
    optimizer = MasterAdamW(named_parameters, args.learning_rate)
    if position_zero_scale.requires_grad:
        optimizer.optimizer = torch.optim.AdamW(
            [
                {
                    "params": optimizer.masters[:-1],
                    "lr": args.learning_rate,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [optimizer.masters[-1]],
                    "lr": args.position_zero_learning_rate,
                    "weight_decay": 0.0,
                },
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        base_learning_rates = [
            args.learning_rate,
            args.position_zero_learning_rate,
        ]
    else:
        optimizer.optimizer.param_groups[0]["weight_decay"] = args.weight_decay
        base_learning_rates = [args.learning_rate]

    prompt_batches_per_epoch = math.ceil(
        len(train_grouped) / args.prompts_per_batch
    )
    accumulation_batches = math.ceil(
        args.gradient_accumulation_prompts / args.prompts_per_batch
    )
    total_steps = args.epochs * math.ceil(
        prompt_batches_per_epoch / accumulation_batches
    )
    warmup_steps = int(round(total_steps * args.warmup_ratio))
    started = time.perf_counter()

    baseline = evaluate(
        domino=domino,
        target_weight=target_weight,
        position_zero_scale=position_zero_scale,
        records=eval_records,
        batch_size=args.eval_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if int(baseline["released_token_mismatches"]) != 0 or int(
        baseline["released_length_mismatches"]
    ) != 0:
        raise RuntimeError(
            "step-0 GFPR decode does not exactly reproduce the collected policy: "
            f"token_mismatches={baseline['released_token_mismatches']} "
            f"length_mismatches={baseline['released_length_mismatches']}"
        )
    checkpoint_provenance = {
        "target": str(args.target.resolve()),
        "base_domino": str(args.domino_draft.resolve()),
        "initial_adaptation": (
            str(args.initial_adaptation.resolve())
            if args.initial_adaptation is not None
            else None
        ),
        "block_size": int(domino.block_size),
        "train_rollouts": [str(path.resolve()) for path in args.train_rollout],
        "train_policy_versions": [
            metadata["policy_version"] for metadata in train_metadata
        ],
    }
    _save_checkpoint(
        args.output / "initial_adaptation.pt",
        domino=domino,
        position_zero_scale=position_zero_scale,
        step=0,
        validation=baseline,
        provenance=checkpoint_provenance,
    )
    best_eal = float(baseline["current_eal_prompt_balanced"])
    best_step = 0
    best_validation = baseline
    last_validation = baseline
    _save_checkpoint(
        args.output / "best_adaptation.pt",
        domino=domino,
        position_zero_scale=position_zero_scale,
        step=0,
        validation=baseline,
        provenance=checkpoint_provenance,
    )
    history: list[dict[str, Any]] = [
        {"epoch": 0, "step": 0, "validation": baseline}
    ]
    print(json.dumps(history[-1], indent=2), flush=True)

    rng = random.Random(args.seed)
    global_step = 0
    optimizer.zero_grad()
    initial_scale = _learning_rate_scale(0, total_steps, warmup_steps)
    for group, base_lr in zip(
        optimizer.optimizer.param_groups, base_learning_rates, strict=True
    ):
        group["lr"] = base_lr * initial_scale
    pending_batches = 0
    accumulation_prompt_denominator = 0
    running = defaultdict(float)
    running_batches = 0
    for epoch in range(1, args.epochs + 1):
        batches = prompt_batches(train_grouped, args.prompts_per_batch, rng)
        _set_module_mode(modules, True)
        for batch_index, records in enumerate(batches):
            if pending_batches == 0:
                upcoming = batches[
                    batch_index : batch_index + accumulation_batches
                ]
                accumulation_prompt_denominator = sum(
                    len(
                        {
                            str(
                                record.get(
                                    "_training_prompt_key", record["sample_id"]
                                )
                            )
                            for record in item
                        }
                    )
                    for item in upcoming
                )
            batch = collate(records)
            batch_prompt_count = len(set(batch["training_prompt_keys"]))
            hidden = batch["hidden"].to("cuda:0", non_blocking=True)
            anchors = batch["anchors"].to("cuda:0", non_blocking=True)
            gold = batch["gold"].to("cuda:0", non_blocking=True)
            weights = batch["block_weights"].to("cuda:0", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = all_position_teacher_logits(
                    domino=domino,
                    target_weight=target_weight,
                    anchors=anchors,
                    gold=gold,
                    hidden=hidden,
                    position_zero_scale=position_zero_scale,
                )
                frontier = normalized_frontier_margin_loss(
                    logits,
                    gold,
                    break_margin=args.break_margin,
                    keep_margin=args.keep_margin,
                    break_weight=args.break_weight,
                    keep_weight=args.keep_weight,
                    block_weights=weights,
                )
                l2 = _delta_l2(parameters, initial_parameters)
                loss = frontier.loss + args.delta_l2 * l2
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite GFPR loss at epoch={epoch} step={global_step}"
                )
            (loss * batch_prompt_count / accumulation_prompt_denominator).backward()
            if position_zero_scale.grad is not None:
                running["position_zero_abs_grad"] += float(
                    position_zero_scale.grad.detach().abs()
                )
            pending_batches += 1
            running["loss"] += float(loss.detach())
            running["repair"] += float(frontier.repair_loss)
            running["keep"] += float(frontier.keep_loss)
            running["l2"] += float(l2.detach())
            running["frontier"] += float(frontier.frontier.float().mean())
            running_batches += 1

            last_batch = batch_index + 1 == len(batches)
            if pending_batches < accumulation_batches and not last_batch:
                continue
            grad_norm = optimizer.step(args.max_grad_norm)
            if not math.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite GFPR gradient norm at step={global_step}"
                )
            optimizer.zero_grad()
            pending_batches = 0
            global_step += 1
            scale = _learning_rate_scale(
                global_step, total_steps, warmup_steps
            )
            for group, base_lr in zip(
                optimizer.optimizer.param_groups,
                base_learning_rates,
                strict=True,
            ):
                group["lr"] = base_lr * scale

            should_evaluate = (
                global_step % args.eval_every_steps == 0
                or (epoch == args.epochs and last_batch)
            )
            if not should_evaluate:
                continue
            validation = evaluate(
                domino=domino,
                target_weight=target_weight,
                position_zero_scale=position_zero_scale,
                records=eval_records,
                batch_size=args.eval_batch_size,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            entry = {
                "epoch": epoch,
                "step": global_step,
                "train": {
                    key: value / max(1, running_batches)
                    for key, value in running.items()
                },
                "grad_norm": float(grad_norm),
                "learning_rate": optimizer.optimizer.param_groups[0]["lr"],
                "position_zero_learning_rate": (
                    optimizer.optimizer.param_groups[1]["lr"]
                    if position_zero_scale.requires_grad
                    else 0.0
                ),
                "position_zero_scale": float(position_zero_scale.detach()),
                "validation": validation,
            }
            history.append(entry)
            last_validation = validation
            if args.save_every_eval:
                periodic_checkpoint = (
                    args.output / f"adaptation_step_{global_step:05d}.pt"
                )
                _save_checkpoint(
                    periodic_checkpoint,
                    domino=domino,
                    position_zero_scale=position_zero_scale,
                    step=global_step,
                    validation=validation,
                    provenance=checkpoint_provenance,
                )
                entry["checkpoint"] = str(periodic_checkpoint.resolve())
            print(json.dumps(entry, indent=2), flush=True)
            running.clear()
            running_batches = 0
            current_eal = float(validation["current_eal_prompt_balanced"])
            if current_eal > best_eal:
                best_eal = current_eal
                best_step = global_step
                best_validation = validation
                _save_checkpoint(
                    args.output / "best_adaptation.pt",
                    domino=domino,
                    position_zero_scale=position_zero_scale,
                    step=global_step,
                    validation=validation,
                    provenance=checkpoint_provenance,
                )
            _set_module_mode(modules, True)

    _save_checkpoint(
        args.output / "last_adaptation.pt",
        domino=domino,
        position_zero_scale=position_zero_scale,
        step=global_step,
        validation=last_validation,
        provenance=checkpoint_provenance,
    )

    report = {
        "status": "capacity_only" if overlap else "completed",
        "train_rollouts": [str(path.resolve()) for path in args.train_rollout],
        "eval_rollout": str(args.eval_rollout.resolve()),
        "train_modes": [metadata["mode"] for metadata in train_metadata],
        "train_policy_versions": [
            metadata["policy_version"] for metadata in train_metadata
        ],
        "eval_mode": eval_metadata["mode"],
        "initial_adaptation": (
            str(args.initial_adaptation.resolve())
            if args.initial_adaptation is not None
            else None
        ),
        "train_prompts": len(train_grouped),
        "train_blocks": sum(len(group) for group in train_grouped.values()),
        "eval_prompts": len(eval_grouped),
        "eval_blocks": len(eval_records),
        "overlapping_prompts": len(overlap),
        "correct_position_zero": args.correct_position_zero,
        "trainable_scope": args.trainable_scope,
        "position_zero_learning_rate": args.position_zero_learning_rate,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "total_optimizer_steps": global_step,
        "best_step": best_step,
        "best_validation": best_validation,
        "proof_of_signal_gate": {
            "eal_at_least_7_55": best_eal >= 7.55,
            "delta_at_least_0_30": float(best_validation["paired_delta"]) >= 0.30,
            "bootstrap_lower_above_zero": float(
                best_validation["paired_bootstrap_95_interval"][0]
            )
            > 0,
            "lost_to_gained_at_most_half": float(
                best_validation["lost_to_gained_ratio"]
            )
            <= 0.5,
            "harmful_prompts_at_most_20pct": float(
                best_validation["harmful_prompt_fraction"]
            )
            <= 0.2,
        },
        "history": history,
        "seconds": time.perf_counter() - started,
        "checkpoint": str((args.output / "best_adaptation.pt").resolve()),
        "last_checkpoint": str((args.output / "last_adaptation.pt").resolve()),
    }
    if overlap:
        report["proof_of_signal_gate"] = {
            "passed": False,
            "suppressed_due_to_train_eval_overlap": True,
        }
    else:
        report["proof_of_signal_gate"]["passed"] = all(
            report["proof_of_signal_gate"].values()
        )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
