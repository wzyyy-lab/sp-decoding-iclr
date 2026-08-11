#!/usr/bin/env python3
"""Train the lightweight Top-K GFPR causal head on policy-induced frontiers."""

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
from torch import Tensor, nn
from transformers import AutoModel

from sph.gfpr import accepted_lengths, paired_prompt_summary
from sph.gfpr_candidate import (
    GFPRCandidateHead,
    candidate_dense_dpace_loss,
    candidate_dense_margin_loss,
    candidate_frontier_margin_loss,
    candidate_target_distillation_loss,
)
from train_domino_cached_head import MasterAdamW, load_tensor_from_checkpoint
from train_gfpr_head import (
    group_prompts,
    load_records,
    prompt_batches,
    validate_rollout_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rollout", type=Path, required=True)
    parser.add_argument("--eval-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation_select")
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument(
        "--include-released-action",
        action="store_true",
        help="Use exact Top-15 plus current-prefix Domino action K16 union.",
    )
    parser.add_argument(
        "--released-union-teacher",
        choices=("exact", "stored_frontier"),
        default="exact",
        help="Exact full-vocab or fast stored-action teacher for K16 union.",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=("calibration", "adapter", "input_rank", "gru_rank"),
        default="gru_rank",
    )
    parser.add_argument("--adapter-rank", type=int, default=16)
    parser.add_argument("--use-target-boundary-feature", action="store_true")
    parser.add_argument("--use-target-anchor-early-feature", action="store_true")
    parser.add_argument("--target-anchor-early-exit-layer", type=int)
    parser.add_argument("--required-alignment-report", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--prompts-per-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation-prompts", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--base-learning-rate", type=float, default=1e-4)
    parser.add_argument("--calibration-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--delta-l2", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--break-margin", type=float, default=1e-4)
    parser.add_argument("--keep-margin", type=float, default=0.05)
    parser.add_argument("--break-weight", type=float, default=1.0)
    parser.add_argument("--keep-weight", type=float, default=0.1)
    parser.add_argument(
        "--frontier-weight",
        type=float,
        default=1.0,
        help="Weight on the moving hard-label frontier objective.",
    )
    parser.add_argument(
        "--dense-dpace-weight",
        type=float,
        default=0.0,
        help="Weight on dense prefix-censored Candidate-D-PACE supervision.",
    )
    parser.add_argument("--dpace-alpha", type=float, default=0.5)
    parser.add_argument(
        "--dense-margin-weight",
        type=float,
        default=0.0,
        help="Weight on dense prefix-censored hinge supervision.",
    )
    parser.add_argument("--dense-margin", type=float, default=0.05)
    parser.add_argument(
        "--target-kl-weight",
        type=float,
        default=0.0,
        help="Candidate-set target KL on released reachable states.",
    )
    parser.add_argument(
        "--target-advantage-weight",
        type=float,
        default=0.0,
        help="Huber regression of target-vs-Domino candidate advantages.",
    )
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--target-huber-delta", type=float, default=0.5)
    parser.add_argument("--target-protect-weight", type=float, default=1.0)
    parser.add_argument("--target-repair-weight", type=float, default=4.0)
    parser.add_argument("--max-train-prompts", type=int)
    parser.add_argument("--max-eval-prompts", type=int)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--capacity-only-allow-overlap", action="store_true")
    parser.add_argument("--save-every-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def collate(
    records: list[dict[str, Any]],
    candidates: int,
    target_context_field: str | None = None,
) -> dict[str, Any]:
    ids = torch.stack(
        [record["base_topk_ids"].long()[..., :candidates] for record in records]
    )
    logits = torch.stack(
        [
            record["base_topk_logits"].float()[..., :candidates]
            for record in records
        ]
    )
    if ids.shape != logits.shape:
        raise ValueError("stored candidate IDs/logits differ in shape")
    target_fields = (
        "target_candidate_logits",
        "target_policy_logits",
        "target_top1_ids",
    )
    teacher_presence = [
        all(field in record for field in target_fields) for record in records
    ]
    if any(teacher_presence) and not all(teacher_presence):
        raise ValueError("target teacher fields are present for only part of a batch")
    target_candidate_logits = None
    target_policy_logits = None
    target_top1_ids = None
    if teacher_presence and teacher_presence[0]:
        target_candidate_logits = torch.stack(
            [
                record["target_candidate_logits"].float()[..., :candidates]
                for record in records
            ]
        )
        target_policy_logits = torch.stack(
            [record["target_policy_logits"].float() for record in records]
        )
        target_top1_ids = torch.stack(
            [record["target_top1_ids"].long() for record in records]
        )
        if target_candidate_logits.shape != ids.shape:
            raise ValueError("target candidate logits differ from lattice shape")
    target_context = None
    if target_context_field is not None:
        context_presence = [target_context_field in record for record in records]
        if not all(context_presence):
            raise ValueError(
                f"{target_context_field} is absent from part of a batch"
            )
        target_context = torch.stack(
            [record[target_context_field].to(torch.bfloat16) for record in records]
        )
    return {
        "sample_ids": [str(record["sample_id"]) for record in records],
        "domains": [str(record["domain"]) for record in records],
        "anchors": torch.tensor(
            [int(record["anchor_token_id"]) for record in records],
            dtype=torch.long,
        ),
        "hidden": torch.stack(
            [record["parallel_hidden"].to(torch.bfloat16) for record in records]
        ),
        "candidate_ids": ids,
        "candidate_logits": logits,
        "gold": torch.stack([record["gold_ids"].long() for record in records]),
        "released_ids": torch.stack(
            [record["policy_ids"].long() for record in records]
        ),
        "released_lengths": torch.tensor(
            [int(record["accepted_length"]) for record in records],
            dtype=torch.long,
        ),
        "target_candidate_logits": target_candidate_logits,
        "target_policy_logits": target_policy_logits,
        "target_top1_ids": target_top1_ids,
        "target_context": target_context,
        "block_weights": torch.tensor(
            [float(record.get("prompt_block_weight", 1.0)) for record in records],
            dtype=torch.float32,
        ),
    }


def _configure_scope(
    head: GFPRCandidateHead, scope: str
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    head.requires_grad_(False)
    base: list[tuple[str, nn.Parameter]] = []
    calibration: list[tuple[str, nn.Parameter]] = []
    if scope in {"input_rank", "gru_rank"}:
        for name, parameter in head.input_projection.named_parameters():
            parameter.requires_grad_(True)
            base.append((f"input_projection.{name}", parameter))
    if scope == "gru_rank":
        for name, parameter in head.prefix_gru.named_parameters():
            parameter.requires_grad_(True)
            base.append((f"prefix_gru.{name}", parameter))
    if scope == "adapter":
        module_names = ["residual_down", "residual_up"]
        if head.boundary_down is not None:
            module_names.append("boundary_down")
        for module_name in module_names:
            module = getattr(head, module_name)
            for name, parameter in module.named_parameters():
                parameter.requires_grad_(True)
                base.append((f"{module_name}.{name}", parameter))
    else:
        for name in ("base_scale", "correction_scale", "rank_bias"):
            parameter = getattr(head, name)
            parameter.requires_grad_(True)
            calibration.append((name, parameter))
    return base, calibration


def _l2(parameters: list[nn.Parameter], references: list[Tensor]) -> Tensor:
    value = torch.zeros((), dtype=torch.float32, device=parameters[0].device)
    for parameter, reference in zip(parameters, references, strict=True):
        value = value + (parameter.float() - reference).square().sum()
    return value


def _lr_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _save(
    path: Path,
    *,
    head: GFPRCandidateHead,
    args: argparse.Namespace,
    step: int,
    validation: dict[str, Any],
) -> None:
    torch.save(
        {
            "format": "gfpr_candidate_v1",
            "state_dict": head.state_dict(),
            "config": {
                "positions": head.positions,
                "candidates": head.candidates,
                "trainable_scope": args.trainable_scope,
                "dense_dpace_weight": args.dense_dpace_weight,
                "dpace_alpha": args.dpace_alpha,
                "dense_margin_weight": args.dense_margin_weight,
                "dense_margin": args.dense_margin,
                "adapter_rank": head.adapter_rank,
                "include_released_action": args.include_released_action,
                "released_union_teacher": args.released_union_teacher,
                "frontier_weight": args.frontier_weight,
                "target_kl_weight": args.target_kl_weight,
                "target_advantage_weight": args.target_advantage_weight,
                "target_temperature": args.target_temperature,
                "target_huber_delta": args.target_huber_delta,
                "target_protect_weight": args.target_protect_weight,
                "target_repair_weight": args.target_repair_weight,
                "use_target_boundary_feature": args.use_target_boundary_feature,
                "use_target_anchor_early_feature": (
                    args.use_target_anchor_early_feature
                ),
                "target_anchor_early_exit_layer": (
                    args.target_anchor_early_exit_layer
                ),
                "target_context_field": args.target_context_field,
                "boundary_width": head.boundary_width,
                "required_alignment_report": str(
                    args.required_alignment_report.resolve()
                )
                if args.required_alignment_report
                else None,
            },
            "provenance": {
                "target": str(args.target.resolve()),
                "base_domino": str(args.domino_draft.resolve()),
                "train_rollout": str(args.train_rollout.resolve()),
                "eval_rollout": str(args.eval_rollout.resolve()),
                "alignment_report": str(args.required_alignment_report.resolve())
                if args.required_alignment_report
                else None,
            },
            "step": step,
            "validation": validation,
        },
        path,
    )


@torch.inference_mode()
def evaluate(
    *,
    head: GFPRCandidateHead,
    records: list[dict[str, Any]],
    candidates: int,
    batch_size: int,
    bootstrap_samples: int,
    seed: int,
    include_released_action: bool,
    target_context_field: str | None,
) -> dict[str, Any]:
    head.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    baseline_lengths: list[int] = []
    current_lengths: list[int] = []
    token_changes = 0
    position_zero_changed = 0
    position_zero_repaired = 0
    position_zero_harmed = 0
    position_zero_baseline_correct = 0
    position_zero_current_correct = 0
    for start in range(0, len(records), batch_size):
        batch = collate(
            records[start : start + batch_size],
            candidates,
            target_context_field,
        )
        hidden = batch["hidden"].to("cuda:0", non_blocking=True)
        anchors = batch["anchors"].to("cuda:0", non_blocking=True)
        candidate_ids = batch["candidate_ids"].to("cuda:0", non_blocking=True)
        candidate_logits = batch["candidate_logits"].to(
            "cuda:0", non_blocking=True
        )
        gold = batch["gold"].to("cuda:0", non_blocking=True)
        target_context = (
            batch["target_context"].to("cuda:0", non_blocking=True)
            if head.boundary_width and batch["target_context"] is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if include_released_action:
                decoded = head.decode_with_released_union(
                    anchors=anchors,
                    hidden=hidden,
                    base_candidate_ids=candidate_ids,
                    target_boundary=target_context,
                )
            else:
                decoded = head.decode(
                    anchors=anchors,
                    hidden=hidden,
                    candidate_ids=candidate_ids,
                    candidate_logits=candidate_logits,
                    target_boundary=target_context,
                )
        current = accepted_lengths(decoded.token_ids, gold)
        released = batch["released_ids"]
        current_cpu = decoded.token_ids.cpu()
        gold_cpu = batch["gold"]
        released_zero = released[:, 0].eq(gold_cpu[:, 0])
        current_zero = current_cpu[:, 0].eq(gold_cpu[:, 0])
        token_changes += int(current_cpu.ne(released).sum())
        position_zero_changed += int(current_cpu[:, 0].ne(released[:, 0]).sum())
        position_zero_repaired += int((~released_zero & current_zero).sum())
        position_zero_harmed += int((released_zero & ~current_zero).sum())
        position_zero_baseline_correct += int(released_zero.sum())
        position_zero_current_correct += int(current_zero.sum())
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        baseline_lengths.extend(batch["released_lengths"].tolist())
        current_lengths.extend(current.cpu().tolist())
    report = paired_prompt_summary(
        sample_ids,
        baseline_lengths,
        current_lengths,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    report.update(
        {
            "tokens_changed_vs_released": token_changes,
            "position_zero_changed_blocks": position_zero_changed,
            "position_zero_baseline_accuracy": position_zero_baseline_correct
            / len(records),
            "position_zero_current_accuracy": position_zero_current_correct
            / len(records),
            "position_zero_repaired_blocks": position_zero_repaired,
            "position_zero_harmed_blocks": position_zero_harmed,
        }
    )
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("candidate GFPR training requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    if args.candidate_topk < 2 or args.epochs < 1 or args.adapter_rank < 1:
        raise ValueError("candidate count and epochs must be positive")
    if min(args.base_learning_rate, args.calibration_learning_rate) <= 0:
        raise ValueError("learning rates must be positive")
    if (
        min(
            args.frontier_weight,
            args.dense_dpace_weight,
            args.dense_margin_weight,
            args.dense_margin,
            args.target_kl_weight,
            args.target_advantage_weight,
            args.target_protect_weight,
            args.target_repair_weight,
        )
        < 0
        or not 0.0 <= args.dpace_alpha <= 1.0
        or args.target_temperature <= 0
        or args.target_huber_delta <= 0
    ):
        raise ValueError("invalid dense candidate objective configuration")
    uses_target_teacher = bool(
        args.target_kl_weight > 0 or args.target_advantage_weight > 0
    )
    if uses_target_teacher and not args.include_released_action:
        raise ValueError("target distillation currently requires released-action union")
    if args.include_released_action and (
        args.trainable_scope != "adapter" or args.candidate_topk != 16
    ):
        raise ValueError(
            "released-action union currently requires the exact K16 adapter scope"
        )
    if args.include_released_action and args.eval_batch_size != 1:
        raise ValueError(
            "released-action identity requires batch-1 evaluation to match runtime"
        )
    if args.use_target_boundary_feature and args.use_target_anchor_early_feature:
        raise ValueError("choose only one target context feature")
    if args.use_target_anchor_early_feature:
        if args.target_anchor_early_exit_layer is None:
            raise ValueError("anchor early feature requires its target layer count")
        if args.target_anchor_early_exit_layer < 1:
            raise ValueError("anchor early-exit layer must be positive")
        args.target_context_field = "target_anchor_early_feature"
    elif args.use_target_boundary_feature:
        if args.target_anchor_early_exit_layer is not None:
            raise ValueError("target anchor layer was set without its feature")
        args.target_context_field = "target_boundary_feature"
    else:
        if args.target_anchor_early_exit_layer is not None:
            raise ValueError("target anchor layer was set without its feature")
        args.target_context_field = None
    alignment_contract: dict[str, Any] | None = None
    if args.required_alignment_report is not None:
        alignment_contract = json.loads(
            args.required_alignment_report.read_text(encoding="utf-8")
        )
        if alignment_contract.get("format") != "r047_anchor_alignment_v1":
            raise ValueError("required alignment report has the wrong format")
        if int(alignment_contract.get("early_layers", -1)) != int(
            args.target_anchor_early_exit_layer or -1
        ):
            raise ValueError("alignment report used a different target layer")
        if not alignment_contract.get("gate", {}).get("passed", False):
            raise RuntimeError("required anchor alignment gate did not pass")
        if "checkpoint_path_alignment" not in alignment_contract:
            raise RuntimeError(
                "Phase3 requires token-path alignment from a trained smoke checkpoint"
            )
        if Path(alignment_contract["collection"]).resolve() != args.eval_rollout.resolve():
            raise ValueError("alignment report does not bind the current eval rollout")
        if Path(alignment_contract["target"]).resolve() != args.target.resolve():
            raise ValueError("alignment report does not bind the current target")
        checkpoint_alignment = alignment_contract["checkpoint_path_alignment"]
        if int(checkpoint_alignment.get("checkpoint_step", 0)) <= 0:
            raise ValueError("alignment report checkpoint was not trained")
        if float(checkpoint_alignment.get("checkpoint_residual_up_norm", 0.0)) <= 0:
            raise ValueError("alignment report checkpoint retained zero residual")
        alignment_provenance = checkpoint_alignment.get(
            "checkpoint_provenance", {}
        )
        expected_alignment_paths = {
            "target": args.target.resolve(),
            "base_domino": args.domino_draft.resolve(),
            "eval_rollout": args.eval_rollout.resolve(),
        }
        for field, expected_path in expected_alignment_paths.items():
            if Path(str(alignment_provenance.get(field, ""))).resolve() != expected_path:
                raise ValueError(
                    f"alignment checkpoint {field} differs from current run"
                )
    elif not args.use_target_anchor_early_feature:
        alignment_contract = None
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_metadata, train_records = load_records(
        args.train_rollout, args.train_split
    )
    eval_metadata, eval_records = load_records(args.eval_rollout, args.eval_split)
    for path, metadata in (
        (args.train_rollout, train_metadata),
        (args.eval_rollout, eval_metadata),
    ):
        validate_rollout_metadata(
            metadata,
            rollout=path,
            target=args.target,
            domino_draft=args.domino_draft,
        )
        if metadata.get("adaptation") is not None:
            raise ValueError("candidate v0 training expects released-policy rollouts")
    train_grouped = group_prompts(train_records, args.max_train_prompts)
    eval_grouped = group_prompts(eval_records, args.max_eval_prompts)
    overlap = sorted(set(train_grouped) & set(eval_grouped))
    if overlap and not args.capacity_only_allow_overlap:
        raise RuntimeError(f"train/eval share {len(overlap)} prompts")
    eval_records = [record for group in eval_grouped.values() for record in group]
    boundary_width = 0
    if args.target_context_field is not None:
        missing_context = sum(
            args.target_context_field not in record
            for groups in (train_grouped.values(), eval_grouped.values())
            for group in groups
            for record in group
        )
        if missing_context:
            raise ValueError(
                f"{args.target_context_field} requested but "
                f"{missing_context} records lack it"
            )
        boundary_widths = {
            int(record[args.target_context_field].numel())
            for groups in (train_grouped.values(), eval_grouped.values())
            for group in groups
            for record in group
        }
        if len(boundary_widths) != 1:
            raise ValueError("target context width changes across rollouts")
        boundary_width = boundary_widths.pop()
    if args.use_target_anchor_early_feature:
        for name, metadata in (
            ("train", train_metadata),
            ("eval", eval_metadata),
        ):
            contract = metadata.get("target_anchor_early_exit_feature", {})
            if not contract.get("stored", False):
                raise ValueError(f"{name} rollout lacks anchor early-exit metadata")
            if int(contract.get("early_layers", -1)) != int(
                args.target_anchor_early_exit_layer
            ):
                raise ValueError(f"{name} rollout used a different early layer")
    if uses_target_teacher:
        missing_teacher = sum(
            "target_candidate_logits" not in record
            or "target_policy_logits" not in record
            for group in train_grouped.values()
            for record in group
        )
        if missing_teacher:
            raise ValueError(
                f"target distillation requested but {missing_teacher} train records "
                "lack target logits"
            )

    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to("cuda:0", torch.bfloat16)
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    head = GFPRCandidateHead.from_domino(
        domino,
        target_weight,
        positions=16,
        candidates=args.candidate_topk,
        adapter_rank=args.adapter_rank,
        boundary_width=boundary_width,
    )
    del domino
    base_named, calibration_named = _configure_scope(head, args.trainable_scope)
    named = base_named + calibration_named
    parameters = [parameter for _, parameter in named]
    trainable_parameter_count = sum(parameter.numel() for parameter in parameters)
    if (
        args.use_target_anchor_early_feature
        and args.adapter_rank == 64
        and boundary_width == 2560
        and trainable_parameter_count != 409_600
    ):
        raise RuntimeError(
            "R047 rank-64 anchor head must have exactly 409,600 trainable parameters"
        )
    references = [parameter.detach().float().clone() for parameter in parameters]
    optimizer = MasterAdamW(named, args.base_learning_rate)
    base_count = len(base_named)
    groups = []
    group_lrs = []
    if base_count:
        groups.append(
            {
                "params": optimizer.masters[:base_count],
                "lr": args.base_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
        group_lrs.append(args.base_learning_rate)
    if calibration_named:
        groups.append(
            {
                "params": optimizer.masters[base_count:],
                "lr": args.calibration_learning_rate,
                "weight_decay": 0.0,
            }
        )
        group_lrs.append(args.calibration_learning_rate)
    optimizer.optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.95), eps=1e-8
    )

    accumulation_batches = math.ceil(
        args.gradient_accumulation_prompts / args.prompts_per_batch
    )
    batches_per_epoch = math.ceil(len(train_grouped) / args.prompts_per_batch)
    total_steps = args.epochs * math.ceil(
        batches_per_epoch / accumulation_batches
    )
    warmup_steps = int(round(total_steps * args.warmup_ratio))
    started = time.perf_counter()
    baseline = evaluate(
        head=head,
        records=eval_records,
        candidates=args.candidate_topk,
        batch_size=args.eval_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        include_released_action=args.include_released_action,
        target_context_field=args.target_context_field,
    )
    if args.include_released_action and int(
        baseline["tokens_changed_vs_released"]
    ) != 0:
        raise RuntimeError(
            "zero-residual released-action union does not exactly reproduce Domino"
        )
    best_eal = float(baseline["current_eal_prompt_balanced"])
    best_step = 0
    best_validation = baseline
    last_validation = baseline
    _save(
        args.output / "initial_candidate.pt",
        head=head,
        args=args,
        step=0,
        validation=baseline,
    )
    _save(
        args.output / "best_candidate.pt",
        head=head,
        args=args,
        step=0,
        validation=baseline,
    )
    history: list[dict[str, Any]] = [
        {"epoch": 0, "step": 0, "validation": baseline}
    ]
    print(json.dumps(history[-1], indent=2), flush=True)

    rng = random.Random(args.seed)
    optimizer.zero_grad()
    global_step = 0
    pending_batches = 0
    accumulation_prompt_denominator = 0
    running = defaultdict(float)
    running_batches = 0
    initial_scale = _lr_scale(0, total_steps, warmup_steps)
    for group, lr in zip(optimizer.optimizer.param_groups, group_lrs, strict=True):
        group["lr"] = lr * initial_scale

    for epoch in range(1, args.epochs + 1):
        batches = prompt_batches(train_grouped, args.prompts_per_batch, rng)
        head.train()
        for batch_index, records in enumerate(batches):
            if pending_batches == 0:
                upcoming = batches[
                    batch_index : batch_index + accumulation_batches
                ]
                accumulation_prompt_denominator = sum(
                    len({str(record["sample_id"]) for record in item})
                    for item in upcoming
                )
            batch = collate(
                records,
                args.candidate_topk,
                args.target_context_field,
            )
            batch_prompt_count = len(set(batch["sample_ids"]))
            hidden = batch["hidden"].to("cuda:0", non_blocking=True)
            anchors = batch["anchors"].to("cuda:0", non_blocking=True)
            candidate_ids = batch["candidate_ids"].to(
                "cuda:0", non_blocking=True
            )
            candidate_logits = batch["candidate_logits"].to(
                "cuda:0", non_blocking=True
            )
            base_candidate_ids = candidate_ids
            gold = batch["gold"].to("cuda:0", non_blocking=True)
            target_context = (
                batch["target_context"].to("cuda:0", non_blocking=True)
                if head.boundary_width and batch["target_context"] is not None
                else None
            )
            weights = batch["block_weights"].to("cuda:0", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.include_released_action:
                    if args.released_union_teacher == "stored_frontier":
                        teacher = head.teacher_stored_union_scores(
                            anchors=anchors,
                            gold=gold,
                            hidden=hidden,
                            base_candidate_ids=candidate_ids,
                            released_token_ids=batch["released_ids"].to(
                                "cuda:0", non_blocking=True
                            ),
                            target_boundary=target_context,
                        )
                    else:
                        teacher = head.teacher_union_scores(
                            anchors=anchors,
                            gold=gold,
                            hidden=hidden,
                            base_candidate_ids=candidate_ids,
                            target_boundary=target_context,
                        )
                    scores = teacher.scores
                    candidate_ids = teacher.candidate_ids
                else:
                    scores = head.teacher_scores(
                        anchors=anchors,
                        gold=gold,
                        hidden=hidden,
                        candidate_ids=candidate_ids,
                        candidate_logits=candidate_logits,
                        target_boundary=target_context,
                    )
                frontier = candidate_frontier_margin_loss(
                    scores,
                    candidate_ids,
                    gold,
                    break_margin=args.break_margin,
                    keep_margin=args.keep_margin,
                    break_weight=args.break_weight,
                    keep_weight=args.keep_weight,
                    block_weights=weights,
                )
                dense = candidate_dense_dpace_loss(
                    scores,
                    candidate_ids,
                    gold,
                    alpha=args.dpace_alpha,
                    block_weights=weights,
                )
                dense_margin = candidate_dense_margin_loss(
                    scores,
                    candidate_ids,
                    gold,
                    margin=args.dense_margin,
                    alpha=args.dpace_alpha,
                    block_weights=weights,
                )
                target_kl = torch.zeros((), device=hidden.device)
                target_advantage = torch.zeros((), device=hidden.device)
                target_active_fraction = torch.zeros((), device=hidden.device)
                target_top1_match_fraction = torch.zeros(
                    (), device=hidden.device
                )
                if uses_target_teacher:
                    target_distill = candidate_target_distillation_loss(
                        scores,
                        candidate_ids,
                        base_candidate_ids,
                        batch["target_candidate_logits"].to(
                            "cuda:0", non_blocking=True
                        ),
                        batch["released_ids"].to(
                            "cuda:0", non_blocking=True
                        ),
                        batch["target_policy_logits"].to(
                            "cuda:0", non_blocking=True
                        ),
                        gold,
                        batch["released_lengths"].to(
                            "cuda:0", non_blocking=True
                        ),
                        temperature=args.target_temperature,
                        huber_delta=args.target_huber_delta,
                        protect_weight=args.target_protect_weight,
                        repair_weight=args.target_repair_weight,
                        block_weights=weights,
                    )
                    target_kl = target_distill.kl_loss
                    target_advantage = target_distill.advantage_loss
                    target_active = target_distill.active_positions
                    target_active_fraction = target_active.float().mean()
                    target_top1_match_fraction = (
                        target_distill.raw_teacher_top1_matches_gold
                        & target_active
                    ).sum() / target_active.sum().clamp_min(1)
                l2 = _l2(parameters, references)
                loss = (
                    args.frontier_weight * frontier.loss
                    + args.dense_dpace_weight * dense.loss
                    + args.dense_margin_weight * dense_margin.loss
                    + args.target_kl_weight * target_kl
                    + args.target_advantage_weight * target_advantage
                    + args.delta_l2 * l2
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite candidate GFPR loss")
            (loss * batch_prompt_count / accumulation_prompt_denominator).backward()
            pending_batches += 1
            running["loss"] += float(loss.detach())
            running["repair"] += float(frontier.repair_loss)
            running["keep"] += float(frontier.keep_loss)
            running["dense_dpace"] += float(dense.loss.detach())
            running["dense_nll"] += float(dense.unweighted_nll)
            running["dense_active_fraction"] += float(
                dense.active_positions.float().mean()
            )
            running["dense_margin"] += float(dense_margin.loss.detach())
            running["dense_margin_violation"] += float(
                dense_margin.violations.sum()
                / dense_margin.active_positions.sum().clamp_min(1)
            )
            running["target_kl"] += float(target_kl.detach())
            running["target_advantage"] += float(target_advantage.detach())
            running["target_active_fraction"] += float(
                target_active_fraction.detach()
            )
            running["target_raw_top1_match_fraction"] += float(
                target_top1_match_fraction.detach()
            )
            running["l2"] += float(l2.detach())
            running["frontier"] += float(frontier.frontier.float().mean())
            running["repairable_fraction"] += float(
                frontier.repairable_blocks.float().mean()
            )
            running_batches += 1
            last_batch = batch_index + 1 == len(batches)
            if pending_batches < accumulation_batches and not last_batch:
                continue
            grad_norm = optimizer.step(args.max_grad_norm)
            if not math.isfinite(grad_norm):
                raise FloatingPointError("non-finite candidate GFPR gradient")
            optimizer.zero_grad()
            pending_batches = 0
            global_step += 1
            scale = _lr_scale(global_step, total_steps, warmup_steps)
            for group, lr in zip(
                optimizer.optimizer.param_groups, group_lrs, strict=True
            ):
                group["lr"] = lr * scale
            should_evaluate = (
                global_step % args.eval_every_steps == 0
                or (epoch == args.epochs and last_batch)
            )
            if not should_evaluate:
                continue
            validation = evaluate(
                head=head,
                records=eval_records,
                candidates=args.candidate_topk,
                batch_size=args.eval_batch_size,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                include_released_action=args.include_released_action,
                target_context_field=args.target_context_field,
            )
            last_validation = validation
            entry = {
                "epoch": epoch,
                "step": global_step,
                "train": {
                    key: value / max(1, running_batches)
                    for key, value in running.items()
                },
                "grad_norm": grad_norm,
                "learning_rates": [
                    group["lr"] for group in optimizer.optimizer.param_groups
                ],
                "base_scale": head.base_scale.detach().cpu().tolist(),
                "correction_scale": head.correction_scale.detach().cpu().tolist(),
                "validation": validation,
            }
            if args.save_every_eval:
                path = args.output / f"candidate_step_{global_step:05d}.pt"
                _save(
                    path,
                    head=head,
                    args=args,
                    step=global_step,
                    validation=validation,
                )
                entry["checkpoint"] = str(path.resolve())
            history.append(entry)
            print(json.dumps(entry, indent=2), flush=True)
            running.clear()
            running_batches = 0
            current_eal = float(validation["current_eal_prompt_balanced"])
            if current_eal > best_eal:
                best_eal = current_eal
                best_step = global_step
                best_validation = validation
                _save(
                    args.output / "best_candidate.pt",
                    head=head,
                    args=args,
                    step=global_step,
                    validation=validation,
                )
            head.train()

    _save(
        args.output / "last_candidate.pt",
        head=head,
        args=args,
        step=global_step,
        validation=last_validation,
    )
    report = {
        "status": "capacity_only" if overlap else "completed",
        "train_rollout": str(args.train_rollout.resolve()),
        "eval_rollout": str(args.eval_rollout.resolve()),
        "train_prompts": len(train_grouped),
        "train_blocks": sum(len(group) for group in train_grouped.values()),
        "eval_prompts": len(eval_grouped),
        "eval_blocks": len(eval_records),
        "overlapping_prompts": len(overlap),
        "candidate_topk": args.candidate_topk,
        "include_released_action": args.include_released_action,
        "released_union_teacher": args.released_union_teacher,
        "trainable_scope": args.trainable_scope,
        "adapter_rank": head.adapter_rank,
        "use_target_boundary_feature": args.use_target_boundary_feature,
        "use_target_anchor_early_feature": args.use_target_anchor_early_feature,
        "target_anchor_early_exit_layer": args.target_anchor_early_exit_layer,
        "target_context_field": args.target_context_field,
        "required_alignment_report": str(args.required_alignment_report.resolve())
        if args.required_alignment_report
        else None,
        "boundary_width": head.boundary_width,
        "trainable_parameters": trainable_parameter_count,
        "frontier_weight": args.frontier_weight,
        "dense_dpace_weight": args.dense_dpace_weight,
        "dpace_alpha": args.dpace_alpha,
        "dense_margin_weight": args.dense_margin_weight,
        "dense_margin": args.dense_margin,
        "target_kl_weight": args.target_kl_weight,
        "target_advantage_weight": args.target_advantage_weight,
        "target_temperature": args.target_temperature,
        "target_huber_delta": args.target_huber_delta,
        "target_protect_weight": args.target_protect_weight,
        "target_repair_weight": args.target_repair_weight,
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
        "checkpoint": str((args.output / "best_candidate.pt").resolve()),
        "last_checkpoint": str((args.output / "last_candidate.pt").resolve()),
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
    if args.use_target_anchor_early_feature:
        early_entries = [
            entry
            for entry in history
            if 0 < int(entry["step"]) <= 200
        ]
        early_values = [
            float(entry["validation"]["current_eal_prompt_balanced"])
            for entry in early_entries
        ]
        report["r047_gates"] = {
            "step_200_observed": global_step >= 200,
            "best_eal_through_step_200": max(early_values)
            if early_values
            else None,
            "step_200_eal_at_least_7_50": (
                max(early_values) >= 7.50
                if global_step >= 200 and early_values
                else None
            ),
            "continue_eal_at_least_7_80": best_eal >= 7.80,
            "bootstrap_lower_above_zero": float(
                best_validation["paired_bootstrap_95_interval"][0]
            )
            > 0,
            "lost_to_gained_at_most_half": float(
                best_validation["lost_to_gained_ratio"]
            )
            <= 0.5,
            "hard_fixed_b16_eal_at_least_8_325485909": (
                best_eal >= 8.325485908649174
            ),
            "exact_409600_trainable_parameters": (
                trainable_parameter_count == 409_600
            ),
        }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
