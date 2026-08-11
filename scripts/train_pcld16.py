#!/usr/bin/env python3
"""Train and evaluate the frozen PCLD-16R full-block one-chain head."""

from __future__ import annotations

import argparse
from collections import defaultdict
import inspect
import json
import math
from pathlib import Path
import random
from typing import Any

from safetensors import safe_open
import torch
from torch import Tensor
from torch.nn import functional as F

from sph.japd import accepted_lengths, strict_joint_two_frontier_metric
from sph.japd_data import load_rollout_records, record_key
from sph.pcld import (
    BLOCK_LENGTH,
    CANDIDATES,
    EXPECTED_PARAMETER_COUNT,
    PCLD16Head,
    PCLDOutput,
    assert_frozen_architecture,
    latent_alpha,
    pcld_per_block_loss,
)
from sph.pcld_data import (
    attach_pcld_sidecar,
    calibrate_epsilon_from_records,
    capacity_expected_j2_denominators,
    collate_pcld_records,
    compute_latent_scale,
    filter_effective_records,
    group_record_indices_by_prompt,
    load_manifest,
    load_pcld_sidecar,
    pcld_forward_inputs,
    sample_prompt_balanced_records,
    select_manifest_group,
    validate_sidecar_receipt,
    validate_sidecar_source,
    validate_capacity_support_receipt,
    validate_manifest_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rollout", type=Path, required=True)
    parser.add_argument("--eval-rollout", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--eval-sidecar", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capacity-support-receipt", type=Path)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="train")
    parser.add_argument(
        "--train-group", choices=("capacity", "fit", "select", "diagnostic"), required=True
    )
    parser.add_argument(
        "--eval-group", choices=("capacity", "fit", "select", "diagnostic"), required=True
    )
    parser.add_argument("--scope", choices=("global", "local"), default="global")
    parser.add_argument("--no-latent", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--eval-every-steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate", choices=("none", "capacity"), default="none")
    parser.add_argument("--require-gate", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--require-parameter-count", type=int, default=EXPECTED_PARAMETER_COUNT
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_target_lm_head_weight(target: Path) -> tuple[Tensor, str]:
    """Resolve the serialized tensor backing authoritative target.lm_head.weight."""

    config = json.loads((target / "config.json").read_text())
    index = json.loads((target / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    if "lm_head.weight" in weight_map:
        key = "lm_head.weight"
    elif bool(config.get("tie_word_embeddings", False)):
        key = "model.embed_tokens.weight"
    else:
        raise RuntimeError("checkpoint cannot resolve target.lm_head.weight")
    shard = target / str(weight_map[key])
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key)
    if weight.ndim != 2 or weight.shape[1] != 2560:
        raise RuntimeError(f"unexpected target LM-head shape {tuple(weight.shape)}")
    return weight, key


def validate_train_eval_groups(
    train_group: str,
    eval_group: str,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> None:
    if train_group == "capacity" or eval_group == "capacity":
        if train_group != "capacity" or eval_group != "capacity":
            raise RuntimeError("P1 capacity must use capacity for both train and eval")
        train_keys = {record_key(record) for record in train_records}
        eval_keys = {record_key(record) for record in eval_records}
        if len(train_records) != 512 or len(eval_records) != 512:
            raise RuntimeError("P1 capacity requires exactly 512 raw records")
        if train_keys != eval_keys or len(train_keys) != 512:
            raise RuntimeError("P1 capacity train/eval semantic sets differ")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def prompt_mean(values: list[float], sample_ids: list[str]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, sample_id in zip(values, sample_ids, strict=True):
        grouped[sample_id].append(float(value))
    if not grouped:
        raise ValueError("prompt-balanced mean has no records")
    return sum(sum(rows) / len(rows) for rows in grouped.values()) / len(grouped)


def prompt_ratio(
    successes: list[int], eligible: list[int], sample_ids: list[str]
) -> tuple[float, int, int, int]:
    grouped_success: dict[str, int] = defaultdict(int)
    grouped_eligible: dict[str, int] = defaultdict(int)
    for success, keep, sample_id in zip(successes, eligible, sample_ids, strict=True):
        grouped_success[sample_id] += int(success)
        grouped_eligible[sample_id] += int(keep)
    ratios = [
        grouped_success[sample_id] / count
        for sample_id, count in grouped_eligible.items()
        if count
    ]
    return (
        sum(ratios) / len(ratios) if ratios else 0.0,
        sum(grouped_success.values()),
        sum(grouped_eligible.values()),
        len(ratios),
    )


def pcld_block_diagnostics(
    output: PCLDOutput,
    batch: dict[str, Any],
    support: Tensor,
    latent_scale: Tensor,
) -> dict[str, Tensor]:
    """Frozen support-aware diagnostics, reduced within each block first."""

    if support.shape != output.scores.shape[:2]:
        raise ValueError("PCLD diagnostic support shape mismatch")
    support_float = support.float()
    support_count = support_float.sum(dim=-1)
    normalizer = support_count.clamp_min(1.0)

    residual_error = (
        output.predicted_residual.float() - batch["target_residual"].float()
    )
    raw_hidden_rmse = (
        (residual_error.square().mean(dim=-1) * support_float).sum(dim=-1)
        / normalizer
    ).sqrt()
    whitened_error = residual_error / latent_scale.float().view(1, 1, -1)
    whitened_hidden_rmse = (
        (whitened_error.square().mean(dim=-1) * support_float).sum(dim=-1)
        / normalizer
    ).sqrt()
    target_correction = (
        batch["target_candidate_logits"].float() - output.base_scores.float()
    )
    correction_error = output.corrections.float() - target_correction
    candidate_correction_rmse = (
        (correction_error.square().mean(dim=-1) * support_float).sum(dim=-1)
        / normalizer
    ).sqrt()

    gold_ranks = batch["gold_candidate_ranks"].clamp(0, CANDIDATES - 1)
    gold_slots = F.one_hot(gold_ranks, num_classes=CANDIDATES).bool()
    student_scores = output.scores.float()
    teacher_scores = batch["target_candidate_logits"].float()
    student_gold = student_scores.gather(
        -1, gold_ranks.unsqueeze(-1)
    ).squeeze(-1)
    teacher_gold = teacher_scores.gather(
        -1, gold_ranks.unsqueeze(-1)
    ).squeeze(-1)
    student_margin = student_gold - student_scores.masked_fill(
        gold_slots, -torch.inf
    ).amax(dim=-1)
    teacher_margin = teacher_gold - teacher_scores.masked_fill(
        gold_slots, -torch.inf
    ).amax(dim=-1)
    teacher_margin_sign_agreement = (
        torch.sign(student_margin).eq(torch.sign(teacher_margin)).float()
        * support_float
    ).sum(dim=-1) / normalizer
    teacher_margin_rmse = (
        ((student_margin - teacher_margin).square() * support_float).sum(dim=-1)
        / normalizer
    ).sqrt()
    residual_cosine = (
        F.cosine_similarity(
            output.predicted_residual.float(),
            batch["target_residual"].float(),
            dim=-1,
        )
        * support_float
    ).sum(dim=-1) / normalizer
    teacher_candidate_ranks = teacher_scores.argmax(dim=-1)
    student_candidate_ranks = student_scores.argmax(dim=-1)
    teacher_candidate_agreement = (
        student_candidate_ranks.eq(teacher_candidate_ranks).float()
        * support_float
    ).sum(dim=-1) / normalizer
    return {
        "eligible": support_count.gt(0),
        "raw_hidden_rmse": raw_hidden_rmse,
        "whitened_hidden_rmse": whitened_hidden_rmse,
        "candidate_correction_rmse": candidate_correction_rmse,
        "teacher_margin_sign_agreement": teacher_margin_sign_agreement,
        "teacher_margin_rmse": teacher_margin_rmse,
        "residual_cosine": residual_cosine,
        "teacher_candidate_agreement": teacher_candidate_agreement,
    }


def cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    minimum: float,
) -> float:
    if warmup_steps and step <= warmup_steps:
        return peak * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def selection_evaluation_step(
    step: int, *, total_steps: int, eval_every_steps: int
) -> bool:
    if step < 1 or total_steps < 1 or eval_every_steps < 1:
        raise ValueError("invalid checkpoint selection cadence")
    return step % eval_every_steps == 0 or step == total_steps


@torch.inference_mode()
def evaluate(
    model: PCLD16Head,
    records: list[dict[str, Any]],
    lm_head_weight: Tensor,
    latent_scale: Tensor,
    epsilon_num: float,
    device: torch.device,
    *,
    batch_size: int,
    no_latent: bool,
    require_identity: bool,
) -> dict[str, Any]:
    model.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    model_lengths: list[int] = []
    base_lengths: list[int] = []
    domino_lengths: list[int] = []
    oracle_lengths: list[int] = []
    teacher_lengths: list[int] = []
    losses: dict[str, list[float]] = defaultdict(list)
    diagnostic_values: dict[str, list[float]] = defaultdict(list)
    diagnostic_sample_ids: list[str] = []
    support_correct = 0
    support_total = 0
    hard_correct = 0
    hard_total = 0
    j2_success: dict[str, list[int]] = {
        name: []
        for name in ("legacy", "authoritative", "authoritative_numeric", "stable")
    }
    j2_eligible: dict[str, list[int]] = {
        name: [] for name in j2_success
    }

    for start in range(0, len(records), batch_size):
        cpu_batch = collate_pcld_records(
            records[start : start + batch_size],
            epsilon_num=epsilon_num,
            require_effective=False,
        )
        batch = move_batch(cpu_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(**pcld_forward_inputs(batch, lm_head_weight))
        if require_identity:
            if not torch.equal(output.scores, batch["candidate_logits"].float()):
                raise RuntimeError("zero-init PCLD scores do not exactly equal base scores")
            if not torch.equal(
                output.scores.argmax(dim=-1),
                torch.zeros_like(batch["gold_ids"], dtype=torch.long),
            ):
                raise RuntimeError("zero-init PCLD selected ranks differ from base Top1")
        loss_output = pcld_per_block_loss(
            output,
            batch["candidate_ids"],
            batch["gold_ids"],
            batch["target_residual"],
            batch["target_candidate_logits"],
            batch["target_top1_ids"],
            batch["stable_rows"],
            latent_scale,
            alpha=0.0 if no_latent else 0.1,
        )
        model_ranks = output.scores.argmax(dim=-1)
        model_ids = batch["candidate_ids"].gather(
            -1, model_ranks.unsqueeze(-1)
        ).squeeze(-1)
        base_ids = batch["candidate_ids"][..., 0]
        oracle_ranks = torch.where(
            batch["gold_candidate_ranks"].ge(0),
            batch["gold_candidate_ranks"],
            torch.zeros_like(batch["gold_candidate_ranks"]),
        )
        oracle_ids = batch["candidate_ids"].gather(
            -1, oracle_ranks.unsqueeze(-1)
        ).squeeze(-1)
        teacher_ranks = batch["target_candidate_logits"].argmax(dim=-1)
        teacher_ids = batch["candidate_ids"].gather(
            -1, teacher_ranks.unsqueeze(-1)
        ).squeeze(-1)
        current_lengths = {
            "model": accepted_lengths(model_ids, batch["gold_ids"]),
            "base": accepted_lengths(base_ids, batch["gold_ids"]),
            "domino": accepted_lengths(batch["policy_ids"], batch["gold_ids"]),
            "oracle": accepted_lengths(oracle_ids, batch["gold_ids"]),
            "teacher": accepted_lengths(teacher_ids, batch["gold_ids"]),
        }
        support = loss_output.support_mask
        block_diagnostics = pcld_block_diagnostics(
            output, batch, support, latent_scale
        )
        diagnostic_indices = block_diagnostics["eligible"].nonzero(
            as_tuple=False
        ).flatten().cpu().tolist()
        diagnostic_sample_ids.extend(
            [batch["sample_ids"][index] for index in diagnostic_indices]
        )
        for name, values in block_diagnostics.items():
            if name == "eligible":
                continue
            values_cpu = values.detach().cpu()
            diagnostic_values[name].extend(
                [float(values_cpu[index]) for index in diagnostic_indices]
            )
        correct = model_ranks.eq(batch["gold_candidate_ranks"])
        hard = support & batch["gold_candidate_ranks"].ne(0)
        support_correct += int((correct & support).sum().item())
        support_total += int(support.sum().item())
        hard_correct += int((correct & hard).sum().item())
        hard_total += int(hard.sum().item())
        joint_metrics = {
            name: strict_joint_two_frontier_metric(
                model_ranks,
                batch["gold_candidate_ranks"],
                batch[f"{name}_j2_target_matches_gold"],
            )
            for name in j2_success
        }

        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        model_lengths.extend(current_lengths["model"].cpu().tolist())
        base_lengths.extend(current_lengths["base"].cpu().tolist())
        domino_lengths.extend(current_lengths["domino"].cpu().tolist())
        oracle_lengths.extend(current_lengths["oracle"].cpu().tolist())
        teacher_lengths.extend(current_lengths["teacher"].cpu().tolist())
        for name, joint in joint_metrics.items():
            j2_success[name].extend(joint.success.cpu().to(torch.int64).tolist())
            j2_eligible[name].extend(joint.eligible.cpu().to(torch.int64).tolist())
        losses["total"].extend(loss_output.per_block_loss.cpu().tolist())
        losses["safe"].extend(loss_output.safe_loss.cpu().tolist())
        losses["latent"].extend(loss_output.latent_loss.cpu().tolist())
        losses["candidate_kl"].extend(loss_output.candidate_kl.cpu().tolist())

    metrics: dict[str, Any] = {
        "records": len(sample_ids),
        "prompts": len(set(sample_ids)),
        "model_eal": prompt_mean(model_lengths, sample_ids),
        "base_eal": prompt_mean(base_lengths, sample_ids),
        "domino_eal": prompt_mean(domino_lengths, sample_ids),
        "oracle_eal": prompt_mean(oracle_lengths, sample_ids),
        "teacher_eal": prompt_mean(teacher_lengths, sample_ids),
        "candidate_accuracy_micro": support_correct / max(1, support_total),
        "hard_candidate_accuracy": hard_correct / max(1, hard_total),
        "support_positions": support_total,
        "hard_positions": hard_total,
        "harmed_fraction": sum(
            int(model < base)
            for model, base in zip(model_lengths, base_lengths, strict=True)
        )
        / len(model_lengths),
    }
    for name, values in losses.items():
        metrics[f"{name}_loss_prompt_balanced"] = prompt_mean(values, sample_ids)
    if not diagnostic_sample_ids:
        raise RuntimeError("PCLD evaluation has no supported diagnostic blocks")
    metrics["diagnostic_records"] = len(diagnostic_sample_ids)
    metrics["diagnostic_prompts"] = len(set(diagnostic_sample_ids))
    for name, values in diagnostic_values.items():
        metrics[f"{name}_prompt_balanced"] = prompt_mean(
            values, diagnostic_sample_ids
        )
    metrics["candidate_accuracy"] = metrics[
        "teacher_candidate_agreement_prompt_balanced"
    ]
    metrics["candidate_accuracy_definition"] = (
        "authoritative_teacher_candidate_argmax; support mean within block; "
        "block mean within prompt; prompt mean"
    )
    gap = metrics["oracle_eal"] - metrics["base_eal"]
    metrics["oracle_gap_recovered"] = (
        (metrics["model_eal"] - metrics["base_eal"]) / gap if gap > 0 else 0.0
    )
    for name in j2_success:
        j2_mean, j2_num, j2_den, j2_prompts = prompt_ratio(
            j2_success[name], j2_eligible[name], sample_ids
        )
        metrics.update(
            {
                f"{name}_j2_prompt_balanced": j2_mean,
                f"{name}_j2_numerator": j2_num,
                f"{name}_j2_denominator": j2_den,
                f"{name}_j2_prompts": j2_prompts,
            }
        )
    metrics["binding_j2_definition"] = "legacy"
    by_domain: dict[str, Any] = {}
    for domain in sorted(set(domains)):
        indices = [index for index, value in enumerate(domains) if value == domain]
        ids = [sample_ids[index] for index in indices]
        by_domain[domain] = {
            "model_eal": prompt_mean([model_lengths[index] for index in indices], ids),
            "base_eal": prompt_mean([base_lengths[index] for index in indices], ids),
            "domino_eal": prompt_mean(
                [domino_lengths[index] for index in indices], ids
            ),
        }
    metrics["by_domain"] = by_domain
    return metrics


def gate_checks(
    metrics: dict[str, Any],
    gate: str,
    *,
    expected_j2_denominator: int | None = None,
) -> dict[str, bool]:
    if gate == "capacity":
        if expected_j2_denominator != 411:
            raise RuntimeError("capacity gate requires frozen J2 denominator 411")
        return {
            "candidate_agreement_at_least_99pct": metrics["candidate_accuracy"]
            >= 0.99,
            "oracle_gap_at_least_95pct": metrics["oracle_gap_recovered"] >= 0.95,
            "harm_at_most_1pct": metrics["harmed_fraction"] <= 0.01,
            "legacy_j2_at_least_99pct": metrics["legacy_j2_prompt_balanced"]
            >= 0.99,
            "legacy_j2_denominator_exactly_411": metrics[
                "legacy_j2_denominator"
            ]
            == expected_j2_denominator,
        }
    return {}


def serialized_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def validate_pcld_checkpoint(
    checkpoint: dict[str, Any],
    *,
    args: argparse.Namespace,
    expected_step: int,
    expected_metrics: dict[str, Any],
    epsilon_num: float,
    latent_scale: Tensor,
) -> dict[str, Tensor]:
    """Fail closed before reloading the internally selected best checkpoint."""

    if checkpoint.get("format") != "pcld16_checkpoint_v1":
        raise RuntimeError("unsupported PCLD checkpoint format")
    if int(checkpoint.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("PCLD checkpoint parameter count mismatch")
    if int(checkpoint.get("step", -1)) != expected_step:
        raise RuntimeError("PCLD checkpoint step differs from selected best step")
    if checkpoint.get("config") != serialized_config(args):
        raise RuntimeError("PCLD checkpoint config/source contract mismatch")
    if checkpoint.get("metrics") != expected_metrics:
        raise RuntimeError("PCLD checkpoint metrics differ from selected metrics")
    stored_epsilon = checkpoint.get("epsilon_num")
    if (
        not isinstance(stored_epsilon, (int, float))
        or not math.isfinite(float(stored_epsilon))
        or float(stored_epsilon) != float(epsilon_num)
    ):
        raise RuntimeError("PCLD checkpoint numerical epsilon mismatch")
    stored_scale = checkpoint.get("latent_scale")
    if not isinstance(stored_scale, Tensor) or not torch.equal(
        stored_scale.cpu(), latent_scale.detach().cpu()
    ):
        raise RuntimeError("PCLD checkpoint latent scale mismatch")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("PCLD checkpoint lacks a model state")
    return state


def save_checkpoint(
    path: Path,
    model: PCLD16Head,
    *,
    step: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    epsilon_num: float,
    latent_scale: Tensor,
) -> None:
    torch.save(
        {
            "format": "pcld16_checkpoint_v1",
            "model": model.state_dict(),
            "step": step,
            "metrics": metrics,
            "epsilon_num": epsilon_num,
            "latent_scale": latent_scale.cpu(),
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "config": serialized_config(args),
        },
        path,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse PCLD output {args.output}")
    if min(args.batch_size, args.eval_batch_size, args.max_steps) < 1:
        raise ValueError("PCLD batch sizes and max steps must be positive")
    if args.warmup_steps >= args.max_steps:
        raise ValueError("PCLD warmup must end before max steps")
    if args.eval_every_steps < 1 or args.gradient_clip <= 0:
        raise ValueError("invalid PCLD cadence or gradient clip")
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("PCLD training requires CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    manifest = load_manifest(args.manifest)
    validate_manifest_source(
        manifest, rollout=args.train_rollout, split=args.train_split
    )
    validate_manifest_source(
        manifest, rollout=args.eval_rollout, split=args.eval_split
    )
    expected_capacity_j2 = (
        capacity_expected_j2_denominators(manifest)
        if args.gate == "capacity"
        else None
    )
    if args.gate == "capacity" and args.capacity_support_receipt is None:
        raise RuntimeError("PCLD capacity gate requires a frozen support receipt")
    if args.gate != "capacity" and args.capacity_support_receipt is not None:
        raise RuntimeError("PCLD support receipt is capacity-only")
    train_source, raw_train = load_rollout_records(
        args.train_rollout, split=args.train_split
    )
    eval_source, raw_eval = load_rollout_records(args.eval_rollout, split=args.eval_split)
    train_records = select_manifest_group(raw_train, manifest, args.train_group)
    eval_records = select_manifest_group(raw_eval, manifest, args.eval_group)
    validate_train_eval_groups(
        args.train_group, args.eval_group, train_records, eval_records
    )

    train_sidecar_metadata, train_sidecar = load_pcld_sidecar(args.train_sidecar)
    eval_sidecar_metadata, eval_sidecar = load_pcld_sidecar(args.eval_sidecar)
    validate_sidecar_source(
        train_sidecar_metadata,
        rollout=args.train_rollout,
        target=args.target,
        split=args.train_split,
        group=args.train_group,
    )
    validate_sidecar_source(
        eval_sidecar_metadata,
        rollout=args.eval_rollout,
        target=args.target,
        split=args.eval_split,
        group=args.eval_group,
    )
    train_receipt = validate_sidecar_receipt(args.train_sidecar, train_sidecar_metadata)
    eval_receipt = validate_sidecar_receipt(args.eval_sidecar, eval_sidecar_metadata)
    train_records = attach_pcld_sidecar(
        train_records, train_sidecar, require_exact_keys=True
    )
    eval_records = attach_pcld_sidecar(
        eval_records, eval_sidecar, require_exact_keys=True
    )

    epsilon_num = calibrate_epsilon_from_records(train_records)
    capacity_support_receipt = None
    if expected_capacity_j2 is not None:
        capacity_support_receipt = validate_capacity_support_receipt(
            args.capacity_support_receipt,
            train_records,
            epsilon_num,
            rollout=args.train_rollout,
            manifest=args.manifest,
            target=args.target,
            sidecar=args.train_sidecar,
            split=args.train_split,
            group=args.train_group,
            replay_report=train_receipt,
        )
    raw_train_count = len(train_records)
    train_records = filter_effective_records(train_records, epsilon_num)
    if args.train_group == "capacity":
        eval_effective_keys = {
            record_key(record)
            for record in filter_effective_records(eval_records, epsilon_num)
        }
        if {record_key(record) for record in train_records} != eval_effective_keys:
            raise RuntimeError("P1 train/eval effective record sets differ")
    latent_scale_cpu, latent_scale_rows = compute_latent_scale(
        train_records, epsilon_num
    )
    latent_scale = latent_scale_cpu.to(device)
    grouped_indices = group_record_indices_by_prompt(train_records)
    sampler_rng = random.Random(args.seed)

    lm_head_weight_cpu, serialized_lm_head_key = load_target_lm_head_weight(args.target)
    lm_head_weight = lm_head_weight_cpu.to(
        device=device, dtype=torch.bfloat16
    ).detach()
    model = PCLD16Head(scope=args.scope).to(device)
    assert_frozen_architecture(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != args.require_parameter_count:
        raise RuntimeError(
            f"PCLD parameter count {parameter_count} != {args.require_parameter_count}"
        )
    forbidden_forward_names = {"gold", "target_hidden", "teacher", "selected_tokens"}
    forward_names = set(inspect.signature(model.forward).parameters)
    if any(any(token in name for token in forbidden_forward_names) for name in forward_names):
        raise RuntimeError("offline target field entered PCLD production forward")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output.mkdir(parents=True, exist_ok=False)
    initial = evaluate(
        model,
        eval_records,
        lm_head_weight,
        latent_scale,
        epsilon_num,
        device,
        batch_size=args.eval_batch_size,
        no_latent=args.no_latent,
        require_identity=True,
    )
    if expected_capacity_j2 is not None:
        for name, expected in expected_capacity_j2.items():
            actual = initial[f"{name}_j2_denominator"]
            if actual != expected:
                raise RuntimeError(
                    f"PCLD capacity step0 {name} J2 denominator "
                    f"{actual} != frozen {expected}"
                )
    best_metrics = initial
    best_step = 0
    save_checkpoint(
        args.output / "best.pt",
        model,
        step=0,
        metrics=initial,
        args=args,
        epsilon_num=epsilon_num,
        latent_scale=latent_scale_cpu,
    )
    history: list[dict[str, Any]] = [{"step": 0, "eval": initial}]
    print(json.dumps(history[-1], ensure_ascii=False), flush=True)

    for step in range(1, args.max_steps + 1):
        model.train()
        sampled = sample_prompt_balanced_records(
            train_records,
            grouped_indices,
            batch_size=args.batch_size,
            rng=sampler_rng,
        )
        batch = move_batch(
            collate_pcld_records(
                sampled, epsilon_num=epsilon_num, require_effective=True
            ),
            device,
        )
        learning_rate = cosine_learning_rate(
            step,
            total_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            peak=args.learning_rate,
            minimum=args.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        alpha = 0.0 if args.no_latent else latent_alpha(step, args.max_steps)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(**pcld_forward_inputs(batch, lm_head_weight))
            loss_output = pcld_per_block_loss(
                output,
                batch["candidate_ids"],
                batch["gold_ids"],
                batch["target_residual"],
                batch["target_candidate_logits"],
                batch["target_top1_ids"],
                batch["stable_rows"],
                latent_scale,
                alpha=alpha,
            )
            # Sampling is exactly uniform-prompt then uniform-effective-block.
            loss = loss_output.per_block_loss.mean()
        if not bool(torch.isfinite(loss).detach().cpu().item()):
            raise FloatingPointError(f"non-finite PCLD loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm).detach().cpu().item()):
            raise FloatingPointError(f"non-finite PCLD gradient at step {step}")
        optimizer.step()

        if selection_evaluation_step(
            step,
            total_steps=args.max_steps,
            eval_every_steps=args.eval_every_steps,
        ):
            metrics = evaluate(
                model,
                eval_records,
                lm_head_weight,
                latent_scale,
                epsilon_num,
                device,
                batch_size=args.eval_batch_size,
                no_latent=args.no_latent,
                require_identity=False,
            )
            record = {
                "step": step,
                "learning_rate": learning_rate,
                "latent_alpha": alpha,
                "train_loss": float(loss.detach()),
                "train_safe": float(loss_output.safe_loss.mean().detach()),
                "train_latent": float(loss_output.latent_loss.mean().detach()),
                "train_candidate_kl": float(
                    loss_output.candidate_kl.mean().detach()
                ),
                "gradient_norm": float(gradient_norm.detach()),
                "eval": metrics,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if metrics["model_eal"] > best_metrics["model_eal"]:
                best_metrics = metrics
                best_step = step
                save_checkpoint(
                    args.output / "best.pt",
                    model,
                    step=step,
                    metrics=metrics,
                    args=args,
                    epsilon_num=epsilon_num,
                    latent_scale=latent_scale_cpu,
                )

    checkpoint = torch.load(
        args.output / "best.pt", map_location=device, weights_only=False
    )
    checkpoint_state = validate_pcld_checkpoint(
        checkpoint,
        args=args,
        expected_step=best_step,
        expected_metrics=best_metrics,
        epsilon_num=epsilon_num,
        latent_scale=latent_scale_cpu,
    )
    model.load_state_dict(checkpoint_state, strict=True)
    selected = evaluate(
        model,
        eval_records,
        lm_head_weight,
        latent_scale,
        epsilon_num,
        device,
        batch_size=args.eval_batch_size,
        no_latent=args.no_latent,
        require_identity=False,
    )
    if expected_capacity_j2 is not None:
        selected["expected_j2_denominators"] = expected_capacity_j2
    checks = gate_checks(
        selected,
        args.gate,
        expected_j2_denominator=(
            expected_capacity_j2["legacy"]
            if expected_capacity_j2 is not None
            else None
        ),
    )
    passed = all(checks.values()) if checks else True
    report = {
        "format": "pcld16_training_v1",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": {
            "type": str(device),
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "architecture": {
            "scope": args.scope,
            "positions": BLOCK_LENGTH,
            "candidates": CANDIDATES,
            "model_dim": 256,
            "heads": 8,
            "layers": 2,
            "ffn_dim": 1024,
            "dropout": 0.0,
            "one_chain": True,
            "selected_token_feedback": False,
        },
        "parameter_count": parameter_count,
        "serialized_lm_head_key": serialized_lm_head_key,
        "epsilon_num": epsilon_num,
        "latent_scale": {
            "rows": latent_scale_rows,
            "min": float(latent_scale_cpu.min()),
            "median": float(latent_scale_cpu.median()),
            "max": float(latent_scale_cpu.max()),
        },
        "train_source": train_source,
        "eval_source": eval_source,
        "train_sidecar_receipt": train_receipt,
        "eval_sidecar_receipt": eval_receipt,
        "capacity_support_receipt": capacity_support_receipt,
        "train_records_raw": raw_train_count,
        "train_records_effective": len(train_records),
        "train_prompts_effective": len(grouped_indices),
        "eval_records": len(eval_records),
        "best_step": best_step,
        "selected": selected,
        "gate": args.gate,
        "gate_checks": checks,
        "gate_passed": passed,
        "history": history,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "best_step": best_step,
                "selected": selected,
                "gate_checks": checks,
                "gate_passed": passed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.require_gate and not passed:
        raise RuntimeError(f"PCLD {args.gate} gate failed: {checks}")
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
