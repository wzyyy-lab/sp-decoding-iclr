#!/usr/bin/env python3
"""Train/evaluate the frozen JAPD-16 full-block one-chain selector."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import partial
import json
import math
from pathlib import Path
import random
from typing import Any

from safetensors import safe_open
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from sph.global_direct_selector import GlobalDirectCandidateSelector
from sph.japd import (
    BLOCK_LENGTH,
    CANDIDATES,
    accepted_lengths,
    clean_support,
    fixed_prompt_balanced_batch_loss,
    japd_per_block_loss,
    matched_candidate_dpace_per_block_loss,
    strict_joint_two_frontier_metric,
)
from sph.japd_data import (
    attach_lse_sidecar,
    collate_japd_records,
    effective_blocks_per_prompt,
    effective_record_mask,
    load_lse_sidecar,
    load_rollout_records,
    record_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rollout", type=Path, required=True)
    parser.add_argument("--eval-rollout", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--eval-sidecar", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="train")
    parser.add_argument(
        "--train-group",
        choices=("fit", "select", "diagnostic", "capacity", "full_fit_diagnostic", "all"),
        required=True,
    )
    parser.add_argument(
        "--eval-group",
        choices=("fit", "select", "diagnostic", "capacity", "full_fit_diagnostic", "all"),
        required=True,
    )
    parser.add_argument("--scope", choices=("global", "local"), required=True)
    parser.add_argument("--objective", choices=("japd", "candidate_dpace"), required=True)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-every-steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate", choices=("none", "capacity", "full_fit"), default="none")
    parser.add_argument("--require-gate", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-parameter-count", type=int, default=0)
    return parser.parse_args()


class RecordDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("JAPD dataset is empty")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_target_embedding(target: Path) -> Tensor:
    index = json.loads((target / "model.safetensors.index.json").read_text())
    key = "model.embed_tokens.weight"
    shard = target / str(index["weight_map"][key])
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key)
    if weight.ndim != 2 or weight.shape[1] != 2560:
        raise RuntimeError(f"unexpected target embedding shape {tuple(weight.shape)}")
    return weight


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "japd_manifest_v1" or not manifest.get("complete"):
        raise RuntimeError(f"invalid JAPD manifest: {path}")
    if manifest.get("label_fields_used_for_selection") != []:
        raise RuntimeError("JAPD manifest used labels for selection")
    return manifest


def capacity_manifest_keys(
    manifest: dict[str, Any],
) -> set[tuple[str, int, int]]:
    items = manifest.get("capacity", {}).get("records")
    if not isinstance(items, list) or len(items) != 512:
        raise RuntimeError("JAPD capacity manifest must list exactly 512 records")
    keys = [
        (
            str(item["sample_id"]),
            int(item["anchor_offset"]),
            int(item["context_length"]),
        )
        for item in items
    ]
    if len(set(keys)) != 512:
        raise RuntimeError("JAPD capacity manifest record keys must be unique")
    if len({key[0] for key in keys}) != 512:
        raise RuntimeError("JAPD capacity manifest must use one record per prompt")
    return set(keys)


def full_fit_manifest_prompts(manifest: dict[str, Any]) -> set[str]:
    values = manifest.get("full_fit_diagnostic", {}).get("prompts")
    if not isinstance(values, list) or len(values) != 512:
        raise RuntimeError("JAPD full-fit manifest must list exactly 512 prompts")
    prompts = {str(value) for value in values}
    if len(prompts) != 512:
        raise RuntimeError("JAPD full-fit manifest prompts must be unique")
    capacity_prompts = {key[0] for key in capacity_manifest_keys(manifest)}
    overlap = prompts.intersection(capacity_prompts)
    if overlap:
        raise RuntimeError(
            f"JAPD capacity/full-fit manifests overlap on {len(overlap)} prompts"
        )
    return prompts


def validate_sidecar_source(
    metadata: dict[str, Any],
    *,
    rollout: Path,
    target: Path,
    split: str,
) -> None:
    """Bind a derived LSE sidecar to its semantic source, without hashes."""

    expected_rollout = str(rollout.resolve())
    expected_target = str(target.resolve())
    if metadata.get("source_rollout") != expected_rollout:
        raise RuntimeError(
            "JAPD sidecar source rollout mismatch: "
            f"{metadata.get('source_rollout')!r} != {expected_rollout!r}"
        )
    if metadata.get("target") != expected_target:
        raise RuntimeError(
            "JAPD sidecar target mismatch: "
            f"{metadata.get('target')!r} != {expected_target!r}"
        )
    if metadata.get("split") != split:
        raise RuntimeError(
            "JAPD sidecar split mismatch: "
            f"{metadata.get('split')!r} != {split!r}"
        )


def validate_sidecar_replay_receipt(
    root: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Require the semantic GPU replay receipt before training consumes LSE."""

    receipt_path = root / "replay_report.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"JAPD sidecar lacks replay receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != "japd_base_lse_replay_v1":
        raise RuntimeError("unsupported JAPD sidecar replay receipt")
    required_true = (
        "verified",
        "top16_ids_exact",
        "stored_dtype_top16_logits_exact",
        "five_scalar_channels_allclose",
        "audit_head_scores_allclose",
        "selected_tokens_exact",
    )
    failed = [name for name in required_true if receipt.get(name) is not True]
    if failed:
        raise RuntimeError(f"JAPD sidecar replay receipt failed: {failed}")
    if int(receipt.get("records", -1)) != int(metadata.get("records", -2)):
        raise RuntimeError("JAPD sidecar replay record count mismatch")
    if receipt.get("source_rollout") != metadata.get("source_rollout"):
        raise RuntimeError("JAPD sidecar replay source mismatch")
    if receipt.get("split") != metadata.get("split"):
        raise RuntimeError("JAPD sidecar replay split mismatch")
    if receipt.get("sidecar") != str(root.resolve()):
        raise RuntimeError("JAPD sidecar replay path mismatch")
    if int(receipt.get("selected_token_mismatches", -1)) != 0:
        raise RuntimeError("JAPD sidecar replay selected-token mismatch")
    return receipt


def select_group(
    records: list[dict[str, Any]], manifest: dict[str, Any], group: str
) -> list[dict[str, Any]]:
    if group == "all":
        return list(records)
    if group in {"fit", "select", "diagnostic"}:
        prompts = set(manifest["prompt_splits"][group])
        selected = [
            record for record in records if str(record["sample_id"]) in prompts
        ]
    elif group == "full_fit_diagnostic":
        prompts = full_fit_manifest_prompts(manifest)
        selected = [
            record for record in records if str(record["sample_id"]) in prompts
        ]
        selected_prompts = {str(record["sample_id"]) for record in selected}
        if selected_prompts != prompts:
            missing = sorted(prompts.difference(selected_prompts))
            raise RuntimeError(
                f"full-fit manifest does not align with rollout: missing {missing[:3]}"
            )
    elif group == "capacity":
        keys = capacity_manifest_keys(manifest)
        selected = [record for record in records if record_key(record) in keys]
        if {record_key(record) for record in selected} != keys:
            raise RuntimeError("capacity manifest does not align with rollout")
    else:
        raise AssertionError(group)
    if not selected:
        raise RuntimeError(f"manifest group {group} selected no records")
    return selected


def validate_m1_train_eval_alignment(
    train_group: str,
    eval_group: str,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    m1_groups = {"capacity", "full_fit_diagnostic"}
    if train_group not in m1_groups and eval_group not in m1_groups:
        return
    if train_group != eval_group:
        raise RuntimeError("M1 same-set gates require identical train/eval groups")
    train_keys = {record_key(record) for record in train_records}
    eval_keys = {record_key(record) for record in eval_records}
    if len(train_keys) != len(train_records) or len(eval_keys) != len(eval_records):
        raise RuntimeError("M1 selected records contain duplicate semantic keys")
    if train_keys != eval_keys:
        raise RuntimeError("M1 train/eval semantic record sets differ")
    if train_group == "capacity":
        if train_keys != capacity_manifest_keys(manifest):
            raise RuntimeError("M1 capacity records differ from frozen manifest")
    else:
        prompts = full_fit_manifest_prompts(manifest)
        if {str(record["sample_id"]) for record in train_records} != prompts:
            raise RuntimeError("M1 full-fit prompts differ from frozen manifest")


def selection_evaluation_step(
    step: int, *, total_steps: int, eval_every_steps: int
) -> bool:
    """Frozen checkpoint cadence: step0, every N updates, and the final step."""

    if step < 1 or total_steps < 1 or eval_every_steps < 1:
        raise ValueError("selection-evaluation schedule arguments must be positive")
    return step % eval_every_steps == 0 or step == total_steps


def should_replace_checkpoint(
    candidate_metrics: dict[str, Any], best_metrics: dict[str, Any]
) -> bool:
    """Select strictly higher EAL; exact ties retain the earlier checkpoint."""

    return float(candidate_metrics["model_eal"]) > float(best_metrics["model_eal"])


def filter_effective_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mask = effective_record_mask(records)
    return [
        record
        for record, keep in zip(records, mask.tolist(), strict=True)
        if keep
    ]


def make_loader(
    records: list[dict[str, Any]],
    *,
    prompt_counts: dict[str, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    require_effective: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        RecordDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=partial(
            collate_japd_records,
            prompt_effective_counts=prompt_counts,
            require_effective=require_effective,
        ),
    )


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
        raise ValueError("prompt mean has no values")
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


def objective_per_block(
    objective: str,
    scores: Tensor,
    batch: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    if objective == "japd":
        output = japd_per_block_loss(
            scores,
            batch["candidate_ids"],
            batch["gold_ids"],
            batch["target_candidate_logits"],
            batch["target_matches_gold"],
        )
        return output.per_block_loss, {
            "all_prefix": output.all_prefix_loss,
            "joint_two_frontier": output.joint_two_frontier_loss,
        }
    if objective == "candidate_dpace":
        per_block = matched_candidate_dpace_per_block_loss(
            scores,
            batch["gold_candidate_ranks"],
            batch["target_matches_gold"],
            alpha=0.5,
        )
        return per_block, {"candidate_dpace": per_block}
    raise AssertionError(objective)


@torch.inference_mode()
def evaluate(
    model: GlobalDirectCandidateSelector,
    loader: DataLoader,
    target_embedding: Tensor,
    device: torch.device,
    *,
    objective: str,
    require_identity: bool = False,
) -> dict[str, Any]:
    model.eval()
    all_sample_ids: list[str] = []
    all_domains: list[str] = []
    loss_values: list[float] = []
    loss_effective: list[bool] = []
    model_lengths: list[int] = []
    base_lengths: list[int] = []
    domino_lengths: list[int] = []
    oracle_lengths: list[int] = []
    j2_success: list[int] = []
    j2_eligible: list[int] = []
    support_correct = 0
    support_total = 0
    hard_correct = 0
    hard_total = 0
    support_correct_by_position = [0] * BLOCK_LENGTH
    support_total_by_position = [0] * BLOCK_LENGTH
    suffix_edits = 0
    suffix_positions = 0
    component_values: dict[str, list[float]] = defaultdict(list)

    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        candidate_embeddings = target_embedding[batch["candidate_ids"]]
        anchor_embeddings = target_embedding[batch["anchor_ids"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                candidate_embeddings,
                batch["candidate_logits"],
                batch["base_logsumexp"],
                anchor_embeddings,
            )
        if require_identity:
            expected_scores = (
                batch["candidate_logits"].float()
                - batch["base_logsumexp"].float().unsqueeze(-1)
            )
            if not torch.equal(output.scores, expected_scores):
                raise RuntimeError("step0 JAPD scores differ from DFlash base")
            if not torch.equal(
                output.scores.argmax(dim=-1),
                torch.zeros_like(batch["gold_ids"], dtype=torch.long),
            ):
                raise RuntimeError("step0 JAPD tokens differ from DFlash base")

        per_block, components = objective_per_block(
            objective, output.scores, batch
        )
        support, horizons = clean_support(
            batch["gold_candidate_ranks"], batch["target_matches_gold"]
        )
        model_ranks = output.scores.argmax(dim=-1)
        model_ids = batch["candidate_ids"].gather(
            -1, model_ranks.unsqueeze(-1)
        ).squeeze(-1)
        base_ids = batch["candidate_ids"][..., 0]
        # policy_ids is an offline baseline metric and never enters model.forward.
        domino_ids = batch["policy_ids"]
        gold_in_lattice = batch["gold_candidate_ranks"].ge(0)
        oracle_ranks = torch.where(
            gold_in_lattice,
            batch["gold_candidate_ranks"],
            torch.zeros_like(batch["gold_candidate_ranks"]),
        )
        oracle_ids = batch["candidate_ids"].gather(
            -1, oracle_ranks.unsqueeze(-1)
        ).squeeze(-1)
        lengths = {
            "model": accepted_lengths(model_ids, batch["gold_ids"]),
            "base": accepted_lengths(base_ids, batch["gold_ids"]),
            "domino": accepted_lengths(domino_ids, batch["gold_ids"]),
            "oracle": accepted_lengths(oracle_ids, batch["gold_ids"]),
        }
        joint = strict_joint_two_frontier_metric(
            model_ranks,
            batch["gold_candidate_ranks"],
            batch["target_matches_gold"],
        )
        correct = model_ranks.eq(batch["gold_candidate_ranks"])
        hard = support & batch["gold_candidate_ranks"].ne(0)
        support_correct += int((correct & support).sum().item())
        support_total += int(support.sum().item())
        for position in range(BLOCK_LENGTH):
            support_correct_by_position[position] += int(
                (correct[:, position] & support[:, position]).sum().item()
            )
            support_total_by_position[position] += int(
                support[:, position].sum().item()
            )
        hard_correct += int((correct & hard).sum().item())
        hard_total += int(hard.sum().item())
        positions = torch.arange(BLOCK_LENGTH, device=device).unsqueeze(0)
        suffix = positions.gt(lengths["base"].unsqueeze(-1))
        suffix_edits += int((model_ranks.ne(0) & suffix).sum().item())
        suffix_positions += int(suffix.sum().item())

        all_sample_ids.extend(batch["sample_ids"])
        all_domains.extend(batch["domains"])
        loss_values.extend(per_block.detach().cpu().tolist())
        loss_effective.extend(horizons.gt(0).cpu().tolist())
        model_lengths.extend(lengths["model"].cpu().tolist())
        base_lengths.extend(lengths["base"].cpu().tolist())
        domino_lengths.extend(lengths["domino"].cpu().tolist())
        oracle_lengths.extend(lengths["oracle"].cpu().tolist())
        j2_success.extend(joint.success.cpu().to(torch.int64).tolist())
        j2_eligible.extend(joint.eligible.cpu().to(torch.int64).tolist())
        for name, values in components.items():
            component_values[name].extend(values.detach().cpu().tolist())

    effective_losses = [
        value for value, keep in zip(loss_values, loss_effective, strict=True) if keep
    ]
    effective_ids = [
        sample_id
        for sample_id, keep in zip(all_sample_ids, loss_effective, strict=True)
        if keep
    ]
    metrics = {
        "records": len(all_sample_ids),
        "prompts": len(set(all_sample_ids)),
        "objective_prompt_balanced": prompt_mean(effective_losses, effective_ids),
        "model_eal": prompt_mean(model_lengths, all_sample_ids),
        "base_eal": prompt_mean(base_lengths, all_sample_ids),
        "domino_eal": prompt_mean(domino_lengths, all_sample_ids),
        "oracle_eal": prompt_mean(oracle_lengths, all_sample_ids),
        "candidate_accuracy": support_correct / max(1, support_total),
        "hard_candidate_accuracy": hard_correct / max(1, hard_total),
        "support_positions": support_total,
        "hard_positions": hard_total,
        "harmed_fraction": sum(
            int(model < base)
            for model, base in zip(model_lengths, base_lengths, strict=True)
        ) / len(model_lengths),
        "suffix_edit_fraction": suffix_edits / max(1, suffix_positions),
        "candidate_accuracy_by_position": [
            correct / total if total else 0.0
            for correct, total in zip(
                support_correct_by_position,
                support_total_by_position,
                strict=True,
            )
        ],
        "support_positions_by_position": support_total_by_position,
    }
    for name, values in sorted(component_values.items()):
        effective_values = [
            value
            for value, keep in zip(values, loss_effective, strict=True)
            if keep
        ]
        metrics[f"{name}_prompt_balanced"] = prompt_mean(
            effective_values, effective_ids
        )
    gap = metrics["oracle_eal"] - metrics["base_eal"]
    metrics["oracle_gap_recovered"] = (
        (metrics["model_eal"] - metrics["base_eal"]) / gap if gap > 0 else 0.0
    )
    j2_mean, j2_num, j2_den, j2_prompts = prompt_ratio(
        j2_success, j2_eligible, all_sample_ids
    )
    metrics.update(
        {
            "j2_prompt_balanced": j2_mean,
            "j2_numerator": j2_num,
            "j2_denominator": j2_den,
            "j2_prompts": j2_prompts,
        }
    )
    by_domain: dict[str, dict[str, float]] = {}
    for domain in sorted(set(all_domains)):
        indices = [index for index, value in enumerate(all_domains) if value == domain]
        ids = [all_sample_ids[index] for index in indices]
        by_domain[domain] = {
            "model_eal": prompt_mean([model_lengths[index] for index in indices], ids),
            "base_eal": prompt_mean([base_lengths[index] for index in indices], ids),
            "domino_eal": prompt_mean([domino_lengths[index] for index in indices], ids),
        }
    metrics["by_domain"] = by_domain
    return metrics


def cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    minimum: float,
) -> float:
    if step <= warmup_steps and warmup_steps > 0:
        return peak * step / warmup_steps
    if total_steps <= warmup_steps:
        return minimum
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def gate_result(metrics: dict[str, Any], gate: str) -> dict[str, bool]:
    if gate == "capacity":
        return {
            "j2_at_least_99pct": metrics["j2_prompt_balanced"] >= 0.99,
            "oracle_gap_at_least_95pct": metrics["oracle_gap_recovered"] >= 0.95,
            "harm_at_most_1pct": metrics["harmed_fraction"] <= 0.01,
        }
    if gate == "full_fit":
        return {
            "j2_at_least_90pct": metrics["j2_prompt_balanced"] >= 0.90,
            "oracle_gap_at_least_80pct": metrics["oracle_gap_recovered"] >= 0.80,
        }
    return {}


def save_checkpoint(
    path: Path,
    model: GlobalDirectCandidateSelector,
    *,
    step: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    parameter_count: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "metrics": metrics,
            "parameter_count": parameter_count,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"training output already exists: {args.output}")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.eval_every_steps < 1:
        raise ValueError("eval-every-steps must be positive")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required")
    manifest = load_manifest(args.manifest)
    train_metadata, raw_train = load_rollout_records(
        args.train_rollout, split=args.train_split
    )
    eval_metadata, raw_eval = load_rollout_records(
        args.eval_rollout, split=args.eval_split
    )
    train_records = select_group(raw_train, manifest, args.train_group)
    eval_records = select_group(raw_eval, manifest, args.eval_group)
    validate_m1_train_eval_alignment(
        args.train_group,
        args.eval_group,
        train_records,
        eval_records,
        manifest,
    )
    train_sidecar_metadata, train_sidecar = load_lse_sidecar(args.train_sidecar)
    eval_sidecar_metadata, eval_sidecar = load_lse_sidecar(args.eval_sidecar)
    validate_sidecar_replay_receipt(args.train_sidecar, train_sidecar_metadata)
    validate_sidecar_replay_receipt(args.eval_sidecar, eval_sidecar_metadata)
    validate_sidecar_source(
        train_sidecar_metadata,
        rollout=args.train_rollout,
        target=args.target,
        split=args.train_split,
    )
    validate_sidecar_source(
        eval_sidecar_metadata,
        rollout=args.eval_rollout,
        target=args.target,
        split=args.eval_split,
    )
    train_records = attach_lse_sidecar(train_records, train_sidecar)
    eval_records = attach_lse_sidecar(eval_records, eval_sidecar)
    raw_train_records = len(train_records)
    train_records = filter_effective_records(train_records)
    if args.train_group in {"capacity", "full_fit_diagnostic"}:
        effective_eval_keys = {
            record_key(record)
            for record in filter_effective_records(eval_records)
        }
        if {record_key(record) for record in train_records} != effective_eval_keys:
            raise RuntimeError("M1 train/eval effective record sets differ")
    train_prompt_counts = effective_blocks_per_prompt(train_records)
    eval_prompt_counts = effective_blocks_per_prompt(eval_records)
    total_effective_blocks = len(train_records)
    total_effective_prompts = len(train_prompt_counts)
    if total_effective_prompts < 1:
        raise RuntimeError("training has no effective prompts")
    train_loader = make_loader(
        train_records,
        prompt_counts=train_prompt_counts,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        require_effective=True,
    )
    eval_loader = make_loader(
        eval_records,
        prompt_counts=eval_prompt_counts,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed,
        require_effective=False,
    )
    target_embedding = load_target_embedding(args.target).to(
        device=device, dtype=torch.bfloat16
    ).detach()
    model = GlobalDirectCandidateSelector(
        hidden_size=2560,
        max_positions=BLOCK_LENGTH,
        max_candidates=CANDIDATES,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        scope=args.scope,
        mixer="axial",
        node_encoder="additive",
        dropout=args.dropout,
        initialization_seed=args.seed,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if args.require_parameter_count and parameter_count != args.require_parameter_count:
        raise RuntimeError(
            f"expected {args.require_parameter_count} parameters, got {parameter_count}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    planned_steps = args.epochs * len(train_loader)
    total_steps = args.max_steps if args.max_steps else planned_steps
    if total_steps < 1:
        raise RuntimeError("training schedule has zero steps")
    if args.warmup_steps >= total_steps:
        raise ValueError("warmup-steps must be smaller than total training steps")

    args.output.mkdir(parents=True, exist_ok=False)
    initial = evaluate(
        model,
        eval_loader,
        target_embedding,
        device,
        objective=args.objective,
        require_identity=True,
    )
    best_metrics = initial
    best_step = 0
    save_checkpoint(
        args.output / "best.pt",
        model,
        step=0,
        metrics=initial,
        args=args,
        parameter_count=parameter_count,
    )
    history: list[dict[str, Any]] = [{"step": 0, "eval": initial}]
    print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    iterator = iter(train_loader)
    model.train()
    for step in range(1, total_steps + 1):
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            cpu_batch = next(iterator)
        batch = move_batch(cpu_batch, device)
        candidate_embeddings = target_embedding[batch["candidate_ids"]]
        anchor_embeddings = target_embedding[batch["anchor_ids"]]
        optimizer.zero_grad(set_to_none=True)
        learning_rate = cosine_learning_rate(
            step,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            peak=args.learning_rate,
            minimum=args.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                candidate_embeddings,
                batch["candidate_logits"],
                batch["base_logsumexp"],
                anchor_embeddings,
            )
            per_block, components = objective_per_block(
                args.objective, output.scores, batch
            )
            loss = fixed_prompt_balanced_batch_loss(
                per_block,
                batch["effective_blocks_per_prompt"],
                total_effective_blocks=total_effective_blocks,
                total_effective_prompts=total_effective_prompts,
            )
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm).detach().cpu()):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()

        should_evaluate = selection_evaluation_step(
            step,
            total_steps=total_steps,
            eval_every_steps=args.eval_every_steps,
        )
        if should_evaluate:
            metrics = evaluate(
                model,
                eval_loader,
                target_embedding,
                device,
                objective=args.objective,
            )
            record = {
                "step": step,
                "learning_rate": learning_rate,
                "train_loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "train_components": {
                    name: float(value.mean().detach())
                    for name, value in components.items()
                },
                "eval": metrics,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if should_replace_checkpoint(metrics, best_metrics):
                best_metrics = metrics
                best_step = step
                save_checkpoint(
                    args.output / "best.pt",
                    model,
                    step=step,
                    metrics=metrics,
                    args=args,
                    parameter_count=parameter_count,
                )
            model.train()

    checkpoint = torch.load(
        args.output / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    selected = evaluate(
        model,
        eval_loader,
        target_embedding,
        device,
        objective=args.objective,
    )
    checks = gate_result(selected, args.gate)
    passed = all(checks.values()) if checks else True
    report = {
        "format": "japd_training_v1",
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
        "parameter_count": parameter_count,
        "train_source": train_metadata,
        "eval_source": eval_metadata,
        "train_records_effective": total_effective_blocks,
        "train_records_raw": raw_train_records,
        "train_prompts_effective": total_effective_prompts,
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
    print(json.dumps({
        "best_step": best_step,
        "selected": selected,
        "gate_checks": checks,
        "gate_passed": passed,
    }, ensure_ascii=False), flush=True)
    if args.require_gate and not passed:
        raise RuntimeError(f"JAPD {args.gate} gate failed: {checks}")
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
