#!/usr/bin/env python3
"""Fine-tune the released Domino correction head on cached exact anchors."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument(
        "--additional-train-canonical", nargs="*", type=Path, default=[]
    )
    parser.add_argument(
        "--eval-canonical",
        type=Path,
        help="Optional separate cache carrying validation_select.",
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=[
            "decay_ce",
            "dpace",
            "dpace_normalized",
            "auf",
            "auf_decay",
            "breaker",
            "breaker_margin",
        ],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Use 1 for exact single-chain BF16 parity with the released evaluator.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-decay-gamma", type=float, default=7.0)
    parser.add_argument("--dpace-smoothing", type=float, default=0.5)
    parser.add_argument("--breaker-prefix-weight", type=float, default=0.1)
    parser.add_argument("--margin-temperature", type=float, default=1.0)
    parser.add_argument("--margin-offset", type=float, default=0.0)
    parser.add_argument("--l2sp-weight", type=float, default=1e-3)
    parser.add_argument(
        "--trainable-scope",
        choices=["gru_rank", "full_head"],
        default="gru_rank",
        help="Start with GRU + first rank projection; full_head also tunes vocab projection.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-blocks", type=int)
    parser.add_argument("--max-eval-blocks", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_tensor_from_checkpoint(root: Path, name: str) -> torch.Tensor:
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filename = index["weight_map"].get(name)
        if filename is None:
            raise KeyError(f"checkpoint has no tensor {name!r}")
        tensor_path = root / filename
    else:
        tensor_path = root / "model.safetensors"
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def load_records(root: Path, split: str, maximum: int | None) -> list[dict[str, Any]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("collection_complete", False):
        raise RuntimeError(f"incomplete Domino cache: {root}")
    records: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        shard_records = torch.load(shard, map_location="cpu", weights_only=False)
        records.extend(record for record in shard_records if record["split"] == split)
    if maximum is not None:
        records = records[:maximum]
    if not records:
        raise ValueError(f"no cached records for split={split!r}")
    return records


class CachedDominoDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    gold = torch.stack([record["gold_ids"].long() for record in records])
    # Training only needs anchor/gold/parallel hidden.  Older DFlash canonical
    # caches therefore remain valid training sources even though they predate
    # the released-Domino rollout fields.  Exact evaluation still uses a
    # Domino cache and checks both proposal IDs and accepted lengths below.
    cached_lengths = torch.tensor(
        [int(record.get("released_accepted_length", -1)) for record in records],
        dtype=torch.long,
    )
    cached_ids = torch.stack(
        [
            record.get(
                "released_onpolicy_ids",
                torch.full_like(record["gold_ids"], -1),
            ).long()
            for record in records
        ]
    )
    return {
        "sample_ids": [str(record["sample_id"]) for record in records],
        "domains": [str(record["domain"]) for record in records],
        "anchors": torch.tensor(
            [int(record["anchor_token_id"]) for record in records], dtype=torch.long
        ),
        "gold": gold,
        "hidden": torch.stack(
            [record["parallel_hidden"].to(torch.bfloat16) for record in records]
        ),
        "cached_released_lengths": cached_lengths,
        "cached_released_ids": cached_ids,
    }


def acceptance_lengths(proposals: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    return (proposals == gold).to(torch.long).cumprod(dim=-1).sum(dim=-1)


def position_decay(length: int, gamma: float, device: torch.device) -> torch.Tensor:
    return torch.exp(-torch.arange(length, device=device, dtype=torch.float32) / gamma)


def auf_reach_mask(predicted: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """One through the first mismatch, zero strictly after it."""
    matches = (predicted == gold).to(torch.float32)
    return torch.cat(
        [torch.ones_like(matches[:, :1]), matches[:, :-1].cumprod(dim=-1)], dim=-1
    )


def dpace_weights(
    logits: torch.Tensor, gold: torch.Tensor, smoothing: float
) -> torch.Tensor:
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("D-PACE smoothing must be in [0, 1]")
    with torch.no_grad():
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            gold.reshape(-1),
            reduction="none",
        ).reshape_as(gold)
        probabilities = torch.exp(-ce)
        smoothed = (1.0 - smoothing) * probabilities + smoothing
        survival = smoothed.cumprod(dim=-1)
        return torch.flip(torch.flip(survival, dims=[-1]).cumsum(dim=-1), dims=[-1])


def best_competitor_margin_loss(
    logits: torch.Tensor,
    gold: torch.Tensor,
    *,
    temperature: float,
    offset: float,
) -> torch.Tensor:
    """Smooth loss on the greedy gold-vs-best-other decision boundary."""

    if temperature <= 0.0:
        raise ValueError("margin temperature must be positive")
    top_values, top_indices = logits.topk(k=2, dim=-1)
    gold_values = logits.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    competitor = torch.where(
        top_indices[..., 0] == gold, top_values[..., 1], top_values[..., 0]
    )
    gap = (competitor.float() - gold_values.float() + offset) / temperature
    return F.softplus(gap) * temperature


def objective_loss(
    *,
    all_logits: torch.Tensor,
    gold: torch.Tensor,
    objective: str,
    gamma: float,
    dpace_smoothing: float,
    breaker_prefix_weight: float = 0.1,
    margin_temperature: float = 1.0,
    margin_offset: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss for corrected positions 1..L-1; position zero is backbone-only."""
    suffix_logits = all_logits[:, 1:]
    suffix_gold = gold[:, 1:]
    if objective == "breaker_margin":
        token_loss = best_competitor_margin_loss(
            suffix_logits,
            suffix_gold,
            temperature=margin_temperature,
            offset=margin_offset,
        )
    else:
        token_loss = F.cross_entropy(
            suffix_logits.float().reshape(-1, suffix_logits.shape[-1]),
            suffix_gold.reshape(-1),
            reduction="none",
        ).reshape_as(suffix_gold)
    decay = position_decay(gold.shape[1], gamma, gold.device)[1:].view(1, -1)
    with torch.no_grad():
        predicted = all_logits.argmax(dim=-1)
        base_reachable = (predicted[:, :1] == gold[:, :1]).to(torch.float32)

    if objective == "decay_ce":
        weights = decay.expand_as(token_loss)
    elif objective == "dpace":
        # Cached index zero is the first draft candidate (the official
        # block-training index zero is the anchor).  Its detached probability
        # therefore enters every suffix survival weight even though the frozen
        # head cannot optimize its CE.  Match D-PACE's batch-size reduction.
        weights = (
            dpace_weights(all_logits, gold, dpace_smoothing)[:, 1:]
        )
    elif objective == "dpace_normalized":
        weights = dpace_weights(all_logits, gold, dpace_smoothing)[:, 1:]
    elif objective in {"auf", "auf_decay"}:
        with torch.no_grad():
            weights = auf_reach_mask(predicted, gold)[:, 1:] * base_reachable
            if objective == "auf_decay":
                weights = weights * decay
    elif objective in {"breaker", "breaker_margin"}:
        if not 0.0 <= breaker_prefix_weight <= 1.0:
            raise ValueError("breaker_prefix_weight must be in [0, 1]")
        with torch.no_grad():
            reach = auf_reach_mask(predicted, gold)[:, 1:] * base_reachable
            suffix_match = (predicted[:, 1:] == gold[:, 1:]).to(torch.float32)
            breaker = reach * (1.0 - suffix_match)
            preserved_prefix = reach * suffix_match
            weights = breaker + breaker_prefix_weight * preserved_prefix
    else:
        raise ValueError(f"unknown objective {objective!r}")

    if objective == "dpace":
        denominator = torch.tensor(
            float(gold.shape[0]), device=gold.device, dtype=torch.float32
        )
    else:
        denominator = weights.sum().clamp_min(1.0)
    loss = (token_loss * weights).sum() / denominator
    with torch.no_grad():
        teacher_accepted = acceptance_lengths(all_logits.argmax(dim=-1), gold)
    return loss, {
        "weight_sum": float(weights.sum().detach()),
        "loss_denominator": float(denominator.detach()),
        "teacher_eal": float(teacher_accepted.float().mean()),
        "teacher_full_horizon": float(
            (teacher_accepted == gold.shape[1]).float().mean()
        ),
        "base_reachable_fraction": float(base_reachable.mean()),
    }


class MasterAdamW:
    """FP32 AdamW masters for a BF16 inference-identical trainable head."""

    def __init__(self, named_parameters: list[tuple[str, nn.Parameter]], lr: float) -> None:
        self.named_parameters = named_parameters
        self.masters = [
            nn.Parameter(parameter.detach().float().clone(), requires_grad=True)
            for _, parameter in named_parameters
        ]
        self.optimizer = torch.optim.AdamW(
            self.masters, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        for _, parameter in self.named_parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self, max_grad_norm: float) -> float:
        squared_norm = torch.zeros((), device=self.masters[0].device)
        for _, parameter in self.named_parameters:
            if parameter.grad is not None:
                squared_norm += parameter.grad.float().square().sum()
        grad_norm = float(torch.sqrt(squared_norm))
        clip = min(1.0, max_grad_norm / (grad_norm + 1e-12))
        for master, (_, parameter) in zip(
            self.masters, self.named_parameters, strict=True
        ):
            master.grad = (
                None if parameter.grad is None else parameter.grad.float() * clip
            )
        self.optimizer.step()
        for master, (_, parameter) in zip(
            self.masters, self.named_parameters, strict=True
        ):
            parameter.copy_(master.to(dtype=parameter.dtype))
        return grad_norm


def trainable_head_parameters(
    domino: nn.Module, scope: str
) -> list[tuple[str, nn.Parameter]]:
    for parameter in domino.parameters():
        parameter.requires_grad_(False)
    named: list[tuple[str, nn.Parameter]] = []
    if scope not in {"gru_rank", "full_head"}:
        raise ValueError(f"unknown trainable scope {scope!r}")
    for module_name in ["prefix_gru", "embed_proj"]:
        module = getattr(domino, module_name)
        for name, parameter in module.named_parameters():
            if scope == "gru_rank" and module_name == "embed_proj" and name.startswith("2."):
                continue
            parameter.requires_grad_(True)
            named.append((f"{module_name}.{name}", parameter))
    return named


def l2sp_penalty(
    named_parameters: list[tuple[str, nn.Parameter]],
    reference: dict[str, torch.Tensor],
) -> torch.Tensor:
    numerator = torch.zeros((), device=named_parameters[0][1].device)
    for name, parameter in named_parameters:
        numerator = numerator + (parameter.float() - reference[name]).square().sum()
    return numerator


def head_state(domino: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for module_name in ["prefix_gru", "embed_proj"]:
        module = getattr(domino, module_name)
        for name, tensor in module.state_dict().items():
            state[f"{module_name}.{name}"] = tensor.detach().cpu().clone()
    return state


@torch.no_grad()
def load_head_state(domino: nn.Module, state: dict[str, torch.Tensor]) -> None:
    for module_name in ["prefix_gru", "embed_proj"]:
        prefix = f"{module_name}."
        module_state = {
            name[len(prefix) :]: tensor
            for name, tensor in state.items()
            if name.startswith(prefix)
        }
        getattr(domino, module_name).load_state_dict(module_state, strict=True)


def teacher_logits(
    *,
    domino: nn.Module,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    gold: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        base_logits = F.linear(hidden, target_weight)
        prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
        prefix_embeddings = F.embedding(prefix_ids, target_weight)
    gru_out, _ = domino.prefix_gru(prefix_embeddings)
    prefix_states = gru_out[:, 1:]
    correction = domino.embed_proj(
        torch.cat([hidden[:, 1:], prefix_states], dim=-1)
    )
    return torch.cat(
        [base_logits[:, :1], base_logits[:, 1:] + correction], dim=1
    )


@torch.inference_mode()
def teacher_alignment_check(
    *,
    domino: nn.Module,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    gold: torch.Tensor,
    hidden: torch.Tensor,
) -> dict[str, float | int]:
    """Compare vectorized teacher states with an explicit position loop."""
    vectorized = teacher_logits(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
    )
    base = F.linear(hidden, target_weight)
    reference = [base[:, :1]]
    first_prefix = torch.cat([anchors[:, None], gold[:, :1]], dim=-1)
    _, state = domino.prefix_gru(F.embedding(first_prefix, target_weight))
    for position in range(1, hidden.shape[1]):
        correction = domino.embed_proj(
            torch.cat(
                [hidden[:, position : position + 1], state.transpose(0, 1)],
                dim=-1,
            )
        )
        reference.append(base[:, position : position + 1] + correction)
        if position + 1 < hidden.shape[1]:
            _, state = domino.prefix_gru(
                F.embedding(gold[:, position : position + 1], target_weight), state
            )
    looped = torch.cat(reference, dim=1)
    difference = (vectorized.float() - looped.float()).abs()
    return {
        "max_abs_logit_difference": float(difference.max()),
        "argmax_mismatches": int(
            (vectorized.argmax(dim=-1) != looped.argmax(dim=-1)).sum()
        ),
    }


@torch.inference_mode()
def onpolicy_ids(
    *,
    domino: nn.Module,
    target_weight: torch.Tensor,
    anchors: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    base_logits = F.linear(hidden, target_weight)
    batch, positions = hidden.shape[:2]
    proposals = torch.empty((batch, positions), dtype=torch.long, device=hidden.device)
    first = base_logits[:, :1].argmax(dim=-1)
    proposals[:, 0] = first[:, 0]
    prefix_ids = torch.cat([anchors[:, None], first], dim=-1)
    _, state = domino.prefix_gru(F.embedding(prefix_ids, target_weight))
    for position in range(1, positions):
        state_for_head = state.transpose(0, 1)
        correction = domino.embed_proj(
            torch.cat([hidden[:, position : position + 1], state_for_head], dim=-1)
        )
        token = (base_logits[:, position : position + 1] + correction).argmax(dim=-1)
        proposals[:, position] = token[:, 0]
        if position + 1 < positions:
            _, state = domino.prefix_gru(F.embedding(token, target_weight), state)
    return proposals


def mean(values: Iterable[int | float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items)


def summarize_lengths(
    sample_ids: list[str], domains: list[str], lengths: list[int], horizon: int
) -> dict[str, Any]:
    by_prompt: dict[str, list[int]] = defaultdict(list)
    by_domain: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for sample_id, domain, length in zip(sample_ids, domains, lengths, strict=True):
        by_prompt[sample_id].append(length)
        by_domain[domain].append((sample_id, length))

    def one(pairs: list[tuple[str, int]]) -> dict[str, float]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for sample_id, length in pairs:
            grouped[sample_id].append(length)
        raw = [length for _, length in pairs]
        return {
            "blocks": len(raw),
            "mean_accepted_draft_tokens_round_weighted": mean(raw),
            "mean_accepted_draft_tokens_prompt_balanced": mean(
                mean(values) for values in grouped.values()
            ),
            "full_horizon_acceptance": mean(length == horizon for length in raw),
        }

    overall_pairs = list(zip(sample_ids, lengths, strict=True))
    return {
        "overall": one(overall_pairs),
        "by_domain": {domain: one(pairs) for domain, pairs in sorted(by_domain.items())},
    }


def prompt_bootstrap_difference(
    sample_ids: list[str], left: list[int], right: list[int], draws: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample_id, lvalue, rvalue in zip(sample_ids, left, right, strict=True):
        grouped[sample_id].append(float(lvalue - rvalue))
    cluster_means = [mean(values) for values in grouped.values()]
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(cluster_means) for _ in cluster_means) for _ in range(draws)
    )
    return {
        "mean_difference_prompt_balanced": mean(cluster_means),
        "ci95_prompt_cluster_bootstrap": [
            estimates[int(0.025 * (draws - 1))],
            estimates[int(0.975 * (draws - 1))],
        ],
    }


@torch.inference_mode()
def evaluate(
    domino: nn.Module,
    target_weight: torch.Tensor,
    loader: DataLoader,
) -> dict[str, Any]:
    domino.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    lengths: list[int] = []
    cached_lengths: list[int] = []
    cached_token_mismatches = 0
    horizon = 0
    for batch in loader:
        anchors = batch["anchors"].to(target_weight.device, non_blocking=True)
        gold = batch["gold"].to(target_weight.device, non_blocking=True)
        hidden = batch["hidden"].to(target_weight.device, non_blocking=True)
        proposals = onpolicy_ids(
            domino=domino,
            target_weight=target_weight,
            anchors=anchors,
            hidden=hidden,
        )
        batch_lengths = acceptance_lengths(proposals, gold).cpu().tolist()
        cached_ids = batch["cached_released_ids"].to(
            target_weight.device, non_blocking=True
        )
        cached_token_mismatches += int((proposals != cached_ids).sum())
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
                left != right for left, right in zip(lengths, cached_lengths, strict=True)
            ),
            "cached_token_mismatches": cached_token_mismatches,
        }
    )
    return summary


def cosine_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("cached Domino head training requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    train_records = load_records(args.canonical, "train", args.max_train_blocks)
    for additional_root in args.additional_train_canonical:
        train_records.extend(load_records(additional_root, "train", None))
    eval_root = args.eval_canonical or args.canonical
    eval_records = load_records(
        eval_root, "validation_select", args.max_eval_blocks
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        CachedDominoDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        CachedDominoDataset(eval_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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
    )
    if int(getattr(domino, "pure_draft_prefix_len", 0)) != 1:
        raise ValueError("trainer expects Domino pure_draft_prefix_len=1")
    named_head = trainable_head_parameters(domino, args.trainable_scope)
    released_reference = {
        name: parameter.detach().float().clone() for name, parameter in named_head
    }
    master_optimizer = MasterAdamW(named_head, args.learning_rate)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        master_optimizer.optimizer,
        lambda step: cosine_schedule(step, total_steps, warmup_steps),
    )

    alignment_batch = collate(eval_records[:1])
    alignment = teacher_alignment_check(
        domino=domino,
        target_weight=target_weight,
        anchors=alignment_batch["anchors"].to(target_weight.device),
        gold=alignment_batch["gold"].to(target_weight.device),
        hidden=alignment_batch["hidden"].to(target_weight.device),
    )
    if alignment["argmax_mismatches"] != 0:
        raise RuntimeError(f"teacher-state alignment failed: {alignment}")
    baseline = evaluate(domino, target_weight, eval_loader)
    if baseline["cached_length_mismatches"] != 0:
        raise RuntimeError(
            "same-code-path released baseline did not reproduce cached accepted "
            f"lengths: {baseline['cached_length_mismatches']} mismatched blocks"
        )
    baseline_eal = baseline["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    print(
        json.dumps(
            {
                "baseline_eal": baseline_eal,
                "cached_length_mismatches": baseline["cached_length_mismatches"],
                "cached_token_mismatches": baseline["cached_token_mismatches"],
                "teacher_alignment": alignment,
            },
            indent=2,
        ),
        flush=True,
    )

    best_state = head_state(domino)
    best_epoch = 0
    best_eval = baseline
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        domino.prefix_gru.train()
        domino.embed_proj.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in train_loader:
            anchors = batch["anchors"].to(target_weight.device, non_blocking=True)
            gold = batch["gold"].to(target_weight.device, non_blocking=True)
            hidden = batch["hidden"].to(target_weight.device, non_blocking=True)
            master_optimizer.zero_grad()
            logits = teacher_logits(
                domino=domino,
                target_weight=target_weight,
                anchors=anchors,
                gold=gold,
                hidden=hidden,
            )
            task_loss, diagnostics = objective_loss(
                all_logits=logits,
                gold=gold,
                objective=args.objective,
                gamma=args.loss_decay_gamma,
                dpace_smoothing=args.dpace_smoothing,
                breaker_prefix_weight=args.breaker_prefix_weight,
                margin_temperature=args.margin_temperature,
                margin_offset=args.margin_offset,
            )
            penalty = l2sp_penalty(named_head, released_reference)
            loss = task_loss + args.l2sp_weight * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            loss.backward()
            grad_norm = master_optimizer.step(args.max_grad_norm)
            scheduler.step()
            totals["loss"] += float(loss.detach())
            totals["task_loss"] += float(task_loss.detach())
            totals["l2sp_penalty"] += float(penalty.detach())
            totals["grad_norm"] += grad_norm
            for key, value in diagnostics.items():
                totals[key] += value
            batches += 1
            global_step += 1
            if global_step % 100 == 0:
                print(
                    f"step={global_step}/{total_steps} epoch={epoch} "
                    f"loss={float(loss.detach()):.6f} grad={grad_norm:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )

        epoch_eval = evaluate(domino, target_weight, eval_loader)
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
                "changed_blocks_vs_cache": epoch_eval["cached_length_mismatches"],
            },
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        if epoch_eal > best_eval["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]:
            best_state = head_state(domino)
            best_epoch = epoch
            best_eval = epoch_eval

    load_head_state(domino, best_state)
    selected = evaluate(domino, target_weight, eval_loader)
    selected_eal = selected["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    paired = prompt_bootstrap_difference(
        selected["sample_ids"],
        selected["lengths"],
        baseline["lengths"],
        args.bootstrap_samples,
        args.seed + 917,
    )
    beneficial = sum(
        left > right
        for left, right in zip(selected["lengths"], baseline["lengths"], strict=True)
    )
    harmful = sum(
        left < right
        for left, right in zip(selected["lengths"], baseline["lengths"], strict=True)
    )
    checkpoint = {
        "head_state_dict": best_state,
        "objective": args.objective,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "trainable_scope": args.trainable_scope,
        "l2sp_weight": args.l2sp_weight,
        "margin_temperature": args.margin_temperature,
        "margin_offset": args.margin_offset,
        "source_domino": str(args.domino_draft.resolve()),
        "validation_select_eal": selected_eal,
        "released_same_run_eal": baseline_eal,
    }
    torch.save(checkpoint, args.output / "best_head.pt")
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "objective": args.objective,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_blocks": len(train_records),
        "validation_blocks": len(eval_records),
        "trainable_parameters": sum(parameter.numel() for _, parameter in named_head),
        "teacher_alignment": alignment,
        "seconds": time.perf_counter() - started,
        "baseline": {"overall": baseline["overall"], "by_domain": baseline["by_domain"]},
        "history": history,
        "best_epoch": best_epoch,
        "selected": {"overall": selected["overall"], "by_domain": selected["by_domain"]},
        "selected_delta_vs_released": selected_eal - baseline_eal,
        "paired_vs_released": paired,
        "beneficial_blocks": beneficial,
        "harmful_blocks": harmful,
        "unchanged_blocks": len(selected["lengths"]) - beneficial - harmful,
        "baseline_cache_mismatches": baseline["cached_length_mismatches"],
        "baseline_cache_token_mismatches": baseline["cached_token_mismatches"],
        "inputs": {
            "canonical": str(args.canonical.resolve()),
            "additional_train_canonical": [
                str(path.resolve()) for path in args.additional_train_canonical
            ],
            "eval_canonical": str(eval_root.resolve()),
            "target": str(args.target.resolve()),
            "domino_draft": str(args.domino_draft.resolve()),
        },
        "checkpoint": str((args.output / "best_head.pt").resolve()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
