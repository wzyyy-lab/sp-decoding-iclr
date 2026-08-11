#!/usr/bin/env python3
"""Jointly adapt released Domino's final parallel-backbone layers with LoRA."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM

from sph.domino_joint_runtime import (
    CanonicalBlock,
    acceptance_lengths,
    domino_onpolicy_ids,
    domino_prediction_hidden,
    domino_teacher_and_base_logits,
    domino_teacher_logits,
    frontier_margin_joint_loss,
    greedy_reachable_joint_loss,
    select_even_prompt_blocks,
    summarize_prompt_balanced_lengths,
    target_distilled_union_joint_loss,
    target_frontier_distilled_union_joint_loss,
    target_full_vocab_distilled_joint_loss,
    union_topk_frontier_protected_joint_loss,
    union_topk_oracle_prefix_joint_loss,
    union_topk_reachable_joint_loss,
)
from sph.fbpf import (
    FBPF_EXPECTED_TRAINABLE_PARAMETERS,
    cosine_warmup_learning_rate,
    count_lora_parameters,
    dpace_loss,
    inject_fbpf_lora,
    lora_disabled,
    named_lora_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--train-canonical", type=Path, required=True)
    parser.add_argument("--eval-canonical", type=Path, required=True)
    parser.add_argument("--eval-domino-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-adaptation", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation_select")
    parser.add_argument("--max-train-prompts", type=int, default=8_000)
    parser.add_argument("--max-eval-prompts", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--peak-learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--gradient-accumulation-prompts", type=int, default=1)
    parser.add_argument(
        "--train-blocks-per-prompt",
        type=int,
        default=4,
        help="Evenly select this many anchors, or use 0 for every available anchor.",
    )
    parser.add_argument("--curriculum-final-base-weight", type=float, default=0.0)
    parser.add_argument("--curriculum-transition-ratio", type=float, default=1.0)
    parser.add_argument(
        "--objective",
        choices=(
            "dpace",
            "greedy_reachable",
            "frontier_margin",
            "domino_curriculum",
            "topk_curriculum",
            "topk_oracle_curriculum",
            "topk_frontier_curriculum",
            "target_distill",
            "target_frontier_distill",
            "target_full_vocab_distill",
        ),
        default="dpace",
    )
    parser.add_argument("--train-causal-head", action="store_true")
    parser.add_argument("--train-full-backbone", action="store_true")
    parser.add_argument(
        "--full-b16",
        action="store_true",
        help="Train/select on all 16 Domino outputs instead of the legacy 15-token cache horizon.",
    )
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument("--feature-anchor-weight", type=float, default=0.0)
    parser.add_argument("--protection-margin", type=float, default=0.05)
    parser.add_argument("--protection-weight", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=2.0)
    parser.add_argument("--target-protect-weight", type=float, default=1.0)
    parser.add_argument("--target-repair-weight", type=float, default=4.0)
    parser.add_argument("--target-future-weight", type=float, default=1.0)
    parser.add_argument("--target-protection-margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def shard_paths(root: Path) -> list[Path]:
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        declared = [root / str(item["path"]) for item in metadata.get("shards", [])]
        if declared:
            return declared
    return sorted(root.glob("shard-*.pt"))


def light_block(record: dict[str, Any]) -> CanonicalBlock:
    return CanonicalBlock(
        sample_id=str(record["sample_id"]),
        domain=str(record["domain"]),
        context_ids=record["context_ids_before_anchor"].to(torch.long),
        anchor_token_id=int(record["anchor_token_id"]),
        gold_ids=record["gold_ids"].to(torch.long),
        anchor_offset=int(record["anchor_offset"]),
    )


def load_prompt_groups(
    root: Path,
    *,
    split: str,
    max_prompts: int | None,
) -> list[tuple[CanonicalBlock, ...]]:
    groups: OrderedDict[str, list[CanonicalBlock]] = OrderedDict()
    for shard in shard_paths(root):
        records = torch.load(shard, map_location="cpu", weights_only=False)
        for raw in records:
            if str(raw["split"]) != split:
                continue
            sample_id = str(raw["sample_id"])
            groups.setdefault(sample_id, []).append(light_block(raw))
        del records
    if not groups:
        raise ValueError(f"no {split!r} records found under {root}")
    result: list[tuple[CanonicalBlock, ...]] = []
    skipped_short = 0
    for values in groups.values():
        ordered = tuple(sorted(values, key=lambda record: record.anchor_offset))
        if len(ordered) < 4:
            skipped_short += 1
            continue
        result.append(ordered)
    if max_prompts is not None:
        if len(result) < max_prompts:
            raise ValueError(
                f"requested {max_prompts} complete prompts but loaded {len(result)}"
            )
        result = result[:max_prompts]
    if skipped_short:
        print(f"skipped {skipped_short} prompts with fewer than four blocks", flush=True)
    return result


def load_released_cache(
    root: Path, *, split: str, full_b16: bool = False
) -> dict[tuple[str, int], tuple[torch.Tensor, int]]:
    result: dict[tuple[str, int], tuple[torch.Tensor, int]] = {}
    for shard in shard_paths(root):
        records = torch.load(shard, map_location="cpu", weights_only=False)
        for record in records:
            if str(record["split"]) != split:
                continue
            key = (str(record["sample_id"]), int(record["anchor_offset"]))
            if full_b16:
                if "teacher_full_ids" not in record:
                    raise ValueError(
                        "full-B16 selection cache has no teacher_full_ids"
                    )
                released_ids = record["teacher_full_ids"]
                released_length = record["teacher_accepted_length"]
            else:
                if "released_onpolicy_ids" in record:
                    released_ids = record["released_onpolicy_ids"]
                    released_length = record["released_accepted_length"]
                else:
                    released_ids = record["teacher_ids"]
                    released_length = min(
                        int(record["teacher_accepted_length"]),
                        int(released_ids.numel()),
                    )
            result[key] = (released_ids.to(torch.long), int(released_length))
        del records
    if not result:
        raise ValueError(f"released Domino cache has no {split!r} records")
    return result


def extract_context_feature(
    hidden_states: tuple[torch.Tensor, ...], layer_ids: list[int]
) -> torch.Tensor:
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


@torch.no_grad()
def materialize_prompt_inputs(
    *,
    target: nn.Module,
    domino: nn.Module,
    records: Sequence[CanonicalBlock],
    include_target_teacher: bool = False,
    full_b16: bool = False,
) -> dict[str, torch.Tensor]:
    longest_record = max(records, key=lambda record: int(record.context_ids.numel()))
    longest = longest_record.context_ids
    for record in records:
        length = int(record.context_ids.numel())
        if not torch.equal(longest[:length], record.context_ids):
            raise ValueError("records in one prompt are not prefix nested")
    canonical_horizon = int(longest_record.gold_ids.numel())
    target_sequence = longest
    if include_target_teacher or full_b16:
        teacher_horizon = canonical_horizon + int(full_b16)
        # Recover every target-teacher state with one causal pass.  Several OPB
        # anchor schedules have a final gap shorter than the 15-token horizon,
        # so a pass ending at the longest context cannot label all selected
        # anchors.  Causality keeps all earlier context features unchanged in
        # real arithmetic while this extension supplies the missing suffix.
        target_sequence = torch.cat(
            [
                longest,
                torch.tensor(
                    [longest_record.anchor_token_id], dtype=torch.long
                ),
                longest_record.gold_ids[: teacher_horizon - 1].to(torch.long),
            ]
        )
    context = target_sequence.unsqueeze(0).to("cuda:0", non_blocking=True)
    target_outputs = target.model(
        context,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    full_features = extract_context_feature(
        target_outputs.hidden_states, list(domino.target_layer_ids)
    ).detach()
    batch = len(records)
    context_lengths = torch.tensor(
        [int(record.context_ids.numel()) for record in records],
        dtype=torch.long,
        device="cuda:0",
    )
    target_hidden = full_features.expand(batch, -1, -1)
    anchors = torch.tensor(
        [record.anchor_token_id for record in records],
        dtype=torch.long,
        device="cuda:0",
    )
    if full_b16:
        # The canonical cache stores the first 15 targets.  Reconstruct the
        # exact 16th target exactly as the independent B16 evaluator does: one
        # next-token argmax after the longest cached continuation, then slice
        # every nested anchor from that single shared target sequence.
        next_token = target.lm_head(
            target_outputs.last_hidden_state[:, -1:]
        ).argmax(dim=-1)
        extended_sequence = torch.cat([context, next_token], dim=-1)
        gold_rows: list[torch.Tensor] = []
        for record, length_tensor in zip(records, context_lengths, strict=True):
            context_length = int(length_tensor)
            row_gold = extended_sequence[
                0, context_length + 1 : context_length + 17
            ]
            if row_gold.shape[0] != 16:
                raise RuntimeError("target replay returned fewer than 16 gold tokens")
            if not torch.equal(
                row_gold[:canonical_horizon].cpu(), record.gold_ids.to(torch.long)
            ):
                raise RuntimeError(
                    "full-B16 target reconstruction changed the canonical first 15 tokens"
                )
            gold_rows.append(row_gold)
        gold = torch.stack(gold_rows)
    else:
        gold = torch.stack([record.gold_ids for record in records]).to(
            "cuda:0", non_blocking=True
        )
    target_teacher_hidden: torch.Tensor | None = None
    if include_target_teacher:
        horizon = int(gold.shape[1])
        teacher_rows: list[torch.Tensor] = []
        for length_tensor in context_lengths:
            context_length = int(length_tensor)
            teacher_stop = context_length + horizon
            row_hidden = target_outputs.last_hidden_state[
                :, context_length:teacher_stop
            ]
            if row_hidden.shape[1] != horizon:
                raise RuntimeError("target replay returned the wrong teacher horizon")
            teacher_rows.append(row_hidden)
        target_teacher_hidden = torch.cat(teacher_rows, dim=0).detach()
    del target_outputs
    block_size = int(domino.block_size)
    block_ids = torch.full(
        (batch, block_size),
        int(domino.mask_token_id),
        dtype=torch.long,
        device="cuda:0",
    )
    block_ids[:, 0] = anchors
    noise_embedding = target.model.embed_tokens(block_ids).detach()
    result = {
        "target_hidden": target_hidden,
        "noise_embedding": noise_embedding,
        "context_lengths": context_lengths,
        "anchors": anchors,
        "gold": gold,
    }
    if target_teacher_hidden is not None:
        result["target_teacher_hidden"] = target_teacher_hidden
    return result


def parallel_hidden_rows(
    domino: nn.Module, inputs: dict[str, torch.Tensor]
) -> list[torch.Tensor]:
    # Variable-length padding is mathematically masked, but SDPA changes its
    # reduction path and changed accepted length on the real BF16 checkpoint.
    # Reuse the one longest target pass while executing each draft block on its
    # exact released context geometry.  Keep the rows separate here: the
    # released implementation applies both the LM head and prefix rollout with
    # batch size one, so batching after the backbone can still alter BF16 ties.
    outputs: list[torch.Tensor] = []
    for row, length_tensor in enumerate(inputs["context_lengths"]):
        context_length = int(length_tensor)
        position_ids = torch.arange(
            context_length + int(domino.block_size),
            dtype=torch.long,
            device=inputs["gold"].device,
        ).unsqueeze(0)
        full_hidden = domino(
            target_hidden=inputs["target_hidden"][row : row + 1, :context_length],
            noise_embedding=inputs["noise_embedding"][row : row + 1],
            position_ids=position_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        # Validate the released shift-label contract, but keep all 16 states.
        # Released Domino applies its LM head and causal rollout to all 16.
        # Legacy caches supervise 15 positions; --full-b16 uses all 16.
        domino_prediction_hidden(
            domino, full_hidden, horizon=int(inputs["gold"].shape[1])
        )
        outputs.append(full_hidden)
    return outputs


@torch.no_grad()
def evaluate(
    *,
    target: nn.Module,
    domino: nn.Module,
    target_weight: torch.Tensor,
    prompt_groups: Sequence[Sequence[CanonicalBlock]],
    released_cache: dict[tuple[str, int], tuple[torch.Tensor, int]],
    full_b16: bool = False,
) -> dict[str, Any]:
    domino.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    lengths: list[int] = []
    released_lengths: list[int] = []
    token_mismatches = 0
    length_mismatches = 0
    horizon = 0
    for prompt_records in prompt_groups:
        # The released cache obtains every anchor's target features by slicing
        # one pass over the prompt's longest context.  Materialize all anchors
        # together so early anchors use that exact same target computation.
        records = prompt_records
        inputs = materialize_prompt_inputs(
            target=target,
            domino=domino,
            records=records,
            full_b16=full_b16,
        )
        hidden_rows = parallel_hidden_rows(domino, inputs)
        horizon = int(inputs["gold"].shape[1])
        for row, (record, full_hidden) in enumerate(
            zip(records, hidden_rows, strict=True)
        ):
            full_proposals = domino_onpolicy_ids(
                domino=domino,
                target_weight=target_weight,
                anchors=inputs["anchors"][row : row + 1],
                hidden=full_hidden,
            )
            proposals = full_proposals[:, :horizon]
            current_length = int(
                acceptance_lengths(
                    proposals, inputs["gold"][row : row + 1]
                )[0]
            )
            key = (record.sample_id, record.anchor_offset)
            cached_ids, cached_length = released_cache[key]
            sample_ids.append(record.sample_id)
            domains.append(record.domain)
            lengths.append(current_length)
            released_lengths.append(cached_length)
            cached_gpu = cached_ids.to(proposals.device)
            token_mismatches += int((proposals[0] != cached_gpu).sum())
            length_mismatches += int(current_length != cached_length)
    current = summarize_prompt_balanced_lengths(
        sample_ids, domains, lengths, horizon=horizon
    )
    released = summarize_prompt_balanced_lengths(
        sample_ids, domains, released_lengths, horizon=horizon
    )
    current_eal = current["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    released_eal = released["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    return {
        "overall": current["overall"],
        "by_domain": current["by_domain"],
        "released_overall": released["overall"],
        "delta_vs_released": current_eal - released_eal,
        "released_token_mismatches": token_mismatches,
        "released_length_mismatches": length_mismatches,
    }


def clone_trainable_state(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_parameters
    }


@torch.no_grad()
def load_trainable_state(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    state: dict[str, torch.Tensor],
) -> None:
    live = dict(named_parameters)
    if tuple(live) != tuple(state):
        raise ValueError("adaptation checkpoint layout differs from live model")
    for name, parameter in live.items():
        parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


class MasterAdamW:
    """AdamW over FP32 masters while the released head stays BF16-exact."""

    def __init__(
        self,
        named_parameters: Sequence[tuple[str, nn.Parameter]],
        *,
        learning_rate: float,
    ) -> None:
        self.named_parameters = tuple(named_parameters)
        self.masters = [
            nn.Parameter(parameter.detach().float().clone(), requires_grad=True)
            for _, parameter in self.named_parameters
        ]
        self.optimizer = torch.optim.AdamW(
            self.masters,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        for _, parameter in self.named_parameters:
            parameter.grad = None

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    @torch.no_grad()
    def step(self, max_grad_norm: float) -> float:
        squared_norm = torch.zeros((), device=self.masters[0].device)
        for _, parameter in self.named_parameters:
            if parameter.grad is not None:
                squared_norm += parameter.grad.float().square().sum()
        if not bool(torch.isfinite(squared_norm).item()):
            raise FloatingPointError("non-finite full-model gradient norm")
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Domino backbone adaptation requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    args.output.mkdir(parents=True)
    if args.feature_anchor_weight < 0:
        raise ValueError("feature-anchor-weight must be non-negative")
    if args.objective != "greedy_reachable" and args.feature_anchor_weight != 0:
        raise ValueError("feature anchoring is only defined for greedy_reachable")
    if args.protection_margin < 0 or args.protection_weight < 0:
        raise ValueError("frontier protection values must be non-negative")
    if args.target_temperature <= 0:
        raise ValueError("target-temperature must be positive")
    if args.target_protection_margin < 0:
        raise ValueError("target-protection-margin must be non-negative")
    if min(
        args.target_protect_weight,
        args.target_repair_weight,
        args.target_future_weight,
    ) < 0:
        raise ValueError("target distillation weights must be non-negative")
    if args.gradient_accumulation_prompts < 1:
        raise ValueError("gradient-accumulation-prompts must be positive")
    if args.train_blocks_per_prompt < 0:
        raise ValueError("train-blocks-per-prompt must be non-negative")
    if not 0.0 <= args.curriculum_final_base_weight <= 1.0:
        raise ValueError("curriculum-final-base-weight must lie in [0, 1]")
    if not 0.0 < args.curriculum_transition_ratio <= 1.0:
        raise ValueError("curriculum-transition-ratio must lie in (0, 1]")
    if args.objective not in {
        "domino_curriculum",
        "topk_curriculum",
        "topk_oracle_curriculum",
        "topk_frontier_curriculum",
    } and (
        args.curriculum_final_base_weight != 0.0
        or args.curriculum_transition_ratio != 1.0
    ):
        raise ValueError("curriculum schedule options require domino_curriculum")
    if args.candidate_topk < 1:
        raise ValueError("candidate-topk must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    train_groups_all = load_prompt_groups(
        args.train_canonical,
        split=args.train_split,
        max_prompts=args.max_train_prompts,
    )
    if args.train_blocks_per_prompt == 0:
        train_groups = train_groups_all
    else:
        train_groups = [
            select_even_prompt_blocks(records, count=args.train_blocks_per_prompt)
            for records in train_groups_all
        ]
    eval_groups = load_prompt_groups(
        args.eval_canonical,
        split=args.eval_split,
        max_prompts=args.max_eval_prompts,
    )
    released_cache = load_released_cache(
        args.eval_domino_cache, split=args.eval_split, full_b16=args.full_b16
    )
    eval_keys = {
        (record.sample_id, record.anchor_offset)
        for group in eval_groups
        for record in group
    }
    if not eval_keys.issubset(released_cache):
        raise ValueError("released cache is missing evaluation anchors")

    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    for model in (target, domino):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    injected: tuple[str, ...] = ()
    backbone_trainable_count = 0
    if args.train_full_backbone:
        for module_name in ("layers", "norm", "fc", "hidden_norm"):
            module = getattr(domino, module_name)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                backbone_trainable_count += parameter.numel()
        lora_trainable_count = 0
    else:
        injected = inject_fbpf_lora(domino, training_seed=args.seed)
        lora_trainable_count = count_lora_parameters(domino)
        if lora_trainable_count != FBPF_EXPECTED_TRAINABLE_PARAMETERS:
            raise RuntimeError(
                f"unexpected LoRA parameter count {lora_trainable_count}"
            )
    causal_head_count = 0
    if args.train_causal_head:
        for module_name in ("prefix_gru", "embed_proj"):
            module = getattr(domino, module_name)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                causal_head_count += parameter.numel()
    named_trainable = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in domino.named_parameters()
                if parameter.requires_grad
            ),
            key=lambda item: item[0],
        )
    )
    trainable_count = sum(parameter.numel() for _, parameter in named_trainable)
    live_trainable_count = sum(
        parameter.numel()
        for model in (target, domino)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    expected_trainable_count = (
        backbone_trainable_count + lora_trainable_count + causal_head_count
    )
    if live_trainable_count != expected_trainable_count:
        raise RuntimeError(
            "parameters outside the requested adaptation scope are trainable: "
            f"live={live_trainable_count}, expected={expected_trainable_count}"
        )
    if trainable_count != expected_trainable_count:
        raise RuntimeError("named trainable parameter accounting is inconsistent")
    trainable = [parameter for _, parameter in named_trainable]
    target_weight = target.lm_head.weight.detach()

    baseline = evaluate(
        target=target,
        domino=domino,
        target_weight=target_weight,
        prompt_groups=eval_groups,
        released_cache=released_cache,
        full_b16=args.full_b16,
    )
    if baseline["released_length_mismatches"]:
        raise RuntimeError(
            "zero-LoRA variable-context evaluator changed released Domino "
            f"lengths on {baseline['released_length_mismatches']} anchors"
        )
    if baseline["released_token_mismatches"]:
        raise RuntimeError(
            "zero-LoRA variable-context evaluator changed the released Domino "
            f"token trace at {baseline['released_token_mismatches']} positions"
        )
    baseline_eal = baseline["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    print(json.dumps({"baseline": baseline}, indent=2), flush=True)

    initial_eval = baseline
    if args.initial_adaptation is not None:
        payload = torch.load(
            args.initial_adaptation, map_location="cpu", weights_only=False
        )
        if "adaptation_state_dict" not in payload:
            raise ValueError("initial adaptation has no adaptation_state_dict")
        load_trainable_state(
            named_trainable, payload["adaptation_state_dict"]
        )
        initial_eval = evaluate(
            target=target,
            domino=domino,
            target_weight=target_weight,
            prompt_groups=eval_groups,
            released_cache=released_cache,
            full_b16=args.full_b16,
        )
        print(json.dumps({"initial_adaptation": initial_eval}, indent=2), flush=True)

    master_optimizer: MasterAdamW | None = None
    optimizer: torch.optim.AdamW | None = None
    if args.train_causal_head or args.train_full_backbone:
        master_optimizer = MasterAdamW(
            named_trainable, learning_rate=args.peak_learning_rate
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.peak_learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

    steps_per_epoch = math.ceil(
        len(train_groups) / args.gradient_accumulation_prompts
    )
    total_steps = args.epochs * steps_per_epoch
    best_state = clone_trainable_state(named_trainable)
    best_step = 0
    best_eval = initial_eval
    history: list[dict[str, Any]] = []
    global_step = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generator = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_groups)))
        generator.shuffle(order)
        domino.train()
        for order_position, group_index in enumerate(order):
            micro_index = order_position % args.gradient_accumulation_prompts
            if micro_index == 0:
                accumulation_target = min(
                    args.gradient_accumulation_prompts,
                    len(order) - order_position,
                )
                if master_optimizer is not None:
                    master_optimizer.zero_grad()
                else:
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
            records = train_groups[group_index]
            inputs = materialize_prompt_inputs(
                target=target,
                domino=domino,
                records=records,
                include_target_teacher=args.objective
                in {
                    "target_distill",
                    "target_frontier_distill",
                    "target_full_vocab_distill",
                },
                full_b16=args.full_b16,
            )
            released_hidden_rows: list[torch.Tensor] | None = None
            if args.objective == "greedy_reachable" and args.feature_anchor_weight:
                with torch.no_grad(), lora_disabled(domino):
                    released_hidden_rows = parallel_hidden_rows(domino, inputs)
            hidden_rows = parallel_hidden_rows(domino, inputs)
            if args.objective == "dpace":
                logits = torch.cat(
                    [
                        domino_teacher_logits(
                            domino=domino,
                            target_weight=target_weight,
                            anchors=inputs["anchors"][row : row + 1],
                            gold=inputs["gold"][row : row + 1],
                            hidden=full_hidden,
                            train_causal_head=args.train_causal_head,
                        )
                        for row, full_hidden in enumerate(hidden_rows)
                    ],
                    dim=0,
                )
                loss = dpace_loss(
                    logits,
                    inputs["gold"],
                    reduction_divisor=int(inputs["gold"].shape[0]),
                )
                loss_parts: dict[str, float] = {"dpace": float(loss.detach())}
            elif args.objective == "greedy_reachable":
                final_and_base = [
                    domino_teacher_and_base_logits(
                        domino=domino,
                        target_weight=target_weight,
                        anchors=inputs["anchors"][row : row + 1],
                        gold=inputs["gold"][row : row + 1],
                        hidden=full_hidden,
                        train_causal_head=args.train_causal_head,
                    )
                    for row, full_hidden in enumerate(hidden_rows)
                ]
                final_logits = torch.cat(
                    [item[0] for item in final_and_base], dim=0
                )
                base_logits = torch.cat([item[1] for item in final_and_base], dim=0)
                loss, tensors = greedy_reachable_joint_loss(
                    final_logits, base_logits, inputs["gold"]
                )
                feature_anchor = loss.new_zeros(())
                if released_hidden_rows is not None:
                    feature_anchor = torch.stack(
                        [
                            F.mse_loss(adapted.float(), released.float())
                            for adapted, released in zip(
                                hidden_rows, released_hidden_rows, strict=True
                            )
                        ]
                    ).mean()
                    loss = loss + args.feature_anchor_weight * feature_anchor
                loss_parts = {
                    name: float(value) for name, value in tensors.items()
                }
                loss_parts["feature_anchor_mse"] = float(feature_anchor.detach())
            elif args.objective == "frontier_margin":
                final_logits = torch.cat(
                    [
                        domino_teacher_logits(
                            domino=domino,
                            target_weight=target_weight,
                            anchors=inputs["anchors"][row : row + 1],
                            gold=inputs["gold"][row : row + 1],
                            hidden=full_hidden,
                            train_causal_head=args.train_causal_head,
                        )
                        for row, full_hidden in enumerate(hidden_rows)
                    ],
                    dim=0,
                )
                loss, tensors = frontier_margin_joint_loss(
                    final_logits,
                    inputs["gold"],
                    protection_margin=args.protection_margin,
                    protection_weight=args.protection_weight,
                )
                loss_parts = {
                    name: float(value) for name, value in tensors.items()
                }
            elif args.objective in {
                "target_distill",
                "target_frontier_distill",
                "target_full_vocab_distill",
            }:
                final_and_base = [
                    domino_teacher_and_base_logits(
                        domino=domino,
                        target_weight=target_weight,
                        anchors=inputs["anchors"][row : row + 1],
                        gold=inputs["gold"][row : row + 1],
                        hidden=full_hidden,
                        train_causal_head=args.train_causal_head,
                    )
                    for row, full_hidden in enumerate(hidden_rows)
                ]
                final_logits = torch.cat(
                    [item[0] for item in final_and_base], dim=0
                )
                base_logits = torch.cat(
                    [item[1] for item in final_and_base], dim=0
                )
                with torch.no_grad():
                    target_logits = F.linear(
                        inputs["target_teacher_hidden"], target_weight
                    ).float()
                if args.objective == "target_distill":
                    loss, tensors = target_distilled_union_joint_loss(
                        final_logits,
                        base_logits,
                        target_logits,
                        inputs["gold"],
                        topk=args.candidate_topk,
                        temperature=args.target_temperature,
                        protect_weight=args.target_protect_weight,
                        repair_weight=args.target_repair_weight,
                    )
                elif args.objective == "target_frontier_distill":
                    loss, tensors = target_frontier_distilled_union_joint_loss(
                        final_logits,
                        base_logits,
                        target_logits,
                        inputs["gold"],
                        topk=args.candidate_topk,
                        temperature=args.target_temperature,
                        protection_margin=args.target_protection_margin,
                        protection_weight=args.target_protect_weight,
                        repair_weight=args.target_repair_weight,
                    )
                else:
                    loss, tensors = target_full_vocab_distilled_joint_loss(
                        final_logits,
                        base_logits,
                        target_logits,
                        inputs["gold"],
                        topk=args.candidate_topk,
                        temperature=args.target_temperature,
                        protection_margin=args.target_protection_margin,
                        protection_weight=args.target_protect_weight,
                        repair_weight=args.target_repair_weight,
                        future_weight=args.target_future_weight,
                    )
                loss_parts = {
                    name: float(value) for name, value in tensors.items()
                }
            elif args.objective == "domino_curriculum":
                final_and_base = [
                    domino_teacher_and_base_logits(
                        domino=domino,
                        target_weight=target_weight,
                        anchors=inputs["anchors"][row : row + 1],
                        gold=inputs["gold"][row : row + 1],
                        hidden=full_hidden,
                        train_causal_head=args.train_causal_head,
                    )
                    for row, full_hidden in enumerate(hidden_rows)
                ]
                final_logits = torch.cat([item[0] for item in final_and_base], dim=0)
                base_logits = torch.cat([item[1] for item in final_and_base], dim=0)
                base_loss = dpace_loss(
                    base_logits,
                    inputs["gold"],
                    reduction_divisor=int(inputs["gold"].shape[0]),
                )
                final_loss = dpace_loss(
                    final_logits,
                    inputs["gold"],
                    reduction_divisor=int(inputs["gold"].shape[0]),
                )
                transition_steps = max(
                    1, int(round(total_steps * args.curriculum_transition_ratio))
                )
                curriculum_progress = min(
                    1.0,
                    global_step / float(max(transition_steps - 1, 1)),
                )
                curriculum_base_weight = (
                    1.0
                    - curriculum_progress
                    * (1.0 - args.curriculum_final_base_weight)
                )
                loss = (
                    curriculum_base_weight * base_loss
                    + (1.0 - curriculum_base_weight) * final_loss
                )
                loss_parts = {
                    "base_dpace": float(base_loss.detach()),
                    "final_dpace": float(final_loss.detach()),
                    "curriculum_base_weight": curriculum_base_weight,
                }
            else:
                final_and_base = [
                    domino_teacher_and_base_logits(
                        domino=domino,
                        target_weight=target_weight,
                        anchors=inputs["anchors"][row : row + 1],
                        gold=inputs["gold"][row : row + 1],
                        hidden=full_hidden,
                        train_causal_head=args.train_causal_head,
                    )
                    for row, full_hidden in enumerate(hidden_rows)
                ]
                final_logits = torch.cat([item[0] for item in final_and_base], dim=0)
                base_logits = torch.cat([item[1] for item in final_and_base], dim=0)
                base_loss = dpace_loss(
                    base_logits,
                    inputs["gold"],
                    reduction_divisor=int(inputs["gold"].shape[0]),
                )
                if args.objective == "topk_curriculum":
                    candidate_loss, candidate_parts = union_topk_reachable_joint_loss(
                        final_logits,
                        base_logits,
                        inputs["gold"],
                        topk=args.candidate_topk,
                    )
                elif args.objective == "topk_oracle_curriculum":
                    candidate_loss, candidate_parts = (
                        union_topk_oracle_prefix_joint_loss(
                            final_logits,
                            base_logits,
                            inputs["gold"],
                            topk=args.candidate_topk,
                        )
                    )
                else:
                    candidate_loss, candidate_parts = (
                        union_topk_frontier_protected_joint_loss(
                            final_logits,
                            base_logits,
                            inputs["gold"],
                            topk=args.candidate_topk,
                            protection_margin=args.protection_margin,
                            protection_weight=args.protection_weight,
                        )
                    )
                transition_steps = max(
                    1, int(round(total_steps * args.curriculum_transition_ratio))
                )
                curriculum_progress = min(
                    1.0,
                    global_step / float(max(transition_steps - 1, 1)),
                )
                curriculum_base_weight = (
                    1.0
                    - curriculum_progress
                    * (1.0 - args.curriculum_final_base_weight)
                )
                loss = (
                    curriculum_base_weight * base_loss
                    + (1.0 - curriculum_base_weight) * candidate_loss
                )
                loss_parts = {
                    "base_dpace": float(base_loss.detach()),
                    "curriculum_base_weight": curriculum_base_weight,
                    **{
                        name: float(value)
                        for name, value in candidate_parts.items()
                    },
                }
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            (loss / float(accumulation_target)).backward()
            if micro_index + 1 < accumulation_target:
                continue
            learning_rate = cosine_warmup_learning_rate(
                global_step,
                total_steps=total_steps,
                peak=args.peak_learning_rate,
                warmup_ratio=args.warmup_ratio,
            )
            if master_optimizer is not None:
                master_optimizer.set_learning_rate(learning_rate)
                grad_norm = master_optimizer.step(1.0)
            else:
                assert optimizer is not None
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
                optimizer.step()
            global_step += 1
            if global_step % 50 == 0:
                print(
                    f"step={global_step}/{total_steps} epoch={epoch} "
                    f"loss={float(loss.detach()):.6f} "
                    f"parts={json.dumps(loss_parts, sort_keys=True)} "
                    f"grad_norm={grad_norm:.4f} lr={learning_rate:.3e}",
                    flush=True,
                )
            if (
                args.eval_every_steps > 0
                and global_step % args.eval_every_steps == 0
            ):
                current = evaluate(
                    target=target,
                    domino=domino,
                    target_weight=target_weight,
                    prompt_groups=eval_groups,
                    released_cache=released_cache,
                    full_b16=args.full_b16,
                )
                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": float(loss.detach()),
                    "train_loss_parts": loss_parts,
                    "grad_norm": grad_norm,
                    "validation": current,
                }
                history.append(record)
                print(json.dumps({"step_validation": record}, indent=2), flush=True)
                current_eal = current["overall"][
                    "mean_accepted_draft_tokens_prompt_balanced"
                ]
                best_eal = best_eval["overall"][
                    "mean_accepted_draft_tokens_prompt_balanced"
                ]
                if current_eal > best_eal:
                    best_state = clone_trainable_state(named_trainable)
                    best_step = global_step
                    best_eval = current
                domino.train()

    if not history or history[-1]["global_step"] != global_step:
        current = evaluate(
            target=target,
            domino=domino,
            target_weight=target_weight,
            prompt_groups=eval_groups,
            released_cache=released_cache,
            full_b16=args.full_b16,
        )
        history.append(
            {
                "epoch": args.epochs,
                "global_step": global_step,
                "train_loss_parts": loss_parts,
                "grad_norm": grad_norm,
                "validation": current,
            }
        )
        current_eal = current["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]
        best_eal = best_eval["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]
        if current_eal > best_eal:
            best_state = clone_trainable_state(named_trainable)
            best_step = global_step
            best_eval = current

    load_trainable_state(named_trainable, best_state)
    selected = evaluate(
        target=target,
        domino=domino,
        target_weight=target_weight,
        prompt_groups=eval_groups,
        released_cache=released_cache,
        full_b16=args.full_b16,
    )
    checkpoint = args.output / "best_adaptation.pt"
    lora_names = {name for name, _ in named_lora_parameters(domino)}
    torch.save(
        {
            "adaptation_state_dict": best_state,
            "lora_state_dict": {
                name: tensor for name, tensor in best_state.items() if name in lora_names
            },
            "best_step": best_step,
            "injected_modules": list(injected),
            "train_causal_head": args.train_causal_head,
            "train_full_backbone": args.train_full_backbone,
            "full_b16": args.full_b16,
        },
        checkpoint,
    )
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "seed": args.seed,
        "train_prompts": len(train_groups),
        "train_blocks": sum(len(group) for group in train_groups),
        "train_blocks_per_prompt": args.train_blocks_per_prompt,
        "gradient_accumulation_prompts": args.gradient_accumulation_prompts,
        "optimizer_steps": global_step,
        "curriculum_final_base_weight": args.curriculum_final_base_weight,
        "curriculum_transition_ratio": args.curriculum_transition_ratio,
        "eval_prompts": len(eval_groups),
        "eval_blocks": sum(len(group) for group in eval_groups),
        "trainable_parameters": trainable_count,
        "lora_trainable_parameters": lora_trainable_count,
        "backbone_trainable_parameters": backbone_trainable_count,
        "causal_head_trainable_parameters": causal_head_count,
        "train_causal_head": args.train_causal_head,
        "train_full_backbone": args.train_full_backbone,
        "full_b16": args.full_b16,
        "attention_implementation": args.attn_implementation,
        "initial_adaptation": (
            None
            if args.initial_adaptation is None
            else str(args.initial_adaptation.resolve())
        ),
        "initial_eal": initial_eval["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "objective": args.objective,
        "candidate_topk": args.candidate_topk,
        "feature_anchor_weight": args.feature_anchor_weight,
        "protection_margin": args.protection_margin,
        "protection_weight": args.protection_weight,
        "target_temperature": args.target_temperature,
        "target_protect_weight": args.target_protect_weight,
        "target_repair_weight": args.target_repair_weight,
        "target_future_weight": args.target_future_weight,
        "target_protection_margin": args.target_protection_margin,
        "draft_forward_mode": "individual_exact_context",
        "baseline_eal": baseline_eal,
        "best_step": best_step,
        "history": history,
        "selected": selected,
        "selected_delta_vs_released": selected["overall"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]
        - baseline_eal,
        "seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated()
        / float(1024**3),
        "checkpoint": str(checkpoint.resolve()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
