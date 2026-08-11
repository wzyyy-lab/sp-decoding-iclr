#!/usr/bin/env python3
"""Evaluate a DFlash/LoRA checkpoint on the complete B16 acceptance metric."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM

from sph.domino_joint_runtime import (
    CanonicalBlock,
    acceptance_lengths,
    domino_onpolicy_ids,
)
from sph.fbpf import inject_fbpf_lora, lora_disabled
from sph.gfpr import paired_prompt_summary
from train_domino_backbone_lora import (
    extract_context_feature,
    load_prompt_groups,
    load_trainable_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--adaptation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--expected-released-eal", type=float, default=7.23955296404276
    )
    parser.add_argument("--baseline-tolerance", type=float, default=1e-9)
    return parser.parse_args()


@torch.no_grad()
def reconstruct_target_prompt(
    *,
    target: nn.Module,
    domino: nn.Module,
    records: Sequence[CanonicalBlock],
) -> tuple[torch.Tensor, torch.Tensor]:
    longest_record = max(records, key=lambda row: int(row.context_ids.numel()))
    longest = longest_record.context_ids.to(torch.long)
    for record in records:
        context_length = int(record.context_ids.numel())
        if not torch.equal(longest[:context_length], record.context_ids.long()):
            raise ValueError("prompt contexts are not prefix nested")
    sequence = torch.cat(
        [
            longest,
            torch.tensor([longest_record.anchor_token_id], dtype=torch.long),
            longest_record.gold_ids.to(torch.long),
        ]
    ).unsqueeze(0).to("cuda:0", non_blocking=True)
    outputs = target.model(
        sequence,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    context_features = extract_context_feature(
        outputs.hidden_states, list(domino.target_layer_ids)
    ).detach()
    next_token = target.lm_head(outputs.last_hidden_state[:, -1:]).argmax(
        dim=-1
    )
    extended = torch.cat([sequence, next_token], dim=-1)
    del outputs
    return context_features, extended


@torch.no_grad()
def draft_one(
    *,
    domino: nn.Module,
    target_weight: torch.Tensor,
    target_features: torch.Tensor,
    context_length: int,
    anchor: torch.Tensor,
) -> torch.Tensor:
    block_size = int(domino.block_size)
    block_ids = torch.full(
        (1, block_size),
        int(domino.mask_token_id),
        dtype=torch.long,
        device=target_features.device,
    )
    block_ids[:, 0] = anchor
    position_ids = torch.arange(
        context_length + block_size,
        dtype=torch.long,
        device=target_features.device,
    ).unsqueeze(0)
    hidden = domino(
        target_hidden=target_features[:, :context_length],
        noise_embedding=F.embedding(block_ids, target_weight),
        position_ids=position_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
    )
    return domino_onpolicy_ids(
        domino=domino,
        target_weight=target_weight,
        anchors=anchor,
        hidden=hidden,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full-B16 LoRA evaluation requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.bootstrap_samples < 1 or args.baseline_tolerance < 0:
        raise ValueError("invalid evaluation tolerance/bootstrap count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)

    groups = load_prompt_groups(
        args.canonical, split=args.split, max_prompts=args.max_prompts
    )
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
    payload = torch.load(args.adaptation, map_location="cpu", weights_only=False)
    train_full_backbone = bool(payload.get("train_full_backbone", False))
    if train_full_backbone:
        injected: tuple[str, ...] = ()
        for module_name in ("layers", "norm", "fc", "hidden_norm"):
            for parameter in getattr(domino, module_name).parameters():
                parameter.requires_grad_(True)
    else:
        injected = inject_fbpf_lora(domino, training_seed=args.seed)
    train_causal_head = bool(payload.get("train_causal_head", False))
    released_domino: nn.Module | None = None
    if train_causal_head:
        for module_name in ("prefix_gru", "embed_proj"):
            for parameter in getattr(domino, module_name).parameters():
                parameter.requires_grad_(True)
    if train_full_backbone or train_causal_head:
        # LoRA can be disabled in-place, but a jointly adapted causal head
        # or backbone needs an untouched released reference for the baseline.
        released_domino = AutoModel.from_pretrained(
            str(args.domino_draft),
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            device_map="cuda:0",
        ).eval()
        for parameter in released_domino.parameters():
            parameter.requires_grad_(False)
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
    state = payload.get("adaptation_state_dict")
    if not isinstance(state, dict):
        raise ValueError("adaptation checkpoint lacks adaptation_state_dict")
    load_trainable_state(named_trainable, state)
    target_weight = target.lm_head.weight.detach()

    sample_ids: list[str] = []
    domains: list[str] = []
    baseline_lengths: list[int] = []
    current_lengths: list[int] = []
    started = time.perf_counter()
    for prompt_index, records in enumerate(groups):
        target_features, extended = reconstruct_target_prompt(
            target=target, domino=domino, records=records
        )
        for record in records:
            context_length = int(record.context_ids.numel())
            anchor = extended[:, context_length]
            gold = extended[:, context_length + 1 : context_length + 17]
            if gold.shape != (1, 16):
                raise RuntimeError("reconstructed prompt lacks a full B16 label")
            if int(anchor.item()) != record.anchor_token_id:
                raise RuntimeError("reconstructed anchor differs from canonical data")
            if not torch.equal(
                gold[:, :15].cpu(), record.gold_ids.long().view(1, -1)
            ):
                raise RuntimeError("reconstructed gold prefix differs from canonical data")
            if released_domino is not None:
                baseline_ids = draft_one(
                    domino=released_domino,
                    target_weight=target_weight,
                    target_features=target_features,
                    context_length=context_length,
                    anchor=anchor,
                )
            else:
                with lora_disabled(domino):
                    baseline_ids = draft_one(
                        domino=domino,
                        target_weight=target_weight,
                        target_features=target_features,
                        context_length=context_length,
                        anchor=anchor,
                    )
            current_ids = draft_one(
                domino=domino,
                target_weight=target_weight,
                target_features=target_features,
                context_length=context_length,
                anchor=anchor,
            )
            sample_ids.append(record.sample_id)
            domains.append(record.domain)
            baseline_lengths.append(int(acceptance_lengths(baseline_ids, gold)[0]))
            current_lengths.append(int(acceptance_lengths(current_ids, gold)[0]))
        if (prompt_index + 1) % 25 == 0:
            print(
                f"prompts={prompt_index + 1}/{len(groups)} blocks={len(sample_ids)}",
                flush=True,
            )

    overall = paired_prompt_summary(
        sample_ids,
        baseline_lengths,
        current_lengths,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    by_domain: dict[str, dict[str, float | int | list[float]]] = {}
    for domain in sorted(set(domains)):
        indices = [i for i, value in enumerate(domains) if value == domain]
        by_domain[domain] = paired_prompt_summary(
            [sample_ids[i] for i in indices],
            [baseline_lengths[i] for i in indices],
            [current_lengths[i] for i in indices],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    baseline_error = abs(
        float(overall["baseline_eal_prompt_balanced"])
        - args.expected_released_eal
    )
    if baseline_error > args.baseline_tolerance:
        raise RuntimeError(
            "full-B16 released baseline mismatch: "
            f"observed={overall['baseline_eal_prompt_balanced']}, "
            f"expected={args.expected_released_eal}"
        )
    report: dict[str, Any] = {
        "status": "completed",
        "metric": "fixed_full_b16_prompt_balanced_eal",
        "split": args.split,
        "adaptation": str(args.adaptation.resolve()),
        "adaptation_best_step": payload.get("best_step"),
        "train_causal_head": train_causal_head,
        "train_full_backbone": train_full_backbone,
        "injected_modules": list(injected),
        "prompts": len(groups),
        "blocks": len(sample_ids),
        "overall": overall,
        "by_domain": by_domain,
        "expected_released_eal": args.expected_released_eal,
        "baseline_error": baseline_error,
        "target_15_percent_eal": args.expected_released_eal * 1.15,
        "passes_7_8_gate": float(overall["current_eal_prompt_balanced"]) >= 7.8,
        "passes_15_percent_target": float(
            overall["current_eal_prompt_balanced"]
        )
        >= args.expected_released_eal * 1.15,
        "seconds": time.perf_counter() - started,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
