#!/usr/bin/env python3
"""Collect Domino base candidates and on-policy/teacher-forced GRU traces."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from collect_canonical_blocks import (
    ShardWriter,
    evenly_spaced_offsets,
    extract_context_feature,
    package_version,
    read_manifest,
    validate_output_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--anchors-per-sample", type=int, default=4)
    parser.add_argument("--continuation-tokens", type=int, default=128)
    parser.add_argument("--shard-blocks", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def correction_logits(
    draft: Any,
    parallel_hidden: torch.Tensor,
    state: torch.Tensor,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    state_for_head = state.transpose(0, 1)
    if bool(getattr(draft, "use_bias_norm", False)):
        state_for_head = draft.bias_norm(state_for_head)
    bias = draft.embed_proj(torch.cat([parallel_hidden, state_for_head], dim=-1))
    if bool(getattr(draft, "use_bias_gate", False)) and hasattr(draft, "bias_gate"):
        bias = torch.sigmoid(draft.bias_gate(parallel_hidden)) * bias
    return base_logits + bias


@torch.inference_mode()
def run_domino_paths(
    *,
    draft: Any,
    target: Any,
    anchor_token: torch.Tensor,
    gold_ids: torch.Tensor,
    parallel_hidden: torch.Tensor,
    base_logits: torch.Tensor,
    top_k: int,
) -> dict[str, torch.Tensor]:
    positions = base_logits.shape[1]
    if int(getattr(draft, "pure_draft_prefix_len", 0)) != 1:
        raise ValueError("diagnostic currently expects pure_draft_prefix_len=1")

    onpolicy_ids = torch.empty((1, positions), dtype=torch.long, device=target.device)
    teacher_ids = torch.empty_like(onpolicy_ids)
    onpolicy_topk_ids = torch.empty(
        (1, positions, top_k), dtype=torch.long, device=target.device
    )
    teacher_topk_ids = torch.empty_like(onpolicy_topk_ids)
    onpolicy_topk_logits = torch.empty(
        (1, positions, top_k), dtype=torch.float32, device=target.device
    )
    teacher_topk_logits = torch.empty_like(onpolicy_topk_logits)
    state_distance = torch.zeros((1, positions), dtype=torch.float32, device=target.device)
    onpolicy_state_norm = torch.zeros_like(state_distance)
    teacher_state_norm = torch.zeros_like(state_distance)

    first_logits = base_logits[:, :1].float()
    first_token = first_logits.argmax(dim=-1)
    first_values, first_indices = torch.topk(first_logits, k=top_k, dim=-1)
    onpolicy_ids[:, 0] = first_token[:, 0]
    teacher_ids[:, 0] = first_token[:, 0]
    onpolicy_topk_ids[:, 0] = first_indices[:, 0]
    teacher_topk_ids[:, 0] = first_indices[:, 0]
    onpolicy_topk_logits[:, 0] = first_values[:, 0]
    teacher_topk_logits[:, 0] = first_values[:, 0]

    onpolicy_prefix = torch.cat([anchor_token.view(1, 1), first_token], dim=1)
    teacher_prefix = torch.cat(
        [anchor_token.view(1, 1), gold_ids[:, :1]], dim=1
    )
    _, onpolicy_state = draft.prefix_gru(target.model.embed_tokens(onpolicy_prefix))
    _, teacher_state = draft.prefix_gru(target.model.embed_tokens(teacher_prefix))

    for position in range(1, positions):
        state_distance[:, position] = torch.linalg.vector_norm(
            (onpolicy_state - teacher_state).float(), dim=-1
        )[0]
        onpolicy_state_norm[:, position] = torch.linalg.vector_norm(
            onpolicy_state.float(), dim=-1
        )[0]
        teacher_state_norm[:, position] = torch.linalg.vector_norm(
            teacher_state.float(), dim=-1
        )[0]
        z_i = parallel_hidden[:, position : position + 1]
        base_i = base_logits[:, position : position + 1]
        on_logits = correction_logits(draft, z_i, onpolicy_state, base_i).float()
        teacher_logits = correction_logits(draft, z_i, teacher_state, base_i).float()
        on_values, on_indices = torch.topk(on_logits, k=top_k, dim=-1)
        teacher_values, teacher_indices = torch.topk(
            teacher_logits, k=top_k, dim=-1
        )
        on_token = on_indices[..., 0]
        teacher_token = teacher_indices[..., 0]
        onpolicy_ids[:, position] = on_token[:, 0]
        teacher_ids[:, position] = teacher_token[:, 0]
        onpolicy_topk_ids[:, position] = on_indices[:, 0]
        teacher_topk_ids[:, position] = teacher_indices[:, 0]
        onpolicy_topk_logits[:, position] = on_values[:, 0]
        teacher_topk_logits[:, position] = teacher_values[:, 0]
        if position + 1 < positions:
            _, onpolicy_state = draft.prefix_gru(
                target.model.embed_tokens(on_token), onpolicy_state
            )
            _, teacher_state = draft.prefix_gru(
                target.model.embed_tokens(gold_ids[:, position : position + 1]),
                teacher_state,
            )

    return {
        "onpolicy_ids": onpolicy_ids,
        "teacher_ids": teacher_ids,
        "onpolicy_topk_ids": onpolicy_topk_ids,
        "teacher_topk_ids": teacher_topk_ids,
        "onpolicy_topk_logits": onpolicy_topk_logits,
        "teacher_topk_logits": teacher_topk_logits,
        "state_distance": state_distance,
        "onpolicy_state_norm": onpolicy_state_norm,
        "teacher_state_norm": teacher_state_norm,
    }


@torch.inference_mode()
def collect_sample(
    *,
    sample: dict[str, Any],
    target: Any,
    draft: Any,
    tokenizer: Any,
    block_size: int,
    top_k: int,
    anchors_per_sample: int,
    continuation_tokens: int,
) -> list[dict[str, Any]]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(text, return_tensors="pt").to(target.device)
    input_ids = encoded.input_ids
    prompt_tokens = int(input_ids.shape[1])
    sequence = target.generate(
        input_ids,
        attention_mask=encoded.attention_mask,
        max_new_tokens=continuation_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    continuation = sequence[0, prompt_tokens:]
    draft_positions = block_size
    maximum_anchor = int(continuation.numel()) - draft_positions - 1
    offsets = evenly_spaced_offsets(maximum_anchor, anchors_per_sample)
    if not offsets:
        return []

    longest_context = prompt_tokens + offsets[-1]
    target_outputs = target.model(
        sequence[:, :longest_context],
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    context_features = extract_context_feature(
        target_outputs.hidden_states, list(draft.target_layer_ids)
    )
    del target_outputs

    records = []
    for anchor_offset in offsets:
        context_length = prompt_tokens + anchor_offset
        anchor_token = continuation[anchor_offset].to(torch.long)
        gold_ids = continuation[
            anchor_offset + 1 : anchor_offset + 1 + draft_positions
        ].view(1, -1).to(torch.long)
        block_ids = torch.full(
            (1, block_size),
            int(draft.mask_token_id),
            dtype=torch.long,
            device=target.device,
        )
        block_ids[0, 0] = anchor_token
        position_ids = torch.arange(
            context_length + block_size, device=target.device
        ).unsqueeze(0)
        parallel_hidden = draft(
            target_hidden=context_features[:, :context_length],
            noise_embedding=target.model.embed_tokens(block_ids),
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        base_logits = target.lm_head(parallel_hidden)
        base_topk_logits, base_topk_ids = torch.topk(
            base_logits.float(), k=top_k, dim=-1
        )
        base_logsumexp = torch.logsumexp(base_logits.float(), dim=-1)
        paths = run_domino_paths(
            draft=draft,
            target=target,
            anchor_token=anchor_token,
            gold_ids=gold_ids,
            parallel_hidden=parallel_hidden,
            base_logits=base_logits,
            top_k=top_k,
        )
        base_matches = base_topk_ids[..., 0] == gold_ids
        onpolicy_matches = paths["onpolicy_ids"] == gold_ids
        teacher_matches = paths["teacher_ids"] == gold_ids
        records.append(
            {
                "sample_id": sample["sample_id"],
                "domain": sample["domain"],
                "source": sample["source"],
                "split": sample["split"],
                "prompt_token_count": prompt_tokens,
                "anchor_offset": anchor_offset,
                "anchor_token_id": int(anchor_token.item()),
                "gold_ids": gold_ids[0].cpu().to(torch.int32),
                "parallel_hidden": parallel_hidden[0].cpu().to(torch.bfloat16),
                "base_topk_ids": base_topk_ids[0].cpu().to(torch.int32),
                "base_topk_logits": base_topk_logits[0].cpu().to(torch.float16),
                "base_logsumexp": base_logsumexp[0].cpu().to(torch.float32),
                "base_top1_match": base_matches[0].cpu(),
                "onpolicy_ids": paths["onpolicy_ids"][0].cpu().to(torch.int32),
                "teacher_ids": paths["teacher_ids"][0].cpu().to(torch.int32),
                "onpolicy_match": onpolicy_matches[0].cpu(),
                "teacher_match": teacher_matches[0].cpu(),
                "onpolicy_topk_ids": paths["onpolicy_topk_ids"][0]
                .cpu()
                .to(torch.int32),
                "teacher_topk_ids": paths["teacher_topk_ids"][0]
                .cpu()
                .to(torch.int32),
                "state_distance": paths["state_distance"][0].cpu(),
                "onpolicy_state_norm": paths["onpolicy_state_norm"][0].cpu(),
                "teacher_state_norm": paths["teacher_state_norm"][0].cpu(),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Domino canonical collection requires CUDA")
    samples = read_manifest(args.manifest, args.max_samples)
    validate_output_directory(args.output)
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(str(args.target), local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    draft = AutoModel.from_pretrained(
        str(args.draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    config = getattr(draft.config, "dflash_config", {})
    if config.get("projector_type") not in {"domino", "causal_v5"}:
        raise ValueError("a Domino checkpoint is required")
    if not bool(config.get("shift_label", False)):
        raise ValueError("diagnostic expects a shift_label Domino checkpoint")
    if int(draft.block_size) != args.block_size:
        raise ValueError("block-size mismatch")

    metadata = {
        "format_version": 1,
        "collector": "domino-canonical",
        "created_unix": time.time(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "manifest": str(args.manifest.resolve()),
        "block_size": args.block_size,
        "draft_positions": args.block_size,
        "top_k": args.top_k,
        "anchors_per_sample": args.anchors_per_sample,
        "continuation_tokens": args.continuation_tokens,
        "num_manifest_samples": len(samples),
        "target_layer_ids": list(draft.target_layer_ids),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    writer = ShardWriter(args.output, args.shard_blocks)
    start = time.perf_counter()
    for index, sample in enumerate(samples):
        records = collect_sample(
            sample=sample,
            target=target,
            draft=draft,
            tokenizer=tokenizer,
            block_size=args.block_size,
            top_k=args.top_k,
            anchors_per_sample=args.anchors_per_sample,
            continuation_tokens=args.continuation_tokens,
        )
        for record in records:
            writer.add(record)
        print(
            f"[{index + 1}/{len(samples)}] {sample['sample_id']}: {len(records)} blocks",
            flush=True,
        )
    writer.flush()
    metadata["num_blocks"] = writer.total_blocks
    metadata["collection_seconds"] = time.perf_counter() - start
    metadata["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    temporary = args.output / f"metadata.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output / "metadata.json")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
