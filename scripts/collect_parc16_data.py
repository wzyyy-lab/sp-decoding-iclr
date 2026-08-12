#!/usr/bin/env python3
"""Materialize full16 train/validation data for joint PARC-DFlash training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from sph.parc import (
    PURE_DFLASH_INPUT_LENGTH,
    greedy_first_topk,
    nonshift_full16_prediction_hidden,
)

try:
    from scripts.collect_canonical_blocks import (
        evenly_spaced_offsets,
        extract_context_feature,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the project root.
    from collect_canonical_blocks import evenly_spaced_offsets, extract_context_feature


BLOCK_LENGTH = 16
CANDIDATES = 16
DEFAULT_CONTINUATION_TOKENS = 129
DEFAULT_ANCHORS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continuation-tokens", type=int, default=DEFAULT_CONTINUATION_TOKENS)
    parser.add_argument("--anchors-per-prompt", type=int, default=DEFAULT_ANCHORS)
    parser.add_argument("--shard-prompts", type=int, default=16)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sample_id = str(record["sample_id"])
                split = str(record["split"])
                domain = str(record["domain"])
                str(record["prompt"])
                int(record["part_index"])
                int(record["part_selection_order"])
                int(record["part_required_usable_count"])
                int(record["part_candidate_count"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise RuntimeError(f"malformed manifest row {line_number}") from error
            if split not in {"train", "validation"}:
                raise RuntimeError(
                    "PARC train-data collector refuses held-out or unknown splits"
                )
            if sample_id in seen:
                raise RuntimeError(f"duplicate prompt {sample_id}")
            seen.add(sample_id)
            records.append(record)
    if not records:
        raise RuntimeError("manifest selected no train/validation prompts")
    records.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["domain"]),
            int(row["part_selection_order"]),
        )
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["split"]), str(record["domain"]))
        grouped.setdefault(key, []).append(record)
    if set(grouped) != {
        (split, domain)
        for split in ("train", "validation")
        for domain in ("chat", "code", "math")
    }:
        raise RuntimeError("PARC part manifest lacks a split/domain reserve group")
    for key, rows in grouped.items():
        quotas = {int(row["part_required_usable_count"]) for row in rows}
        candidates = {int(row["part_candidate_count"]) for row in rows}
        orders = [int(row["part_selection_order"]) for row in rows]
        if len(quotas) != 1 or len(candidates) != 1:
            raise RuntimeError(f"inconsistent reserve receipt for {key}")
        if candidates != {len(rows)} or orders != list(range(len(rows))):
            raise RuntimeError(f"non-contiguous reserve order for {key}")
        if not 0 < next(iter(quotas)) < len(rows):
            raise RuntimeError(f"reserve pool for {key} cannot fill its quota")
    return records


def manifest_group_quotas(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], int]:
    return {
        (str(record["split"]), str(record["domain"])): int(
            record["part_required_usable_count"]
        )
        for record in records
    }


def full16_anchor_offsets(
    continuation_length: int, anchors_per_prompt: int
) -> list[int]:
    if anchors_per_prompt < 1:
        raise ValueError("anchors_per_prompt must be positive")
    maximum_anchor = continuation_length - BLOCK_LENGTH - 1
    return evenly_spaced_offsets(maximum_anchor, anchors_per_prompt)


def accepted_length(proposal: Tensor, gold: Tensor) -> int:
    if proposal.shape != (BLOCK_LENGTH,) or gold.shape != (BLOCK_LENGTH,):
        raise ValueError("proposal and gold must have shape [16]")
    mismatches = proposal.ne(gold).nonzero(as_tuple=False)
    return BLOCK_LENGTH if mismatches.numel() == 0 else int(mismatches[0, 0])


def reference_margin_summary(
    *,
    hidden: Tensor,
    projection_weight: Tensor,
    topk_ids: Tensor,
    topk_logits: Tensor,
    accepted: int,
) -> tuple[float, float]:
    """Return FP32 protected margin and BF16-vs-FP32 margin error."""

    if hidden.shape[0] != BLOCK_LENGTH:
        raise ValueError("reference hidden must contain all 16 positions")
    if topk_ids.shape != (BLOCK_LENGTH, CANDIDATES):
        raise ValueError("topk_ids must have shape [16,16]")
    if topk_logits.shape != (BLOCK_LENGTH, CANDIDATES):
        raise ValueError("topk_logits must have shape [16,16]")
    if not 0 <= accepted <= BLOCK_LENGTH:
        raise ValueError("accepted length lies outside [0,16]")
    if accepted == 0:
        return 0.0, 0.0
    selected_rows = torch.nn.functional.embedding(topk_ids.long(), projection_weight)
    fp32_scores = torch.einsum(
        "ph,pkh->pk", hidden.float(), selected_rows.float()
    )
    bf16_margin = topk_logits[:accepted, 0].float() - topk_logits[
        :accepted, 1:
    ].float().amax(dim=-1)
    fp32_margin = fp32_scores[:accepted, 0] - fp32_scores[
        :accepted, 1:
    ].amax(dim=-1)
    return (
        float(fp32_margin.amin().item()),
        float((bf16_margin - fp32_margin).abs().amax().item()),
    )


class PromptShardWriter:
    def __init__(self, root: Path, prompts_per_shard: int) -> None:
        if prompts_per_shard < 1:
            raise ValueError("prompts_per_shard must be positive")
        self.root = root
        self.prompts_per_shard = prompts_per_shard
        self.buffers: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "validation": [],
        }
        self.shards: list[dict[str, Any]] = []
        self.prompts = 0
        self.blocks = 0

    def add(self, record: dict[str, Any]) -> None:
        split = str(record["split"])
        if split not in self.buffers:
            raise ValueError(f"unsupported materialized split {split!r}")
        self.buffers[split].append(record)
        if len(self.buffers[split]) >= self.prompts_per_shard:
            self.flush(split)

    def flush(self, split: str) -> None:
        buffer = self.buffers[split]
        if not buffer:
            return
        split_index = sum(1 for item in self.shards if item["split"] == split)
        path = self.root / f"{split}-shard-{split_index:05d}.pt"
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        torch.save(buffer, temporary)
        os.replace(temporary, path)
        prompts = len(buffer)
        blocks = sum(len(record["anchors"]) for record in buffer)
        counts = Counter(
            f"{record['domain']}/{record['split']}" for record in buffer
        )
        self.prompts += prompts
        self.blocks += blocks
        self.shards.append(
            {
                "path": path.name,
                "split": split,
                "prompts": prompts,
                "blocks": blocks,
                "bytes": path.stat().st_size,
                "sample_ids": [str(record["sample_id"]) for record in buffer],
                "counts_by_domain_split": dict(sorted(counts.items())),
                "prompt_reference_summaries": [
                    {
                        "sample_id": str(record["sample_id"]),
                        "domain": str(record["domain"]),
                        "numeric_margin_error": max(
                            float(anchor["numeric_margin_error"])
                            for anchor in record["anchors"]
                        ),
                        "reference_deltas": [
                            float(anchor["reference_delta_fp32"])
                            for anchor in record["anchors"]
                        ],
                        "reference_accepted_lengths": [
                            int(anchor["reference_accepted_length"])
                            for anchor in record["anchors"]
                        ],
                        "reference_domino_accepted_lengths": (
                            [
                                int(anchor["reference_domino_accepted_length"])
                                for anchor in record["anchors"]
                            ]
                            if split == "validation"
                            else None
                        ),
                    }
                    for record in buffer
                ],
            }
        )
        self.buffers[split] = []

    def flush_all(self) -> None:
        for split in tuple(self.buffers):
            self.flush(split)


@torch.inference_mode()
def released_domino_proposal(
    *,
    domino: Any,
    target_weight: Tensor,
    hidden: Tensor,
    base_logits: Tensor,
    anchor: Tensor,
) -> Tensor:
    """Run the released Domino serial head only as a fixed comparator."""

    if hidden.shape[1] != BLOCK_LENGTH or base_logits.shape[:2] != (
        hidden.shape[0],
        BLOCK_LENGTH,
    ):
        raise ValueError("Domino comparator expects one full16 hidden sequence")
    first = base_logits[:, 0].float().argmax(dim=-1)
    selected = [first]
    prefix = torch.stack([anchor, first], dim=-1)
    _, state = domino.prefix_gru(F.embedding(prefix, target_weight))
    for position in range(1, BLOCK_LENGTH):
        state_for_head = state.transpose(0, 1)
        if bool(getattr(domino, "use_bias_norm", False)):
            state_for_head = domino.bias_norm(state_for_head)
        joined = torch.cat(
            [hidden[:, position : position + 1], state_for_head], dim=-1
        )
        code = domino.embed_proj[1](domino.embed_proj[0](joined))[:, 0]
        correction = F.linear(code, domino.embed_proj[2].weight)
        token = (base_logits[:, position] + correction).argmax(dim=-1)
        selected.append(token)
        if position + 1 < BLOCK_LENGTH:
            _, state = domino.prefix_gru(
                F.embedding(token[:, None], target_weight), state
            )
    return torch.stack(selected, dim=1)


@torch.inference_mode()
def collect_prompt(
    *,
    sample: dict[str, Any],
    tokenizer: Any,
    target: Any,
    draft: Any,
    domino: Any,
    continuation_tokens: int,
    anchors_per_prompt: int,
) -> dict[str, Any] | None:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(sample["prompt"])}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(text, return_tensors="pt").to(target.device)
    prompt_tokens = int(encoded.input_ids.shape[1])
    sequence = target.generate(
        encoded.input_ids,
        attention_mask=encoded.attention_mask,
        max_new_tokens=continuation_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    continuation = sequence[0, prompt_tokens:]
    if int(continuation.numel()) != continuation_tokens:
        return None
    offsets = full16_anchor_offsets(int(continuation.numel()), anchors_per_prompt)
    if len(offsets) != anchors_per_prompt:
        return None
    longest_context = prompt_tokens + offsets[-1]
    target_outputs = target.model(
        sequence[:, :longest_context],
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    context_features = extract_context_feature(
        target_outputs.hidden_states, list(draft.target_layer_ids)
    )[0].to(torch.bfloat16)
    del target_outputs

    projection_weight = target.lm_head.weight.detach()
    anchors: list[dict[str, Any]] = []
    for anchor_offset in offsets:
        context_length = prompt_tokens + anchor_offset
        anchor_token = continuation[anchor_offset].long()
        gold = continuation[
            anchor_offset + 1 : anchor_offset + 1 + BLOCK_LENGTH
        ].long()
        if gold.shape != (BLOCK_LENGTH,):
            raise RuntimeError("collector produced a non-full16 gold block")
        block_ids = torch.full(
            (1, PURE_DFLASH_INPUT_LENGTH),
            int(draft.mask_token_id),
            dtype=torch.long,
            device=target.device,
        )
        block_ids[0, 0] = anchor_token
        position_ids = torch.arange(
            context_length + PURE_DFLASH_INPUT_LENGTH, device=target.device
        ).unsqueeze(0)
        raw_hidden = draft(
            target_hidden=context_features[None, :context_length],
            noise_embedding=target.model.embed_tokens(block_ids),
            position_ids=position_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        hidden = nonshift_full16_prediction_hidden(raw_hidden)
        if hidden.shape != (1, BLOCK_LENGTH, projection_weight.shape[1]):
            raise RuntimeError(f"pure DFlash returned shape {tuple(hidden.shape)}")
        base_logits = target.lm_head(hidden)
        topk_logits, topk_ids = greedy_first_topk(base_logits, CANDIDATES)
        greedy = base_logits.float().argmax(dim=-1)
        if not bool(torch.isfinite(topk_logits).all().item()):
            raise FloatingPointError("non-finite pure-DFlash candidate logits")
        if not torch.equal(greedy, topk_ids[..., 0]):
            raise RuntimeError("greedy-first Top16 contract lost pure-DFlash argmax")
        if bool(
            (
                topk_ids.unsqueeze(-1)
                == topk_ids.unsqueeze(-2)
            ).triu(diagonal=1).any().item()
        ):
            raise RuntimeError("greedy-first Top16 produced duplicate candidate IDs")
        accepted = accepted_length(greedy[0], gold)
        reference_delta, numeric_error = reference_margin_summary(
            hidden=hidden[0],
            projection_weight=projection_weight,
            topk_ids=topk_ids[0],
            topk_logits=topk_logits[0],
            accepted=accepted,
        )
        anchors.append(
            {
                "anchor_offset": anchor_offset,
                "context_length": context_length,
                "anchor_token_id": int(anchor_token.item()),
                "gold_ids": gold.cpu().to(torch.int32),
                "reference_topk_ids": topk_ids[0].cpu().to(torch.int32),
                "reference_topk_logits": topk_logits[0].cpu().to(torch.float16),
                "reference_proposal_ids": greedy[0].cpu().to(torch.int32),
                "reference_rank0_tie_rows": int(
                    topk_logits[0, :, 0].eq(topk_logits[0, :, 1]).sum().item()
                ),
                "reference_accepted_length": accepted,
                "reference_delta_fp32": reference_delta,
                "numeric_margin_error": numeric_error,
            }
        )
        if str(sample["split"]) == "validation":
            domino_block_ids = torch.full(
                (1, BLOCK_LENGTH),
                int(domino.mask_token_id),
                dtype=torch.long,
                device=target.device,
            )
            domino_block_ids[0, 0] = anchor_token
            domino_position_ids = torch.arange(
                context_length + BLOCK_LENGTH, device=target.device
            ).unsqueeze(0)
            domino_hidden = domino(
                target_hidden=context_features[None, :context_length],
                noise_embedding=target.model.embed_tokens(domino_block_ids),
                position_ids=domino_position_ids,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )
            domino_logits = target.lm_head(domino_hidden)
            domino_proposal = released_domino_proposal(
                domino=domino,
                target_weight=projection_weight,
                hidden=domino_hidden,
                base_logits=domino_logits,
                anchor=anchor_token.view(1),
            )[0]
            anchors[-1]["reference_domino_proposal_ids"] = (
                domino_proposal.cpu().to(torch.int32)
            )
            anchors[-1]["reference_domino_accepted_length"] = accepted_length(
                domino_proposal, gold
            )
            del domino_hidden, domino_logits, domino_proposal
        del raw_hidden, hidden, base_logits, topk_logits, topk_ids, greedy

    return {
        "sample_id": str(sample["sample_id"]),
        "domain": str(sample["domain"]),
        "source": str(sample["source"]),
        "split": str(sample["split"]),
        "prompt_token_count": prompt_tokens,
        "continuation_length": int(continuation.numel()),
        "target_context_features": context_features.cpu(),
        "anchors": anchors,
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("PARC full16 materialization requires CUDA")
    if args.continuation_tokens != DEFAULT_CONTINUATION_TOKENS:
        raise ValueError("claim-bearing PARC collection requires exactly 129 tokens")
    if args.anchors_per_prompt != DEFAULT_ANCHORS:
        raise ValueError("claim-bearing PARC data requires exactly 8 anchors")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    samples = read_manifest(args.manifest)
    quotas = manifest_group_quotas(samples)
    work = args.output.with_name(
        f"{args.output.name}.incomplete-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )
    if work.exists():
        raise FileExistsError(f"refusing to reuse incomplete output {work}")
    work.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir()

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
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    if int(draft.block_size) != BLOCK_LENGTH:
        raise RuntimeError("PARC requires a B16 pure-DFlash checkpoint")
    if getattr(draft.config, "dflash_config", {}).get("projector_type") is not None:
        raise RuntimeError("PARC data must use pure DFlash, not Domino")
    if bool(getattr(draft.config, "dflash_config", {}).get("shift_label", False)):
        raise RuntimeError("raw17/slice full16 requires a non-shift pure DFlash checkpoint")
    if getattr(domino.config, "dflash_config", {}).get("projector_type") != "domino":
        raise RuntimeError("validation comparator is not released Domino")
    if int(domino.block_size) != BLOCK_LENGTH or not bool(
        getattr(domino.config, "dflash_config", {}).get("shift_label", False)
    ):
        raise RuntimeError("released Domino comparator must use shift-label B16")
    for model in (target, draft, domino):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    writer = PromptShardWriter(work, args.shard_prompts)
    counts: Counter[str] = Counter()
    rejected_short = 0
    unused_reserve = 0
    selected_by_group: Counter[tuple[str, str]] = Counter()
    max_numeric_error = 0.0
    rank0_tie_rows = 0
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        group = (str(sample["split"]), str(sample["domain"]))
        if selected_by_group[group] >= quotas[group]:
            unused_reserve += 1
            continue
        prompt_started = time.perf_counter()
        record = collect_prompt(
            sample=sample,
            tokenizer=tokenizer,
            target=target,
            draft=draft,
            domino=domino,
            continuation_tokens=args.continuation_tokens,
            anchors_per_prompt=args.anchors_per_prompt,
        )
        if record is None:
            rejected_short += 1
            print(
                f"[{index}/{len(samples)}] {sample['sample_id']}: "
                "reserve rejected (fewer than 129 pre-EOS tokens)",
                flush=True,
            )
            continue
        writer.add(record)
        selected_by_group[group] += 1
        counts[f"{record['domain']}/{record['split']}"] += 1
        max_numeric_error = max(
            max_numeric_error,
            max(float(anchor["numeric_margin_error"]) for anchor in record["anchors"]),
        )
        rank0_tie_rows += sum(
            int(anchor["reference_rank0_tie_rows"])
            for anchor in record["anchors"]
        )
        print(
            f"[{index}/{len(samples)}] {record['sample_id']}: "
            f"{len(record['anchors'])} full16 blocks in "
            f"{time.perf_counter() - prompt_started:.2f}s",
            flush=True,
        )
    missing = {
        f"{split}/{domain}": quotas[(split, domain)]
        - selected_by_group[(split, domain)]
        for split, domain in quotas
        if selected_by_group[(split, domain)] != quotas[(split, domain)]
    }
    if missing:
        raise RuntimeError(f"reserve pool exhausted before exact quotas: {missing}")
    writer.flush_all()
    metadata = {
        "format": "parc16_full_prompt_data_v1",
        "collection_complete": True,
        "manifest": str(args.manifest.resolve()),
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "domino_draft": str(args.domino_draft.resolve()),
        "block_length": BLOCK_LENGTH,
        "pure_dflash_input_length": PURE_DFLASH_INPUT_LENGTH,
        "pure_dflash_geometry": "non_shift_raw17_slice_rows_1_through_16",
        "released_domino_geometry": "shift_label_raw16_all_rows",
        "candidates": CANDIDATES,
        "anchors_per_prompt": args.anchors_per_prompt,
        "continuation_tokens": args.continuation_tokens,
        "attention_implementation": args.attn_implementation,
        "manifest_candidate_prompts": len(samples),
        "prompt_records": writer.prompts,
        "blocks": writer.blocks,
        "required_prompt_records": sum(quotas.values()),
        "rejected_short_candidates": rejected_short,
        "unused_reserve_candidates": unused_reserve,
        "selected_counts_by_group": {
            f"{split}/{domain}": selected_by_group[(split, domain)]
            for split, domain in sorted(quotas)
        },
        "counts_by_domain_split": dict(sorted(counts.items())),
        "max_numeric_margin_error_all_local_splits": max_numeric_error,
        "greedy_first_top16_contract": True,
        "reference_rank0_tie_rows": rank0_tie_rows,
        "target_feature_width": len(draft.target_layer_ids)
        * int(target.config.hidden_size),
        "old_15_position_cache_used": False,
        "heldout_present": False,
        "validation_domino_comparator_materialized": True,
        "seconds": time.perf_counter() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "shards": writer.shards,
    }
    (work / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(work, args.output)
    return metadata


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
