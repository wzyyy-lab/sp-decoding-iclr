#!/usr/bin/env python3
"""Materialize the exact B16 on-policy Domino teacher for PLC training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM

from collect_canonical_blocks import extract_context_feature
from sph.candidate_ceiling import accepted_draft_prefix_lengths
from sph.data import validate_stored_canonical_contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation_select", "validation_gate"],
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-blocks", type=int, default=256)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def load_records(root: Path, splits: set[str]) -> list[dict[str, Any]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("collection_complete", False):
        raise RuntimeError(f"source collection is incomplete: {root}")
    records: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        records.extend(
            record
            for record in torch.load(shard, map_location="cpu", weights_only=False)
            if str(record["split"]) in splits
        )
    if not records:
        raise ValueError("no source records matched the requested splits")
    return records


class ShardWriter:
    def __init__(self, root: Path, shard_blocks: int) -> None:
        self.root = root
        self.shard_blocks = shard_blocks
        self.buffer: list[dict[str, Any]] = []
        self.shards: list[dict[str, int | str]] = []
        self.total = 0

    def add(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_blocks:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.root / f"shard-{len(self.shards):05d}.pt"
        torch.save(self.buffer, path)
        count = len(self.buffer)
        self.total += count
        self.shards.append(
            {"path": path.name, "blocks": count, "bytes": path.stat().st_size}
        )
        self.buffer = []


@torch.inference_mode()
def runtime_teacher(
    *,
    domino: Any,
    target_weight: torch.Tensor,
    anchor: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return base prefix, 15 corrected tokens, and 15 on-policy W_s s codes."""

    if hidden.shape[1] != 16:
        raise ValueError("runtime teacher requires 16 parallel hidden positions")
    base_logits = F.linear(hidden, target_weight)
    prefix = base_logits[:, 0].argmax(dim=-1)
    prefix_pair = torch.stack([anchor, prefix], dim=-1)
    _, state = domino.prefix_gru(F.embedding(prefix_pair, target_weight))
    first_projection = domino.embed_proj[0]
    hidden_width = hidden.shape[-1]
    w_s = first_projection.weight[:, hidden_width:]
    tokens: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    for position in range(1, 16):
        state_for_head = state.transpose(0, 1)
        if bool(getattr(domino, "use_bias_norm", False)):
            state_for_head = domino.bias_norm(state_for_head)
        delta = F.linear(state_for_head[:, 0], w_s)
        bias = domino.embed_proj(
            torch.cat([hidden[:, position : position + 1], state_for_head], dim=-1)
        )
        if bool(getattr(domino, "use_bias_gate", False)):
            bias = torch.sigmoid(
                domino.bias_gate(hidden[:, position : position + 1])
            ) * bias
        token = (base_logits[:, position : position + 1] + bias).argmax(dim=-1)
        deltas.append(delta)
        tokens.append(token[:, 0])
        if position < 15:
            _, state = domino.prefix_gru(F.embedding(token, target_weight), state)
    return prefix, torch.stack(tokens, dim=1), torch.stack(deltas, dim=1)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    args.output.mkdir(parents=True)
    records = load_records(args.canonical, set(args.splits))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(record)
    sample_ids = sorted(grouped)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
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
    target.requires_grad_(False)
    domino.requires_grad_(False)
    if int(domino.block_size) != 16:
        raise ValueError("PLC runtime teacher currently requires Domino B16")
    if int(domino.pure_draft_prefix_len) != 1:
        raise ValueError("PLC runtime teacher requires pure_draft_prefix_len=1")

    writer = ShardWriter(args.output, args.shard_blocks)
    counts: Counter[str] = Counter()
    teacher_lengths: dict[str, list[int]] = defaultdict(list)
    started = time.perf_counter()
    for sample_index, sample_id in enumerate(sample_ids):
        sample_records = sorted(
            grouped[sample_id], key=lambda item: int(item["context_length"])
        )
        validate_stored_canonical_contexts(sample_records, sample_id)
        last = sample_records[-1]
        full_sequence = torch.cat(
            [
                last["context_ids_before_anchor"].long(),
                torch.tensor([int(last["anchor_token_id"])], dtype=torch.long),
                last["gold_ids"].long(),
            ]
        ).to("cuda:0")
        target_outputs = target.model(
            full_sequence.unsqueeze(0),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        context_features = extract_context_feature(
            target_outputs.hidden_states, list(domino.target_layer_ids)
        )
        final_hidden = target_outputs.last_hidden_state

        for source_record in sample_records:
            context_length = int(source_record["context_length"])
            existing_gold = source_record["gold_ids"].long().to("cuda:0")
            expected = torch.cat(
                [
                    torch.tensor(
                        [int(source_record["anchor_token_id"])],
                        device="cuda:0",
                        dtype=torch.long,
                    ),
                    existing_gold,
                ]
            )
            if not torch.equal(
                full_sequence[context_length : context_length + 16], expected
            ):
                raise ValueError(f"non-nested continuation for {sample_id}")
            next_index = context_length + 16
            if next_index < full_sequence.numel():
                next_gold = full_sequence[next_index]
            else:
                next_gold = target.lm_head(
                    final_hidden[:, context_length + 15 : context_length + 16]
                )[0, 0].argmax(dim=-1)
            gold_full = torch.cat([existing_gold, next_gold.view(1)])

            anchor = torch.tensor(
                [int(source_record["anchor_token_id"])],
                device="cuda:0",
                dtype=torch.long,
            )
            block_ids = torch.full(
                (1, 16),
                int(domino.mask_token_id),
                device="cuda:0",
                dtype=torch.long,
            )
            block_ids[0, 0] = anchor[0]
            position_ids = torch.arange(
                context_length + 16, device="cuda:0"
            ).unsqueeze(0)
            hidden = domino(
                target_hidden=context_features[:, :context_length],
                noise_embedding=target.model.embed_tokens(block_ids),
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )
            if hidden.shape[1] != 16:
                raise RuntimeError(f"Domino returned shape {tuple(hidden.shape)}")
            prefix, teacher_ids, teacher_delta = runtime_teacher(
                domino=domino,
                target_weight=target.model.embed_tokens.weight,
                anchor=anchor,
                hidden=hidden,
            )
            teacher_full = torch.cat([prefix.view(1), teacher_ids[0]])
            accepted = int(
                accepted_draft_prefix_lengths(teacher_full == gold_full).item()
            )
            split = str(source_record["split"])
            domain = str(source_record["domain"])
            writer.add(
                {
                    "sample_id": sample_id,
                    "domain": domain,
                    "source": str(source_record["source"]),
                    "split": split,
                    "anchor_offset": int(source_record["anchor_offset"]),
                    "anchor_token_id": int(anchor.item()),
                    "base_prefix_token_id": int(prefix.item()),
                    "gold_full_ids": gold_full.cpu().to(torch.int32),
                    "gold_ids": gold_full[1:].cpu().to(torch.int32),
                    "parallel_hidden": hidden[0, 1:16]
                    .cpu()
                    .to(torch.bfloat16),
                    "teacher_ids": teacher_ids[0].cpu().to(torch.int32),
                    "teacher_delta": teacher_delta[0].cpu().to(torch.bfloat16),
                    "teacher_full_ids": teacher_full.cpu().to(torch.int32),
                    "teacher_accepted_length": accepted,
                }
            )
            counts[f"{domain}/{split}"] += 1
            teacher_lengths[split].append(accepted)

        del target_outputs, context_features, final_hidden
        if (sample_index + 1) % 10 == 0 or sample_index + 1 == len(sample_ids):
            print(
                f"[{sample_index + 1}/{len(sample_ids)}] blocks={writer.total + len(writer.buffer)}",
                flush=True,
            )

    writer.flush()
    metadata = {
        "format": "plc_runtime_teacher_v1",
        "collection_complete": True,
        "source_canonical": str(args.canonical.resolve()),
        "target": str(args.target.resolve()),
        "domino_draft": str(args.domino_draft.resolve()),
        "samples": len(sample_ids),
        "blocks": writer.total,
        "corrected_positions": 15,
        "counts_by_domain_split": dict(sorted(counts.items())),
        "teacher_round_weighted_eal_by_split": {
            split: sum(values) / len(values)
            for split, values in teacher_lengths.items()
        },
        "seconds": time.perf_counter() - started,
        "shards": writer.shards,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

