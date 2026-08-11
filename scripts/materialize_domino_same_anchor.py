#!/usr/bin/env python3
"""Cache released-Domino backbone features on exact canonical anchors.

The cache is intentionally minimal: it stores semantic identifiers, labels,
the released Domino parallel hidden states, and released on-policy outcomes.
It does not hash source shards or duplicate stored contexts.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM

from collect_canonical_blocks import extract_context_feature
from diagnose_domino_bias_scale import (
    accepted_length,
    domino_scaled_onpolicy_ids,
)
from sph.data import validate_stored_canonical_contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume-work-output",
        type=Path,
        help="Resume a previously interrupted .incomplete-* directory.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "validation_select", "validation_gate"])
    parser.add_argument("--shard-blocks", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--max-samples-per-domain",
        type=int,
        help="Take this many sorted prompts from every domain (balanced subset).",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def load_records(root: Path, splits: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("collection_complete") is False:
        raise RuntimeError(f"canonical collection is incomplete: {root}")
    records: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        shard_records = torch.load(shard, map_location="cpu", weights_only=False)
        records.extend(
            record for record in shard_records if str(record.get("split")) in splits
        )
    if not records:
        raise ValueError(f"no records matched splits {sorted(splits)}")
    return metadata, records


class MinimalShardWriter:
    def __init__(self, root: Path, shard_blocks: int) -> None:
        if shard_blocks < 1:
            raise ValueError("shard_blocks must be positive")
        self.root = root
        self.shard_blocks = shard_blocks
        self.buffer: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []
        self.total = 0

    def add(self, record: dict[str, Any]) -> None:
        for key, value in record.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                raise ValueError(f"cached tensor {key!r} still requires gradients")
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_blocks:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        name = f"shard-{len(self.shards):05d}.pt"
        path = self.root / name
        torch.save(self.buffer, path)
        count = len(self.buffer)
        self.shards.append({"path": name, "blocks": count, "bytes": path.stat().st_size})
        self.total += count
        self.buffer = []

    def restore_existing_shards(self):
        """Restore a flushed prefix and yield its records one shard at a time."""

        if self.buffer or self.shards or self.total:
            raise RuntimeError("restore_existing_shards requires a fresh writer")
        paths = sorted(self.root.glob("shard-*.pt"))
        if not paths:
            raise ValueError(f"resume directory has no flushed shards: {self.root}")
        for index, path in enumerate(paths):
            expected_name = f"shard-{index:05d}.pt"
            if path.name != expected_name:
                raise ValueError(
                    f"resume shards are not contiguous: expected {expected_name}, got {path.name}"
                )
            records = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(records, list) or not records:
                raise ValueError(f"resume shard is empty or malformed: {path}")
            count = len(records)
            self.shards.append(
                {"path": path.name, "blocks": count, "bytes": path.stat().st_size}
            )
            self.total += count
            yield records


def validate_domino_contract(domino: Any, draft_positions: int) -> None:
    """Fail early if a checkpoint would change the released B16 alignment."""

    config = getattr(domino, "config", None)
    dflash_config = getattr(config, "dflash_config", {}) or {}
    projector_type = getattr(
        domino, "projector_type", dflash_config.get("projector_type")
    )
    if projector_type not in {"domino", "causal_v5"}:
        raise ValueError(f"expected a Domino projector, got {projector_type!r}")
    if int(getattr(domino, "pure_draft_prefix_len", 0)) != 1:
        raise ValueError("materializer expects Domino pure_draft_prefix_len=1")
    if not bool(dflash_config.get("shift_label", False)):
        raise ValueError("same-anchor B16 materialization requires shift_label=true")
    block_size = int(getattr(domino, "block_size", 0))
    if block_size != int(draft_positions) + 1:
        raise ValueError(
            f"canonical horizon={draft_positions} is incompatible with "
            f"Domino block_size={block_size}"
        )


def prompt_balanced_mean(values_by_prompt: dict[str, list[int]]) -> float:
    if not values_by_prompt or any(not values for values in values_by_prompt.values()):
        raise ValueError("prompt-balanced mean requires non-empty prompt groups")
    prompt_means = [sum(values) / len(values) for values in values_by_prompt.values()]
    return sum(prompt_means) / len(prompt_means)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Domino feature materialization requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse existing output {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_tag = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    if args.resume_work_output is None:
        work_output = args.output.with_name(f"{args.output.name}.incomplete-{run_tag}")
        if work_output.exists():
            raise FileExistsError(f"refusing to reuse incomplete output {work_output}")
        work_output.mkdir()
    else:
        work_output = args.resume_work_output
        if not work_output.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {work_output}")
    source_metadata, source_records = load_records(args.canonical, set(args.splits))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        grouped[str(record["sample_id"])].append(record)
    sample_ids = sorted(grouped)
    if args.max_samples is not None and args.max_samples_per_domain is not None:
        raise ValueError("use only one of max-samples and max-samples-per-domain")
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]
    if args.max_samples_per_domain is not None:
        if args.max_samples_per_domain < 1:
            raise ValueError("max-samples-per-domain must be positive")
        by_domain: dict[str, list[str]] = defaultdict(list)
        for sample_id in sample_ids:
            domains = {str(record["domain"]) for record in grouped[sample_id]}
            if len(domains) != 1:
                raise ValueError(f"sample {sample_id} spans domains {sorted(domains)}")
            by_domain[next(iter(domains))].append(sample_id)
        sample_ids = sorted(
            sample_id
            for domain_ids in by_domain.values()
            for sample_id in domain_ids[: args.max_samples_per_domain]
        )
    selected_sample_count = len(sample_ids)

    torch.cuda.set_device(0)
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
    horizon = int(source_metadata["draft_positions"])
    validate_domino_contract(domino, horizon)

    writer = MinimalShardWriter(work_output, args.shard_blocks)
    counts: Counter[str] = Counter()
    released_lengths: dict[str, list[int]] = defaultdict(list)
    released_lengths_by_prompt: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    completed_record_counts: Counter[str] = Counter()

    def account_record(record: dict[str, Any]) -> None:
        sample_id = str(record["sample_id"])
        split = str(record["split"])
        domain = str(record["domain"])
        released_length = int(record["released_accepted_length"])
        counts[f"{domain}/{split}"] += 1
        released_lengths[split].append(released_length)
        released_lengths_by_prompt[split][sample_id].append(released_length)

    resumed_blocks = 0
    if args.resume_work_output is not None:
        for shard_records in writer.restore_existing_shards():
            for record in shard_records:
                sample_id = str(record["sample_id"])
                completed_record_counts[sample_id] += 1
                account_record(record)
        resumed_blocks = writer.total
        for sample_id, count in completed_record_counts.items():
            if sample_id not in grouped:
                raise ValueError(f"resumed sample is absent from source: {sample_id}")
            expected = len(grouped[sample_id])
            if count != expected:
                raise ValueError(
                    f"resumed sample {sample_id} has {count}/{expected} records; "
                    "only complete-sample shard prefixes can be resumed"
                )
        sample_ids = [
            sample_id for sample_id in sample_ids if sample_id not in completed_record_counts
        ]
        print(
            f"resuming {resumed_blocks} blocks across "
            f"{len(completed_record_counts)} complete samples; "
            f"{len(sample_ids)} samples remain",
            flush=True,
        )

    started = time.perf_counter()
    for sample_index, sample_id in enumerate(sample_ids):
        records = sorted(grouped[sample_id], key=lambda item: int(item["anchor_offset"]))
        longest_context_ids = validate_stored_canonical_contexts(records, sample_id)
        target_outputs = target.model(
            longest_context_ids.unsqueeze(0).to(target.device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        context_features = extract_context_feature(
            target_outputs.hidden_states, list(domino.target_layer_ids)
        )
        del target_outputs

        for record in records:
            context_length = int(record["context_ids_before_anchor"].numel())
            anchor = torch.tensor(
                int(record["anchor_token_id"]), dtype=torch.long, device=target.device
            )
            block_ids = torch.full(
                (1, int(domino.block_size)),
                int(domino.mask_token_id),
                dtype=torch.long,
                device=target.device,
            )
            block_ids[0, 0] = anchor
            position_ids = torch.arange(
                context_length + int(domino.block_size), device=target.device
            ).unsqueeze(0)
            parallel_hidden = domino(
                target_hidden=context_features[:, :context_length],
                noise_embedding=target.model.embed_tokens(block_ids),
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )
            base_logits = target.lm_head(parallel_hidden)
            released_ids = domino_scaled_onpolicy_ids(
                domino, target, anchor, parallel_hidden, base_logits, [1.0]
            )[0]
            gold = record["gold_ids"].long().to(target.device)[:horizon]
            if gold.numel() != horizon:
                raise ValueError(
                    f"{sample_id} anchor {record['anchor_offset']} has "
                    f"{gold.numel()} gold tokens, expected {horizon}"
                )
            if parallel_hidden.shape[1] < horizon or released_ids.shape[-1] < horizon:
                raise ValueError("Domino rollout returned fewer positions than the horizon")
            released_length = accepted_length(released_ids[:horizon], gold)
            split = str(record["split"])
            domain = str(record["domain"])
            cached_record = {
                "sample_id": sample_id,
                "domain": domain,
                "source": str(record["source"]),
                "split": split,
                "anchor_offset": int(record["anchor_offset"]),
                "anchor_token_id": int(record["anchor_token_id"]),
                "gold_ids": gold.detach().to(device="cpu", dtype=torch.int32),
                "parallel_hidden": parallel_hidden[0, :horizon]
                .detach()
                .to(device="cpu", dtype=torch.bfloat16),
                "released_onpolicy_ids": released_ids[:horizon]
                .detach()
                .to(device="cpu", dtype=torch.int32),
                "released_accepted_length": released_length,
            }
            writer.add(cached_record)
            account_record(cached_record)
            del parallel_hidden, base_logits, released_ids
        print(
            f"[{sample_index + 1}/{len(sample_ids)}] {sample_id}: {len(records)} anchors",
            flush=True,
        )

    writer.flush()
    metadata = {
        "format": "domino_same_anchor_hidden_v1",
        "collection_complete": True,
        "source_canonical": str(args.canonical.resolve()),
        "target": str(args.target.resolve()),
        "domino_draft": str(args.domino_draft.resolve()),
        "splits": args.splits,
        "samples": selected_sample_count,
        "blocks": writer.total,
        "draft_positions": int(source_metadata["draft_positions"]),
        "dtype": "bfloat16",
        "attention_implementation": args.attn_implementation,
        "seconds": time.perf_counter() - started,
        "resumed_blocks": resumed_blocks,
        "counts_by_domain_split": dict(sorted(counts.items())),
        "released_round_weighted_eal_by_split": {
            split: sum(values) / len(values) for split, values in released_lengths.items()
        },
        "released_prompt_balanced_eal_by_split": {
            split: prompt_balanced_mean(grouped)
            for split, grouped in released_lengths_by_prompt.items()
        },
        "shards": writer.shards,
        "semantic_checks": "stored contexts must be prefix-nested with matching anchor/gold",
    }
    (work_output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(work_output, args.output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
