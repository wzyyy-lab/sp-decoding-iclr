#!/usr/bin/env python3
"""Collect pure-DFlash candidate blocks at target-greedy canonical anchors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


PROJECT = Path(__file__).resolve().parents[1]


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


def read_manifest(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    required = {"sample_id", "domain", "source", "prompt", "split"}
    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"manifest record {index} is missing {sorted(missing)}")
    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest sample_id values must be unique")
    return records if max_samples is None else records[:max_samples]


def evenly_spaced_offsets(maximum: int, count: int) -> list[int]:
    if maximum < 0 or count < 1:
        return []
    if count == 1 or maximum == 0:
        return [0]
    count = min(count, maximum + 1)
    offsets = {
        int(round(index * maximum / (count - 1))) for index in range(count)
    }
    return sorted(offsets)


def extract_context_feature(
    hidden_states: tuple[torch.Tensor, ...], layer_ids: list[int]
) -> torch.Tensor:
    # Hugging Face hidden_states[0] is the embedding output, so layer i is i+1.
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


class ShardWriter:
    def __init__(self, output: Path, blocks_per_shard: int) -> None:
        if blocks_per_shard < 1:
            raise ValueError("blocks_per_shard must be positive")
        self.output = output
        self.blocks_per_shard = blocks_per_shard
        self.records: list[dict[str, Any]] = []
        self.shard_index = 0
        self.total_blocks = 0
        self.sample_ids: set[str] = set()
        self.counts: dict[str, int] = defaultdict(int)
        self.shards: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self.sample_ids.add(record["sample_id"])
        self.counts[f"{record['domain']}/{record['split']}"] += 1
        if len(self.records) >= self.blocks_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        path = self.output / f"shard-{self.shard_index:05d}.pt"
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        torch.save(self.records, temporary)
        os.replace(temporary, path)
        print(f"wrote {path} ({len(self.records)} blocks)", flush=True)
        self.shards.append(
            {
                "path": path.name,
                "blocks": len(self.records),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        self.total_blocks += len(self.records)
        self.records = []
        self.shard_index += 1


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_is_dirty(path: Path) -> bool | None:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def checkpoint_fingerprint(root: Path) -> list[dict[str, Any]]:
    """Hash every top-level checkpoint file, including weights and remote code."""

    files = sorted(path for path in root.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(f"checkpoint directory is empty: {root}")
    return [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_output_directory(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to mix with an existing collection in {output}; "
            "choose a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)


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
    maximum_anchor = int(continuation.numel()) - block_size
    offsets = evenly_spaced_offsets(maximum_anchor, anchors_per_sample)
    if not offsets:
        print(
            f"skip {sample['sample_id']}: continuation has only "
            f"{continuation.numel()} tokens",
            flush=True,
        )
        return []

    # One causal target pass provides every prefix feature: prefix token states
    # are invariant to later suffix tokens, so each anchor is still canonical.
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

    mask_token_id = int(draft.mask_token_id)
    records: list[dict[str, Any]] = []
    for anchor_offset in offsets:
        context_length = prompt_tokens + anchor_offset
        anchor_token_id = continuation[anchor_offset].to(torch.long)
        gold_ids = continuation[
            anchor_offset + 1 : anchor_offset + block_size
        ].to(torch.long)
        if gold_ids.numel() != block_size - 1:
            raise AssertionError("gold block length invariant failed")

        block_ids = torch.full(
            (1, block_size),
            mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        block_ids[0, 0] = anchor_token_id
        noise_embedding = target.model.embed_tokens(block_ids)
        position_ids = torch.arange(
            context_length + block_size, device=target.device
        ).unsqueeze(0)
        parallel_hidden = draft(
            target_hidden=context_features[:, :context_length],
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )[:, 1 - block_size :, :]
        base_logits = target.lm_head(parallel_hidden)
        topk_logits, topk_ids = torch.topk(
            base_logits.float(), k=top_k, dim=-1, sorted=True
        )
        base_logsumexp = torch.logsumexp(base_logits.float(), dim=-1)
        gold_gpu = gold_ids.unsqueeze(0)
        base_top1_match = topk_ids[..., 0] == gold_gpu
        gold_in_saved_topk = topk_ids == gold_gpu.unsqueeze(-1)
        rank_axis = torch.arange(1, top_k + 1, device=target.device)
        missing_rank = torch.full_like(rank_axis, top_k + 1)
        gold_rank = torch.where(
            gold_in_saved_topk, rank_axis, missing_rank
        ).amin(dim=-1)

        records.append(
            {
                "sample_id": sample["sample_id"],
                "domain": sample["domain"],
                "source": sample["source"],
                "split": sample["split"],
                "prompt_token_count": prompt_tokens,
                "anchor_offset": anchor_offset,
                "context_length": context_length,
                # Exact token IDs strictly before the anchor.  Storing these
                # makes same-anchor replay independent of generation-kernel
                # tie behavior on another GPU or attention backend.
                "context_ids_before_anchor": sequence[
                    0, :context_length
                ].cpu().to(torch.int32),
                "anchor_token_id": int(anchor_token_id.item()),
                "gold_ids": gold_ids.cpu().to(torch.int32),
                "parallel_hidden": parallel_hidden[0].cpu().to(torch.bfloat16),
                "base_topk_ids": topk_ids[0].cpu().to(torch.int32),
                "base_topk_logits": topk_logits[0].cpu().to(torch.float16),
                "base_logsumexp": base_logsumexp[0].cpu().to(torch.float32),
                "base_top1_match": base_top1_match[0].cpu(),
                "gold_rank": gold_rank[0].cpu().to(torch.int16),
            }
        )
        del parallel_hidden, base_logits, topk_logits, topk_ids

    return records


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("canonical collection requires a CUDA GPU")
    if args.block_size < 2:
        raise ValueError("block_size must be at least 2")
    if args.top_k < 1:
        raise ValueError("top_k must be positive")
    samples = read_manifest(args.manifest, args.max_samples)
    validate_output_directory(args.output)
    run_provenance = {
        "project_commit": git_revision(PROJECT),
        "project_dirty_at_start": git_is_dirty(PROJECT),
        "collector_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": sha256_file(args.manifest),
        "target_files": checkpoint_fingerprint(args.target),
        "draft_files": checkpoint_fingerprint(args.draft),
        "dflash_commit": git_revision(PROJECT / "third_party" / "dflash"),
        "dflash_dirty_at_start": git_is_dirty(
            PROJECT / "third_party" / "dflash"
        ),
        "domino_commit": git_revision(PROJECT / "third_party" / "Domino"),
        "domino_dirty_at_start": git_is_dirty(
            PROJECT / "third_party" / "Domino"
        ),
    }
    metadata = {
        "format_version": 2,
        "collection_complete": False,
        "created_unix": time.time(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "manifest": str(args.manifest.resolve()),
        "block_size": args.block_size,
        "draft_positions": args.block_size - 1,
        "top_k": args.top_k,
        "anchors_per_sample": args.anchors_per_sample,
        "continuation_tokens": args.continuation_tokens,
        "attention_implementation": args.attn_implementation,
        "dtype": "bfloat16",
        "num_manifest_samples": len(samples),
        "provenance": run_provenance,
    }
    incomplete_path = args.output / "INCOMPLETE.json"
    atomic_write_json(incomplete_path, metadata)
    torch.cuda.set_device(0)

    load_start = time.perf_counter()
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
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    if int(draft.block_size) != args.block_size:
        raise ValueError(
            f"checkpoint block_size={draft.block_size}, requested {args.block_size}"
        )
    if args.top_k > int(target.config.vocab_size):
        raise ValueError("top_k exceeds target vocabulary")
    if getattr(draft.config, "dflash_config", {}).get("projector_type") is not None:
        raise ValueError("Gate 1 must use the pure DFlash checkpoint, not Domino")

    metadata.update(
        {
            "device": torch.cuda.get_device_name(0),
            "load_seconds": load_seconds,
            "target_layer_ids": list(draft.target_layer_ids),
        }
    )
    atomic_write_json(incomplete_path, metadata)

    writer = ShardWriter(args.output, args.shard_blocks)
    collection_start = time.perf_counter()
    for sample_index, sample in enumerate(samples):
        sample_start = time.perf_counter()
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
            f"[{sample_index + 1}/{len(samples)}] {sample['sample_id']}: "
            f"{len(records)} blocks in {time.perf_counter() - sample_start:.2f}s",
            flush=True,
        )
    writer.flush()
    metadata["num_blocks"] = writer.total_blocks
    metadata["num_collected_samples"] = len(writer.sample_ids)
    metadata["block_counts_by_domain_split"] = dict(sorted(writer.counts.items()))
    metadata["shards"] = writer.shards
    metadata["collection_seconds"] = time.perf_counter() - collection_start
    metadata["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    metadata["collection_complete"] = True
    atomic_write_json(args.output / "metadata.json", metadata)
    incomplete_path.unlink()
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
