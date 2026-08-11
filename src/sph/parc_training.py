"""Data and optimization utilities for the claim-bearing PARC-16 run."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor

from sph.parc import BLOCK_LENGTH, CANDIDATES


EXPECTED_ANCHORS = 8
EXPECTED_TRAIN_PROMPTS = 90_000
EXPECTED_VALIDATION_PROMPTS = 5_000
AUDIT_DOMAIN_QUOTAS = {"chat": 1_667, "code": 1_666, "math": 1_667}


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    split: str
    prompts: int
    blocks: int
    sample_ids: tuple[str, ...]
    summaries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DataCatalog:
    root: Path
    shards: tuple[ShardInfo, ...]
    target: Path
    draft: Path
    domino_draft: Path

    def for_split(self, split: str) -> tuple[ShardInfo, ...]:
        selected = tuple(shard for shard in self.shards if shard.split == split)
        if not selected:
            raise RuntimeError(f"PARC catalog has no {split!r} shards")
        return selected

    def prompt_count(self, split: str) -> int:
        return sum(shard.prompts for shard in self.for_split(split))


def _metadata_paths(root: Path) -> list[Path]:
    direct = root / "metadata.json"
    if direct.exists():
        return [direct]
    paths = sorted(root.glob("part-*/metadata.json"))
    if not paths:
        raise FileNotFoundError(f"no PARC part metadata under {root}")
    return paths


def load_data_catalog(
    root: Path,
    *,
    target: Path,
    draft: Path,
    domino_draft: Path,
) -> DataCatalog:
    """Load semantic receipts without touching the sealed held-out manifest."""

    shards: list[ShardInfo] = []
    seen_ids: set[str] = set()
    expected_paths = {
        "target": target.resolve(),
        "draft": draft.resolve(),
        "domino_draft": domino_draft.resolve(),
    }
    for metadata_path in _metadata_paths(root):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("format") != "parc16_full_prompt_data_v1":
            raise RuntimeError(f"invalid PARC metadata {metadata_path}")
        if metadata.get("collection_complete") is not True:
            raise RuntimeError(f"incomplete PARC collection {metadata_path.parent}")
        if metadata.get("heldout_present") is not False:
            raise RuntimeError("training catalog contains held-out data")
        if metadata.get("old_15_position_cache_used") is not False:
            raise RuntimeError("PARC training cannot use the old 15-position cache")
        if int(metadata.get("block_length", -1)) != BLOCK_LENGTH:
            raise RuntimeError("PARC collection is not full16")
        if int(metadata.get("pure_dflash_input_length", -1)) != BLOCK_LENGTH + 1:
            raise RuntimeError("PARC collection did not use raw17 pure-DFlash input")
        if metadata.get("pure_dflash_geometry") != (
            "non_shift_raw17_slice_rows_1_through_16"
        ):
            raise RuntimeError("PARC pure-DFlash row alignment is not frozen full16")
        if metadata.get("released_domino_geometry") != "shift_label_raw16_all_rows":
            raise RuntimeError("PARC Domino comparator row alignment drifted")
        if int(metadata.get("candidates", -1)) != CANDIDATES:
            raise RuntimeError("PARC collection candidate count drifted")
        if int(metadata.get("anchors_per_prompt", -1)) != EXPECTED_ANCHORS:
            raise RuntimeError("PARC collection must use exactly eight anchors")
        if metadata.get("validation_domino_comparator_materialized") is not True:
            raise RuntimeError("validation Domino comparator is missing")
        for key, expected in expected_paths.items():
            if Path(str(metadata.get(key, ""))).resolve() != expected:
                raise RuntimeError(f"PARC {key} source differs across collection parts")
        for item in metadata.get("shards", []):
            split = str(item.get("split"))
            if split not in {"train", "validation"}:
                raise RuntimeError(f"materialized forbidden split {split!r}")
            sample_ids = tuple(str(value) for value in item.get("sample_ids", []))
            prompts = int(item.get("prompts", -1))
            blocks = int(item.get("blocks", -1))
            summaries = tuple(item.get("prompt_reference_summaries", []))
            if prompts <= 0 or len(sample_ids) != prompts or len(summaries) != prompts:
                raise RuntimeError("PARC shard prompt receipt is inconsistent")
            if blocks != prompts * EXPECTED_ANCHORS:
                raise RuntimeError("PARC shard does not have eight full16 blocks per prompt")
            if len(set(sample_ids)) != prompts or seen_ids.intersection(sample_ids):
                raise RuntimeError("PARC materialized prompt IDs overlap")
            if tuple(str(row.get("sample_id")) for row in summaries) != sample_ids:
                raise RuntimeError("PARC shard summary order differs from sample IDs")
            if split == "validation" and any(
                row.get("reference_domino_accepted_lengths") is None
                for row in summaries
            ):
                raise RuntimeError("validation shard lacks released Domino results")
            seen_ids.update(sample_ids)
            shard_path = metadata_path.parent / str(item["path"])
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            shards.append(
                ShardInfo(
                    path=shard_path,
                    split=split,
                    prompts=prompts,
                    blocks=blocks,
                    sample_ids=sample_ids,
                    summaries=summaries,
                )
            )
    catalog = DataCatalog(
        root=root.resolve(),
        shards=tuple(sorted(shards, key=lambda item: str(item.path))),
        target=target.resolve(),
        draft=draft.resolve(),
        domino_draft=domino_draft.resolve(),
    )
    train_count = catalog.prompt_count("train")
    validation_count = catalog.prompt_count("validation")
    if train_count != EXPECTED_TRAIN_PROMPTS:
        raise RuntimeError(f"unexpected effective train prompt count {train_count}")
    if validation_count != EXPECTED_VALIDATION_PROMPTS:
        raise RuntimeError(
            f"unexpected effective validation prompt count {validation_count}"
        )
    return catalog


def validate_prompt_record(record: dict[str, Any], split: str) -> None:
    if str(record.get("split")) != split:
        raise RuntimeError("record split differs from its shard")
    if str(record.get("domain")) not in {"chat", "code", "math"}:
        raise RuntimeError("record domain is outside the frozen task")
    features = record.get("target_context_features")
    anchors = record.get("anchors")
    if not isinstance(features, Tensor) or features.ndim != 2:
        raise RuntimeError("record target context features are malformed")
    if features.dtype != torch.bfloat16 or features.shape[1] != 12_800:
        raise RuntimeError("record target context features differ from full DFlash")
    if not isinstance(anchors, list) or len(anchors) != EXPECTED_ANCHORS:
        raise RuntimeError("record does not contain exactly eight anchors")
    previous_offset = -1
    for anchor in anchors:
        offset = int(anchor["anchor_offset"])
        context_length = int(anchor["context_length"])
        if offset <= previous_offset:
            raise RuntimeError("anchor offsets are not strictly increasing")
        previous_offset = offset
        if context_length > features.shape[0]:
            raise RuntimeError("anchor context exceeds stored target features")
        for key, shape in (
            ("gold_ids", (BLOCK_LENGTH,)),
            ("reference_proposal_ids", (BLOCK_LENGTH,)),
            ("reference_topk_ids", (BLOCK_LENGTH, CANDIDATES)),
            ("reference_topk_logits", (BLOCK_LENGTH, CANDIDATES)),
        ):
            value = anchor.get(key)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise RuntimeError(f"anchor field {key} is malformed")
        accepted = int(anchor["reference_accepted_length"])
        if not 0 <= accepted <= BLOCK_LENGTH:
            raise RuntimeError("reference accepted length lies outside full16")
        if split == "validation":
            domino = anchor.get("reference_domino_proposal_ids")
            domino_accepted = int(anchor.get("reference_domino_accepted_length", -1))
            if not isinstance(domino, Tensor) or tuple(domino.shape) != (BLOCK_LENGTH,):
                raise RuntimeError("validation anchor lacks Domino proposal")
            if not 0 <= domino_accepted <= BLOCK_LENGTH:
                raise RuntimeError("Domino accepted length lies outside full16")


def load_shard(shard: ShardInfo) -> list[dict[str, Any]]:
    records = torch.load(shard.path, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or len(records) != shard.prompts:
        raise RuntimeError(f"malformed PARC shard {shard.path}")
    sample_ids = tuple(str(record.get("sample_id")) for record in records)
    if sample_ids != shard.sample_ids:
        raise RuntimeError(f"PARC shard sample order drifted: {shard.path}")
    for record in records:
        validate_prompt_record(record, shard.split)
    return records


class BlockStream:
    """Deterministic two-level shuffle over prompt-uniform full16 blocks."""

    def __init__(
        self,
        catalog: DataCatalog,
        *,
        seed: int,
        state: dict[str, int] | None = None,
    ) -> None:
        self.shards = catalog.for_split("train")
        self.seed = int(seed)
        state = state or {"epoch": 0, "shard_cursor": 0, "block_cursor": 0}
        self.epoch = int(state["epoch"])
        self.shard_cursor = int(state["shard_cursor"])
        self.block_cursor = int(state["block_cursor"])
        if min(self.epoch, self.shard_cursor, self.block_cursor) < 0:
            raise RuntimeError("negative PARC sampler state")
        self._loaded_key: tuple[int, int] | None = None
        self._records: list[dict[str, Any]] = []
        self._pairs: list[tuple[int, int]] = []

    def _ordered_shards(self) -> list[int]:
        order = list(range(len(self.shards)))
        random.Random(self.seed + 1_000_003 * self.epoch).shuffle(order)
        return order

    def _ensure_loaded(self) -> None:
        if self.shard_cursor == len(self.shards):
            self.epoch += 1
            self.shard_cursor = 0
            self.block_cursor = 0
        if self.shard_cursor > len(self.shards):
            raise RuntimeError("PARC sampler shard cursor exceeds the dataset")
        key = (self.epoch, self.shard_cursor)
        if self._loaded_key == key:
            return
        shard_index = self._ordered_shards()[self.shard_cursor]
        shard = self.shards[shard_index]
        self._records = load_shard(shard)
        self._pairs = [
            (record_index, anchor_index)
            for record_index in range(len(self._records))
            for anchor_index in range(EXPECTED_ANCHORS)
        ]
        random.Random(
            self.seed + 2_000_033 * self.epoch + 7_919 * shard_index
        ).shuffle(self._pairs)
        if self.block_cursor > len(self._pairs):
            raise RuntimeError("PARC sampler block cursor exceeds its shard")
        self._loaded_key = key

    def next_block(self) -> tuple[dict[str, Any], int]:
        self._ensure_loaded()
        if self.block_cursor == len(self._pairs):
            self.shard_cursor += 1
            self.block_cursor = 0
            self._loaded_key = None
            return self.next_block()
        pair = self._pairs[self.block_cursor]
        self.block_cursor += 1
        return self._records[pair[0]], pair[1]

    def next_batch(self, batch_size: int) -> list[tuple[dict[str, Any], int]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return [self.next_block() for _ in range(batch_size)]

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "shard_cursor": self.shard_cursor,
            "block_cursor": self.block_cursor,
        }


def iter_prompt_records(
    catalog: DataCatalog,
    split: str,
    *,
    sample_ids: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    for shard in catalog.for_split(split):
        if sample_ids is not None and not sample_ids.intersection(shard.sample_ids):
            continue
        for record in load_shard(shard):
            if sample_ids is None or str(record["sample_id"]) in sample_ids:
                yield record


def numeric_certificate(catalog: DataCatalog) -> dict[str, float | int]:
    summaries = [
        row for shard in catalog.for_split("train") for row in shard.summaries
    ]
    max_error = max(float(row["numeric_margin_error"]) for row in summaries)
    delta_min = max(2.0 * max_error, 2.0 / 64.0, 2.0**-14)
    prompt_rates: list[float] = []
    ambiguous_blocks = 0
    for row in summaries:
        deltas = [float(value) for value in row["reference_deltas"]]
        accepted = [int(value) for value in row["reference_accepted_lengths"]]
        flags = [int(length > 0 and delta <= delta_min) for length, delta in zip(accepted, deltas, strict=True)]
        ambiguous_blocks += sum(flags)
        prompt_rates.append(sum(flags) / len(flags))
    prompt_mean = sum(prompt_rates) / len(prompt_rates)
    return {
        "train_prompts": len(summaries),
        "train_blocks": len(summaries) * EXPECTED_ANCHORS,
        "max_numeric_margin_error": max_error,
        "delta_min": delta_min,
        "ambiguous_blocks": ambiguous_blocks,
        "prompt_mean_ambiguous": prompt_mean,
    }


def select_train_audit_ids(catalog: DataCatalog) -> set[str]:
    remaining = dict(AUDIT_DOMAIN_QUOTAS)
    selected: set[str] = set()
    for shard in catalog.for_split("train"):
        for row in shard.summaries:
            domain = str(row["domain"])
            if remaining[domain] > 0:
                selected.add(str(row["sample_id"]))
                remaining[domain] -= 1
        if all(value == 0 for value in remaining.values()):
            break
    if remaining != {domain: 0 for domain in remaining} or len(selected) != 5_000:
        raise RuntimeError("could not freeze the 5K train-audit prompt subset")
    return selected


def cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    minimum_ratio: float = 0.1,
) -> float:
    if not 0 <= step <= total_steps:
        raise ValueError("learning-rate step lies outside the run")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("invalid warmup duration")
    if peak <= 0 or not 0 < minimum_ratio <= 1:
        raise ValueError("invalid learning-rate scale")
    if step < warmup_steps:
        return peak * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def accepted_lengths(proposal: Tensor, gold: Tensor) -> Tensor:
    if proposal.shape != gold.shape or proposal.shape[-1] != BLOCK_LENGTH:
        raise ValueError("proposal/gold must share shape [B,16]")
    return proposal.eq(gold).to(torch.long).cumprod(dim=-1).sum(dim=-1)


def grouped_prompt_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty PARC metrics")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    domains: dict[str, str] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        domain = str(row["domain"])
        if sample_id in domains and domains[sample_id] != domain:
            raise RuntimeError("one prompt appears under multiple domains")
        domains[sample_id] = domain
        grouped[sample_id].append(row)

    prompt_rows: list[dict[str, Any]] = []
    numeric_keys = sorted(
        key for key, value in rows[0].items() if isinstance(value, (int, float))
    )
    for sample_id, values in grouped.items():
        prompt = {"sample_id": sample_id, "domain": domains[sample_id]}
        for key in numeric_keys:
            prompt[key] = sum(float(value[key]) for value in values) / len(values)
        prompt_rows.append(prompt)

    def summarize(values: Sequence[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "prompts": len(values),
            **{
                key: sum(float(row[key]) for row in values) / len(values)
                for key in numeric_keys
            },
        }

    return {
        "overall": summarize(prompt_rows),
        "by_domain": {
            domain: summarize(
                [row for row in prompt_rows if row["domain"] == domain]
            )
            for domain in ("chat", "code", "math")
            if any(row["domain"] == domain for row in prompt_rows)
        },
    }


def checkpoint_is_better(
    candidate: dict[str, Any], best: dict[str, Any] | None
) -> bool:
    overall = candidate["overall"]
    if float(overall["actual_harm"]) > 0.01:
        return False
    if best is None:
        return True
    return float(overall["eal"]) > float(best["overall"]["eal"])
