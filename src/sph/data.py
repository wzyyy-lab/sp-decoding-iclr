"""Dataset helpers for canonical DFlash/Domino block shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CanonicalBlockDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str | None = None,
        domains: Iterable[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        allowed_domains = None if domains is None else set(domains)
        records: list[dict[str, Any]] = []
        for shard in sorted(self.root.glob("shard-*.pt")):
            shard_records = torch.load(
                shard, map_location="cpu", weights_only=False
            )
            for record in shard_records:
                if split is not None and record["split"] != split:
                    continue
                if allowed_domains is not None and record["domain"] not in allowed_domains:
                    continue
                records.append(record)
        if not records:
            raise ValueError(
                f"no canonical blocks matched split={split!r}, domains={allowed_domains}"
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate_canonical_blocks(
    records: list[dict[str, Any]], candidate_k: int
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot collate an empty batch")
    saved_k = records[0]["base_topk_ids"].shape[-1]
    if not 1 <= candidate_k <= saved_k:
        raise ValueError(f"candidate_k must be in [1, {saved_k}]")
    candidate_ids = torch.stack(
        [record["base_topk_ids"][:, :candidate_k].long() for record in records]
    )
    candidate_logits = torch.stack(
        [record["base_topk_logits"][:, :candidate_k].float() for record in records]
    )
    gold_ids = torch.stack([record["gold_ids"].long() for record in records])
    matches = candidate_ids == gold_ids.unsqueeze(-1)
    gold_in_lattice = matches.any(dim=-1)
    gold_candidate_indices = matches.to(torch.int64).argmax(dim=-1)
    return {
        "sample_ids": [record["sample_id"] for record in records],
        "domains": [record["domain"] for record in records],
        "hidden": torch.stack(
            [record["parallel_hidden"].to(torch.bfloat16) for record in records]
        ),
        "anchor_ids": torch.tensor(
            [record["anchor_token_id"] for record in records], dtype=torch.long
        ),
        "candidate_ids": candidate_ids,
        "candidate_logits": candidate_logits,
        "base_logsumexp": torch.stack(
            [record["base_logsumexp"].float() for record in records]
        ),
        "gold_ids": gold_ids,
        "gold_in_lattice": gold_in_lattice,
        "gold_candidate_indices": gold_candidate_indices,
    }
