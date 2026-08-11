"""Dataset helpers for canonical DFlash/Domino block shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_collection_files(
    root: Path, metadata: dict[str, Any], *, verify_sha256: bool
) -> None:
    """Reject incomplete, missing, extra, or corrupted protocol-v2 shards."""

    if int(metadata.get("format_version", 1)) < 2:
        return
    if not metadata.get("collection_complete", False):
        raise RuntimeError(f"canonical collection is not complete: {root}")
    if (root / "INCOMPLETE.json").exists():
        raise RuntimeError(f"canonical collection still has INCOMPLETE marker: {root}")
    shard_entries = metadata.get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        raise RuntimeError(f"canonical collection has no shard manifest: {root}")
    expected_names = [str(entry["path"]) for entry in shard_entries]
    actual_names = [path.name for path in sorted(root.glob("shard-*.pt"))]
    if actual_names != expected_names:
        raise RuntimeError(
            f"canonical shard set differs from metadata: expected "
            f"{expected_names}, found {actual_names}"
        )
    for entry in shard_entries:
        path = root / str(entry["path"])
        expected_bytes = int(entry["bytes"])
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"canonical shard size mismatch: {path}")
        if verify_sha256 and _sha256_file(path) != str(entry["sha256"]):
            raise RuntimeError(f"canonical shard SHA256 mismatch: {path}")


class CanonicalBlockDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str | None = None,
        domains: Iterable[str] | None = None,
        verify_integrity: bool = True,
    ) -> None:
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        _validate_collection_files(
            self.root, self.metadata, verify_sha256=verify_integrity
        )
        allowed_domains = None if domains is None else set(domains)
        records: list[dict[str, Any]] = []
        total_loaded_blocks = 0
        shard_block_counts = {
            str(entry["path"]): int(entry["blocks"])
            for entry in self.metadata.get("shards", [])
        }
        for shard in sorted(self.root.glob("shard-*.pt")):
            shard_records = torch.load(
                shard, map_location="cpu", weights_only=False
            )
            total_loaded_blocks += len(shard_records)
            if (
                shard.name in shard_block_counts
                and len(shard_records) != shard_block_counts[shard.name]
            ):
                raise RuntimeError(f"canonical shard block count mismatch: {shard}")
            for record in shard_records:
                if split is not None and record["split"] != split:
                    continue
                if allowed_domains is not None and record["domain"] not in allowed_domains:
                    continue
                records.append(record)
        expected_total = self.metadata.get("num_blocks")
        if expected_total is not None and total_loaded_blocks != int(expected_total):
            raise RuntimeError(
                f"canonical block count mismatch: expected {expected_total}, "
                f"loaded {total_loaded_blocks}"
            )
        greedy_witness_flags = [
            "base_greedy_ids" in record for record in records
        ]
        if any(greedy_witness_flags) and not all(greedy_witness_flags):
            raise RuntimeError(
                "canonical collection mixes records with and without the "
                "exact released-DFlash greedy-token witness"
            )
        declared_witness = self.metadata.get("candidate_zero_invariant")
        if declared_witness is not None and not all(greedy_witness_flags):
            raise RuntimeError(
                "canonical metadata declares a rank-zero witness but one or "
                "more records do not contain base_greedy_ids"
            )
        if all(greedy_witness_flags):
            for record in records:
                if not torch.equal(
                    record["base_topk_ids"][:, 0].long(),
                    record["base_greedy_ids"].long(),
                ):
                    raise RuntimeError(
                        "canonical rank zero differs from its exact "
                        "released-DFlash greedy-token witness"
                    )
            self.base_greedy_witness_status = "complete"
        else:
            self.base_greedy_witness_status = "none_legacy"
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
    if not torch.all(
        candidate_logits[..., :-1] >= candidate_logits[..., 1:]
    ):
        raise RuntimeError(
            "canonical DFlash candidates are not sorted by descending logit; "
            "candidate index zero must be the frozen DFlash top-1 action"
        )
    greedy_fields = ["base_greedy_ids" in record for record in records]
    if any(greedy_fields) and not all(greedy_fields):
        raise RuntimeError(
            "canonical batch mixes records with and without the exact "
            "released-DFlash greedy-token witness"
        )
    if all(greedy_fields):
        base_greedy_ids = torch.stack(
            [record["base_greedy_ids"].long() for record in records]
        )
        if not torch.equal(candidate_ids[..., 0], base_greedy_ids):
            raise RuntimeError(
                "candidate index zero differs from the stored exact "
                "released-DFlash greedy action"
            )
    gold_ids = torch.stack([record["gold_ids"].long() for record in records])
    matches = candidate_ids == gold_ids.unsqueeze(-1)
    gold_in_lattice = matches.any(dim=-1)
    gold_candidate_indices = matches.to(torch.int64).argmax(dim=-1)
    batch = {
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
    if all(greedy_fields):
        batch["base_greedy_ids"] = base_greedy_ids
    return batch


def validate_stored_canonical_contexts(
    records: list[dict[str, Any]], sample_id: str
) -> Tensor:
    """Validate prefix nesting and return the longest exact stored context.

    A canonical sample's later anchors must extend every earlier anchor.  This
    check catches corrupt or accidentally mixed shards before a same-anchor
    comparison feeds their contexts to a model.
    """

    if not records:
        raise ValueError(f"no canonical records for {sample_id}")
    if any("context_ids_before_anchor" not in record for record in records):
        raise ValueError(f"stored context is missing for {sample_id}")
    longest_record = max(
        records,
        key=lambda item: int(item["context_ids_before_anchor"].numel()),
    )
    longest_context_ids = longest_record["context_ids_before_anchor"].long()
    for record in records:
        context_ids = record["context_ids_before_anchor"].long()
        context_length = int(context_ids.numel())
        if int(record.get("context_length", context_length)) != context_length:
            raise RuntimeError(f"stored context length mismatch for {sample_id}")
        if not torch.equal(context_ids, longest_context_ids[:context_length]):
            raise RuntimeError(
                f"stored contexts are not prefix-nested for {sample_id}"
            )
        if context_length >= int(longest_context_ids.numel()):
            continue
        if int(longest_context_ids[context_length]) != int(
            record["anchor_token_id"]
        ):
            raise RuntimeError(
                f"stored anchor is inconsistent with longer context for {sample_id}"
            )
        available_gold = min(
            int(record["gold_ids"].numel()),
            int(longest_context_ids.numel()) - context_length - 1,
        )
        if not torch.equal(
            record["gold_ids"][:available_gold].long(),
            longest_context_ids[
                context_length + 1 : context_length + 1 + available_gold
            ],
        ):
            raise RuntimeError(
                f"stored gold is inconsistent with longer context for {sample_id}"
            )
    return longest_context_ids
