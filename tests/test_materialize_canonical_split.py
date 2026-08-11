from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.materialize_canonical_prompt_subset import sha256_file
from scripts.materialize_canonical_split import materialize
from sph.data import CanonicalBlockDataset


def _record(sample_id: str, domain: str, split: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "domain": domain,
        "split": split,
        "base_topk_ids": torch.tensor([[1, 2], [3, 4]]),
        "base_topk_logits": torch.tensor([[2.0, 1.0], [2.0, 1.0]]),
        "base_logsumexp": torch.tensor([2.1, 2.1]),
        "parallel_hidden": torch.zeros(2, 3),
        "anchor_token_id": 0,
        "gold_ids": torch.tensor([1, 3]),
    }


class CanonicalSplitMaterializationTest(unittest.TestCase):
    def _make_source(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        manifest = root / "manifest.jsonl"
        manifest_records = [
            {"sample_id": "train-a", "domain": "math", "split": "train"},
            {
                "sample_id": "select-a",
                "domain": "code",
                "split": "validation_select",
            },
            {
                "sample_id": "gate-a",
                "domain": "chat",
                "split": "validation_gate",
            },
            {
                "sample_id": "select-empty",
                "domain": "math",
                "split": "validation_select",
            },
        ]
        manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in manifest_records),
            encoding="utf-8",
        )
        records = [
            _record("train-a", "math", "train"),
            _record("select-a", "code", "validation_select"),
            _record("select-a", "code", "validation_select"),
            _record("gate-a", "chat", "validation_gate"),
        ]
        shard = source / "shard-00000.pt"
        torch.save(records, shard)
        metadata = {
            "format_version": 2,
            "collection_complete": True,
            "manifest": str(manifest),
            "block_size": 2,
            "draft_positions": 2,
            "top_k": 2,
            "anchors_per_sample": 1,
            "continuation_tokens": 2,
            "attention_implementation": "sdpa",
            "dtype": "bfloat16",
            "target_layer_ids": [1],
            "num_manifest_samples": 4,
            "num_collected_samples": 3,
            "num_blocks": 4,
            "block_counts_by_domain_split": {
                "chat/validation_gate": 1,
                "code/validation_select": 2,
                "math/train": 1,
            },
            "shards": [
                {
                    "path": shard.name,
                    "blocks": len(records),
                    "bytes": shard.stat().st_size,
                    "sha256": sha256_file(shard),
                }
            ],
            "provenance": {"target_files": [], "draft_files": []},
        }
        (source / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return source

    def test_isolates_only_requested_split_and_rewrites_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._make_source(root)
            output = root / "select-only"
            report = materialize(
                argparse.Namespace(
                    source=source,
                    output=output,
                    split="validation_select",
                    expected_prompts=1,
                    expected_blocks=2,
                    shard_blocks=1,
                )
            )
            dataset = CanonicalBlockDataset(output)
            selected_manifest = [
                json.loads(line)
                for line in (output / "selected_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        self.assertEqual(len(dataset), 2)
        self.assertEqual({record["split"] for record in dataset.records}, {"validation_select"})
        self.assertEqual({record["sample_id"] for record in dataset.records}, {"select-a"})
        self.assertEqual([record["sample_id"] for record in selected_manifest], ["select-a"])
        self.assertEqual(report["num_manifest_samples"], 1)
        self.assertEqual(
            report["provenance"]["split_materialization"]["source_manifest"]["declared_split_prompts"],
            2,
        )

    def test_fails_closed_on_record_manifest_split_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._make_source(root)
            shard = source / "shard-00000.pt"
            records = torch.load(shard, weights_only=False)
            records[1] = copy.deepcopy(records[1])
            records[1]["split"] = "validation_gate"
            torch.save(records, shard)
            metadata_path = source / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["shards"][0]["bytes"] = shard.stat().st_size
            metadata["shards"][0]["sha256"] = sha256_file(shard)
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(RuntimeError, "split differs"):
                materialize(
                    argparse.Namespace(
                        source=source,
                        output=root / "bad-output",
                        split="validation_select",
                        expected_prompts=1,
                        expected_blocks=2,
                        shard_blocks=2,
                    )
                )


if __name__ == "__main__":
    unittest.main()
