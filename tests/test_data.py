import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from sph.data import (
    CanonicalBlockDataset,
    collate_canonical_blocks,
    validate_stored_canonical_contexts,
)


class CanonicalDataTest(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_protocol_v2_shard_integrity_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shard-00000.pt"
            record = {
                "sample_id": "sample",
                "domain": "math",
                "split": "train",
            }
            torch.save([record], shard)
            metadata = {
                "format_version": 2,
                "collection_complete": True,
                "num_blocks": 1,
                "shards": [
                    {
                        "path": shard.name,
                        "blocks": 1,
                        "bytes": shard.stat().st_size,
                        "sha256": self._sha256(shard),
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))
            dataset = CanonicalBlockDataset(root, split="train")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(
                dataset.base_greedy_witness_status, "none_legacy"
            )

            shard.write_bytes(shard.read_bytes() + b"corruption")
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                CanonicalBlockDataset(root, split="train")

    def test_collection_rejects_mixed_greedy_witness_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shard-00000.pt"
            records = [
                {
                    "sample_id": "a",
                    "domain": "chat",
                    "split": "train",
                    "base_topk_ids": torch.tensor([[1, 2]]),
                    "base_greedy_ids": torch.tensor([1]),
                },
                {
                    "sample_id": "b",
                    "domain": "chat",
                    "split": "train",
                    "base_topk_ids": torch.tensor([[3, 4]]),
                },
            ]
            torch.save(records, shard)
            metadata = {
                "format_version": 2,
                "collection_complete": True,
                "num_blocks": 2,
                "shards": [
                    {
                        "path": shard.name,
                        "blocks": 2,
                        "bytes": shard.stat().st_size,
                        "sha256": self._sha256(shard),
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(RuntimeError, "mixes records"):
                CanonicalBlockDataset(root, split="train")

    def test_collate_builds_gold_indices_and_other_mask(self) -> None:
        record = {
            "sample_id": "sample",
            "domain": "math",
            "parallel_hidden": torch.randn(2, 4),
            "anchor_token_id": 10,
            "base_topk_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "base_greedy_ids": torch.tensor([1, 4]),
            "base_topk_logits": torch.tensor(
                [[3.0, 2.0, 1.0], [4.0, 1.0, -2.0]]
            ),
            "base_logsumexp": torch.randn(2),
            "gold_ids": torch.tensor([2, 9]),
        }
        batch = collate_canonical_blocks([record], candidate_k=2)
        torch.testing.assert_close(
            batch["gold_in_lattice"], torch.tensor([[True, False]])
        )
        torch.testing.assert_close(
            batch["gold_candidate_indices"], torch.tensor([[1, 0]])
        )
        self.assertEqual(tuple(batch["candidate_ids"].shape), (1, 2, 2))
        torch.testing.assert_close(
            batch["base_greedy_ids"], torch.tensor([[1, 4]])
        )

    def test_collate_rejects_candidates_without_top1_at_index_zero(
        self,
    ) -> None:
        record = {
            "sample_id": "sample",
            "domain": "math",
            "parallel_hidden": torch.randn(1, 4),
            "anchor_token_id": 10,
            "base_topk_ids": torch.tensor([[1, 2, 3]]),
            "base_topk_logits": torch.tensor([[1.0, 2.0, 0.0]]),
            "base_logsumexp": torch.tensor([3.0]),
            "gold_ids": torch.tensor([1]),
        }
        with self.assertRaisesRegex(RuntimeError, "not sorted"):
            collate_canonical_blocks([record], candidate_k=3)

    def test_collate_rejects_rank_zero_that_differs_from_greedy_witness(
        self,
    ) -> None:
        record = {
            "sample_id": "sample",
            "domain": "math",
            "parallel_hidden": torch.randn(1, 4),
            "anchor_token_id": 10,
            "base_topk_ids": torch.tensor([[1, 2, 3]]),
            "base_greedy_ids": torch.tensor([2]),
            "base_topk_logits": torch.tensor([[3.0, 2.0, 1.0]]),
            "base_logsumexp": torch.tensor([3.5]),
            "gold_ids": torch.tensor([1]),
        }
        with self.assertRaisesRegex(RuntimeError, "exact released-DFlash"):
            collate_canonical_blocks([record], candidate_k=3)

    def test_stored_contexts_are_prefix_nested(self) -> None:
        records = [
            {
                "context_ids_before_anchor": torch.tensor([1, 2]),
                "context_length": 2,
                "anchor_token_id": 3,
                "gold_ids": torch.tensor([4, 5, 6]),
            },
            {
                "context_ids_before_anchor": torch.tensor([1, 2, 3, 4]),
                "context_length": 4,
                "anchor_token_id": 5,
                "gold_ids": torch.tensor([6, 7, 8]),
            },
        ]
        longest = validate_stored_canonical_contexts(records, "sample")
        torch.testing.assert_close(longest, torch.tensor([1, 2, 3, 4]))

    def test_stored_context_corruption_is_rejected(self) -> None:
        records = [
            {
                "context_ids_before_anchor": torch.tensor([1, 2]),
                "anchor_token_id": 9,
                "gold_ids": torch.tensor([4]),
            },
            {
                "context_ids_before_anchor": torch.tensor([1, 2, 3, 4]),
                "anchor_token_id": 5,
                "gold_ids": torch.tensor([6]),
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "stored anchor"):
            validate_stored_canonical_contexts(records, "sample")


if __name__ == "__main__":
    unittest.main()
