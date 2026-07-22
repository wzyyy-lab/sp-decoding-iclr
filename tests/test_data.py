import unittest

import torch

from sph.data import (
    collate_canonical_blocks,
    validate_stored_canonical_contexts,
)


class CanonicalDataTest(unittest.TestCase):
    def test_collate_builds_gold_indices_and_other_mask(self) -> None:
        record = {
            "sample_id": "sample",
            "domain": "math",
            "parallel_hidden": torch.randn(2, 4),
            "anchor_token_id": 10,
            "base_topk_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "base_topk_logits": torch.randn(2, 3),
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
