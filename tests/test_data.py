import unittest

import torch

from sph.data import collate_canonical_blocks


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


if __name__ == "__main__":
    unittest.main()
