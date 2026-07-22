import unittest

import torch

from sph.candidate_ceiling import (
    accepted_draft_prefix_lengths,
    first_top1_miss_gold_rank,
    gold_candidate_ranks,
    gold_in_candidates,
    prefix_coverage,
)


class CandidateCeilingTest(unittest.TestCase):
    def setUp(self):
        self.topk = torch.tensor(
            [
                [[1, 9, 8], [2, 7, 6], [5, 3, 4], [4, 0, 2]],
                [[8, 1, 2], [7, 2, 3], [6, 3, 4], [5, 4, 0]],
            ]
        )
        self.gold = torch.tensor([[1, 2, 3, 4], [8, 2, 6, 9]])

    def test_gold_coverage_and_prefix(self):
        coverage = gold_in_candidates(self.topk, self.gold, k=2)
        torch.testing.assert_close(
            coverage,
            torch.tensor([[True, True, True, True], [True, True, True, False]]),
        )
        torch.testing.assert_close(
            prefix_coverage(coverage),
            torch.tensor([[True, True, True, True], [True, True, True, False]]),
        )

    def test_accepted_prefix_lengths(self):
        matches = torch.tensor(
            [[True, True, False, True], [False, True, True, True], [True] * 4]
        )
        torch.testing.assert_close(
            accepted_draft_prefix_lengths(matches), torch.tensor([2, 0, 4])
        )

    def test_gold_ranks(self):
        torch.testing.assert_close(
            gold_candidate_ranks(self.topk, self.gold),
            torch.tensor([[1, 1, 2, 1], [1, 2, 1, 4]]),
        )

    def test_first_top1_miss_rank(self):
        torch.testing.assert_close(
            first_top1_miss_gold_rank(self.topk, self.gold), torch.tensor([2, 2])
        )
        perfect_gold = self.topk[..., 0]
        torch.testing.assert_close(
            first_top1_miss_gold_rank(self.topk, perfect_gold), torch.tensor([0, 0])
        )

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            gold_in_candidates(self.topk, self.gold, k=0)
        with self.assertRaises(ValueError):
            gold_in_candidates(self.topk, self.gold, k=4)


if __name__ == "__main__":
    unittest.main()
