from __future__ import annotations

import itertools
import unittest

import torch

from sph.candidate_lattice_selector import (
    CandidateLatticeSelector,
    candidate_selector_loss,
    dpace_position_weights,
    first_divergence_margin,
    prefix_candidate_mask,
    viterbi_decode,
)


def make_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(41)
    batch, length, candidates, hidden_size, vocab = 2, 3, 4, 12, 31
    candidate_ids = torch.randint(vocab, (batch, length, candidates))
    candidate_logits = torch.randn(batch, length, candidates)
    return {
        "hidden": torch.randn(batch, length, hidden_size),
        "candidate_ids": candidate_ids,
        "candidate_embeddings": torch.randn(
            batch, length, candidates, hidden_size
        ),
        "candidate_logits": candidate_logits,
        "base_logsumexp": torch.logsumexp(
            torch.cat(
                [candidate_logits, torch.randn(batch, length, 5)], dim=-1
            ),
            dim=-1,
        ),
        "anchor_ids": torch.randint(vocab, (batch,)),
    }


class CandidateLatticeSelectorTest(unittest.TestCase):
    def test_matched_scopes_have_identical_parameter_counts(self) -> None:
        common = {
            "hidden_size": 12,
            "vocab_size": 31,
            "max_positions": 3,
            "max_candidates": 4,
            "model_dim": 16,
            "token_dim": 8,
            "transition_dim": 4,
            "num_heads": 4,
            "num_layers": 2,
        }
        local = CandidateLatticeSelector(scope="local", **common)
        global_head = CandidateLatticeSelector(scope="global", **common)
        self.assertEqual(
            sum(parameter.numel() for parameter in local.parameters()),
            sum(parameter.numel() for parameter in global_head.parameters()),
        )

    def test_local_scope_cannot_observe_other_positions(self) -> None:
        inputs = make_inputs()
        model = CandidateLatticeSelector(
            hidden_size=12,
            vocab_size=31,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            token_dim=8,
            transition_dim=4,
            num_heads=4,
            num_layers=2,
            scope="local",
        ).eval()
        first = model(**inputs).unary_scores[:, 0]
        changed = {key: value.clone() for key, value in inputs.items()}
        changed["candidate_ids"][:, 2] = changed["candidate_ids"][:, 2].roll(
            1, dims=-1
        )
        changed["candidate_embeddings"][:, 2] = changed[
            "candidate_embeddings"
        ][:, 2].roll(1, dims=-2)
        changed["candidate_logits"][:, 2] = changed[
            "candidate_logits"
        ][:, 2].roll(1, dims=-1)
        second = model(**changed).unary_scores[:, 0]
        torch.testing.assert_close(first, second)

    def test_loss_is_finite_and_reaches_all_main_branches(self) -> None:
        inputs = make_inputs()
        model = CandidateLatticeSelector(
            hidden_size=12,
            vocab_size=31,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            token_dim=8,
            transition_dim=4,
            num_heads=4,
            num_layers=1,
            scope="global",
        )
        output = model(**inputs)
        gold_indices = torch.tensor([[0, 1, 0], [2, 0, 3]])
        in_lattice = torch.tensor(
            [[True, True, False], [True, True, True]]
        )
        gold_ids = inputs["candidate_ids"].gather(
            -1, gold_indices.unsqueeze(-1)
        ).squeeze(-1)
        loss = candidate_selector_loss(
            output,
            inputs["candidate_ids"],
            gold_ids,
            gold_indices,
            in_lattice,
        )
        self.assertTrue(torch.isfinite(loss.loss))
        loss.loss.backward()
        self.assertIsNotNone(model.residual_projection.weight.grad)
        self.assertIsNotNone(model.in_lattice_projection.weight.grad)
        self.assertIsNotNone(model.base_correct_projection.weight.grad)

    def test_prefix_mask_stops_before_first_other(self) -> None:
        in_lattice = torch.tensor(
            [[True, True, False, True], [True, True, True, True]]
        )
        expected = torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )
        torch.testing.assert_close(
            prefix_candidate_mask(in_lattice), expected
        )

    def test_dpace_weights_are_masked_and_normalized(self) -> None:
        probabilities = torch.tensor(
            [[0.8, 0.7, 0.2, 0.9], [0.6, 0.5, 0.4, 0.3]]
        )
        active = torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )
        weights = dpace_position_weights(probabilities, active)
        self.assertEqual(float(weights[0, 2:].sum()), 0.0)
        self.assertAlmostEqual(
            float(weights.sum()), float(active.sum()), places=5
        )
        # Appending censored positions cannot change the relative weights of
        # the observable prefix.
        short = dpace_position_weights(
            probabilities[:1, :2], active[:1, :2]
        )
        long_prefix = weights[:1, :2] / weights[:1, :2].sum()
        short_prefix = short / short.sum()
        torch.testing.assert_close(long_prefix, short_prefix)

    def test_viterbi_matches_brute_force(self) -> None:
        torch.manual_seed(43)
        edge_scores = torch.randn(1, 4, 3, 3)
        decoded = viterbi_decode(edge_scores)
        scored = []
        for values in itertools.product(range(3), repeat=4):
            previous = 0
            score = 0.0
            for position, current in enumerate(values):
                score += float(
                    edge_scores[0, position, previous, current]
                )
                previous = current
            scored.append((score, values))
        score, values = max(scored)
        self.assertEqual(decoded.path.tolist(), [list(values)])
        self.assertAlmostEqual(float(decoded.score), score, places=5)

    def test_first_divergence_margin_uses_the_decision_row(self) -> None:
        probabilities = torch.tensor(
            [
                [
                    [[0.7, 0.3], [0.7, 0.3]],
                    [[0.2, 0.8], [0.6, 0.4]],
                    [[0.5, 0.5], [0.5, 0.5]],
                ]
            ]
        )
        path = torch.tensor([[0, 1, 0]])
        margin = first_divergence_margin(probabilities.log(), path)
        self.assertAlmostEqual(
            float(margin), math_log_ratio := float(torch.log(torch.tensor(4.0)))
        )
        self.assertGreater(math_log_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
