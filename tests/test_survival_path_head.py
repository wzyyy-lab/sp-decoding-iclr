from __future__ import annotations

import itertools
import unittest

import torch

from sph.survival_path_head import (
    BidirectionalSurvivalPathHead,
    SurvivalPathHead,
    absorbing_prefix_crf_conditionals,
    absorbing_prefix_crf_nll,
    chain_crf_conditionals,
    chain_crf_nll,
    expected_prefix_utility,
    global_survival_decode,
    greedy_markov_decode,
    prefix_censored_nll,
    survival_decode,
    viterbi_decode,
)


class SurvivalDecodeTest(unittest.TestCase):
    def test_matches_brute_force(self) -> None:
        torch.manual_seed(7)
        batch, length, candidates = 1, 4, 3
        raw = torch.randn(batch, length, candidates, candidates)
        # Leave some probability for the full-vocabulary outside state.
        log_probs = torch.log_softmax(
            torch.cat([raw, torch.zeros(batch, length, candidates, 1)], dim=-1),
            dim=-1,
        )[..., :candidates]

        decoded = survival_decode(log_probs)
        best_path = None
        best_utility = -1.0
        for values in itertools.product(range(candidates), repeat=length):
            path = torch.tensor([values])
            utility = expected_prefix_utility(log_probs, path).item()
            if utility > best_utility:
                best_utility = utility
                best_path = path

        self.assertTrue(torch.equal(decoded.path, best_path))
        self.assertAlmostEqual(decoded.predicted_utility.item(), best_utility, places=6)

    def test_future_can_change_first_token(self) -> None:
        # Local greedy chooses A (0.51 > 0.49).  A has a poor suffix, whereas B
        # has a nearly deterministic suffix, so prefix-utility decoding chooses B.
        probabilities = torch.tensor(
            [
                [
                    [[0.51, 0.49], [0.51, 0.49]],
                    [[0.10, 0.10], [0.99, 0.005]],
                ]
            ]
        )
        log_probs = probabilities.log()

        greedy = greedy_markov_decode(log_probs)
        survival = survival_decode(log_probs)

        self.assertEqual(greedy.path[0, 0].item(), 0)
        self.assertEqual(survival.path[0, 0].item(), 1)
        self.assertGreater(
            survival.predicted_utility.item(), greedy.predicted_utility.item()
        )

    def test_viterbi_is_a_distinct_control(self) -> None:
        probabilities = torch.tensor(
            [
                [
                    [[0.60, 0.40], [0.60, 0.40]],
                    [[0.40, 0.39], [0.90, 0.01]],
                ]
            ]
        )
        log_probs = probabilities.log()
        survival = survival_decode(log_probs)
        viterbi = viterbi_decode(log_probs)

        self.assertFalse(torch.equal(survival.path, viterbi.path))
        self.assertGreaterEqual(
            survival.predicted_utility.item(), viterbi.predicted_utility.item()
        )

    def test_global_crf_and_survival_decode_match_enumeration(self) -> None:
        torch.manual_seed(11)
        batch, length, candidates = 1, 3, 2
        edge_scores = torch.randn(batch, length, candidates, candidates)
        crf = chain_crf_conditionals(edge_scores)

        paths = []
        scores = []
        for values in itertools.product(range(candidates), repeat=length):
            path = torch.tensor([values])
            paths.append(path)
            previous = 0
            score = 0.0
            for position, current in enumerate(values):
                score += edge_scores[0, position, previous, current].item()
                previous = current
            scores.append(score)

        brute_log_partition = torch.logsumexp(torch.tensor(scores), dim=0)
        self.assertAlmostEqual(
            crf.log_partition.item(), brute_log_partition.item(), places=6
        )

        decoded = global_survival_decode(edge_scores)
        utilities = [
            expected_prefix_utility(crf.log_conditionals, path).item()
            for path in paths
        ]
        best = max(range(len(paths)), key=utilities.__getitem__)
        self.assertTrue(torch.equal(decoded.path, paths[best]))
        self.assertAlmostEqual(
            decoded.predicted_utility.item(), utilities[best], places=6
        )

    def test_crf_nll_is_path_probability(self) -> None:
        torch.manual_seed(13)
        edge_scores = torch.randn(1, 3, 2, 2)
        path = torch.tensor([[1, 0, 1]])
        crf = chain_crf_conditionals(edge_scores)
        nll = chain_crf_nll(edge_scores, path)
        selected = []
        previous = 0
        for position, current in enumerate(path[0].tolist()):
            selected.append(crf.log_conditionals[0, position, previous, current])
            previous = current
        self.assertTrue(torch.allclose(nll, -torch.stack(selected).sum()[None]))

    def test_absorbing_prefix_partition_matches_variable_length_enumeration(self) -> None:
        torch.manual_seed(23)
        batch, length, candidates = 1, 3, 2
        edge_log_weights = torch.randn(batch, length, candidates, candidates)
        outside_log_weights = torch.randn(batch, length, candidates)
        crf = absorbing_prefix_crf_conditionals(
            edge_log_weights,
            outside_log_weights,
            torch.zeros(batch, length),
        )

        scores = []
        for prefix_length in range(length + 1):
            for values in itertools.product(range(candidates), repeat=prefix_length):
                previous = 0
                score = torch.tensor(0.0)
                for position, current in enumerate(values):
                    score = score + edge_log_weights[
                        0, position, previous, current
                    ]
                    previous = current
                if prefix_length < length:
                    score = score + outside_log_weights[
                        0, prefix_length, previous
                    ]
                scores.append(score)

        brute_log_partition = torch.logsumexp(torch.stack(scores), dim=0)
        torch.testing.assert_close(crf.log_partition[0], brute_log_partition)
        total = torch.exp(crf.log_conditionals).sum(dim=-1) + torch.exp(
            crf.outside_log_conditionals
        )
        torch.testing.assert_close(total, torch.ones_like(total))

    def test_absorbing_prefix_censored_nll_uses_first_other(self) -> None:
        torch.manual_seed(29)
        edge_scores = torch.randn(1, 3, 2, 2)
        outside_log_mass = torch.randn(1, 3, 2)
        base_logsumexp = torch.randn(1, 3)
        gold_indices = torch.tensor([[1, 0, 1]])
        in_lattice = torch.tensor([[True, True, False]])
        crf = absorbing_prefix_crf_conditionals(
            edge_scores, outside_log_mass, base_logsumexp
        )
        nll = absorbing_prefix_crf_nll(
            edge_scores,
            outside_log_mass,
            base_logsumexp,
            gold_indices,
            in_lattice,
        )
        expected = -(
            crf.log_conditionals[0, 0, 0, 1]
            + crf.log_conditionals[0, 1, 1, 0]
            + crf.outside_log_conditionals[0, 2, 0]
        )
        torch.testing.assert_close(nll[0], expected)

    def test_absorbing_prefix_nll_has_finite_gradients(self) -> None:
        torch.manual_seed(31)
        edge_scores = torch.randn(2, 4, 3, 3, requires_grad=True)
        outside_log_mass = torch.randn(2, 4)
        base_logsumexp = torch.randn(2, 4)
        gold_indices = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 1]])
        in_lattice = torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )
        loss = absorbing_prefix_crf_nll(
            edge_scores,
            outside_log_mass,
            base_logsumexp,
            gold_indices,
            in_lattice,
        ).mean()
        loss.backward()
        self.assertIsNotNone(edge_scores.grad)
        self.assertTrue(torch.isfinite(edge_scores.grad).all())


class SurvivalPathHeadTest(unittest.TestCase):
    def test_zero_initialized_residual_preserves_base_distribution(self) -> None:
        torch.manual_seed(3)
        batch, length, candidates, dim = 2, 5, 4, 16
        head = SurvivalPathHead(dim, rank=8)

        hidden = torch.randn(batch, length, dim)
        anchor = torch.randn(batch, dim)
        candidate_embeddings = torch.randn(batch, length, candidates, dim)
        candidate_logits = torch.randn(batch, length, candidates)
        outside_logits = torch.randn(batch, length, 7)
        full_logsumexp = torch.logsumexp(
            torch.cat([candidate_logits, outside_logits], dim=-1), dim=-1
        )

        output = head(
            hidden,
            anchor,
            candidate_embeddings,
            candidate_logits,
            full_logsumexp,
        )
        expected = candidate_logits - full_logsumexp[:, :, None]
        expected = expected[:, :, None, :].expand_as(output.log_probs)

        self.assertTrue(torch.allclose(output.residual_logits, torch.zeros_like(output.residual_logits)))
        self.assertTrue(torch.allclose(output.log_probs, expected, atol=1e-6))

    def test_absorbing_prefix_crf_preserves_base_at_zero_residual(self) -> None:
        torch.manual_seed(37)
        batch, length, candidates, dim = 2, 5, 4, 16
        head = SurvivalPathHead(dim, rank=8)
        candidate_logits = torch.randn(batch, length, candidates)
        outside_logits = torch.randn(batch, length, 11)
        full_logsumexp = torch.logsumexp(
            torch.cat([candidate_logits, outside_logits], dim=-1), dim=-1
        )
        output = head(
            torch.randn(batch, length, dim),
            torch.randn(batch, dim),
            torch.randn(batch, length, candidates, dim),
            candidate_logits,
            full_logsumexp,
        )
        crf = absorbing_prefix_crf_conditionals(
            output.edge_scores,
            output.outside_log_mass,
            full_logsumexp,
        )
        expected_candidates = candidate_logits - full_logsumexp[:, :, None]
        expected_candidates = expected_candidates[:, :, None, :].expand_as(
            crf.log_conditionals
        )
        expected_outside = output.outside_log_mass - full_logsumexp
        expected_outside = expected_outside[:, :, None].expand_as(
            crf.outside_log_conditionals
        )
        torch.testing.assert_close(crf.log_conditionals, expected_candidates)
        torch.testing.assert_close(
            crf.outside_log_conditionals, expected_outside
        )
        torch.testing.assert_close(
            crf.log_partition, torch.zeros_like(crf.log_partition), atol=1e-6, rtol=0
        )
        expected_path = candidate_logits.argmax(dim=-1)
        torch.testing.assert_close(
            survival_decode(crf.log_conditionals).path, expected_path
        )

    def test_candidate_and_outside_mass_are_normalized(self) -> None:
        torch.manual_seed(4)
        batch, length, candidates, dim = 1, 3, 5, 12
        head = SurvivalPathHead(dim, rank=4)
        with torch.no_grad():
            head.residual_scale.fill_(0.7)

        candidate_logits = torch.randn(batch, length, candidates)
        outside_logits = torch.randn(batch, length, 11)
        full_logsumexp = torch.logsumexp(
            torch.cat([candidate_logits, outside_logits], dim=-1), dim=-1
        )
        output = head(
            torch.randn(batch, length, dim),
            torch.randn(batch, dim),
            torch.randn(batch, length, candidates, dim),
            candidate_logits,
            full_logsumexp,
        )

        adjusted_outside = torch.exp(
            output.outside_log_mass[:, :, None]
            - torch.logaddexp(
                torch.logsumexp(
                    candidate_logits[:, :, None, :] + output.residual_logits,
                    dim=-1,
                ),
                output.outside_log_mass[:, :, None],
            )
        )
        total = torch.exp(output.log_probs).sum(dim=-1) + adjusted_outside
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-6))

    def test_bidirectional_head_preserves_base_at_zero_residual(self) -> None:
        torch.manual_seed(17)
        batch, length, candidates, dim = 2, 6, 4, 16
        head = BidirectionalSurvivalPathHead(
            dim, rank=8, model_dim=16, num_heads=4
        )
        candidate_logits = torch.randn(batch, length, candidates)
        outside_logits = torch.randn(batch, length, 9)
        full_logsumexp = torch.logsumexp(
            torch.cat([candidate_logits, outside_logits], dim=-1), dim=-1
        )
        output = head(
            torch.randn(batch, length, dim),
            torch.randn(batch, dim),
            torch.randn(batch, length, candidates, dim),
            candidate_logits,
            full_logsumexp,
        )
        expected = candidate_logits - full_logsumexp[:, :, None]
        expected = expected[:, :, None].expand_as(output.log_probs)
        self.assertTrue(torch.allclose(output.log_probs, expected, atol=1e-6))
        total = torch.exp(output.log_probs).sum(dim=-1) + torch.exp(
            output.outside_log_probs
        )
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-6))

    def test_bidirectional_mixer_propagates_suffix_evidence(self) -> None:
        torch.manual_seed(19)
        batch, length, candidates, dim = 1, 5, 3, 12
        head = BidirectionalSurvivalPathHead(
            dim, rank=4, model_dim=12, num_heads=3
        )
        with torch.no_grad():
            head.residual_scale.fill_(1.0)
        hidden = torch.randn(batch, length, dim)
        anchor = torch.randn(batch, dim)
        embeddings = torch.randn(batch, length, candidates, dim)
        logits = torch.randn(batch, length, candidates)
        outside = torch.randn(batch, length, 7)
        logsumexp = torch.logsumexp(torch.cat([logits, outside], dim=-1), dim=-1)
        first = head(hidden, anchor, embeddings, logits, logsumexp)
        changed = embeddings.clone()
        changed[:, -1] = changed[:, -1] + 3.0
        second = head(hidden, anchor, changed, logits, logsumexp)
        self.assertFalse(
            torch.allclose(first.residual_logits[:, 0], second.residual_logits[:, 0])
        )

    def test_prefix_censored_nll_stops_after_other(self) -> None:
        probabilities = torch.tensor(
            [
                [
                    [[0.6, 0.3], [0.5, 0.2]],
                    [[0.7, 0.1], [0.4, 0.4]],
                    [[0.2, 0.2], [0.3, 0.3]],
                ]
            ]
        )
        outside = 1.0 - probabilities.sum(dim=-1)
        gold_indices = torch.tensor([[0, 0, 1]])
        in_lattice = torch.tensor([[True, False, True]])
        nll = prefix_censored_nll(
            probabilities.log(), outside.log(), gold_indices, in_lattice
        )
        expected = -torch.log(torch.tensor(0.6)) - torch.log(torch.tensor(0.2))
        self.assertTrue(torch.allclose(nll, expected[None]))


if __name__ == "__main__":
    unittest.main()
