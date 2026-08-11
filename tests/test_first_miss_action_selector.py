from __future__ import annotations

import itertools
import unittest

import torch

from sph.first_miss_action_selector import (
    FirstMissActionSelector,
    action_logits_from_scores,
    canonical_first_miss_actions,
    decode_action_indices,
    encode_edit_actions,
    first_miss_action_loss,
    num_first_miss_actions,
    realized_prefix_lengths,
)
from sph.global_direct_selector import GlobalDirectCandidateSelector


def make_small_model() -> FirstMissActionSelector:
    return FirstMissActionSelector(
        GlobalDirectCandidateSelector(
            hidden_size=8,
            max_positions=3,
            max_candidates=4,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            scope="global",
            mixer="axial",
            node_encoder="additive",
            initialization_seed=17,
        )
    )


def make_model_inputs(
    *, requires_grad: bool = False
) -> dict[str, torch.Tensor]:
    torch.manual_seed(19)
    batch, length, candidates, hidden_size = 2, 3, 4, 8
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0]).expand(
        batch, length, candidates
    ).clone()
    inputs = {
        "hidden": torch.randn(batch, length, hidden_size),
        "candidate_embeddings": torch.randn(
            batch, length, candidates, hidden_size
        ),
        "candidate_logits": logits,
        "base_logsumexp": torch.logsumexp(
            torch.cat([logits, torch.zeros(batch, length, 3)], dim=-1),
            dim=-1,
        ),
        "anchor_embeddings": torch.randn(batch, hidden_size),
    }
    if requires_grad:
        inputs = {
            name: value.clone().requires_grad_(True)
            for name, value in inputs.items()
        }
    return inputs


class FirstMissActionSemanticsTest(unittest.TestCase):
    def test_target_handles_repair_keep_and_out_of_k(self) -> None:
        gold_ranks = torch.tensor(
            [
                [0, 2, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        available = torch.tensor(
            [
                [True, True, True],
                [True, True, True],
                [True, False, True],
            ]
        )
        actions = canonical_first_miss_actions(
            gold_ranks, available, candidates=4
        )
        expected_edit = 1 + 1 * 3 + (2 - 1)
        torch.testing.assert_close(
            actions, torch.tensor([expected_edit, 0, 0])
        )

    def test_action_encoding_and_decoding_are_bijective(self) -> None:
        length, candidates = 3, 4
        action_count = num_first_miss_actions(length, candidates)
        actions = torch.arange(action_count)
        paths = decode_action_indices(
            actions, length=length, candidates=candidates
        )
        self.assertEqual(paths.shape, (action_count, length))
        self.assertTrue(torch.equal(paths[0], torch.zeros(length)))
        self.assertTrue(
            torch.all((paths > 0).sum(dim=-1)[1:].eq(1))
        )
        for action in range(1, action_count):
            nonzero = paths[action].nonzero().squeeze(1)
            position = nonzero[0]
            rank = paths[action, position]
            encoded = encode_edit_actions(
                position[None],
                rank[None],
                length=length,
                candidates=candidates,
            )
            self.assertEqual(int(encoded), action)

    def test_canonical_target_is_pointwise_eal_optimal_exhaustively(
        self,
    ) -> None:
        length, candidates = 3, 3
        # State 0: base correct; 1/2: gold at that non-base rank; 3: out of K.
        for states in itertools.product(range(4), repeat=length):
            candidate_ids = torch.empty(1, length, candidates, dtype=torch.long)
            gold_ids = torch.arange(100, 100 + length)[None]
            gold_ranks = torch.zeros(1, length, dtype=torch.long)
            available = torch.ones(1, length, dtype=torch.bool)
            for position, state in enumerate(states):
                gold = int(gold_ids[0, position])
                ids = [1000 + 10 * position + rank for rank in range(candidates)]
                if state < candidates:
                    ids[state] = gold
                    gold_ranks[0, position] = state
                else:
                    available[0, position] = False
                candidate_ids[0, position] = torch.tensor(ids)

            target = canonical_first_miss_actions(
                gold_ranks, available, candidates=candidates
            )
            actions = torch.arange(
                num_first_miss_actions(length, candidates)
            )
            paths = decode_action_indices(
                actions, length=length, candidates=candidates
            )
            rewards = realized_prefix_lengths(
                paths,
                candidate_ids.expand(paths.shape[0], -1, -1),
                gold_ids.expand(paths.shape[0], -1),
            )
            self.assertEqual(
                int(rewards[int(target)]), int(rewards.max()), states
            )

    def test_keep_is_canonical_for_neutral_after_miss_ties(self) -> None:
        gold_ranks = torch.tensor([[0, 0, 2]])
        available = torch.tensor([[False, True, True]])
        target = canonical_first_miss_actions(
            gold_ranks, available, candidates=3
        )
        self.assertEqual(int(target), 0)

    def test_action_logits_are_keep_plus_score_differences(self) -> None:
        scores = torch.tensor(
            [[[4.0, 3.0, 1.0], [2.0, 2.0, 0.0]]]
        )
        got = action_logits_from_scores(scores)
        expected = torch.tensor([[0.0, -1.0, -3.0, 0.0, -2.0]])
        torch.testing.assert_close(got, expected)
        actions = got.argmax(dim=-1)
        self.assertEqual(int(actions), 0)

    def test_invalid_actions_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            decode_action_indices(
                torch.tensor([7]), length=2, candidates=3
            )
        with self.assertRaisesRegex(ValueError, "non-base"):
            encode_edit_actions(
                torch.tensor([0]),
                torch.tensor([0]),
                length=2,
                candidates=3,
            )


class FirstMissActionGradientTest(unittest.TestCase):
    def test_identity_then_upstream_gradient_contract(self) -> None:
        model = make_small_model()
        inputs = make_model_inputs(requires_grad=True)
        gold_ranks = torch.tensor([[1, 0, 0], [2, 0, 0]])
        available = torch.ones_like(gold_ranks, dtype=torch.bool)

        output = model(**inputs)
        base_scores = (
            inputs["candidate_logits"].detach()
            - inputs["base_logsumexp"].detach().unsqueeze(-1)
        )
        torch.testing.assert_close(output.direct_output.scores, base_scores)
        actions = output.action_logits.argmax(dim=-1)
        self.assertTrue(torch.equal(actions, torch.zeros_like(actions)))

        first_loss = first_miss_action_loss(
            output, gold_ranks, available
        ).loss
        first_loss.backward()
        projection = model.backbone.residual_projection.weight
        self.assertIsNotNone(projection.grad)
        self.assertTrue(torch.isfinite(projection.grad).all())
        self.assertGreater(float(projection.grad.abs().sum()), 0.0)
        upstream = [
            parameter.grad
            for name, parameter in model.backbone.named_parameters()
            if name != "residual_projection.weight"
            and parameter.grad is not None
        ]
        self.assertTrue(
            all(float(gradient.abs().sum()) == 0.0 for gradient in upstream)
        )
        self.assertTrue(
            all(value.grad is None for value in inputs.values())
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        second_loss = first_miss_action_loss(
            model(**inputs), gold_ranks, available
        ).loss
        second_loss.backward()
        projection_gradient = (
            model.backbone.residual_projection.weight.grad
        )
        self.assertIsNotNone(projection_gradient)
        self.assertTrue(torch.isfinite(projection_gradient).all())
        self.assertGreater(
            float(projection_gradient.abs().sum()), 0.0
        )
        upstream = [
            parameter.grad
            for name, parameter in model.backbone.named_parameters()
            if name != "residual_projection.weight"
            and parameter.grad is not None
        ]
        self.assertTrue(
            any(float(gradient.abs().sum()) > 0.0 for gradient in upstream)
        )
        self.assertTrue(
            all(
                gradient is None or torch.isfinite(gradient).all()
                for gradient in upstream
            )
        )
        self.assertTrue(
            all(value.grad is None for value in inputs.values())
        )

    def test_loss_uses_canonical_action(self) -> None:
        model = make_small_model()
        output = model(**make_model_inputs())
        gold_ranks = torch.tensor([[0, 2, 0], [1, 0, 0]])
        available = torch.ones_like(gold_ranks, dtype=torch.bool)
        loss = first_miss_action_loss(output, gold_ranks, available)
        expected = canonical_first_miss_actions(
            gold_ranks, available, candidates=4
        )
        torch.testing.assert_close(loss.target_actions, expected)
        self.assertTrue(torch.isfinite(loss.loss))


if __name__ == "__main__":
    unittest.main()
