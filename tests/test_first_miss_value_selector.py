from __future__ import annotations

import unittest

import torch

from sph.first_miss_action_selector import (
    decode_action_indices,
    encode_edit_actions,
    num_first_miss_actions,
)
from sph.first_miss_value_selector import (
    FirstMissValueOutput,
    FirstMissValueSelector,
    action_values_from_residual_scores,
    decode_strict_positive_actions,
    dense_signed_action_values,
    first_miss_value_loss,
)
from sph.global_direct_selector import (
    GlobalDirectCandidateSelector,
    GlobalDirectOutput,
)


def make_model() -> FirstMissValueSelector:
    return FirstMissValueSelector(
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
            initialization_seed=31,
        )
    )


def make_inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(23)
    return (
        torch.randn(3, 3, 8, generator=generator),
        torch.randn(3, 3, 4, 8, generator=generator),
        torch.randn(3, 3, 4, generator=generator),
        torch.randn(3, 3, generator=generator),
        torch.randn(3, 8, generator=generator),
    )


class DenseSignedValueSemanticsTest(unittest.TestCase):
    def test_dense_targets_equal_bruteforce_paths(self) -> None:
        generator = torch.Generator().manual_seed(17)
        batch, length, candidates = 11, 5, 6
        available = torch.rand(
            batch, length, generator=generator
        ).gt(0.2)
        ranks = torch.randint(
            0, candidates, (batch, length), generator=generator
        )
        ranks = torch.where(available, ranks, torch.zeros_like(ranks))
        got = dense_signed_action_values(
            ranks, available, candidates=candidates
        )

        actions = torch.arange(num_first_miss_actions(length, candidates))
        paths = decode_action_indices(
            actions, length=length, candidates=candidates
        )
        expected = torch.empty_like(got)
        for item in range(batch):
            accepted: list[int] = []
            for path in paths:
                correct = available[item] & path.eq(ranks[item])
                accepted.append(
                    int(correct.to(torch.int64).cumprod(dim=0).sum())
                )
            base = accepted[0]
            expected[item] = torch.tensor(
                [(value - base) / length for value in accepted]
            )
        torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
        self.assertTrue(torch.equal(got[:, 0], torch.zeros(batch)))

    def test_hand_fixtures_cover_benefit_neutral_and_harm(self) -> None:
        ranks = torch.tensor(
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]
        )
        available = torch.tensor(
            [
                [True, True, True],
                [True, True, True],
                [False, True, True],
            ]
        )
        values = dense_signed_action_values(
            ranks, available, candidates=4
        )
        repair = int(
            encode_edit_actions(
                torch.tensor(1),
                torch.tensor(1),
                length=3,
                candidates=4,
            )
        )
        harmful_first = int(
            encode_edit_actions(
                torch.tensor(0),
                torch.tensor(1),
                length=3,
                candidates=4,
            )
        )
        neutral_wrong_repair = int(
            encode_edit_actions(
                torch.tensor(1),
                torch.tensor(2),
                length=3,
                candidates=4,
            )
        )
        harmful_last = int(
            encode_edit_actions(
                torch.tensor(2),
                torch.tensor(1),
                length=3,
                candidates=4,
            )
        )
        self.assertAlmostEqual(float(values[0, repair]), 2.0 / 3.0)
        self.assertAlmostEqual(float(values[0, harmful_first]), -1.0 / 3.0)
        self.assertEqual(float(values[0, neutral_wrong_repair]), 0.0)
        self.assertEqual(float(values[1, harmful_first]), -1.0)
        self.assertAlmostEqual(float(values[1, harmful_last]), -1.0 / 3.0)
        self.assertTrue(torch.equal(values[2], torch.zeros_like(values[2])))

    def test_residual_differences_remove_position_gauge(self) -> None:
        residuals = torch.tensor(
            [[[1.0, 3.0, -2.0], [4.0, 4.0, 8.0]]]
        )
        offsets = torch.tensor([[[10.0], [-7.0]]])
        expected = torch.tensor([[0.0, 2.0, -3.0, 0.0, 4.0]])
        torch.testing.assert_close(
            action_values_from_residual_scores(residuals), expected
        )
        torch.testing.assert_close(
            action_values_from_residual_scores(residuals + offsets), expected
        )

    def test_strict_positive_decoder_keeps_zero_and_negative_ties(self) -> None:
        values = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, -0.1, -0.2],
                [0.0, 0.4, 0.4],
            ]
        )
        self.assertTrue(
            torch.equal(
                decode_strict_positive_actions(values),
                torch.tensor([0, 0, 1]),
            )
        )
        with self.assertRaisesRegex(ValueError, "KEEP"):
            decode_strict_positive_actions(torch.tensor([[0.1, 0.2]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            decode_strict_positive_actions(torch.tensor([[0.0, float("nan")]]))

    def test_loss_uses_every_nonkeep_action_uniformly(self) -> None:
        batch, length, candidates = 2, 15, 16
        residuals = torch.zeros(batch, length, candidates)
        direct = GlobalDirectOutput(
            scores=residuals,
            log_probs=torch.log_softmax(residuals, dim=-1),
            residual_scores=residuals,
            base_log_probs=residuals,
        )
        output = FirstMissValueOutput(
            action_values=action_values_from_residual_scores(residuals),
            direct_output=direct,
        )
        ranks = torch.zeros(batch, length, dtype=torch.long)
        ranks[0, 0] = 1
        available = torch.ones(batch, length, dtype=torch.bool)
        losses = first_miss_value_loss(output, ranks, available)
        self.assertEqual(losses.squared_errors.shape, (batch, 225))
        torch.testing.assert_close(
            losses.per_block_mse,
            losses.squared_errors.mean(dim=-1),
        )
        torch.testing.assert_close(losses.loss, losses.squared_errors.mean())


class FirstMissValueGradientTest(unittest.TestCase):
    def test_identity_and_two_backward_gradient_contract(self) -> None:
        model = make_model()
        inputs = make_inputs()
        ranks = torch.tensor([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        available = torch.ones(3, 3, dtype=torch.bool)

        output = model(*inputs)
        self.assertTrue(
            torch.equal(
                output.action_values,
                torch.zeros_like(output.action_values),
            )
        )
        actions = decode_strict_positive_actions(output.action_values)
        self.assertTrue(torch.equal(actions, torch.zeros_like(actions)))

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        first = first_miss_value_loss(output, ranks, available)
        first.loss.backward()
        projection = model.backbone.residual_projection.weight
        self.assertIsNotNone(projection.grad)
        self.assertGreater(float(projection.grad.norm()), 0.0)
        upstream_first = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name != "backbone.residual_projection.weight"
        ]
        self.assertTrue(
            all(
                gradient is None or int(torch.count_nonzero(gradient)) == 0
                for gradient in upstream_first
            )
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        second_output = model(*inputs)
        second = first_miss_value_loss(second_output, ranks, available)
        second.loss.backward()
        upstream_second = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name != "backbone.residual_projection.weight"
        ]
        self.assertTrue(
            any(
                gradient is not None
                and int(torch.count_nonzero(gradient)) > 0
                for gradient in upstream_second
            )
        )
        self.assertTrue(all(not value.requires_grad for value in inputs))


if __name__ == "__main__":
    unittest.main()
