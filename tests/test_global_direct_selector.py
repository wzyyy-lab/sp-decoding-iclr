from __future__ import annotations

import inspect
import unittest

import torch

from sph.global_direct_selector import (
    GlobalDirectCandidateSelector,
    GlobalDirectOutput,
    accepted_reach_survival,
    base_accepted_prefix_mask,
    exact_dpace_position_weights,
    global_direct_candidate_loss,
    prediction_conditioned_prefix_mask,
    prefix_candidate_mask,
)


def make_inputs(
    *,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(71)
    batch, length, candidates, hidden_size = 2, 3, 4, 12
    candidate_logits = torch.randn(batch, length, candidates).sort(
        dim=-1, descending=True
    ).values
    tail_logits = torch.randn(batch, length, 7) - 1.0
    values = {
        "hidden": torch.randn(batch, length, hidden_size),
        "candidate_embeddings": torch.randn(
            batch, length, candidates, hidden_size
        ),
        "candidate_logits": candidate_logits,
        "base_logsumexp": torch.logsumexp(
            torch.cat([candidate_logits, tail_logits], dim=-1), dim=-1
        ),
        "anchor_embeddings": torch.randn(batch, hidden_size),
    }
    if requires_grad:
        values = {
            key: value.clone().requires_grad_(True)
            for key, value in values.items()
        }
    return values


def make_model(
    scope: str, mixer: str = "flat"
) -> GlobalDirectCandidateSelector:
    return GlobalDirectCandidateSelector(
        hidden_size=12,
        max_positions=3,
        max_candidates=4,
        model_dim=16,
        num_heads=4,
        num_layers=2,
        scope=scope,
        mixer=mixer,
    )


def enable_residual_readout(
    model: GlobalDirectCandidateSelector,
) -> None:
    torch.manual_seed(73)
    with torch.no_grad():
        model.residual_projection.weight.normal_(mean=0.0, std=0.1)


class ExactDPACETest(unittest.TestCase):
    def test_matches_official_naive_reference(self) -> None:
        probabilities = torch.tensor(
            [[0.8, 0.7, 0.2, 0.9], [0.6, 0.5, 0.4, 0.3]],
            requires_grad=True,
        )
        active = torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )
        alpha = 0.5
        got = exact_dpace_position_weights(
            probabilities, active, alpha=alpha
        )

        smoothed = (1.0 - alpha) * probabilities.detach() + alpha
        smoothed = torch.where(active, smoothed, torch.ones_like(smoothed))
        prefix = torch.cumprod(smoothed, dim=-1)
        expected = torch.flip(
            torch.cumsum(
                torch.flip(prefix * active.float(), dims=[-1]), dim=-1
            ),
            dims=[-1],
        )
        torch.testing.assert_close(got, expected)
        self.assertFalse(got.requires_grad)

    def test_prefix_is_inclusive_and_weights_are_not_normalized(self) -> None:
        probability = torch.tensor([[0.2]])
        active = torch.tensor([[True]])
        weight = exact_dpace_position_weights(
            probability, active, alpha=0.5
        )
        self.assertAlmostEqual(float(weight), 0.6, places=6)
        self.assertNotEqual(float(weight.sum()), float(active.sum()))

    def test_censored_suffix_cannot_change_observable_prefix(self) -> None:
        probabilities = torch.tensor([[0.8, 0.7, 0.01, 0.99]])
        active = torch.tensor([[True, True, False, False]])
        long_weights = exact_dpace_position_weights(
            probabilities, active
        )
        short_weights = exact_dpace_position_weights(
            probabilities[:, :2], active[:, :2]
        )
        torch.testing.assert_close(long_weights[:, :2], short_weights)
        self.assertEqual(float(long_weights[:, 2:].sum()), 0.0)


class AcceptedReachRiskTest(unittest.TestCase):
    @staticmethod
    def output_from_scores(scores: torch.Tensor) -> GlobalDirectOutput:
        log_probs = torch.log_softmax(scores, dim=-1)
        return GlobalDirectOutput(
            scores=scores,
            log_probs=log_probs,
            residual_scores=torch.zeros_like(scores),
            base_log_probs=log_probs.detach(),
        )

    def test_survival_matches_manual_prefix_products(self) -> None:
        probabilities = torch.tensor(
            [[0.8, 0.5, 0.25], [0.4, 0.3, 0.2]]
        )
        available = torch.tensor(
            [[True, True, True], [True, False, True]]
        )
        got = accepted_reach_survival(probabilities, available)
        expected = torch.tensor(
            [[0.8, 0.4, 0.1], [0.4, 0.0, 0.0]]
        )
        torch.testing.assert_close(got, expected)

    def test_reach_log_loss_gradient_matches_formula_and_finite_difference(
        self,
    ) -> None:
        log_losses = torch.tensor(
            [[0.2, 0.5, 1.1]], dtype=torch.float32, requires_grad=True
        )
        available = torch.ones_like(log_losses, dtype=torch.bool)

        def risk(values: torch.Tensor) -> torch.Tensor:
            survival = accepted_reach_survival(
                torch.exp(-values), available
            )
            return 1.0 - survival.sum() / values.shape[-1]

        loss = risk(log_losses)
        loss.backward()
        survival = accepted_reach_survival(
            torch.exp(-log_losses.detach()), available
        )
        expected = torch.flip(
            torch.cumsum(torch.flip(survival, dims=[-1]), dim=-1),
            dims=[-1],
        ) / log_losses.shape[-1]
        torch.testing.assert_close(log_losses.grad, expected)

        epsilon = 1e-3
        finite_differences = torch.empty_like(log_losses)
        for position in range(log_losses.shape[-1]):
            plus = log_losses.detach().clone()
            minus = log_losses.detach().clone()
            plus[0, position] += epsilon
            minus[0, position] -= epsilon
            finite_differences[0, position] = (
                risk(plus) - risk(minus)
            ) / (2.0 * epsilon)
        torch.testing.assert_close(
            log_losses.grad,
            finite_differences,
            rtol=2e-3,
            atol=2e-4,
        )

    def test_reach_arithmetic_is_float32_for_bfloat16_scores(self) -> None:
        scores = torch.tensor(
            [[[2.0, 0.0], [1.5, -0.5], [1.0, -1.0]]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        gold = torch.zeros((1, 3), dtype=torch.long)
        available = torch.ones((1, 3), dtype=torch.bool)
        loss = global_direct_candidate_loss(
            self.output_from_scores(scores),
            gold,
            available,
            weighting="accepted_reach",
        )
        self.assertEqual(loss.gold_probabilities.dtype, torch.float32)
        self.assertEqual(
            loss.components["soft_expected_accepted_tokens"].dtype,
            torch.float32,
        )
        loss.loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_base_mask_contains_only_contiguous_rank_one_prefix(self) -> None:
        gold = torch.tensor([[0, 0, 2, 0], [0, 0, 0, 0]])
        available = torch.tensor(
            [[True, True, True, True], [True, False, True, True]]
        )
        expected = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )
        torch.testing.assert_close(
            base_accepted_prefix_mask(gold, available), expected
        )

    def test_arr_gradient_matches_alpha_zero_candidate_dpace(self) -> None:
        torch.manual_seed(79)
        scores_arr = torch.randn(2, 4, 5, requires_grad=True)
        scores_dpace = scores_arr.detach().clone().requires_grad_(True)
        gold = torch.tensor([[0, 2, 1, 0], [1, 0, 3, 2]])
        available = torch.tensor(
            [[True, True, False, True], [True, True, True, True]]
        )
        arr = global_direct_candidate_loss(
            self.output_from_scores(scores_arr),
            gold,
            available,
            weighting="accepted_reach",
        )
        dpace = global_direct_candidate_loss(
            self.output_from_scores(scores_dpace),
            gold,
            available,
            weighting="candidate_dpace",
            dpace_alpha=0.0,
        )
        arr.loss.backward()
        dpace.loss.backward()
        torch.testing.assert_close(
            scores_arr.grad,
            scores_dpace.grad,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_base_safety_penalizes_a_greedy_prefix_flip(self) -> None:
        safe_scores = torch.tensor(
            [[[3.0, 1.0, 0.0], [2.0, 1.0, 0.0]]],
            requires_grad=True,
        )
        unsafe_scores = safe_scores.detach().clone()
        unsafe_scores[0, 0, 1] = 4.0
        unsafe_scores.requires_grad_(True)
        gold = torch.tensor([[0, 0]])
        available = torch.ones_like(gold, dtype=torch.bool)
        safe = global_direct_candidate_loss(
            self.output_from_scores(safe_scores),
            gold,
            available,
            weighting="accepted_reach",
            base_safety_weight=0.25,
            base_safety_margin=0.1,
        )
        unsafe = global_direct_candidate_loss(
            self.output_from_scores(unsafe_scores),
            gold,
            available,
            weighting="accepted_reach",
            base_safety_weight=0.25,
            base_safety_margin=0.1,
        )
        self.assertEqual(
            float(safe.components["base_safety"].detach()), 0.0
        )
        self.assertGreater(
            float(unsafe.components["base_safety"].detach()), 0.0
        )
        self.assertGreater(
            float(unsafe.loss.detach()), float(safe.loss.detach())
        )


class ReachableDPACETest(unittest.TestCase):
    @staticmethod
    def output_from_scores(scores: torch.Tensor) -> GlobalDirectOutput:
        log_probs = torch.log_softmax(scores.float(), dim=-1)
        return GlobalDirectOutput(
            scores=scores,
            log_probs=log_probs,
            residual_scores=torch.zeros_like(scores),
            base_log_probs=log_probs.detach(),
        )

    def test_reachable_mask_keeps_in_lattice_breaker(self) -> None:
        predicted = torch.tensor([[0, 1, 2, 0]])
        gold = torch.tensor([[0, 0, 2, 0]])
        available = torch.ones_like(gold, dtype=torch.bool)
        reachable = prediction_conditioned_prefix_mask(
            predicted, gold, available
        )
        expected = torch.tensor([[True, True, False, False]])
        torch.testing.assert_close(reachable, expected)
        coverage = prefix_candidate_mask(available)
        suffix = coverage & ~reachable
        self.assertFalse(bool((reachable & suffix).any()))
        torch.testing.assert_close(reachable | suffix, coverage)

    def test_out_of_lattice_breaker_and_suffix_are_not_supervised(
        self,
    ) -> None:
        predicted = torch.tensor([[0, 0, 0, 0]])
        gold = torch.tensor([[0, 0, 0, 0]])
        available = torch.tensor([[True, True, False, True]])
        expected = torch.tensor([[True, True, False, False]])
        torch.testing.assert_close(
            prediction_conditioned_prefix_mask(
                predicted, gold, available
            ),
            expected,
        )

    def test_lambda_one_matches_candidate_dpace_value_and_gradient(
        self,
    ) -> None:
        torch.manual_seed(83)
        cases = [
            (
                torch.tensor([[0, 2, 1, 0], [1, 0, 3, 2]]),
                torch.tensor(
                    [[True, True, False, True], [True, True, True, True]]
                ),
            ),
            (
                torch.zeros((2, 4), dtype=torch.long),
                torch.ones((2, 4), dtype=torch.bool),
            ),
            (
                torch.zeros((2, 4), dtype=torch.long),
                torch.tensor(
                    [[False, True, True, True], [True, False, True, True]]
                ),
            ),
        ]
        for gold, available in cases:
            with self.subTest(gold=gold.tolist(), available=available.tolist()):
                control_scores = torch.randn(2, 4, 5, requires_grad=True)
                treatment_scores = (
                    control_scores.detach().clone().requires_grad_(True)
                )
                control = global_direct_candidate_loss(
                    self.output_from_scores(control_scores),
                    gold,
                    available,
                    weighting="candidate_dpace",
                    dpace_alpha=0.5,
                )
                treatment = global_direct_candidate_loss(
                    self.output_from_scores(treatment_scores),
                    gold,
                    available,
                    weighting="reachable_dpace",
                    dpace_alpha=0.5,
                    post_break_weight=1.0,
                )
                torch.testing.assert_close(
                    treatment.loss, control.loss, rtol=0, atol=0
                )
                control.loss.backward()
                treatment.loss.backward()
                torch.testing.assert_close(
                    treatment_scores.grad,
                    control_scores.grad,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    treatment.active_positions,
                    prefix_candidate_mask(available),
                )

    def test_lambda_zero_has_zero_post_break_score_gradient(self) -> None:
        scores = torch.tensor(
            [
                [
                    [4.0, 1.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [1.0, 4.0, 0.0],
                    [1.0, 0.0, 4.0],
                ]
            ],
            requires_grad=True,
        )
        gold = torch.tensor([[0, 1, 1, 2]])
        available = torch.ones_like(gold, dtype=torch.bool)
        result = global_direct_candidate_loss(
            self.output_from_scores(scores),
            gold,
            available,
            weighting="reachable_dpace",
            post_break_weight=0.0,
        )
        expected_training = torch.tensor(
            [[True, True, False, False]]
        )
        torch.testing.assert_close(
            result.training_positions, expected_training
        )
        torch.testing.assert_close(
            result.post_break_positions, ~expected_training
        )
        result.loss.backward()
        self.assertGreater(float(scores.grad[:, :2].abs().sum()), 0.0)
        self.assertEqual(float(scores.grad[:, 2:].abs().sum()), 0.0)

    def test_empty_coverage_has_finite_zero_direct_loss(self) -> None:
        scores = torch.randn(2, 3, 4, requires_grad=True)
        gold = torch.zeros((2, 3), dtype=torch.long)
        available = torch.zeros((2, 3), dtype=torch.bool)
        result = global_direct_candidate_loss(
            self.output_from_scores(scores),
            gold,
            available,
            weighting="reachable_dpace",
            post_break_weight=0.0,
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertEqual(float(result.loss.detach()), 0.0)
        self.assertEqual(
            float(result.components["reachable_fraction_of_coverage"]),
            0.0,
        )


class GlobalDirectCandidateSelectorTest(unittest.TestCase):
    def test_matched_scopes_have_identical_parameter_counts(self) -> None:
        counts = {
            scope: sum(
                parameter.numel()
                for parameter in make_model(scope).parameters()
            )
            for scope in ("local", "causal", "global")
        }
        self.assertEqual(len(set(counts.values())), 1)

    def test_axial_scopes_have_identical_parameter_counts(self) -> None:
        counts = {
            scope: sum(
                parameter.numel()
                for parameter in make_model(
                    scope, mixer="axial"
                ).parameters()
            )
            for scope in ("local", "causal", "global")
        }
        self.assertEqual(len(set(counts.values())), 1)

    def test_attention_starts_with_local_signal_preservation_prior(self) -> None:
        model = make_model("global")
        expected = torch.full(
            (4,), torch.tensor(3.0).log().item()
        )
        for block in model.blocks:
            torch.testing.assert_close(
                block.same_position_bias.detach(), expected
            )

    def test_forward_has_no_candidate_ids_or_gold_inputs(self) -> None:
        parameters = inspect.signature(
            GlobalDirectCandidateSelector.forward
        ).parameters
        self.assertNotIn("candidate_ids", parameters)
        self.assertNotIn("gold_ids", parameters)
        self.assertNotIn("gold_candidate_indices", parameters)

    def test_identity_initialization_exactly_matches_dflash(self) -> None:
        inputs = make_inputs()
        model = make_model("global").eval()
        output = model(**inputs)
        expected = (
            inputs["candidate_logits"]
            - inputs["base_logsumexp"].unsqueeze(-1)
        )
        torch.testing.assert_close(output.scores, expected)
        torch.testing.assert_close(
            output.residual_scores,
            torch.zeros_like(output.residual_scores),
        )
        self.assertTrue(
            torch.equal(
                output.scores.argmax(dim=-1),
                torch.zeros_like(output.scores[..., 0], dtype=torch.long),
            )
        )

    def test_feature_logits_can_differ_from_identity_score_logits(self) -> None:
        inputs = make_inputs()
        model = make_model("global").eval()
        score_logits = torch.flip(inputs["candidate_logits"], dims=[-1]) + 0.25
        score_lse = torch.logsumexp(score_logits, dim=-1) + 0.7
        output = model(
            **inputs,
            score_candidate_logits=score_logits,
            score_logsumexp=score_lse,
        )
        expected = score_logits - score_lse.unsqueeze(-1)
        torch.testing.assert_close(output.scores, expected)
        torch.testing.assert_close(
            output.residual_scores,
            torch.zeros_like(output.residual_scores),
        )

    def test_local_scope_is_invariant_to_other_positions(self) -> None:
        inputs = make_inputs()
        model = make_model("local").eval()
        enable_residual_readout(model)
        first = model(**inputs).scores[:, 0]

        changed = {key: value.clone() for key, value in inputs.items()}
        changed["hidden"][:, 2].add_(50.0)
        changed["candidate_embeddings"][:, 2].mul_(-7.0)
        changed["candidate_logits"][:, 2].add_(
            torch.tensor([10.0, 3.0, -4.0, -9.0])
        )
        changed["base_logsumexp"][:, 2].add_(2.0)
        second = model(**changed).scores[:, 0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_causal_scope_cannot_observe_future_positions(self) -> None:
        inputs = make_inputs()
        model = make_model("causal").eval()
        enable_residual_readout(model)
        first = model(**inputs).scores[:, 0]

        changed = {key: value.clone() for key, value in inputs.items()}
        changed["hidden"][:, 2].add_(50.0)
        changed["candidate_embeddings"][:, 2].mul_(-7.0)
        changed["candidate_logits"][:, 2].add_(
            torch.tensor([10.0, 3.0, -4.0, -9.0])
        )
        changed["base_logsumexp"][:, 2].add_(2.0)
        second = model(**changed).scores[:, 0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_global_scope_uses_other_positions(self) -> None:
        inputs = make_inputs()
        model = make_model("global").eval()
        enable_residual_readout(model)
        first = model(**inputs).scores[:, 0]

        changed = {key: value.clone() for key, value in inputs.items()}
        changed["hidden"][:, 2].add_(50.0)
        changed["candidate_embeddings"][:, 2].mul_(-7.0)
        changed["candidate_logits"][:, 2].add_(
            torch.tensor([10.0, 3.0, -4.0, -9.0])
        )
        changed["base_logsumexp"][:, 2].add_(2.0)
        second = model(**changed).scores[:, 0]
        self.assertGreater(
            float((first - second).abs().max().detach()), 1e-6
        )

    def test_axial_scope_respects_local_causal_and_global_boundaries(
        self,
    ) -> None:
        inputs = make_inputs()
        changed = {key: value.clone() for key, value in inputs.items()}
        changed["hidden"][:, 2].add_(50.0)
        changed["candidate_embeddings"][:, 2].mul_(-7.0)
        changed["candidate_logits"][:, 2].add_(
            torch.tensor([10.0, 3.0, -4.0, -9.0])
        )
        changed["base_logsumexp"][:, 2].add_(2.0)
        differences = {}
        for scope in ("local", "causal", "global"):
            model = make_model(scope, mixer="axial").eval()
            enable_residual_readout(model)
            first = model(**inputs).scores[:, 0]
            second = model(**changed).scores[:, 0]
            differences[scope] = float(
                (first - second).abs().max().detach()
            )
        self.assertEqual(differences["local"], 0.0)
        self.assertEqual(differences["causal"], 0.0)
        self.assertGreater(differences["global"], 1e-6)

    def test_axial_identity_initialization_matches_dflash(self) -> None:
        inputs = make_inputs()
        model = make_model("global", mixer="axial").eval()
        output = model(**inputs)
        expected = (
            inputs["candidate_logits"]
            - inputs["base_logsumexp"].unsqueeze(-1)
        )
        torch.testing.assert_close(output.scores, expected)

    def test_compatibility_encoder_preserves_identity_initialization(
        self,
    ) -> None:
        inputs = make_inputs()
        model = GlobalDirectCandidateSelector(
            hidden_size=12,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            num_heads=4,
            num_layers=2,
            scope="global",
            mixer="flat",
            node_encoder="compatibility",
        ).eval()
        output = model(**inputs)
        expected = (
            inputs["candidate_logits"]
            - inputs["base_logsumexp"].unsqueeze(-1)
        )
        torch.testing.assert_close(output.scores, expected)

    def test_compatibility_encoder_receives_gradients(self) -> None:
        inputs = make_inputs()
        model = GlobalDirectCandidateSelector(
            hidden_size=12,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            scope="global",
            mixer="flat",
            node_encoder="compatibility",
        )
        enable_residual_readout(model)
        output = model(**inputs)
        loss = global_direct_candidate_loss(
            output,
            torch.tensor([[0, 1, 0], [2, 0, 3]]),
            torch.tensor(
                [[True, True, False], [True, True, True]]
            ),
            weighting="accepted_reach",
            base_safety_weight=0.1,
        )
        loss.loss.backward()
        self.assertIsNotNone(
            model.compatibility_projection[0].weight.grad
        )

    def test_frozen_inputs_are_detached_but_head_receives_gradients(self) -> None:
        inputs = make_inputs(requires_grad=True)
        model = make_model("global")
        enable_residual_readout(model)
        output = model(**inputs)
        gold_indices = torch.tensor([[0, 1, 0], [2, 0, 3]])
        in_lattice = torch.tensor(
            [[True, True, False], [True, True, True]]
        )
        loss = global_direct_candidate_loss(
            output,
            gold_indices,
            in_lattice,
            weighting="dpace",
        )
        loss.loss.backward()
        for value in inputs.values():
            self.assertIsNone(value.grad)
        self.assertIsNotNone(model.residual_projection.weight.grad)
        self.assertIsNotNone(model.hidden_projection.weight.grad)

    def test_shared_parameters_ignore_treatment_module_rng_consumption(
        self,
    ) -> None:
        common = {
            "hidden_size": 12,
            "max_positions": 3,
            "max_candidates": 4,
            "model_dim": 16,
            "num_heads": 4,
            "num_layers": 2,
            "scope": "global",
            "mixer": "flat",
            "initialization_seed": 19,
        }
        additive = GlobalDirectCandidateSelector(
            **common, node_encoder="additive"
        )
        compatibility = GlobalDirectCandidateSelector(
            **common, node_encoder="compatibility"
        )
        additive_parameters = dict(additive.named_parameters())
        compatibility_parameters = dict(
            compatibility.named_parameters()
        )
        shared_names = sorted(
            set(additive_parameters) & set(compatibility_parameters)
        )
        self.assertTrue(shared_names)
        for name in shared_names:
            torch.testing.assert_close(
                additive_parameters[name],
                compatibility_parameters[name],
                rtol=0,
                atol=0,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_axial_and_flat_share_input_parameter_initialization(self) -> None:
        common = {
            "hidden_size": 12,
            "max_positions": 3,
            "max_candidates": 4,
            "model_dim": 16,
            "num_heads": 4,
            "num_layers": 2,
            "scope": "global",
            "node_encoder": "additive",
            "initialization_seed": 23,
        }
        axial = GlobalDirectCandidateSelector(**common, mixer="axial")
        flat = GlobalDirectCandidateSelector(**common, mixer="flat")
        axial_parameters = dict(axial.named_parameters())
        flat_parameters = dict(flat.named_parameters())
        shared_prefixes = (
            "hidden_projection.",
            "token_projection.",
            "position_embedding.",
            "rank_embedding.",
            "scalar_projection.",
            "input_norm.",
            "output_norm.",
            "residual_projection.",
        )
        names = [
            name
            for name in axial_parameters
            if name.startswith(shared_prefixes)
        ]
        self.assertTrue(names)
        for name in names:
            torch.testing.assert_close(
                axial_parameters[name],
                flat_parameters[name],
                rtol=0,
                atol=0,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_model_construction_does_not_advance_cpu_rng(self) -> None:
        torch.manual_seed(29)
        before = torch.random.get_rng_state()
        GlobalDirectCandidateSelector(
            hidden_size=12,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            initialization_seed=31,
        )
        after = torch.random.get_rng_state()
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_initialization_seed_changes_random_shared_parameters(self) -> None:
        first = GlobalDirectCandidateSelector(
            hidden_size=12,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            initialization_seed=37,
        )
        second = GlobalDirectCandidateSelector(
            hidden_size=12,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            initialization_seed=38,
        )
        self.assertFalse(
            torch.equal(
                first.hidden_projection.weight,
                second.hidden_projection.weight,
            )
        )

    def test_loss_trains_the_same_direct_logits_used_for_decoding(self) -> None:
        inputs = make_inputs()
        model = make_model("global")
        output = model(**inputs)
        gold_indices = torch.tensor([[0, 1, 0], [2, 0, 3]])
        in_lattice = torch.tensor(
            [[True, True, False], [True, True, True]]
        )
        loss = global_direct_candidate_loss(
            output,
            gold_indices,
            in_lattice,
            weighting="dpace",
        )
        self.assertTrue(torch.isfinite(loss.loss))
        self.assertEqual(
            int(loss.active_positions.sum()),
            int(prefix_candidate_mask(in_lattice).sum()),
        )
        loss.loss.backward()
        self.assertIsNotNone(model.residual_projection.weight.grad)

    def test_no_trainable_vocabulary_table_exists(self) -> None:
        model = GlobalDirectCandidateSelector(
            hidden_size=2560,
            max_positions=15,
            max_candidates=16,
            model_dim=128,
            num_heads=8,
            num_layers=2,
            scope="global",
        )
        names = dict(model.named_parameters())
        self.assertFalse(
            any("vocab" in name or "token_embedding" in name for name in names)
        )
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters()),
            2_000_000,
        )


if __name__ == "__main__":
    unittest.main()
