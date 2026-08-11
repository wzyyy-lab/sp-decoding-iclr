from __future__ import annotations

from argparse import Namespace
import unittest

import torch

from scripts.train_first_miss_value_selector import (
    capacity_gate_report,
    checkpoint_selection_key,
    evaluate,
    initial_projection_gradient_diagnostics,
)
from sph.first_miss_value_selector import (
    FirstMissValueOutput,
    FirstMissValueSelector,
    action_values_from_residual_scores,
)
from sph.global_direct_selector import (
    GlobalDirectCandidateSelector,
    GlobalDirectOutput,
)


def make_batch() -> dict[str, object]:
    candidate_ids = torch.tensor(
        [
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
            [[40, 41, 42, 43], [50, 51, 52, 53], [60, 61, 62, 63]],
            [[70, 71, 72, 73], [80, 51, 82, 83], [60, 91, 92, 93]],
        ]
    )
    gold_ids = torch.tensor(
        [[10, 21, 30], [40, 50, 60], [99, 51, 60]]
    )
    matches = candidate_ids.eq(gold_ids.unsqueeze(-1))
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0]).expand(3, 3, 4)
    return {
        "sample_ids": ["repair", "full", "outside"],
        "domains": ["chat", "code", "math"],
        "hidden": torch.randn(3, 3, 8),
        "anchor_ids": torch.tensor([1, 2, 3]),
        "candidate_ids": candidate_ids,
        "candidate_logits": logits,
        "base_logsumexp": torch.logsumexp(
            torch.cat([logits, torch.zeros(3, 3, 2)], dim=-1), dim=-1
        ),
        "gold_ids": gold_ids,
        "gold_in_lattice": matches.any(dim=-1),
        "gold_candidate_indices": matches.to(torch.int64).argmax(dim=-1),
    }


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
            initialization_seed=37,
        )
    )


class OneBatchLoader:
    def __init__(self, batch: dict[str, object]) -> None:
        self.batch = batch
        self.dataset = list(range(len(batch["sample_ids"])))

    def __iter__(self):
        yield self.batch


class ControlledValueModel(torch.nn.Module):
    """Select beneficial, harmful, and neutral edits for three fixtures."""

    def eval(self) -> ControlledValueModel:
        return self

    def forward(
        self,
        hidden: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_logits: torch.Tensor,
        base_logsumexp: torch.Tensor,
        anchor_embeddings: torch.Tensor,
    ) -> FirstMissValueOutput:
        del hidden, candidate_embeddings, anchor_embeddings
        base_scores = (
            candidate_logits.float() - base_logsumexp.float().unsqueeze(-1)
        )
        residuals = torch.zeros_like(base_scores)
        residuals[0, 1, 1] = 5.0  # repair first miss
        residuals[1, 0, 1] = 5.0  # harm fully correct block
        residuals[2, 1, 1] = 5.0  # neutral after out-of-K first miss
        direct = GlobalDirectOutput(
            scores=base_scores + residuals,
            log_probs=torch.log_softmax(base_scores + residuals, dim=-1),
            residual_scores=residuals,
            base_log_probs=base_scores,
        )
        return FirstMissValueOutput(
            action_values=action_values_from_residual_scores(residuals),
            direct_output=direct,
        )


class FirstMissValueTrainingTest(unittest.TestCase):
    def test_epoch_zero_is_exact_base_and_no_edit_precision_is_na(self) -> None:
        report = evaluate(
            make_model(),
            [make_batch()],
            torch.randn(100, 8),
            torch.device("cpu"),
            candidate_k=4,
            include_examples=True,
            require_base_identity=True,
        )
        self.assertEqual(
            report["base"]["mean_accepted_draft_tokens_prompt_balanced"],
            report["savs"]["mean_accepted_draft_tokens_prompt_balanced"],
        )
        self.assertEqual(report["decision"]["selected_edits"], 0)
        self.assertIsNone(report["decision"]["edit_selective_precision"])
        self.assertEqual(report["decision"]["harmed_fraction"], 0.0)
        self.assertEqual(
            report["decision"]["single_edit_oracle_gap_recovered"], 0.0
        )

    def test_decision_accounting_reconstructs_all_outcomes(self) -> None:
        report = evaluate(
            ControlledValueModel(),
            [make_batch()],
            torch.randn(100, 8),
            torch.device("cpu"),
            candidate_k=4,
            include_examples=True,
        )
        decision = report["decision"]
        self.assertEqual(decision["selected_edits"], 3)
        self.assertEqual(decision["beneficial_selected_actions"], 1)
        self.assertEqual(decision["harmful_selected_actions"], 1)
        self.assertEqual(decision["neutral_selected_edits"], 1)
        self.assertAlmostEqual(decision["harmed_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(decision["edit_selective_precision"], 1.0 / 3.0)
        self.assertEqual(decision["repair_opportunities"], 1)
        self.assertEqual(decision["repair_recall"], 1.0)
        self.assertEqual(decision["no_benefit_blocks"], 2)
        self.assertEqual(decision["no_benefit_false_edit_rate"], 1.0)
        self.assertAlmostEqual(
            decision["mean_selected_action_regret_normalized"], 1.0 / 3.0
        )
        self.assertAlmostEqual(
            decision["mean_selected_action_regret_tokens"], 1.0
        )
        self.assertAlmostEqual(
            report["savs"]["mean_accepted_draft_tokens_prompt_balanced"],
            1.0,
        )
        self.assertAlmostEqual(
            report["single_edit_oracle"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ],
            2.0,
        )

    def test_epoch_zero_gradient_decomposition_is_sign_complete(self) -> None:
        batch = make_batch()
        report = initial_projection_gradient_diagnostics(
            make_model(),
            OneBatchLoader(batch),
            torch.randn(100, 8),
            torch.device("cpu"),
            candidate_k=4,
        )
        components = report["components"]
        total_actions = sum(
            components[name]["actions"]
            for name in ("beneficial", "neutral", "harmful")
        )
        self.assertEqual(total_actions, 3 * 3 * 3)
        self.assertGreater(components["beneficial"]["actions"], 0)
        self.assertGreater(components["harmful"]["actions"], 0)
        self.assertGreater(
            components["beneficial"]["projection_gradient_norm"], 0.0
        )
        self.assertGreater(
            components["harmful"]["projection_gradient_norm"], 0.0
        )
        self.assertEqual(
            components["neutral"]["projection_gradient_norm"], 0.0
        )
        self.assertGreater(report["total_projection_gradient_norm"], 0.0)

    def test_capacity_gate_uses_all_six_frozen_checks(self) -> None:
        report = {
            "loss": {"all_action_rmse": 0.02},
            "signed_value": {
                "beneficial_sign_recall": 0.99,
                "harmful_nonpositive_recall": 0.99,
                "classes": {"beneficial": {"count": 256}},
            },
            "decision": {
                "single_edit_oracle_gap_recovered": 0.95,
                "harmed_fraction": 0.01,
            },
        }
        args = Namespace(
            max_value_rmse=0.02,
            min_beneficial_sign_recall=0.99,
            min_harmful_nonpositive_recall=0.99,
            min_oracle_gap_recovered=0.95,
            max_harmed_fraction=0.01,
            expected_beneficial_actions=256,
        )
        gate = capacity_gate_report(report, args)
        self.assertTrue(gate["passed"])
        self.assertEqual(len(gate["checks"]), 6)
        report["signed_value"]["classes"]["beneficial"]["count"] = 255
        gate = capacity_gate_report(report, args)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["beneficial_actions"])

    def test_checkpoint_keys_follow_frozen_rules(self) -> None:
        low_mse = {
            "loss": {"objective": 0.1},
            "savs": {"mean_accepted_draft_tokens_prompt_balanced": 1.0},
            "decision": {"harmed_fraction": 0.2},
        }
        high_eal = {
            "loss": {"objective": 0.2},
            "savs": {"mean_accepted_draft_tokens_prompt_balanced": 2.0},
            "decision": {"harmed_fraction": 0.1},
        }
        self.assertGreater(
            checkpoint_selection_key(low_mse, evidence_tier="capacity_probe"),
            checkpoint_selection_key(high_eal, evidence_tier="capacity_probe"),
        )
        self.assertGreater(
            checkpoint_selection_key(high_eal, evidence_tier="development"),
            checkpoint_selection_key(low_mse, evidence_tier="development"),
        )
        tied_eal_lower_harm = {
            **high_eal,
            "decision": {"harmed_fraction": 0.05},
        }
        self.assertGreater(
            checkpoint_selection_key(
                tied_eal_lower_harm, evidence_tier="development"
            ),
            checkpoint_selection_key(high_eal, evidence_tier="development"),
        )


if __name__ == "__main__":
    unittest.main()
