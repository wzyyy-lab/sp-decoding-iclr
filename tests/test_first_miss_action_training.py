from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.train_first_miss_action_selector import (
    capacity_gate_report,
    checkpoint_selection_key,
    evaluate,
    _load_datasets,
)
from sph.first_miss_action_selector import (
    FirstMissActionOutput,
    FirstMissActionSelector,
    action_logits_from_scores,
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


def make_model() -> FirstMissActionSelector:
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
            initialization_seed=29,
        )
    )


class ControlledActionModel(torch.nn.Module):
    """Choose repair, harmful, and neutral edits for the three fixtures."""

    def eval(self) -> ControlledActionModel:
        return self

    def forward(
        self,
        hidden: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_logits: torch.Tensor,
        base_logsumexp: torch.Tensor,
        anchor_embeddings: torch.Tensor,
    ) -> FirstMissActionOutput:
        del hidden, candidate_embeddings, anchor_embeddings
        scores = candidate_logits.float() - base_logsumexp.float().unsqueeze(-1)
        scores = scores.clone()
        # Example 0 repairs its first miss at position 1/rank 1.
        scores[0, 1, 1] = scores[0, 1, 0] + 5.0
        # Example 1 damages a fully correct block at position 0/rank 1.
        scores[1, 0, 1] = scores[1, 0, 0] + 5.0
        # Example 2 edits after its out-of-K first miss, hence is neutral.
        scores[2, 1, 1] = scores[2, 1, 0] + 5.0
        direct_output = GlobalDirectOutput(
            scores=scores,
            log_probs=torch.log_softmax(scores, dim=-1),
            residual_scores=scores
            - (
                candidate_logits.float()
                - base_logsumexp.float().unsqueeze(-1)
            ),
            base_log_probs=(
                candidate_logits.float()
                - base_logsumexp.float().unsqueeze(-1)
            ),
        )
        return FirstMissActionOutput(
            action_logits=action_logits_from_scores(scores),
            direct_output=direct_output,
        )


class FirstMissActionTrainingTest(unittest.TestCase):
    def test_external_training_accepts_physically_isolated_validation(self) -> None:
        validation = SimpleNamespace(
            metadata={"top_k": 16},
            records=[
                {
                    "sample_id": "select-a",
                    "split": "validation_select",
                }
            ],
        )
        training = SimpleNamespace(
            metadata={"top_k": 16},
            records=[{"sample_id": "train-a", "split": "train"}],
        )
        args = Namespace(
            data=Path("validation"),
            train_data=[Path("training")],
            train_split="train",
            validation_split="validation_select",
            candidate_k=16,
            memorization_blocks=0,
            max_train_prompts=0,
        )
        with patch(
            "scripts.train_first_miss_action_selector.CanonicalBlockDataset",
            side_effect=[validation, training],
        ), patch(
            "scripts.train_first_miss_action_selector.direct."
            "assert_canonical_collection_compatible"
        ), patch(
            "scripts.train_first_miss_action_selector.direct."
            "assert_prompt_disjoint_splits"
        ):
            train_dataset, validation_dataset, *_ = _load_datasets(args)
        self.assertEqual(train_dataset.records, training.records)
        self.assertEqual(validation_dataset.records, validation.records)

    def test_external_training_rejects_mixed_validation_collection(self) -> None:
        mixed = SimpleNamespace(
            metadata={"top_k": 16},
            records=[
                {
                    "sample_id": "select-a",
                    "split": "validation_select",
                },
                {"sample_id": "gate-a", "split": "validation_gate"},
            ],
        )
        args = Namespace(
            data=Path("mixed"),
            train_data=[Path("training")],
            train_split="train",
            validation_split="validation_select",
            candidate_k=16,
            memorization_blocks=0,
            max_train_prompts=0,
        )
        with patch(
            "scripts.train_first_miss_action_selector.CanonicalBlockDataset",
            return_value=mixed,
        ), self.assertRaisesRegex(RuntimeError, "physically isolated"):
            _load_datasets(args)

    def test_epoch_zero_evaluation_is_exact_base_and_split_keep_reasons(
        self,
    ) -> None:
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
            report["fmas"]["mean_accepted_draft_tokens_prompt_balanced"],
        )
        self.assertAlmostEqual(
            report["base"]["mean_accepted_draft_tokens_prompt_balanced"],
            4.0 / 3.0,
        )
        self.assertAlmostEqual(
            report["single_edit_oracle"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ],
            2.0,
        )
        self.assertEqual(
            report["action_classification"]["target_kind_counts"],
            {"edit": 1, "keep_full_correct": 1, "keep_out_of_k": 1},
        )
        self.assertAlmostEqual(
            report["action_classification"]["accuracy"], 2.0 / 3.0
        )
        self.assertEqual(
            report["action_classification"]["repairable_action_recall"],
            0.0,
        )
        self.assertEqual(report["fmas_diagnostics"]["harmed_blocks"], 0)
        self.assertEqual(
            report["fmas_diagnostics"][
                "single_edit_oracle_gap_recovered"
            ],
            0.0,
        )

    def test_capacity_gate_uses_all_four_frozen_checks(self) -> None:
        report = {
            "action_classification": {
                "accuracy": 0.97,
                "repairable_action_recall": 0.95,
            },
            "fmas_diagnostics": {
                "single_edit_oracle_gap_recovered": 0.95,
                "harmed_fraction": 0.01,
            },
        }
        args = Namespace(
            min_action_accuracy=0.97,
            min_repairable_action_recall=0.95,
            min_oracle_gap_recovered=0.95,
            max_harmed_fraction=0.01,
        )
        self.assertTrue(capacity_gate_report(report, args)["passed"])
        report["action_classification"]["accuracy"] = 0.969
        gate = capacity_gate_report(report, args)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["action_accuracy"])

    def test_changed_action_accounting_covers_repair_harm_and_neutral(
        self,
    ) -> None:
        report = evaluate(
            ControlledActionModel(),
            [make_batch()],
            torch.randn(100, 8),
            torch.device("cpu"),
            candidate_k=4,
            include_examples=True,
        )
        diagnostics = report["fmas_diagnostics"]
        self.assertEqual(diagnostics["changed_blocks"], 3)
        self.assertEqual(diagnostics["improved_blocks"], 1)
        self.assertEqual(diagnostics["harmed_blocks"], 1)
        self.assertEqual(diagnostics["neutral_changed_blocks"], 1)
        self.assertAlmostEqual(diagnostics["harmed_fraction"], 1.0 / 3.0)
        self.assertEqual(diagnostics["first_miss_repairs"], 1)

    def test_checkpoint_keys_follow_preregistered_metrics(self) -> None:
        low_loss = {
            "loss": {"objective": 0.2},
            "action_classification": {
                "accuracy": 0.8,
                "repairable_action_recall": 0.7,
            },
            "fmas": {"mean_accepted_draft_tokens_prompt_balanced": 2.0},
            "fmas_diagnostics": {"harmed_fraction": 0.1},
        }
        high_eal = {
            "loss": {"objective": 0.3},
            "action_classification": {
                "accuracy": 0.7,
                "repairable_action_recall": 0.6,
            },
            "fmas": {"mean_accepted_draft_tokens_prompt_balanced": 2.2},
            "fmas_diagnostics": {"harmed_fraction": 0.05},
        }
        self.assertGreater(
            checkpoint_selection_key(low_loss, evidence_tier="capacity_probe"),
            checkpoint_selection_key(high_eal, evidence_tier="capacity_probe"),
        )
        self.assertGreater(
            checkpoint_selection_key(high_eal, evidence_tier="development"),
            checkpoint_selection_key(low_loss, evidence_tier="development"),
        )

    def test_capacity_key_ignores_repair_recall_after_ce_accuracy_tie(
        self,
    ) -> None:
        left = {
            "loss": {"objective": 0.2},
            "action_classification": {
                "accuracy": 0.8,
                "repairable_action_recall": 0.1,
            },
            "fmas": {"mean_accepted_draft_tokens_prompt_balanced": 1.0},
            "fmas_diagnostics": {"harmed_fraction": 0.0},
        }
        right = {
            **left,
            "action_classification": {
                "accuracy": 0.8,
                "repairable_action_recall": 0.9,
            },
        }
        self.assertEqual(
            checkpoint_selection_key(left, evidence_tier="capacity_probe"),
            checkpoint_selection_key(right, evidence_tier="capacity_probe"),
        )


if __name__ == "__main__":
    unittest.main()
