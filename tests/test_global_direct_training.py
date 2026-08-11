from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import unittest

import torch

from scripts.train_global_direct_selector import (
    assert_canonical_collection_compatible,
    assert_prompt_disjoint_splits,
    capacity_gate_report,
    checkpoint_selection_key,
    cosine_warmup_scheduler,
    deterministic_capacity_subset,
    deterministic_prompt_subset,
    evaluate,
    selection_key,
    serializable_config,
    summarize_margin_calibration,
    tune_margin_threshold,
)
from sph.global_direct_selector import GlobalDirectCandidateSelector


def make_identity_batch() -> dict[str, object]:
    candidate_ids = torch.tensor(
        [
            [
                [10, 11, 12, 13],
                [20, 21, 22, 23],
                [30, 31, 32, 33],
            ],
            [
                [40, 41, 42, 43],
                [50, 51, 52, 53],
                [60, 61, 62, 63],
            ],
        ]
    )
    gold_ids = torch.tensor([[10, 21, 30], [41, 99, 60]])
    matches = candidate_ids == gold_ids.unsqueeze(-1)
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0]).expand(2, 3, 4)
    return {
        "sample_ids": ["prompt-a", "prompt-b"],
        "domains": ["chat", "code"],
        "hidden": torch.randn(2, 3, 12),
        "anchor_ids": torch.tensor([1, 2]),
        "candidate_ids": candidate_ids,
        "candidate_logits": logits,
        "base_logsumexp": torch.logsumexp(
            torch.cat(
                [logits, torch.zeros(2, 3, 3)],
                dim=-1,
            ),
            dim=-1,
        ),
        "gold_ids": gold_ids,
        "gold_in_lattice": matches.any(dim=-1),
        "gold_candidate_indices": matches.to(torch.int64).argmax(dim=-1),
    }


class _Records:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records


class GlobalDirectTrainingProtocolTest(unittest.TestCase):
    def test_epoch_zero_evaluation_is_exact_dflash(self) -> None:
        torch.manual_seed(91)
        model = GlobalDirectCandidateSelector(
            hidden_size=12,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            scope="global",
        )
        target_embedding = torch.randn(100, 12)
        report = evaluate(
            model,
            [make_identity_batch()],
            target_embedding,
            torch.device("cpu"),
            candidate_k=4,
            loss_weighting="dpace",
            dpace_alpha=0.5,
            exponential_gamma=7.0,
            include_examples=True,
            require_base_identity=True,
        )
        self.assertEqual(
            report["base"]["mean_accepted_draft_tokens"],
            report["direct"]["mean_accepted_draft_tokens"],
        )
        self.assertEqual(
            report["base"]["first_token_accuracy"],
            report["direct"]["first_token_accuracy"],
        )
        self.assertEqual(
            report["chosen_candidate_rank_counts"],
            {"1": 6},
        )
        self.assertEqual(
            report["direct_diagnostics"]["path_changed_blocks"],
            0,
        )
        self.assertEqual(
            report["candidate_classification"]["positions"],
            4,
        )
        self.assertAlmostEqual(
            report["candidate_classification"]["accuracy"], 0.5
        )
        self.assertAlmostEqual(
            report["base"]["mean_accepted_draft_tokens"], 0.5
        )
        self.assertAlmostEqual(
            report["oracle"]["mean_accepted_draft_tokens"], 2.0
        )

    def test_runtime_identity_check_rejects_nonzero_residual(self) -> None:
        torch.manual_seed(92)
        model = GlobalDirectCandidateSelector(
            hidden_size=12,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            scope="global",
        )
        with torch.no_grad():
            model.residual_projection.weight.fill_(0.1)
        with self.assertRaisesRegex(RuntimeError, "epoch-zero selector"):
            evaluate(
                model,
                [make_identity_batch()],
                torch.randn(100, 12),
                torch.device("cpu"),
                candidate_k=4,
                loss_weighting="accepted_reach",
                dpace_alpha=0.5,
                exponential_gamma=7.0,
                require_base_identity=True,
            )

    def test_reachable_loss_keeps_fixed_classification_denominator(
        self,
    ) -> None:
        torch.manual_seed(93)
        model = GlobalDirectCandidateSelector(
            hidden_size=12,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            scope="global",
        )
        report = evaluate(
            model,
            [make_identity_batch()],
            torch.randn(100, 12),
            torch.device("cpu"),
            candidate_k=4,
            loss_weighting="reachable_dpace",
            dpace_alpha=0.5,
            exponential_gamma=7.0,
            post_break_weight=0.0,
            require_base_identity=True,
        )
        self.assertEqual(
            report["candidate_classification"]["positions"], 4
        )
        self.assertEqual(
            report["candidate_classification"]["non_top1_positions"],
            2,
        )
        components = report["loss"]["components"]
        self.assertLess(
            components["reachable_fraction_of_coverage"], 1.0
        )
        self.assertGreater(
            components["post_break_positions_per_block"], 0.0
        )

    def test_evaluation_handles_empty_candidate_coverage(self) -> None:
        torch.manual_seed(94)
        model = GlobalDirectCandidateSelector(
            hidden_size=12,
            max_positions=3,
            max_candidates=4,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            scope="global",
        )
        batch = make_identity_batch()
        batch["gold_ids"] = torch.full((2, 3), 99)
        batch["gold_in_lattice"] = torch.zeros(2, 3, dtype=torch.bool)
        batch["gold_candidate_indices"] = torch.zeros(2, 3, dtype=torch.long)
        report = evaluate(
            model,
            [batch],
            torch.randn(100, 12),
            torch.device("cpu"),
            candidate_k=4,
            loss_weighting="reachable_dpace",
            dpace_alpha=0.5,
            exponential_gamma=7.0,
            post_break_weight=0.0,
            require_base_identity=True,
        )
        classification = report["candidate_classification"]
        self.assertEqual(classification["positions"], 0)
        self.assertIsNone(classification["accuracy"])
        self.assertEqual(report["loss"]["objective"], 0.0)
        self.assertEqual(selection_key(report)[-1], float("-inf"))
        gate = capacity_gate_report(
            report,
            Namespace(
                min_candidate_accuracy=0.99,
                min_hard_candidate_accuracy=0.97,
                min_first_miss_repair_rate=0.95,
                min_oracle_gap_recovered=0.95,
                max_harmed_fraction=0.01,
            ),
        )
        self.assertFalse(gate["passed"])

    def test_checkpoint_key_orders_by_eal_before_accuracy(self) -> None:
        better_eal = {
            "direct": {
                "mean_accepted_draft_tokens_prompt_balanced": 2.1,
                "first_token_accuracy": 0.7,
            },
            "candidate_classification": {"accuracy": 0.7},
            "by_domain": {
                "chat": {
                    "base": {"mean_accepted_draft_tokens": 1.0},
                    "direct": {"mean_accepted_draft_tokens": 1.1},
                }
            },
        }
        better_accuracy = {
            "direct": {
                "mean_accepted_draft_tokens_prompt_balanced": 2.0,
                "first_token_accuracy": 1.0,
            },
            "candidate_classification": {"accuracy": 1.0},
            "by_domain": {
                "chat": {
                    "base": {"mean_accepted_draft_tokens": 1.0},
                    "direct": {"mean_accepted_draft_tokens": 2.0},
                }
            },
        }
        self.assertGreater(
            selection_key(better_eal),
            selection_key(better_accuracy),
        )

    def test_prompt_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "prompt leakage"):
            assert_prompt_disjoint_splits(
                {
                    "train": _Records([{"sample_id": "same"}]),
                    "validation": _Records([{"sample_id": "same"}]),
                }
            )

    def test_external_collection_requires_identical_model_fingerprints(
        self,
    ) -> None:
        reference = {
            "format_version": 2,
            "block_size": 16,
            "draft_positions": 15,
            "attention_implementation": "sdpa",
            "dtype": "bfloat16",
            "target_layer_ids": [1, 9],
            "provenance": {
                "target_files": [
                    {"path": "target", "bytes": 1, "sha256": "a"}
                ],
                "draft_files": [
                    {"path": "draft", "bytes": 2, "sha256": "b"}
                ],
            },
        }
        candidate = {
            **reference,
            "provenance": {
                "target_files": [
                    {"path": "target", "bytes": 1, "sha256": "a"}
                ],
                "draft_files": [
                    {"path": "draft", "bytes": 2, "sha256": "changed"}
                ],
            },
        }
        with self.assertRaisesRegex(RuntimeError, "draft_files"):
            assert_canonical_collection_compatible(
                reference, candidate, path=Path("/tmp/part")
            )

    def test_config_snapshot_serializes_repeated_path_arguments(self) -> None:
        snapshot = serializable_config(
            Namespace(
                data=Path("/validation"),
                train_data=[Path("/train-a"), Path("/train-b")],
            )
        )
        self.assertEqual(snapshot["data"], "/validation")
        self.assertEqual(
            snapshot["train_data"], ["/train-a", "/train-b"]
        )

    def test_capacity_subset_includes_repair_opportunities(self) -> None:
        def record(index: int, gold_rank: int) -> dict[str, object]:
            topk = torch.arange(12).reshape(3, 4) + index * 20
            gold = topk[:, 0].clone()
            if gold_rank:
                gold[0] = topk[0, gold_rank]
            return {
                "sample_id": str(index),
                "base_topk_ids": topk,
                "gold_ids": gold,
            }

        records = [
            record(index, 1 if index < 5 else 0)
            for index in range(10)
        ]
        selected = deterministic_capacity_subset(
            records,
            count=6,
            seed=5,
            opportunity_fraction=0.5,
            candidate_k=4,
        )
        opportunities = sum(
            int(item["gold_ids"][0] == item["base_topk_ids"][0, 1])
            for item in selected
        )
        self.assertEqual(opportunities, 3)

    def test_prompt_subsets_are_nested_and_keep_whole_prompts(self) -> None:
        records = [
            {
                "sample_id": f"prompt-{prompt}",
                "domain": "chat" if prompt % 2 else "code",
                "block": block,
            }
            for prompt in range(10)
            for block in range(prompt % 3 + 1)
        ]
        small = deterministic_prompt_subset(
            records, max_prompts=3, seed=17
        )
        large = deterministic_prompt_subset(
            records, max_prompts=7, seed=17
        )
        small_ids = {str(record["sample_id"]) for record in small}
        large_ids = {str(record["sample_id"]) for record in large}
        self.assertEqual(len(small_ids), 3)
        self.assertEqual(len(large_ids), 7)
        self.assertLessEqual(small_ids, large_ids)
        self.assertEqual(
            small,
            [
                record
                for record in records
                if str(record["sample_id"]) in small_ids
            ],
        )
        for sample_id in small_ids:
            self.assertEqual(
                sum(
                    record["sample_id"] == sample_id
                    for record in small
                ),
                sum(
                    record["sample_id"] == sample_id
                    for record in records
                ),
            )

    def test_margin_calibration_retracts_low_confidence_harm(self) -> None:
        examples = [
            {
                "sample_id": "prompt-a",
                "domain": "chat",
                "accepted_draft_tokens": {
                    "base": 2,
                    "direct": 1,
                },
                "candidate_path_indices": {
                    "direct": [0, 1, 0],
                },
                "direct_margin_over_base": [0.0, 0.1, 0.0],
                "base_position_correct": [True, True, False],
                "direct_position_correct": [True, False, False],
            },
            {
                "sample_id": "prompt-b",
                "domain": "code",
                "accepted_draft_tokens": {
                    "base": 0,
                    "direct": 1,
                },
                "candidate_path_indices": {
                    "direct": [1, 0, 0],
                },
                "direct_margin_over_base": [0.8, 0.0, 0.0],
                "base_position_correct": [False, False, False],
                "direct_position_correct": [True, False, False],
            },
        ]
        evaluation = {
            "examples": examples,
            "base": {"first_token_accuracy": 0.5},
            "by_domain": {
                "chat": {
                    "base": {"mean_accepted_draft_tokens": 2.0}
                },
                "code": {
                    "base": {"mean_accepted_draft_tokens": 0.0}
                },
            },
        }
        threshold, calibrated = tune_margin_threshold(
            evaluation,
            max_first_token_drop=0.0,
            max_domain_drop=0.0,
        )
        self.assertGreater(threshold, 0.1)
        self.assertLessEqual(threshold, 0.8)
        self.assertAlmostEqual(
            calibrated[
                "mean_accepted_draft_tokens_prompt_balanced"
            ],
            1.5,
        )
        self.assertEqual(
            calibrated["diagnostics"]["improved_blocks"], 1
        )
        self.assertEqual(
            calibrated["diagnostics"]["harmed_blocks"], 0
        )
        self.assertEqual(
            calibrated["diagnostics"]["alternative_positions_used"],
            1,
        )
        raw = summarize_margin_calibration(
            evaluation, threshold=0.0
        )
        self.assertEqual(raw["diagnostics"]["harmed_blocks"], 1)

    def test_capacity_gate_requires_every_safety_check(self) -> None:
        evaluation = {
            "candidate_classification": {
                "accuracy": 0.995,
                "non_top1_accuracy": 0.98,
            },
            "direct_diagnostics": {
                "first_miss_repair_rate_given_k": 0.97,
                "oracle_gap_recovered": 0.96,
                "harmed_fraction": 0.02,
            },
        }
        args = Namespace(
            min_candidate_accuracy=0.99,
            min_hard_candidate_accuracy=0.97,
            min_first_miss_repair_rate=0.95,
            min_oracle_gap_recovered=0.95,
            max_harmed_fraction=0.01,
        )
        report = capacity_gate_report(evaluation, args)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["harmed_fraction"])

    def test_capacity_checkpoint_key_keeps_a_passing_epoch(self) -> None:
        def evaluation(*, eal: float, harm: float) -> dict[str, object]:
            return {
                "candidate_classification": {
                    "accuracy": 0.995,
                    "non_top1_accuracy": 0.98,
                },
                "direct_diagnostics": {
                    "first_miss_repair_rate_given_k": 0.97,
                    "oracle_gap_recovered": 0.96,
                    "harmed_fraction": harm,
                },
                "direct": {
                    "mean_accepted_draft_tokens_prompt_balanced": eal,
                    "first_token_accuracy": 1.0,
                },
                "by_domain": {
                    "chat": {
                        "base": {"mean_accepted_draft_tokens": 1.0},
                        "direct": {"mean_accepted_draft_tokens": eal},
                    }
                },
            }

        args = Namespace(
            memorization_blocks=128,
            min_candidate_accuracy=0.99,
            min_hard_candidate_accuracy=0.97,
            min_first_miss_repair_rate=0.95,
            min_oracle_gap_recovered=0.95,
            max_harmed_fraction=0.01,
        )
        passing = checkpoint_selection_key(
            evaluation(eal=2.0, harm=0.0), args
        )
        higher_eal_but_failing = checkpoint_selection_key(
            evaluation(eal=3.0, harm=0.02), args
        )
        self.assertGreater(passing, higher_eal_but_failing)

    def test_scheduler_warms_then_reaches_zero(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler, warmup_steps = cosine_warmup_scheduler(
            optimizer, total_steps=10, warmup_ratio=0.2
        )
        self.assertEqual(warmup_steps, 2)
        learning_rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])
        self.assertLess(learning_rates[0], learning_rates[1])
        self.assertAlmostEqual(learning_rates[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
