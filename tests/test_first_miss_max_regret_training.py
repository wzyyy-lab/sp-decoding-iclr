from __future__ import annotations

from argparse import Namespace
from copy import deepcopy

import torch

from scripts.train_first_miss_max_regret_selector import (
    _competitor_churn,
    capacity_gate_report,
    evaluate,
    projection_gradient_diagnostics,
)
from sph.first_miss_max_regret_selector import FirstMissMaxRegretSelector
from sph.first_miss_value_selector import (
    FirstMissValueOutput,
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
    gold_ids = torch.tensor([[10, 21, 30], [40, 50, 60], [99, 51, 60]])
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


def make_model() -> FirstMissMaxRegretSelector:
    return FirstMissMaxRegretSelector(
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
            initialization_seed=41,
        )
    )


class OneBatchLoader:
    def __init__(self, batch: dict[str, object]) -> None:
        self.batch = batch
        self.dataset = list(range(len(batch["sample_ids"])))

    def __iter__(self):
        yield self.batch


class ControlledModel(torch.nn.Module):
    """Select one beneficial, one harmful, and one neutral edit."""

    def eval(self):
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
        residuals[0, 1, 1] = 5.0
        residuals[1, 0, 1] = 5.0
        residuals[2, 1, 1] = 5.0
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


def gate_args() -> Namespace:
    return Namespace(
        min_harmful_nonpositive_recall=0.99,
        expected_harmful_actions=57_765,
        max_mean_hinge=0.0030078125,
        expected_beneficial_actions=256,
        min_beneficial_positive_count=254,
        min_utility_optimal_count=244,
        min_prompt_oracle_gap_recovered=0.95,
        max_harmed_blocks=5,
        max_no_benefit_false_edits=2,
        expected_oracle_gain_tokens=462,
    )


def passing_evaluation() -> dict[str, object]:
    return {
        "loss": {"mean_block_hinge": 0.0030078125},
        "bound": {
            "violations_beyond_tolerance": 0,
            "minimum_slack": -1e-6,
        },
        "signed_score": {
            "beneficial_actions": 256,
            "beneficial_predicted_positive": 254,
            "harmful_actions": 57_765,
            "harmful_predicted_nonpositive": 57_188,
        },
        "decision": {
            "repairable_blocks": 256,
            "utility_optimal_selected": 244,
            "prompt_balanced_oracle_gap_recovered": 0.95,
            "selected_harmful": 5,
            "no_benefit_blocks": 256,
            "no_benefit_false_edits": 2,
            "oracle_gain_tokens_block_weighted": 462,
        },
        "blocks": 512,
        "prompts": 459,
    }


def test_epoch_zero_evaluation_is_exact_dflash_and_bound_reconstructs() -> None:
    report = evaluate(
        make_model(),
        [make_batch()],
        torch.randn(100, 8),
        torch.device("cpu"),
        candidate_k=4,
        include_examples=True,
        require_base_identity=True,
    )
    assert report["base"]["mean_accepted_draft_tokens_prompt_balanced"] == report[
        "camrs"
    ]["mean_accepted_draft_tokens_prompt_balanced"]
    assert report["decision"]["selected_edits"] == 0
    assert report["decision"]["selected_harmful"] == 0
    assert report["bound"]["violations_beyond_tolerance"] == 0
    for example in report["examples"]:
        assert example["hinge"] - example["decoded_regret"] >= -1e-6


def test_decision_and_competitor_accounting_cover_all_outcomes() -> None:
    report = evaluate(
        ControlledModel(),
        [make_batch()],
        torch.randn(100, 8),
        torch.device("cpu"),
        candidate_k=4,
        include_examples=True,
    )
    decision = report["decision"]
    assert decision["selected_edits"] == 3
    assert decision["selected_beneficial"] == 1
    assert decision["selected_harmful"] == 1
    assert decision["selected_neutral"] == 1
    assert decision["repairable_blocks"] == 1
    assert decision["utility_optimal_selected"] == 1
    assert decision["no_benefit_blocks"] == 2
    assert decision["no_benefit_false_edits"] == 2
    assert decision["harmed_fraction"] == 1.0 / 3.0
    assert report["camrs"]["mean_accepted_draft_tokens_prompt_balanced"] == 1.0
    assert report["single_edit_oracle"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ] == 2.0
    assert report["bound"]["violations_beyond_tolerance"] == 0
    assert len(report["competitor"]["actions_by_evaluation_order"]) == 3


def test_projection_diagnostics_split_oracle_and_competitor_gradients() -> None:
    batch = make_batch()
    report = projection_gradient_diagnostics(
        make_model(),
        OneBatchLoader(batch),
        torch.randn(100, 8),
        torch.device("cpu"),
    )
    assert report["blocks"] == 3
    assert report["active_blocks"] > 0
    assert report["active_repairable_blocks"] == 1
    assert report["oracle_upward_projection_gradient_norm"] > 0
    assert report["competitor_downward_projection_gradient_norm"] > 0
    assert report["total_projection_gradient_norm"] > 0


def test_projection_diagnostics_allow_keep_only_oracles() -> None:
    batch = make_batch()
    keep_only: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            keep_only[key] = value[1:2]
        elif isinstance(value, list):
            keep_only[key] = value[1:2]
        else:
            keep_only[key] = value
    report = projection_gradient_diagnostics(
        make_model(),
        OneBatchLoader(keep_only),
        torch.randn(100, 8),
        torch.device("cpu"),
    )
    assert report["active_repairable_blocks"] == 0
    assert report["oracle_upward_projection_gradient_norm"] == 0
    assert report["competitor_downward_projection_gradient_norm"] > 0


def test_capacity_gate_uses_exact_discrete_conjunction() -> None:
    evaluation = passing_evaluation()
    gate = capacity_gate_report(
        evaluation, gate_args(), epoch_zero_identity=True
    )
    assert gate["passed"]
    assert len(gate["checks"]) == 18
    assert gate["thresholds"]["mean_block_hinge"] == 0.0030078125
    assert gate["thresholds"]["harmful_predicted_nonpositive"] == 57_188
    evaluation["decision"]["no_benefit_false_edits"] = 3
    gate = capacity_gate_report(
        evaluation, gate_args(), epoch_zero_identity=True
    )
    assert not gate["passed"]
    assert not gate["checks"]["no_benefit_false_edits"]


def test_capacity_gate_rejects_nonfinite_and_impossible_values() -> None:
    mutations = [
        ("loss", "mean_block_hinge", -torch.inf),
        ("loss", "mean_block_hinge", -0.001),
        ("bound", "minimum_slack", torch.inf),
        ("decision", "prompt_balanced_oracle_gap_recovered", torch.inf),
        ("decision", "prompt_balanced_oracle_gap_recovered", 1.000002),
        ("signed_score", "beneficial_predicted_positive", 257),
        ("signed_score", "beneficial_predicted_positive", -1),
        ("signed_score", "harmful_predicted_nonpositive", 57_766),
        ("signed_score", "harmful_predicted_nonpositive", -1),
        ("decision", "utility_optimal_selected", 257),
        ("decision", "utility_optimal_selected", -1),
        ("decision", "selected_harmful", -1),
        ("decision", "selected_harmful", 513),
        ("decision", "no_benefit_false_edits", -1),
        ("decision", "no_benefit_false_edits", 257),
        (None, "blocks", 511),
        (None, "prompts", 458),
    ]
    for section, key, value in mutations:
        evaluation = deepcopy(passing_evaluation())
        if section is None:
            evaluation[key] = value
        else:
            evaluation[section][key] = value
        gate = capacity_gate_report(
            evaluation, gate_args(), epoch_zero_identity=True
        )
        assert not gate["passed"], (section, key, value)


def test_epoch_zero_identity_is_a_binding_gate() -> None:
    gate = capacity_gate_report(
        passing_evaluation(), gate_args(), epoch_zero_identity=False
    )
    assert not gate["passed"]
    assert not gate["checks"]["epoch_zero_identity"]


def test_competitor_churn_is_order_sensitive_and_fail_closed() -> None:
    assert _competitor_churn([1, 2, 3], None) is None
    assert _competitor_churn([1, 4, 3], [1, 2, 3]) == 1.0 / 3.0
    try:
        _competitor_churn([1], [1, 2])
    except RuntimeError as error:
        assert "length changed" in str(error)
    else:
        raise AssertionError("mismatched competitor vectors must fail")
