from __future__ import annotations

import torch

from analyze_r049_depth_probe import (
    centered_candidate_scores,
    depth_gate_decision,
    force_keep_rows,
    frontier_token_ranking_lengths,
    parse_layers,
)


def test_centered_scores_preserve_argmax_and_zero_proposal() -> None:
    proposal = torch.tensor([[3, 5]])
    candidates = torch.tensor([[[3, 4], [7, 5]]])
    scores = torch.tensor([[[2.0, 4.0], [1.0, 3.0]]])
    centered = centered_candidate_scores(scores, candidates, proposal)
    assert centered.tolist() == [[[0.0, 2.0], [-2.0, 0.0]]]
    assert centered.argmax(dim=-1).tolist() == scores.argmax(dim=-1).tolist()


def test_true_frontier_token_ranking_uses_exact_rerun_length() -> None:
    proposal = torch.tensor([[1, 9, 3], [1, 2, 9]])
    verifier = torch.tensor([[1, 2, 8], [1, 2, 3]])
    candidates = torch.tensor(
        [[[1, 7], [9, 2], [3, 8]], [[1, 7], [2, 8], [9, 3]]]
    )
    scores = torch.tensor(
        [[[2.0, 1.0], [1.0, 3.0], [2.0, 1.0]], [[2.0, 1.0], [2.0, 1.0], [3.0, 1.0]]]
    )
    lengths = frontier_token_ranking_lengths(
        proposal=proposal,
        verifier_top1=verifier,
        candidate_ids=candidates,
        candidate_scores=scores,
        baseline_lengths=torch.tensor([1, 2]),
        oracle_lengths=torch.tensor([3, 3]),
    )
    assert lengths.tolist() == [3, 2]


def test_numerically_ambiguous_rows_are_forced_to_keep() -> None:
    proposal = torch.tensor([[3, 5]])
    candidates = torch.tensor([[[3, 4], [7, 5]]])
    scores = torch.tensor([[[0.0, 2.0], [3.0, 0.0]]])
    forced = force_keep_rows(
        scores,
        candidates,
        proposal,
        torch.tensor([[True, False]]),
    )
    assert forced[0, 0].tolist() == [0.0, -torch.inf]
    assert forced[0, 1].tolist() == [3.0, 0.0]


def test_depth_gate_prefers_direct_then_residual_and_closes() -> None:
    def reports(token: float, policy: float) -> dict[int, dict]:
        return {
            4: {
                "token_ranking": {"oracle_gain_recovery": token, "gain_block_recovery": token},
                "policy": {"oracle_gain_recovery": policy, "gain_block_recovery": policy},
            },
            16: {
                "token_ranking": {"oracle_gain_recovery": 1.0, "gain_block_recovery": 1.0},
                "policy": {"oracle_gain_recovery": 1.0, "gain_block_recovery": 1.0},
            },
        }

    assert depth_gate_decision(reports(0.95, 0.92))["decision"] == "GO_DISJOINT_DIRECT"
    assert depth_gate_decision(reports(0.95, 0.40))["decision"] == "GO_ONE_KEEP_GATE_CAPACITY"
    assert depth_gate_decision(reports(0.85, 0.40))["decision"] == "GO_ONE_RESIDUAL_GATE_CAPACITY"
    assert depth_gate_decision(reports(0.70, 0.40))["decision"] == "CLOSE_EARLY_TARGET_ROUTE"
    assert parse_layers("12,4,8,8", 36) == [4, 8, 12]
