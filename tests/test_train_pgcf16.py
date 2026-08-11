from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_pgcf16 import (  # noqa: E402
    CAPACITY_GATE_KEYS,
    accepted_lengths,
    capacity_gold_ce_loss,
    candidate_ranks,
    collate_records,
    effective_loss_progress,
    evaluate,
    fixed_stride_diagnostic_records,
    make_loader,
    prompt_balanced,
    record_identity,
    validate_capacity_mode_pair,
    validate_record,
)
from sph.parallel_global_candidate_fusion import (  # noqa: E402
    ParallelGlobalCandidateFusionHead,
    PGCFOutput,
)


def synthetic_record(sample_id: str, *, base_correct: int = 4) -> dict:
    candidate_ids = torch.arange(16, dtype=torch.int32).repeat(16, 1)
    gold = candidate_ids[:, 1].clone()
    gold[:base_correct] = candidate_ids[:base_correct, 0]
    policy = gold.clone()
    target_logits = torch.zeros(16, 16)
    target_logits.scatter_(1, torch.ones(16, 1, dtype=torch.long), 3.0)
    return {
        "sample_id": sample_id,
        "domain": "code",
        "split": "train",
        "anchor_token_id": 31,
        "parallel_hidden": torch.randn(16, 2560).bfloat16(),
        "base_topk_ids": candidate_ids,
        "base_topk_logits": torch.linspace(2.0, -2.0, 16).repeat(16, 1),
        "gold_ids": gold,
        "policy_ids": policy,
        "target_candidate_logits": target_logits,
        "target_top1_ids": gold.clone(),
    }


def test_candidate_rank_uses_minus_one_only_for_unsupported_labels() -> None:
    ids = torch.tensor([[[5, 4, 3], [8, 7, 6]]])
    labels = torch.tensor([[4, 99]])
    assert torch.equal(candidate_ranks(ids, labels), torch.tensor([[1, -1]]))


def test_collate_preserves_all_sixteen_positions_and_builds_labels() -> None:
    batch = collate_records([synthetic_record("a"), synthetic_record("b")])
    assert batch["hidden"].shape == (2, 16, 2560)
    assert batch["candidate_ids"].shape == (2, 16, 16)
    assert batch["gold_candidate_ranks"].shape == (2, 16)
    assert batch["teacher_candidate_ranks"].shape == (2, 16)
    assert batch["target_candidate_logits"].shape == (2, 16, 16)
    assert batch["target_matches_gold"].all()
    assert torch.equal(batch["gold_candidate_ranks"][:, :4], torch.zeros(2, 4))
    assert torch.equal(batch["gold_candidate_ranks"][:, 4:], torch.ones(2, 12))


def test_record_geometry_rejects_fifteen_position_cache() -> None:
    record = synthetic_record("a")
    record["parallel_hidden"] = record["parallel_hidden"][:15]
    with pytest.raises(RuntimeError, match="parallel_hidden"):
        validate_record(record)


def test_accepted_lengths_and_prompt_balance() -> None:
    proposal = torch.tensor([[1, 2, 9, 4], [1, 0, 3, 4]])
    gold = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    assert torch.equal(accepted_lengths(proposal, gold), torch.tensor([2, 1]))
    assert prompt_balanced(["a", "a", "b"], [2.0, 4.0, 9.0]) == 6.0


def test_fixed_stride_train_diagnostic_is_label_independent() -> None:
    records = []
    for index in range(20):
        record = synthetic_record(f"sample-{index}")
        record["anchor_offset"] = index * 3
        record["context_length"] = 40 + index
        records.append(record)
    indices, selected = fixed_stride_diagnostic_records(
        records, count=4, stride=5
    )
    assert indices == [0, 5, 10, 15]
    assert [record["sample_id"] for record in selected] == [
        "sample-0",
        "sample-5",
        "sample-10",
        "sample-15",
    ]
    assert set(record_identity(selected[0])) == {
        "sample_id",
        "domain",
        "anchor_offset",
        "context_length",
    }
    selected[0]["gold_ids"].fill_(999)
    assert fixed_stride_diagnostic_records(records, count=4, stride=5)[0] == indices


def test_fixed_stride_train_diagnostic_fails_closed_on_invalid_manifest() -> None:
    records = [synthetic_record(str(index)) for index in range(5)]
    with pytest.raises(ValueError, match="exceeds canonical"):
        fixed_stride_diagnostic_records(records, count=3, stride=3)
    with pytest.raises(ValueError, match="positive"):
        fixed_stride_diagnostic_records(records, count=1, stride=0)


def test_cpu_identity_evaluator_matches_base_on_full16_records() -> None:
    records = [synthetic_record("a", base_correct=4), synthetic_record("b", base_correct=6)]
    loader = make_loader(records, batch_size=2, shuffle=False)
    model = ParallelGlobalCandidateFusionHead(
        hidden_size=2560,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        ff_multiplier=2,
    ).bfloat16()
    target_embedding = torch.randn(64, 2560).bfloat16()
    metrics = evaluate(
        model,
        loader,
        target_embedding,
        torch.device("cpu"),
        require_identity=True,
    )
    assert metrics["model_eal"] == metrics["base_eal"] == 5.0
    assert metrics["released_eal"] == 16.0
    assert metrics["harmed_fraction"] == 0.0
    assert metrics["residual_abs_max"] == 0.0


def test_capacity_gate_modes_match_preregistered_independent_witnesses() -> None:
    assert CAPACITY_GATE_KEYS["target"] == (
        "candidate_accuracy",
        "hard_candidate_accuracy",
        "oracle_gap_recovered",
        "harmed_fraction",
    )
    assert CAPACITY_GATE_KEYS["teacher"] == ("teacher_action_accuracy",)
    assert set(CAPACITY_GATE_KEYS["combined"]) == (
        set(CAPACITY_GATE_KEYS["target"]) | set(CAPACITY_GATE_KEYS["teacher"])
    )


def test_teacher_only_progress_and_gate_pairing_fail_closed() -> None:
    assert effective_loss_progress("teacher_only", 0.91) == 0.0
    assert effective_loss_progress("curriculum", 0.91) == 0.91
    assert effective_loss_progress("gold_ce", 0.91) == 0.0
    validate_capacity_mode_pair("teacher_only", "teacher")
    validate_capacity_mode_pair("curriculum", "target")
    validate_capacity_mode_pair("gold_ce", "target")
    with pytest.raises(ValueError, match="allows loss modes"):
        validate_capacity_mode_pair("curriculum", "teacher")
    with pytest.raises(ValueError, match="allows loss modes"):
        validate_capacity_mode_pair("teacher_only", "target")


def test_capacity_gold_ce_uses_all_supported_rows_only() -> None:
    scores = torch.zeros(1, 16, 16, requires_grad=True)
    output = PGCFOutput(
        scores=scores,
        residual_scores=torch.zeros_like(scores),
        candidate_states=torch.empty(1, 16, 16, 1),
    )
    ranks = torch.full((1, 16), -1, dtype=torch.long)
    ranks[0, 0] = 2
    ranks[0, 9] = 4
    loss = capacity_gold_ce_loss(output, ranks)
    assert loss.gold_support.sum().item() == 2
    assert loss.lambda_prefix == loss.lambda_target_kl == loss.lambda_teacher == 0
    assert not loss.target_kl_positions.any()
    assert not loss.teacher_positions.any()
    loss.loss.backward()
    assert scores.grad[0, 9].abs().sum() > 0
