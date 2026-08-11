from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.train_parc16 import resolve_run_status

from sph.parc_training import (
    BlockStream,
    DataCatalog,
    ShardInfo,
    checkpoint_is_better,
    cosine_learning_rate,
    grouped_prompt_metrics,
    numeric_certificate,
)


def prompt_record(sample_id: str, split: str = "train") -> dict:
    anchors = []
    for index in range(8):
        anchor = {
            "anchor_offset": index,
            "context_length": index + 1,
            "anchor_token_id": index,
            "gold_ids": torch.arange(16, dtype=torch.int32),
            "reference_proposal_ids": torch.arange(16, dtype=torch.int32),
            "reference_topk_ids": torch.arange(16, dtype=torch.int32)
            .view(1, 16)
            .expand(16, 16)
            .clone(),
            "reference_topk_logits": torch.zeros(16, 16, dtype=torch.float16),
            "reference_accepted_length": 4,
            "reference_delta_fp32": 1.0,
            "numeric_margin_error": 0.0,
        }
        if split == "validation":
            anchor["reference_domino_proposal_ids"] = torch.arange(
                16, dtype=torch.int32
            )
            anchor["reference_domino_accepted_length"] = 5
        anchors.append(anchor)
    return {
        "sample_id": sample_id,
        "domain": "chat",
        "split": split,
        "target_context_features": torch.zeros(8, 12_800, dtype=torch.bfloat16),
        "anchors": anchors,
    }


def shard(tmp_path: Path, name: str, records: list[dict]) -> ShardInfo:
    path = tmp_path / f"{name}.pt"
    torch.save(records, path)
    summaries = tuple(
        {
            "sample_id": record["sample_id"],
            "domain": record["domain"],
            "numeric_margin_error": 0.0,
            "reference_deltas": [1.0] * 8,
            "reference_accepted_lengths": [4] * 8,
            "reference_domino_accepted_lengths": None,
        }
        for record in records
    )
    return ShardInfo(
        path=path,
        split="train",
        prompts=len(records),
        blocks=8 * len(records),
        sample_ids=tuple(record["sample_id"] for record in records),
        summaries=summaries,
    )


def test_block_stream_is_exactly_resumable_and_visits_each_block(tmp_path: Path) -> None:
    shards = (
        shard(tmp_path, "a", [prompt_record("a"), prompt_record("b")]),
        shard(tmp_path, "b", [prompt_record("c")]),
    )
    catalog = DataCatalog(tmp_path, shards, tmp_path, tmp_path, tmp_path)
    stream = BlockStream(catalog, seed=0)
    first_epoch = [stream.next_block() for _ in range(24)]
    assert len({(record["sample_id"], anchor) for record, anchor in first_epoch}) == 24

    prefix = [stream.next_block() for _ in range(7)]
    state = stream.state_dict()
    expected = [stream.next_block() for _ in range(19)]
    resumed = BlockStream(catalog, seed=0, state=state)
    actual = [resumed.next_block() for _ in range(19)]
    assert [(row["sample_id"], anchor) for row, anchor in expected] == [
        (row["sample_id"], anchor) for row, anchor in actual
    ]
    assert len(prefix) == 7


def test_numeric_certificate_uses_train_summaries_only(tmp_path: Path) -> None:
    item = shard(tmp_path, "train", [prompt_record("a")])
    summary = dict(item.summaries[0])
    summary["numeric_margin_error"] = 0.02
    summary["reference_deltas"] = [0.01] + [1.0] * 7
    item = ShardInfo(
        item.path,
        item.split,
        item.prompts,
        item.blocks,
        item.sample_ids,
        (summary,),
    )
    result = numeric_certificate(DataCatalog(tmp_path, (item,), tmp_path, tmp_path, tmp_path))
    assert result["delta_min"] == pytest.approx(0.04)
    assert result["prompt_mean_ambiguous"] == pytest.approx(1 / 8)


def test_prompt_metrics_reduce_within_prompt_before_across_prompts() -> None:
    rows = [
        {"sample_id": "a", "domain": "chat", "eal": 0.0, "actual_harm": 0.0},
        {"sample_id": "a", "domain": "chat", "eal": 2.0, "actual_harm": 0.0},
        {"sample_id": "b", "domain": "code", "eal": 9.0, "actual_harm": 1.0},
    ]
    result = grouped_prompt_metrics(rows)
    assert result["overall"]["eal"] == pytest.approx(5.0)
    assert result["overall"]["actual_harm"] == pytest.approx(0.5)


def test_checkpoint_selection_is_validation_eal_under_harm_gate() -> None:
    best = {"overall": {"eal": 8.0, "actual_harm": 0.01}}
    assert checkpoint_is_better(
        {"overall": {"eal": 8.1, "actual_harm": 0.01}}, best
    )
    assert not checkpoint_is_better(
        {"overall": {"eal": 9.0, "actual_harm": 0.01001}}, best
    )
    assert not checkpoint_is_better(
        {"overall": {"eal": 8.0, "actual_harm": 0.0}}, best
    )


def test_frozen_learning_rate_schedule() -> None:
    assert cosine_learning_rate(
        0, total_steps=180_000, warmup_steps=2_000, peak=3e-4
    ) == pytest.approx(3e-4 / 2_000)
    assert cosine_learning_rate(
        1_999, total_steps=180_000, warmup_steps=2_000, peak=3e-4
    ) == pytest.approx(3e-4)
    assert cosine_learning_rate(
        180_000, total_steps=180_000, warmup_steps=2_000, peak=3e-4
    ) == pytest.approx(3e-5)


def test_terminal_scientific_stops_are_not_resumable_or_heldout_authorized() -> None:
    status, reason, authorized = resolve_run_status(
        global_step=40_000,
        total_steps=180_000,
        eval_every=10_000,
        stop_reason="constraint_infeasible_support_drop",
        best_step=30_000,
    )
    assert (status, reason, authorized) == (
        "stopped_infeasible",
        "constraint_infeasible_support_drop",
        False,
    )
    assert resolve_run_status(
        global_step=40_000,
        total_steps=180_000,
        eval_every=10_000,
        stop_reason="scheduler_checkpoint_request",
        best_step=30_000,
    )[0] == "interrupted_resumable"


def test_step_zero_never_authorizes_heldout() -> None:
    status, reason, authorized = resolve_run_status(
        global_step=180_000,
        total_steps=180_000,
        eval_every=10_000,
        stop_reason=None,
        best_step=0,
    )
    assert status == "stopped_infeasible"
    assert reason == "no_trained_validation_checkpoint_passed_harm_gate"
    assert not authorized
    assert resolve_run_status(
        global_step=180_000,
        total_steps=180_000,
        eval_every=10_000,
        stop_reason=None,
        best_step=10_000,
    ) == ("complete", None, True)
