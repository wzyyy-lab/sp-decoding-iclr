from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_pgcf16_gate2 import (  # noqa: E402
    assert_matched_checkpoint_configs,
    block_identity,
    build_donor_map,
    coherent_remote_inputs,
    context_quartile,
    paired_prompt_bootstrap,
)


def donor_record(index: int) -> dict:
    return {
        "sample_id": f"prompt-{index:02d}",
        "anchor_offset": index * 17,
        "context_length": 20 + index * 10,
        "domain": "code",
        "gold_ids": torch.full((16,), index),
    }


def test_donor_map_is_cross_prompt_within_cell_and_label_independent() -> None:
    records = [donor_record(index) for index in range(16)]
    donors, protocol = build_donor_map(records)
    boundaries = tuple(protocol["context_quartile_boundaries"])
    for recipient_index, donor_index in enumerate(donors):
        assert recipient_index != donor_index
        assert records[recipient_index]["sample_id"] != records[donor_index]["sample_id"]
        assert context_quartile(
            records[recipient_index]["context_length"], boundaries
        ) == context_quartile(records[donor_index]["context_length"], boundaries)
    before = list(donors)
    for record in records:
        record["gold_ids"].random_(0, 1000)
    after, after_protocol = build_donor_map(records)
    assert before == after
    assert protocol == after_protocol
    assert protocol["label_fields_used"] == []


def test_donor_map_rejects_same_prompt_only_cell() -> None:
    records = [donor_record(index) for index in range(8)]
    for record in records:
        record["sample_id"] = "one-prompt"
    with pytest.raises(RuntimeError, match="cross-prompt donor"):
        build_donor_map(records)


def test_coherent_remote_inputs_preserve_only_the_recipient_diagonal() -> None:
    batch = 2
    recipient = {
        "hidden": torch.arange(batch * 16 * 3).reshape(batch, 16, 3),
        "candidate_ids": torch.arange(batch * 16 * 4).reshape(batch, 16, 4),
        "candidate_logits": torch.arange(batch * 16 * 4).reshape(batch, 16, 4).float(),
        "anchor_ids": torch.tensor([7, 9]),
    }
    donor = {
        key: value + 10_000 if key != "anchor_ids" else value + 100
        for key, value in recipient.items()
    }
    mixed = coherent_remote_inputs(recipient, donor)
    assert torch.equal(mixed["anchor_ids"].reshape(batch, 16)[:, 0], torch.tensor([7, 9]))
    for key in ("hidden", "candidate_ids", "candidate_logits"):
        shaped = mixed[key].reshape(batch, 16, *recipient[key].shape[1:])
        for intervention in range(16):
            for position in range(16):
                expected = (
                    recipient[key][:, position]
                    if position == intervention
                    else donor[key][:, position]
                )
                assert torch.equal(shaped[:, intervention, position], expected)


def test_paired_bootstrap_clusters_by_prompt_and_is_deterministic() -> None:
    records = [
        {"sample_id": "a"},
        {"sample_id": "a"},
        {"sample_id": "b"},
    ]
    first = [3.0, 5.0, 9.0]
    second = [1.0, 1.0, 8.0]
    first_report = paired_prompt_bootstrap(
        records, first, second, draws=1000, seed=123
    )
    second_report = paired_prompt_bootstrap(
        records, first, second, draws=1000, seed=123
    )
    assert first_report == second_report
    assert first_report["prompts"] == 2
    assert first_report["point_delta"] == 2.0


def test_checkpoint_config_comparison_allows_only_head_and_output() -> None:
    global_checkpoint = {
        "config": {"head": "global", "output": "g", "seed": 0, "max_steps": 20_000}
    }
    local_checkpoint = {
        "config": {"head": "local", "output": "l", "seed": 0, "max_steps": 20_000}
    }
    assert assert_matched_checkpoint_configs(
        global_checkpoint, local_checkpoint
    )["matched"]
    local_checkpoint["config"]["seed"] = 1
    with pytest.raises(RuntimeError, match="configurations differ"):
        assert_matched_checkpoint_configs(global_checkpoint, local_checkpoint)


def test_block_identity_includes_anchor_offset() -> None:
    first = donor_record(0)
    second = dict(first, anchor_offset=17)
    assert block_identity(first) != block_identity(second)
