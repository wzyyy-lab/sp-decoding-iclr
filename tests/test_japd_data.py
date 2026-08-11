from __future__ import annotations

import torch
import pytest

from sph.japd import BLOCK_LENGTH, CANDIDATES
from sph.japd_data import (
    FORBIDDEN_ONLINE_FEATURE_FIELDS,
    HEAD_BATCH_FIELDS,
    attach_lse_sidecar,
    collate_japd_records,
    record_key,
    stratified_prompt_split,
)


def make_record(sample_id: str, domain: str, offset: int = 0) -> dict:
    candidate_ids = torch.arange(
        BLOCK_LENGTH * CANDIDATES, dtype=torch.long
    ).reshape(BLOCK_LENGTH, CANDIDATES) + 1000 * offset
    gold = candidate_ids[:, 0].clone()
    return {
        "sample_id": sample_id,
        "domain": domain,
        "split": "train",
        "anchor_offset": offset,
        "context_length": 32 + offset,
        "anchor_token_id": 7,
        "parallel_hidden": torch.zeros((BLOCK_LENGTH, 2560), dtype=torch.bfloat16),
        "base_topk_ids": candidate_ids,
        "base_topk_logits": torch.zeros((BLOCK_LENGTH, CANDIDATES)),
        "base_logsumexp": torch.ones(BLOCK_LENGTH),
        "gold_ids": gold,
        "target_candidate_logits": torch.zeros((BLOCK_LENGTH, CANDIDATES)),
        "target_top1_ids": gold.clone(),
        "policy_ids": gold.clone(),
        # It may exist in the canonical rollout, but collate must never expose it.
        "target_anchor_early_feature": torch.randn(2560),
    }


def test_collate_is_whitelist_only_and_ignores_target_online_feature() -> None:
    record = make_record("p0", "code")
    batch = collate_japd_records(
        [record], prompt_effective_counts={"p0": 1}
    )
    assert set(batch) == HEAD_BATCH_FIELDS
    assert not set(batch).intersection(FORBIDDEN_ONLINE_FEATURE_FIELDS)
    assert batch["hidden"].shape == (1, BLOCK_LENGTH, 2560)
    assert batch["candidate_ids"].shape == (1, BLOCK_LENGTH, CANDIDATES)
    assert torch.equal(batch["policy_ids"][0], record["policy_ids"])


def test_collate_rejects_horizon_zero_even_if_prompt_has_other_valid_blocks() -> None:
    record = make_record("p0", "code")
    record["gold_ids"][0] = 999_999
    with pytest.raises(RuntimeError, match="horizon-zero"):
        collate_japd_records(
            [record], prompt_effective_counts={"p0": 2}
        )


def test_sidecar_attach_allows_selecting_subset_without_mutating_source() -> None:
    first = make_record("p0", "code", 0)
    second = make_record("p1", "math", 1)
    first.pop("base_logsumexp")
    sidecar = {
        record_key(first): torch.arange(BLOCK_LENGTH).float(),
        record_key(second): torch.ones(BLOCK_LENGTH),
    }
    attached = attach_lse_sidecar([first], sidecar)
    assert "base_logsumexp" not in first
    assert torch.equal(attached[0]["base_logsumexp"], sidecar[record_key(first)])


def test_stratified_split_is_label_independent_disjoint_and_exact() -> None:
    records = []
    counts = {"chat": (2, 1, 1), "code": (2, 1, 1), "math": (2, 1, 1)}
    for domain in counts:
        for index in range(4):
            records.append(make_record(f"{domain}-{index}", domain, index))
    first = stratified_prompt_split(
        records, seed=17, split_counts_by_domain=counts
    )
    second = stratified_prompt_split(
        list(reversed(records)), seed=17, split_counts_by_domain=counts
    )
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "fit": 6,
        "select": 3,
        "diagnostic": 3,
    }
    assert not first["fit"] & first["select"]
    assert not first["fit"] & first["diagnostic"]
    assert not first["select"] & first["diagnostic"]
