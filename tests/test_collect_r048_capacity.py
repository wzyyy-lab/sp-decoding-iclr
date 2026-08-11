from __future__ import annotations

import torch

from collect_r048_capacity import (
    authoritative_frontier_contract,
    prompt_balanced,
    select_balanced_prompts,
)


def test_select_balanced_prompts_round_robins_domains() -> None:
    records = []
    for domain, names in {"chat": ["c0", "c1"], "code": ["d0", "d1"], "math": ["m0"]}.items():
        for name in names:
            records.append({"sample_id": name, "domain": domain})
            records.append({"sample_id": name, "domain": domain})
    selected = select_balanced_prompts(records, 4)
    assert selected == ["c0", "d0", "m0", "c1"]


def test_capacity_prompt_balance_is_not_block_weighted() -> None:
    value = prompt_balanced(["a", "a", "b"], [1, 3, 8])
    assert value == 5.0


def test_authoritative_frontier_is_contiguous_after_mismatch_then_match() -> None:
    proposal = torch.tensor([[1, 2, 3, 4]])
    # The clean verifier first rejects row 1, then happens to match again.
    verifier_top1 = torch.tensor([[1, 9, 3, 4]])
    candidates = torch.tensor(
        [[[1, 8], [2, 9], [3, 7], [4, 6]]]
    )
    accepted, valid, repair = authoritative_frontier_contract(
        proposal,
        verifier_top1,
        candidates,
    )
    assert accepted.tolist() == [1]
    assert valid.tolist() == [[True, True, False, False]]
    assert repair.frontier.tolist() == [1]
    assert repair.repair_available.tolist() == [True]
    assert repair.token_ids.tolist() == [[1, 9, 3, 4]]
