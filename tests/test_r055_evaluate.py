from __future__ import annotations

import pytest
import torch

from evaluate_r055_padded_forest import (
    clean_domino_control,
    enforce_forest_controls,
    select_balanced_smoke_records,
)


def test_balanced_smoke_uses_distinct_prompts_and_typical_contexts() -> None:
    records = []
    for domain in ("chat", "code", "math"):
        for prompt_index, context in enumerate((20, 150, 170)):
            sample_id = f"{domain}-{prompt_index}"
            records.extend(
                [
                    {
                        "domain": domain,
                        "sample_id": sample_id,
                        "context_length": context,
                    },
                    {
                        "domain": domain,
                        "sample_id": sample_id,
                        "context_length": context + 5,
                    },
                ]
            )

    selected = select_balanced_smoke_records(records, 6)
    assert len(selected) == 6
    assert {record["domain"] for record in selected} == {"chat", "code", "math"}
    for domain in ("chat", "code", "math"):
        domain_records = [record for record in selected if record["domain"] == domain]
        assert len(domain_records) == 2
        assert len({record["sample_id"] for record in domain_records}) == 2
        assert all(record["context_length"] >= 150 for record in domain_records)


def test_fast_control_may_differ_but_baseline_uses_released_domino() -> None:
    released = torch.tensor([[2, 3, 4]])
    fast = torch.tensor([[2, 9, 4]])
    clean = torch.tensor([[2, 3, 4]])
    length, diagnostic = clean_domino_control(
        released_control=released,
        stored_released=released.clone(),
        fast_control=fast,
        clean_gold=clean,
    )
    assert length == 3
    assert diagnostic == 1


def test_forest_trunk_mismatch_remains_a_hard_failure() -> None:
    with pytest.raises(RuntimeError, match="changed Fast-K64 trunk"):
        enforce_forest_controls(
            same_job_domino_mismatches=0,
            trunk_mismatches={4: 0, 8: 1, 16: 0},
        )
