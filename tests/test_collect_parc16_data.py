from __future__ import annotations

import json

import pytest
import torch

from scripts.collect_parc16_data import (
    accepted_length,
    full16_anchor_offsets,
    manifest_group_quotas,
    read_manifest,
    reference_margin_summary,
)


def test_full16_offsets_require_anchor_plus_sixteen_labels() -> None:
    offsets = full16_anchor_offsets(129, 8)
    assert len(offsets) == 8
    assert offsets[0] == 0
    assert offsets[-1] == 112
    assert offsets[-1] + 16 < 129
    assert full16_anchor_offsets(16, 8) == []


def test_accepted_length_covers_first_reject_and_full_accept() -> None:
    gold = torch.arange(16)
    proposal = gold.clone()
    assert accepted_length(proposal, gold) == 16
    proposal[7] = 99
    assert accepted_length(proposal, gold) == 7


def test_reference_margin_summary_uses_only_protected_rows() -> None:
    hidden = torch.zeros(16, 16, dtype=torch.bfloat16)
    hidden[:, 0] = 1.0
    weight = torch.zeros(32, 16, dtype=torch.bfloat16)
    weight[0, 0] = 2.0
    topk_ids = torch.arange(16).view(1, 16).expand(16, 16).clone()
    topk_logits = torch.einsum(
        "ph,pkh->pk", hidden, torch.nn.functional.embedding(topk_ids, weight)
    ).float()
    delta, error = reference_margin_summary(
        hidden=hidden,
        projection_weight=weight,
        topk_ids=topk_ids,
        topk_logits=topk_logits,
        accepted=4,
    )
    assert delta == 2.0
    assert error == 0.0


def test_reserve_manifest_requires_six_ordered_groups(tmp_path) -> None:
    rows = []
    for split in ("train", "validation"):
        for domain in ("chat", "code", "math"):
            for order in range(2):
                rows.append(
                    {
                        "sample_id": f"{split}:{domain}:{order}",
                        "split": split,
                        "domain": domain,
                        "source": "fixture",
                        "prompt": "valid prompt",
                        "part_index": 0,
                        "part_selection_order": order,
                        "part_required_usable_count": 1,
                        "part_candidate_count": 2,
                    }
                )
    path = tmp_path / "part.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    loaded = read_manifest(path)
    quotas = manifest_group_quotas(loaded)
    assert len(quotas) == 6
    assert set(quotas.values()) == {1}
    assert [row["part_selection_order"] for row in loaded[:2]] == [0, 1]

    broken = rows[:-1]
    path.write_text("".join(json.dumps(row) + "\n" for row in broken))
    with pytest.raises(RuntimeError, match="non-contiguous|reserve"):
        read_manifest(path)
