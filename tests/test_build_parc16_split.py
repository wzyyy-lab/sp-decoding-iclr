from __future__ import annotations

from collections import Counter

import pytest

from scripts import build_parc16_split as split


def synthetic_records() -> list[dict[str, str]]:
    counts = {"chat": 90_000, "code": 90_000, "math": 90_000}
    return [
        {
            "sample_id": f"{domain}:{index}",
            "domain": domain,
            "source": "fixture",
            "prompt": f"prompt {domain} {index}",
            "split": "train",
        }
        for domain, count in counts.items()
        for index in range(count)
    ]


def test_frozen_split_is_disjoint_balanced_and_deterministic() -> None:
    records = synthetic_records()
    first = split.assign_splits(records, split.DEFAULT_SEED)
    second = split.assign_splits(list(reversed(records)), split.DEFAULT_SEED)
    assert {name: len(rows) for name, rows in first.items()} == {
        "train": 240_000,
        "validation": 15_000,
        "heldout": 15_000,
    }
    assert {
        name: [row["sample_id"] for row in rows] for name, rows in first.items()
    } == {
        name: [row["sample_id"] for row in rows] for name, rows in second.items()
    }
    ids = {
        name: {row["sample_id"] for row in rows} for name, rows in first.items()
    }
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["heldout"])
    assert ids["validation"].isdisjoint(ids["heldout"])
    assert Counter(row["domain"] for row in first["train"]) == {
        "chat": 80_000,
        "code": 80_000,
        "math": 80_000,
    }
    assert all(row["required_usable_count"] == 30_000 for row in first["train"])


def test_wrong_source_cardinality_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="exactly"):
        split.assign_splits(synthetic_records()[:-1], split.DEFAULT_SEED)
