from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from sph.first_miss_capacity import (
    build_capacity_manifest,
    describe_capacity_record,
    verify_capacity_manifest,
)


def make_record(
    *, sample_id: str, states: list[int], candidate_k: int = 3
) -> dict[str, object]:
    length = len(states)
    gold = torch.arange(100, 100 + length)
    topk = torch.empty(length, candidate_k, dtype=torch.long)
    for position, state in enumerate(states):
        values = torch.arange(
            1000 + 10 * position,
            1000 + 10 * position + candidate_k,
        )
        if state < candidate_k:
            values[state] = gold[position]
        topk[position] = values
    return {
        "sample_id": sample_id,
        "anchor_offset": 7,
        "context_length": 11,
        "anchor_token_id": 42,
        "domain": "chat",
        "split": "train",
        "base_topk_ids": topk,
        "gold_ids": gold,
    }


class FirstMissCapacityManifestTest(unittest.TestCase):
    def test_describes_all_target_kinds_and_gain(self) -> None:
        full = describe_capacity_record(
            make_record(sample_id="full", states=[0, 0, 0]),
            candidate_k=3,
        )
        out = describe_capacity_record(
            make_record(sample_id="out", states=[0, 3, 1]),
            candidate_k=3,
        )
        edit = describe_capacity_record(
            make_record(sample_id="edit", states=[0, 2, 0]),
            candidate_k=3,
        )
        self.assertEqual(full["target_kind"], "keep_full_correct")
        self.assertEqual(out["target_kind"], "keep_out_of_k")
        self.assertEqual(edit["target_kind"], "edit")
        self.assertEqual(edit["first_miss_position"], 1)
        self.assertEqual(edit["gold_rank"], 2)
        self.assertEqual(edit["gain"], 2)
        self.assertEqual(edit["action"], 1 + 1 * 2 + 1)

    def test_manifest_is_deterministic_and_verifies_fail_closed(self) -> None:
        records = [
            make_record(sample_id="a", states=[0, 1, 0]),
            make_record(sample_id="b", states=[0, 0, 0]),
            make_record(sample_id="c", states=[3, 0, 0]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary) / "metadata.json"
            metadata.write_text(json.dumps({"stable": True}))
            first = build_capacity_manifest(
                records,
                source_metadata_path=metadata,
                candidate_k=3,
                seed=0,
                opportunity_fraction=0.5,
            )
            second = build_capacity_manifest(
                records,
                source_metadata_path=metadata,
                candidate_k=3,
                seed=0,
                opportunity_fraction=0.5,
            )
            self.assertEqual(first, second)
            verify_capacity_manifest(
                first,
                records,
                source_metadata_path=metadata,
                candidate_k=3,
                seed=0,
                opportunity_fraction=0.5,
            )
            corrupted = json.loads(json.dumps(first))
            corrupted["entries"][0]["action"] += 1
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                verify_capacity_manifest(
                    corrupted,
                    records,
                    source_metadata_path=metadata,
                    candidate_k=3,
                    seed=0,
                    opportunity_fraction=0.5,
                )


if __name__ == "__main__":
    unittest.main()
