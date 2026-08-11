from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch

from scripts.materialize_canonical_prompt_subset import (
    load_verified_shard,
    main,
    read_collected_prompt_ids,
    select_prompt_ids,
    sha256_file,
    write_selected_manifest,
)
from scripts.train_global_direct_selector import deterministic_prompt_subset


class PromptSubsetMaterializationTest(unittest.TestCase):
    def test_selection_exactly_matches_trainer_hash_ranking(self) -> None:
        records = [
            {"sample_id": f"sample-{index}", "domain": "math"}
            for index in range(20)
            for _ in range(2)
        ]
        trainer = deterministic_prompt_subset(
            records, max_prompts=7, seed=20260730
        )
        expected = {record["sample_id"] for record in trainer}
        materialized = select_prompt_ids(
            {f"sample-{index}": "math" for index in range(20)},
            {f"sample-{index}" for index in range(20)},
            max_prompts=7,
            seed=20260730,
        )
        self.assertEqual(materialized, expected)

    def test_selection_is_strictly_hash_ranked(self) -> None:
        sample_ids = {"a", "b", "c", "d"}
        selected = select_prompt_ids(
            {sample_id: "code" for sample_id in sample_ids},
            sample_ids,
            max_prompts=2,
            seed=17,
        )
        expected = set(
            sorted(
                sample_ids,
                key=lambda item: hashlib.sha256(
                    f"17\0{item}".encode("utf-8")
                ).digest(),
            )[:2]
        )
        self.assertEqual(selected, expected)

    def test_collection_log_parser_keeps_only_nonempty_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "collection.out"
            path.write_text(
                "noise\n"
                "[1/3] sample:a: 8 blocks in 1.00s\n"
                "[2/3] sample:b: 0 blocks in 0.10s\n"
                "[3/3] sample:c: 4 blocks in 1.20s\n",
                encoding="utf-8",
            )
            result = read_collected_prompt_ids([path])
        self.assertEqual(result, {"sample:a", "sample:c"})

    def test_selected_manifest_spans_sources_in_original_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = [root / "part0.jsonl", root / "part1.jsonl"]
            sources[0].write_text(
                json.dumps({"sample_id": "a", "domain": "math"}) + "\n"
                + json.dumps({"sample_id": "b", "domain": "code"})
                + "\n",
                encoding="utf-8",
            )
            sources[1].write_text(
                json.dumps({"sample_id": "c", "domain": "chat"}) + "\n",
                encoding="utf-8",
            )
            output = root / "selected.jsonl"
            digest = write_selected_manifest(sources, {"a", "c"}, output)
            records = [
                json.loads(line) for line in output.read_text().splitlines()
            ]
            observed_digest = sha256_file(output)
        self.assertEqual([record["sample_id"] for record in records], ["a", "c"])
        self.assertEqual(digest, observed_digest)

    def test_verified_shard_rejects_same_size_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shard.pt"
            torch.save([{"sample_id": "a"}], path)
            entry = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            self.assertEqual(load_verified_shard(path, entry)[0]["sample_id"], "a")
            payload = bytearray(path.read_bytes())
            payload[len(payload) // 2] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                load_verified_shard(path, entry)

    def test_materializer_artifact_failure_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stream = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "materialize_canonical_prompt_subset.py",
                    "--source",
                    str(root / "missing-source"),
                    "--collection-log",
                    str(root / "missing.log"),
                    "--output",
                    str(root / "output"),
                    "--max-prompts",
                    "2",
                ],
            ), redirect_stdout(stream):
                with self.assertRaises(SystemExit) as raised:
                    main()
        self.assertEqual(int(raised.exception.code), 2)
        self.assertIn('"status": "artifact_error"', stream.getvalue())


if __name__ == "__main__":
    unittest.main()
