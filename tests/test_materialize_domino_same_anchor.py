from __future__ import annotations

from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from materialize_domino_same_anchor import (  # noqa: E402
    MinimalShardWriter,
    prompt_balanced_mean,
    validate_domino_contract,
)


def test_minimal_shard_writer_flushes_exact_block_counts(tmp_path: Path) -> None:
    writer = MinimalShardWriter(tmp_path, shard_blocks=2)
    writer.add({"value": torch.tensor(1)})
    writer.add({"value": torch.tensor(2)})
    writer.add({"value": torch.tensor(3)})
    writer.flush()
    assert writer.total == 3
    assert [item["blocks"] for item in writer.shards] == [2, 1]
    assert [path.name for path in sorted(tmp_path.glob("shard-*.pt"))] == [
        "shard-00000.pt",
        "shard-00001.pt",
    ]

    resumed = MinimalShardWriter(tmp_path, shard_blocks=2)
    restored = [record for shard in resumed.restore_existing_shards() for record in shard]
    assert [int(record["value"]) for record in restored] == [1, 2, 3]
    assert resumed.total == 3
    resumed.add({"value": torch.tensor(4)})
    resumed.flush()
    assert [item["blocks"] for item in resumed.shards] == [2, 1, 1]
    assert (tmp_path / "shard-00002.pt").is_file()


def test_writer_rejects_autograd_tensors(tmp_path: Path) -> None:
    writer = MinimalShardWriter(tmp_path, shard_blocks=2)
    value = torch.ones(2, requires_grad=True)
    try:
        writer.add({"parallel_hidden": value})
    except ValueError as error:
        assert "requires gradients" in str(error)
    else:
        raise AssertionError("writer accepted a tensor with an attached graph")


class _Config:
    dflash_config = {"projector_type": "domino", "shift_label": True}


class _Domino:
    config = _Config()
    projector_type = "domino"
    pure_draft_prefix_len = 1
    block_size = 16


def test_domino_contract_and_prompt_balanced_mean() -> None:
    validate_domino_contract(_Domino(), draft_positions=15)
    assert prompt_balanced_mean({"short": [1], "long": [3, 5, 7]}) == 3.0

    bad = _Domino()
    bad.block_size = 15
    try:
        validate_domino_contract(bad, draft_positions=15)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("15/16 alignment mismatch was not rejected")
