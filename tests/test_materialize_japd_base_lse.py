from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from scripts.materialize_japd_base_lse import run
from sph.japd import BLOCK_LENGTH, CANDIDATES
from sph.japd_data import load_lse_sidecar


def make_fixture(root: Path) -> tuple[Path, Path, torch.Tensor]:
    target = root / "target"
    target.mkdir()
    generator = torch.Generator().manual_seed(9)
    weight = torch.randn((64, 2560), generator=generator).to(torch.bfloat16)
    save_file(
        {"model.embed_tokens.weight": weight},
        target / "model.safetensors",
    )
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model.safetensors"
                }
            }
        )
    )

    rollout = root / "rollout"
    rollout.mkdir()
    hidden = torch.randn(
        (BLOCK_LENGTH, 2560), generator=generator
    ).to(torch.bfloat16)
    logits = F.linear(hidden, weight)
    top_logits, top_ids = logits.float().topk(CANDIDATES, dim=-1)
    gold = top_ids[:, 0].clone()
    record = {
        "sample_id": "fixture-0",
        "domain": "code",
        "source": "unit",
        "split": "validation_select",
        "anchor_offset": 0,
        "context_length": 32,
        "anchor_token_id": 3,
        "parallel_hidden": hidden,
        "base_topk_ids": top_ids.to(torch.int32),
        "base_topk_logits": top_logits.to(torch.float16),
        "gold_ids": gold.to(torch.int32),
        "target_candidate_logits": top_logits.float(),
        "target_top1_ids": gold.to(torch.int32),
        "policy_ids": gold.to(torch.int32),
    }
    torch.save([record], rollout / "shard-00000.pt")
    (rollout / "metadata.json").write_text(
        json.dumps(
            {
                "format": "gfpr_rollout_v1",
                "collection_complete": True,
            }
        )
    )
    expected_lse = torch.logsumexp(logits.float(), dim=-1)
    return rollout, target, expected_lse


def make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "rollout": None,
        "target": None,
        "split": "validation_select",
        "mode": "materialize",
        "output": None,
        "sidecar": None,
        "max_records": 0,
        "shard_size": 1,
        "lse_atol": 1e-5,
        "lse_rtol": 1e-6,
        "require_cuda": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_materialize_then_verify_exact_sidecar(tmp_path: Path) -> None:
    rollout, target, expected_lse = make_fixture(tmp_path)
    sidecar = tmp_path / "sidecar"
    report = run(
        make_args(rollout=rollout, target=target, output=sidecar)
    )
    assert report["collection_complete"] is True
    assert report["records"] == 1
    metadata, values = load_lse_sidecar(sidecar)
    assert metadata["geometry"] == "batch1_full16_bf16_f_linear_then_fp32_logsumexp"
    value = next(iter(values.values()))
    assert torch.equal(value, expected_lse)
    replay = run(
        make_args(
            rollout=rollout,
            target=target,
            mode="verify",
            sidecar=sidecar,
        )
    )
    assert replay["verified"] is True
    assert replay["max_lse_abs_error"] == 0.0
    assert replay["five_scalar_channels_allclose"] is True
    assert replay["max_scalar_abs_error"] == 0.0
    assert replay["audit_head_scores_allclose"] is True
    assert replay["max_audit_head_score_abs_error"] == 0.0
    assert replay["selected_tokens_exact"] is True
    assert replay["selected_token_mismatches"] == 0


def test_materializer_fails_closed_on_top16_geometry_mismatch(tmp_path: Path) -> None:
    rollout, target, _ = make_fixture(tmp_path)
    records = torch.load(
        rollout / "shard-00000.pt", map_location="cpu", weights_only=False
    )
    records[0]["base_topk_ids"][0, 0] = 999
    torch.save(records, rollout / "shard-00000.pt")
    try:
        run(
            make_args(
                rollout=rollout,
                target=target,
                output=tmp_path / "bad-sidecar",
            )
        )
    except RuntimeError as error:
        assert "Top-16 ID replay mismatch" in str(error)
    else:
        raise AssertionError("geometry mismatch did not fail closed")
