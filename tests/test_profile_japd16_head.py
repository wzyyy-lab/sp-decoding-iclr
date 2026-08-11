from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from profile_japd16_head import build_japd


def profile_args(
    *, model_dim: int, num_heads: int, num_layers: int
) -> Namespace:
    return Namespace(
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        seed=0,
    )


def test_profile_builds_exact_frozen_d64_and_d256_sizes() -> None:
    d64 = build_japd(
        None, profile_args(model_dim=64, num_heads=4, num_layers=1)
    )
    d256 = build_japd(
        None, profile_args(model_dim=256, num_heads=8, num_layers=2)
    )
    assert sum(parameter.numel() for parameter in d64.parameters()) == 433_852
    assert sum(parameter.numel() for parameter in d256.parameters()) == 4_539_888


def test_profile_checkpoint_architecture_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    d64_args = profile_args(model_dim=64, num_heads=4, num_layers=1)
    d64 = build_japd(None, d64_args)
    checkpoint = tmp_path / "d64.pt"
    torch.save(
        {
            "config": {
                "scope": "global",
                "model_dim": 64,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": 0.1,
                "seed": 0,
            },
            "model": d64.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="model_dim"):
        build_japd(
            checkpoint,
            profile_args(model_dim=256, num_heads=8, num_layers=2),
        )
