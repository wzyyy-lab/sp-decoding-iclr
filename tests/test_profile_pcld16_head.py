from __future__ import annotations

from pathlib import Path

import pytest
import torch

from profile_pcld16_head import build_pcld
from sph.pcld import EXPECTED_PARAMETER_COUNT, PCLD16Head


def test_profile_builds_exact_frozen_global_head() -> None:
    model = build_pcld(None)
    assert model.scope == "global"
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETER_COUNT


def test_profile_rejects_local_checkpoint(tmp_path: Path) -> None:
    model = PCLD16Head(scope="local")
    checkpoint = tmp_path / "local.pt"
    torch.save(
        {
            "format": "pcld16_checkpoint_v1",
            "model": model.state_dict(),
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "config": {"scope": "local"},
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="local"):
        build_pcld(checkpoint)

