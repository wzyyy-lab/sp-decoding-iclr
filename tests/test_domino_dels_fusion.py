from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_domino_dels_fusion import parse_configs  # noqa: E402


def test_parse_fusion_configs_requires_released_domino() -> None:
    configs = parse_configs(
        ["domino:1:0:0", "joint:0.75:0.3:0.2"]
    )
    assert configs == [
        {"name": "domino", "gamma": 1.0, "alpha": 0.0, "beta": 0.0},
        {"name": "joint", "gamma": 0.75, "alpha": 0.3, "beta": 0.2},
    ]
