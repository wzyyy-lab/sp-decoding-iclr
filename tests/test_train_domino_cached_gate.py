from __future__ import annotations

from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_domino_cached_gate import AdaptiveCorrectionGate  # noqa: E402


def test_position_gate_initializes_to_unit_scale_and_has_gradient() -> None:
    gate = AdaptiveCorrectionGate(
        gate_type="position",
        positions=4,
        hidden_size=8,
        state_size=3,
        width=5,
    )
    hidden = torch.randn(2, 4, 8)
    state = torch.randn(2, 4, 3)
    scales = gate(hidden, state, torch.arange(4))
    assert torch.equal(scales, torch.ones_like(scales))
    scales.sum().backward()
    assert gate.position_logits.grad is not None


def test_state_position_gate_initializes_to_unit_scale() -> None:
    gate = AdaptiveCorrectionGate(
        gate_type="state_position",
        positions=4,
        hidden_size=8,
        state_size=3,
        width=5,
    )
    scales = gate(torch.randn(2, 4, 8), torch.randn(2, 4, 3), torch.arange(4))
    assert torch.equal(scales, torch.ones_like(scales))
