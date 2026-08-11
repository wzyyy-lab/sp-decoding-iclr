from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_domino_hidden_adapter import (  # noqa: E402
    GlobalFinalHiddenAdapter,
    onpolicy_ids,
    teacher_logits,
)


def test_zero_initialized_hidden_adapter_is_exact_identity() -> None:
    adapter = GlobalFinalHiddenAdapter(
        hidden_size=12,
        positions=5,
        model_dim=8,
        num_heads=2,
        num_layers=1,
    )
    hidden = torch.randn(3, 5, 12)
    adapted, delta = adapter(hidden)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(adapted, hidden.float())


def test_hidden_adapter_rejects_wrong_block_length() -> None:
    adapter = GlobalFinalHiddenAdapter(
        hidden_size=12,
        positions=5,
        model_dim=8,
        num_heads=2,
        num_layers=1,
    )
    with pytest.raises(ValueError, match="block length"):
        adapter(torch.randn(2, 4, 12))


def test_hidden_adapter_training_path_reaches_zero_initialized_output() -> None:
    torch.manual_seed(4)
    hidden_size = 12
    state_size = 6
    vocabulary = 19
    positions = 5
    adapter = GlobalFinalHiddenAdapter(
        hidden_size=hidden_size,
        positions=positions,
        model_dim=8,
        num_heads=2,
        num_layers=1,
    )

    class FakeDomino(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prefix_gru = torch.nn.GRU(
                hidden_size, state_size, batch_first=True
            )
            self.embed_proj = torch.nn.Linear(
                hidden_size + state_size, vocabulary, bias=False
            )

    domino = FakeDomino()
    for parameter in domino.parameters():
        parameter.requires_grad_(False)
    target_weight = torch.randn(vocabulary, hidden_size)
    hidden = torch.randn(2, positions, hidden_size)
    anchors = torch.tensor([1, 2])
    gold = torch.randint(0, vocabulary, (2, positions))
    logits, delta = teacher_logits(
        domino=domino,
        adapter=adapter,
        target_weight=target_weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
    )
    assert logits.shape == (2, positions, vocabulary)
    assert torch.count_nonzero(delta) == 0
    torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), gold.flatten()
    ).backward()
    assert adapter.output_projection.weight.grad is not None
    assert torch.count_nonzero(adapter.output_projection.weight.grad) > 0
    base_only = onpolicy_ids(
        domino=domino,
        adapter=adapter,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        application="base_only",
    )
    joint = onpolicy_ids(
        domino=domino,
        adapter=adapter,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        application="joint",
    )
    assert base_only.shape == gold.shape
    assert torch.equal(base_only, joint)
