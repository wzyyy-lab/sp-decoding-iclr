from __future__ import annotations

from pathlib import Path
import sys

import torch
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_domino_global_refiner import (  # noqa: E402
    GlobalCausalRefiner,
    GlobalDirectCandidateSelector,
    apply_candidate_residual,
    all_position_breaker_loss,
    direct_selector_logits,
)


def test_refiner_starts_as_exact_zero_residual_and_trains_output() -> None:
    refiner = GlobalCausalRefiner(hidden_size=8, rank=4, positions=5)
    output = refiner(torch.randn(2, 5, 8), torch.randn(2, 5, 8))
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert refiner.residual_out.weight.grad is not None


def test_all_position_breaker_can_repair_position_zero() -> None:
    logits = torch.zeros((2, 4, 5), requires_grad=True)
    gold = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]])
    loss, _ = all_position_breaker_loss(
        logits=logits,
        gold=gold,
        objective="breaker",
        prefix_weight=0.1,
        margin_temperature=1.0,
        margin_offset=0.0,
    )
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 1] < 0
    assert torch.equal(logits.grad[0, 1:], torch.zeros_like(logits.grad[0, 1:]))


def test_decay_ce_trains_all_positions_with_early_emphasis() -> None:
    logits = torch.zeros((1, 4, 5), requires_grad=True)
    gold = torch.tensor([[1, 2, 3, 4]])
    loss, diagnostics = all_position_breaker_loss(
        logits=logits,
        gold=gold,
        objective="decay_ce",
        prefix_weight=0.1,
        margin_temperature=1.0,
        margin_offset=0.0,
        loss_decay_gamma=2.0,
    )
    loss.backward()
    assert torch.count_nonzero(logits.grad) > 0
    assert diagnostics["weight_sum"] > 1.0


def test_candidate_dpace_censors_after_first_missing_candidate() -> None:
    logits = torch.zeros((1, 4, 5), requires_grad=True)
    gold = torch.tensor([[1, 2, 3, 4]])
    coverage = torch.tensor([[True, True, False, True]])
    loss, _ = all_position_breaker_loss(
        logits=logits,
        gold=gold,
        objective="candidate_dpace",
        prefix_weight=0.1,
        margin_temperature=1.0,
        margin_offset=0.0,
        dpace_alpha=0.5,
        trainable_mask=coverage,
    )
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :2]) > 0
    assert torch.equal(logits.grad[:, 2:], torch.zeros_like(logits.grad[:, 2:]))


def test_candidate_residual_changes_only_base_or_released_topk() -> None:
    fixed = torch.tensor([[[9.0, 8.0, 1.0, 0.0]]])
    base = torch.tensor([[[9.0, 1.0, 8.0, 0.0]]])
    delta = torch.tensor([[[0.5, 0.5, 0.5, 100.0]]])
    corrected, mask = apply_candidate_residual(
        fixed_logits=fixed,
        base_logits=base,
        delta=delta,
        candidate_topk=2,
    )
    assert mask is not None
    assert mask.sum() == 3
    assert corrected[0, 0, 3] < fixed.min()
    assert torch.equal(corrected[0, 0, :3], fixed[0, 0, :3] + 0.5)
    assert corrected.argmax(dim=-1).item() in {0, 1, 2}


def test_direct_selector_identity_reranks_exact_domino_topk() -> None:
    selector = GlobalDirectCandidateSelector(
        hidden_size=4,
        max_positions=2,
        max_candidates=2,
        model_dim=4,
        num_heads=1,
        num_layers=1,
        scope="global",
        mixer="axial",
        node_encoder="compatibility",
    )
    fixed = torch.tensor([[[9.0, 8.0, 1.0, 0.0], [2.0, 7.0, 6.0, 1.0]]])
    corrected, mask = direct_selector_logits(
        refiner=selector,
        target_weight=torch.randn(4, 4),
        anchors=torch.tensor([0]),
        hidden=torch.randn(1, 2, 4),
        fixed_logits=fixed,
        base_logits=fixed,
        candidate_topk=2,
        candidate_source="released_topk",
    )
    assert mask.sum().item() == 4
    assert torch.equal(corrected.argmax(dim=-1), fixed.argmax(dim=-1))
    assert corrected[0, 0, 2] < corrected[0, 0, :2].min()
    assert corrected[0, 1, 0] < corrected[0, 1, 1:3].min()


def test_direct_selector_uses_base_topk_but_never_drops_released_token() -> None:
    class CapturingIdentitySelector(torch.nn.Module):
        def forward(self, **kwargs):
            self.candidate_logits = kwargs["candidate_logits"].detach().clone()
            self.score_candidate_logits = kwargs[
                "score_candidate_logits"
            ].detach().clone()
            scores = (
                kwargs["score_candidate_logits"].float()
                - kwargs["score_logsumexp"].unsqueeze(-1)
            )
            return SimpleNamespace(scores=scores)

    selector = CapturingIdentitySelector()
    fixed = torch.tensor([[[1.0, 2.0, 3.0, 9.0]]])
    base = torch.tensor([[[8.0, 7.0, 1.0, 0.0]]])
    corrected, mask = direct_selector_logits(
        refiner=selector,
        target_weight=torch.randn(4, 4),
        anchors=torch.tensor([0]),
        hidden=torch.randn(1, 1, 4),
        fixed_logits=fixed,
        base_logits=base,
        candidate_topk=2,
        candidate_source="base_topk_plus_released",
    )
    assert mask[0, 0, 0]
    assert mask[0, 0, 3]
    assert mask.sum().item() == 2
    assert corrected.argmax(dim=-1).item() == 3
    assert selector.candidate_logits[0, 0].tolist() == [8.0, 0.0]
    assert selector.score_candidate_logits[0, 0].tolist() == [1.0, 9.0]
