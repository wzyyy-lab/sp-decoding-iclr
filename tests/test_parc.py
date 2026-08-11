from __future__ import annotations

import copy

import pytest
import torch

from sph.parc import (
    PARC16Head,
    conditional_gain_loss,
    nonshift_full16_prediction_hidden,
    parc_fixed_reference_loss,
)


def inputs(hidden_size: int = 8) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, 16, hidden_size, generator=generator)
    logits = torch.randn(2, 16, 16, generator=generator)
    anchor = torch.randn(2, hidden_size, generator=generator)
    candidates = torch.randn(2, 16, 16, hidden_size, generator=generator)
    ids = torch.arange(16).view(1, 1, 16).expand(2, 16, 16).clone()
    return hidden, logits, anchor, candidates, ids


def test_one_call_full16_identity_and_rank0_gauge() -> None:
    model = PARC16Head(
        hidden_size=8, model_dim=16, num_heads=4, num_layers=1
    )
    hidden, logits, anchor, candidates, ids = inputs()
    output = model(hidden, logits, anchor, candidates)
    assert output.scores.shape == (2, 16, 16)
    assert model.proposal_ids(ids, output).shape == (2, 16)
    assert torch.equal(output.scores.argmax(-1), logits.argmax(-1))
    assert torch.equal(output.scores[..., 0], torch.zeros_like(output.scores[..., 0]))


def test_nonshift_full16_drops_anchor_carrier_and_keeps_rows_one_to_sixteen() -> None:
    raw = torch.arange(17).view(1, 17, 1).expand(2, 17, 3)
    prediction = nonshift_full16_prediction_hidden(raw)
    assert prediction.shape == (2, 16, 3)
    assert torch.equal(prediction[0, :, 0], torch.arange(1, 17))
    with pytest.raises(ValueError, match="17"):
        nonshift_full16_prediction_hidden(raw[:, :16])


def test_global_head_has_cross_position_gradient_but_local_control_does_not() -> None:
    global_model = PARC16Head(
        hidden_size=8, model_dim=16, num_heads=4, num_layers=1
    )
    local_model = PARC16Head(
        hidden_size=8,
        model_dim=16,
        num_heads=4,
        num_layers=1,
        local_control=True,
    )
    local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
    with torch.no_grad():
        global_model.residual_projection.weight.zero_()
        local_model.residual_projection.weight.zero_()
        global_model.residual_projection.weight[0, 0] = 0.1
        local_model.residual_projection.weight[0, 0] = 0.1
    hidden, logits, anchor, candidates, _ = inputs()
    hidden_global = hidden.clone().requires_grad_(True)
    hidden_local = hidden.clone().requires_grad_(True)
    global_model(hidden_global, logits, anchor, candidates).scores[0, 0, 1].backward()
    local_model(hidden_local, logits, anchor, candidates).scores[0, 0, 1].backward()
    assert float(hidden_global.grad[0, 15].abs().sum()) > 0.0
    assert torch.equal(hidden_local.grad[0, 15], torch.zeros_like(hidden_local.grad[0, 15]))


def test_detached_gain_loss_matches_direct_gain_gradient() -> None:
    scores_a = torch.randn(3, 16, 16, generator=torch.Generator().manual_seed(9), requires_grad=True)
    ranks = torch.zeros(3, 16, dtype=torch.long)
    ranks[1, 7] = -1
    accepted = torch.tensor([0, 3, 16])
    loss, gain, _ = conditional_gain_loss(scores_a, ranks, accepted)
    grad_a = torch.autograd.grad(loss, scores_a)[0]

    scores_b = scores_a.detach().clone().requires_grad_(True)
    _, direct_gain, _ = conditional_gain_loss(scores_b, ranks, accepted)
    direct = -direct_gain.mean() / 16.0
    grad_b = torch.autograd.grad(direct, scores_b)[0]
    assert torch.allclose(gain.detach(), direct_gain.detach())
    assert torch.allclose(grad_a, grad_b, atol=1e-7, rtol=1e-6)


def test_harm_upper_bound_covers_deterministic_harm_and_support_drop() -> None:
    model = PARC16Head(hidden_size=8, model_dim=16, num_heads=4, num_layers=1)
    hidden, logits, anchor, candidates, ids = inputs()
    gold = torch.zeros(2, 16, dtype=torch.long)
    accepted = torch.tensor([4, 4])
    delta = torch.tensor([2.0, 2.0])
    output = model(hidden, logits, anchor, candidates)
    result = parc_fixed_reference_loss(
        output, ids, gold, accepted, delta, delta_min=0.03125
    )
    assert torch.all(result.actual_harm <= result.harm_upper_bound)

    dropped = ids.clone()
    dropped[0, 2] += 100
    dropped[0, 2, 0] = 99
    result_drop = parc_fixed_reference_loss(
        output, dropped, gold, accepted, delta, delta_min=0.03125
    )
    assert bool(result_drop.support_drop[0])
    assert float(result_drop.harm_upper_bound[0].detach()) == 1.0
    assert not bool(result_drop.gain_positions[0].any())
