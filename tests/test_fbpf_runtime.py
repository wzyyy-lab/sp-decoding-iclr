from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sph.fbpf import LoRALinear, TransactionState, TransactionalAdamW, dpace_loss
from sph.fbpf_runtime import (
    FunctionalDFlashBatch,
    FunctionalDFlashForward,
    copy_flat_lora_,
    engineering_gold_with_protected_prefix,
    evaluate_flat_transaction,
    flatten_current_lora,
)


class TinyFunctionalDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(17)
        base = nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            base.weight.copy_(torch.eye(6))
        self.proj = LoRALinear(base, rank=2, alpha=2, generator=generator)

    def forward(
        self,
        *,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        del position_ids
        return self.proj(noise_embedding + target_hidden[:, :1])


def make_runtime() -> tuple[
    TinyFunctionalDraft, FunctionalDFlashBatch, FunctionalDFlashForward, torch.Tensor
]:
    draft = TinyFunctionalDraft()
    target_weight = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ]
    )
    noise = torch.zeros(4, 4, 6)
    for row in range(4):
        for position in range(4):
            noise[row, position, (position + row) % 6] = 2.0
    target_hidden = torch.zeros(4, 1, 6)
    position_ids = torch.arange(5).repeat(4, 1)
    layout, theta = flatten_current_lora(draft)
    provisional = FunctionalDFlashBatch(
        target_hidden=target_hidden,
        noise_embedding=noise,
        position_ids=position_ids,
        gold=torch.zeros(4, 3, dtype=torch.long),
        base_logits=torch.zeros(4, 3, 7),
    )
    provisional_forward = FunctionalDFlashForward(
        draft=draft,
        target_weight=target_weight,
        batch=provisional,
        layout=layout,
    )
    with torch.no_grad():
        base_logits = provisional_forward(theta)
    gold = engineering_gold_with_protected_prefix(base_logits, mismatch_position=1)
    batch = FunctionalDFlashBatch(
        target_hidden=target_hidden,
        noise_embedding=noise,
        position_ids=position_ids,
        gold=gold,
        base_logits=base_logits,
    )
    forward = FunctionalDFlashForward(
        draft=draft,
        target_weight=target_weight,
        batch=batch,
        layout=layout,
    )
    return draft, batch, forward, theta


def test_functional_zero_adapter_matches_live_model() -> None:
    draft, batch, forward, theta = make_runtime()
    with torch.no_grad():
        hidden = draft(
            target_hidden=batch.target_hidden,
            noise_embedding=batch.noise_embedding,
            position_ids=batch.position_ids,
        )[:, -3:]
        expected = F.linear(hidden, forward.target_weight).float()
        actual = forward(theta)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_real_model_linearization_contract_geometry() -> None:
    _, batch, forward, theta = make_runtime()
    evaluation = evaluate_flat_transaction(
        theta=theta,
        forward_logits=forward,
        batch=batch,
        arm="D",
        need_task=True,
        need_vjp=True,
    )
    assert evaluation.row_ids == (0, 1, 2, 3)
    assert evaluation.constraint_values.shape == (4,)
    assert evaluation.constraint_gradients is not None
    assert evaluation.constraint_gradients.shape == (4, theta.numel())
    assert evaluation.task_gradient is not None
    assert evaluation.task_gradient.shape == theta.shape
    assert evaluation.feasible()


def test_functional_flat_lora_dpace_gradient_matches_manual_reference() -> None:
    _, batch, forward, initial = make_runtime()

    reference_theta = initial.detach().clone().requires_grad_(True)
    reference_logits = forward(reference_theta)
    per_token = F.cross_entropy(
        reference_logits.reshape(-1, reference_logits.shape[-1]),
        batch.gold.reshape(-1),
        reduction="none",
    ).view_as(batch.gold)
    with torch.no_grad():
        smooth = 0.5 * torch.exp(-per_token) + 0.5
        prefix = torch.cumprod(smooth, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix, dims=(-1,)), dim=-1), dims=(-1,)
        )
    reference_loss = (per_token * weights).sum() / 4.0
    reference_gradient = torch.autograd.grad(reference_loss, reference_theta)[0]

    ours_theta = initial.detach().clone().requires_grad_(True)
    ours_logits = forward(ours_theta)
    ours_loss = dpace_loss(ours_logits, batch.gold, reduction_divisor=4)
    ours_gradient = torch.autograd.grad(ours_loss, ours_theta)[0]

    torch.testing.assert_close(ours_loss, reference_loss, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        ours_gradient, reference_gradient, atol=5e-7, rtol=1e-5
    )
    assert torch.count_nonzero(ours_gradient) > 0


def test_transaction_and_copy_flat_lora() -> None:
    draft, batch, forward, theta = make_runtime()
    engine = TransactionalAdamW(candidate_alphas=(1.0, 0.5, 0.25))

    def evaluate(
        candidate: torch.Tensor, need_task: bool, need_vjp: bool
    ):
        return evaluate_flat_transaction(
            theta=candidate,
            forward_logits=forward,
            batch=batch,
            arm="D",
            need_task=need_task,
            need_vjp=need_vjp,
        )

    result = engine.step(
        TransactionState.initialize(theta), evaluate, learning_rate=1e-4
    )
    assert not result.aborted
    layout, _ = flatten_current_lora(draft)
    copy_flat_lora_(draft, layout, result.state.theta)
    _, copied = flatten_current_lora(draft)
    torch.testing.assert_close(copied, result.state.theta, atol=0, rtol=0)
