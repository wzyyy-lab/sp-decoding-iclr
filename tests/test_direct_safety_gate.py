from __future__ import annotations

from copy import deepcopy
import math

import pytest
import torch
from torch.nn import functional as F

from sph.direct_safety_gate import (
    BLOCK_LENGTH,
    CANDIDATE_COUNT,
    SCALAR_COMPARATOR_DIMENSION,
    SIDECAR_PARAMETER_COUNT,
    DirectSafetySidecar,
    binary_outcomes_from_tokens,
    capacity_gain_weighted_unit_hinge,
    direct_safety_position_features,
    freeze_direct_producer,
    frozen_direct_forward_with_states,
    gain_weighted_unit_hinge,
    normalized_signed_gain,
    prompt_balanced_example_weights,
    prompt_balanced_mean,
    scalar_comparator_features,
    validated_oracle_recovery,
)
from sph.global_direct_selector import (
    GlobalDirectCandidateSelector,
    GlobalDirectOutput,
)


def make_feature_fixture() -> tuple[
    torch.Tensor,
    GlobalDirectOutput,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(20260805)
    batch = 3
    node_states = torch.randn(
        batch,
        BLOCK_LENGTH,
        CANDIDATE_COUNT,
        64,
        generator=generator,
    )
    one_position = -torch.arange(CANDIDATE_COUNT, dtype=torch.float32)
    logits = one_position.expand(batch, BLOCK_LENGTH, -1).clone()
    base_logsumexp = torch.logsumexp(logits, dim=-1) + 0.75
    base_log_probs = logits - base_logsumexp.unsqueeze(-1)
    residual = torch.zeros_like(base_log_probs)
    residual[1, 2, 3] = 20.0
    residual[1, 7, 1] = 20.0
    residual[2, 0, 2] = 20.0
    residual[2, 14, 15] = 30.0
    residual = residual - residual.mean(dim=-1, keepdim=True)
    scores = base_log_probs + residual
    direct = GlobalDirectOutput(
        scores=scores,
        log_probs=torch.log_softmax(scores, dim=-1),
        residual_scores=residual,
        base_log_probs=base_log_probs,
    )
    return node_states, direct, logits, base_logsumexp


def test_exact_feature_shape_paths_and_scalar_formulas() -> None:
    node_states, direct, logits, base_logsumexp = make_feature_fixture()
    result = direct_safety_position_features(
        node_states, direct, logits, base_logsumexp
    )
    assert result.position_features.shape == (3, 15, 200)
    assert result.direct_path[0].eq(0).all()
    assert result.direct_path[1, 2].item() == 3
    assert result.direct_path[1, 7].item() == 1
    assert result.direct_path[2, 0].item() == 2
    assert result.direct_path[2, 14].item() == 15
    assert torch.equal(result.change_mask, result.direct_path.ne(0))

    chosen = node_states[1, 2, 3].float()
    base = node_states[1, 2, 0].float()
    torch.testing.assert_close(result.position_features[1, 2, :64], chosen)
    torch.testing.assert_close(result.position_features[1, 2, 64:128], base)
    torch.testing.assert_close(
        result.position_features[1, 2, 128:192], chosen - base
    )
    scalars = result.position_features[1, 2, 192:]
    expected_total = direct.scores[1, 2, 3] - direct.scores[1, 2, 0]
    expected_residual = (
        direct.residual_scores[1, 2, 3]
        - direct.residual_scores[1, 2, 0]
    )
    expected_base = (
        direct.base_log_probs[1, 2, 3]
        - direct.base_log_probs[1, 2, 0]
    )
    assert scalars[0].item() == expected_total.item()
    assert scalars[1].item() == expected_residual.item()
    assert scalars[2].item() == expected_base.item()
    assert scalars[3].item() == pytest.approx(3 / 15)
    assert scalars[4].item() == pytest.approx(2 / 14)
    assert scalars[5].item() == 1.0
    q = torch.softmax(logits[1, 2], dim=-1)
    entropy = -(q * q.log()).sum() / math.log(16)
    retained = torch.tanh(
        (torch.logsumexp(logits[1, 2], dim=-1) - base_logsumexp[1, 2])
        / 2
    )
    assert scalars[6].item() == pytest.approx(entropy.item())
    assert scalars[7].item() == pytest.approx(retained.item())


def test_sidecar_parameter_count_identity_and_identical_path_zero() -> None:
    node_states, direct, logits, base_logsumexp = make_feature_fixture()
    features = direct_safety_position_features(
        node_states, direct, logits, base_logsumexp
    )
    sidecar = DirectSafetySidecar(initialization_seed=0)
    assert sum(value.numel() for value in sidecar.parameters()) == (
        SIDECAR_PARAMETER_COUNT
    )
    assert sidecar.position_norm.eps == 1e-5
    assert sidecar.position_norm.elementwise_affine
    assert sidecar.position_input.bias is not None
    assert sidecar.final_projection.bias is None
    assert torch.equal(
        sidecar.final_projection.weight,
        torch.zeros_like(sidecar.final_projection.weight),
    )
    output = sidecar(
        features.position_features,
        features.change_mask,
        features.direct_path,
    )
    assert torch.equal(output.scores, torch.zeros(3))
    assert not bool(output.apply_direct.any())
    assert output.scores[0].item() == 0.0

    layer_norms = [sidecar.position_norm, sidecar.block_norm]
    assert all(module.eps == 1e-5 for module in layer_norms)
    assert all(module.elementwise_affine for module in layer_norms)
    assert sidecar.position_norm.normalized_shape == (200,)
    assert sidecar.block_norm.normalized_shape == (257,)
    linears = [
        sidecar.position_input,
        sidecar.position_output,
        sidecar.block_input,
        sidecar.block_output,
        sidecar.final_projection,
    ]
    assert all(module.weight.dtype == torch.float32 for module in linears)
    assert all(module.bias is not None for module in linears[:-1])
    assert linears[-1].bias is None
    assert [(module.in_features, module.out_features) for module in linears] == [
        (200, 64),
        (64, 64),
        (257, 64),
        (64, 64),
        (64, 1),
    ]


def _manual_sidecar_pool(
    sidecar: DirectSafetySidecar,
    position_features: torch.Tensor,
    change_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = position_features.detach().float()
    encoded = sidecar.position_output(
        F.silu(sidecar.position_input(sidecar.position_norm(features)))
    )
    change_float = change_mask.float()
    change_count = change_float.sum(dim=1)
    changed_mean = (
        encoded * change_float.unsqueeze(-1)
    ).sum(dim=1) / change_count.clamp_min(1.0).unsqueeze(-1)
    masked = encoded.masked_fill(
        ~change_mask.unsqueeze(-1), torch.finfo(encoded.dtype).min
    )
    changed_max = masked.amax(dim=1)
    changed_max = torch.where(
        change_count.gt(0).unsqueeze(-1),
        changed_max,
        torch.zeros_like(changed_max),
    )
    pooled = torch.cat(
        [
            encoded.mean(dim=1),
            changed_mean,
            changed_max,
            encoded[:, 0],
            (change_count / 15.0).unsqueeze(-1),
        ],
        dim=-1,
    )
    block_state = sidecar.block_output(
        F.silu(sidecar.block_input(sidecar.block_norm(pooled)))
    )
    return pooled, block_state


def test_exact_pools_nonzero_decoder_and_identical_path_mask() -> None:
    node_states, direct, logits, base_logsumexp = make_feature_fixture()
    features = direct_safety_position_features(
        node_states, direct, logits, base_logsumexp
    )
    sidecar = DirectSafetySidecar(initialization_seed=0)
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().clone())

    handle = sidecar.block_norm.register_forward_pre_hook(capture)
    try:
        initial = sidecar(
            features.position_features,
            features.change_mask,
            features.direct_path,
        )
    finally:
        handle.remove()
    expected_pool, block_state = _manual_sidecar_pool(
        sidecar, features.position_features, features.change_mask
    )
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected_pool, rtol=0, atol=0)
    assert captured[0].dtype == torch.float32
    assert block_state.dtype == torch.float32
    assert initial.scores.dtype == torch.float32
    assert torch.equal(expected_pool[0, 64:192], torch.zeros(128))

    direction = block_state[1] / block_state[1].square().sum()
    with torch.no_grad():
        sidecar.final_projection.weight.copy_(direction.unsqueeze(0))
    positive = sidecar(
        features.position_features,
        features.change_mask,
        features.direct_path,
    )
    assert positive.scores[1].item() == pytest.approx(1.0, abs=1e-5)
    assert positive.apply_direct[1].item() is True
    assert positive.scores[0].item() == 0.0
    assert positive.apply_direct[0].item() is False

    with torch.no_grad():
        sidecar.final_projection.weight.copy_(-direction.unsqueeze(0))
    negative = sidecar(
        features.position_features,
        features.change_mask,
        features.direct_path,
    )
    assert negative.scores[1].item() == pytest.approx(-1.0, abs=1e-5)
    assert negative.apply_direct[1].item() is False

    with torch.no_grad():
        sidecar.final_projection.weight.zero_()
    tied = sidecar(
        features.position_features,
        features.change_mask,
        features.direct_path,
    )
    assert tied.scores[1].item() == 0.0
    assert tied.apply_direct[1].item() is False


def test_scalar_comparator_exact_21_feature_golden_vectors() -> None:
    values = torch.zeros(2, 15, 200, dtype=torch.float64, requires_grad=True)
    change = torch.zeros(2, 15, dtype=torch.bool)
    change[0, [0, 2]] = True
    with torch.no_grad():
        values[0, 0, 192:195] = torch.tensor([2.0, 4.0, -1.0])
        values[0, 2, 192:195] = torch.tensor([6.0, -2.0, 3.0])
        values[0, change[0], 197] = 1.0
        values[0, :, 198] = torch.arange(15) / 14.0
        values[0, :, 199] = torch.arange(-7, 8)
        values[1, 0, 192] = 11.0
        values[1, :, 198] = 0.25
        values[1, :, 199] = -0.5
    result = scalar_comparator_features(values, change)
    assert result.shape == (2, SCALAR_COMPARATOR_DIMENSION)
    assert result.dtype == torch.float32
    assert not result.requires_grad
    expected_first = torch.tensor(
        [
            2 / 15,
            8,
            4,
            2,
            6,
            2,
            1,
            -2,
            4,
            2,
            1,
            -1,
            3,
            0.5,
            0,
            1,
            0,
            -7,
            7,
            1,
            2,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result[0], expected_first)
    expected_empty = torch.tensor(
        [0] * 13 + [0.25, 0.25, 0.25, -0.5, -0.5, -0.5, 0, 11],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result[1], expected_empty)
    with pytest.raises(ValueError, match="15x200"):
        scalar_comparator_features(values[:, :, :-1], change)
    with pytest.raises(ValueError, match="boolean"):
        scalar_comparator_features(values, change.float())
    inconsistent = change.clone()
    inconsistent[0, 0] = False
    with pytest.raises(ValueError, match="differs"):
        scalar_comparator_features(values, inconsistent)


def test_feature_boundary_detaches_every_direct_source_tensor() -> None:
    node_states = torch.randn(
        1, 15, 16, 64, dtype=torch.float64, requires_grad=True
    )
    logits = torch.randn(1, 15, 16, requires_grad=True)
    full_lse = (
        torch.logsumexp(logits.detach(), dim=-1) + 0.5
    ).requires_grad_()
    base = logits - full_lse.unsqueeze(-1)
    residual_source = torch.randn(1, 15, 16, requires_grad=True)
    residual = residual_source - residual_source.mean(dim=-1, keepdim=True)
    scores = base + residual
    direct = GlobalDirectOutput(
        scores=scores,
        log_probs=torch.log_softmax(scores, dim=-1),
        residual_scores=residual,
        base_log_probs=base,
    )
    sidecar = DirectSafetySidecar(initialization_seed=0)
    with torch.no_grad():
        sidecar.final_projection.weight.fill_(0.01)
    output = sidecar.forward_from_direct(
        node_states, direct, logits, full_lse
    )
    output.scores.sum().backward()
    assert node_states.grad is None
    assert logits.grad is None
    assert full_lse.grad is None
    assert residual_source.grad is None


def test_named_initialization_is_reproducible_and_rng_neutral() -> None:
    torch.manual_seed(71)
    before = torch.random.get_rng_state()
    left = DirectSafetySidecar(initialization_seed=13)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    torch.manual_seed(999)
    right = DirectSafetySidecar(initialization_seed=13)
    for name, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[name])


def test_frozen_producer_state_capture_preserves_native_forward() -> None:
    generator = torch.Generator().manual_seed(83)
    producer = GlobalDirectCandidateSelector(
        hidden_size=8,
        max_positions=3,
        max_candidates=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        scope="global",
        mixer="axial",
        node_encoder="additive",
        initialization_seed=17,
    )
    inputs = (
        torch.randn(2, 3, 8, generator=generator),
        torch.randn(2, 3, 4, 8, generator=generator),
        torch.randn(2, 3, 4, generator=generator),
        torch.randn(2, 3, generator=generator),
        torch.randn(2, 8, generator=generator),
    )
    freeze_direct_producer(producer)
    frozen_state = deepcopy(producer.state_dict())
    with torch.no_grad():
        native_before = producer(*inputs)
    captured_output, states = frozen_direct_forward_with_states(producer, *inputs)
    with torch.no_grad():
        native_after = producer(*inputs)
    assert states.shape == (2, 3, 4, 8)
    assert not states.requires_grad
    for field in ("scores", "log_probs", "residual_scores", "base_log_probs"):
        assert torch.equal(getattr(native_before, field), getattr(captured_output, field))
        assert torch.equal(getattr(native_before, field), getattr(native_after, field))
    for name, value in frozen_state.items():
        assert torch.equal(value, producer.state_dict()[name])
    assert all(not value.requires_grad for value in producer.parameters())


def test_capture_rejects_trainable_or_training_producer() -> None:
    producer = GlobalDirectCandidateSelector(
        hidden_size=4,
        max_positions=2,
        max_candidates=2,
        model_dim=4,
        num_heads=1,
        num_layers=1,
        scope="local",
        mixer="axial",
        initialization_seed=3,
    )
    inputs = (
        torch.randn(1, 2, 4),
        torch.randn(1, 2, 2, 4),
        torch.randn(1, 2, 2),
        torch.randn(1, 2),
        torch.randn(1, 4),
    )
    with pytest.raises(ValueError, match="evaluation mode"):
        frozen_direct_forward_with_states(producer, *inputs)
    producer.eval()
    with pytest.raises(ValueError, match="frozen"):
        frozen_direct_forward_with_states(producer, *inputs)


def test_capture_hook_cleanup_after_producer_exception() -> None:
    class RaisingProducer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.residual_projection = torch.nn.Linear(4, 1, bias=False)

        def forward(self, value: torch.Tensor) -> GlobalDirectOutput:
            self.residual_projection(value)
            raise RuntimeError("synthetic producer failure")

    producer = freeze_direct_producer(RaisingProducer())
    observed: list[int] = []
    persistent = producer.residual_projection.register_forward_pre_hook(
        lambda _module, _inputs: observed.append(1)
    )
    before = tuple(producer.residual_projection._forward_pre_hooks)
    try:
        with pytest.raises(RuntimeError, match="synthetic producer failure"):
            frozen_direct_forward_with_states(producer, torch.ones(1, 4))
        after = tuple(producer.residual_projection._forward_pre_hooks)
        assert after == before
        assert observed == [1]
    finally:
        persistent.remove()


def test_pinned_topology_and_end_to_end_gradient_isolation() -> None:
    producer = GlobalDirectCandidateSelector(
        hidden_size=8,
        max_positions=15,
        max_candidates=16,
        model_dim=64,
        num_heads=4,
        num_layers=1,
        scope="global",
        mixer="axial",
        node_encoder="additive",
        dropout=0.0,
        initialization_seed=0,
    )
    with torch.no_grad():
        generator = torch.Generator().manual_seed(203)
        producer.residual_projection.weight.copy_(
            4.0 * torch.randn(1, 64, generator=generator)
        )
    inputs = (
        torch.randn(1, 15, 8, requires_grad=True),
        torch.randn(1, 15, 16, 8, requires_grad=True),
        torch.randn(1, 15, 16, requires_grad=True),
        torch.randn(1, 15, requires_grad=True),
        torch.randn(1, 8, requires_grad=True),
    )
    freeze_direct_producer(producer)
    direct, states = frozen_direct_forward_with_states(producer, *inputs)
    assert states.shape == (1, 15, 16, 64)
    sidecar = DirectSafetySidecar(initialization_seed=0)
    with torch.no_grad():
        sidecar.final_projection.weight.fill_(0.01)
    output = sidecar.forward_from_direct(
        states,
        direct,
        inputs[2],
        inputs[3],
    )
    output.scores.sum().backward()
    assert all(value.grad is None for value in inputs)
    assert all(value.grad is None for value in producer.parameters())
    assert any(
        value.grad is not None for value in sidecar.parameters()
    )


def test_gain_hinge_initial_gradients_neutral_and_regret_bound() -> None:
    scores = torch.zeros(3, requires_grad=True)
    gains = torch.tensor([1 / 15, -1.0, 0.0])
    result = gain_weighted_unit_hinge(scores, gains)
    torch.testing.assert_close(
        result.per_block_loss, torch.tensor([1 / 15, 1.0, 0.0])
    )
    torch.testing.assert_close(
        result.decoded_regret, torch.tensor([1 / 15, 0.0, 0.0])
    )
    assert float(result.bound_slack.detach().min()) >= -1e-6
    result.loss.backward()
    torch.testing.assert_close(
        scores.grad, torch.tensor([-1 / 45, 1 / 3, 0.0])
    )


def test_capacity_probability_masses_equal_arithmetic_mean_and_gradient() -> None:
    scores = torch.zeros(512, requires_grad=True)
    gains = torch.linspace(-1.0, 1.0, 512)
    report = capacity_gain_weighted_unit_hinge(scores, gains)
    expected_loss = gains.abs().mean()
    torch.testing.assert_close(report.loss, expected_loss)
    report.loss.backward()
    torch.testing.assert_close(scores.grad, -gains / 512.0)

    multiplier_loss = gain_weighted_unit_hinge(
        torch.zeros(512),
        gains,
        example_weights=torch.full((512,), 1.0 / 512.0),
    ).loss
    torch.testing.assert_close(multiplier_loss, expected_loss / 512.0)


def test_gain_hinge_conditional_collision_follows_expected_gain() -> None:
    positive_gains = torch.full((9,), 1 / 15)
    gains = torch.cat([positive_gains, torch.tensor([-1.0])])
    assert gains.mean().item() < 0
    negative_score = torch.full((10,), -1.0)
    positive_score = torch.full((10,), 1.0)
    negative_risk = gain_weighted_unit_hinge(negative_score, gains).loss
    positive_risk = gain_weighted_unit_hinge(positive_score, gains).loss
    assert negative_risk < positive_risk


def test_random_gain_hinge_bound_and_strict_positive_decoder() -> None:
    generator = torch.Generator().manual_seed(97)
    for _ in range(32):
        scores = torch.randn(41, generator=generator)
        gains = torch.randint(-15, 16, (41,), generator=generator).float() / 15
        result = gain_weighted_unit_hinge(scores, gains)
        assert torch.equal(result.apply_direct, scores > 0)
        assert float(result.bound_slack.detach().min()) >= -1e-6


def test_prompt_weights_and_mean_match_mean_of_prompt_means() -> None:
    sample_ids = ["a", "a", "b"]
    weights = prompt_balanced_example_weights(sample_ids)
    torch.testing.assert_close(weights, torch.tensor([0.75, 0.75, 1.5]))
    assert weights.mean().item() == 1.0
    double_weights = prompt_balanced_example_weights(
        sample_ids, dtype=torch.float64
    )
    assert double_weights.dtype == torch.float64
    torch.testing.assert_close(
        double_weights, torch.tensor([0.75, 0.75, 1.5], dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="float32 or float64"):
        prompt_balanced_example_weights(sample_ids, dtype=torch.float16)
    values = torch.tensor([1.0, 3.0, 5.0])
    assert prompt_balanced_mean(values, sample_ids).item() == 3.5


def test_first_projection_then_upstream_gradient() -> None:
    node_states, direct, logits, base_logsumexp = make_feature_fixture()
    features = direct_safety_position_features(
        node_states, direct, logits, base_logsumexp
    )
    sidecar = DirectSafetySidecar(initialization_seed=0)
    optimizer = torch.optim.SGD(sidecar.parameters(), lr=0.1)
    selected_features = features.position_features[1:]
    selected_changes = features.change_mask[1:]
    selected_paths = features.direct_path[1:]
    gains = torch.tensor([0.4, 0.7])

    first_output = sidecar(
        selected_features, selected_changes, selected_paths
    )
    first = gain_weighted_unit_hinge(first_output.scores, gains)
    first.loss.backward()
    assert sidecar.final_projection.weight.grad is not None
    assert float(sidecar.final_projection.weight.grad.norm()) > 0
    for name, value in sidecar.named_parameters():
        if name == "final_projection.weight":
            continue
        assert value.grad is None or int(torch.count_nonzero(value.grad)) == 0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_output = sidecar(
        selected_features, selected_changes, selected_paths
    )
    second = gain_weighted_unit_hinge(second_output.scores, gains)
    second.loss.backward()
    upstream = [
        value.grad
        for name, value in sidecar.named_parameters()
        if name != "final_projection.weight"
    ]
    assert any(
        value is not None and int(torch.count_nonzero(value)) > 0
        for value in upstream
    )


def test_binary_outcomes_reconstruct_benefit_harm_and_oracle() -> None:
    candidate_ids = torch.empty(2, 15, 16, dtype=torch.long)
    for batch in range(2):
        for position in range(15):
            candidate_ids[batch, position] = (
                batch * 10_000 + position * 100 + torch.arange(16)
            )
    gold = candidate_ids[:, :, 0].clone()
    gold[0, 0] = candidate_ids[0, 0, 1]
    path = torch.zeros(2, 15, dtype=torch.long)
    path[0, 0] = 1
    path[1, 2] = 1
    report = binary_outcomes_from_tokens(path, candidate_ids, gold)
    assert report.base_lengths.tolist() == [0, 15]
    assert report.direct_lengths.tolist() == [15, 2]
    torch.testing.assert_close(
        report.normalized_gains, torch.tensor([1.0, -13 / 15])
    )
    assert report.oracle_apply_direct.tolist() == [True, False]


def test_normalized_gain_and_recovery_fail_closed() -> None:
    gains = normalized_signed_gain(
        torch.tensor([0, 15]), torch.tensor([15, 2])
    )
    torch.testing.assert_close(gains, torch.tensor([1.0, -13 / 15]))
    assert validated_oracle_recovery(3.0, 1.0, 3.0) == 1.0
    with pytest.raises(ValueError, match="positive"):
        validated_oracle_recovery(1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="outside"):
        validated_oracle_recovery(3.1, 1.0, 3.0)
    with pytest.raises(ValueError, match="finite"):
        validated_oracle_recovery(float("nan"), 1.0, 3.0)
