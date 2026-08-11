import copy

import torch
from torch import nn
from torch.func import functional_call

from sph.fbpf import (
    FBPF_EXPECTED_TRAINABLE_PARAMETERS,
    LoRALinear,
    count_lora_parameters,
    expected_dflash_lora_parameter_count,
    expected_lora_module_paths,
    inject_fbpf_lora,
    iter_lora_modules,
    lora_disabled,
    lora_parameter_hashes,
    merge_fbpf_lora_,
    named_lora_parameters,
)


class _MockAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)


class _MockMlp(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, width * 2, bias=False)
        self.up_proj = nn.Linear(width, width * 2, bias=False)
        self.down_proj = nn.Linear(width * 2, width, bias=False)


class _MockLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = _MockAttention(width)
        self.mlp = _MockMlp(width)


class _MockDraft(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(width) for _ in range(5)])


def test_frozen_configuration_count_is_exact() -> None:
    assert expected_dflash_lora_parameter_count() == 1_835_008
    assert expected_dflash_lora_parameter_count() == FBPF_EXPECTED_TRAINABLE_PARAMETERS
    assert len(expected_lora_module_paths()) == 14
    assert expected_lora_module_paths() == tuple(sorted(expected_lora_module_paths()))


def test_injection_is_seeded_sorted_and_does_not_advance_global_rng() -> None:
    torch.manual_seed(17)
    original = _MockDraft()
    left = copy.deepcopy(original)
    right = copy.deepcopy(original)

    rng_before = torch.random.get_rng_state().clone()
    paths = inject_fbpf_lora(left, training_seed=2)
    rng_after = torch.random.get_rng_state().clone()
    inject_fbpf_lora(right, training_seed=2)

    assert torch.equal(rng_before, rng_after)
    assert paths == expected_lora_module_paths()
    assert lora_parameter_hashes(left) == lora_parameter_hashes(right)
    assert all(module.lora_A.dtype == torch.float32 for _, module in iter_lora_modules(left))
    assert all(module.lora_B.dtype == torch.float32 for _, module in iter_lora_modules(left))
    assert all(torch.count_nonzero(module.lora_B) == 0 for _, module in iter_lora_modules(left))
    assert count_lora_parameters(left) == sum(
        parameter.numel() for _, parameter in named_lora_parameters(left)
    )


def test_zero_adapter_disable_and_bf16_merge_preserve_outputs() -> None:
    torch.manual_seed(23)
    root = nn.Sequential(nn.Linear(4, 5, bias=True).to(dtype=torch.bfloat16))
    inputs = torch.randn(7, 4, dtype=torch.bfloat16)
    released = root(inputs)
    inject_fbpf_lora(root, training_seed=0, module_paths=("0",), rank=2, alpha=2)

    enabled_zero = root(inputs)
    with lora_disabled(root):
        disabled = root(inputs)
    assert torch.equal(enabled_zero, released)
    assert torch.equal(disabled, released)

    wrapper = root[0]
    assert isinstance(wrapper, LoRALinear)
    with torch.no_grad():
        wrapper.lora_B.fill_(0.01)
    adapter_output = root(inputs)
    adapter_argmax = adapter_output.float().argmax(dim=-1)
    merged_paths = merge_fbpf_lora_(root)
    merged_output = root(inputs)

    assert merged_paths == ("0",)
    assert isinstance(root[0], nn.Linear)
    assert root[0].weight.dtype == torch.bfloat16
    torch.testing.assert_close(merged_output, adapter_output, atol=0.02, rtol=0.02)
    assert torch.equal(merged_output.float().argmax(dim=-1), adapter_argmax)
    assert not tuple(iter_lora_modules(root))


def test_partial_functional_call_overrides_only_lora_parameters() -> None:
    torch.manual_seed(29)
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    inject_fbpf_lora(model, training_seed=1, module_paths=("0",), rank=2, alpha=2)
    inputs = torch.randn(4, 3)
    zero_output = model(inputs)
    wrapper = model[0]
    assert isinstance(wrapper, LoRALinear)

    override_b = torch.full_like(wrapper.lora_B, 0.25, requires_grad=True)
    changed = functional_call(
        model,
        {"0.lora_B": override_b},
        (inputs,),
        strict=False,
    )
    assert not torch.equal(changed, zero_output)
    changed.sum().backward()
    assert override_b.grad is not None
    assert wrapper.lora_B.grad is None
