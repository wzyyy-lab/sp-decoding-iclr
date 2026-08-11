from __future__ import annotations

import torch

from sph.r048_layer_split import causal_additive_mask


def test_cached_causal_mask_is_lower_right_aligned() -> None:
    mask = causal_additive_mask(
        batch_size=1,
        query_length=3,
        past_length=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )[0, 0]
    allowed = mask.eq(0)
    assert allowed.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


def test_cached_causal_mask_supports_batches() -> None:
    mask = causal_additive_mask(
        batch_size=2,
        query_length=1,
        past_length=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert mask.shape == (2, 1, 1, 5)
    assert torch.equal(mask[0], mask[1])
    assert torch.count_nonzero(mask).item() == 0
