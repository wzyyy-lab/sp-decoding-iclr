from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_domino_bias_scale import (  # noqa: E402
    domino_scaled_onpolicy_ids,
    normalize_scales,
    scale_key,
    summarize,
)


class ConstantBias(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*inputs.shape[:-1], 4), dtype=inputs.dtype)
        logits[..., 1] = 2.0
        return logits


def test_scaled_rollout_changes_only_causally_corrected_positions() -> None:
    torch.manual_seed(0)
    draft = SimpleNamespace(
        pure_draft_prefix_len=1,
        prefix_gru=torch.nn.GRU(2, 1, batch_first=True),
        embed_proj=ConstantBias(),
        use_bias_norm=False,
        use_bias_gate=False,
    )
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(4, 2))
    )
    base_logits = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]] * 3], dtype=torch.float32
    )
    proposals = domino_scaled_onpolicy_ids(
        draft,
        target,
        torch.tensor(2),
        torch.zeros((1, 3, 1)),
        base_logits,
        [0.0, 1.0],
    )
    assert proposals.tolist() == [[0, 0, 0], [0, 1, 1]]


def test_scale_normalization_and_prompt_balanced_summary() -> None:
    assert normalize_scales([0.5, 1, 0.5]) == [0.5, 1.0]
    assert scale_key(1.25) == "domino_scale_1p25"
    records = [
        {"sample_id": "many", "matched_horizon": 3, "method": 3},
        {"sample_id": "many", "matched_horizon": 3, "method": 3},
        {"sample_id": "one", "matched_horizon": 3, "method": 0},
    ]
    result = summarize(records, ["method"])["method"]
    assert result["mean_accepted_draft_tokens_round_weighted"] == 2.0
    assert result["mean_accepted_draft_tokens_prompt_balanced"] == 1.5
