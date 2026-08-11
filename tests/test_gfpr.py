from __future__ import annotations

import torch
from torch import nn
import pytest

from sph.gfpr import (
    accepted_lengths,
    adaptation_state_dict,
    all_position_onpolicy_decode,
    all_position_teacher_logits,
    frontier_masks,
    next_anchor_offsets,
    normalized_frontier_margin_loss,
    oracle_prefix_lengths,
    paired_prompt_summary,
    load_adaptation,
    topk_oracle_matches,
)
from collect_gfpr_rollouts import _fixed_offsets, _reconstruct_sequence


class TinyDomino(nn.Module):
    def __init__(self, width: int, state: int, vocab: int) -> None:
        super().__init__()
        self.prefix_gru = nn.GRU(width, state, batch_first=True, bias=False)
        self.embed_proj = nn.Sequential(
            nn.Linear(width + state, 5, bias=False),
            nn.SiLU(),
            nn.Linear(5, vocab, bias=False),
        )
        self.use_bias_norm = False
        self.use_bias_gate = False


def test_acceptance_and_bonus_advance() -> None:
    proposals = torch.tensor([[1, 2, 9, 4], [1, 2, 3, 4], [9, 2, 3, 4]])
    gold = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    lengths = accepted_lengths(proposals, gold)
    assert lengths.tolist() == [2, 4, 0]
    assert next_anchor_offsets(torch.tensor([0, 7, 20]), lengths).tolist() == [3, 12, 21]


def test_position_zero_identity_and_teacher_state_agree() -> None:
    torch.manual_seed(4)
    width, state, vocab, positions = 6, 4, 11, 4
    domino = TinyDomino(width, state, vocab)
    target_weight = torch.randn(vocab, width)
    hidden = torch.randn(3, positions, width)
    anchors = torch.tensor([1, 2, 3])
    decoded = all_position_onpolicy_decode(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        position_zero_scale=0.0,
        topk=5,
    )
    base = torch.nn.functional.linear(hidden, target_weight)
    assert torch.equal(decoded.token_ids[:, 0], base[:, 0].argmax(dim=-1))
    teacher, teacher_base = all_position_teacher_logits(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        gold=decoded.token_ids,
        hidden=hidden,
        position_zero_scale=0.0,
        return_base_logits=True,
    )
    assert torch.equal(teacher[:, 0].argmax(dim=-1), decoded.token_ids[:, 0])
    assert torch.equal(teacher.argmax(dim=-1), decoded.token_ids)
    assert torch.equal(teacher_base[:, 0], base[:, 0].float())


def test_frontier_masks_cover_prefix_and_one_repair() -> None:
    logits = torch.full((3, 4, 5), -3.0)
    gold = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    predictions = torch.tensor([[0, 1, 4, 3], [4, 1, 2, 3], [0, 1, 2, 3]])
    logits.scatter_(-1, predictions.unsqueeze(-1), 3.0)
    frontier, protected, repair = frontier_masks(logits, gold)
    assert frontier.tolist() == [2, 0, 4]
    assert protected.sum(dim=-1).tolist() == [2, 0, 4]
    assert repair.sum(dim=-1).tolist() == [1, 1, 0]


def test_keep_budget_is_normalized_per_block() -> None:
    # Every protected gold margin is zero and keep_margin is one.  A block
    # with three protected positions and one with one protected position must
    # therefore each contribute a per-block keep loss of one.
    logits = torch.zeros(2, 4, 3, requires_grad=True)
    gold = torch.zeros(2, 4, dtype=torch.long)
    with torch.no_grad():
        logits[0, 3, 1] = 2.0
        logits[1, 1, 1] = 2.0
    result = normalized_frontier_margin_loss(
        logits,
        gold,
        break_margin=0.0,
        keep_margin=1.0,
        break_weight=0.0,
        keep_weight=1.0,
    )
    assert result.frontier.tolist() == [3, 1]
    assert torch.isclose(result.keep_loss, torch.tensor(1.0))
    result.loss.backward()
    assert logits.grad is not None


def test_topk_union_contract_and_oracle_prefix() -> None:
    base = torch.tensor(
        [
            [
                list(range(10, 26)),
                list(range(30, 46)),
                list(range(50, 66)),
            ]
        ]
    )
    released = torch.tensor([[10, 99, 99]])
    # pos0 in base; pos1 is base rank16 but is displaced by released action;
    # pos2 is rescued by the released action.
    gold = torch.tensor([[10, 45, 99]])
    matches = topk_oracle_matches(
        base_topk_ids=base, released_ids=released, gold=gold
    )
    assert matches["base16"].tolist() == [[True, True, False]]
    assert matches["k17"].tolist() == [[True, True, True]]
    assert matches["k16"].tolist() == [[True, False, True]]
    assert oracle_prefix_lengths(matches["base16"]).tolist() == [2]
    assert oracle_prefix_lengths(matches["k17"]).tolist() == [3]
    assert oracle_prefix_lengths(matches["k16"]).tolist() == [1]


def test_reconstruct_nested_canonical_sequence() -> None:
    prompt = torch.tensor([8, 9])
    continuation = torch.arange(20, 30)
    full = torch.cat([prompt, continuation])
    records = []
    for offset in (0, 3):
        context_length = len(prompt) + offset
        records.append(
            {
                "sample_id": "sample",
                "prompt_token_count": len(prompt),
                "context_length": context_length,
                "context_ids_before_anchor": full[:context_length].clone(),
                "anchor_token_id": int(continuation[offset]),
                "gold_ids": continuation[offset + 1 : offset + 6].clone(),
                "anchor_offset": offset,
            }
        )
    sequence, prompt_tokens = _reconstruct_sequence(records, "sample", torch.device("cpu"))
    assert prompt_tokens == 2
    assert torch.equal(sequence, full[: 2 + 3 + 1 + 5])
    assert _fixed_offsets(records) == [0, 3]


def test_paired_prompt_summary_clusters_by_prompt_and_reports_harm() -> None:
    report = paired_prompt_summary(
        ["a", "a", "b", "b"],
        [1, 3, 4, 4],
        [3, 3, 3, 4],
        bootstrap_samples=200,
        seed=7,
    )
    # Prompt a improves from 2 to 3; prompt b falls from 4 to 3.5.
    assert report["baseline_eal_prompt_balanced"] == 3.0
    assert report["current_eal_prompt_balanced"] == 3.25
    assert report["paired_delta"] == 0.25
    assert report["gained_accepted_tokens"] == 2.0
    assert report["lost_accepted_tokens"] == 1.0
    assert report["lost_to_gained_ratio"] == 0.5
    assert report["harmful_prompt_fraction"] == 0.5


def test_head_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(9)
    source = TinyDomino(4, 3, 7)
    checkpoint = tmp_path / "head.pt"
    torch.save(adaptation_state_dict(source, torch.tensor(0.25)), checkpoint)
    target = TinyDomino(4, 3, 7)
    scale = load_adaptation(target, checkpoint)
    assert float(scale) == 0.25
    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        assert torch.equal(source_parameter, target_parameter)


def test_claim_bearing_checkpoint_load_checks_provenance(tmp_path) -> None:
    source = TinyDomino(4, 3, 7)
    payload = adaptation_state_dict(source, torch.tensor(0.0))
    payload["provenance"] = {
        "target": str(tmp_path / "target"),
        "base_domino": str(tmp_path / "domino"),
    }
    checkpoint = tmp_path / "provenance.pt"
    torch.save(payload, checkpoint)
    load_adaptation(
        TinyDomino(4, 3, 7),
        checkpoint,
        expected_target=tmp_path / "target",
        expected_base_domino=tmp_path / "domino",
    )
    with pytest.raises(ValueError, match="does not match"):
        load_adaptation(
            TinyDomino(4, 3, 7),
            checkpoint,
            expected_target=tmp_path / "wrong-target",
        )
