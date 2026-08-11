from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from types import SimpleNamespace

from sph.domino_joint_runtime import (
    CanonicalBlock,
    dflash_positions_and_mask,
    domino_onpolicy_ids,
    domino_prediction_hidden,
    domino_teacher_and_base_logits,
    domino_teacher_logits,
    frontier_margin_joint_loss,
    greedy_reachable_joint_loss,
    released_domino_corrected_logits,
    select_even_prompt_blocks,
    summarize_prompt_balanced_lengths,
    target_distilled_union_joint_loss,
    target_frontier_distilled_union_joint_loss,
    target_full_vocab_distilled_joint_loss,
    union_topk_frontier_protected_joint_loss,
    union_topk_oracle_prefix_joint_loss,
    union_topk_reachable_joint_loss,
)


def test_even_prompt_selection_preserves_nested_offsets() -> None:
    full = torch.arange(40)
    records = [
        CanonicalBlock(
            sample_id="p",
            domain="code",
            context_ids=full[: 5 + index],
            anchor_token_id=index,
            gold_ids=torch.arange(15),
            anchor_offset=index,
        )
        for index in range(8)
    ]
    selected = select_even_prompt_blocks(records)
    assert [record.anchor_offset for record in selected] == [0, 2, 5, 7]


def test_variable_context_positions_and_mask_keep_noise_absolute_positions() -> None:
    positions, mask = dflash_positions_and_mask(
        torch.tensor([2, 4]),
        maximum_context=4,
        block_size=3,
        dtype=torch.float32,
        device="cpu",
    )
    assert positions.tolist() == [[0, 1, 0, 0, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6]]
    assert mask.shape == (2, 1, 3, 7)
    assert torch.all(mask[0, :, :, :2] == 0)
    assert torch.all(mask[0, :, :, 2:4] < -1e30)
    assert torch.all(mask[0, :, :, 4:] == 0)
    assert torch.all(mask[1] == 0)


def test_domino_shift_label_uses_hidden_zero_through_fourteen() -> None:
    domino = SimpleNamespace(
        block_size=16,
        config=SimpleNamespace(
            dflash_config={"shift_label": True, "pure_draft_prefix_len": 1}
        ),
    )
    full = torch.arange(2 * 16 * 3).view(2, 16, 3)
    selected = domino_prediction_hidden(domino, full, horizon=15)
    assert torch.equal(selected, full[:, :15])
    selected_full = domino_prediction_hidden(domino, full, horizon=16)
    assert torch.equal(selected_full, full)


def test_released_domino_adds_base_and_bias_before_float_argmax() -> None:
    base = torch.tensor([-0.79296875, -1.8125], dtype=torch.bfloat16)
    bias = torch.tensor([-4.3125, -3.28125], dtype=torch.bfloat16)
    released = released_domino_corrected_logits(base, bias)
    assert released.dtype == torch.bfloat16
    assert released.tolist() == [-5.09375, -5.09375]
    assert int(released.argmax()) == 0
    assert int((base.float() + bias.float()).argmax()) == 1


class TinyDomino(nn.Module):
    def __init__(self, hidden: int, vocabulary: int) -> None:
        super().__init__()
        self.prefix_gru = nn.GRU(hidden, 3, batch_first=True, bias=False)
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden + 3, 5, bias=False),
            nn.SiLU(),
            nn.Linear(5, vocabulary, bias=False),
        )


def test_domino_teacher_logits_match_explicit_correct_prefix_loop() -> None:
    torch.manual_seed(41)
    batch, positions, hidden_size, vocabulary = 4, 6, 7, 11
    domino = TinyDomino(hidden_size, vocabulary)
    weight = torch.randn(vocabulary, hidden_size)
    hidden = torch.randn(batch, positions, hidden_size, requires_grad=True)
    anchors = torch.randint(0, vocabulary, (batch,))
    gold = torch.randint(0, vocabulary, (batch, positions))

    vectorized = domino_teacher_logits(
        domino=domino,
        target_weight=weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
    )
    base = F.linear(hidden, weight).float()
    reference = [base[:, :1]]
    first_prefix = torch.cat([anchors[:, None], gold[:, :1]], dim=-1)
    _, state = domino.prefix_gru(F.embedding(first_prefix, weight))
    for position in range(1, positions):
        correction = domino.embed_proj(
            torch.cat(
                [hidden[:, position : position + 1], state.transpose(0, 1)],
                dim=-1,
            )
        ).float()
        reference.append(base[:, position : position + 1] + correction)
        if position + 1 < positions:
            _, state = domino.prefix_gru(
                F.embedding(gold[:, position : position + 1], weight), state
            )
    looped = torch.cat(reference, dim=1)
    torch.testing.assert_close(vectorized, looped)
    vectorized.sum().backward()
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) > 0


def test_domino_teacher_logits_keeps_full_lm_head_geometry() -> None:
    torch.manual_seed(42)
    hidden_size, vocabulary = 7, 11
    domino = TinyDomino(hidden_size, vocabulary)
    weight = torch.randn(vocabulary, hidden_size)
    # Released inference sends all 16 backbone states through the LM head, but
    # the canonical acceptance target contains only the first 15 tokens.
    hidden = torch.randn(1, 16, hidden_size, requires_grad=True)
    anchors = torch.randint(0, vocabulary, (1,))
    gold = torch.randint(0, vocabulary, (1, 15))
    logits = domino_teacher_logits(
        domino=domino,
        target_weight=weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
    )
    assert logits.shape == (1, 15, vocabulary)
    logits.sum().backward()
    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad[:, :15]) > 0
    assert torch.count_nonzero(hidden.grad[:, 15:]) == 0


def test_teacher_and_base_logits_share_released_full_geometry() -> None:
    torch.manual_seed(44)
    hidden_size, vocabulary = 7, 11
    domino = TinyDomino(hidden_size, vocabulary)
    weight = torch.randn(vocabulary, hidden_size)
    hidden = torch.randn(1, 16, hidden_size)
    anchors = torch.randint(0, vocabulary, (1,))
    gold = torch.randint(0, vocabulary, (1, 15))
    final, base = domino_teacher_and_base_logits(
        domino=domino,
        target_weight=weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
    )
    assert final.shape == base.shape == (1, 15, vocabulary)
    torch.testing.assert_close(base, F.linear(hidden, weight)[:, :15].float())
    torch.testing.assert_close(final[:, :1], base[:, :1])


def test_teacher_logits_optionally_backpropagates_through_prefix_gru() -> None:
    torch.manual_seed(45)
    hidden_size, vocabulary = 7, 11
    domino = TinyDomino(hidden_size, vocabulary)
    weight = torch.randn(vocabulary, hidden_size)
    hidden = torch.randn(1, 6, hidden_size, requires_grad=True)
    anchors = torch.randint(0, vocabulary, (1,))
    gold = torch.randint(0, vocabulary, (1, 6))

    frozen_head = domino_teacher_logits(
        domino=domino,
        target_weight=weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
        train_causal_head=False,
    )
    frozen_head.sum().backward()
    assert domino.prefix_gru.weight_ih_l0.grad is None

    domino.zero_grad(set_to_none=True)
    hidden.grad = None
    trainable_head = domino_teacher_logits(
        domino=domino,
        target_weight=weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
        train_causal_head=True,
    )
    trainable_head.sum().backward()
    assert domino.prefix_gru.weight_ih_l0.grad is not None
    assert torch.count_nonzero(domino.prefix_gru.weight_ih_l0.grad) > 0


def test_greedy_reachable_joint_loss_stops_after_first_miss() -> None:
    # Row 0 misses at position 0, so only base-zero CE is active.  Row 1 is
    # correct at positions 0 and 1 then misses at 2, so final CE is active at
    # positions 1 and 2, but not at position 3.
    gold = torch.zeros((2, 4), dtype=torch.long)
    final = torch.tensor(
        [
            [[0.0, 2.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]],
        ],
        requires_grad=True,
    )
    base = torch.zeros_like(final, requires_grad=True)
    loss, parts = greedy_reachable_joint_loss(final, base, gold)
    loss.backward()
    assert float(parts["reachable_suffix_positions_per_block"]) == 1.0
    assert torch.count_nonzero(final.grad[0]) == 0
    assert torch.count_nonzero(final.grad[1, 0]) == 0
    assert torch.count_nonzero(final.grad[1, 1]) > 0
    assert torch.count_nonzero(final.grad[1, 2]) > 0
    assert torch.count_nonzero(final.grad[1, 3]) == 0
    assert torch.count_nonzero(base.grad[:, 0]) > 0
    assert torch.count_nonzero(base.grad[:, 1:]) == 0


def test_frontier_margin_repairs_one_boundary_and_protects_only_prefix() -> None:
    gold = torch.zeros((3, 3), dtype=torch.long)
    logits = torch.tensor(
        [
            # Miss at zero: only position zero is the repair frontier.
            [[0.0, 2.0], [2.0, 0.0], [2.0, 0.0]],
            # Correct zero, miss one: zero has a safe margin and no gradient.
            [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]],
            # Fully correct but with 0.01 margins: protect all three positions.
            [[0.01, 0.0], [0.01, 0.0], [0.01, 0.0]],
        ],
        requires_grad=True,
    )
    loss, parts = frontier_margin_joint_loss(
        logits, gold, protection_margin=0.05
    )
    loss.backward()
    torch.testing.assert_close(
        parts["accepted_prefix_positions_per_block"], torch.tensor(4 / 3)
    )
    assert int(parts["repairable_blocks_per_batch"]) == 2
    assert torch.count_nonzero(logits.grad[0, 0]) > 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0
    assert torch.count_nonzero(logits.grad[1, 0]) == 0
    assert torch.count_nonzero(logits.grad[1, 1]) > 0
    assert torch.count_nonzero(logits.grad[1, 2]) == 0
    assert torch.count_nonzero(logits.grad[2]) > 0


def test_union_topk_listwise_loss_stops_after_current_first_miss() -> None:
    gold = torch.full((1, 3), 2, dtype=torch.long)
    final = torch.tensor(
        [
            [
                [0.0, 1.0, 3.0, -2.0, -3.0],
                [0.0, 3.0, 2.0, -2.0, -3.0],
                [0.0, 1.0, 3.0, -2.0, -3.0],
            ]
        ],
        requires_grad=True,
    )
    base = torch.tensor(
        [[[0.0, 1.5, 2.0, -1.0, -2.0]] * 3], requires_grad=True
    )
    loss, parts = union_topk_reachable_joint_loss(
        final, base, gold, topk=2
    )
    loss.backward()
    assert float(parts["reachable_covered_positions_per_block"]) == 2.0
    assert torch.count_nonzero(final.grad[0, 0]) > 0
    assert torch.count_nonzero(final.grad[0, 1]) > 0
    assert torch.count_nonzero(final.grad[0, 2]) == 0
    # Vocabulary item four is outside both Top-2 lists and receives no gradient.
    assert torch.count_nonzero(final.grad[..., 4]) == 0
    assert base.grad is None


def test_union_topk_oracle_prefix_trains_past_current_greedy_miss() -> None:
    gold = torch.full((1, 4), 2, dtype=torch.long)
    final = torch.tensor(
        [[
            [0.0, 3.0, 2.0, -2.0, -3.0],
            [0.0, 3.0, 2.0, -2.0, -3.0],
            [0.0, 3.0, -2.0, 2.5, -3.0],
            [0.0, 3.0, 2.0, -2.0, -3.0],
        ]],
        requires_grad=True,
    )
    # Gold is in the union Top-2 at positions 0 and 1, outside it at 2, and
    # back inside at 3.  The oracle prefix must train 0 and 1 together, then
    # stop permanently at the first unavailable token.
    base = torch.tensor(
        [[
            [0.0, 1.5, 2.0, -1.0, -2.0],
            [0.0, 1.5, 2.0, -1.0, -2.0],
            [0.0, 2.0, -2.0, 1.5, -3.0],
            [0.0, 1.5, 2.0, -1.0, -2.0],
        ]],
        requires_grad=True,
    )
    loss, parts = union_topk_oracle_prefix_joint_loss(
        final, base, gold, topk=2
    )
    loss.backward()
    assert float(parts["oracle_prefix_positions_per_block"]) == 2.0
    assert torch.count_nonzero(final.grad[0, 0]) > 0
    assert torch.count_nonzero(final.grad[0, 1]) > 0
    assert torch.count_nonzero(final.grad[0, 2:]) == 0
    assert base.grad is None


def test_union_topk_frontier_repairs_chain_without_easy_prefix_ce() -> None:
    gold = torch.full((1, 4), 2, dtype=torch.long)
    final = torch.tensor(
        [[
            [0.0, -1.0, 3.0, -2.0, -3.0],
            [0.0, 3.0, 2.0, -2.0, -3.0],
            [0.0, 3.0, 2.0, -2.0, -3.0],
            [0.0, 3.0, -2.0, 2.5, -3.0],
        ]],
        requires_grad=True,
    )
    base = torch.tensor(
        [[
            [0.0, 1.5, 2.0, -1.0, -2.0],
            [0.0, 1.5, 2.0, -1.0, -2.0],
            [0.0, 1.5, 2.0, -1.0, -2.0],
            [0.0, 2.0, -2.0, 1.5, -3.0],
        ]],
        requires_grad=True,
    )
    loss, parts = union_topk_frontier_protected_joint_loss(
        final,
        base,
        gold,
        topk=2,
        protection_margin=0.05,
    )
    loss.backward()
    assert float(parts["accepted_prefix_positions_per_block"]) == 1.0
    assert float(parts["repair_positions_per_block"]) == 2.0
    # Position zero is already safely correct and gets no easy-token CE.
    assert torch.count_nonzero(final.grad[0, 0]) == 0
    assert torch.count_nonzero(final.grad[0, 1]) > 0
    assert torch.count_nonzero(final.grad[0, 2]) > 0
    assert torch.count_nonzero(final.grad[0, 3]) == 0
    assert base.grad is None


def test_target_distilled_union_masks_suffix_and_target_replay_mismatch() -> None:
    # Row 0 is correct at zero and misses at one, so positions zero/one are
    # eligible.  The target replay intentionally disagrees with canonical gold
    # at position zero, leaving only the repair frontier active.  Position two
    # must remain masked even though the target and gold agree there.
    gold = torch.zeros((1, 3), dtype=torch.long)
    final = torch.tensor(
        [[[3.0, 1.0, 0.0], [1.0, 3.0, 0.0], [3.0, 1.0, 0.0]]],
        requires_grad=True,
    )
    base = torch.tensor(
        [[[2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]
    )
    target = torch.tensor(
        [[[1.0, 2.0, 0.0], [4.0, 0.0, -1.0], [4.0, 0.0, -1.0]]]
    )
    loss, parts = target_distilled_union_joint_loss(
        final,
        base,
        target,
        gold,
        topk=2,
        temperature=2.0,
    )
    loss.backward()
    assert float(parts["active_positions_per_block"]) == 1.0
    assert torch.count_nonzero(final.grad[0, 0]) == 0
    assert torch.count_nonzero(final.grad[0, 1]) > 0
    assert torch.count_nonzero(final.grad[0, 2]) == 0


def test_target_distilled_union_inserts_current_action_and_reaches_gold() -> None:
    # Base Top-2 excludes vocabulary ID 2, but the current Domino action is 2;
    # replacing the last slot must make the gold frontier repairable.
    gold = torch.tensor([[2]])
    final = torch.tensor([[[0.0, 1.0, 3.0, 2.0]]], requires_grad=True)
    base = torch.tensor([[[4.0, 3.0, 1.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0, 4.0, 2.0]]])
    loss, parts = target_distilled_union_joint_loss(
        final, base, target, gold, topk=2
    )
    loss.backward()
    assert float(parts["gold_available_active_fraction"]) == 1.0
    assert torch.count_nonzero(final.grad[..., 2]) > 0


def test_target_frontier_distill_repairs_boundary_and_hinges_prefix_only() -> None:
    gold = torch.zeros((1, 3), dtype=torch.long)
    final = torch.tensor(
        [[[0.01, 0.0, -1.0], [0.0, 2.0, -1.0], [2.0, 0.0, -1.0]]],
        requires_grad=True,
    )
    base = torch.tensor(
        [[[2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]
    )
    target = torch.tensor(
        [[[3.0, 0.0, -1.0], [4.0, 0.0, -1.0], [4.0, 0.0, -1.0]]]
    )
    loss, parts = target_frontier_distilled_union_joint_loss(
        final,
        base,
        target,
        gold,
        topk=2,
        protection_margin=0.05,
    )
    loss.backward()
    assert float(parts["accepted_prefix_positions_per_block"]) == 1.0
    assert float(parts["repairable_frontier_fraction"]) == 1.0
    assert torch.count_nonzero(final.grad[0, 0]) > 0
    assert torch.count_nonzero(final.grad[0, 1]) > 0
    assert torch.count_nonzero(final.grad[0, 2]) == 0


def test_target_full_vocab_distill_reaches_missing_gold_and_exact_future() -> None:
    # Position zero is already correct but below the safety margin. Position
    # one is the live frontier and its gold ID 0 is outside base Top-2.
    # Position two is trained under its exact gold prefix even though it is not
    # currently reachable.
    gold = torch.zeros((1, 3), dtype=torch.long)
    final = torch.tensor(
        [
            [
                [0.01, 0.0, -1.0, -2.0],
                [0.0, 3.0, 2.0, 1.0],
                [0.0, 2.0, 1.0, -1.0],
            ]
        ],
        requires_grad=True,
    )
    base = torch.tensor(
        [
            [
                [3.0, 2.0, 1.0, 0.0],
                [0.0, 4.0, 3.0, 2.0],
                [0.0, 3.0, 2.0, 1.0],
            ]
        ]
    )
    target = torch.tensor(
        [[[4.0, 0.0, -1.0, -2.0], [4.0, 1.0, 0.0, -1.0], [4.0, 0.0, -1.0, -2.0]]]
    )
    loss, parts = target_full_vocab_distilled_joint_loss(
        final,
        base,
        target,
        gold,
        topk=2,
        future_weight=1.0,
    )
    loss.backward()
    assert float(parts["frontier_gold_in_base_topk_fraction"]) == 0.0
    assert float(parts["repair_target_positions_per_block"]) == 1.0
    assert float(parts["future_target_positions_per_block"]) == 1.0
    assert torch.count_nonzero(final.grad[0, 0]) > 0
    assert final.grad[0, 1, 0] < 0
    assert torch.count_nonzero(final.grad[0, 2]) > 0


def test_target_full_vocab_distill_masks_replay_mismatch_and_optional_future() -> None:
    gold = torch.zeros((1, 3), dtype=torch.long)
    final = torch.tensor(
        [[[3.0, 0.0, -1.0], [0.0, 3.0, -1.0], [0.0, 2.0, -1.0]]],
        requires_grad=True,
    )
    base = final.detach().clone()
    target = torch.tensor(
        [[[4.0, 0.0, -1.0], [4.0, 0.0, -1.0], [0.0, 4.0, -1.0]]]
    )
    loss, parts = target_full_vocab_distilled_joint_loss(
        final,
        base,
        target,
        gold,
        topk=2,
        future_weight=0.0,
        protection_margin=0.0,
    )
    loss.backward()
    assert float(parts["repair_target_positions_per_block"]) == 1.0
    assert float(parts["future_target_positions_per_block"]) == 0.0
    assert torch.count_nonzero(final.grad[0, 0]) == 0
    assert torch.count_nonzero(final.grad[0, 1]) > 0
    assert torch.count_nonzero(final.grad[0, 2]) == 0


def test_onpolicy_ids_and_prompt_balanced_summary() -> None:
    torch.manual_seed(43)
    domino = TinyDomino(5, 9)
    weight = torch.randn(9, 5)
    hidden = torch.randn(2, 4, 5)
    anchors = torch.tensor([1, 2])
    proposals = domino_onpolicy_ids(
        domino=domino, target_weight=weight, anchors=anchors, hidden=hidden
    )
    assert proposals.shape == (2, 4)

    summary = summarize_prompt_balanced_lengths(
        ["a", "a", "b"],
        ["math", "math", "code"],
        [2, 4, 3],
        horizon=4,
    )
    assert summary["overall"]["mean_accepted_draft_tokens_prompt_balanced"] == 3
    assert summary["overall"]["mean_accepted_draft_tokens_round_weighted"] == 3
