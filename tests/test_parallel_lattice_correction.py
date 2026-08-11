from __future__ import annotations

import torch

from sph.parallel_lattice_correction import ParallelLatticeCorrectionHead


def make_head() -> ParallelLatticeCorrectionHead:
    torch.manual_seed(123)
    return ParallelLatticeCorrectionHead(
        w_h=torch.randn(8, 24),
        w_out=torch.randn(41, 8),
        max_positions=5,
        candidates=4,
        modes=2,
        width=16,
        heads=4,
        feed_forward_width=32,
    )


def test_forward_is_single_parallel_chain() -> None:
    head = make_head()
    output = head(
        parallel_hiddens=torch.randn(2, 5, 24),
        base_logits=torch.randn(2, 5, 41),
        anchor_ids=torch.tensor([2, 3]),
        prefix_ids=torch.tensor([4, 5]),
    )
    assert output.token_ids.shape == (2, 5)
    assert output.correction_codes.shape == (2, 5, 8)
    assert output.route_weights.shape == (2, 5, 2)
    torch.testing.assert_close(
        output.route_weights.sum(dim=-1), torch.ones(2, 5)
    )
    assert output.corrected_logits is not None
    torch.testing.assert_close(
        output.token_ids, output.corrected_logits.argmax(dim=-1)
    )


def test_preprojected_inference_matches_training_path() -> None:
    head = make_head().eval()
    inputs = {
        "parallel_hiddens": torch.randn(1, 5, 24),
        "base_logits": torch.randn(1, 5, 41),
        "anchor_ids": torch.tensor([2]),
        "prefix_ids": torch.tensor([4]),
    }
    before = head(**inputs)
    head.prepare_inference()
    after = head(**inputs)
    torch.testing.assert_close(before.correction_codes, after.correction_codes)
    torch.testing.assert_close(before.corrected_logits, after.corrected_logits)
    torch.testing.assert_close(before.token_ids, after.token_ids)


def test_semantic_table_and_full_hidden_path_match_after_freeze() -> None:
    torch.manual_seed(124)
    head = ParallelLatticeCorrectionHead(
        w_h=torch.randn(8, 24),
        w_out=torch.randn(41, 8),
        token_embeddings=torch.randn(41, 24),
        use_full_hidden=True,
        max_positions=5,
        candidates=4,
        modes=2,
        width=16,
        heads=4,
        feed_forward_width=32,
    ).eval()
    inputs = {
        "parallel_hiddens": torch.randn(1, 5, 24),
        "base_logits": torch.randn(1, 5, 41),
        "anchor_ids": torch.tensor([2]),
        "prefix_ids": torch.tensor([4]),
    }
    before = head(**inputs)
    head.prepare_inference()
    after = head(**inputs)
    torch.testing.assert_close(before.correction_codes, after.correction_codes)
    torch.testing.assert_close(before.corrected_logits, after.corrected_logits)


def test_parameter_budget_excludes_shared_released_basis() -> None:
    head = ParallelLatticeCorrectionHead(
        w_h=torch.empty(256, 2560, device="meta"),
        w_out=torch.empty(151936, 256, device="meta"),
    )
    assert head.trainable_parameter_count < 600_000
    assert head.active_parameter_count < 40_200_000


def test_all_main_trainable_branches_receive_gradient() -> None:
    head = make_head()
    output = head(
        parallel_hiddens=torch.randn(2, 5, 24),
        base_logits=torch.randn(2, 5, 41),
        anchor_ids=torch.tensor([2, 3]),
        prefix_ids=torch.tensor([4, 5]),
    )
    assert output.corrected_logits is not None
    loss = output.corrected_logits.square().mean()
    loss.backward()
    assert head.lexical_projection.weight.grad is not None
    assert head.local_attention.query.weight.grad is not None
    assert head.global_blocks[0].attention.query.weight.grad is not None
    assert head.mode_route.weight.grad is not None
    assert head.code_projection.weight.grad is not None
