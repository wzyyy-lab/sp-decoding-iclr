#!/usr/bin/env python3
"""GPU parity gate for the R056 SGLang proposal and forest primitives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel

from collect_r048_capacity import load_source
from profile_r052_exact_prefix import released_domino_head
from profile_r053_beam_graph import median_context_record
from sph.r053_tree import fast_candidate_domino_beam_from_base
from sph.r055_forest import traverse_padded_forest
from train_domino_cached_head import load_tensor_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("R056 proposal parity requires CUDA")
    sys.path.insert(0, str(args.sglang / "python"))
    from sglang.srt.speculative.domino_forest_proposal import (
        DominoProposalCudaGraph,
        fast_k64_domino_beam,
        released_domino_proposal,
    )
    from sglang.srt.speculative.domino_forest_utils import (
        build_domino_forest_positions,
        build_domino_forest_visibility,
        traverse_domino_forest,
    )

    _, records = load_source(args.source_rollout, "validation_select")
    record = median_context_record(records)
    device = torch.device("cuda:0")
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to(device=device, dtype=torch.bfloat16)
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    domino.requires_grad_(False)
    hidden = record["parallel_hidden"].to(device, torch.bfloat16)[None]
    anchor = torch.tensor(
        [int(record["anchor_token_id"])], dtype=torch.long, device=device
    )
    base_logits = F.linear(hidden, target_weight)

    reference_domino = released_domino_head(
        domino=domino,
        target_weight=target_weight,
        hidden=hidden,
        base_logits=base_logits,
        anchor=anchor,
    )
    ported_domino = released_domino_proposal(
        domino=domino,
        target_weight=target_weight,
        hidden=hidden,
        base_logits=base_logits,
        anchor=anchor,
    )
    if not torch.equal(reference_domino, ported_domino):
        raise RuntimeError("SGLang released Domino proposal differs from reference")

    reference_beam = fast_candidate_domino_beam_from_base(
        domino=domino,
        target_weight=target_weight,
        anchors=anchor,
        hidden=hidden,
        base_logits=base_logits,
        candidate_pool_topk=64,
        tree_support_size=16,
        beam_width=8,
    )
    ported_beam = fast_k64_domino_beam(
        domino=domino,
        target_weight=target_weight,
        anchor=anchor,
        hidden=hidden,
        base_logits=base_logits,
    )
    if not torch.equal(reference_beam.token_ids, ported_beam.token_ids):
        raise RuntimeError("SGLang W8 paths differ from frozen R055 reference")
    if not torch.equal(
        reference_beam.trunk_token_ids, ported_beam.trunk_token_ids
    ):
        raise RuntimeError("SGLang protected trunk differs from R055 reference")

    released_graph = DominoProposalCudaGraph(
        domino=domino, target_weight=target_weight, forest=False
    )
    graph_domino = released_graph(hidden=hidden, anchor=anchor).clone()
    forest_graph = DominoProposalCudaGraph(
        domino=domino, target_weight=target_weight, forest=True
    )
    graph_paths = forest_graph(hidden=hidden, anchor=anchor).clone()
    torch.cuda.synchronize()
    if not torch.equal(graph_domino, reference_domino):
        raise RuntimeError("released Domino graph changed proposal IDs")
    if not torch.equal(graph_paths, reference_beam.token_ids):
        raise RuntimeError("W8 graph changed path IDs")

    visibility = build_domino_forest_visibility(
        width=8, horizon=16, device=device
    )
    positions = build_domino_forest_positions(
        torch.tensor([int(record["context_length"])], device=device),
        width=8,
        horizon=16,
    )
    if visibility.shape != (129, 129) or positions.shape != (1, 129):
        raise RuntimeError("R056 fixed forest geometry changed")
    logits = torch.randn(
        (1, 129, target_weight.shape[0]),
        dtype=torch.bfloat16,
        device=device,
    )
    ref_traversal = traverse_padded_forest(reference_beam.token_ids[0], logits)
    posterior = logits.float().argmax(dim=-1)
    ported_traversal = traverse_domino_forest(
        paths=reference_beam.token_ids, posterior=posterior
    )
    for left, right, name in (
        (ref_traversal.accepted, ported_traversal.accepted, "accepted"),
        (
            ref_traversal.selected_path,
            ported_traversal.selected_path,
            "selected_path",
        ),
        (ref_traversal.next_token, ported_traversal.next_token, "next_token"),
        (
            ref_traversal.per_path_accepted,
            ported_traversal.per_path_accepted[0],
            "per_path_accepted",
        ),
    ):
        if not torch.equal(left, right):
            raise RuntimeError(f"R056 traversal differs in {name}")

    result = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "record_context_length": int(record["context_length"]),
        "released_domino_token_parity": True,
        "w8_path_parity": True,
        "protected_trunk_parity": True,
        "released_cuda_graph_parity": True,
        "forest_cuda_graph_parity": True,
        "forest_geometry_rows": 129,
        "forest_traversal_parity": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
