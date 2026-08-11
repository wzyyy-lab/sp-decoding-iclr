#!/usr/bin/env python3
"""J000: audit JAPD metrics, full16 dataflow, and immutable invariants."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from sph.global_direct_selector import GlobalDirectCandidateSelector
from sph.japd import (
    BLOCK_LENGTH,
    CANDIDATES,
    candidate_gold_ranks,
    clean_support,
    strict_joint_two_frontier_metric,
)
from sph.japd_data import (
    FORBIDDEN_ONLINE_FEATURE_FIELDS,
    HEAD_BATCH_FIELDS,
    load_rollout_records,
)


EXPECTED_HISTORICAL = {
    "eligible": 745,
    "global": 15,
    "local": 0,
    "released_domino": 207,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-rollout", type=Path, required=True)
    parser.add_argument("--gate2-report", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-historical-counts", action="store_true")
    return parser.parse_args()


def proposal_candidate_ranks(candidate_ids: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
    matches = candidate_ids.eq(proposal.unsqueeze(-1))
    ranks = matches.to(torch.int64).argmax(dim=-1)
    return torch.where(matches.any(dim=-1), ranks, torch.full_like(ranks, -2))


def reference_joint_counts(
    records: list[dict[str, Any]],
    proposals: dict[str, torch.Tensor],
) -> dict[str, int]:
    counts = {name: 0 for name in proposals}
    eligible = 0
    for index, record in enumerate(records):
        ids = record["base_topk_ids"].long()
        gold = record["gold_ids"].long()
        target_matches = record["target_top1_ids"].long().eq(gold)
        matches = ids.eq(gold.unsqueeze(-1))
        ranks = torch.where(
            matches.any(dim=-1),
            matches.to(torch.int64).argmax(dim=-1),
            torch.full((BLOCK_LENGTH,), -1, dtype=torch.long),
        )
        first_invalid = (ranks.lt(0) | ~target_matches).nonzero().flatten()
        horizon = int(first_invalid[0]) if first_invalid.numel() else BLOCK_LENGTH
        base_errors = ranks[:horizon].ne(0).nonzero().flatten()
        if base_errors.numel() < 2:
            continue
        eligible += 1
        second_error = int(base_errors[1])
        for name, proposal in proposals.items():
            counts[name] += int(
                proposal[index, : second_error + 1]
                .eq(gold[: second_error + 1])
                .all()
            )
    return {"eligible": eligible, **counts}


@torch.inference_mode()
def architecture_audit() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(20260810)
    model = GlobalDirectCandidateSelector(
        hidden_size=2560,
        max_positions=BLOCK_LENGTH,
        max_candidates=CANDIDATES,
        model_dim=64,
        num_heads=4,
        num_layers=1,
        scope="global",
        mixer="axial",
        node_encoder="additive",
        dropout=0.1,
        initialization_seed=0,
    ).eval()
    local = GlobalDirectCandidateSelector(
        hidden_size=2560,
        max_positions=BLOCK_LENGTH,
        max_candidates=CANDIDATES,
        model_dim=64,
        num_heads=4,
        num_layers=1,
        scope="local",
        mixer="axial",
        node_encoder="additive",
        dropout=0.1,
        initialization_seed=0,
    ).eval()
    hidden = torch.randn((1, BLOCK_LENGTH, 2560), generator=generator)
    embeddings = torch.randn(
        (1, BLOCK_LENGTH, CANDIDATES, 2560), generator=generator
    )
    logits = torch.randn(
        (1, BLOCK_LENGTH, CANDIDATES), generator=generator
    ).sort(dim=-1, descending=True).values
    lse = torch.logsumexp(
        torch.cat(
            [logits, torch.randn((1, BLOCK_LENGTH, 32), generator=generator)],
            dim=-1,
        ),
        dim=-1,
    )
    anchors = torch.randn((1, 2560), generator=generator)
    identity = model(hidden, embeddings, logits, lse, anchors)
    expected_identity_scores = logits.float() - lse.float().unsqueeze(-1)
    if not torch.equal(identity.scores, expected_identity_scores):
        raise RuntimeError("zero-init JAPD head does not exactly reproduce base scores")
    if not torch.equal(
        identity.scores.argmax(dim=-1),
        torch.zeros((1, BLOCK_LENGTH), dtype=torch.long),
    ):
        raise RuntimeError("zero-init JAPD head does not reproduce the base token path")
    # A constant readout is orthogonal to LayerNorm output because the latter
    # has zero feature mean.  Use one non-uniform readout direction.
    model.residual_projection.weight.zero_()
    model.residual_projection.weight[0, 0] = 1.0
    local.residual_projection.weight.copy_(model.residual_projection.weight)
    baseline_global = model(hidden, embeddings, logits, lse, anchors).scores
    baseline_local = local(hidden, embeddings, logits, lse, anchors).scores
    changed_hidden = hidden.clone()
    # A uniform shift would be intentionally removed by the parameter-free
    # input LayerNorm.  Perturb one feature direction so the visibility test
    # changes information rather than only the discarded mean.
    changed_hidden[:, 15, 0].add_(30.0)
    changed_global = model(changed_hidden, embeddings, logits, lse, anchors).scores
    changed_local = local(changed_hidden, embeddings, logits, lse, anchors).scores
    global_remote_delta = float(
        (changed_global[:, 0] - baseline_global[:, 0]).abs().max()
    )
    local_remote_delta = float(
        (changed_local[:, 0] - baseline_local[:, 0]).abs().max()
    )
    if global_remote_delta <= 1e-8:
        raise RuntimeError("global position 0 did not respond to position 15")
    if local_remote_delta != 0.0:
        raise RuntimeError("matched local control responded to a remote position")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 433_852:
        raise RuntimeError(f"unexpected D64 parameter count {parameter_count}")
    forward_parameters = set(inspect.signature(model.forward).parameters)
    forbidden_signature = sorted(
        name for name in forward_parameters if name.startswith("target_")
    )
    if forbidden_signature:
        raise RuntimeError(
            f"online head accepts target-derived features: {forbidden_signature}"
        )
    if HEAD_BATCH_FIELDS & FORBIDDEN_ONLINE_FEATURE_FIELDS:
        raise RuntimeError("head batch whitelist contains forbidden target features")
    return {
        "parameter_count": parameter_count,
        "output_shape": list(baseline_global.shape),
        "zero_init_scores_exact": True,
        "zero_init_tokens_exact": True,
        "scope": model.scope,
        "mixer": model.mixer,
        "global_position15_to_position0_max_delta": global_remote_delta,
        "local_position15_to_position0_max_delta": local_remote_delta,
        "online_forward_parameters": sorted(forward_parameters),
        "forbidden_target_signature": forbidden_signature,
        "one_call_one_score_tensor": list(baseline_global.shape)
        == [1, BLOCK_LENGTH, CANDIDATES],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _, records = load_rollout_records(
        args.eval_rollout, split=args.split
    )
    gate2 = json.loads(args.gate2_report.read_text())
    per_block = gate2.get("per_block")
    if not isinstance(per_block, list) or len(per_block) != len(records):
        raise RuntimeError("Gate2 per_block rows do not align with rollout")
    names = ("global", "local", "released_domino")
    proposal_lists: dict[str, list[list[int]]] = {name: [] for name in names}
    policy_mismatches = 0
    for record, row in zip(records, per_block, strict=True):
        expected_identity = f"{record['sample_id']}@{int(record['anchor_offset'])}"
        if row.get("identity") != expected_identity:
            raise RuntimeError(
                f"Gate2/rollout identity mismatch: {row.get('identity')} vs {expected_identity}"
            )
        for name in names:
            proposal = row["proposals"][name]
            if len(proposal) != BLOCK_LENGTH:
                raise RuntimeError(f"{name} proposal is not full16")
            proposal_lists[name].append(proposal)
        policy_mismatches += int(
            row["proposals"]["released_domino"]
            != record["policy_ids"].long().tolist()
        )
    if policy_mismatches:
        raise RuntimeError(
            f"Gate2 released Domino differs from stored policy in {policy_mismatches} blocks"
        )

    candidate_ids = torch.stack(
        [record["base_topk_ids"].long() for record in records]
    )
    gold_ids = torch.stack([record["gold_ids"].long() for record in records])
    target_matches = torch.stack(
        [
            record["target_top1_ids"].long().eq(record["gold_ids"].long())
            for record in records
        ]
    )
    gold_ranks = candidate_gold_ranks(candidate_ids, gold_ids)
    proposals = {
        name: torch.tensor(values, dtype=torch.long)
        for name, values in proposal_lists.items()
    }
    vectorized: dict[str, int] = {}
    vectorized_eligible: int | None = None
    for name, proposal in proposals.items():
        predicted_ranks = proposal_candidate_ranks(candidate_ids, proposal)
        metric = strict_joint_two_frontier_metric(
            predicted_ranks, gold_ranks, target_matches
        )
        vectorized[name] = metric.numerator
        if vectorized_eligible is None:
            vectorized_eligible = metric.denominator
        elif metric.denominator != vectorized_eligible:
            raise AssertionError("J2 denominator depends on the evaluated method")
    assert vectorized_eligible is not None
    vectorized_counts = {"eligible": vectorized_eligible, **vectorized}
    reference_counts = reference_joint_counts(records, proposals)
    if vectorized_counts != reference_counts:
        raise RuntimeError(
            f"vectorized/reference J2 mismatch: {vectorized_counts} vs {reference_counts}"
        )
    if args.require_historical_counts and vectorized_counts != EXPECTED_HISTORICAL:
        raise RuntimeError(
            f"strict historical counts changed: {vectorized_counts} vs {EXPECTED_HISTORICAL}"
        )
    support, horizons = clean_support(gold_ranks, target_matches)
    architecture = architecture_audit()
    report = {
        "format": "japd_contract_audit_v1",
        "passed": True,
        "rollout": str(args.eval_rollout.resolve()),
        "gate2_report": str(args.gate2_report.resolve()),
        "records": len(records),
        "strict_clean_horizon": {
            "effective_blocks": int(horizons.gt(0).sum()),
            "supported_rows": int(support.sum()),
        },
        "joint_two_frontier": {
            "vectorized": vectorized_counts,
            "reference": reference_counts,
            "inclusive_endpoint": True,
            "historical_expected": EXPECTED_HISTORICAL,
        },
        "released_policy_report_mismatches": policy_mismatches,
        "architecture": architecture,
        "forbidden_online_features": sorted(FORBIDDEN_ONLINE_FEATURE_FIELDS),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
