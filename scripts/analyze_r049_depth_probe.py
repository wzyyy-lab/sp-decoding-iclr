#!/usr/bin/env python3
"""Zero-parameter multi-depth target-logit probe for R049-A."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM

from collect_r048_capacity import (
    authoritative_frontier_contract,
    load_source,
    prompt_balanced,
    select_balanced_prompts,
)
from sph.fast_r048 import candidate_union_with_proposal, fast_candidate_domino_decode
from sph.gfpr import accepted_lengths
from sph.r048_capacity import select_zero_harm_threshold
from sph.r048_layer_split import clone_dynamic_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--candidate-topk", type=int, default=64)
    parser.add_argument("--probe-layers", default="4,8,12,16,24,32,36")
    return parser.parse_args()


def parse_layers(specification: str, depth: int) -> list[int]:
    layers = sorted({int(value.strip()) for value in specification.split(",")})
    if not layers or layers[0] < 1 or layers[-1] > depth:
        raise ValueError("probe layers lie outside target depth")
    return layers


def centered_candidate_scores(
    scores: Tensor,
    candidate_ids: Tensor,
    proposal: Tensor,
) -> Tensor:
    if scores.shape != candidate_ids.shape or proposal.shape != scores.shape[:2]:
        raise ValueError("candidate score contract differs in shape")
    matches = candidate_ids.eq(proposal.unsqueeze(-1))
    if not bool(matches.any(dim=-1).all().item()):
        raise ValueError("candidate support does not retain the proposal")
    index = matches.to(torch.long).argmax(dim=-1)
    proposal_score = scores.float().gather(-1, index.unsqueeze(-1))
    return scores.float() - proposal_score


def force_keep_rows(
    scores: Tensor,
    candidate_ids: Tensor,
    proposal: Tensor,
    keep_mask: Tensor,
) -> Tensor:
    """Conservatively make the proposal the only selectable token on rows."""

    if keep_mask.shape != proposal.shape or scores.shape != candidate_ids.shape:
        raise ValueError("KEEP mask differs from candidate lattice")
    matches = candidate_ids.eq(proposal.unsqueeze(-1))
    if not bool(matches.any(dim=-1).all().item()):
        raise ValueError("candidate support does not retain the proposal")
    proposal_index = matches.to(torch.long).argmax(dim=-1)
    output = scores.float().clone().masked_fill(keep_mask.unsqueeze(-1), -torch.inf)
    proposal_values = output.gather(-1, proposal_index.unsqueeze(-1))
    proposal_values = torch.where(
        keep_mask.unsqueeze(-1),
        torch.zeros_like(proposal_values),
        proposal_values,
    )
    return output.scatter(-1, proposal_index.unsqueeze(-1), proposal_values)


def frontier_token_ranking_lengths(
    *,
    proposal: Tensor,
    verifier_top1: Tensor,
    candidate_ids: Tensor,
    candidate_scores: Tensor,
    baseline_lengths: Tensor,
    oracle_lengths: Tensor,
) -> Tensor:
    """Exact reward if only candidate ranking at the true frontier is tested."""

    if proposal.ndim != 2 or verifier_top1.shape != proposal.shape:
        raise ValueError("proposal/verifier shape mismatch")
    batch, positions = proposal.shape
    if baseline_lengths.shape != (batch,) or oracle_lengths.shape != (batch,):
        raise ValueError("length shape mismatch")
    best_index = candidate_scores.float().argmax(dim=-1)
    best_token = candidate_ids.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    frontier = baseline_lengths.clamp_max(positions - 1)
    predicted = best_token.gather(1, frontier[:, None])[:, 0]
    target = verifier_top1.gather(1, frontier[:, None])[:, 0]
    correct = baseline_lengths.lt(positions) & predicted.eq(target)
    return torch.where(correct, oracle_lengths, baseline_lengths)


def depth_gate_decision(
    layer_reports: dict[int, dict[str, Any]],
    *,
    shallow_limit: int = 12,
    policy_gate: float = 0.90,
    token_gate: float = 0.80,
) -> dict[str, Any]:
    shallow = sorted(layer for layer in layer_reports if layer <= shallow_limit)
    direct = [
        layer
        for layer in shallow
        if float(layer_reports[layer]["policy"]["oracle_gain_recovery"])
        >= policy_gate
        and float(layer_reports[layer]["policy"]["gain_block_recovery"])
        >= policy_gate
    ]
    if direct:
        return {
            "decision": "GO_DISJOINT_DIRECT",
            "selected_layer": direct[0],
            "reason": "shallow zero-parameter policy recovered at least 90%",
        }
    gate_only = [
        layer
        for layer in shallow
        if float(layer_reports[layer]["token_ranking"]["oracle_gain_recovery"])
        >= policy_gate
        and float(layer_reports[layer]["token_ranking"]["gain_block_recovery"])
        >= policy_gate
    ]
    if gate_only:
        return {
            "decision": "GO_ONE_KEEP_GATE_CAPACITY",
            "selected_layer": gate_only[0],
            "reason": "shallow token ranking passed 90% but global policy did not",
        }
    residual = [
        layer
        for layer in shallow
        if float(layer_reports[layer]["token_ranking"]["oracle_gain_recovery"])
        >= token_gate
        and float(layer_reports[layer]["token_ranking"]["gain_block_recovery"])
        >= token_gate
    ]
    if residual:
        return {
            "decision": "GO_ONE_RESIDUAL_GATE_CAPACITY",
            "selected_layer": residual[0],
            "reason": "shallow token information passed 80% but policy did not pass 90%",
        }
    return {
        "decision": "CLOSE_EARLY_TARGET_ROUTE",
        "selected_layer": None,
        "reason": "all layers <=12 missed the 80% token-information gate",
    }


def domain_prompt_balanced(
    sample_ids: Sequence[str],
    domains: Sequence[str],
    values: Sequence[int],
) -> dict[str, float]:
    by_domain: dict[str, tuple[list[str], list[int]]] = {}
    for sample_id, domain, value in zip(sample_ids, domains, values, strict=True):
        ids, lengths = by_domain.setdefault(str(domain), ([], []))
        ids.append(str(sample_id))
        lengths.append(int(value))
    return {
        domain: prompt_balanced(ids, lengths)
        for domain, (ids, lengths) in sorted(by_domain.items())
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R049 depth probe requires CUDA")
    source_metadata, all_records = load_source(args.source_rollout, args.split)
    if Path(str(source_metadata["target"])).resolve() != args.target.resolve():
        raise ValueError("source target provenance differs")
    if Path(str(source_metadata["domino_draft"])).resolve() != args.domino_draft.resolve():
        raise ValueError("source Domino provenance differs")
    selected_ids = set(select_balanced_prompts(all_records, args.max_prompts))
    records = [row for row in all_records if str(row["sample_id"]) in selected_ids]
    if len(selected_ids) != args.max_prompts:
        raise ValueError("source has fewer prompts than requested")

    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    target.requires_grad_(False)
    domino.requires_grad_(False)
    depth = len(target.model.layers)
    layers = parse_layers(args.probe_layers, depth)
    if args.max_prompts != 64 or args.candidate_topk != 64:
        raise ValueError("official R049-A is frozen to 64 prompts and K64")
    if layers != [4, 8, 12, 16, 24, 32, 36]:
        raise ValueError("official R049-A probe depths are frozen")
    if depth not in layers:
        raise ValueError("R049 numerical control requires the final target layer")
    proposal_weight = target.model.embed_tokens.weight
    lm_head_weight = target.lm_head.weight

    sample_ids: list[str] = []
    domains: list[str] = []
    proposals: list[Tensor] = []
    candidates: list[Tensor] = []
    verifier_top1_rows: list[Tensor] = []
    baseline_lengths: list[int] = []
    oracle_lengths: list[int] = []
    teacher_scores: list[Tensor] = []
    depth_scores: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    started = time.perf_counter()

    for index, row in enumerate(records, start=1):
        context = row["context_ids_before_anchor"].long().to(device)[None]
        prefix_length = int(context.shape[1])
        prefix = target.model(context, use_cache=True, return_dict=True)
        prefix_cache = prefix.past_key_values
        anchor = torch.tensor(
            [int(row["anchor_token_id"])], dtype=torch.long, device=device
        )
        hidden = row["parallel_hidden"].to(device, torch.bfloat16)[None]
        proposal_output = fast_candidate_domino_decode(
            domino=domino,
            target_weight=proposal_weight,
            anchors=anchor,
            hidden=hidden,
            candidate_topk=args.candidate_topk,
        )
        support = candidate_union_with_proposal(
            proposal_output.candidate_ids,
            proposal_output.token_ids,
            support_size=args.candidate_topk,
        )
        verifier_ids = torch.cat([anchor[:, None], proposal_output.token_ids], dim=1)

        captured: dict[int, Tensor] = {}
        handles = []
        for layer in layers:
            def capture(_module: Any, _inputs: Any, output: Tensor, *, layer: int = layer) -> None:
                captured[layer] = output.detach()

            handles.append(target.model.layers[layer - 1].register_forward_hook(capture))
        teacher_cache = clone_dynamic_cache(prefix_cache, config=target.config)
        try:
            verifier = target(
                verifier_ids,
                past_key_values=teacher_cache,
                use_cache=True,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        if sorted(captured) != layers:
            raise RuntimeError("target hooks did not capture every requested depth")

        full_logits = verifier.logits[:, :16].float()
        target_top1 = full_logits.argmax(dim=-1)
        accepted, _, repair = authoritative_frontier_contract(
            proposal_output.token_ids,
            target_top1,
            support,
        )
        if bool(repair.repair_available[0]):
            repaired_ids = torch.cat([anchor[:, None], repair.token_ids], dim=1)
            repair_cache = clone_dynamic_cache(prefix_cache, config=target.config)
            repaired_verifier = target(
                repaired_ids,
                past_key_values=repair_cache,
                use_cache=True,
                return_dict=True,
            )
            repaired_top1 = repaired_verifier.logits[:, :16].float().argmax(dim=-1)
            oracle_accepted = accepted_lengths(repair.token_ids, repaired_top1)
            if int(oracle_accepted[0]) <= int(accepted[0]):
                raise RuntimeError("available authoritative repair did not advance")
        else:
            oracle_accepted = accepted

        basis = F.embedding(support, lm_head_weight)
        teacher_candidate = full_logits.gather(-1, support)
        teacher_scores.append(
            centered_candidate_scores(
                teacher_candidate,
                support,
                proposal_output.token_ids,
            )[0].cpu()
        )
        for layer in layers:
            hidden_at_layer = captured[layer][:, :16]
            normalized = target.model.norm(hidden_at_layer)
            scores = torch.einsum("blh,blkh->blk", normalized, basis)
            depth_scores[layer].append(
                centered_candidate_scores(
                    scores,
                    support,
                    proposal_output.token_ids,
                )[0].cpu()
            )

        sample_ids.append(str(row["sample_id"]))
        domains.append(str(row["domain"]))
        proposals.append(proposal_output.token_ids[0].cpu())
        candidates.append(support[0].cpu())
        verifier_top1_rows.append(target_top1[0].cpu())
        baseline_lengths.append(int(accepted[0]))
        oracle_lengths.append(int(oracle_accepted[0]))
        if index % 32 == 0 or index == len(records):
            print(f"probed {index}/{len(records)}", flush=True)

    proposal_tensor = torch.stack(proposals).long()
    candidate_tensor = torch.stack(candidates).long()
    verifier_tensor = torch.stack(verifier_top1_rows).long()
    baseline_tensor = torch.tensor(baseline_lengths, dtype=torch.long)
    oracle_tensor = torch.tensor(oracle_lengths, dtype=torch.long)
    teacher_tensor = torch.stack(teacher_scores).float()
    baseline_eal = prompt_balanced(sample_ids, baseline_lengths)
    oracle_eal = prompt_balanced(sample_ids, oracle_lengths)
    oracle_gain = oracle_eal - baseline_eal
    if oracle_gain <= 0:
        raise RuntimeError("R049-A requires a positive exact one-repair oracle gap")

    teacher_policy = select_zero_harm_threshold(
        sample_ids=sample_ids,
        proposal=proposal_tensor,
        verifier_top1=verifier_tensor,
        candidate_ids=candidate_tensor,
        adjusted_scores=teacher_tensor,
        baseline_lengths=baseline_tensor,
        oracle_lengths=oracle_tensor,
    )
    axes = torch.arange(16).view(1, -1)
    protected = axes.lt(baseline_tensor[:, None])
    valid = axes.le(baseline_tensor[:, None])
    repairable = oracle_tensor.gt(baseline_tensor)
    safe_frontier = baseline_tensor.clamp_max(15)
    batch_axis = torch.arange(len(records))

    # The candidate-only gather-dot and the full-vocabulary GEMM may differ by
    # a few BF16 ulps.  Use the measured L36 centered discrepancy to define a
    # conservative ambiguity band.  Such rows remain in the oracle denominator
    # but are forced to KEEP for every deployable probe policy.
    final_scores_raw = torch.stack(depth_scores[depth]).float()
    centered_delta = (final_scores_raw - teacher_tensor).abs()
    epsilon = float(centered_delta[valid].max())
    teacher_top2 = teacher_tensor.topk(2, dim=-1).values
    teacher_margin = teacher_top2[..., 0] - teacher_top2[..., 1]
    teacher_ambiguous = valid & teacher_margin.le(2.0 * epsilon)
    conservative_teacher = force_keep_rows(
        teacher_tensor,
        candidate_tensor,
        proposal_tensor,
        teacher_ambiguous,
    )
    conservative_teacher_policy = select_zero_harm_threshold(
        sample_ids=sample_ids,
        proposal=proposal_tensor,
        verifier_top1=verifier_tensor,
        candidate_ids=candidate_tensor,
        adjusted_scores=conservative_teacher,
        baseline_lengths=baseline_tensor,
        oracle_lengths=oracle_tensor,
    )
    ambiguous_frontier = (
        repairable & teacher_ambiguous[batch_axis, safe_frontier]
    )
    oracle_gain_blocks = int(repairable.sum())
    oracle_gain_tokens = int((oracle_tensor - baseline_tensor).sum())
    stable_oracle_gain_blocks = oracle_gain_blocks - int(ambiguous_frontier.sum())
    stable_oracle_gain_tokens = oracle_gain_tokens - int(
        ((oracle_tensor - baseline_tensor) * ambiguous_frontier.long()).sum()
    )
    conservative_lengths = torch.tensor(
        conservative_teacher_policy["lengths"], dtype=torch.long
    )
    conservative_gain_blocks = int(conservative_lengths.gt(baseline_tensor).sum())

    layer_reports: dict[int, dict[str, Any]] = {}

    for layer in layers:
        scores_raw = torch.stack(depth_scores[layer]).float()
        layer_top2 = scores_raw.topk(2, dim=-1).values
        # This is the deployable fail-closed rule: a layer uses its own
        # candidate margin and the fixed measured numerical tolerance.  The
        # authoritative target ambiguity mask is reserved for L36 control and
        # is never exposed to an earlier-layer policy.
        layer_ambiguous = valid & (layer_top2[..., 0] - layer_top2[..., 1]).le(
            2.0 * epsilon
        )
        scores = force_keep_rows(
            scores_raw,
            candidate_tensor,
            proposal_tensor,
            layer_ambiguous,
        )
        best_index = scores.argmax(dim=-1)
        best_token = candidate_tensor.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
        token_lengths = frontier_token_ranking_lengths(
            proposal=proposal_tensor,
            verifier_top1=verifier_tensor,
            candidate_ids=candidate_tensor,
            candidate_scores=scores,
            baseline_lengths=baseline_tensor,
            oracle_lengths=oracle_tensor,
        )
        token_values = [int(value) for value in token_lengths]
        token_eal = prompt_balanced(sample_ids, token_values)
        policy = select_zero_harm_threshold(
            sample_ids=sample_ids,
            proposal=proposal_tensor,
            verifier_top1=verifier_tensor,
            candidate_ids=candidate_tensor,
            adjusted_scores=scores,
            baseline_lengths=baseline_tensor,
            oracle_lengths=oracle_tensor,
        )
        policy_lengths = torch.tensor(policy["lengths"], dtype=torch.long)
        policy_gain_blocks = int(policy_lengths.gt(baseline_tensor).sum())
        policy["gain_blocks"] = policy_gain_blocks
        policy["gain_block_recovery"] = (
            policy_gain_blocks / oracle_gain_blocks if oracle_gain_blocks else 1.0
        )
        policy["no_effect_blocks"] = (
            int(policy["changed_blocks"])
            - policy_gain_blocks
            - int(policy["harmful_blocks"])
        )
        frontier_predictions = best_token[batch_axis, safe_frontier]
        frontier_targets = verifier_tensor[batch_axis, safe_frontier]
        token_gain_blocks = int(token_lengths.gt(baseline_tensor).sum())
        layer_reports[layer] = {
            "token_ranking": {
                "eal_prompt_balanced": token_eal,
                "oracle_gain_recovery": (
                    (token_eal - baseline_eal) / oracle_gain
                    if oracle_gain > 0
                    else 1.0
                ),
                "gain_blocks": token_gain_blocks,
                "gain_block_recovery": (
                    token_gain_blocks / oracle_gain_blocks
                    if oracle_gain_blocks
                    else 1.0
                ),
                "gain_tokens": int((token_lengths - baseline_tensor).sum()),
                "frontier_correct_blocks": int(
                    (repairable & frontier_predictions.eq(frontier_targets)).sum()
                ),
                "repairable_frontier_blocks": int(repairable.sum()),
                "domain_eal": domain_prompt_balanced(sample_ids, domains, token_values),
            },
            "policy": {
                key: value
                for key, value in policy.items()
                if key != "lengths"
            },
            "policy_domain_eal": domain_prompt_balanced(
                sample_ids,
                domains,
                [int(value) for value in policy["lengths"]],
            ),
            "protected_candidate_argmax_errors": int(
                (protected & best_token.ne(proposal_tensor)).sum()
            ),
            "protected_rows": int(protected.sum()),
            "self_margin_ambiguous_valid_rows_forced_keep": int(
                layer_ambiguous.sum()
            ),
        }

    stable = valid & ~teacher_ambiguous
    final_argmax = final_scores_raw.argmax(dim=-1)
    teacher_argmax = teacher_tensor.argmax(dim=-1)
    parity_matches = stable & final_argmax.eq(teacher_argmax)
    parity_rows = int(stable.sum())
    parity_count = int(parity_matches.sum())
    parity = {
        "layer": depth,
        "valid_numerically_stable_rows": parity_rows,
        "matching_candidate_argmax_rows": parity_count,
        "candidate_argmax_match_rate": parity_count / max(parity_rows, 1),
        "maximum_centered_score_absolute_delta": epsilon,
        "ambiguity_margin_rule": "authoritative candidate top1 margin <= 2 * epsilon",
        "ambiguous_valid_rows_forced_keep": int(teacher_ambiguous.sum()),
    }
    controls_passed = (
        parity_rows > 0
        and parity_count == parity_rows
        and int(conservative_teacher_policy["harmful_blocks"]) == 0
        and int(conservative_teacher_policy["gained_tokens"])
        == stable_oracle_gain_tokens
        and conservative_gain_blocks == stable_oracle_gain_blocks
    )
    gate = depth_gate_decision(layer_reports) if controls_passed else {
        "decision": "STOP_INVALID_NUMERICAL_CONTROL",
        "selected_layer": None,
        "reason": "L36 parity or authoritative teacher policy control failed",
    }
    report = {
        "status": "completed" if controls_passed else "invalid_control",
        "format": "r049_depth_probe_v1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "source_rollout": str(args.source_rollout.resolve()),
        "target": str(args.target.resolve()),
        "domino_draft": str(args.domino_draft.resolve()),
        "prompts": len(selected_ids),
        "blocks": len(records),
        "candidate_topk": args.candidate_topk,
        "probe_layers": layers,
        "baseline_eal_prompt_balanced": baseline_eal,
        "oracle_eal_prompt_balanced": oracle_eal,
        "oracle_gain": oracle_gain,
        "oracle_gain_blocks": oracle_gain_blocks,
        "oracle_gain_tokens": oracle_gain_tokens,
        "numerical_ambiguity": {
            "epsilon": epsilon,
            "valid_rows_forced_keep": int(teacher_ambiguous.sum()),
            "repairable_frontier_blocks_forced_keep": int(ambiguous_frontier.sum()),
            "oracle_gain_tokens_forced_keep": int(
                ((oracle_tensor - baseline_tensor) * ambiguous_frontier.long()).sum()
            ),
        },
        "teacher_policy_control": {
            key: value for key, value in teacher_policy.items() if key != "lengths"
        },
        "conservative_teacher_policy": {
            key: value
            for key, value in conservative_teacher_policy.items()
            if key != "lengths"
        }
        | {
            "stable_oracle_gain_blocks": stable_oracle_gain_blocks,
            "stable_oracle_gain_tokens": stable_oracle_gain_tokens,
            "gain_blocks": conservative_gain_blocks,
            "stable_gain_block_recovery": (
                conservative_gain_blocks / stable_oracle_gain_blocks
                if stable_oracle_gain_blocks
                else 1.0
            ),
            "stable_gain_token_recovery": (
                int(conservative_teacher_policy["gained_tokens"])
                / stable_oracle_gain_tokens
                if stable_oracle_gain_tokens
                else 1.0
            ),
        },
        "layer36_numerical_control": parity,
        "layers": {str(layer): layer_reports[layer] for layer in layers},
        "gate": gate,
        "controls_passed": controls_passed,
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not controls_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
