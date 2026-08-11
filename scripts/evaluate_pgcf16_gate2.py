#!/usr/bin/env python3
"""Evaluate disjoint PGCF-16 global use and the offline remote intervention.

The deployed method remains one full16, non-causal, single-chain head call.
The 16-pass construction in this file is diagnostic-only: each pass preserves
one recipient position, replaces all other complete lattice triplets with one
label-independent donor block, and retains only the preserved-position score.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from sph.parallel_global_candidate_fusion import (
    BLOCK_LENGTH,
    CANDIDATES,
    DEFAULT_PARAMETER_COUNT,
    MatchedLocalCandidateFusionHead,
    ParallelGlobalCandidateFusionHead,
)
from train_pgcf16 import (
    accepted_lengths,
    collate_records,
    load_rollout,
    load_target_embedding,
    make_loader,
    move_tensors,
    prompt_balanced,
)


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_810
MIN_GLOBAL_LOCAL_DELTA = 0.15
MIN_REMOTE_ERASURE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--eval-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-split", default="validation_select")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--remote-batch-size", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def block_identity(record: dict[str, Any]) -> str:
    return f"{record['sample_id']}@{int(record['anchor_offset'])}"


def context_quartile_boundaries(records: list[dict[str, Any]]) -> tuple[float, ...]:
    if not records:
        raise ValueError("cannot define quartiles for an empty collection")
    lengths = torch.tensor(
        [int(record["context_length"]) for record in records],
        dtype=torch.float64,
    )
    boundaries = torch.quantile(
        lengths, torch.tensor([0.25, 0.50, 0.75], dtype=torch.float64)
    )
    return tuple(float(value) for value in boundaries)


def context_quartile(length: int, boundaries: tuple[float, ...]) -> int:
    if len(boundaries) != 3:
        raise ValueError("exactly three context quartile boundaries are required")
    return sum(float(length) > boundary for boundary in boundaries)


def build_donor_map(
    records: list[dict[str, Any]],
) -> tuple[list[int], dict[str, Any]]:
    """Build a deterministic, label-independent within-cell derangement."""

    identities = [block_identity(record) for record in records]
    if len(identities) != len(set(identities)):
        raise RuntimeError("rollout block identities are not unique")
    boundaries = context_quartile_boundaries(records)
    cells: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        cell = (
            str(record["domain"]),
            context_quartile(int(record["context_length"]), boundaries),
        )
        cells[cell].append(index)

    donor_indices = [-1] * len(records)
    cell_report: dict[str, Any] = {}
    for (domain, quartile), indices in sorted(cells.items()):
        ordered = sorted(
            indices,
            key=lambda index: (
                str(records[index]["sample_id"]),
                int(records[index]["anchor_offset"]),
                int(records[index]["context_length"]),
                index,
            ),
        )
        sample_ids = [str(records[index]["sample_id"]) for index in ordered]
        selected_shift = None
        for shift in range(1, len(ordered)):
            if all(
                sample_ids[position] != sample_ids[(position + shift) % len(ordered)]
                for position in range(len(ordered))
            ):
                selected_shift = shift
                break
        if selected_shift is None:
            counts = Counter(sample_ids)
            raise RuntimeError(
                "cannot construct a cross-prompt donor derangement for "
                f"domain={domain}, quartile={quartile}, counts={dict(counts)}"
            )
        for position, recipient_index in enumerate(ordered):
            donor_indices[recipient_index] = ordered[
                (position + selected_shift) % len(ordered)
            ]
        cell_report[f"{domain}/q{quartile}"] = {
            "blocks": len(ordered),
            "prompts": len(set(sample_ids)),
            "circular_shift": selected_shift,
            "max_blocks_per_prompt": max(Counter(sample_ids).values()),
        }

    for recipient_index, donor_index in enumerate(donor_indices):
        if donor_index < 0 or donor_index == recipient_index:
            raise RuntimeError("donor map is not a complete row derangement")
        recipient = records[recipient_index]
        donor = records[donor_index]
        if str(recipient["sample_id"]) == str(donor["sample_id"]):
            raise RuntimeError("donor and recipient prompts must differ")
        if str(recipient["domain"]) != str(donor["domain"]):
            raise RuntimeError("donor domain mismatch")
        if context_quartile(
            int(recipient["context_length"]), boundaries
        ) != context_quartile(int(donor["context_length"]), boundaries):
            raise RuntimeError("donor context quartile mismatch")

    return donor_indices, {
        "algorithm": (
            "stable (sample_id,anchor_offset,context_length,index) order; "
            "smallest circular shift with donor sample_id != recipient sample_id"
        ),
        "label_fields_used": [],
        "context_quartile_boundaries": list(boundaries),
        "cells": cell_report,
    }


def coherent_remote_inputs(
    recipient: dict[str, Tensor], donor: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Create Bx16 intervention inputs with a coherent retained diagonal."""

    tensor_keys = ("hidden", "candidate_ids", "candidate_logits")
    batch = int(recipient["hidden"].shape[0])
    positions = torch.arange(BLOCK_LENGTH, device=recipient["hidden"].device)
    mixed: dict[str, Tensor] = {}
    for key in tensor_keys:
        recipient_value = recipient[key]
        donor_value = donor[key]
        if recipient_value.shape != donor_value.shape:
            raise ValueError(f"recipient/donor {key} shapes differ")
        value = donor_value[:, None].expand(
            batch, BLOCK_LENGTH, *donor_value.shape[1:]
        ).clone()
        value[:, positions, positions] = recipient_value[:, positions]
        mixed[key] = value.reshape(
            batch * BLOCK_LENGTH, *donor_value.shape[1:]
        )
    mixed["anchor_ids"] = recipient["anchor_ids"][:, None].expand(
        batch, BLOCK_LENGTH
    ).reshape(batch * BLOCK_LENGTH)
    return mixed


def _model_from_checkpoint(
    path: Path,
    *,
    expected_head: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("head") != expected_head:
        raise RuntimeError(
            f"checkpoint {path} is not the expected {expected_head} head"
        )
    model_type = (
        ParallelGlobalCandidateFusionHead
        if expected_head == "global"
        else MatchedLocalCandidateFusionHead
    )
    model = model_type(
        hidden_size=2560,
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        ff_multiplier=int(config["ff_multiplier"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != DEFAULT_PARAMETER_COUNT:
        raise RuntimeError("checkpoint does not use the frozen PGCF-16 parameter count")
    return model.to(device).eval(), checkpoint


def assert_matched_checkpoint_configs(
    global_checkpoint: dict[str, Any], local_checkpoint: dict[str, Any]
) -> dict[str, Any]:
    global_config = dict(global_checkpoint["config"])
    local_config = dict(local_checkpoint["config"])
    ignored = {"head", "output"}
    keys = sorted((set(global_config) | set(local_config)) - ignored)
    differences = {
        key: {"global": global_config.get(key), "local": local_config.get(key)}
        for key in keys
        if global_config.get(key) != local_config.get(key)
    }
    if differences:
        raise RuntimeError(
            "global/local training configurations differ beyond head visibility: "
            f"{differences}"
        )
    return {
        "matched": True,
        "ignored_fields": sorted(ignored),
        "compared_fields": keys,
    }


@torch.inference_mode()
def predict_proposals(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    target_embedding: Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> Tensor:
    proposals = []
    loader = make_loader(records, batch_size=batch_size, shuffle=False)
    for cpu_batch in loader:
        batch = move_tensors(cpu_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                batch["candidate_logits"],
                target_embedding[batch["anchor_ids"]],
                candidate_embeddings=target_embedding[batch["candidate_ids"]],
            )
        ranks = output.scores.argmax(dim=-1)
        proposals.append(
            batch["candidate_ids"].gather(-1, ranks.unsqueeze(-1)).squeeze(-1).cpu()
        )
    return torch.cat(proposals, dim=0)


@torch.inference_mode()
def predict_remote_proposals(
    global_model: torch.nn.Module,
    local_model: torch.nn.Module,
    records: list[dict[str, Any]],
    donor_indices: list[int],
    target_embedding: Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Run the diagnostic 16-pass intervention in vectorized batches."""

    proposals = []
    local_decisions = 0
    local_mismatches = 0
    positions = torch.arange(BLOCK_LENGTH, device=device)
    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        recipient_cpu = collate_records(records[start:stop])
        donor_cpu = collate_records(
            [records[donor_indices[index]] for index in range(start, stop)]
        )
        recipient = move_tensors(recipient_cpu, device)
        donor = move_tensors(donor_cpu, device)
        remote = coherent_remote_inputs(recipient, donor)
        self_control = coherent_remote_inputs(recipient, recipient)
        expanded_batch = stop - start

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            global_output = global_model(
                remote["hidden"],
                remote["candidate_logits"],
                target_embedding[remote["anchor_ids"]],
                candidate_embeddings=target_embedding[remote["candidate_ids"]],
            )
            local_remote_output = local_model(
                remote["hidden"],
                remote["candidate_logits"],
                target_embedding[remote["anchor_ids"]],
                candidate_embeddings=target_embedding[remote["candidate_ids"]],
            )
            local_self_output = local_model(
                self_control["hidden"],
                self_control["candidate_logits"],
                target_embedding[self_control["anchor_ids"]],
                candidate_embeddings=target_embedding[
                    self_control["candidate_ids"]
                ],
            )

        global_scores = global_output.scores.view(
            expanded_batch, BLOCK_LENGTH, BLOCK_LENGTH, CANDIDATES
        )[:, positions, positions]
        global_ranks = global_scores.argmax(dim=-1)
        proposal = recipient["candidate_ids"].gather(
            -1, global_ranks.unsqueeze(-1)
        ).squeeze(-1)
        proposals.append(proposal.cpu())

        local_remote_scores = local_remote_output.scores.view(
            expanded_batch, BLOCK_LENGTH, BLOCK_LENGTH, CANDIDATES
        )[:, positions, positions]
        local_self_scores = local_self_output.scores.view(
            expanded_batch, BLOCK_LENGTH, BLOCK_LENGTH, CANDIDATES
        )[:, positions, positions]
        local_remote_tokens = recipient["candidate_ids"].gather(
            -1, local_remote_scores.argmax(dim=-1).unsqueeze(-1)
        ).squeeze(-1)
        local_self_tokens = recipient["candidate_ids"].gather(
            -1, local_self_scores.argmax(dim=-1).unsqueeze(-1)
        ).squeeze(-1)
        local_decisions += int(local_remote_tokens.numel())
        local_mismatches += int(local_remote_tokens.ne(local_self_tokens).sum())

    if local_mismatches:
        raise RuntimeError(
            "matched-local negative control changed under remote intervention: "
            f"{local_mismatches}/{local_decisions} positions"
        )
    return torch.cat(proposals, dim=0), {
        "positions": local_decisions,
        "token_mismatches": local_mismatches,
        "passed": local_mismatches == 0,
    }


def proposal_lengths(proposal: Tensor, records: list[dict[str, Any]]) -> list[float]:
    gold = torch.stack([record["gold_ids"].long() for record in records])
    return accepted_lengths(proposal.long(), gold).float().tolist()


def metric_summary(
    records: list[dict[str, Any]], lengths: list[float]
) -> dict[str, Any]:
    if len(records) != len(lengths):
        raise ValueError("record/length count mismatch")
    sample_ids = [str(record["sample_id"]) for record in records]
    domains = sorted({str(record["domain"]) for record in records})
    per_domain = {}
    for domain in domains:
        indices = [
            index
            for index, record in enumerate(records)
            if str(record["domain"]) == domain
        ]
        per_domain[domain] = prompt_balanced(
            [sample_ids[index] for index in indices],
            [lengths[index] for index in indices],
        )
    return {
        "prompt_balanced_eal": prompt_balanced(sample_ids, lengths),
        "block_mean_eal": sum(lengths) / len(lengths),
        "domains": per_domain,
    }


def paired_prompt_bootstrap(
    records: list[dict[str, Any]],
    first: list[float],
    second: list[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record, first_value, second_value in zip(
        records, first, second, strict=True
    ):
        grouped[str(record["sample_id"])].append(
            float(first_value) - float(second_value)
        )
    prompt_ids = sorted(grouped)
    prompt_deltas = torch.tensor(
        [sum(grouped[prompt_id]) / len(grouped[prompt_id]) for prompt_id in prompt_ids],
        dtype=torch.float64,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    samples = prompt_deltas[
        torch.randint(
            len(prompt_ids),
            (draws, len(prompt_ids)),
            generator=generator,
        )
    ].mean(dim=1)
    lower, upper = torch.quantile(
        samples, torch.tensor([0.025, 0.975], dtype=torch.float64)
    ).tolist()
    return {
        "unit": "prompt_cluster",
        "method": "paired percentile bootstrap",
        "draws": draws,
        "seed": seed,
        "prompts": len(prompt_ids),
        "point_delta": float(prompt_deltas.mean()),
        "ci95": [float(lower), float(upper)],
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.eval_batch_size < 1 or args.remote_batch_size < 1:
        raise ValueError("evaluation batch sizes must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("claim-bearing Gate2 evaluation requires CUDA")

    metadata, records = load_rollout(
        args.eval_rollout,
        split=args.eval_split,
        max_records=0,
    )
    global_model, global_checkpoint = _model_from_checkpoint(
        args.global_checkpoint, expected_head="global", device=device
    )
    local_model, local_checkpoint = _model_from_checkpoint(
        args.local_checkpoint, expected_head="local", device=device
    )
    config_match = assert_matched_checkpoint_configs(
        global_checkpoint, local_checkpoint
    )
    target_embedding = load_target_embedding(args.target).to(
        device=device, dtype=torch.bfloat16
    )

    global_proposal = predict_proposals(
        global_model,
        records,
        target_embedding,
        device,
        batch_size=args.eval_batch_size,
    )
    local_proposal = predict_proposals(
        local_model,
        records,
        target_embedding,
        device,
        batch_size=args.eval_batch_size,
    )
    base_proposal = torch.stack(
        [record["base_topk_ids"][:, 0].long() for record in records]
    )
    released_proposal = torch.stack(
        [record["policy_ids"].long() for record in records]
    )
    gold = torch.stack([record["gold_ids"].long() for record in records])
    candidates = torch.stack(
        [record["base_topk_ids"].long() for record in records]
    )
    gold_matches = candidates.eq(gold.unsqueeze(-1))
    oracle_proposal = torch.where(
        gold_matches.any(dim=-1), gold, base_proposal
    )

    donor_indices, donor_protocol = build_donor_map(records)
    remote_proposal, local_negative_control = predict_remote_proposals(
        global_model,
        local_model,
        records,
        donor_indices,
        target_embedding,
        device,
        batch_size=args.remote_batch_size,
    )

    proposals = {
        "global": global_proposal,
        "local": local_proposal,
        "base": base_proposal,
        "released_domino": released_proposal,
        "base16_oracle": oracle_proposal,
        "remote": remote_proposal,
    }
    lengths = {
        name: proposal_lengths(proposal, records)
        for name, proposal in proposals.items()
    }
    metrics = {
        name: metric_summary(records, values) for name, values in lengths.items()
    }
    global_local_delta = (
        metrics["global"]["prompt_balanced_eal"]
        - metrics["local"]["prompt_balanced_eal"]
    )
    bootstrap = paired_prompt_bootstrap(
        records, lengths["global"], lengths["local"]
    )
    erasure = (
        1.0
        - (
            metrics["remote"]["prompt_balanced_eal"]
            - metrics["local"]["prompt_balanced_eal"]
        )
        / global_local_delta
        if global_local_delta != 0.0
        else None
    )
    domain_non_regression = {
        domain: metrics["global"]["domains"][domain]
        >= metrics["base"]["domains"][domain]
        for domain in metrics["base"]["domains"]
    }
    gates = {
        "global_local_delta_at_least_0_15": global_local_delta
        >= MIN_GLOBAL_LOCAL_DELTA,
        "paired_bootstrap_ci_lower_positive": bootstrap["ci95"][0] > 0.0,
        "global_above_base": metrics["global"]["prompt_balanced_eal"]
        > metrics["base"]["prompt_balanced_eal"],
        "all_domains_global_not_below_base": all(domain_non_regression.values()),
        "remote_erasure_at_least_0_50": erasure is not None
        and erasure >= MIN_REMOTE_ERASURE,
        "local_remote_negative_control": local_negative_control["passed"],
    }

    donor_mapping = []
    per_block = []
    for index, record in enumerate(records):
        donor = records[donor_indices[index]]
        donor_mapping.append(
            {
                "recipient_index": index,
                "recipient": block_identity(record),
                "donor_index": donor_indices[index],
                "donor": block_identity(donor),
                "domain": str(record["domain"]),
                "recipient_context_length": int(record["context_length"]),
                "donor_context_length": int(donor["context_length"]),
                "quartile": context_quartile(
                    int(record["context_length"]),
                    tuple(donor_protocol["context_quartile_boundaries"]),
                ),
            }
        )
        per_block.append(
            {
                "index": index,
                "identity": block_identity(record),
                "sample_id": str(record["sample_id"]),
                "domain": str(record["domain"]),
                "context_length": int(record["context_length"]),
                "lengths": {name: values[index] for name, values in lengths.items()},
                "proposals": {
                    name: proposal[index].tolist() for name, proposal in proposals.items()
                },
            }
        )

    report = {
        "protocol": {
            "method": "PGCF-16 full-block global non-causal one-chain head",
            "online_head_calls": 1,
            "online_output_shape": [16],
            "remote_intervention": "diagnostic_only",
            "remote_head_calls_per_block": 16,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": str(device),
        "metadata": metadata,
        "blocks": len(records),
        "prompts": len({str(record["sample_id"]) for record in records}),
        "global_checkpoint_step": int(global_checkpoint["step"]),
        "local_checkpoint_step": int(local_checkpoint["step"]),
        "checkpoint_config_match": config_match,
        "donor_protocol": donor_protocol,
        "donor_mapping": donor_mapping,
        "local_remote_negative_control": local_negative_control,
        "metrics": metrics,
        "global_local_delta": global_local_delta,
        "paired_prompt_bootstrap": bootstrap,
        "domain_global_not_below_base": domain_non_regression,
        "remote_erasure_raw": erasure,
        "gates": gates,
        "gate2_passed": all(gates.values()),
        "per_block": per_block,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "metrics": metrics,
                "global_local_delta": global_local_delta,
                "paired_prompt_bootstrap": bootstrap,
                "remote_erasure_raw": erasure,
                "gates": gates,
                "gate2_passed": report["gate2_passed"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not report["gate2_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
