#!/usr/bin/env python3
"""Compare cached full-replay anchor features with incremental target KV use."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from sph.gfpr import accepted_lengths, paired_prompt_summary
from sph.gfpr_candidate import (
    GFPRCandidateHead,
    select_anchor_early_exit_feature,
)
from train_gfpr_head import load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--early-layers", type=int, default=4)
    parser.add_argument("--max-blocks", type=int, default=24)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--domino-draft", type=Path)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    parser.add_argument("--maximum-relative-rms", type=float, default=0.05)
    parser.add_argument("--maximum-path-change-fraction", type=float, default=0.0)
    parser.add_argument("--maximum-eal-absolute-delta", type=float, default=0.02)
    parser.add_argument("--fail-on-material-delta", action="store_true")
    return parser.parse_args()


def _balanced_records(
    records: list[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("max-blocks must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["domain"])].append(record)
    result: list[dict[str, Any]] = []
    domains = sorted(grouped)
    while len(result) < maximum:
        changed = False
        for domain in domains:
            if grouped[domain]:
                result.append(grouped[domain].pop(0))
                changed = True
                if len(result) == maximum:
                    break
        if not changed:
            break
    return result


def _feature_metrics(reference: torch.Tensor, current: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    current = current.float()
    difference = current - reference
    reference_rms = reference.square().mean().sqrt().clamp_min(1e-12)
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference.unsqueeze(0), current.unsqueeze(0)
            ).item()
        ),
        "relative_rms": float(difference.square().mean().sqrt() / reference_rms),
        "max_absolute": float(difference.abs().max()),
    }


def _summarize_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "mean_cosine": sum(row["cosine"] for row in rows) / len(rows),
        "minimum_cosine": min(row["cosine"] for row in rows),
        "mean_relative_rms": sum(row["relative_rms"] for row in rows) / len(rows),
        "maximum_relative_rms": max(row["relative_rms"] for row in rows),
        "maximum_absolute": max(row["max_absolute"] for row in rows),
    }


def _load_early_target(path: Path, layers: int) -> nn.Module:
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    full_depth = int(config.num_hidden_layers)
    if not 1 <= layers <= full_depth:
        raise ValueError(f"early layer count must be in [1, {full_depth}]")
    config.num_hidden_layers = layers
    target = AutoModelForCausalLM.from_pretrained(
        str(path),
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    # A full target exposes hidden_states[layers] before the next decoder
    # layer.  A truncated model would apply the final RMSNorm at this point, so
    # remove it to recover the same intermediate representation.
    target.model.norm = nn.Identity()
    target.requires_grad_(False)
    return target


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("anchor alignment check requires CUDA")
    metadata, all_records = load_records(args.collection, args.split)
    if Path(str(metadata.get("target", ""))).resolve() != args.target.resolve():
        raise ValueError("collection target provenance differs from --target")
    contract = metadata.get("target_anchor_early_exit_feature", {})
    if not contract.get("stored", False):
        raise ValueError("collection lacks the anchor early-exit feature")
    if int(contract.get("early_layers", -1)) != args.early_layers:
        raise ValueError("collection and requested early layer differ")
    records = _balanced_records(all_records, args.max_blocks)
    if not records:
        raise ValueError("alignment selection is empty")

    target = _load_early_target(args.target, args.early_layers)
    stored_incremental_rows: list[dict[str, float]] = []
    local_incremental_rows: list[dict[str, float]] = []
    stored_local_rows: list[dict[str, float]] = []
    incremental_features: list[torch.Tensor] = []
    local_features: list[torch.Tensor] = []
    for index, record in enumerate(records, start=1):
        context = record["context_ids_before_anchor"].long().to("cuda:0")
        anchor = torch.tensor(
            [[int(record["anchor_token_id"])]],
            dtype=torch.long,
            device="cuda:0",
        )
        gold = record["gold_ids"].long().to("cuda:0")
        replay = torch.cat([context, anchor[0], gold], dim=0).unsqueeze(0)
        local_outputs = target.model(
            replay,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        local = select_anchor_early_exit_feature(
            local_outputs.hidden_states,
            context_length=int(context.numel()),
            early_layers=args.early_layers,
        )

        prefill = target.model(
            context.unsqueeze(0),
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        incremental_outputs = target.model(
            anchor,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        incremental = select_anchor_early_exit_feature(
            incremental_outputs.hidden_states,
            context_length=0,
            early_layers=args.early_layers,
        )
        stored = record["target_anchor_early_feature"].to(
            "cuda:0", torch.bfloat16
        )
        stored_incremental_rows.append(_feature_metrics(stored, incremental))
        local_incremental_rows.append(_feature_metrics(local, incremental))
        stored_local_rows.append(_feature_metrics(stored, local))
        incremental_features.append(incremental.cpu().to(torch.bfloat16))
        local_features.append(local.cpu().to(torch.bfloat16))
        print(f"alignment {index}/{len(records)}", flush=True)

    report: dict[str, Any] = {
        "format": "r047_anchor_alignment_v1",
        "collection": str(args.collection.resolve()),
        "target": str(args.target.resolve()),
        "collection_target": str(Path(metadata["target"]).resolve()),
        "collection_domino": str(Path(metadata["domino_draft"]).resolve()),
        "split": args.split,
        "early_layers": args.early_layers,
        "blocks": len(records),
        "stored_full_prompt_vs_incremental": _summarize_metrics(
            stored_incremental_rows
        ),
        "local_full_replay_vs_incremental": _summarize_metrics(
            local_incremental_rows
        ),
        "stored_full_prompt_vs_local_full_replay": _summarize_metrics(
            stored_local_rows
        ),
    }
    stored_incremental_summary = report["stored_full_prompt_vs_incremental"]
    feature_gate = {
        "minimum_cosine_at_least_threshold": (
            stored_incremental_summary["minimum_cosine"] >= args.minimum_cosine
        ),
        "maximum_relative_rms_at_most_threshold": (
            stored_incremental_summary["maximum_relative_rms"]
            <= args.maximum_relative_rms
        ),
    }
    report["thresholds"] = {
        "minimum_cosine": args.minimum_cosine,
        "maximum_relative_rms": args.maximum_relative_rms,
        "maximum_path_change_fraction": args.maximum_path_change_fraction,
        "maximum_eal_absolute_delta": args.maximum_eal_absolute_delta,
    }
    report["gate"] = {
        "feature": feature_gate,
        "checkpoint_path": None,
    }

    if args.checkpoint is not None:
        if args.domino_draft is None:
            raise ValueError("checkpoint comparison requires --domino-draft")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_step = int(checkpoint.get("step", 0))
        if checkpoint_step <= 0:
            raise ValueError("checkpoint alignment requires a trained step>0 checkpoint")
        residual_up = checkpoint.get("state_dict", {}).get("residual_up.weight")
        if residual_up is None or not bool(torch.isfinite(residual_up).all()):
            raise ValueError("checkpoint residual_up is absent or non-finite")
        residual_up_norm = float(residual_up.float().norm())
        if residual_up_norm <= 0:
            raise ValueError("checkpoint residual_up is still the zero fallback")
        provenance = checkpoint.get("provenance", {})
        expected_provenance = {
            "target": args.target.resolve(),
            "base_domino": args.domino_draft.resolve(),
            "eval_rollout": args.collection.resolve(),
        }
        for field, expected_path in expected_provenance.items():
            if Path(str(provenance.get(field, ""))).resolve() != expected_path:
                raise ValueError(
                    f"checkpoint {field} provenance differs from {expected_path}"
                )
        config = checkpoint["config"]
        if config.get("target_context_field") != "target_anchor_early_feature":
            raise ValueError("checkpoint is not an anchor early-feature head")
        domino = AutoModel.from_pretrained(
            str(args.domino_draft),
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda:0",
        ).eval()
        domino.requires_grad_(False)
        target_weight = target.model.embed_tokens.weight
        head = GFPRCandidateHead.from_domino(
            domino,
            target_weight,
            positions=int(config["positions"]),
            candidates=int(config["candidates"]),
            adapter_rank=int(config["adapter_rank"]),
            boundary_width=int(config["boundary_width"]),
        )
        head.load_state_dict(checkpoint["state_dict"], strict=True)
        head.eval()
        stored_lengths: list[int] = []
        incremental_lengths: list[int] = []
        sample_ids: list[str] = []
        token_changes = 0
        path_changes = 0
        for record, incremental in zip(records, incremental_features, strict=True):
            anchors = torch.tensor(
                [int(record["anchor_token_id"])],
                dtype=torch.long,
                device="cuda:0",
            )
            hidden = record["parallel_hidden"].to(
                "cuda:0", torch.bfloat16
            )[None]
            candidates = record["base_topk_ids"].to("cuda:0").long()[None]
            gold = record["gold_ids"].to("cuda:0").long()[None]
            stored_feature = record["target_anchor_early_feature"].to(
                "cuda:0", torch.bfloat16
            )[None]
            incremental_feature = incremental.to("cuda:0")[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                stored_decode = head.decode_with_released_union(
                    anchors=anchors,
                    hidden=hidden,
                    base_candidate_ids=candidates,
                    target_boundary=stored_feature,
                ).token_ids
                incremental_decode = head.decode_with_released_union(
                    anchors=anchors,
                    hidden=hidden,
                    base_candidate_ids=candidates,
                    target_boundary=incremental_feature,
                ).token_ids
            stored_lengths.append(int(accepted_lengths(stored_decode, gold).item()))
            incremental_lengths.append(
                int(accepted_lengths(incremental_decode, gold).item())
            )
            token_changes += int(stored_decode.ne(incremental_decode).sum())
            path_changes += int(not torch.equal(stored_decode, incremental_decode))
            sample_ids.append(str(record["sample_id"]))
        paired_acceptance = paired_prompt_summary(
            sample_ids,
            stored_lengths,
            incremental_lengths,
            bootstrap_samples=10_000,
            seed=0,
        )
        path_change_fraction = path_changes / len(records)
        report["checkpoint_path_alignment"] = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": checkpoint_step,
            "checkpoint_residual_up_norm": residual_up_norm,
            "checkpoint_provenance": provenance,
            "tokens_changed": token_changes,
            "blocks_with_any_path_change": path_changes,
            "path_change_fraction": path_change_fraction,
            "paired_acceptance": paired_acceptance,
        }
        report["gate"]["checkpoint_path"] = {
            "path_change_fraction_at_most_threshold": (
                path_change_fraction <= args.maximum_path_change_fraction
            ),
            "eal_absolute_delta_at_most_threshold": (
                abs(float(paired_acceptance["paired_delta"]))
                <= args.maximum_eal_absolute_delta
            ),
        }

    gate_values = list(feature_gate.values())
    checkpoint_gate = report["gate"]["checkpoint_path"]
    if checkpoint_gate is not None:
        gate_values.extend(checkpoint_gate.values())
    report["gate"]["passed"] = all(gate_values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.fail_on_material_delta and not report["gate"]["passed"]:
        raise RuntimeError("anchor cached-replay/incremental alignment gate failed")


if __name__ == "__main__":
    main()
