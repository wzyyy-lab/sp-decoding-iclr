#!/usr/bin/env python3
"""Collect exact all-16 Domino blocks at fixed or policy-induced anchors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM

from collect_canonical_blocks import extract_context_feature
from materialize_domino_same_anchor import (
    MinimalShardWriter,
    load_records,
    validate_domino_contract,
)
from sph.data import validate_stored_canonical_contexts
from sph.gfpr import accepted_lengths, all_position_onpolicy_decode, load_adaptation
from sph.gfpr_candidate import select_anchor_early_exit_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("dynamic", "fixed"), required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        help="Explicit split list; sealed validation_gate is rejected by default.",
    )
    parser.add_argument("--allow-sealed-validation-gate", action="store_true")
    parser.add_argument("--policy-version", default="released-v0")
    parser.add_argument("--adaptation", type=Path)
    parser.add_argument("--position-zero-scale", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument(
        "--store-target-boundary-feature",
        action="store_true",
        help="Store the last verified target multi-layer state before the anchor.",
    )
    parser.add_argument(
        "--store-anchor-early-exit-layer",
        type=int,
        help=(
            "Store the current anchor hidden state after this many target "
            "decoder layers (for example 4)."
        ),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-shards", type=int, default=1)
    parser.add_argument("--sample-shard-index", type=int, default=0)
    parser.add_argument("--shard-blocks", type=int, default=256)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def _prompt_balanced(values: dict[str, list[int]]) -> float:
    if not values:
        raise ValueError("prompt-balanced metric requires records")
    return sum(sum(group) / len(group) for group in values.values()) / len(values)


def _select_sample_ids(
    grouped: dict[str, list[dict[str, Any]]], args: argparse.Namespace
) -> list[str]:
    if args.sample_shards < 1:
        raise ValueError("sample-shards must be positive")
    if not 0 <= args.sample_shard_index < args.sample_shards:
        raise ValueError("sample-shard-index lies outside sample-shards")
    sample_ids = sorted(grouped)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("max-samples must be positive")
        sample_ids = sample_ids[: args.max_samples]
    return sample_ids[args.sample_shard_index :: args.sample_shards]


def _reconstruct_sequence(
    records: list[dict[str, Any]], sample_id: str, device: torch.device
) -> tuple[Tensor, int]:
    longest_context = validate_stored_canonical_contexts(records, sample_id)
    last = max(records, key=lambda row: int(row["context_length"]))
    # validate_stored_canonical_contexts returns the longest stored context and
    # proves that every shorter context is its exact prefix.  Append the last
    # anchor and its 15 canonical gold tokens to recover the full continuation.
    if not torch.equal(longest_context, last["context_ids_before_anchor"].long()):
        raise ValueError(f"longest context mismatch for {sample_id}")
    sequence = torch.cat(
        [
            longest_context.long(),
            torch.tensor([int(last["anchor_token_id"])], dtype=torch.long),
            last["gold_ids"].long(),
        ]
    ).to(device)
    prompt_lengths = {int(record["prompt_token_count"]) for record in records}
    if len(prompt_lengths) != 1:
        raise ValueError(f"prompt-token count changes within {sample_id}")
    return sequence, prompt_lengths.pop()


def _fixed_offsets(records: list[dict[str, Any]]) -> list[int]:
    offsets = sorted({int(record["anchor_offset"]) for record in records})
    if len(offsets) != len(records):
        raise ValueError("fixed source contains duplicate anchors")
    return offsets


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GFPR collection requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    if args.topk < 16:
        raise ValueError("Gate A requires at least the DFlash Top-16")
    if (
        args.store_anchor_early_exit_layer is not None
        and args.store_anchor_early_exit_layer < 1
    ):
        raise ValueError("anchor early-exit layer must be positive")
    if "validation_gate" in args.splits and not args.allow_sealed_validation_gate:
        raise ValueError(
            "validation_gate is sealed until method freeze; pass the explicit "
            "final-evaluation override only after freezing the method"
        )
    if args.adaptation is not None and args.policy_version == "released-v0":
        raise ValueError(
            "an adapted rollout requires an explicit non-released --policy-version"
        )

    source_metadata, source_records = load_records(
        args.canonical, set(args.splits)
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        grouped[str(record["sample_id"])].append(record)
    sample_ids = _select_sample_ids(grouped, args)
    if not sample_ids:
        raise ValueError("sample shard selected no prompts")

    work = args.output.with_name(
        f"{args.output.name}.incomplete-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )
    if work.exists():
        raise FileExistsError(f"refusing to reuse incomplete output {work}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    target_layers = int(target.config.num_hidden_layers)
    if (
        args.store_anchor_early_exit_layer is not None
        and args.store_anchor_early_exit_layer > target_layers
    ):
        raise ValueError(
            f"anchor early-exit layer exceeds target depth {target_layers}"
        )
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    if int(domino.block_size) != 16:
        raise ValueError("GFPR currently requires a B16 Domino checkpoint")
    validate_domino_contract(domino, 15)
    position_zero_scale = torch.tensor(args.position_zero_scale, device=device)
    if args.adaptation is not None:
        position_zero_scale = load_adaptation(
            domino,
            args.adaptation,
            map_location=device,
            expected_target=args.target,
            expected_base_domino=args.domino_draft,
        ).to(device)
    target_weight = target.model.embed_tokens.weight
    if target.lm_head.weight.shape != target_weight.shape:
        raise ValueError("target LM head and embedding shapes differ")
    target_boundary_width = len(domino.target_layer_ids) * int(
        target_weight.shape[1]
    )

    writer = MinimalShardWriter(work, args.shard_blocks)
    counts: Counter[str] = Counter()
    accepted_by_split_prompt: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    full_accepts = 0
    target_teacher_positions = 0
    target_teacher_top1_mismatches = 0
    started = time.perf_counter()

    for sample_index, sample_id in enumerate(sample_ids):
        records = sorted(
            grouped[sample_id], key=lambda row: int(row["anchor_offset"])
        )
        sequence, prompt_tokens = _reconstruct_sequence(records, sample_id, device)
        target_outputs = target.model(
            sequence.unsqueeze(0),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        context_features = extract_context_feature(
            target_outputs.hidden_states, list(domino.target_layer_ids)
        )
        # The canonical record ends with 15 gold tokens.  One extra target
        # argmax supplies the sixteenth runtime label at the final fixed anchor.
        next_token = target.lm_head(
            target_outputs.last_hidden_state[:, -1:]
        ).float().argmax(dim=-1)[0, 0]
        extended = torch.cat([sequence, next_token.view(1)])
        continuation = extended[prompt_tokens:]

        if args.mode == "fixed":
            pending_offsets = _fixed_offsets(records)
        else:
            pending_offsets = [0]

        seen_offsets: set[int] = set()
        while pending_offsets:
            anchor_offset = pending_offsets.pop(0)
            if anchor_offset in seen_offsets:
                raise RuntimeError(
                    f"policy revisited anchor {anchor_offset} for {sample_id}"
                )
            seen_offsets.add(anchor_offset)
            # A full B16 label requires continuation[o+1:o+17].
            if anchor_offset + 16 >= int(continuation.numel()):
                if args.mode == "fixed":
                    raise ValueError(
                        f"fixed anchor {anchor_offset} lacks 16 gold tokens"
                    )
                break
            context_length = prompt_tokens + anchor_offset
            if context_length >= context_features.shape[1]:
                break
            anchor = continuation[anchor_offset].view(1).long()
            gold = continuation[
                anchor_offset + 1 : anchor_offset + 17
            ].view(1, 16).long()
            block_ids = torch.full(
                (1, 16),
                int(domino.mask_token_id),
                dtype=torch.long,
                device=device,
            )
            block_ids[0, 0] = anchor[0]
            position_ids = torch.arange(
                context_length + 16, device=device
            ).unsqueeze(0)
            hidden = domino(
                target_hidden=context_features[:, :context_length],
                noise_embedding=target.model.embed_tokens(block_ids),
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )
            if hidden.shape[1:] != (16, target_weight.shape[1]):
                raise RuntimeError(
                    f"Domino returned unexpected shape {tuple(hidden.shape)}"
                )
            decoded = all_position_onpolicy_decode(
                domino=domino,
                target_weight=target_weight,
                anchors=anchor,
                hidden=hidden,
                position_zero_scale=position_zero_scale,
                topk=args.topk,
            )
            # Dense target supervision at the same gold prefix as canonical
            # acceptance.  The target hidden at the anchor predicts position
            # zero; the following 15 states predict positions 1..15.  Store
            # only the deployable DFlash lattice and released-Domino action,
            # rather than a 152K-way distribution.
            target_teacher_hidden = target_outputs.last_hidden_state[
                :, context_length : context_length + 16
            ]
            if target_teacher_hidden.shape[1] != 16:
                raise RuntimeError("target teacher slice has fewer than 16 positions")
            target_teacher_logits = target.lm_head(target_teacher_hidden).float()
            target_top1_logits, target_top1_ids = target_teacher_logits.max(
                dim=-1
            )
            target_candidate_logits = target_teacher_logits.gather(
                -1, decoded.base_topk_ids.long()
            )
            target_policy_logits = target_teacher_logits.gather(
                -1, decoded.token_ids.long().unsqueeze(-1)
            ).squeeze(-1)
            target_logsumexp = torch.logsumexp(target_teacher_logits, dim=-1)
            accepted = int(accepted_lengths(decoded.token_ids, gold).item())
            next_offset = anchor_offset + accepted + 1
            bonus_available = anchor_offset + 17 < int(continuation.numel())
            bonus_id = (
                int(continuation[anchor_offset + 17].item())
                if bonus_available
                else -1
            )
            if args.mode == "dynamic" and accepted == 16 and not bonus_available:
                # The current block is still valid for EAL, but there is no
                # stored position-16 bonus with which to construct another cycle.
                next_offset = -1

            source = records[0]
            cached = {
                "sample_id": sample_id,
                "domain": str(source["domain"]),
                "source": str(source["source"]),
                "split": str(source["split"]),
                "mode": args.mode,
                "policy_version": args.policy_version,
                "prompt_token_count": prompt_tokens,
                "anchor_offset": anchor_offset,
                "context_length": context_length,
                "context_ids_before_anchor": extended[:context_length]
                .cpu()
                .to(torch.int32),
                "anchor_token_id": int(anchor.item()),
                "gold_ids": gold[0].cpu().to(torch.int32),
                "bonus_token_id": bonus_id,
                "parallel_hidden": hidden[0].cpu().to(torch.bfloat16),
                "base_topk_ids": decoded.base_topk_ids[0]
                .cpu()
                .to(torch.int32),
                "base_topk_logits": decoded.base_topk_logits[0]
                .cpu()
                .to(torch.float16),
                "target_candidate_logits": target_candidate_logits[0]
                .cpu()
                .to(torch.float32),
                "target_candidate_advantages": (
                    target_candidate_logits[0]
                    - target_policy_logits[0].unsqueeze(-1)
                ).cpu().to(torch.float32),
                "target_policy_logits": target_policy_logits[0]
                .cpu()
                .to(torch.float32),
                "target_top1_ids": target_top1_ids[0]
                .cpu()
                .to(torch.int32),
                "target_top1_logits": target_top1_logits[0]
                .cpu()
                .to(torch.float16),
                "target_logsumexp": target_logsumexp[0]
                .cpu()
                .to(torch.float32),
                "policy_ids": decoded.token_ids[0].cpu().to(torch.int32),
                "accepted_length": accepted,
                "next_anchor_offset": next_offset,
            }
            if args.store_target_boundary_feature:
                if context_length < 1:
                    raise RuntimeError("target boundary requires non-empty context")
                cached["target_boundary_feature"] = context_features[
                    0, context_length - 1
                ].cpu().to(torch.bfloat16)
            if args.store_anchor_early_exit_layer is not None:
                cached["target_anchor_early_feature"] = (
                    select_anchor_early_exit_feature(
                        target_outputs.hidden_states,
                        context_length=context_length,
                        early_layers=args.store_anchor_early_exit_layer,
                    )
                    .cpu()
                    .to(torch.bfloat16)
                )
            # Stored semantics must agree before the record leaves the GPU loop.
            if accepted < 16:
                if not torch.equal(decoded.token_ids[0, :accepted], gold[0, :accepted]):
                    raise RuntimeError("accepted prefix does not equal target labels")
                if decoded.token_ids[0, accepted] == gold[0, accepted]:
                    raise RuntimeError("stored first mismatch is not a mismatch")
            elif not torch.equal(decoded.token_ids, gold):
                raise RuntimeError("full-accept record contains a mismatch")
            writer.add(cached)
            split = str(source["split"])
            domain = str(source["domain"])
            counts[f"{domain}/{split}"] += 1
            accepted_by_split_prompt[split][sample_id].append(accepted)
            full_accepts += int(accepted == 16)
            target_teacher_positions += int(gold.numel())
            target_teacher_top1_mismatches += int(
                target_top1_ids.ne(gold).sum().item()
            )

            if args.mode == "dynamic":
                if next_offset < 0 or next_offset + 16 >= int(continuation.numel()):
                    break
                pending_offsets.append(next_offset)

            del hidden, decoded, target_teacher_logits

        del target_outputs, context_features, sequence, extended, continuation
        if (sample_index + 1) % 10 == 0 or sample_index + 1 == len(sample_ids):
            print(
                f"[{sample_index + 1}/{len(sample_ids)}] blocks="
                f"{writer.total + len(writer.buffer)}",
                flush=True,
            )

    writer.flush()
    metadata = {
        "format": "gfpr_rollout_v1",
        "collection_complete": True,
        "mode": args.mode,
        "policy_version": args.policy_version,
        "position_zero_scale": float(position_zero_scale.item()),
        "adaptation": str(args.adaptation.resolve()) if args.adaptation else None,
        "source_canonical": str(args.canonical.resolve()),
        "target": str(args.target.resolve()),
        "domino_draft": str(args.domino_draft.resolve()),
        "source_format": source_metadata.get("format_version", source_metadata.get("format")),
        "sample_shards": args.sample_shards,
        "sample_shard_index": args.sample_shard_index,
        "samples": len(sample_ids),
        "blocks": writer.total,
        "full_accept_blocks": full_accepts,
        "target_teacher": {
            "support": "dflash_topk_plus_released_action",
            "prefix": "canonical_gold",
            "positions": target_teacher_positions,
            "top1_mismatches_vs_canonical_gold": target_teacher_top1_mismatches,
            "top1_match_fraction": 1.0
            - target_teacher_top1_mismatches / max(1, target_teacher_positions),
        },
        "target_boundary_feature": {
            "stored": args.store_target_boundary_feature,
            "prefix_position": "immediately_before_anchor",
            "width": target_boundary_width
            if args.store_target_boundary_feature else 0,
        },
        "target_anchor_early_exit_feature": {
            "stored": args.store_anchor_early_exit_layer is not None,
            "prefix_position": "current_anchor",
            "early_layers": args.store_anchor_early_exit_layer,
            "width": int(target_weight.shape[1])
            if args.store_anchor_early_exit_layer is not None
            else 0,
        },
        "counts_by_domain_split": dict(sorted(counts.items())),
        "prompt_balanced_eal_by_split": {
            split: _prompt_balanced(values)
            for split, values in sorted(accepted_by_split_prompt.items())
        },
        "seconds": time.perf_counter() - started,
        "shards": writer.shards,
    }
    (work / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(work, args.output)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
