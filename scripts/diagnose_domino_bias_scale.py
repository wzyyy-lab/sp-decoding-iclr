#!/usr/bin/env python3
"""Measure whether Domino's causal correction is miscalibrated.

The experiment replays the exact stored phase-3 contexts, runs the frozen
Domino backbone once per anchor, and rolls out several multiplicative scales
of the causal correction head in parallel.  It deliberately performs semantic
context checks but skips shard hashing: the scientific question is whether a
calibrated causal correction can improve accepted prefix length.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModel, AutoModelForCausalLM

from collect_canonical_blocks import extract_context_feature
from sph.candidate_ceiling import accepted_draft_prefix_lengths
from sph.data import validate_stored_canonical_contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--matched-horizon", type=int, default=15)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def scale_key(scale: float) -> str:
    rendered = f"{scale:.6f}".rstrip("0").rstrip(".")
    rendered = rendered.replace("-", "m").replace(".", "p")
    return f"domino_scale_{rendered}"


def normalize_scales(scales: Iterable[float]) -> list[float]:
    normalized: list[float] = []
    for scale in scales:
        value = float(scale)
        if value < 0.0:
            raise ValueError("correction scales must be nonnegative")
        if value not in normalized:
            normalized.append(value)
    if 1.0 not in normalized:
        raise ValueError("--scales must include the released Domino scale 1.0")
    return normalized


def load_split_records(
    root: Path, split: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("collection_complete") is False:
        raise RuntimeError(f"canonical collection is incomplete: {root}")
    records: list[dict[str, Any]] = []
    shard_paths = sorted(root.glob("shard-*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no canonical shards found under {root}")
    for shard in shard_paths:
        shard_records = torch.load(shard, map_location="cpu", weights_only=False)
        records.extend(
            record for record in shard_records if str(record.get("split")) == split
        )
    if not records:
        raise ValueError(f"no canonical records matched split={split!r}")
    return metadata, records


def accepted_length(proposal: torch.Tensor, gold: torch.Tensor) -> int:
    if proposal.shape != gold.shape:
        raise ValueError("proposal and gold must have identical shapes")
    return int(accepted_draft_prefix_lengths(proposal == gold).item())


@torch.inference_mode()
def domino_scaled_onpolicy_ids(
    draft: Any,
    target: Any,
    anchor_token: torch.Tensor,
    parallel_hidden: torch.Tensor,
    base_logits: torch.Tensor,
    scales: list[float],
) -> torch.Tensor:
    """Roll out all correction scales together; returns [scale, position]."""

    if int(getattr(draft, "pure_draft_prefix_len", 0)) != 1:
        raise ValueError("bias-scale diagnostic expects pure_draft_prefix_len=1")
    scale_count = len(scales)
    positions = int(base_logits.shape[1])
    proposals = torch.empty(
        (scale_count, positions), dtype=torch.long, device=base_logits.device
    )
    first_token = base_logits[:, :1].argmax(dim=-1)
    proposals[:, 0] = first_token[0, 0]

    realized_prefix = torch.cat([anchor_token.view(1, 1), first_token], dim=1)
    realized_prefix = realized_prefix.expand(scale_count, -1)
    _, state = draft.prefix_gru(target.model.embed_tokens(realized_prefix))
    scale_tensor: torch.Tensor | None = None

    for position in range(1, positions):
        state_for_head = state.transpose(0, 1)
        if bool(getattr(draft, "use_bias_norm", False)):
            state_for_head = draft.bias_norm(state_for_head)
        hidden_i = parallel_hidden[:, position : position + 1].expand(
            scale_count, -1, -1
        )
        bias = draft.embed_proj(torch.cat([hidden_i, state_for_head], dim=-1))
        if bool(getattr(draft, "use_bias_gate", False)) and hasattr(
            draft, "bias_gate"
        ):
            bias = torch.sigmoid(draft.bias_gate(hidden_i)) * bias
        if scale_tensor is None or scale_tensor.dtype != bias.dtype:
            scale_tensor = torch.tensor(
                scales, dtype=bias.dtype, device=bias.device
            ).view(scale_count, 1, 1)
        base_i = base_logits[:, position : position + 1].expand(
            scale_count, -1, -1
        )
        token = (base_i + scale_tensor * bias).argmax(dim=-1)
        proposals[:, position] = token[:, 0]
        if position + 1 < positions:
            _, state = draft.prefix_gru(target.model.embed_tokens(token), state)
    return proposals


def mean(values: Iterable[int | float]) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("cannot average an empty sequence")
    return sum(items) / len(items)


def summarize(
    block_results: list[dict[str, Any]], methods: list[str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in methods:
        by_prompt: dict[str, list[int]] = defaultdict(list)
        values: list[int] = []
        for record in block_results:
            value = int(record[method])
            values.append(value)
            by_prompt[str(record["sample_id"])].append(value)
        horizon = int(block_results[0]["matched_horizon"])
        summary[method] = {
            "blocks": len(values),
            "mean_accepted_draft_tokens_round_weighted": mean(values),
            "mean_accepted_draft_tokens_prompt_balanced": mean(
                mean(prompt_values) for prompt_values in by_prompt.values()
            ),
            "full_horizon_acceptance": mean(value == horizon for value in values),
        }
    return summary


def paired_prompt_bootstrap(
    block_results: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in block_results:
        grouped[str(record["sample_id"])].append(
            float(record[left]) - float(record[right])
        )
    cluster_means = [mean(values) for values in grouped.values()]
    point = mean(cluster_means)
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(cluster_means) for _ in cluster_means)
        for _ in range(draws)
    )
    return {
        "mean_difference_prompt_balanced": point,
        "ci95_prompt_cluster_bootstrap": [
            estimates[int(0.025 * (draws - 1))],
            estimates[int(0.975 * (draws - 1))],
        ],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Domino bias-scale diagnostic requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    scales = normalize_scales(args.scales)
    metadata, records = load_split_records(args.canonical, args.split)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(record)
    sample_ids = sorted(grouped)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]
    if args.matched_horizon > int(metadata["draft_positions"]):
        raise ValueError("matched horizon exceeds the stored canonical horizon")

    torch.cuda.set_device(0)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    config = getattr(domino.config, "dflash_config", {})
    if config.get("projector_type") not in {"domino", "causal_v5"}:
        raise ValueError("the supplied draft checkpoint is not Domino")

    methods = [scale_key(scale) for scale in scales]
    block_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for sample_index, sample_id in enumerate(sample_ids):
        sample_records = sorted(
            grouped[sample_id], key=lambda item: int(item["anchor_offset"])
        )
        longest_context_ids = validate_stored_canonical_contexts(
            sample_records, sample_id
        )
        target_outputs = target.model(
            longest_context_ids.unsqueeze(0).to(target.device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        context_features = extract_context_feature(
            target_outputs.hidden_states, list(domino.target_layer_ids)
        )
        del target_outputs

        for record in sample_records:
            context_length = int(record["context_ids_before_anchor"].numel())
            anchor = torch.tensor(
                int(record["anchor_token_id"]),
                dtype=torch.long,
                device=target.device,
            )
            block_ids = torch.full(
                (1, int(domino.block_size)),
                int(domino.mask_token_id),
                dtype=torch.long,
                device=target.device,
            )
            block_ids[0, 0] = anchor
            position_ids = torch.arange(
                context_length + int(domino.block_size), device=target.device
            ).unsqueeze(0)
            parallel_hidden = domino(
                target_hidden=context_features[:, :context_length],
                noise_embedding=target.model.embed_tokens(block_ids),
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )
            base_logits = target.lm_head(parallel_hidden)
            proposals = domino_scaled_onpolicy_ids(
                domino, target, anchor, parallel_hidden, base_logits, scales
            )
            horizon = args.matched_horizon
            gold = record["gold_ids"].long().to(target.device)[:horizon]
            result: dict[str, Any] = {
                "sample_id": sample_id,
                "domain": str(record["domain"]),
                "source": str(record["source"]),
                "split": str(record["split"]),
                "anchor_offset": int(record["anchor_offset"]),
                "matched_horizon": horizon,
            }
            for method, proposal in zip(methods, proposals, strict=True):
                result[method] = accepted_length(proposal[:horizon], gold)
            block_results.append(result)
            del parallel_hidden, base_logits, proposals
        print(
            f"[{sample_index + 1}/{len(sample_ids)}] {sample_id}: "
            f"{len(sample_records)} anchors",
            flush=True,
        )

    overall = summarize(block_results, methods)
    by_domain = {
        domain: summarize(
            [record for record in block_results if record["domain"] == domain],
            methods,
        )
        for domain in sorted({str(record["domain"]) for record in block_results})
    }
    released_key = scale_key(1.0)
    paired_vs_released = {
        method: paired_prompt_bootstrap(
            block_results,
            method,
            released_key,
            draws=args.bootstrap_samples,
            seed=args.seed + index,
        )
        for index, method in enumerate(methods)
        if method != released_key
    }
    best_key = max(
        methods,
        key=lambda method: overall[method][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
    )
    report = {
        "status": "completed",
        "experiment": "domino_causal_correction_scale",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "split": args.split,
        "matched_horizon": args.matched_horizon,
        "samples": len(sample_ids),
        "blocks": len(block_results),
        "seconds": time.perf_counter() - started,
        "scales": scales,
        "released_domino_key": released_key,
        "best_key": best_key,
        "best_scale": scales[methods.index(best_key)],
        "overall": overall,
        "by_domain": by_domain,
        "paired_vs_released_domino": paired_vs_released,
        "metric_convention": {
            "primary": "prompt-balanced accepted draft prefix over 15 positions",
            "bootstrap_unit": "prompt/sample_id with all anchors kept together",
            "context_check": "semantic prefix/anchor/gold consistency; no hashing",
        },
        "inputs": {
            "canonical": str(args.canonical.resolve()),
            "target": str(args.target.resolve()),
            "domino_draft": str(args.domino_draft.resolve()),
        },
        "block_results": block_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "best_scale": report["best_scale"],
                "best": overall[best_key],
                "released": overall[released_key],
                "paired_best_vs_released": paired_vs_released.get(best_key),
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
