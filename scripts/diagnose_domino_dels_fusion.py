#!/usr/bin/env python3
"""Screen a residual fusion of Domino and the released DeLS local expert.

For position t>0 the candidate score is

    DominoBackbone + gamma * DominoInteraction
                    + alpha * DeLSLocal - beta * UnigramPrior.

All policies are rolled out on-policy on exactly the same stored contexts.
The script also evaluates the released DeLS recipe on the stored DFlash
backbone as a checkpoint/runtime sanity reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM


PROJECT = Path(__file__).resolve().parents[1]
DELS_CODE = PROJECT / "third_party" / "DeLS-Spec" / "code"
sys.path.insert(0, str(DELS_CODE))

from dels import DeLSLocalHead, load_unigram_log_prior  # noqa: E402
from collect_canonical_blocks import extract_context_feature  # noqa: E402
from diagnose_domino_bias_scale import (  # noqa: E402
    accepted_length,
    load_split_records,
    paired_prompt_bootstrap,
    summarize,
)
from sph.data import validate_stored_canonical_contexts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--dels-local-head", type=Path, required=True)
    parser.add_argument("--dels-unigram", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Policies formatted as name:domino_scale:dels_alpha:dels_beta",
    )
    parser.add_argument("--matched-horizon", type=int, default=15)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def parse_configs(values: list[str]) -> list[dict[str, float | str]]:
    configs: list[dict[str, float | str]] = []
    names: set[str] = set()
    for value in values:
        parts = value.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"invalid fusion config {value!r}; expected name:gamma:alpha:beta"
            )
        name = parts[0]
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"invalid fusion config name {name!r}")
        if name in names:
            raise ValueError(f"duplicate fusion config name {name!r}")
        gamma, alpha, beta = (float(part) for part in parts[1:])
        if gamma < 0.0 or alpha < 0.0 or beta < 0.0:
            raise ValueError("gamma, alpha, and beta must be nonnegative")
        configs.append(
            {"name": name, "gamma": gamma, "alpha": alpha, "beta": beta}
        )
        names.add(name)
    released = [
        config
        for config in configs
        if config["gamma"] == 1.0
        and config["alpha"] == 0.0
        and config["beta"] == 0.0
    ]
    if len(released) != 1:
        raise ValueError("configs must contain exactly one released Domino (1:0:0)")
    return configs


def domino_interaction_bias(
    draft: Any,
    hidden_i: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    state_for_head = state.transpose(0, 1)
    if bool(getattr(draft, "use_bias_norm", False)):
        state_for_head = draft.bias_norm(state_for_head)
    bias = draft.embed_proj(torch.cat([hidden_i, state_for_head], dim=-1))
    if bool(getattr(draft, "use_bias_gate", False)) and hasattr(
        draft, "bias_gate"
    ):
        bias = torch.sigmoid(draft.bias_gate(hidden_i)) * bias
    return bias


@torch.inference_mode()
def domino_dels_fusion_ids(
    *,
    domino: Any,
    local_head: DeLSLocalHead,
    target: Any,
    anchor_token: torch.Tensor,
    parallel_hidden: torch.Tensor,
    base_logits: torch.Tensor,
    unigram: torch.Tensor,
    configs: list[dict[str, float | str]],
) -> torch.Tensor:
    """Return on-policy proposals with shape [configuration, position]."""

    count = len(configs)
    positions = int(base_logits.shape[1])
    proposals = torch.empty(
        (count, positions), dtype=torch.long, device=base_logits.device
    )
    first_token = base_logits[:, :1].argmax(dim=-1)
    proposals[:, 0] = first_token[0, 0]
    repeated_first = first_token.expand(count, -1)

    domino_prefix = torch.cat([anchor_token.view(1, 1), first_token], dim=1)
    domino_prefix = domino_prefix.expand(count, -1)
    _, domino_state = domino.prefix_gru(
        target.model.embed_tokens(domino_prefix)
    )
    local_state = local_head.init_empty_hidden(count, device=base_logits.device)
    local_state = local_head.advance(repeated_first, local_state)

    gamma = torch.tensor(
        [float(config["gamma"]) for config in configs],
        dtype=base_logits.dtype,
        device=base_logits.device,
    ).view(count, 1, 1)
    alpha = torch.tensor(
        [float(config["alpha"]) for config in configs],
        dtype=base_logits.dtype,
        device=base_logits.device,
    ).view(count, 1, 1)
    beta = torch.tensor(
        [float(config["beta"]) for config in configs],
        dtype=base_logits.dtype,
        device=base_logits.device,
    ).view(count, 1, 1)
    prior = unigram.to(device=base_logits.device, dtype=base_logits.dtype)

    for position in range(1, positions):
        hidden_i = parallel_hidden[:, position : position + 1].expand(
            count, -1, -1
        )
        interaction = domino_interaction_bias(domino, hidden_i, domino_state)
        local_logits = local_head.logits_from_hidden(local_state)
        base_i = base_logits[:, position : position + 1].expand(count, -1, -1)
        logits = base_i + gamma * interaction + alpha * local_logits - beta * prior
        token = logits.argmax(dim=-1)
        proposals[:, position] = token[:, 0]
        if position + 1 < positions:
            token_embeddings = target.model.embed_tokens(token)
            _, domino_state = domino.prefix_gru(token_embeddings, domino_state)
            local_state = local_head.advance(token, local_state)
    return proposals


@torch.inference_mode()
def dflash_dels_ids(
    *,
    local_head: DeLSLocalHead,
    base_logits: torch.Tensor,
    unigram: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.3,
) -> torch.Tensor:
    positions = int(base_logits.shape[1])
    proposal = torch.empty((positions,), dtype=torch.long, device=base_logits.device)
    first_token = base_logits[:, :1].argmax(dim=-1)
    proposal[0] = first_token[0, 0]
    state = local_head.init_empty_hidden(1, device=base_logits.device)
    state = local_head.advance(first_token, state)
    prior = unigram.to(device=base_logits.device, dtype=base_logits.dtype)
    for position in range(1, positions):
        short_logits = local_head.logits_from_hidden(state)
        logits = (
            base_logits[:, position : position + 1]
            + float(alpha) * short_logits
            - float(beta) * prior
        )
        token = logits.argmax(dim=-1)
        proposal[position] = token[0, 0]
        if position + 1 < positions:
            state = local_head.advance(token, state)
    return proposal


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Domino-DeLS fusion diagnostic requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    configs = parse_configs(args.configs)
    metadata, records = load_split_records(args.canonical, args.split)
    if args.matched_horizon > int(metadata["draft_positions"]):
        raise ValueError("matched horizon exceeds stored canonical horizon")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(record)
    sample_ids = sorted(grouped)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

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
    local_head = DeLSLocalHead.from_checkpoint(
        checkpoint_path=str(args.dels_local_head),
        target_model=target,
        dtype=torch.bfloat16,
        device=target.device,
    ).eval()
    unigram = load_unigram_log_prior(
        str(args.dels_unigram),
        vocab_size=int(local_head.vocab_size),
        dtype=torch.bfloat16,
        device=target.device,
    )
    if int(local_head.vocab_size) != int(target.lm_head.weight.shape[0]):
        raise ValueError("DeLS and target vocab sizes do not match")

    fusion_methods = [str(config["name"]) for config in configs]
    methods = ["dflash_dels_released", *fusion_methods]
    released_domino_key = next(
        str(config["name"])
        for config in configs
        if config["gamma"] == 1.0
        and config["alpha"] == 0.0
        and config["beta"] == 0.0
    )
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
            domino_hidden = domino(
                target_hidden=context_features[:, :context_length],
                noise_embedding=target.model.embed_tokens(block_ids),
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )
            domino_base_logits = target.lm_head(domino_hidden)
            fusion_ids = domino_dels_fusion_ids(
                domino=domino,
                local_head=local_head,
                target=target,
                anchor_token=anchor,
                parallel_hidden=domino_hidden,
                base_logits=domino_base_logits,
                unigram=unigram,
                configs=configs,
            )
            dflash_hidden = record["parallel_hidden"].to(
                device=target.device, dtype=torch.bfloat16
            ).unsqueeze(0)
            dflash_base_logits = target.lm_head(dflash_hidden)
            released_dels = dflash_dels_ids(
                local_head=local_head,
                base_logits=dflash_base_logits,
                unigram=unigram,
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
                "dflash_dels_released": accepted_length(
                    released_dels[:horizon], gold
                ),
            }
            for method, proposal in zip(
                fusion_methods, fusion_ids, strict=True
            ):
                result[method] = accepted_length(proposal[:horizon], gold)
            block_results.append(result)
            del domino_hidden, domino_base_logits, dflash_base_logits, fusion_ids
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
    paired_vs_domino = {
        method: paired_prompt_bootstrap(
            block_results,
            method,
            released_domino_key,
            draws=args.bootstrap_samples,
            seed=args.seed + index,
        )
        for index, method in enumerate(methods)
        if method != released_domino_key
    }
    best_key = max(
        fusion_methods,
        key=lambda method: overall[method][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
    )
    report = {
        "status": "completed",
        "experiment": "domino_dels_residual_fusion",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "split": args.split,
        "matched_horizon": args.matched_horizon,
        "samples": len(sample_ids),
        "blocks": len(block_results),
        "seconds": time.perf_counter() - started,
        "configs": configs,
        "released_domino_key": released_domino_key,
        "best_fusion_key": best_key,
        "overall": overall,
        "by_domain": by_domain,
        "paired_vs_released_domino": paired_vs_domino,
        "metric_convention": {
            "primary": "prompt-balanced accepted draft prefix over 15 positions",
            "bootstrap_unit": "prompt/sample_id with all anchors kept together",
            "context_check": "semantic prefix/anchor/gold consistency; no hashing",
        },
        "inputs": {
            "canonical": str(args.canonical.resolve()),
            "target": str(args.target.resolve()),
            "domino_draft": str(args.domino_draft.resolve()),
            "dels_local_head": str(args.dels_local_head.resolve()),
            "dels_unigram": str(args.dels_unigram.resolve()),
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
                "best_fusion_key": best_key,
                "best": overall[best_key],
                "released_domino": overall[released_domino_key],
                "released_dflash_dels": overall["dflash_dels_released"],
                "paired_best_vs_domino": paired_vs_domino.get(best_key),
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
