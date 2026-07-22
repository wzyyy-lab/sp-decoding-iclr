#!/usr/bin/env python3
"""Run DFlash and Domino eager baselines on one fixed prompt manifest."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "third_party" / "dflash"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--dflash-draft", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def load_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records if limit is None else records[:limit]


def synchronize() -> None:
    torch.cuda.synchronize()


def acceptance_profile(accepted_draft_tokens: list[int], maximum: int) -> dict[str, Any]:
    if not accepted_draft_tokens:
        return {"reach": [], "conditional_hazard": [], "full_acceptance": None}
    reaches = []
    hazards = []
    for position in range(1, maximum + 1):
        reached = sum(value >= position - 1 for value in accepted_draft_tokens)
        accepted = sum(value >= position for value in accepted_draft_tokens)
        reaches.append(reached / len(accepted_draft_tokens))
        hazards.append(1.0 - accepted / reached if reached else None)
    return {
        "reach": reaches,
        "conditional_hazard": hazards,
        "full_acceptance": sum(value == maximum for value in accepted_draft_tokens)
        / len(accepted_draft_tokens),
    }


def first_mismatch_index(left: torch.Tensor, right: torch.Tensor) -> int | None:
    shared = min(left.shape[-1], right.shape[-1])
    differences = (left[..., :shared] != right[..., :shared]).nonzero()
    if differences.numel() > 0:
        return int(differences[0, -1].item())
    if left.shape[-1] != right.shape[-1]:
        return shared
    return None


def prepare_inputs(
    records: list[dict[str, Any]], tokenizer: Any, device: torch.device
) -> list[dict[str, Any]]:
    prepared = []
    for record in records:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(text, return_tensors="pt").to(device)
        prepared.append(
            {
                **record,
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
            }
        )
    return prepared


@torch.inference_mode()
def greedy_targets(
    prepared: list[dict[str, Any]],
    target: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> list[torch.Tensor]:
    outputs = []
    for index, record in enumerate(prepared):
        output = target.generate(
            record["input_ids"],
            attention_mask=record["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs.append(output[:, record["input_ids"].shape[1] :].cpu())
        print(f"target greedy [{index + 1}/{len(prepared)}]", flush=True)
    return outputs


def aggregate_method(
    method: str,
    sample_results: list[dict[str, Any]],
    drafted_per_round: int,
) -> dict[str, Any]:
    all_advances = [
        advance
        for sample in sample_results
        for advance in sample["acceptance_lengths"]
    ]
    accepted_draft = [advance - 1 for advance in all_advances]
    per_domain: dict[str, list[int]] = defaultdict(list)
    for sample in sample_results:
        per_domain[sample["domain"]].extend(
            advance - 1 for advance in sample["acceptance_lengths"]
        )
    total_tokens = sum(sample["num_output_tokens"] for sample in sample_results)
    total_decode = sum(sample["decode_seconds"] for sample in sample_results)
    total_wall = sum(sample["wall_seconds"] for sample in sample_results)
    return {
        "method": method,
        "drafted_tokens_per_round": drafted_per_round,
        "samples": len(sample_results),
        "rounds": len(all_advances),
        "token_exact_samples": sum(sample["token_exact"] for sample in sample_results),
        "mean_accepted_draft_tokens": sum(accepted_draft) / len(accepted_draft),
        "mean_verification_advance": sum(all_advances) / len(all_advances),
        "decode_tokens_per_second": total_tokens / total_decode,
        "wall_tokens_per_second": total_tokens / total_wall,
        "acceptance_profile": acceptance_profile(accepted_draft, drafted_per_round),
        "by_domain": {
            domain: {
                "rounds": len(values),
                "mean_accepted_draft_tokens": sum(values) / len(values),
                "mean_verification_advance": 1.0 + sum(values) / len(values),
                "acceptance_profile": acceptance_profile(values, drafted_per_round),
            }
            for domain, values in sorted(per_domain.items())
        },
        "sample_results": sample_results,
    }


@torch.inference_mode()
def run_method(
    *,
    method: str,
    draft_path: Path,
    prepared: list[dict[str, Any]],
    gold_outputs: list[torch.Tensor],
    target: Any,
    tokenizer: Any,
    max_new_tokens: int,
    block_size: int,
    attn_implementation: str,
) -> dict[str, Any]:
    print(f"loading {method}: {draft_path}", flush=True)
    draft = AutoModel.from_pretrained(
        str(draft_path),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map="cuda:0",
    ).eval()
    shift_label = bool(
        getattr(draft.config, "dflash_config", {}).get("shift_label", False)
    )
    drafted_per_round = block_size if shift_label else block_size - 1

    if method == "dflash":
        from dflash.model import dflash_generate

        def generate(input_ids: torch.Tensor, token_limit: int) -> Any:
            return dflash_generate(
                draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=token_limit,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                block_size=block_size,
                return_stats=True,
            )

    elif method == "domino":

        def generate(input_ids: torch.Tensor, token_limit: int) -> Any:
            return draft.spec_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=token_limit,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                block_size=block_size,
                graph_runner=None,
                use_bias=True,
                return_dict=True,
            )

    else:
        raise ValueError(method)

    # One short untimed generation initializes kernels and allocator state.
    generate(prepared[0]["input_ids"], min(16, max_new_tokens))
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    sample_results = []
    for index, (record, gold) in enumerate(zip(prepared, gold_outputs, strict=True)):
        synchronize()
        wall_start = time.perf_counter()
        result = generate(record["input_ids"], max_new_tokens)
        synchronize()
        wall_seconds = time.perf_counter() - wall_start
        generated = result.output_ids[:, result.num_input_tokens :].cpu()
        token_exact = torch.equal(generated, gold)
        mismatch_index = first_mismatch_index(generated, gold)
        sample_result = {
            "sample_id": record["sample_id"],
            "domain": record["domain"],
            "split": record["split"],
            "num_input_tokens": int(result.num_input_tokens),
            "num_output_tokens": int(result.num_output_tokens),
            "time_to_first_token": float(result.time_to_first_token),
            "decode_seconds": float(result.time_per_output_token)
            * int(result.num_output_tokens),
            "wall_seconds": wall_seconds,
            "acceptance_lengths": [int(value) for value in result.acceptance_lengths],
            "token_exact": token_exact,
            "first_mismatch_index": mismatch_index,
            "generated_token_ids": generated[0].tolist(),
            "target_greedy_token_ids": gold[0].tolist(),
        }
        sample_results.append(sample_result)
        print(
            f"{method} [{index + 1}/{len(prepared)}] "
            f"exact={token_exact} mismatch={mismatch_index} "
            f"advances={sample_result['acceptance_lengths']}",
            flush=True,
        )

    summary = aggregate_method(method, sample_results, drafted_per_round)
    summary["shift_label"] = shift_label
    summary["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    del draft
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("baseline requires a CUDA GPU")
    torch.cuda.set_device(0)
    records = load_manifest(args.manifest, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(str(args.target), local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    prepared = prepare_inputs(records, tokenizer, target.device)
    gold_outputs = greedy_targets(prepared, target, tokenizer, args.max_new_tokens)

    report: dict[str, Any] = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "target": str(args.target.resolve()),
        "manifest": str(args.manifest.resolve()),
        "block_size_argument": args.block_size,
        "max_new_tokens": args.max_new_tokens,
        "metric_convention": {
            "acceptance_lengths": "raw verification advance returned by official loops",
            "accepted_draft_tokens": "acceptance_length - 1 (known anchor removed)",
        },
        "methods": {},
    }
    for method, draft_path in [
        ("dflash", args.dflash_draft),
        ("domino", args.domino_draft),
    ]:
        report["methods"][method] = run_method(
            method=method,
            draft_path=draft_path,
            prepared=prepared,
            gold_outputs=gold_outputs,
            target=target,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            attn_implementation=args.attn_implementation,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    concise = {
        method: {
            key: value
            for key, value in summary.items()
            if key
            in {
                "mean_accepted_draft_tokens",
                "mean_verification_advance",
                "decode_tokens_per_second",
                "wall_tokens_per_second",
                "token_exact_samples",
            }
        }
        for method, summary in report["methods"].items()
    }
    print(json.dumps(concise, indent=2), flush=True)


if __name__ == "__main__":
    main()
