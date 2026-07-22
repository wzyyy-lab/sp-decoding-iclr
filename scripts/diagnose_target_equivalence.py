#!/usr/bin/env python3
"""Diagnose target-logit equivalence across cache, block shape, and backend.

All comparisons are teacher-forced on one immutable token sequence.  This
prevents an early near-tie from cascading into unrelated prefixes.  The report
separates deterministic same-shape replay, cached-one-token versus full-prefix,
block verification versus full-prefix, and eager versus SDPA full-prefix
execution.  A top-1 disagreement is marked numerically explainable only when
the reference top-1 margin is no larger than the measured error on the union of
the two executions' top-2 tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends", nargs="+", default=["eager", "sdpa"]
    )
    parser.add_argument("--reference-backend", default="eager")
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--continuation-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--anchors-per-sample", type=int, default=2)
    parser.add_argument("--error-atol", type=float, default=1e-6)
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit nonzero after writing the report when the diagnostic gate fails.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def evenly_spaced_offsets(maximum: int, count: int) -> list[int]:
    if maximum < 0 or count < 1:
        return []
    if maximum == 0 or count == 1:
        return [0]
    count = min(count, maximum + 1)
    return sorted(
        {
            int(round(index * maximum / (count - 1)))
            for index in range(count)
        }
    )


def load_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records[:limit]


def load_target(path: Path, backend: str) -> Any:
    return AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=backend,
        device_map="cuda:0",
    ).eval()


def comparison_event(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    sample_id: str,
    domain: str,
    token_index: int,
    comparison: str,
    error_atol: float,
    anchor_offset: int | None = None,
) -> dict[str, Any]:
    reference = reference.float()
    candidate = candidate.float()
    reference_values, reference_ids = torch.topk(reference, k=2)
    candidate_values, candidate_ids = torch.topk(candidate, k=2)
    union_ids = torch.unique(torch.cat([reference_ids, candidate_ids]))
    pair_error = torch.max(
        torch.abs(reference[union_ids] - candidate[union_ids])
    )
    max_error = torch.max(torch.abs(reference - candidate))
    margin = reference_values[0] - reference_values[1]
    equal = bool(reference_ids[0] == candidate_ids[0])
    explainable = equal or bool(margin <= 2.0 * pair_error + error_atol)
    return {
        "sample_id": sample_id,
        "domain": domain,
        "comparison": comparison,
        "token_index": token_index,
        "anchor_offset": anchor_offset,
        "reference_top1_id": int(reference_ids[0]),
        "candidate_top1_id": int(candidate_ids[0]),
        "top1_equal": equal,
        "reference_top1_margin": float(margin),
        "max_abs_logit_error": float(max_error),
        "top2_union_max_abs_error": float(pair_error),
        "disagreement_within_measured_error": explainable,
    }


@torch.inference_mode()
def generate_fixed_sequences(
    target: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    continuation_tokens: int,
) -> list[dict[str, Any]]:
    fixed = []
    for index, record in enumerate(records):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(text, return_tensors="pt").to(target.device)
        output = target.generate(
            encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=continuation_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        fixed.append(
            {
                **record,
                "prompt_ids": encoded.input_ids[0].cpu(),
                "continuation_ids": output[0, encoded.input_ids.shape[1] :].cpu(),
            }
        )
        print(f"reference generation [{index + 1}/{len(records)}]", flush=True)
    return fixed


@torch.inference_mode()
def full_prefix_logits(target: Any, sequence: torch.Tensor, prompt_length: int) -> torch.Tensor:
    logits = []
    positions = torch.arange(sequence.numel(), device=target.device).unsqueeze(0)
    sequence = sequence.to(target.device)
    continuation_length = int(sequence.numel()) - prompt_length
    for token_index in range(continuation_length):
        prefix_length = prompt_length + token_index
        output = target(
            sequence[:prefix_length].unsqueeze(0),
            position_ids=positions[:, :prefix_length],
            use_cache=False,
            logits_to_keep=1,
        )
        logits.append(output.logits[0, -1].float().cpu())
    return torch.stack(logits)


@torch.inference_mode()
def cached_single_logits(
    target: Any, prompt: torch.Tensor, continuation: torch.Tensor
) -> torch.Tensor:
    prompt = prompt.to(target.device)
    continuation = continuation.to(target.device)
    cache = DynamicCache()
    prompt_positions = torch.arange(prompt.numel(), device=target.device).unsqueeze(0)
    output = target(
        prompt.unsqueeze(0),
        position_ids=prompt_positions,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    logits = []
    for token_index in range(continuation.numel()):
        logits.append(output.logits[0, -1].float().cpu())
        if token_index + 1 < continuation.numel():
            position = prompt.numel() + token_index
            output = target(
                continuation[token_index : token_index + 1].view(1, 1),
                position_ids=torch.tensor(
                    [[position]], dtype=torch.long, device=target.device
                ),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
    return torch.stack(logits)


@torch.inference_mode()
def cached_block_logits(
    target: Any,
    sequence: torch.Tensor,
    prompt_length: int,
    anchor_offset: int,
    block_size: int,
) -> torch.Tensor:
    sequence = sequence.to(target.device)
    prefix_length = prompt_length + anchor_offset
    cache = DynamicCache()
    positions = torch.arange(sequence.numel(), device=target.device).unsqueeze(0)
    target(
        sequence[:prefix_length].unsqueeze(0),
        position_ids=positions[:, :prefix_length],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    block = sequence[prefix_length : prefix_length + block_size]
    output = target(
        block.unsqueeze(0),
        position_ids=positions[:, prefix_length : prefix_length + block_size],
        past_key_values=cache,
        use_cache=True,
    )
    return output.logits[0].float().cpu()


def aggregate(events: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(event["comparison"], []).append(event)
    result = {}
    for comparison, values in sorted(grouped.items()):
        disagreements = [value for value in values if not value["top1_equal"]]
        unexplained = [
            value
            for value in disagreements
            if not value["disagreement_within_measured_error"]
        ]
        result[comparison] = {
            "predictions": len(values),
            "top1_equal": len(values) - len(disagreements),
            "top1_disagreements": len(disagreements),
            "unexplained_disagreements": len(unexplained),
            "max_abs_logit_error": max(
                value["max_abs_logit_error"] for value in values
            ),
            "max_top2_union_abs_error": max(
                value["top2_union_max_abs_error"] for value in values
            ),
            "minimum_margin_among_unexplained": (
                min(value["reference_top1_margin"] for value in unexplained)
                if unexplained
                else None
            ),
        }
    return result


@torch.inference_mode()
def evaluate_backend(
    *,
    backend: str,
    target_path: Path,
    fixed: list[dict[str, Any]],
    block_size: int,
    anchors_per_sample: int,
    error_atol: float,
    cross_backend_reference: list[torch.Tensor] | None,
    reference_backend: str,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    target = load_target(target_path, backend)
    events: list[dict[str, Any]] = []
    full_outputs: list[torch.Tensor] = []
    for sample_index, record in enumerate(fixed):
        prompt = record["prompt_ids"]
        continuation = record["continuation_ids"]
        sequence = torch.cat([prompt, continuation])
        full = full_prefix_logits(target, sequence, int(prompt.numel()))
        cached_first = cached_single_logits(target, prompt, continuation)
        cached_replay = cached_single_logits(target, prompt, continuation)
        full_outputs.append(full)
        for token_index in range(continuation.numel()):
            events.append(
                comparison_event(
                    full[token_index],
                    cached_first[token_index],
                    sample_id=record["sample_id"],
                    domain=record["domain"],
                    token_index=token_index,
                    comparison=f"{backend}:cached_single_vs_full_prefix",
                    error_atol=error_atol,
                )
            )
            events.append(
                comparison_event(
                    cached_first[token_index],
                    cached_replay[token_index],
                    sample_id=record["sample_id"],
                    domain=record["domain"],
                    token_index=token_index,
                    comparison=f"{backend}:cached_same_shape_replay",
                    error_atol=error_atol,
                )
            )
            if cross_backend_reference is not None:
                events.append(
                    comparison_event(
                        cross_backend_reference[sample_index][token_index],
                        full[token_index],
                        sample_id=record["sample_id"],
                        domain=record["domain"],
                        token_index=token_index,
                        comparison=f"{reference_backend}_vs_{backend}:full_prefix",
                        error_atol=error_atol,
                    )
                )

        maximum_anchor = int(continuation.numel()) - block_size - 1
        for anchor_offset in evenly_spaced_offsets(
            maximum_anchor, anchors_per_sample
        ):
            block = cached_block_logits(
                target,
                sequence,
                int(prompt.numel()),
                anchor_offset,
                block_size,
            )
            for block_position in range(block_size):
                prediction_index = anchor_offset + block_position + 1
                events.append(
                    comparison_event(
                        full[prediction_index],
                        block[block_position],
                        sample_id=record["sample_id"],
                        domain=record["domain"],
                        token_index=prediction_index,
                        anchor_offset=anchor_offset,
                        comparison=f"{backend}:cached_block_vs_full_prefix",
                        error_atol=error_atol,
                    )
                )
        print(
            f"{backend} [{sample_index + 1}/{len(fixed)}] "
            f"tokens={continuation.numel()}",
            flush=True,
        )
    del target
    torch.cuda.empty_cache()
    return events, full_outputs


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("target equivalence diagnosis requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.reference_backend not in args.backends:
        raise ValueError("reference backend must be included in --backends")
    if args.backends[0] != args.reference_backend:
        raise ValueError("reference backend must be the first backend")
    torch.cuda.set_device(0)
    records = load_manifest(args.manifest, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.target), local_files_only=True
    )

    start = time.perf_counter()
    reference_target = load_target(args.target, args.reference_backend)
    fixed = generate_fixed_sequences(
        reference_target, tokenizer, records, args.continuation_tokens
    )
    del reference_target
    torch.cuda.empty_cache()

    all_events: list[dict[str, Any]] = []
    reference_full: list[torch.Tensor] | None = None
    for backend_index, backend in enumerate(args.backends):
        events, full = evaluate_backend(
            backend=backend,
            target_path=args.target,
            fixed=fixed,
            block_size=args.block_size,
            anchors_per_sample=args.anchors_per_sample,
            error_atol=args.error_atol,
            cross_backend_reference=(reference_full if backend_index > 0 else None),
            reference_backend=args.reference_backend,
        )
        all_events.extend(events)
        if backend_index == 0:
            reference_full = full

    summary = aggregate(all_events)
    gate_comparisons = [
        key
        for key in summary
        if "cached_same_shape_replay" not in key
    ]
    gate_pass = all(
        summary[key]["unexplained_disagreements"] == 0
        for key in gate_comparisons
    ) and all(
        summary[key]["top1_disagreements"] == 0
        for key in summary
        if "cached_same_shape_replay" in key
    )
    report = {
        "evidence_tier": "gate0_numerical_diagnostic",
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "target": str(args.target.resolve()),
        "manifest": str(args.manifest.resolve()),
        "backends": args.backends,
        "reference_backend": args.reference_backend,
        "samples": len(fixed),
        "continuation_tokens_requested": args.continuation_tokens,
        "block_size": args.block_size,
        "anchors_per_sample": args.anchors_per_sample,
        "error_rule": (
            "a disagreement is explainable iff reference margin <= "
            "2 * max error on the union of both top-2 sets + error_atol"
        ),
        "error_atol": args.error_atol,
        "gate_pass": gate_pass,
        "seconds": time.perf_counter() - start,
        "provenance": {
            "project_commit": git_revision(PROJECT),
            "manifest_sha256": sha256_file(args.manifest),
            "target_config_sha256": sha256_file(args.target / "config.json"),
            "script_sha256": sha256_file(Path(__file__)),
            "dflash_commit": git_revision(PROJECT / "third_party" / "dflash"),
            "domino_commit": git_revision(PROJECT / "third_party" / "Domino"),
        },
        "summary": summary,
        "events": all_events,
        "fixed_sequences": [
            {
                "sample_id": record["sample_id"],
                "domain": record["domain"],
                "prompt_token_count": int(record["prompt_ids"].numel()),
                "continuation_token_ids": record["continuation_ids"].tolist(),
            }
            for record in fixed
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {"gate_pass": gate_pass, "summary": summary, "output": str(args.output)},
            indent=2,
        ),
        flush=True,
    )
    if args.fail_on_gate and not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
