#!/usr/bin/env python3
"""Load the local Qwen3 target and draft checkpoints on one GPU.

This is intentionally small: it validates the exact Python environment, CUDA
runtime, local model files, and Transformers remote-code entry points before we
launch longer baseline or trace-collection jobs.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument(
        "--draft-model",
        action="append",
        type=Path,
        required=True,
        help="Repeat for DFlash and Domino checkpoints.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synchronize(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def load_and_generate(
    *,
    torch_module: Any,
    auto_model: Any,
    target: Any,
    input_ids: Any,
    draft_path: Path,
    eos_token_id: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    load_start = time.perf_counter()
    draft = auto_model.from_pretrained(
        str(draft_path),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch_module.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    synchronize(torch_module)
    load_seconds = time.perf_counter() - load_start

    dflash_config = getattr(draft.config, "dflash_config", {})
    projector_type = dflash_config.get("projector_type")
    method = "domino" if projector_type in {"domino", "causal_v5"} else "dflash"
    signature = inspect.signature(draft.spec_generate)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "target": target,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "stop_token_ids": [eos_token_id],
    }
    if "return_dict" in signature.parameters:
        kwargs["return_dict"] = True

    synchronize(torch_module)
    generate_start = time.perf_counter()
    result = draft.spec_generate(**kwargs)
    synchronize(torch_module)
    generate_seconds = time.perf_counter() - generate_start

    if hasattr(result, "output_ids"):
        output_ids = result.output_ids
        acceptance_lengths = [int(x) for x in result.acceptance_lengths]
        time_per_output_token = float(result.time_per_output_token)
    else:
        output_ids = result
        acceptance_lengths = None
        time_per_output_token = None

    generated = output_ids[:, input_ids.shape[1] :]
    record = {
        "method": method,
        "draft_model": str(draft_path),
        "projector_type": projector_type,
        "block_size": int(getattr(draft, "block_size", -1)),
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "num_generated_tokens": int(generated.shape[1]),
        "generated_token_ids": generated[0].tolist(),
        "acceptance_lengths": acceptance_lengths,
        "time_per_output_token": time_per_output_token,
        "peak_memory_gib": torch_module.cuda.max_memory_allocated() / 2**30,
    }
    del result, generated, output_ids, draft
    torch_module.cuda.empty_cache()
    return record


def main() -> None:
    args = parse_args()
    for path in [args.target_model, *args.draft_model]:
        if not path.is_dir():
            raise FileNotFoundError(path)

    # Delay heavyweight imports until after argument/path validation so failures
    # in the Slurm log are easier to interpret.
    import torch
    import transformers
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this job")
    torch.cuda.set_device(0)

    report: dict[str, Any] = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "target_model": str(args.target_model),
        "drafts": [],
    }
    print(json.dumps(report, indent=2), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.target_model), local_files_only=True
    )
    prompt = "How many positive whole-number divisors does 196 have?"
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to("cuda:0")

    load_start = time.perf_counter()
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target_model),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    synchronize(torch)
    report["target_load_seconds"] = time.perf_counter() - load_start
    report["input_tokens"] = int(input_ids.shape[1])

    for draft_path in args.draft_model:
        print(f"loading draft: {draft_path}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        record = load_and_generate(
            torch_module=torch,
            auto_model=AutoModel,
            target=target,
            input_ids=input_ids,
            draft_path=draft_path,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
        )
        record["text"] = tokenizer.decode(
            record["generated_token_ids"], skip_special_tokens=True
        )
        report["drafts"].append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
