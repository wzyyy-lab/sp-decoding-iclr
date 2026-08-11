#!/usr/bin/env python3
"""Compare DFlash ceilings and Domino on exactly the same canonical anchors.

This script closes Gate 1b.  It treats the existing pure-DFlash canonical
collection as immutable, reconstructs every target prefix from its manifest,
fails if the reconstruction does not reproduce the stored anchor/gold tokens,
and evaluates Domino's frozen backbone and on-policy GRU proposal at those same
prefixes.  The primary comparison truncates every method to 15 draft positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from collect_canonical_blocks import extract_context_feature, package_version
from collect_domino_canonical import correction_logits
from sph.candidate_ceiling import accepted_draft_prefix_lengths
from sph.data import validate_stored_canonical_contexts


PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-horizon", type=int, default=15)
    parser.add_argument("--oracle-k", type=int, nargs="+", default=[8, 16])
    parser.add_argument(
        "--split",
        help=(
            "Optional manifest split filter applied after verifying the "
            "canonical collection's immutable source manifest."
        ),
    )
    parser.add_argument(
        "--allow-missing-canonical-samples",
        action="store_true",
        help=(
            "Explicitly restrict evaluation to the manifest/canonical "
            "intersection when selected manifest samples produced no canonical "
            "blocks. Missing sample IDs are recorded in the output report."
        ),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit nonzero after writing the report when the K=16 gate fails.",
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


def git_is_dirty(path: Path) -> bool | None:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    ids = [record["sample_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest sample_id values must be unique")
    return records


def load_canonical(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("collection_complete") is False:
        raise RuntimeError(f"canonical collection is incomplete: {path}")
    shard_paths = sorted(path.glob("shard-*.pt"))
    expected_shards = metadata.get("shards")
    if expected_shards is not None:
        expected_names = [entry["path"] for entry in expected_shards]
        actual_names = [shard.name for shard in shard_paths]
        if actual_names != expected_names:
            raise RuntimeError("canonical shard manifest does not match the directory")
        for shard, expected in zip(
            shard_paths, expected_shards, strict=True
        ):
            if shard.stat().st_size != int(expected["bytes"]):
                raise RuntimeError(f"canonical shard size mismatch: {shard}")
            if sha256_file(shard) != expected["sha256"]:
                raise RuntimeError(f"canonical shard hash mismatch: {shard}")
    records: list[dict[str, Any]] = []
    for shard in shard_paths:
        records.extend(torch.load(shard, map_location="cpu", weights_only=False))
    if not records:
        raise FileNotFoundError(f"no canonical records found under {path}")
    return metadata, records


def accepted_length(proposal: torch.Tensor, gold: torch.Tensor) -> int:
    if proposal.shape != gold.shape:
        raise ValueError("proposal and gold must have identical shapes")
    return int(
        accepted_draft_prefix_lengths(proposal == gold).item()
    )


@torch.inference_mode()
def domino_onpolicy_ids(
    draft: Any,
    target: Any,
    anchor_token: torch.Tensor,
    parallel_hidden: torch.Tensor,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the eager Domino proposal without invoking target verification."""

    positions = int(base_logits.shape[1])
    prefix_len = int(getattr(draft, "pure_draft_prefix_len", 0))
    if prefix_len != 1:
        raise ValueError(f"expected Domino pure_draft_prefix_len=1, got {prefix_len}")
    proposal = torch.empty((1, positions), dtype=torch.long, device=target.device)
    first_token = base_logits[:, :1].argmax(dim=-1)
    proposal[:, :1] = first_token
    realized_prefix = torch.cat([anchor_token.view(1, 1), first_token], dim=1)
    _, state = draft.prefix_gru(target.model.embed_tokens(realized_prefix))
    for position in range(1, positions):
        logits = correction_logits(
            draft,
            parallel_hidden[:, position : position + 1],
            state,
            base_logits[:, position : position + 1],
        )
        token = logits.argmax(dim=-1)
        proposal[:, position : position + 1] = token
        if position + 1 < positions:
            _, state = draft.prefix_gru(target.model.embed_tokens(token), state)
    return proposal


def mean(values: list[int | float]) -> float:
    return sum(float(value) for value in values) / len(values)


def cluster_bootstrap_difference(
    block_results: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    draws: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in block_results:
        grouped[record["sample_id"]].append(
            float(record[left]) - float(record[right])
        )
    cluster_means = [mean(values) for values in grouped.values()]
    rng = random.Random(seed)
    estimates = [
        mean([rng.choice(cluster_means) for _ in cluster_means])
        for _ in range(draws)
    ]
    estimates.sort()
    return [
        estimates[int(0.025 * (draws - 1))],
        estimates[int(0.975 * (draws - 1))],
    ]


def summarize(
    block_results: list[dict[str, Any]], methods: list[str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in methods:
        values = [record[method] for record in block_results]
        by_prompt: dict[str, list[int]] = defaultdict(list)
        for record, value in zip(block_results, values, strict=True):
            by_prompt[record["sample_id"]].append(int(value))
        summary[method] = {
            "blocks": len(values),
            "mean_accepted_draft_tokens_round_weighted": mean(values),
            "mean_verification_advance_round_weighted": mean(values) + 1.0,
            "mean_accepted_draft_tokens_prompt_balanced": mean(
                [mean(items) for items in by_prompt.values()]
            ),
            "full_horizon_acceptance": mean(
                [value == block_results[0]["matched_horizon"] for value in values]
            ),
        }
    return summary


def paired_summary(
    block_results: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    round_differences = []
    for record in block_results:
        difference = float(record[left]) - float(record[right])
        round_differences.append(difference)
        grouped[record["sample_id"]].append(difference)
    prompt_balanced_difference = mean(
        [mean(values) for values in grouped.values()]
    )
    return {
        "mean_difference": prompt_balanced_difference,
        "mean_difference_prompt_balanced": prompt_balanced_difference,
        "mean_difference_round_weighted": mean(round_differences),
        "ci95_prompt_cluster_bootstrap": cluster_bootstrap_difference(
            block_results, left, right, draws=draws, seed=seed
        ),
    }


def main() -> None:
    args = parse_args()
    metadata_path = args.canonical / "metadata.json"
    run_provenance = {
        "project_commit": git_revision(PROJECT),
        "project_dirty_at_start": git_is_dirty(PROJECT),
        "manifest_sha256": sha256_file(args.manifest),
        "canonical_metadata_sha256": sha256_file(metadata_path),
        "target_config_sha256": sha256_file(args.target / "config.json"),
        "domino_config_sha256": sha256_file(args.domino_draft / "config.json"),
        "domino_remote_code_sha256": sha256_file(
            args.domino_draft / "dflash.py"
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "dflash_commit": git_revision(PROJECT / "third_party" / "dflash"),
        "dflash_dirty_at_start": git_is_dirty(PROJECT / "third_party" / "dflash"),
        "domino_commit": git_revision(PROJECT / "third_party" / "Domino"),
        "domino_dirty_at_start": git_is_dirty(PROJECT / "third_party" / "Domino"),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("same-anchor Domino comparison requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.cuda.set_device(0)

    canonical_metadata, canonical_records = load_canonical(args.canonical)
    manifest = load_manifest(args.manifest)
    if Path(canonical_metadata["manifest"]).resolve() != args.manifest.resolve():
        raise ValueError("canonical metadata and requested manifest do not match")
    saved_horizon = int(canonical_metadata["draft_positions"])
    saved_k = int(canonical_metadata["top_k"])
    if args.matched_horizon > saved_horizon:
        raise ValueError("matched horizon exceeds stored DFlash horizon")
    if any(k < 1 or k > saved_k for k in args.oracle_k):
        raise ValueError(f"oracle K must be within the saved top-{saved_k}")

    if args.split is not None:
        manifest = [
            record
            for record in manifest
            if str(record.get("split")) == args.split
        ]
        if not manifest:
            raise ValueError(f"manifest contains no split {args.split!r}")
    if args.max_samples is not None:
        manifest = manifest[: args.max_samples]
    requested_samples = len(manifest)
    selected_ids = {record["sample_id"] for record in manifest}
    canonical_records = [
        record
        for record in canonical_records
        if record["sample_id"] in selected_ids
    ]
    grouped_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in canonical_records:
        grouped_canonical[record["sample_id"]].append(record)
    missing = sorted(selected_ids - grouped_canonical.keys())
    if missing:
        if not args.allow_missing_canonical_samples:
            raise ValueError(f"canonical collection is missing samples: {missing}")
        manifest = [
            record
            for record in manifest
            if record["sample_id"] in grouped_canonical
        ]
    if not manifest:
        raise ValueError("no manifest samples have canonical blocks")
    stored_context_flags = {
        "context_ids_before_anchor" in record for record in canonical_records
    }
    if len(stored_context_flags) != 1:
        raise RuntimeError("canonical collection mixes stored and regenerated contexts")
    use_stored_context = stored_context_flags == {True}

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.target), local_files_only=True
    )
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
    if not bool(config.get("shift_label", False)):
        raise ValueError("the same-anchor comparison expects shift_label=true")

    block_results: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for sample_index, sample in enumerate(manifest):
        records = sorted(
            grouped_canonical[sample["sample_id"]],
            key=lambda item: int(item["anchor_offset"]),
        )
        if use_stored_context:
            longest_context_ids = validate_stored_canonical_contexts(
                records, sample["sample_id"]
            )
            sequence_for_features = longest_context_ids.unsqueeze(0).to(
                target.device
            )
            longest_context = int(longest_context_ids.numel())
            continuation = None
            prompt_tokens = None
        else:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": sample["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(text, return_tensors="pt").to(target.device)
            prompt_tokens = int(encoded.input_ids.shape[1])
            if any(
                int(record["prompt_token_count"]) != prompt_tokens
                for record in records
            ):
                raise RuntimeError(
                    f"prompt tokenization drift for {sample['sample_id']}"
                )
            sequence = target.generate(
                encoded.input_ids,
                attention_mask=encoded.attention_mask,
                max_new_tokens=int(canonical_metadata["continuation_tokens"]) + 1,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = sequence[0, prompt_tokens:]
            longest_context = prompt_tokens + max(
                int(record["anchor_offset"]) for record in records
            )
            sequence_for_features = sequence[:, :longest_context]
        target_outputs = target.model(
            sequence_for_features,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        context_features = extract_context_feature(
            target_outputs.hidden_states, list(domino.target_layer_ids)
        )
        del target_outputs

        for record in records:
            offset = int(record["anchor_offset"])
            stored_anchor = int(record["anchor_token_id"])
            stored_gold = record["gold_ids"].long().to(target.device)
            if use_stored_context:
                anchor = torch.tensor(
                    stored_anchor, dtype=torch.long, device=target.device
                )
                context_length = int(
                    record["context_ids_before_anchor"].numel()
                )
            else:
                assert continuation is not None and prompt_tokens is not None
                anchor = continuation[offset].to(torch.long)
                reconstructed_gold = continuation[
                    offset + 1 : offset + 1 + saved_horizon
                ].long()
                if int(anchor) != stored_anchor or not torch.equal(
                    reconstructed_gold, stored_gold
                ):
                    raise RuntimeError(
                        f"canonical reconstruction mismatch for "
                        f"{sample['sample_id']} at offset {offset}"
                    )
                context_length = prompt_tokens + offset
            block_size = int(domino.block_size)
            block_ids = torch.full(
                (1, block_size),
                int(domino.mask_token_id),
                dtype=torch.long,
                device=target.device,
            )
            block_ids[0, 0] = anchor
            position_ids = torch.arange(
                context_length + block_size, device=target.device
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
            domino_base = base_logits.argmax(dim=-1)[0]
            domino_onpolicy = domino_onpolicy_ids(
                domino, target, anchor, parallel_hidden, base_logits
            )[0]

            horizon = args.matched_horizon
            gold = stored_gold[:horizon]
            dflash_topk = record["base_topk_ids"].long().to(target.device)[
                :horizon
            ]
            result: dict[str, Any] = {
                "sample_id": sample["sample_id"],
                "domain": sample["domain"],
                "source": sample["source"],
                "split": sample["split"],
                "anchor_offset": offset,
                "matched_horizon": horizon,
                "dflash_top1": accepted_length(dflash_topk[:, 0], gold),
                "domino_backbone_top1": accepted_length(
                    domino_base[:horizon], gold
                ),
                "domino_onpolicy": accepted_length(
                    domino_onpolicy[:horizon], gold
                ),
                "gold_ids": gold.cpu().tolist(),
                "dflash_top1_ids": dflash_topk[:, 0].cpu().tolist(),
                "domino_backbone_top1_ids": domino_base[:horizon].cpu().tolist(),
                "domino_onpolicy_ids": domino_onpolicy[:horizon].cpu().tolist(),
            }
            for k in sorted(set(args.oracle_k)):
                oracle_matches = (dflash_topk[:, :k] == gold[:, None]).any(dim=-1)
                result[f"dflash_oracle_k{k}"] = int(
                    accepted_draft_prefix_lengths(oracle_matches).item()
                )
            block_results.append(result)
            del parallel_hidden, base_logits

        print(
            f"[{sample_index + 1}/{len(manifest)}] {sample['sample_id']}: "
            f"{len(records)} anchors",
            flush=True,
        )

    methods = [
        "dflash_top1",
        "domino_backbone_top1",
        "domino_onpolicy",
        *[f"dflash_oracle_k{k}" for k in sorted(set(args.oracle_k))],
    ]
    overall = summarize(block_results, methods)
    by_domain = {
        domain: summarize(
            [record for record in block_results if record["domain"] == domain],
            methods,
        )
        for domain in sorted({record["domain"] for record in block_results})
    }
    paired: dict[str, Any] = {}
    paired["domino_onpolicy_minus_dflash_top1"] = paired_summary(
        block_results,
        "domino_onpolicy",
        "dflash_top1",
        draws=args.bootstrap_samples,
        seed=args.seed + 101,
    )
    paired["domino_onpolicy_minus_domino_backbone_top1"] = paired_summary(
        block_results,
        "domino_onpolicy",
        "domino_backbone_top1",
        draws=args.bootstrap_samples,
        seed=args.seed + 102,
    )
    domino_mean = overall["domino_onpolicy"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    threshold = max(0.5, 0.10 * domino_mean)
    for k in sorted(set(args.oracle_k)):
        oracle = f"dflash_oracle_k{k}"
        difference = overall[oracle][
            "mean_accepted_draft_tokens_prompt_balanced"
        ] - domino_mean
        interval = cluster_bootstrap_difference(
            block_results,
            oracle,
            "domino_onpolicy",
            draws=args.bootstrap_samples,
            seed=args.seed + k,
        )
        paired[f"{oracle}_minus_domino_onpolicy"] = {
            "mean_difference": difference,
            "ci95_prompt_cluster_bootstrap": interval,
            "gate_threshold": threshold,
            "point_estimate_exceeds_threshold": difference >= threshold,
            "ci_excludes_zero": interval[0] > 0.0,
        }
    paired_by_domain: dict[str, Any] = {}
    for domain_index, domain in enumerate(sorted(by_domain)):
        subset = [
            record for record in block_results if record["domain"] == domain
        ]
        paired_by_domain[domain] = {
            "domino_onpolicy_minus_dflash_top1": paired_summary(
                subset,
                "domino_onpolicy",
                "dflash_top1",
                draws=args.bootstrap_samples,
                seed=args.seed + 200 + domain_index,
            )
        }
        for k in sorted(set(args.oracle_k)):
            oracle = f"dflash_oracle_k{k}"
            paired_by_domain[domain][
                f"{oracle}_minus_domino_onpolicy"
            ] = paired_summary(
                subset,
                oracle,
                "domino_onpolicy",
                draws=args.bootstrap_samples,
                seed=args.seed + 300 + 10 * domain_index + k,
            )
    primary_key = "dflash_oracle_k16_minus_domino_onpolicy"
    if primary_key not in paired:
        raise ValueError("Gate 1b requires --oracle-k to include 16")
    primary_gate = paired[primary_key]
    domains_with_positive_primary_gain = sum(
        value[primary_key]["mean_difference"] > 0.0
        for value in paired_by_domain.values()
    )
    gate_pass = bool(
        primary_gate["point_estimate_exceeds_threshold"]
        and primary_gate["ci_excludes_zero"]
        and domains_with_positive_primary_gain >= 2
    )

    report = {
        "evidence_tier": "gate1b_same_anchor",
        "status": "completed",
        "gate_pass": gate_pass,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "attention_implementation": args.attn_implementation,
        "dtype": "bfloat16",
        "matched_horizon": args.matched_horizon,
        "split_filter": args.split,
        "requested_samples_before_missing_filter": requested_samples,
        "missing_canonical_samples": missing,
        "samples": len(manifest),
        "blocks": len(block_results),
        "seconds": time.perf_counter() - start_time,
        "inputs": {
            "canonical": str(args.canonical.resolve()),
            "target": str(args.target.resolve()),
            "domino_draft": str(args.domino_draft.resolve()),
            "manifest": str(args.manifest.resolve()),
        },
        "provenance": run_provenance,
        "metric_convention": {
            "primary": "accepted draft tokens over the shared first 15 positions",
            "verification_advance": "accepted draft tokens + 1",
            "bootstrap_unit": "prompt/sample_id; all anchors resampled together",
            "canonical_reconstruction": (
                "exact stored context replay with shard integrity verification"
                if use_stored_context
                else "legacy regeneration with hard failure on anchor/gold mismatch"
            ),
        },
        "context_replay_mode": (
            "stored_exact_context" if use_stored_context else "legacy_regeneration"
        ),
        "overall": overall,
        "by_domain": by_domain,
        "paired_comparisons": paired,
        "paired_comparisons_by_domain": paired_by_domain,
        "domains_with_positive_k16_oracle_gain": domains_with_positive_primary_gain,
        "block_results": block_results,
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
            {
                "overall": overall,
                "paired_comparisons": paired,
                "gate_pass": gate_pass,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.fail_on_gate and not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
