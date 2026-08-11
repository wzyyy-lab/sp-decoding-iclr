#!/usr/bin/env python3
"""Train a local or globally normalized path head on frozen block features."""

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
from safetensors import safe_open
from torch.utils.data import DataLoader

from sph.candidate_ceiling import accepted_draft_prefix_lengths
from sph.data import CanonicalBlockDataset, collate_canonical_blocks
from sph.survival_path_head import (
    BidirectionalSurvivalPathHead,
    SurvivalPathHead,
    absorbing_prefix_crf_conditionals,
    gold_prefix_survival_loss,
    greedy_markov_decode,
    prefix_censored_nll,
    survival_decode,
    viterbi_decode,
)


PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument(
        "--head-type",
        choices=["no_mixer", "bidirectional"],
        default="no_mixer",
        help="Use the minimal pairwise head by default; the mixer is an ablation.",
    )
    parser.add_argument(
        "--normalization",
        choices=["absorbing_crf", "local"],
        default="absorbing_crf",
        help="Training distribution. absorbing_crf is the proposed model.",
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--base-regularization", type=float, default=0.01)
    parser.add_argument(
        "--survival-loss-weight",
        type=float,
        default=0.1,
        help="Weight on negative predicted utility of the observed gold prefix.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument(
        "--gate-split",
        help=(
            "Optional held-out development gate evaluated exactly once after "
            "validation-only checkpoint selection."
        ),
    )
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help=(
            "Do not load or evaluate a test split. Use this for development "
            "selection before the reserved formal test is unsealed."
        ),
    )
    parser.add_argument(
        "--evidence-tier",
        choices=["plumbing_smoke", "development", "formal"],
        default="development",
        help="Recorded verbatim so smoke outputs cannot be mistaken for evidence.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_target_embedding(target: Path) -> torch.Tensor:
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "model.embed_tokens.weight"
    shard_name = index["weight_map"][key]
    with safe_open(target / shard_name, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def validate_target_embedding_identity(
    data_metadata: dict[str, Any], target: Path
) -> list[dict[str, Any]]:
    """Bind training-time token embeddings to the collection checkpoint."""

    if int(data_metadata.get("format_version", 1)) < 2:
        return []
    expected_records = data_metadata.get("provenance", {}).get("target_files")
    if not isinstance(expected_records, list):
        raise RuntimeError("protocol-v2 data is missing target file fingerprints")
    expected = {str(record["path"]): record for record in expected_records}
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    embedding_shard = str(index["weight_map"]["model.embed_tokens.weight"])
    required_names = ["config.json", index_path.name, embedding_shard]
    verified = []
    for name in required_names:
        if name not in expected:
            raise RuntimeError(f"collection fingerprint is missing target file {name}")
        path = target / name
        actual = {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        reference = expected[name]
        if actual["bytes"] != int(reference["bytes"]):
            raise RuntimeError(f"training target size differs from collection: {name}")
        if actual["sha256"] != str(reference["sha256"]):
            raise RuntimeError(f"training target hash differs from collection: {name}")
        verified.append(actual)
    return verified


def make_loader(
    dataset: CanonicalBlockDataset,
    *,
    candidate_k: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda records: collate_canonical_blocks(records, candidate_k),
    )


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def assert_prompt_disjoint_splits(
    named_datasets: dict[str, CanonicalBlockDataset | None],
) -> None:
    sample_ids = {
        name: {str(record["sample_id"]) for record in dataset.records}
        for name, dataset in named_datasets.items()
        if dataset is not None
    }
    names = list(sample_ids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sample_ids[left] & sample_ids[right]
            if overlap:
                examples = sorted(overlap)[:3]
                raise RuntimeError(
                    f"prompt leakage between {left} and {right}: {examples}"
                )


def realized_prefix(
    path: torch.Tensor, candidate_ids: torch.Tensor, gold_ids: torch.Tensor
) -> torch.Tensor:
    selected_ids = candidate_ids.gather(-1, path.unsqueeze(-1)).squeeze(-1)
    return accepted_draft_prefix_lengths(selected_ids == gold_ids)


@torch.inference_mode()
def evaluate(
    head: torch.nn.Module,
    loader: DataLoader,
    embedding: torch.Tensor,
    device: torch.device,
    normalization: str,
    include_examples: bool = False,
) -> dict[str, Any]:
    head.eval()
    accumulators: dict[str, list[torch.Tensor]] = {
        "base": [],
        "local": [],
        "local_map": [],
        "local_survival": [],
        "global_map": [],
        "global_survival": [],
        "first_base": [],
        "first_local": [],
        "first_local_map": [],
        "first_local_survival": [],
        "first_global_map": [],
        "first_global_survival": [],
        "predicted_global_survival_utility": [],
    }
    selected_losses = []
    local_losses = []
    global_losses = []
    example_records: list[dict[str, Any]] = []
    decoder_pairs = [
        ("global_survival", "global_map"),
        ("global_survival", "local_survival"),
        ("global_survival", "base"),
        ("global_map", "base"),
        ("local_survival", "local"),
    ]
    disagreement_stats = {
        f"{left}_vs_{right}": {
            "examples": 0,
            "path_disagreements": 0,
            "first_token_disagreements": 0,
            "realized_delta_on_path_disagreements": 0.0,
        }
        for left, right in decoder_pairs
    }
    score_diagnostics = {
        "residual_abs_sum": 0.0,
        "residual_square_sum": 0.0,
        "residual_count": 0,
        "residual_abs_max": 0.0,
        "base_top1_margin_sum": 0.0,
        "base_top1_margin_count": 0,
        "residual_next_range_sum": 0.0,
        "residual_next_range_count": 0,
        "residual_range_exceeds_margin_count": 0,
    }
    for batch in loader:
        batch = to_device(batch, device)
        anchor_embedding = embedding[batch["anchor_ids"]]
        candidate_embeddings = embedding[batch["candidate_ids"]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = head(
                batch["hidden"],
                anchor_embedding,
                candidate_embeddings,
                batch["candidate_logits"],
                batch["base_logsumexp"],
            )
            local_loss = prefix_censored_nll(
                output.log_probs,
                output.outside_log_probs,
                batch["gold_candidate_indices"],
                batch["gold_in_lattice"],
            )
            global_distribution = absorbing_prefix_crf_conditionals(
                output.edge_scores,
                output.outside_log_mass,
                batch["base_logsumexp"],
            )
            global_loss = prefix_censored_nll(
                global_distribution.log_conditionals,
                global_distribution.outside_log_conditionals,
                batch["gold_candidate_indices"],
                batch["gold_in_lattice"],
            )
            selected_loss = (
                global_loss if normalization == "absorbing_crf" else local_loss
            )
        residual = output.residual_logits.float()
        # Position zero has one real anchor predecessor; its K identical rows
        # are only a static-shape implementation detail and must not be counted
        # K times in calibration diagnostics.
        real_residual = torch.cat(
            [residual[:, :1, :1].flatten(), residual[:, 1:].flatten()]
        )
        residual_abs = real_residual.abs()
        score_diagnostics["residual_abs_sum"] += float(residual_abs.sum())
        score_diagnostics["residual_square_sum"] += float(
            real_residual.square().sum()
        )
        score_diagnostics["residual_count"] += real_residual.numel()
        score_diagnostics["residual_abs_max"] = max(
            score_diagnostics["residual_abs_max"], float(residual_abs.max())
        )
        if batch["candidate_logits"].shape[-1] >= 2:
            top_two = batch["candidate_logits"].float().topk(2, dim=-1).values
            base_margin = top_two[..., 0] - top_two[..., 1]
            residual_range = residual.amax(dim=-1) - residual.amin(dim=-1)
            real_residual_range = torch.cat(
                [
                    residual_range[:, :1, :1].flatten(),
                    residual_range[:, 1:].flatten(),
                ]
            )
            comparable_base_margin = torch.cat(
                [
                    base_margin[:, :1].flatten(),
                    base_margin[:, 1:, None]
                    .expand_as(residual_range[:, 1:])
                    .flatten(),
                ]
            )
            score_diagnostics["base_top1_margin_sum"] += float(base_margin.sum())
            score_diagnostics["base_top1_margin_count"] += base_margin.numel()
            score_diagnostics["residual_next_range_sum"] += float(
                real_residual_range.sum()
            )
            score_diagnostics["residual_next_range_count"] += (
                real_residual_range.numel()
            )
            score_diagnostics["residual_range_exceeds_margin_count"] += int(
                (real_residual_range > comparable_base_margin).sum().item()
            )
        selected_losses.append(selected_loss.float().cpu())
        local_losses.append(local_loss.float().cpu())
        global_losses.append(global_loss.float().cpu())
        paths = {
            "base": torch.zeros_like(batch["gold_candidate_indices"]),
            "local": greedy_markov_decode(output.log_probs).path,
            "local_map": viterbi_decode(output.log_probs).path,
            "local_survival": survival_decode(output.log_probs).path,
            "global_map": viterbi_decode(
                global_distribution.log_conditionals
            ).path,
            "global_survival": survival_decode(
                global_distribution.log_conditionals
            ).path,
        }
        realized_by_method: dict[str, torch.Tensor] = {}
        first_by_method: dict[str, torch.Tensor] = {}
        for name, path in paths.items():
            prefix = realized_prefix(
                path, batch["candidate_ids"], batch["gold_ids"]
            )
            realized_by_method[name] = prefix
            accumulators[name].append(prefix.cpu())
            selected_first = batch["candidate_ids"][:, 0].gather(
                -1, path[:, :1]
            )[:, 0]
            first_correct = selected_first == batch["gold_ids"][:, 0]
            first_by_method[name] = first_correct
            accumulators[f"first_{name}"].append(first_correct.cpu())
        accumulators["predicted_global_survival_utility"].append(
            survival_decode(
                global_distribution.log_conditionals
            ).predicted_utility.float().cpu()
        )
        for left, right in decoder_pairs:
            path_diff = (paths[left] != paths[right]).any(dim=1)
            first_diff = paths[left][:, 0] != paths[right][:, 0]
            key = f"{left}_vs_{right}"
            stats = disagreement_stats[key]
            stats["examples"] += int(path_diff.numel())
            stats["path_disagreements"] += int(path_diff.sum().item())
            stats["first_token_disagreements"] += int(first_diff.sum().item())
            realized_delta = (
                realized_by_method[left] - realized_by_method[right]
            ).float()
            stats["realized_delta_on_path_disagreements"] += float(
                realized_delta[path_diff].sum().item()
            )
        for item_index, (sample_id, domain) in enumerate(
            zip(batch["sample_ids"], batch["domains"], strict=True)
        ):
            example_record = {
                "sample_id": sample_id,
                "domain": domain,
                "accepted_draft_tokens": {
                    name: int(values[item_index].item())
                    for name, values in realized_by_method.items()
                },
                "first_token_correct": {
                    name: bool(values[item_index].item())
                    for name, values in first_by_method.items()
                },
            }
            if include_examples:
                example_record["candidate_path_indices"] = {
                    name: path[item_index].detach().cpu().tolist()
                    for name, path in paths.items()
                }
            example_records.append(example_record)

    report: dict[str, Any] = {
        "normalization": normalization,
        "nll": float(torch.cat(selected_losses).mean()),
        "local_nll": float(torch.cat(local_losses).mean()),
        "absorbing_crf_nll": float(torch.cat(global_losses).mean()),
    }
    for name in [
        "base",
        "local",
        "local_map",
        "local_survival",
        "global_map",
        "global_survival",
    ]:
        values = torch.cat(accumulators[name]).float()
        first = torch.cat(accumulators[f"first_{name}"]).float()
        report[name] = {
            "mean_accepted_draft_tokens": float(values.mean()),
            "mean_verification_advance": float(values.mean() + 1.0),
            "first_token_accuracy": float(first.mean()),
        }
        prompt_values: dict[str, list[int]] = defaultdict(list)
        for record in example_records:
            prompt_values[record["sample_id"]].append(
                record["accepted_draft_tokens"][name]
            )
        report[name]["mean_accepted_draft_tokens_prompt_balanced"] = sum(
            sum(items) / len(items) for items in prompt_values.values()
        ) / len(prompt_values)
        report[name]["mean_verification_advance_prompt_balanced"] = (
            report[name]["mean_accepted_draft_tokens_prompt_balanced"] + 1.0
        )
    report["predicted_global_survival_utility"] = float(
        torch.cat(accumulators["predicted_global_survival_utility"]).mean()
    )
    residual_count = score_diagnostics["residual_count"]
    range_count = score_diagnostics["residual_next_range_count"]
    margin_count = score_diagnostics["base_top1_margin_count"]
    report["score_diagnostics"] = {
        "residual_abs_mean": score_diagnostics["residual_abs_sum"]
        / residual_count,
        "residual_rms": (
            score_diagnostics["residual_square_sum"] / residual_count
        )
        ** 0.5,
        "residual_abs_max": score_diagnostics["residual_abs_max"],
        "base_top1_margin_mean": (
            score_diagnostics["base_top1_margin_sum"] / margin_count
            if margin_count
            else None
        ),
        "residual_next_range_mean": (
            score_diagnostics["residual_next_range_sum"] / range_count
            if range_count
            else None
        ),
        "fraction_predecessor_rows_residual_range_exceeds_base_margin": (
            score_diagnostics["residual_range_exceeds_margin_count"]
            / range_count
            if range_count
            else None
        ),
        "learned_residual_scale": float(
            head.residual_scale.detach().float().item()
        ),
    }
    report["decoder_disagreement"] = {}
    for key, stats in disagreement_stats.items():
        examples = stats["examples"]
        path_disagreements = stats["path_disagreements"]
        report["decoder_disagreement"][key] = {
            **stats,
            "path_disagreement_fraction": path_disagreements / examples,
            "first_token_disagreement_fraction": stats[
                "first_token_disagreements"
            ]
            / examples,
            "mean_realized_delta_when_path_diff": (
                stats["realized_delta_on_path_disagreements"]
                / path_disagreements
                if path_disagreements
                else None
            ),
        }
    report["by_domain"] = {}
    for domain in sorted({record["domain"] for record in example_records}):
        subset = [record for record in example_records if record["domain"] == domain]
        report["by_domain"][domain] = {}
        for name in paths:
            accepted = [record["accepted_draft_tokens"][name] for record in subset]
            first = [record["first_token_correct"][name] for record in subset]
            report["by_domain"][domain][name] = {
                "blocks": len(subset),
                "mean_accepted_draft_tokens": sum(accepted) / len(accepted),
                "first_token_accuracy": sum(first) / len(first),
            }
    if include_examples:
        report["examples"] = example_records
    return report


def main() -> None:
    args = parse_args()
    if args.evidence_tier == "formal" and args.skip_test:
        raise ValueError("formal evidence cannot skip its frozen test split")
    if args.gate_split is not None and args.gate_split == args.validation_split:
        raise ValueError("gate split must differ from checkpoint-selection split")
    data_metadata_path = args.data / "metadata.json"
    data_metadata = json.loads(data_metadata_path.read_text())
    verified_target_embedding_files = validate_target_embedding_identity(
        data_metadata, args.target
    )
    run_provenance = {
        "project_commit": git_revision(PROJECT),
        "project_dirty_at_start": git_is_dirty(PROJECT),
        "data_metadata_sha256": sha256_file(data_metadata_path),
        "target_config_sha256": sha256_file(args.target / "config.json"),
        "trainer_sha256": sha256_file(Path(__file__)),
        "head_source_sha256": sha256_file(
            PROJECT / "src" / "sph" / "survival_path_head.py"
        ),
        "verified_target_embedding_files": verified_target_embedding_files,
        "dflash_commit": git_revision(PROJECT / "third_party" / "dflash"),
        "dflash_dirty_at_start": git_is_dirty(PROJECT / "third_party" / "dflash"),
        "domino_commit": git_revision(PROJECT / "third_party" / "Domino"),
        "domino_dirty_at_start": git_is_dirty(PROJECT / "third_party" / "Domino"),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    train_dataset = CanonicalBlockDataset(args.data, split=args.train_split)
    validation_dataset = CanonicalBlockDataset(
        args.data, split=args.validation_split, verify_integrity=False
    )
    gate_dataset = (
        None
        if args.gate_split is None
        else CanonicalBlockDataset(
            args.data, split=args.gate_split, verify_integrity=False
        )
    )
    test_dataset = (
        None
        if args.skip_test
        else CanonicalBlockDataset(
            args.data, split=args.test_split, verify_integrity=False
        )
    )
    assert_prompt_disjoint_splits(
        {
            "train": train_dataset,
            "validation": validation_dataset,
            "gate": gate_dataset,
            "test": test_dataset,
        }
    )
    train_loader = make_loader(
        train_dataset,
        candidate_k=args.candidate_k,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_dataset,
        candidate_k=args.candidate_k,
        batch_size=args.batch_size,
        shuffle=False,
    )
    gate_loader = (
        None
        if gate_dataset is None
        else make_loader(
            gate_dataset,
            candidate_k=args.candidate_k,
            batch_size=args.batch_size,
            shuffle=False,
        )
    )
    test_loader = (
        None
        if test_dataset is None
        else make_loader(
            test_dataset,
            candidate_k=args.candidate_k,
            batch_size=args.batch_size,
            shuffle=False,
        )
    )
    embedding = load_target_embedding(args.target).to(
        device=device, dtype=torch.bfloat16
    )
    hidden_size = int(embedding.shape[1])
    block_length = int(train_dataset.records[0]["gold_ids"].numel())
    if args.head_type == "no_mixer":
        head = SurvivalPathHead(hidden_size, rank=args.rank).to(device)
    else:
        head = BidirectionalSurvivalPathHead(
            hidden_size,
            rank=args.rank,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            max_positions=block_length,
        ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_advance = float("-inf")
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        train_losses = []
        train_nlls = []
        train_survival_losses = []
        train_regularizations = []
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            anchor_embedding = embedding[batch["anchor_ids"]]
            candidate_embeddings = embedding[batch["candidate_ids"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = head(
                    batch["hidden"],
                    anchor_embedding,
                    candidate_embeddings,
                    batch["candidate_logits"],
                    batch["base_logsumexp"],
                )
                if args.normalization == "absorbing_crf":
                    distribution = absorbing_prefix_crf_conditionals(
                        output.edge_scores,
                        output.outside_log_mass,
                        batch["base_logsumexp"],
                    )
                    censored_nll = prefix_censored_nll(
                        distribution.log_conditionals,
                        distribution.outside_log_conditionals,
                        batch["gold_candidate_indices"],
                        batch["gold_in_lattice"],
                    ).mean()
                    training_log_probs = distribution.log_conditionals
                else:
                    censored_nll = prefix_censored_nll(
                        output.log_probs,
                        output.outside_log_probs,
                        batch["gold_candidate_indices"],
                        batch["gold_in_lattice"],
                    ).mean()
                    training_log_probs = output.log_probs
                survival_loss = gold_prefix_survival_loss(
                    training_log_probs,
                    batch["gold_candidate_indices"],
                    batch["gold_in_lattice"],
                ).mean()
                regularization = output.residual_logits.square().mean()
                loss = (
                    censored_nll
                    + args.survival_loss_weight * survival_loss
                    + args.base_regularization * regularization
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            train_nlls.append(float(censored_nll.detach()))
            train_survival_losses.append(float(survival_loss.detach()))
            train_regularizations.append(float(regularization.detach()))

        validation = evaluate(
            head,
            validation_loader,
            embedding,
            device,
            args.normalization,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / len(train_losses),
            "train_nll": sum(train_nlls) / len(train_nlls),
            "train_survival_loss": sum(train_survival_losses)
            / len(train_survival_losses),
            "train_residual_regularization": sum(train_regularizations)
            / len(train_regularizations),
            "validation": validation,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2), flush=True)
        selection_decoder = (
            "global_survival"
            if args.normalization == "absorbing_crf"
            else "local_survival"
        )
        # Anchors from one prompt are correlated and short generations can
        # yield fewer anchors.  Select checkpoints with one equal-weight vote
        # per prompt instead of silently overweighting prompts with more
        # collected blocks.
        advance = validation[selection_decoder][
            "mean_verification_advance_prompt_balanced"
        ]
        if advance > best_advance:
            best_advance = advance
            torch.save(
                {
                    "model": head.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "validation": validation,
                },
                args.output / "best.pt",
            )

    best_checkpoint = torch.load(
        args.output / "best.pt", map_location=device, weights_only=False
    )
    head.load_state_dict(best_checkpoint["model"])
    final_validation = evaluate(
        head,
        validation_loader,
        embedding,
        device,
        args.normalization,
        include_examples=True,
    )
    final_gate = (
        None
        if gate_loader is None
        else evaluate(
            head,
            gate_loader,
            embedding,
            device,
            args.normalization,
            include_examples=True,
        )
    )
    # This is a post-selection diagnostic only. It is never used for checkpoint
    # selection, but makes underfitting distinguishable from memorization.
    final_train = evaluate(
        head,
        make_loader(
            train_dataset,
            candidate_k=args.candidate_k,
            batch_size=args.batch_size,
            shuffle=False,
        ),
        embedding,
        device,
        args.normalization,
    )
    final_test = (
        None
        if test_loader is None
        else evaluate(
            head,
            test_loader,
            embedding,
            device,
            args.normalization,
            include_examples=True,
        )
    )

    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config": vars(args),
        "data": str(args.data.resolve()),
        "target": str(args.target.resolve()),
        "train_blocks": len(train_dataset),
        "validation_blocks": len(validation_dataset),
        "gate_blocks": len(gate_dataset) if gate_dataset is not None else 0,
        "gate_status": (
            "not_configured"
            if gate_dataset is None
            else "evaluated_once_after_selection"
        ),
        "test_blocks": len(test_dataset) if test_dataset is not None else 0,
        "test_status": (
            "skipped_reserved_unobserved"
            if test_dataset is None
            else "evaluated_once_after_selection"
        ),
        "head_type": args.head_type,
        "normalization": args.normalization,
        "evidence_tier": args.evidence_tier,
        "seed": args.seed,
        "candidate_k": args.candidate_k,
        "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "seconds": time.perf_counter() - start,
        "selected_epoch": int(best_checkpoint["epoch"]),
        "final_train_diagnostic": final_train,
        "final_validation": final_validation,
        "final_gate": final_gate,
        "final_test": final_test,
        "history": history,
        "provenance": run_provenance,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "history"},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
