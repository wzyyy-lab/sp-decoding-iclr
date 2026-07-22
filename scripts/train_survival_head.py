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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
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


def load_target_embedding(target: Path) -> torch.Tensor:
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "model.embed_tokens.weight"
    shard_name = index["weight_map"][key]
    with safe_open(target / shard_name, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


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
        for item_index, (sample_id, domain) in enumerate(
            zip(batch["sample_ids"], batch["domains"], strict=True)
        ):
            example_records.append(
                {
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
            )

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
    report["predicted_global_survival_utility"] = float(
        torch.cat(accumulators["predicted_global_survival_utility"]).mean()
    )
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
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    train_dataset = CanonicalBlockDataset(args.data, split=args.train_split)
    validation_dataset = CanonicalBlockDataset(
        args.data, split=args.validation_split
    )
    test_dataset = CanonicalBlockDataset(args.data, split=args.test_split)
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
    test_loader = make_loader(
        test_dataset,
        candidate_k=args.candidate_k,
        batch_size=args.batch_size,
        shuffle=False,
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
                else:
                    censored_nll = prefix_censored_nll(
                        output.log_probs,
                        output.outside_log_probs,
                        batch["gold_candidate_indices"],
                        batch["gold_in_lattice"],
                    ).mean()
                regularization = output.residual_logits.square().mean()
                loss = censored_nll + args.base_regularization * regularization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))

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
            "validation": validation,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2), flush=True)
        selection_decoder = (
            "global_survival"
            if args.normalization == "absorbing_crf"
            else "local_survival"
        )
        advance = validation[selection_decoder]["mean_verification_advance"]
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
    final_test = evaluate(
        head,
        test_loader,
        embedding,
        device,
        args.normalization,
        include_examples=True,
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
        "test_blocks": len(test_dataset),
        "head_type": args.head_type,
        "normalization": args.normalization,
        "evidence_tier": args.evidence_tier,
        "seed": args.seed,
        "candidate_k": args.candidate_k,
        "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "seconds": time.perf_counter() - start,
        "selected_epoch": int(best_checkpoint["epoch"]),
        "final_validation": final_validation,
        "final_test": final_test,
        "history": history,
        "provenance": {
            "project_commit": git_revision(PROJECT),
            "data_metadata_sha256": sha256_file(args.data / "metadata.json"),
            "target_config_sha256": sha256_file(args.target / "config.json"),
            "trainer_sha256": sha256_file(Path(__file__)),
            "head_source_sha256": sha256_file(
                PROJECT / "src" / "sph" / "survival_path_head.py"
            ),
            "dflash_commit": git_revision(PROJECT / "third_party" / "dflash"),
            "domino_commit": git_revision(PROJECT / "third_party" / "Domino"),
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
