#!/usr/bin/env python3
"""Claim-bearing 180K-step joint DFlash + parallel PARC-16 training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any, Sequence

from safetensors import safe_open
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel

from sph.fbpf import dpace_loss
from sph.parc import (
    BLOCK_LENGTH,
    CANDIDATES,
    EXPECTED_PARAMETER_COUNT,
    PARC16Head,
    PARCOutput,
    PURE_DFLASH_INPUT_LENGTH,
    assert_frozen_architecture,
    nonshift_full16_prediction_hidden,
    parc_fixed_reference_loss,
)
from sph.parc_training import (
    BlockStream,
    DataCatalog,
    accepted_lengths,
    checkpoint_is_better,
    cosine_learning_rate,
    grouped_prompt_metrics,
    iter_prompt_records,
    load_data_catalog,
    numeric_certificate,
    select_train_audit_ids,
)


FROZEN_BATCH_SIZE = 8
FROZEN_TOTAL_STEPS = 180_000
FROZEN_EVAL_EVERY = 10_000
FROZEN_SAVE_EVERY = 1_000
FROZEN_WARMUP = 2_000
FROZEN_HEAD_LR = 3e-4
FROZEN_DFLASH_LR = 1e-5
FROZEN_SEED = 0
FROZEN_WEIGHT_DECAY = 0.0
FROZEN_GRADIENT_CLIP = 1.0
EXPECTED_DFLASH_PARAMETERS = 537_427_200
CONSTRAINT_LIMIT = 0.01
DUAL_EMA_DECAY = 0.95
DUAL_LEARNING_RATE = 0.05
DUAL_MAXIMUM = 100.0


_STOP_REQUESTED = False


def _request_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"received signal {signum}; checkpointing after the current step", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    parser.add_argument("--total-steps", type=int, default=FROZEN_TOTAL_STEPS)
    parser.add_argument("--eval-every", type=int, default=FROZEN_EVAL_EVERY)
    parser.add_argument("--save-every", type=int, default=FROZEN_SAVE_EVERY)
    parser.add_argument("--warmup-steps", type=int, default=FROZEN_WARMUP)
    parser.add_argument("--head-learning-rate", type=float, default=FROZEN_HEAD_LR)
    parser.add_argument("--dflash-learning-rate", type=float, default=FROZEN_DFLASH_LR)
    parser.add_argument("--weight-decay", type=float, default=FROZEN_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip", type=float, default=FROZEN_GRADIENT_CLIP)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def enforce_frozen_recipe(args: argparse.Namespace) -> None:
    expected = {
        "batch_size": FROZEN_BATCH_SIZE,
        "total_steps": FROZEN_TOTAL_STEPS,
        "eval_every": FROZEN_EVAL_EVERY,
        "save_every": FROZEN_SAVE_EVERY,
        "warmup_steps": FROZEN_WARMUP,
        "head_learning_rate": FROZEN_HEAD_LR,
        "dflash_learning_rate": FROZEN_DFLASH_LR,
        "weight_decay": FROZEN_WEIGHT_DECAY,
        "gradient_clip": FROZEN_GRADIENT_CLIP,
        "seed": FROZEN_SEED,
    }
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise RuntimeError(
                f"claim-bearing PARC recipe requires --{key.replace('_', '-')}={value}"
            )
    if args.attn_implementation != "sdpa":
        raise RuntimeError("claim-bearing PARC training is frozen to SDPA")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_target_weight(target: Path) -> tuple[Tensor, str]:
    config = json.loads((target / "config.json").read_text())
    index_path = target / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        key = (
            "lm_head.weight"
            if "lm_head.weight" in weight_map
            else "model.embed_tokens.weight"
        )
        if key != "lm_head.weight" and not bool(config.get("tie_word_embeddings")):
            raise RuntimeError("target checkpoint has no resolvable LM-head weight")
        shard = target / str(weight_map[key])
    else:
        shard = target / "model.safetensors"
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        key = "lm_head.weight" if "lm_head.weight" in keys else "model.embed_tokens.weight"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key)
    if weight.ndim != 2 or tuple(weight.shape) != (151_936, 2_560):
        raise RuntimeError(f"unexpected target lexical table {tuple(weight.shape)}")
    return weight, key


def configure_trainable_dflash(draft: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    for parameter in draft.parameters():
        parameter.requires_grad_(False)
    for module_name in ("layers", "norm", "fc", "hidden_norm"):
        module = getattr(draft, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    named = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in draft.named_parameters()
                if parameter.requires_grad
            ),
            key=lambda item: item[0],
        )
    )
    count = sum(parameter.numel() for _, parameter in named)
    if count != EXPECTED_DFLASH_PARAMETERS:
        raise RuntimeError(
            f"trainable DFlash parameters {count} != {EXPECTED_DFLASH_PARAMETERS}"
        )
    return named


class MasterAdamW:
    """FP32-master AdamW for the BF16 DFlash backbone."""

    def __init__(
        self,
        named_parameters: Sequence[tuple[str, nn.Parameter]],
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        self.named_parameters = tuple(named_parameters)
        self.masters = [
            nn.Parameter(parameter.detach().float().clone(), requires_grad=True)
            for _, parameter in self.named_parameters
        ]
        self.optimizer = torch.optim.AdamW(
            self.masters,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        for _, parameter in self.named_parameters:
            parameter.grad = None

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    @torch.no_grad()
    def step_from_live_gradients(self) -> None:
        for master, (_, parameter) in zip(
            self.masters, self.named_parameters, strict=True
        ):
            master.grad = (
                None if parameter.grad is None else parameter.grad.float().clone()
            )
        self.optimizer.step()
        for master, (_, parameter) in zip(
            self.masters, self.named_parameters, strict=True
        ):
            parameter.copy_(master.to(dtype=parameter.dtype))

    def state_dict(self) -> dict[str, Any]:
        return {
            "names": [name for name, _ in self.named_parameters],
            "masters": [master.detach().cpu() for master in self.masters],
            "optimizer": self.optimizer.state_dict(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        names = [name for name, _ in self.named_parameters]
        if state.get("names") != names or len(state.get("masters", [])) != len(names):
            raise RuntimeError("DFlash optimizer parameter layout changed on resume")
        for master, saved in zip(self.masters, state["masters"], strict=True):
            master.copy_(saved.to(device=master.device, dtype=torch.float32))
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in tuple(optimizer_state.items()):
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to(self.masters[0].device)


@dataclass
class ForwardBatch:
    output: PARCOutput
    candidate_ids: Tensor
    base_logits: Tensor
    gold_ids: Tensor
    reference_accepted: Tensor
    reference_delta: Tensor
    sample_ids: list[str]
    domains: list[str]
    reference_dflash_accepted: list[int]
    reference_domino_accepted: list[int | None]


def _block_input(
    record: dict[str, Any], anchor_index: int
) -> tuple[dict[str, Any], Tensor]:
    anchors = record["anchors"]
    if not 0 <= anchor_index < len(anchors):
        raise ValueError("anchor index lies outside the record")
    return anchors[anchor_index], record["target_context_features"]


def forward_blocks(
    *,
    draft: nn.Module,
    head: PARC16Head,
    target_weight: Tensor,
    items: Sequence[tuple[dict[str, Any], int]],
) -> ForwardBatch:
    if not items:
        raise ValueError("PARC forward batch is empty")
    device = target_weight.device
    hidden_rows: list[Tensor] = []
    gold_rows: list[Tensor] = []
    anchors: list[int] = []
    reference_accepted: list[int] = []
    reference_delta: list[float] = []
    sample_ids: list[str] = []
    domains: list[str] = []
    domino_accepted: list[int | None] = []
    context_cache: dict[int, Tensor] = {}
    for record, anchor_index in items:
        anchor, cpu_features = _block_input(record, anchor_index)
        record_key = id(record)
        if record_key not in context_cache:
            context_cache[record_key] = cpu_features.to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
        features = context_cache[record_key]
        context_length = int(anchor["context_length"])
        anchor_token = int(anchor["anchor_token_id"])
        block_ids = torch.full(
            (1, PURE_DFLASH_INPUT_LENGTH),
            int(draft.mask_token_id),
            dtype=torch.long,
            device=device,
        )
        block_ids[0, 0] = anchor_token
        position_ids = torch.arange(
            context_length + PURE_DFLASH_INPUT_LENGTH,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        raw_hidden = draft(
            target_hidden=features[None, :context_length],
            noise_embedding=F.embedding(block_ids, target_weight),
            position_ids=position_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        hidden = nonshift_full16_prediction_hidden(raw_hidden)
        if tuple(hidden.shape) != (1, BLOCK_LENGTH, 2_560):
            raise RuntimeError(f"DFlash returned {tuple(hidden.shape)}, expected [1,16,2560]")
        hidden_rows.append(hidden)
        gold_rows.append(anchor["gold_ids"].long().to(device, non_blocking=True))
        anchors.append(anchor_token)
        reference_accepted.append(int(anchor["reference_accepted_length"]))
        reference_delta.append(float(anchor["reference_delta_fp32"]))
        sample_ids.append(str(record["sample_id"]))
        domains.append(str(record["domain"]))
        domino_accepted.append(
            int(anchor["reference_domino_accepted_length"])
            if "reference_domino_accepted_length" in anchor
            else None
        )

    output_rows: list[PARCOutput] = []
    candidate_id_rows: list[Tensor] = []
    base_logit_rows: list[Tensor] = []
    for hidden, anchor_token in zip(hidden_rows, anchors, strict=True):
        # Preserve the production batch-1 GEMM and head geometry for every chain.
        base_logits = F.linear(hidden, target_weight)
        candidate_logits, candidate_ids = base_logits.float().topk(
            CANDIDATES, dim=-1, sorted=True
        )
        anchor_tensor = torch.tensor([anchor_token], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = head(
                hidden,
                candidate_logits,
                F.embedding(anchor_tensor, target_weight),
                F.embedding(candidate_ids, target_weight),
            )
        output_rows.append(output)
        candidate_id_rows.append(candidate_ids)
        base_logit_rows.append(base_logits)
    return ForwardBatch(
        output=PARCOutput(
            scores=torch.cat([row.scores for row in output_rows], dim=0),
            residual_advantages=torch.cat(
                [row.residual_advantages for row in output_rows], dim=0
            ),
            candidate_states=torch.cat(
                [row.candidate_states for row in output_rows], dim=0
            ),
        ),
        candidate_ids=torch.cat(candidate_id_rows, dim=0),
        base_logits=torch.cat(base_logit_rows, dim=0),
        gold_ids=torch.stack(gold_rows),
        reference_accepted=torch.tensor(
            reference_accepted, dtype=torch.long, device=device
        ),
        reference_delta=torch.tensor(
            reference_delta, dtype=torch.float32, device=device
        ),
        sample_ids=sample_ids,
        domains=domains,
        reference_dflash_accepted=reference_accepted,
        reference_domino_accepted=domino_accepted,
    )


def training_loss(
    forward: ForwardBatch,
    *,
    dual_lambda: float,
    delta_min: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    base = dpace_loss(
        forward.base_logits,
        forward.gold_ids,
        reduction_divisor=int(forward.gold_ids.shape[0]),
    ) / float(BLOCK_LENGTH)
    fixed = parc_fixed_reference_loss(
        forward.output,
        forward.candidate_ids,
        forward.gold_ids,
        forward.reference_accepted,
        forward.reference_delta,
        delta_min=delta_min,
    )
    constraint = fixed.harm_upper_bound.mean() - CONSTRAINT_LIMIT
    primal = base + fixed.gain_loss + float(dual_lambda) * constraint
    return primal, {
        "base_loss": base,
        "gain_loss": fixed.gain_loss,
        "constraint": constraint,
        "harm_upper_bound": fixed.harm_upper_bound.mean(),
        "actual_harm": fixed.actual_harm.mean(),
        "support_drop": fixed.support_drop.float().mean(),
        "ambiguous": fixed.ambiguous.float().mean(),
        "conditional_gain": fixed.conditional_gain.mean(),
    }


@torch.no_grad()
def gradient_norms(
    dflash_parameters: Sequence[tuple[str, nn.Parameter]],
    head: nn.Module,
) -> tuple[float, float, float]:
    dflash_squared = torch.zeros((), device="cuda:0")
    head_squared = torch.zeros((), device="cuda:0")
    for _, parameter in dflash_parameters:
        if parameter.grad is not None:
            dflash_squared += parameter.grad.float().square().sum()
    for parameter in head.parameters():
        if parameter.grad is not None:
            head_squared += parameter.grad.float().square().sum()
    total = dflash_squared + head_squared
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("non-finite PARC joint gradient norm")
    return (
        float(torch.sqrt(total).item()),
        float(torch.sqrt(dflash_squared).item()),
        float(torch.sqrt(head_squared).item()),
    )


@torch.no_grad()
def scale_gradients(parameters: Sequence[nn.Parameter], scale: float) -> None:
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


@torch.inference_mode()
def evaluate(
    *,
    catalog: DataCatalog,
    split: str,
    draft: nn.Module,
    head: PARC16Head,
    target_weight: Tensor,
    delta_min: float,
    sample_ids: set[str] | None = None,
    require_reference_identity: bool = False,
) -> dict[str, Any]:
    draft_was_training = draft.training
    head_was_training = head.training
    draft.eval()
    head.eval()
    rows: list[dict[str, Any]] = []
    token_mismatches = 0
    length_mismatches = 0
    started = time.perf_counter()
    for prompt_index, record in enumerate(
        iter_prompt_records(catalog, split, sample_ids=sample_ids), start=1
    ):
        items = [(record, anchor_index) for anchor_index in range(8)]
        forward = forward_blocks(
            draft=draft, head=head, target_weight=target_weight, items=items
        )
        proposal = PARC16Head.proposal_ids(forward.candidate_ids, forward.output)
        lengths = accepted_lengths(proposal, forward.gold_ids)
        fixed = parc_fixed_reference_loss(
            forward.output,
            forward.candidate_ids,
            forward.gold_ids,
            forward.reference_accepted,
            forward.reference_delta,
            delta_min=delta_min,
        )
        if require_reference_identity:
            for row_index, anchor in enumerate(record["anchors"]):
                reference_ids = anchor["reference_topk_ids"].long().to(proposal.device)
                reference_proposal = anchor["reference_proposal_ids"].long().to(
                    proposal.device
                )
                token_mismatches += int(
                    not torch.equal(forward.candidate_ids[row_index], reference_ids)
                    or not torch.equal(proposal[row_index], reference_proposal)
                )
                length_mismatches += int(
                    int(lengths[row_index])
                    != int(anchor["reference_accepted_length"])
                )
        for row_index in range(len(items)):
            domino_length = forward.reference_domino_accepted[row_index]
            rows.append(
                {
                    "sample_id": forward.sample_ids[row_index],
                    "domain": forward.domains[row_index],
                    "eal": int(lengths[row_index]),
                    "actual_harm": float(fixed.actual_harm[row_index]),
                    "harm_upper_bound": float(fixed.harm_upper_bound[row_index]),
                    "support_drop": float(fixed.support_drop[row_index]),
                    "ambiguous": float(fixed.ambiguous[row_index]),
                    "dflash_eal": forward.reference_dflash_accepted[row_index],
                    **(
                        {"domino_eal": int(domino_length)}
                        if domino_length is not None
                        else {}
                    ),
                }
            )
        if prompt_index % 100 == 0:
            print(
                f"evaluation {split}: {prompt_index} prompts in "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    metrics = grouped_prompt_metrics(rows)
    overall = metrics["overall"]
    if "domino_eal" in overall:
        overall["ratio_vs_domino"] = float(overall["eal"]) / float(
            overall["domino_eal"]
        )
        for domain in metrics["by_domain"].values():
            domain["ratio_vs_domino"] = float(domain["eal"]) / float(
                domain["domino_eal"]
            )
    metrics.update(
        {
            "split": split,
            "seconds": time.perf_counter() - started,
            "reference_topk_or_proposal_mismatches": token_mismatches,
            "reference_length_mismatches": length_mismatches,
        }
    )
    if require_reference_identity and (token_mismatches or length_mismatches):
        raise RuntimeError(
            "step-0 validation no longer reproduces the materialized pure-DFlash reference"
        )
    if draft_was_training:
        draft.train()
    if head_was_training:
        head.train()
    return metrics


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_run_status(
    *,
    global_step: int,
    total_steps: int,
    eval_every: int,
    stop_reason: str | None,
    best_step: int | None,
) -> tuple[str, str | None, bool]:
    """Resolve terminal/resumable state without masking scientific stops."""

    eligible_best = best_step in range(
        eval_every, total_steps + 1, eval_every
    )
    if stop_reason is not None and stop_reason != "scheduler_checkpoint_request":
        status = "stopped_infeasible"
    elif global_step == total_steps:
        if not eligible_best:
            stop_reason = "no_trained_validation_checkpoint_passed_harm_gate"
            status = "stopped_infeasible"
        else:
            status = "complete"
    elif stop_reason == "scheduler_checkpoint_request":
        status = "interrupted_resumable"
    else:
        status = "stopped_infeasible"
    heldout_authorized = status == "complete" and eligible_best
    return status, stop_reason, heldout_authorized


def cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def run_config(
    args: argparse.Namespace,
    *,
    catalog: DataCatalog,
    certificate: dict[str, Any],
    audit_ids: set[str],
) -> dict[str, Any]:
    return {
        "format": "parc16_joint_training_config_v1",
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "domino_draft": str(args.domino_draft.resolve()),
        "data_root": str(catalog.root),
        "attention_implementation": args.attn_implementation,
        "batch_size": args.batch_size,
        "total_steps": args.total_steps,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "warmup_steps": args.warmup_steps,
        "head_learning_rate": args.head_learning_rate,
        "dflash_learning_rate": args.dflash_learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "head_parameters": EXPECTED_PARAMETER_COUNT,
        "dflash_parameters": EXPECTED_DFLASH_PARAMETERS,
        "train_prompt_records": catalog.prompt_count("train"),
        "validation_prompt_records": catalog.prompt_count("validation"),
        "numeric_certificate": certificate,
        "train_audit_sample_ids": sorted(audit_ids),
        "online_target_model_used": False,
        "old_15_position_cache_used": False,
        "serial_selected_token_feedback": False,
        "multi_path_output": False,
        "pure_dflash_geometry": "non_shift_raw17_slice_rows_1_through_16",
        "released_domino_geometry": "shift_label_raw16_all_rows",
        "extra_pure_dflash_carrier_row_must_be_costed": True,
    }


def main() -> None:
    args = parse_args()
    enforce_frozen_recipe(args)
    if not torch.cuda.is_available():
        raise RuntimeError("formal PARC training requires CUDA")
    torch.cuda.set_device(0)
    seed_everything(args.seed)
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    if args.resume is None:
        if args.output.exists():
            raise FileExistsError(f"refusing existing PARC output {args.output}")
        args.output.mkdir(parents=True)
    else:
        if not args.resume.is_file() or args.resume.parent != args.output:
            raise RuntimeError("resume checkpoint must be inside the requested output")
        args.output.mkdir(parents=True, exist_ok=True)

    catalog = load_data_catalog(
        args.data_root,
        target=args.target,
        draft=args.draft,
        domino_draft=args.domino_draft,
    )
    certificate = numeric_certificate(catalog)
    if float(certificate["prompt_mean_ambiguous"]) > 0.01:
        raise RuntimeError(
            "train-only numeric ambiguity exceeds the frozen 1% launch gate"
        )
    audit_ids = select_train_audit_ids(catalog)
    config = run_config(
        args, catalog=catalog, certificate=certificate, audit_ids=audit_ids
    )

    target_weight_cpu, target_weight_key = load_target_weight(args.target)
    target_weight = target_weight_cpu.to("cuda:0", torch.bfloat16)
    del target_weight_cpu
    draft = AutoModel.from_pretrained(
        str(args.draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    )
    if int(draft.block_size) != BLOCK_LENGTH:
        raise RuntimeError("formal PARC training requires pure DFlash B16")
    if getattr(draft.config, "dflash_config", {}).get("projector_type") is not None:
        raise RuntimeError("formal PARC training loaded a non-pure DFlash checkpoint")
    if bool(getattr(draft.config, "dflash_config", {}).get("shift_label", False)):
        raise RuntimeError("formal PARC raw17/slice path requires non-shift pure DFlash")
    dflash_named = configure_trainable_dflash(draft)
    head = PARC16Head().to("cuda:0")
    assert_frozen_architecture(head)
    head_named = tuple(sorted(head.named_parameters(), key=lambda item: item[0]))

    dflash_optimizer = MasterAdamW(
        dflash_named,
        learning_rate=args.dflash_learning_rate,
        weight_decay=args.weight_decay,
    )
    head_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in head_named],
        lr=args.head_learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    global_step = 0
    dual_lambda = 0.0
    dual_ema = 0.0
    best_metrics: dict[str, Any] | None = None
    best_step: int | None = None
    support_drop_windows: list[float] = []
    sampler_state: dict[str, int] | None = None
    if args.resume is not None:
        report_path = args.output / "report.json"
        if report_path.exists():
            previous_status = str(json.loads(report_path.read_text()).get("status"))
            if previous_status != "interrupted_resumable":
                raise RuntimeError(
                    f"refusing to resume terminal PARC status {previous_status!r}"
                )
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("format") != "parc16_joint_latest_v1":
            raise RuntimeError("resume checkpoint format differs from formal PARC")
        if payload.get("config") != config:
            raise RuntimeError("resume config differs from the frozen PARC run")
        saved_stop = payload.get("stop_reason")
        if saved_stop not in {None, "scheduler_checkpoint_request"}:
            raise RuntimeError(f"refusing terminal PARC checkpoint {saved_stop!r}")
        if int(payload.get("global_step", -1)) >= args.total_steps:
            raise RuntimeError("refusing to resume a completed PARC checkpoint")
        draft.load_state_dict(payload["draft_state"], strict=True)
        head.load_state_dict(payload["head_state"], strict=True)
        dflash_optimizer.load_state_dict(payload["dflash_optimizer"])
        head_optimizer.load_state_dict(payload["head_optimizer"])
        for optimizer_state in head_optimizer.state.values():
            for key, value in tuple(optimizer_state.items()):
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to("cuda:0")
        global_step = int(payload["global_step"])
        dual_lambda = float(payload["dual_lambda"])
        dual_ema = float(payload["dual_ema"])
        best_metrics = payload["best_metrics"]
        best_step = payload["best_step"]
        support_drop_windows = [float(value) for value in payload["support_drop_windows"]]
        sampler_state = payload["sampler_state"]
        random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        print(f"resumed formal PARC at step {global_step}", flush=True)
    else:
        (args.output / "config.json").write_text(
            json.dumps(
                {**config, "target_weight_key": target_weight_key},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    stream = BlockStream(catalog, seed=args.seed, state=sampler_state)
    delta_min = float(certificate["delta_min"])
    metrics_path = args.output / "metrics.jsonl"
    started = time.perf_counter()

    if global_step == 0:
        validation = evaluate(
            catalog=catalog,
            split="train",
            draft=draft,
            head=head,
            target_weight=target_weight,
            delta_min=delta_min,
            sample_ids=audit_ids,
            require_reference_identity=True,
        )
        event = {
            "type": "step0_train_audit_parity_selection_ineligible",
            "step": 0,
            "metrics": validation,
        }
        append_jsonl(metrics_path, event)
        print(json.dumps(event, ensure_ascii=False), flush=True)

    rolling: list[dict[str, float]] = []
    stop_reason: str | None = None

    def save_latest() -> None:
        atomic_torch_save(
            args.output / "latest.pt",
            {
                "format": "parc16_joint_latest_v1",
                "config": config,
                "global_step": global_step,
                "dual_lambda": dual_lambda,
                "dual_ema": dual_ema,
                "best_metrics": best_metrics,
                "best_step": best_step,
                "stop_reason": stop_reason,
                "support_drop_windows": support_drop_windows,
                "sampler_state": stream.state_dict(),
                "draft_state": cpu_state_dict(draft),
                "head_state": cpu_state_dict(head),
                "dflash_optimizer": dflash_optimizer.state_dict(),
                "head_optimizer": head_optimizer.state_dict(),
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
        )

    while global_step < args.total_steps:
        draft.train()
        head.train()
        dflash_optimizer.zero_grad()
        head_optimizer.zero_grad(set_to_none=True)
        dflash_lr = cosine_learning_rate(
            global_step,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
            peak=args.dflash_learning_rate,
        )
        head_lr = cosine_learning_rate(
            global_step,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
            peak=args.head_learning_rate,
        )
        dflash_optimizer.set_learning_rate(dflash_lr)
        for group in head_optimizer.param_groups:
            group["lr"] = head_lr

        items = stream.next_batch(args.batch_size)
        forward = forward_blocks(
            draft=draft, head=head, target_weight=target_weight, items=items
        )
        primal, parts = training_loss(
            forward, dual_lambda=dual_lambda, delta_min=delta_min
        )
        if not bool(torch.isfinite(primal).item()):
            raise FloatingPointError("non-finite PARC primal loss")
        primal.backward()
        total_norm, dflash_norm, head_norm = gradient_norms(dflash_named, head)
        clip_scale = min(1.0, args.gradient_clip / (total_norm + 1e-12))
        scale_gradients(
            [parameter for _, parameter in dflash_named]
            + [parameter for _, parameter in head_named],
            clip_scale,
        )
        dflash_optimizer.step_from_live_gradients()
        head_optimizer.step()
        global_step += 1

        violation = float(parts["constraint"].detach())
        dual_ema = DUAL_EMA_DECAY * dual_ema + (1.0 - DUAL_EMA_DECAY) * violation
        dual_lambda = min(
            DUAL_MAXIMUM,
            max(0.0, dual_lambda + DUAL_LEARNING_RATE * dual_ema),
        )
        row = {
            key: float(value.detach()) for key, value in parts.items()
        }
        row.update(
            {
                "primal": float(primal.detach()),
                "gradient_norm": total_norm,
                "dflash_gradient_norm": dflash_norm,
                "head_gradient_norm": head_norm,
                "clip_scale": clip_scale,
                "dual_lambda": dual_lambda,
                "dual_ema": dual_ema,
                "dflash_lr": dflash_lr,
                "head_lr": head_lr,
            }
        )
        rolling.append(row)
        if global_step % args.log_every == 0:
            averaged = {
                key: sum(item[key] for item in rolling) / len(rolling)
                for key in rolling[0]
            }
            event = {"type": "train", "step": global_step, "metrics": averaged}
            append_jsonl(metrics_path, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
            rolling = []

        if dual_lambda >= DUAL_MAXIMUM and dual_ema > 0:
            stop_reason = "constraint_infeasible_dual_saturation"

        if global_step % args.eval_every == 0:
            validation = evaluate(
                catalog=catalog,
                split="validation",
                draft=draft,
                head=head,
                target_weight=target_weight,
                delta_min=delta_min,
            )
            validation_event = {
                "type": "validation",
                "step": global_step,
                "metrics": validation,
            }
            append_jsonl(metrics_path, validation_event)
            print(json.dumps(validation_event, ensure_ascii=False), flush=True)
            if checkpoint_is_better(validation, best_metrics):
                best_metrics = validation
                best_step = global_step
                atomic_torch_save(
                    args.output / "best.pt",
                    {
                        "format": "parc16_joint_best_v1",
                        "config": config,
                        "step": global_step,
                        "metrics": validation,
                        "draft_state": cpu_state_dict(draft),
                        "head_state": cpu_state_dict(head),
                    },
                )
            audit = evaluate(
                catalog=catalog,
                split="train",
                draft=draft,
                head=head,
                target_weight=target_weight,
                delta_min=delta_min,
                sample_ids=audit_ids,
            )
            audit_event = {
                "type": "train_audit",
                "step": global_step,
                "metrics": audit,
            }
            append_jsonl(metrics_path, audit_event)
            print(json.dumps(audit_event, ensure_ascii=False), flush=True)
            if global_step >= 20_000:
                support_drop_windows.append(float(audit["overall"]["support_drop"]))
                if len(support_drop_windows) >= 4:
                    window = support_drop_windows[-4:]
                    if all(value > 0.01 for value in window) and window[-1] > 0.8 * window[0]:
                        stop_reason = "constraint_infeasible_support_drop"

        if _STOP_REQUESTED and stop_reason is None:
            stop_reason = "scheduler_checkpoint_request"
        if global_step % args.save_every == 0 or stop_reason:
            save_latest()
        if stop_reason:
            break

    status, stop_reason, heldout_authorized = resolve_run_status(
        global_step=global_step,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
        stop_reason=stop_reason,
        best_step=best_step,
    )
    save_latest()
    if status == "complete" and not heldout_authorized:
        raise RuntimeError("complete PARC run has no eligible trained checkpoint")
    report = {
        "format": "parc16_joint_training_report_v1",
        "status": status,
        "stop_reason": stop_reason,
        "global_step": global_step,
        "best_step": best_step,
        "best_validation": best_metrics,
        "elapsed_seconds_this_process": time.perf_counter() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "heldout_opened": False,
        "heldout_authorized": heldout_authorized,
        "training_eal_is_claim_evidence": False,
        "validation_used_only_for_checkpoint_selection": True,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if heldout_authorized:
        (args.output / "checkpoint_lock.json").write_text(
            json.dumps(
                {
                    "format": "parc16_checkpoint_lock_v1",
                    "best_checkpoint": str((args.output / "best.pt").resolve()),
                    "best_step": best_step,
                    "best_validation": best_metrics,
                    "weights_and_config_locked": True,
                    "heldout_opened": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
