"""Diagnostic-only R056 production-logit tracing.

This module is injected with ``PYTHONPATH`` only by the parity-microscope
launcher.  It deliberately monkey-patches the live SGLang methods rather than
adding synchronization or logging to the production/timed implementation.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any


_TRACE_PATH = os.environ.get("R056_TRACE_PATH")
_TRACE_MODE = os.environ.get("R056_TRACE_MODE", "").lower()
_TRACE_RIDS = {
    value for value in os.environ.get("R056_TRACE_RIDS", "").split(",") if value
}
_TARGET_STEPS: dict[str, int] = defaultdict(int)
_VERIFY_CYCLES: dict[tuple[str, str], int] = defaultdict(int)


def _load_authority() -> dict[str, list[int]]:
    source = os.environ.get("R056_TRACE_AUTHORITY_JSON")
    if not source:
        return {}
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    return {
        str(row["sample_id"]): [int(token) for token in row["output_ids"]]
        for row in payload.get("records", [])
    }


_AUTHORITY = _load_authority() if _TRACE_PATH else {}


def _enabled(rid: Any) -> bool:
    return bool(_TRACE_PATH) and str(rid) in _TRACE_RIDS


def _append(record: dict[str, Any]) -> None:
    if not _TRACE_PATH:
        return
    record["pid"] = os.getpid()
    encoded = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(
        _TRACE_PATH,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _cpu_ints(tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _cpu_floats(tensor) -> list[float]:
    return [float(value) for value in tensor.detach().float().cpu().reshape(-1).tolist()]


def _seq_len(batch: Any, index: int) -> int | None:
    values = getattr(batch, "seq_lens_cpu", None)
    if values is None:
        return None
    return int(values[index].item())


def _authority_fields(
    *,
    rid: str,
    output_before: list[int],
    decision_logits,
    proposed_tokens: list[int],
) -> dict[str, Any]:
    authority = _AUTHORITY.get(rid, [])
    offset = len(output_before)
    window = authority[offset : offset + int(decision_logits.shape[0])]
    prefix_exact = output_before == authority[:offset]
    authority_logits: list[float | None] = []
    for row, token in enumerate(window):
        authority_logits.append(float(decision_logits[row, int(token)].float().item()))
    authority_logits.extend([None] * (int(decision_logits.shape[0]) - len(window)))

    prefix_matches: list[bool] = []
    still_matches = prefix_exact
    for depth, token in enumerate(proposed_tokens):
        still_matches = bool(
            still_matches
            and depth < len(window)
            and int(token) == int(window[depth])
        )
        prefix_matches.append(still_matches)
    return {
        "authority_prefix_exact": prefix_exact,
        "authority_token_ids": window,
        "authority_token_logits": authority_logits,
        "proposal_prefix_matches_authority": prefix_matches,
    }


def _install_target_trace() -> None:
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner.sample, "_r056_trace_patch", False):
        return
    original = ModelRunner.sample

    def traced_sample(self, logits_output, forward_batch):
        if not isinstance(logits_output, tuple):
            rids = list(getattr(forward_batch, "rids", None) or [])
            logits = getattr(logits_output, "next_token_logits", None)
            active = [index for index, rid in enumerate(rids) if _enabled(rid)]
            if (
                logits is not None
                and active
                and int(logits.shape[0]) >= len(rids)
            ):
                values, indices = logits[: len(rids)].float().topk(2, dim=-1)
                values_cpu = values.detach().cpu()
                indices_cpu = indices.detach().cpu()
                for index, rid_value in enumerate(rids):
                    rid = str(rid_value)
                    if not _enabled(rid):
                        continue
                    step = _TARGET_STEPS[rid]
                    _TARGET_STEPS[rid] += 1
                    _append(
                        {
                            "event": "target_authority",
                            "mode": _TRACE_MODE,
                            "rid": rid,
                            "output_position": step,
                            "forward_mode": str(forward_batch.forward_mode),
                            "seq_len": _seq_len(forward_batch, index),
                            "top2_ids": [int(v) for v in indices_cpu[index].tolist()],
                            "top2_logits": [float(v) for v in values_cpu[index].tolist()],
                            "margin": float(values_cpu[index, 0] - values_cpu[index, 1]),
                        }
                    )
        return original(self, logits_output, forward_batch)

    traced_sample._r056_trace_patch = True
    ModelRunner.sample = traced_sample


def _install_domino_trace() -> None:
    import torch

    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_utils import compute_dflash_accept_len_and_bonus

    if getattr(DFlashVerifyInput.verify, "_r056_trace_patch", False):
        return
    original = DFlashVerifyInput.verify

    def traced_verify(self, *, batch, logits_output, page_size):
        if not any(_enabled(req.rid) for req in batch.reqs):
            return original(
                self=self,
                batch=batch,
                logits_output=logits_output,
                page_size=page_size,
            )
        bs = batch.batch_size()
        rows = int(self.draft_token_num)
        flat_logits = logits_output.next_token_logits.float().view(bs, rows, -1)
        candidates = self.draft_token.view(bs, rows)
        target_predict = flat_logits.argmax(dim=-1)
        accept, bonus = compute_dflash_accept_len_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        before = [list(req.output_ids) for req in batch.reqs]
        seq_lens_before = [_seq_len(batch, index) for index in range(bs)]
        result = original(
            self=self,
            batch=batch,
            logits_output=logits_output,
            page_size=page_size,
        )
        new_verified, commit_lens, _, accepted_cpu = result
        for index, req in enumerate(batch.reqs):
            rid = str(req.rid)
            if not _enabled(rid):
                continue
            decision_logits = flat_logits[index]
            top_values, top_ids = decision_logits.topk(2, dim=-1)
            proposed_tokens = _cpu_ints(candidates[index, 1:])
            cycle_key = ("domino", rid)
            cycle = _VERIFY_CYCLES[cycle_key]
            _VERIFY_CYCLES[cycle_key] += 1
            record = {
                "event": "spec_verify",
                "algorithm": "domino",
                "rid": rid,
                "cycle": cycle,
                "rows": rows,
                "width": 1,
                "horizon": rows - 1,
                "output_len_before": len(before[index]),
                "seq_len": seq_lens_before[index],
                "anchor_id": int(candidates[index, 0].item()),
                "proposed_tokens": proposed_tokens,
                "decision_row_indices": list(range(rows)),
                "row_top2_ids": top_ids.detach().cpu().tolist(),
                "row_top2_logits": top_values.detach().cpu().tolist(),
                "row_margins": _cpu_floats(top_values[:, 0] - top_values[:, 1]),
                "accepted_pre": int(accept[index].item()),
                "accepted": int(accepted_cpu[index]),
                "next_token_pre": int(bonus[index].item()),
                "commit_len": int(commit_lens[index].item()),
                "new_verified_id": int(new_verified[index].item()),
                "emitted_ids": [int(v) for v in req.output_ids[len(before[index]) :]],
            }
            record.update(
                _authority_fields(
                    rid=rid,
                    output_before=before[index],
                    decision_logits=decision_logits,
                    proposed_tokens=proposed_tokens,
                )
            )
            _append(record)
        return result

    traced_verify._r056_trace_patch = True
    DFlashVerifyInput.verify = traced_verify


def _install_forest_trace() -> None:
    import torch

    from sglang.srt.speculative.domino_forest_info import DominoForestVerifyInput
    from sglang.srt.speculative.domino_forest_utils import traverse_domino_forest

    if getattr(DominoForestVerifyInput.verify, "_r056_trace_patch", False):
        return
    original = DominoForestVerifyInput.verify

    def traced_verify(self, *, batch, logits_output, page_size):
        if not any(_enabled(req.rid) for req in batch.reqs):
            return original(
                self=self,
                batch=batch,
                logits_output=logits_output,
                page_size=page_size,
            )
        bs = batch.batch_size()
        rows = int(self.draft_token_num)
        width = int(self.width)
        horizon = int(self.horizon)
        flat_logits = logits_output.next_token_logits.float().view(bs, rows, -1)
        posterior = flat_logits.argmax(dim=-1)
        traversal = traverse_domino_forest(paths=self.paths, posterior=posterior)
        before = [list(req.output_ids) for req in batch.reqs]
        seq_lens_before = [_seq_len(batch, index) for index in range(bs)]

        diagnostic: list[dict[str, Any] | None] = []
        for index, req in enumerate(batch.reqs):
            rid = str(req.rid)
            if not _enabled(rid):
                diagnostic.append(None)
                continue
            selected = int(traversal.selected_path[index].item())
            path_start = 1 + selected * horizon
            decision_rows = [0, *range(path_start, path_start + horizon)]
            decision_index = torch.tensor(
                decision_rows, dtype=torch.long, device=flat_logits.device
            )
            decision_logits = flat_logits[index].index_select(0, decision_index)
            top_values, top_ids = decision_logits.topk(2, dim=-1)
            proposed_tokens = _cpu_ints(self.paths[index, selected])

            root = posterior[index, 0].expand(width, 1)
            path_predictions = posterior[index, 1:].view(width, horizon)
            draft_predictions = torch.cat(
                [root, path_predictions[:, : horizon - 1]], dim=1
            )
            per_path_accepted = (
                draft_predictions.eq(self.paths[index])
                .to(torch.int32)
                .cumprod(dim=1)
                .sum(dim=1)
            )
            diagnostic.append(
                {
                    "rid": rid,
                    "selected_path": selected,
                    "decision_rows": decision_rows,
                    "decision_logits": decision_logits,
                    "top_values": top_values,
                    "top_ids": top_ids,
                    "proposed_tokens": proposed_tokens,
                    "per_path_accepted": _cpu_ints(per_path_accepted),
                    "accepted_pre": int(traversal.accepted[index].item()),
                    "next_token_pre": int(traversal.next_token[index].item()),
                    "paths": self.paths[index].detach().cpu().tolist(),
                }
            )

        result = original(
            self=self,
            batch=batch,
            logits_output=logits_output,
            page_size=page_size,
        )
        new_verified, commit_lens, _, accepted_cpu = result
        for index, req in enumerate(batch.reqs):
            item = diagnostic[index]
            if item is None:
                continue
            rid = item["rid"]
            cycle_key = (f"forest_w{width}", rid)
            cycle = _VERIFY_CYCLES[cycle_key]
            _VERIFY_CYCLES[cycle_key] += 1
            record = {
                "event": "spec_verify",
                "algorithm": f"forest_w{width}",
                "rid": rid,
                "cycle": cycle,
                "rows": rows,
                "width": width,
                "horizon": horizon,
                "output_len_before": len(before[index]),
                "seq_len": seq_lens_before[index],
                "anchor_id": int(self.draft_token[index * rows].item()),
                "paths": item["paths"],
                "proposed_tokens": item["proposed_tokens"],
                "decision_row_indices": item["decision_rows"],
                "row_top2_ids": item["top_ids"].detach().cpu().tolist(),
                "row_top2_logits": item["top_values"].detach().cpu().tolist(),
                "row_margins": _cpu_floats(
                    item["top_values"][:, 0] - item["top_values"][:, 1]
                ),
                "per_path_accepted": item["per_path_accepted"],
                "selected_path": int(item["selected_path"]),
                "accepted_pre": int(item["accepted_pre"]),
                "accepted": int(accepted_cpu[index]),
                "next_token_pre": int(item["next_token_pre"]),
                "commit_len": int(commit_lens[index].item()),
                "new_verified_id": int(new_verified[index].item()),
                "emitted_ids": [int(v) for v in req.output_ids[len(before[index]) :]],
            }
            record.update(
                _authority_fields(
                    rid=rid,
                    output_before=before[index],
                    decision_logits=item["decision_logits"],
                    proposed_tokens=item["proposed_tokens"],
                )
            )
            _append(record)
        return result

    traced_verify._r056_trace_patch = True
    DominoForestVerifyInput.verify = traced_verify


if _TRACE_PATH:
    if _TRACE_MODE == "target":
        _install_target_trace()
    elif _TRACE_MODE == "domino":
        _install_domino_trace()
    elif _TRACE_MODE.startswith("forest"):
        _install_forest_trace()
    else:
        raise RuntimeError(f"unknown R056_TRACE_MODE={_TRACE_MODE!r}")
