"""Functional exact-Domino forward used by constrained FBPF adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .domino_joint_runtime import domino_prediction_hidden, domino_teacher_logits
from .fbpf import FlatParameterLayout, named_lora_parameters


@dataclass(frozen=True)
class DominoFunctionalInputs:
    """Four exact-context Domino blocks sharing one materialized target pass."""

    target_hidden: Tensor
    noise_embedding: Tensor
    context_lengths: Tensor
    anchors: Tensor
    gold: Tensor

    def __post_init__(self) -> None:
        if self.gold.ndim != 2 or self.gold.shape[0] != 4:
            raise ValueError("FBPF requires exactly four [4, L] gold rows")
        if self.target_hidden.shape[0] != 4:
            raise ValueError("target features must contain four rows")
        if self.noise_embedding.shape[0] != 4:
            raise ValueError("noise embeddings must contain four rows")
        if self.context_lengths.shape != (4,):
            raise ValueError("context lengths must have shape [4]")
        if self.anchors.shape != (4,):
            raise ValueError("anchors must have shape [4]")


class FunctionalDominoTeacherForward:
    """Evaluate teacher-forced Domino logits at a flat LoRA parameter vector.

    Each row keeps the released batch-one context geometry.  Only the injected
    LoRA tensors are replaced by ``theta``; the released backbone, causal head,
    and target vocabulary head remain frozen.
    """

    def __init__(
        self,
        *,
        domino: nn.Module,
        target_weight: Tensor,
        inputs: DominoFunctionalInputs,
        layout: FlatParameterLayout | None = None,
    ) -> None:
        self.domino = domino
        self.target_weight = target_weight.detach()
        self.inputs = inputs
        self.layout = (
            FlatParameterLayout.from_named_parameters(named_lora_parameters(domino))
            if layout is None
            else layout
        )
        if self.layout.total_numel == 0:
            raise ValueError("Domino has no injected LoRA parameters")

    def __call__(self, theta: Tensor) -> Tensor:
        if theta.ndim != 1 or theta.numel() != self.layout.total_numel:
            raise ValueError("flat LoRA vector has the wrong geometry")
        parameters = self.layout.unflatten(theta)
        horizon = int(self.inputs.gold.shape[1])
        rows: list[Tensor] = []
        for row, length_tensor in enumerate(self.inputs.context_lengths):
            context_length = int(length_tensor)
            position_ids = torch.arange(
                context_length + int(self.domino.block_size),
                dtype=torch.long,
                device=self.inputs.gold.device,
            ).unsqueeze(0)
            hidden = functional_call(
                self.domino,
                parameters,
                args=(),
                kwargs={
                    "target_hidden": self.inputs.target_hidden[
                        row : row + 1, :context_length
                    ],
                    "noise_embedding": self.inputs.noise_embedding[row : row + 1],
                    "position_ids": position_ids,
                    "attention_mask": None,
                    "past_key_values": None,
                    "use_cache": False,
                    "is_causal": False,
                },
                tie_weights=False,
                strict=False,
            )
            domino_prediction_hidden(self.domino, hidden, horizon=horizon)
            rows.append(
                domino_teacher_logits(
                    domino=self.domino,
                    target_weight=self.target_weight,
                    anchors=self.inputs.anchors[row : row + 1],
                    gold=self.inputs.gold[row : row + 1],
                    hidden=hidden,
                    train_causal_head=False,
                )
            )
        return torch.cat(rows, dim=0)


__all__ = ["DominoFunctionalInputs", "FunctionalDominoTeacherForward"]
