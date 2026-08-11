# Round 2 Proposal: Greedy-Boundary Reachable Domino Adaptation

The fixed evaluator is prompt-balanced accepted draft tokens on exact
Qwen3-4B greedy B16 single-chain anchors.  Released Domino is
`7.0157920311/15`; the selection goal remains `>=7.5`, followed by one frozen
validation-gate confirmation.

Static calibration is closed by local evidence: global correction scaling adds
only `0.0186` with an interval crossing zero, and positive released DeLS fusion
hurts.  Released Domino's parallel backbone/correction decomposition is
`5.9385 -> 7.0158`; the DFlash K16 oracle of `9.7267` only demonstrates that
better single-chain paths exist among local candidates.

## Head-only stage

Use cached released parallel hidden states.  Freeze target, backbone, and the
large vocabulary projection; train the released-initialized causal GRU and
rank/input projection with L2-SP.  A block whose frozen base token is wrong has
zero head loss.  For a reachable block, optimize the first current mismatch
and lightly preserve its preceding correct prefix.

Screen two losses on identical data and initialization:

1. breaker cross entropy;
2. breaker gold-versus-best-competitor softplus margin, directly aligned with
   the greedy argmax boundary.

Released-style decay CE and D-PACE are controls.  On-policy prompt-balanced EAL
selects checkpoints; teacher loss does not.  A vectorized/position-loop state
alignment check and same-code-path released baseline run before optimization.
Head-only is stopped if it cannot add at least `0.10`.

## Joint stage when head-only is insufficient

Unfreeze the final parallel-backbone layer(s) and use

`L_joint = L_base,0 + sum_{j>=1} I[pred_<j=gold_<j] L_final,j
           + lambda_anchor L_released-base`.

`L_base,0` is always active, including on previously unreachable blocks, so
the backbone can repair first-token errors.  Later losses retain detached
clean-prefix reachability.  The anchor term distills released base logits and/
or penalizes feature drift so repairing position zero does not destroy the
strong released branch.  Full-head tuning, actual-prefix target replay, and a
small Jacobi refiner are subsequent measured escalations.

Selection requires `>=7.5` and a meaningful same-run gain before opening the
gate split.  Final evidence reports paired prompt-balanced EAL, conditional
acceptance by position, first breaker, full-horizon rate, domains, and
beneficial/harmful blocks.  No hash audit is part of this route.
