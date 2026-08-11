# Initial Proposal: Frontier-Aligned Domino Adaptation

## Problem Anchor

- **Bottom-line problem:** maximize accepted draft-prefix length for Qwen3-4B,
  greedy decoding, block size 16, exact single-chain verification.
- **Hard comparator:** released Domino, prompt-balanced same-anchor EAL
  `7.0157920311` over 15 draft positions.
- **First target:** at least `7.5` on model-selection data and strictly exceed
  a freshly rerun Domino comparator on clean held-out data.
- **Non-goals:** no more frozen candidate selectors; no oracle/gold inference;
  no hash/provenance bureaucracy; no change to exact target verification.
- **Freedom:** extra causal-head capacity, joint backbone tuning, target-logit
  replay, and a few iterative draft refinements are allowed because EAL is the
  primary target.  Latency is reported, not used to veto a real EAL gain.

## Diagnosis

The local decomposition is unusually informative.  Domino's backbone improves
released DFlash from `5.1120` to `5.9385`, and its causal correction adds another
`1.0773` to reach `7.0158`.  The K16 oracle remains at `9.7267`.  Frozen
selectors repeatedly fail to turn this oracle availability into realized EAL,
including a 27.5M-parameter teacher trained on the 100K corpus.

Global correction scaling and released DeLS fusion were tested before proposing
a larger intervention.  A scale of 0.9 yielded only `+0.0186 EAL`, with a 95%
interval crossing zero.  Every tested positive DeLS weight hurt relative to the
best Domino-only policy.  This closes fixed post-hoc fusion and identifies the
remaining problem as learned, state-dependent correction under the local data
distribution.

## Method Thesis

Initialize from released Domino, retain its strong parallel backbone, and
adapt the causal correction at the first acceptance frontier using exact local
anchors.  Clean-prefix teacher states are the primary supervision because only
they can extend the current accepted prefix.  If this saturates, add target
distillation on actual draft-induced prefixes, then jointly adapt the final
backbone layers.  A state/position gate or parallel iterative refiner is added
only when a measured failure demands it.

## Stage 1: Cached head adaptation

### Data

Replay all 18,253 existing phase-3 canonical blocks through the released Domino
parallel backbone and cache only:

- exact anchor and 15 target-greedy labels;
- Domino parallel hidden states;
- sample/domain/split identifiers.

The existing prompt-level train/validation-select/validation-gate split is
retained.  No shard hashes are recomputed.  Stored contexts are checked only
for semantic prefix, anchor, and label consistency.

### Trainable parameters

Freeze the target model and Domino parallel backbone.  Train the released
single-layer GRU and rank-256 correction projection, initialized from the public
checkpoint.  The first draft token remains the parallel backbone top-1, exactly
as in Domino.

For position `i>=2`, teacher-forced state `s_i` is computed from the anchor and
the correct preceding draft tokens.  Corrected logits are

`l_i = l_i^base + Delta(z_i, s_i)`.

### Compact objective screen

Run the following matched variants with identical data, initialization,
optimizer, steps, and on-policy evaluator:

1. **DECAY-CE:** released Domino-style exponential position CE.
2. **DYNAMIC-FRONTIER:** D-PACE-style detached weights computed from current
   gold probabilities, with asymmetric smoothing so one weak early token does
   not erase all suffix learning.
3. **AUF:** retain supervision only through the current greedy first mismatch;
   this directly concentrates updates at the acceptance breaker.

The primary metric during selection is on-policy prompt-balanced EAL, not
teacher-forced token accuracy.  Learning rate and early stopping are chosen on
validation-select only.  A variant must improve the same-run released
checkpoint by at least `+0.10` before receiving a second seed; otherwise it is
stopped.

## Stage 2: Target replay on proposal prefixes

This stage is conditional on Stage 1 failing to reach at least `+0.20` EAL.
For each training anchor, run the current drafter on-policy, append its proposed
tokens to the exact context, and obtain target conditionals in one causal target
forward.  Distill the target distribution on those actual states:

- accepted/clean-prefix positions: forward KL or hard CE to reinforce the
  target path;
- rejected-prefix positions: clipped reverse KL with depth decay, following
  Draft-OPD's stability result.

The target label for a wrong prefix is derived from the target evaluated on
that prefix, never from the original clean continuation.  Refresh replay once
after the first adaptation round if proposal drift is substantial.

## Stage 3: Capacity escalation

If head-only adaptation remains below `7.5`:

1. jointly tune the final one/two Domino backbone layers and head using the
   official SpecForge Domino implementation and base-anchor curriculum;
2. add a scalar correction gate conditioned on position, base margin/entropy,
   parallel hidden state, and GRU state, initialized to reproduce scale one;
3. if sequential correction remains capacity-limited, implement a 2/4/6-pass
   Jacobi block refiner over fixed parallel features, following the mechanism
   validated by xPress.

Only one escalation is active at a time.  The smallest stage that clears the
held-out Domino comparator becomes the final method.

## Evaluation and decision rules

- **Selection:** phase-3 validation-select, paired prompt-balanced EAL.
- **Confirmation:** validation-gate, evaluated once after freezing method,
  hyperparameters, and checkpoint.
- **Success:** point estimate strictly exceeds a same-job released Domino run;
  preferred final evidence is a positive prompt-cluster 95% interval.  Target
  `>=7.5` remains the performance goal.
- **Diagnostics:** per-domain EAL, first-token acceptance, full-horizon rate,
  harm/benefit counts, and draft latency.
- **Escalate rather than rationalize:** a tiny positive but uncertain result is
  not success; it triggers the next mechanism stage.

## Why this is the right next intervention

It starts from the only local mechanism already proven to recover more than one
EAL token, uses the exact target distribution and anchors of interest, and can
be tested in minutes from cached features.  It also retains clear expansion
paths supported by recent evidence: target replay, richer target-conditioned
backbones, and iterative block refinement.  Static selectors and fixed DeLS
fusion are excluded by direct local negative results rather than preference.
