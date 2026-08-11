# Initial Proposal: Binary Direct Safety Gate (BDSG)

## Problem anchor

The goal remains to increase real accepted draft length over released DFlash
without target calls or autoregressive candidate selection. Three action-level
routes have now isolated a more specific failure:

- flat 226-way action CE selected many cost-insensitive edits and harmed
  `31.7%--35.0%` of held-out blocks;
- action-uniform signed MSE hid rare positives even on a 512-block capacity
  set;
- CAMRS fit that same subset exactly, but on full-data development a harmful
  nonpositive rate of `98.5%--99.3%` still left `637--1326` positive harmful
  scores among 90,120 harmful actions. Max selection converted that thin tail
  into `67--108` harmed blocks while trained EAL fell by `0.075--0.128`.

The immediate issue is therefore not candidate availability or the absence of
any Direct signal. The frozen matched Direct-native policy reaches raw
prompt-balanced EAL `5.334669582118561`, versus DFlash
`5.112001943634597`. It improves 141 blocks, is neutral on 972, and harms 62.
An exact oracle restricted to choosing DFlash or that fixed Direct path reaches
`5.430758017492711` with zero harm, above the unchanged strict target
`5.396991943634597`. Adding Direct-one-edit as a third option raises the oracle
only to `5.438411078717201`.

BDSG removes the 225-edit maximum. It learns exactly one binary decision:

```text
KEEP:  emit the released DFlash rank-zero path;
APPLY: emit the complete path selected by the frozen matched Direct model.
```

This is a new mechanism and a fresh model. It is not threshold calibration,
continuation, reweighting, or reinterpretation of CAMRS.

## Frozen producer and gold-free forward path

The APPLY policy is the hash-frozen Direct checkpoint from job `10133585`:

- global axial-additive D64/H4/L1, K16, seed 0;
- checkpoint SHA256
  `9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e`;
- metrics SHA256
  `9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef`.

It is loaded strictly, frozen, placed in evaluation mode, and must reproduce
the existing Direct-native control exactly before any gate training. For block
`x`, its per-position argmax path is `p_D(x)`; DFlash is `p_B(x)=0`.

The trainable gate uses a fresh zero-output-initialized axial-additive
D64/H4/L1 lattice backbone. Let its residual node scores be `rho_G(i,r;x)`.
Define one scalar APPLY score

```text
z(x) = (1/L) sum_i [rho_G(i,p_D[i];x) - rho_G(i,0;x)],   L=15.
s(KEEP,x)=0,  s(APPLY,x)=z(x).
```

The gate backbone sees the same deployable frozen DFlash hidden states,
candidate embeddings/logits, anchor embedding, and the frozen Direct path. It
never sees gold tokens. Gold is used only to form supervised accepted-prefix
gain. If Direct equals DFlash, every summand is identically zero. At fresh
initialization the residual projection is exactly zero, so every block chooses
KEEP and reproduces DFlash bit-for-bit.

Deployment uses the fixed rule

```text
APPLY iff z(x) > 0; otherwise KEEP.
```

There is no learned KEEP bias, temperature, margin offset, domain rule,
calibration threshold, or validation-time threshold search.

## Exact signed policy gain and binary regret hinge

Let `A_B(x)` and `A_D(x)` be the realized accepted-prefix lengths of DFlash and
frozen Direct-native on the stored gold continuation. Define

```text
g(x) = [A_D(x) - A_B(x)] / L in [-1,1].
```

KEEP has utility zero and APPLY has utility `g`. Ties prefer KEEP. The exact
two-action cost-augmented hinge simplifies to

```text
if g > 0:  loss(x) = ReLU(g - z)
else:      loss(x) = ReLU(z - g).
```

At `z=g` the loss is zero. At `z=0`, beneficial blocks push APPLY upward,
harmful blocks push it downward, and neutral blocks have zero loss and the
declared zero ReLU subgradient. Every nonneutral block contributes one policy
constraint; there is no division by 225 actions and no extreme-value selection
over action-score noise.

For the deployed decision `a_hat`, the loss pointwise upper-bounds normalized
binary policy regret:

```text
loss(x) >= max(0,g) - utility(a_hat,x).
```

The output is a decision score, not a calibrated probability or a claim that
the exact gain is identifiable.

## Mechanism predictions and falsifiers

BDSG makes four falsifiable predictions:

1. same-subset capacity reaches the correct sign for beneficial and harmful
   Direct outcomes without a rare-positive gradient deficit;
2. on held-out prompts, a single negative tail is controllable at much lower
   precision than 90,120 action-level negatives;
3. a trained checkpoint suppresses a material share of Direct-native's 113
   lost block-weighted tokens while preserving most of its 374 gained tokens;
4. raw EAL improves without post-hoc thresholding and without first-token harm.

Failure modes remain explicit:

- the frozen inputs may not distinguish whether Direct helps or harms;
- the trainable gate may fit producer-specific in-sample artifacts;
- the binary oracle has only `0.03377` EAL slack above the absolute gate, so
  even moderate classification regret can make the route infeasible;
- running a second small lattice backbone adds latency and must later be
  profiled if offline gates pass.

## Required diagnostics

Every evaluation must record enough per-block evidence to reconstruct:

1. DFlash, Direct-native, BDSG, and exact binary-oracle accepted length, path,
   first-token outcome, and prompt-balanced summaries;
2. exact counts and token mass for beneficial, neutral, and harmful Direct
   outcomes;
3. APPLY coverage, improved/neutral/harmed selected blocks, selective
   precision, benefit recall, harm recall/avoidance, false-APPLY rate on
   nonbeneficial blocks, and binary-oracle gap recovery;
4. score sign by true gain sign, score quantiles, maximum harmful score,
   minimum beneficial score, and tie counts at zero;
5. mean/max hinge, decoded binary regret, minimum bound slack, and all
   violations beyond `1e-6`;
6. epoch-zero beneficial-upward and harmful-downward output-projection gradient
   norms, cosine/cancellation, total norm, clipping, and finiteness;
7. frozen Direct producer identity before/after, gate source hashes before/after,
   exact prompt/cardinality hashes, and checkpoint selection for every epoch.

## Gate 0: CPU semantics

Tests must establish before GPU use:

- Direct/base accepted-prefix gains match brute-force token reconstruction;
- Direct paths are produced without gold and the frozen checkpoint loads
  strictly with no trainable producer parameter;
- `z` is exactly zero when the output projection is zero and whenever
  `p_D=p_B`;
- strict-positive APPLY and KEEP-preferred ties are exact;
- the piecewise hinge matches loss-augmented two-action enumeration, is finite,
  nonnegative, zero at `z=g`, and pointwise upper-bounds decoded regret;
- at zero, beneficial/harmful gradients have the declared opposite signs and
  neutral examples have zero gradient;
- frozen producer, canonical inputs, target embedding, and target IDs never
  receive gradients;
- only the zero-initialized gate output projection receives the first backward
  gradient, while upstream gate parameters receive gradients after one update;
- saved examples independently reconstruct every gate and oracle summary.

## Gate 1: fresh capacity falsifier

First materialize a deterministic outcome manifest from OPB `train` using only
the frozen Direct checkpoint and exact token-ID realization. The manifest and
producer/data/source hashes must be frozen before training. Select 512 unique
blocks by hash, with exact composition:

```text
256 Direct-beneficial
128 Direct-harmful
128 Direct-neutral with p_D != p_B
```

This is an adaptive same-subset engineering probe only. Train a fresh gate
backbone with global axial-additive D64/H4/L1, K16, batch 32, seed 0, dropout
and weight decay 0, LR `6e-4`, warmup `0.04`, clip 1.0, and exactly 5,120
updates. The Direct producer is frozen. Select the earliest checkpoint with
minimum uniform-block mean binary hinge; behavior cannot choose it.

All conditions are conjunctive:

- zero bound violations and finite values/gradients;
- mean hinge at most 5% of the manifest's exact mean binary-oracle advantage;
- beneficial score `>0` on at least 254/256 blocks;
- harmful score `<=0` on at least 127/128 blocks;
- changed-neutral score `<=0` on at least 127/128 blocks;
- utility-optimal binary decision on at least 507/512 blocks;
- binary-oracle gap recovery at least `0.95`;
- selected harmful blocks at most 1/512;
- false APPLY on nonbeneficial blocks at most 2/256;
- exact manifest composition, producer identity, epoch-zero DFlash identity,
  and complete source/data/checkpoint provenance.

A failure closes this exact binary score/architecture/objective/schedule. No
longer training, wider model, threshold, weights, extra seed, or alternate
manifest rescue is allowed. A pass supports only same-subset capacity and
requires fresh result review and development-code review.

## Conditional full-data development

Only after Gate 1 passes may one seed-0 development run be considered. The
frozen Direct producer remains unchanged. The exact OPB 99,356-prompt /
793,989-block collection, three epochs, batch 64, and 37,221-update budget are
retained for a direct comparison. Validation must remain the physically
isolated 147-prompt / 1,175-block `validation_select` collection. Select by raw
prompt-balanced BDSG EAL, then lower harm, then lower binary hinge; strict
improvement preserves the earliest exact tie. Epoch 0 is eligible.

Advance remains conjunctive and unchanged:

- `EAL_BDSG - EAL_DFlash > 0.28499`;
- `EAL_BDSG - EAL_Direct-native >= 0.05`;
- `EAL_BDSG - EAL_Direct-one-edit >= 0.05`;
- harmed fraction `<=0.05`;
- first-token-correct count trails Direct-native by at most one block;
- exact binary-oracle EAL is frozen as `5.430758017492711`, and reported gate
  recovery must be finite and in `[0,1+1e-6]`.

Development is not paper evidence. No additional seeds, formal test, rollout,
calibration, threshold sweep, policy mixture, or latency claim is authorized
without a subsequent result-to-claim review.

## Novelty boundary

Selective prediction, abstention, mixture-of-experts routing, signed policy
value prediction, and cost-augmented binary margins are prior art. SpecDec++
predicts acceptance to adapt speculative length; Hybrid Verified Decoding
predicts payoff to route among draft sources; SelectiveNet formalizes learned
coverage/risk. BDSG therefore claims no novelty for binary gating itself.

The narrow candidate contribution is the DFlash-specific composition:
counterfactual accepted-prefix supervision for a fixed full-lattice Direct
policy, an exact KEEP identity, a zero-threshold binary regret bound, and an
explicit response to the observed max-over-225 harmful-score tail. Novelty and
practical value remain provisional until positive held-out and latency evidence.

## Authorization boundary

- This document authorizes independent research review only.
- A fresh review score of at least 9.0/10 and `READY` may authorize CPU
  implementation and tests, not GPU execution.
- Outcome-manifest generation and every GPU job require a separate
  experiment-bridge review and explicit bounded GO.
- CAMRS checkpoints/optimizer state are never reused.
- Formal test, additional seeds, thresholding, widening, continuation, and
  post-hoc policy mixtures remain closed.
