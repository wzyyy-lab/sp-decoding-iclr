# Round 1 Refinement: Producer-Reusing Out-of-Sample Direct Safety Gate

## Decision

Retain the two actions but replace every weak part of the first proposal. The
refined route is **PROS-Gate**: a small producer-reusing, producer-out-of-sample
safety sidecar that chooses between released DFlash and the complete path of
one hash-frozen Direct producer.

```text
KEEP:  released DFlash rank-zero path
APPLY: complete frozen Direct-native path
```

No OPB outcome may supervise the gate. No second lattice backbone is trained.
No threshold is calibrated. The raw deployment rule remains `APPLY iff z > 0`
with KEEP on ties.

This document answers the blocking Round-1 review. It authorizes a new
independent method review only; it does not authorize implementation or GPU
execution.

## Frozen opportunity and feasibility bar

The producer remains job `10133585`, checkpoint
`9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e`
and metrics
`9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef`.
On the isolated 1,175-block development control:

| Policy | Prompt-balanced EAL |
|---|---:|
| DFlash | 5.112001943634597 |
| Direct-native | 5.334669582118561 |
| Exact KEEP/Direct oracle | 5.430758017492711 |

Direct improves 141 blocks for 374 unweighted tokens, is neutral on 972, and
harms 62 for 113 tokens. The unchanged absolute target is strictly above
5.396991943634597. A successful gate must therefore recover more than 89.4069%
of the binary-oracle gap and approximately 64.9% of Direct's avoidable loss.
This narrow `0.0337661` oracle slack is treated as a feasibility constraint,
not hidden by a softer development threshold.

## Response to Review Blocker 1: utility-consistent objective

For block `x`, let the realized accepted-prefix lengths under released DFlash
and frozen Direct be `A_B(x)` and `A_D(x)`. With `L=15`, define

```text
g(x) = [A_D(x) - A_B(x)] / L
y(x) = sign(g(x)).
```

The trainable sidecar emits one dimensionless score `z(x)`. Its loss is

```text
if g != 0:  ell(x) = abs(g) * ReLU(1 - y*z)
if g == 0:  ell(x) = 0.
```

This corrects the first proposal's sign-count inconsistency. For colliding
deployable features, the conditional hinge decision follows the sign of
`E[g | features]`, so token magnitude rather than positive/negative example
count determines the action. A one-token mistake and a fifteen-token mistake
cannot contribute the same active gradient.

For prompt `p` with `n_p` stored blocks, the exact empirical estimand is

```text
L_train = (1 / |P|) * sum_p [(1 / n_p) * sum_{x in p} ell(x)].
```

Each record stores the fixed SGD weight

```text
w_x = N_blocks / (N_prompts * n_p),
```

so a mean over all weighted records is exactly `L_train` and mini-batch
gradients are unbiased for it.

The decoded normalized regret is

```text
regret = g       if g > 0 and z <= 0
regret = -g      if g < 0 and z > 0
regret = 0       otherwise.
```

Whenever the decoded decision is wrong, `y*z <= 0`, hence
`ell >= abs(g) = regret`; otherwise regret is zero. Every evaluation must
reconstruct this bound independently and fail on slack below `-1e-6`.

At `z=0`, nonneutral output gradients are scaled signed gains. Neutral blocks
have zero loss because either action has exactly zero accepted-token regret.
Neutral APPLY behavior is reported separately and is not mislabeled as harm.

## Response to Review Blocker 2: producer-out-of-sample split

The OPB-100K manifest explicitly excluded the complete Phase-3 development
manifest. The frozen Direct producer was trained only on OPB. The existing
Phase-3 canonical `train` split is therefore producer-out-of-sample and has
exactly 1,987 collected prompts / 15,886 blocks:

| Domain | Prompts | Blocks |
|---|---:|---:|
| chat | 655 | 5,230 |
| code | 665 | 5,320 |
| math | 667 | 5,336 |

Before Direct outcomes are computed, create a split manifest from canonical
prompt identity only. Within each domain, order collected `sample_id` values by

```text
SHA256("pros-gate-phase3-oos-v1\0" || domain || "\0" || sample_id ||
       "\0" || canonical_metadata_sha256).
```

Assign exact prefix counts:

| Domain | Fit | Checkpoint | Falsifier |
|---|---:|---:|---:|
| chat | 525 | 65 | 65 |
| code | 531 | 67 | 67 |
| math | 531 | 68 | 68 |
| **Total** | **1,587** | **200** | **200** |

The manifest is frozen and hashed before outcome computation. It must prove:

- exact cardinalities and prompt-domain membership;
- no prompt in more than one gate split;
- zero sample-ID overlap with all eight OPB producer collections,
  `validation_gate`, `validation_select`, and reserved test;
- agreement with the Phase-3 manifest and canonical collection hashes;
- OPB's recorded exclusion/decontamination contract for the entire Phase-3
  development manifest.

Only `fit` may update sidecar parameters. Only `checkpoint` may choose a
checkpoint. The chosen model is then frozen, `falsifier` is evaluated exactly
once, and the model is never refit. The historically inspected
`validation_gate` split is excluded. `validation_select` stays unopened by
this route until a falsifier pass. Formal test remains sealed.

OPB producer outcomes are allowed only as explicitly labeled in-sample
descriptive diagnostics. They can neither train nor select PROS-Gate.

## Response to Review Blocker 3: one-shot contextual falsifier

The 200-prompt falsifier is the binding feasibility gate before
`validation_select`. The frozen raw `z > 0` decision must satisfy every
condition:

1. all values and gradients are finite and there are zero regret-bound
   violations beyond `1e-6`;
2. prompt-balanced binary-oracle recovery is at least `0.90`, computed without
   clipping as
   `(EAL_PROS - EAL_DFlash) / (EAL_binary_oracle - EAL_DFlash)`;
3. raw prompt-balanced `EAL_PROS` is strictly above both `EAL_DFlash` and
   `EAL_Direct`;
4. harmed fraction, defined as APPLY with `g < 0` divided by all blocks, is at
   most `0.05`;
5. first-token-correct count trails Direct by at most one block;
6. recovery exceeds the best frozen scalar-only comparator by at least `0.05`;
7. exact block/prompt counts, split identity, producer identity, and native
   Direct reproduction all match the frozen contracts.

Failure closes this exact feature family, loss, fit/checkpoint schedule, seed,
and decision rule. It cannot be rescued with a threshold, another split,
refitting, longer training, a wider sidecar, another seed, OPB labels, or
opening `validation_select`.

Every falsifier report includes beneficial/neutral/harmful counts and token
mass, APPLY coverage, benefit recall, harm avoidance, harmful APPLY count,
neutral APPLY count, tie-policy agreement, zero-regret utility optimality,
false APPLY on all nonbeneficial blocks, oracle regret, score/gain quantiles,
per-domain summaries, and a deterministic 10,000-replicate prompt-cluster
bootstrap interval. Point estimates bind; intervals are diagnostic.

### Frozen scalar-only comparator

To make condition 6 executable, fit one deterministic prompt-weighted ridge
regression on `fit` only to predict signed gain `g`. Use float64 closed-form
weighted least squares, an unpenalized intercept, standardized nonconstant
features using fit statistics only, and fixed ridge coefficient `1e-3`. The
21 block scalars are:

- normalized Direct change count;
- sum/mean/min/max over changed positions for Direct total-score margin,
  Direct residual margin, and DFlash log-probability difference (12 values);
- mean/min/max DFlash top-K entropy (3 values);
- mean/min/max retained top-K log mass (3 values);
- first-position change flag and first-position Direct total-score margin.

Missing changed-position summaries are zero and accompanied by zero change
count. Deployment is prediction `>0`; identical paths force KEEP. Coefficients,
fit normalizers, predictions, and hashes are frozen before either checkpoint or
falsifier evaluation. Always-KEEP and always-Direct are also reported. The
best comparator is the maximum falsifier recovery among the three, without
using falsifier labels to alter any comparator.

If a scalar comparator itself reaches 0.90 recovery, the contextual sidecar
must still clear the declared `+0.05` contribution gate. Otherwise the
sidecar-specific claim closes and any simpler scalar deployment requires a new
protocol.

## Response to Review Blocker 4: producer-reusing 38,674-parameter sidecar

Run the frozen Direct selector once in evaluation mode and capture its detached
final normalized node states `h_D` immediately before the existing residual
projection. Capturing these states may not change the Direct forward result or
state dictionary. Native Direct scores, paths, accepted lengths, and summaries
must match the frozen control bit-for-bit before and after sidecar attachment.

For each of 15 positions, gather the Direct-selected node and rank-zero node.
The exact 200-dimensional position vector concatenates:

```text
h_D(i, p_D[i])                         64
h_D(i, 0)                              64
h_D(i, p_D[i]) - h_D(i, 0)             64
eight fixed scalars                     8
```

The scalars are Direct total-score margin, Direct residual margin, DFlash
log-probability difference, selected rank divided by 15, position divided by
14, change indicator, normalized top-K entropy, and smoothly bounded retained
top-K log mass. All producer tensors are detached.

The shared position encoder is exactly

```text
LayerNorm(200) -> Linear(200,64) -> SiLU -> Linear(64,64).
```

Concatenate the mean over all positions, mean over changed positions, max over
changed positions, encoded first position, and normalized change count. Empty
changed-position pools are zero. This 257-vector passes through

```text
LayerNorm(257) -> Linear(257,64) -> SiLU -> Linear(64,64)
-> bias-free Linear(64,1).
```

The last 64-to-1 weight is exactly zero initialized; all other parameters use
name-seeded initialization with seed 0. The exact trainable parameter count is
38,674, or 8.92% of the 433,772-parameter producer. No producer parameter is
trainable. The emitted scalar is multiplied by the indicator that at least one
position changed, forcing exact `z=0` for identical paths.

This design reuses already computed globally contextual Direct states and does
not repeat target-embedding projection or lattice mixing. It needs no target
forward call, target token, gold ID, accepted length, or outcome at inference.
At epoch zero every block keeps DFlash exactly. Only the final projection gets
the first nonzero backward gradient; upstream sidecar layers receive gradients
after its first update.

Offline success authorizes only a later fused latency evaluation. Since the
strict target adds only about 0.98% verification advance over Direct and the
perfect oracle only about 1.52%, no throughput claim is permitted without an
end-to-end online benchmark.

## Response to Review Blocker 5: exact capacity construction and metrics

After the outcome computation is separately reviewed and hash-frozen, form a
same-subset capacity set only from `fit`. Its unique block key is

```text
(sample_id, anchor_offset, context_length).
```

Define strata exactly:

```text
beneficial:      g > 0
harmful:         g < 0
changed-neutral: g == 0 and p_D != p_B.
```

Within the scarcity order harmful, changed-neutral, beneficial, rank blocks by
SHA256 of `pros-capacity-v1`, block key, producer checkpoint/metrics hashes,
canonical metadata hash, and split-manifest hash. Select 128 harmful, 128
changed-neutral, and 256 beneficial blocks while greedily excluding every
already selected prompt. Fail closed if exact prompt-unique composition is not
available. The manifest is then immutable.

Capacity is only a generous same-subset plumbing/memorization bound. Use a
fresh sidecar, batch 32, at most 5,120 updates, and evaluate each complete
16-step pass. Report first passage; select the earliest checkpoint with minimum
prompt-weighted loss. Conjunctive passage requires:

- finite values/gradients and zero regret-bound violations;
- mean weighted hinge at most 5% of its epoch-zero value;
- benefit recall at least 254/256;
- harm avoidance at least 127/128;
- at least 509/512 zero-regret utility-optimal decisions;
- prompt-balanced binary-oracle recovery at least 0.95;
- at most one harmful APPLY block.

Changed-neutral KEEP agreement, neutral APPLY, and false APPLY over the full
nonbeneficial denominator are reported but do not gate accepted-token utility.
This explicitly separates the tie policy from zero-regret optimality.

## Response to Review Blocker 6: frozen fit optimization and selection

Conditional on CPU semantics and capacity review/pass, train one fresh seed-0
sidecar on `fit` with:

```text
batch size                 64
maximum updates            5,120
learning rate              6e-4
AdamW betas / epsilon      PyTorch defaults (0.9, 0.999) / 1e-8
weight decay               0
dropout                    0
gradient clip              1.0
warmup                     round(0.04 * exact_total_updates)
post-warmup schedule       cosine to zero
```

Let `S = ceil(number_of_fit_blocks / 64)`. Run exactly
`floor(5120 / S)` complete deterministic passes, hence no more than 5,120
updates and no partial final pass. Freeze `S`, pass count, total steps, warmup,
and prompt-order hashes after the outcome manifest exists and before training.
Each pass uses a seed-0 deterministic permutation.

Evaluate the untouched checkpoint split at epoch zero and after every complete
pass. Select strictly lexicographically by

```text
(prompt-balanced EAL, -harmed_count, -prompt-weighted_gain_hinge).
```

Epoch zero is eligible; strict improvement is required, so exact ties retain
the earliest checkpoint. Fit behavior cannot select a checkpoint. Falsifier
and `validation_select` are absent from the training process and cannot alter
the selection.

## Gate 0: CPU semantic and isolation tests

Before any GPU action, tests must establish:

- exact Direct/base accepted-prefix gains from brute-force token IDs;
- strict loading and freezing of the producer, with no producer gradient;
- exact latent capture without changing native Direct output/state;
- exact 200-to-64 position and 257-to-64 block feature dimensions and the
  38,674 trainable parameter count;
- gold-free inference inputs and detached producer/target/canonical tensors;
- zero initialization, strict-positive APPLY, KEEP ties, and forced identical-
  path zero;
- the gain-weighted hinge against hand enumeration, prompt weighting identity,
  conditional collision examples, decoded regret bound, and neutral zero loss;
- signed gain-scaled initialization gradients, first-step projection-only
  gradient, and later upstream sidecar gradient;
- deterministic split/capacity identities, exact denominators, native-control
  reconstruction, and independent saved-example reconstruction;
- no code path can load `validation_gate`, `validation_select`, or formal test
  during fit/checkpoint/falsifier preparation.

## Conditional development evaluation

Only a reviewed falsifier pass may authorize opening the physically isolated
147-prompt / 1,175-block `validation_select` collection once with the already
frozen model. No training, refit, threshold selection, or checkpoint change is
allowed. The original conjunctive development gate remains:

- `EAL_PROS - EAL_DFlash > 0.28499`;
- `EAL_PROS - EAL_Direct-native >= 0.05`;
- `EAL_PROS - EAL_Direct-one-edit >= 0.05`;
- harmed fraction at most `0.05`;
- first-token-correct count trails Direct-native by at most one;
- all values finite, exact control/data alignment, and zero regret-bound
  violations.

Report unclipped recovery against exact binary-oracle EAL
5.430758017492711 and oracle regret. Development is not paper evidence. Formal
test, seeds 1/2, online rollout, calibration, widening, refitting, policy
mixtures, and positive latency claims remain unauthorized pending a fresh
result-to-claim review.

## Provenance contract

Every material action must record and verify before and after:

- full runtime Python import closure and launch wrapper;
- Phase-3, OPB, validation, and reserved manifests and canonical shard hashes;
- split, outcome, capacity, scalar-comparator, and prompt-order artifacts;
- producer checkpoint, metrics, config, source closure, target embedding, and
  exact native-output witness;
- sidecar sources/config/checkpoints, all checkpoint selection keys, optimizer
  and scheduler state, and environment versions;
- prompt sets, block keys, split cardinalities, denominators, and overlap
  proofs.

Scientific-gate failure must still write an atomic complete report before a
nonzero exit. Artifacts are immutable after review.

## Claim and novelty boundary

Selective prediction, acceptance/payoff heads, draft-source routing, weighted
binary margins, and abstention are established ideas. PROS-Gate claims none of
them in isolation. The candidate contribution is limited to a DFlash-specific,
producer-reusing sidecar trained from counterfactual accepted-prefix outcomes
of a fixed full-path Direct policy, with exact KEEP identity and a strict
producer-out-of-sample stacking protocol motivated by the observed CAMRS
max-tail failure.

Even a positive development result supports only one frozen producer, one gate
seed, offline anchors, and development prompts. Paper-facing efficacy requires
formal-test, multiple seeds, prompt-cluster uncertainty, online
verification-boundary behavior, and fused end-to-end latency evidence.

## Authorization boundary

- A fresh independent review must score at least 9.0/10 and return `READY`
  before CPU implementation.
- CPU implementation does not authorize full-data outcome materialization or
  any GPU job.
- Outcome preparation, capacity execution, fit execution, falsifier opening,
  and development opening each require their own experiment-bridge/code/result
  review and an explicit bounded GO.
- The pending formal D64/D640 array remains untouched, and its pinned Direct
  trainer/head sources are not modified.

