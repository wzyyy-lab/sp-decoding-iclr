# Round 1 Refinement: Canonical One-Edit Oracle-Action Imitation

## Anchor Check

- **Problem preserved:** improve DFlash accepted length and reduce direct-head
  harm without recurrence, tree verification, target inference, or backbone
  training.
- **Causal hypothesis preserved:** the deployed decision should be a
  base-anchored block intervention rather than 15 independently edited tokens.
- **Claim narrowed:** FMAS CE imitates a canonical pointwise-optimal one-edit
  action.  It is not claimed to be a Bayes-consistent expected-EAL surrogate.
- **Evidence boundary:** all existing oracle and perfect-gate results are
  gold-aware availability bounds, not proof of held-out learnability.

## Changes Made

### 1. Separated model selection from evidence

`validation_select` may select checkpoints and make an engineering routing
decision only.  The existing `validation_gate` cannot repair this because the
repository records that it was already inspected.  A claim-grade comparison
therefore requires the frozen 600-prompt reserved formal test, read exactly
once after all method choices and seeds are frozen.  Bootstrap intervals on
`validation_select` are descriptive only.

### 2. Corrected the optimization claim

The label is now called the **canonical one-edit oracle action**.  KEEP_BASE is
the declared safety tie-break when multiple actions have the same EAL.  Plain
CE is a cheap imitation-learning falsifier; realized prompt-balanced EAL is the
primary metric and action accuracy remains diagnostic.

### 3. Added the missing causal control

Every matched direct checkpoint is evaluated two ways:

1. native unconstrained per-position argmax;
2. KEEP/max-margin one-edit argmax using exactly FMAS's action decoder.

FMAS training uses decoder 2.  Direct-one-edit versus FMAS isolates the
supervision change; direct-native versus FMAS measures the deployable system
change.  The development route requires material improvement over both and
must also exceed the historical best direct delta `+0.28499`.

### 4. Added a conditional rollout gate

Offline canonical evaluation is explicitly a selection/diagnostic layer.  An
end-to-end rollout on disjoint formal prompts must compare DFlash, the matched
direct selector, FMAS, and same-protocol Domino, including latency and tokens/s,
before any deployability or throughput claim.

### 5. Narrowed novelty

D-PACE already supplies acceptance-aware parallel-drafter weighting; VSD also
optimizes acceptance/path utility; DiffuSpec performs candidate-path selection
with prefix-aware scoring.  FMAS's only potential method distinction is the
combination of a frozen DFlash top-K lattice, explicit KEEP_BASE, one
base-preserving edit, and exact DFlash identity initialization.  If FMAS does
not beat the matched direct-one-edit decoder, it is recorded only as an
objective/action-space diagnostic.

## Revised Method Contract

### Problem and constraints

The historical best axial global D64 selector reaches raw prompt-balanced EAL
delta `+0.28499` over DFlash, harms `7.32%` of blocks, repairs `18.29%` of 984
in-K first-miss opportunities, and closes `6.18%` of the K16 oracle gap.  A
gold-aware one-edit oracle reaches EAL `6.64407` from base `5.11200`, leaving
`+1.53207` available in the declared action space.  This motivates a test of
learnability; it does not imply that the frozen features identify the action.

The first admissible model is the unchanged axial-global, additive
D64/H4/L1 `GlobalDirectCandidateSelector`, with frozen released DFlash hidden
states/logits and a frozen Qwen3-4B target-token embedding lookup.  There is no
Qwen forward pass and no gold/target label in `forward`.  K=16, L=15, greedy
temperature zero, and the base/target checkpoints remain frozen.

### Canonical one-edit oracle action

Let `c_i` indicate that DFlash rank zero equals the gold token, and let
`m=min{i: not c_i}`.  For a full-correct block, define `a*=KEEP_BASE`.  If a
first miss exists but its gold token is outside K16, also define
`a*=KEEP_BASE`; KEEP is the canonical safety tie-break among actions that may
be EAL-tied.  Otherwise, if the gold rank at `m` is `r>0`, define
`a*=(m,r)`.

For L positions and K candidates:

```text
A = {KEEP_BASE} union {(i,r): 0<=i<L, 1<=r<K}
|A| = 1 + L(K-1) = 226
index(KEEP_BASE) = 0
index(i,r) = 1 + i(K-1) + (r-1)
```

Among one-edit paths, `a*` is pointwise EAL-optimal.  It need not be the unique
optimal action, and CE on `a*` is not claimed to minimize expected EAL under
feature ambiguity.

### Action logits and identity

Reuse the unchanged direct selector scores `s[i,r]`:

```text
z_KEEP = 0
z_(i,r) = s[i,r] - s[i,0], r>0
L_FMAS = cross_entropy(z, a*)
```

At zero residual initialization, sorted DFlash logits make every edit logit
non-positive.  KEEP is action index zero and wins deterministic ties, so the
decoded path is exactly the rank-zero DFlash path.  The decoder chooses one
action, emits rank zero everywhere for KEEP, or changes exactly one position
for an edit.  It is a fixed-depth O(LK) decision with no recurrent state.

### Formal metric definitions

For action `a`, let `R_b(a)` be the number of consecutive gold tokens from
position zero in the emitted one-edit path.  The canonical one-edit oracle EAL
is the mean of `R_b(a*_b)` within each prompt followed by an equal mean across
prompts.  All primary offline EAL values use this prompt-balanced aggregation.

The reported single-edit oracle-gap recovery is:

```text
(EAL_FMAS - EAL_DFlash) /
(EAL_one_edit_oracle - EAL_DFlash)
```

with a fail-closed undefined result if the denominator is non-positive.  In
addition to exact action recall, gain-weighted repair recall is:

```text
sum_b gain_b * 1[predicted action = a*_b] / sum_b gain_b
gain_b = R_b(a*_b) - R_b(KEEP_BASE)
```

over repairable blocks.  Diagnostics separately report KEEP because the block
is fully correct, KEEP because the first miss is out of K, predicted edits,
neutral edits, improvements, harms, first-token accuracy, selected margins,
position/rank buckets, and domain breakdowns.

## Revised Validation Gates

### Gate 0 — semantics and properties

CPU tests must cover hand examples and exhaustive small L/K lattices:

1. the canonical target is pointwise EAL-optimal among every legal action;
2. full-correct and out-of-K misses map to KEEP, including neutral-after-miss
   ties;
3. encode/decode is bijective and changes at most one position;
4. zero residual scores reproduce DFlash even with tied candidate logits;
5. realized prefix, repair, neutral, and harm accounting are exact;
6. gradients reach edit scores/backbone parameters but never frozen inputs.

### Gate 1 — pinned 512-block capacity probe

Before submission, materialize a manifest of the exact deterministic subset,
including ordered `(sample_id, anchor_offset)` keys, source collection metadata
SHA256, subset SHA256, KEEP-full-correct count, KEEP-out-of-K count, edit count,
position/rank histograms, and gain histogram.  The job must verify the manifest
fail-closed; no dynamically drifting subset is accepted.

Use D64 axial global additive on that same subset for train/evaluation.  Select
by action CE then action accuracy.  Pass only if action accuracy `>=0.97`,
repairable-action recall `>=0.95`, single-edit oracle-gap recovery `>=0.95`,
harm `<=0.01`, exact initialization identity, and finite gradients.  If D64
fails, one D640 diagnostic may distinguish capacity from formulation, but it
cannot authorize D64 development.

### Gate 2 — prompt-diverse engineering development

Use full OPB 99,356 training prompts, 793,989 blocks, batch64, three epochs,
37,221 optimizer updates, LR `6e-4`, zero dropout/weight decay, warmup `0.04`,
seed0, and the exact D64/H4/L1 axial-additive backbone.  Use only
`validation_select` for checkpoint selection and routing.  The matched
Candidate-D-PACE compact cell must match data, model, seed, step budget,
optimizer, frozen target identity, and validation identities fail-closed.

Report these three paths:

| comparison cell | training | decoding |
|---|---|---|
| Direct-native | Candidate-D-PACE | independent per-position argmax |
| Direct-one-edit | same exact Direct-native checkpoint | KEEP/max-margin one edit |
| FMAS | canonical action CE | same KEEP/max-margin one edit |

Advance only if all hold on raw `validation_select` point estimates:

- `EAL_FMAS - max(EAL_Direct-native, EAL_Direct-one-edit) >= 0.05`;
- `EAL_FMAS - EAL_DFlash > 0.28499`, the current historical-best direct gain;
- FMAS harmed fraction `<=0.05` and first-token accuracy is no lower than
  Direct-native by more than `0.001`.

Any bootstrap interval here is descriptive and explicitly selection-biased.
A result within `0.01` of Direct-native with lower harm may be logged as a
safety diagnostic but cannot advance.

### Gate 3 — seed stability, not confirmation

If Gate 2 passes, freeze every method choice and run seeds 1/2 for FMAS and the
matched direct baseline with identical selection rules on `validation_select`.
Require FMAS minus both direct decoders to remain positive in every seed.  This
tests optimization stability only; it cannot produce a claim-grade CI because
the prompts and route have already been observed.

### Gate 4 — one-shot formal offline and online evaluation

Only after Gate 3 passes may the pre-registered 600-prompt reserved formal-test
manifest be collected/evaluated once.  The method, checkpoints/seed aggregation,
decoder, and thresholds must already be frozen.  Offline success requires:

- prompt-cluster 95% CI lower bound above zero for FMAS minus Direct-native
  and FMAS minus Direct-one-edit EAL;
- a one-sided prompt-cluster 95% UCB for FMAS harmed fraction `<=0.05`;
- fixed domain reporting with no post-hoc threshold calibration.

On the same disjoint prompt protocol, run end-to-end rollouts for released
DFlash, Direct-native, FMAS, and same-protocol Domino.  Report accepted length,
first-token accuracy, block harm, selector latency, end-to-end latency, and
tokens/s.  A deployability or throughput claim requires positive tokens/s
benefit after selector overhead; offline EAL alone is insufficient.

## Closest-Work and Contribution Boundary

- D-PACE provides dynamic position-aware acceptance training for parallel
  drafting; FMAS does not claim the first acceptance-aware objective.
- VSD already connects training to verification/path utility; FMAS does not
  claim generic expected-acceptance optimization.
- DiffuSpec already performs prefix-aware candidate-path selection; FMAS does
  not claim the first lattice/path selector.
- The conditional candidate contribution is only a frozen-DFlash,
  base-preserving one-edit intervention action space with explicit KEEP and
  exact identity initialization, and only if FMAS beats the exact matched
  Direct-one-edit decoder.

Failure at capacity closes this parameterization.  Failure at development
closes FMAS as a performance route but may retain a lower-harm diagnostic.
Neither failure proves a frozen-feature information ceiling or that multi-edit
decoding is necessary.

