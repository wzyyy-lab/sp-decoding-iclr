# Final Proposal: Producer-Reusing Out-of-Sample Direct Safety Gate

## Thesis

PROS-Gate learns one conservative block-level decision: keep released DFlash's
rank-zero path or apply the complete path of one hash-frozen Direct producer.
It addresses CAMRS's observed max-over-225 harmful-score tail by eliminating
action-level maximization, reuses the producer's already contextualized node
states through a 38,674-parameter sidecar, and trains only on prompts that the
producer did not see.

This is a narrow DFlash systems component. It is not a general routing,
calibration, or selective-prediction claim.

## Frozen evidence and target

The released-DFlash development baseline is prompt-balanced EAL
`5.112001943634597`. The frozen matched Direct-native producer reaches
`5.334669582118561`. Its exact KEEP/Direct binary oracle reaches
`5.430758017492711` with zero harm.

Direct improves 141/1,175 blocks for 374 unweighted accepted tokens, is neutral
on 972, and harms 62 for 113 tokens. The strict target is above
`5.396991943634597`, requiring more than 89.4069% of the binary-oracle gap and
approximately 64.9% recovery of Direct's avoidable loss. Oracle slack is only
`0.033766073858114` EAL.

The frozen producer is job `10133585`:

- checkpoint SHA256
  `9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e`;
- metrics SHA256
  `9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef`;
- global axial-additive D64/H4/L1, K16, seed 0;
- 433,772 parameters, frozen, evaluation mode, no gradients.

Native Direct scores, paths, accepted lengths, summaries, config, state, and
source identities must reproduce exactly before and after sidecar attachment.

## Actions and inference

For each stored or online block:

```text
KEEP:  emit released DFlash path p_B[i] = 0
APPLY: emit complete frozen Direct argmax path p_D
```

The sidecar outputs scalar `z`. Deployment is fixed:

```text
APPLY iff z > 0; otherwise KEEP.
```

There is no bias threshold, temperature, calibration, domain rule, validation
sweep, or mixture. If `p_D == p_B`, force `z=0`. The sidecar never sees gold
tokens, accepted lengths, target outputs, or target IDs at inference.

## Producer-reusing sidecar

Capture the frozen Direct selector's detached final normalized node states
`h_D` immediately before its existing residual projection. Capturing states
must not change the native Direct result or state dictionary.

For position `i`, gather the Direct-selected and rank-zero nodes. Concatenate
the following 200 values:

```text
h_D(i,p_D[i])                            64
h_D(i,0)                                 64
h_D(i,p_D[i]) - h_D(i,0)                 64
Direct total-score margin                 1
Direct residual margin                    1
DFlash log-probability difference         1
selected rank / 15                        1
zero-based position / 14                  1
change indicator                          1
normalized top-K entropy                  1
bounded retained top-K log mass           1
```

For top-K logits `l`, `K=16`, define in float32:

```text
q = softmax(l)
entropy = -sum_k q_k*log(q_k) / log(16)
retained_mass = tanh((logsumexp(l)-base_logsumexp)/2)
```

Margins are exactly:

```text
Direct total = direct_scores[i,p_D[i]] - direct_scores[i,0]
Direct residual = direct_residual[i,p_D[i]] - direct_residual[i,0]
DFlash difference = base_log_probs[i,p_D[i]] - base_log_probs[i,0].
```

All producer values are detached; node states are converted to float32 before
the sidecar. The shared position encoder is

```text
LayerNorm(200, affine=True, eps=1e-5)
-> Linear(200,64,bias=True) -> SiLU
-> Linear(64,64,bias=True).
```

Concatenate four 64-dimensional pools—mean over all positions, mean over
changed positions, max over changed positions, and encoded first position—plus
`number_changed/15`. Empty changed-position pools are zero. The resulting 257
values pass through

```text
LayerNorm(257, affine=True, eps=1e-5)
-> Linear(257,64,bias=True) -> SiLU
-> Linear(64,64,bias=True)
-> Linear(64,1,bias=False).
```

The final weight is exactly zero initialized; all other parameters use
deterministic name-seeded initialization with seed 0. The exact trainable count
is:

| Component | Parameters |
|---|---:|
| position LayerNorm | 400 |
| position 200-to-64 | 12,864 |
| position 64-to-64 | 4,160 |
| block LayerNorm | 514 |
| block 257-to-64 | 16,512 |
| block 64-to-64 | 4,160 |
| final projection | 64 |
| **total** | **38,674** |

The raw output is multiplied by `1[number_changed>0]`. Epoch zero therefore
reproduces released DFlash exactly. Only the final projection receives the
first nonzero backward gradient; upstream sidecar layers receive gradients
after its first update.

## Utility-consistent supervision

Let `A_B` and `A_D` be realized accepted-prefix lengths of DFlash and frozen
Direct for a 15-position block. Define

```text
g = (A_D-A_B)/15
y = sign(g)
loss = abs(g)*ReLU(1-y*z),  if g != 0
loss = 0,                   if g == 0.
```

For prompt `p` with `n_p` blocks, train the prompt-balanced empirical risk

```text
(1/number_of_prompts) * sum_p [(1/n_p)*sum_{x in p} loss(x)].
```

Each record weight is
`N_blocks/(N_prompts*n_p)`, recomputed independently within each split. A mean
over all weighted records exactly equals the declared risk.

This weighted unit-margin hinge makes the conditional decision sign follow
`E[g|deployable features]`. If a beneficial block is wrongly kept (`z<=0`),
loss is at least `g`; if a harmful block is wrongly applied (`z>0`), loss is
greater than `-g`. Thus the loss pointwise upper-bounds decoded normalized
regret. Neutral blocks have zero accepted-token regret and zero loss regardless
of action.

Every evaluation reconstructs decoded regret, loss, slack, and all violations
beyond `1e-6` from saved block records.

## Producer-out-of-sample data protocol

OPB-100K construction explicitly excluded the Phase-3 development manifest.
The frozen Direct producer trained only on OPB. Phase-3 canonical `train` is
therefore producer-out-of-sample and contains 1,987 collected prompts / 15,886
blocks:

| Domain | Collected | Fit | Checkpoint | Falsifier |
|---|---:|---:|---:|---:|
| chat | 655 | 523 | 66 | 66 |
| code | 665 | 531 | 67 | 67 |
| math | 667 | 533 | 67 | 67 |
| **total** | **1,987** | **1,587** | **200** | **200** |

Before any Direct outcome computation, within each domain sort collected
prompt IDs by

```text
SHA256("pros-gate-phase3-oos-v1\0" || domain || "\0" || sample_id ||
       "\0" || canonical_metadata_sha256).
```

Assign the exact fit prefix, then checkpoint prefix, then falsifier suffix.
Freeze/hash the manifest. Prove exact cardinality, domain membership, mutual
disjointness, zero sample-ID overlap with all OPB producer collections and all
validation/reserved splits, and agreement with the frozen OPB exclusion and
Phase-3 collection identities.

- Only fit updates sidecar parameters.
- Only checkpoint selects the checkpoint.
- The selected checkpoint is frozen, opened once on falsifier, and never refit.
- Historically inspected `validation_gate` is never loaded.
- `validation_select` stays closed until a reviewed falsifier pass.
- Formal test remains sealed.

OPB producer outcomes may be labeled in-sample diagnostics only.

## Deterministic block and pass identity

The unique block key is `(sample_id, anchor_offset, context_length)`, serialized
as UTF-8

```text
sample_id || "\0" || decimal(anchor_offset) || "\0" ||
decimal(context_length)
```

with base-10 nonnegative integers and no whitespace.

For each zero-based pass, sort records by

```text
SHA256("pros-fit-order-v1\0" || decimal(pass_index) || "\0" ||
       serialized_block_key || "\0" || training_manifest_sha256)
```

and use the unhashed key as collision fallback. Freeze each ordered sequence
and its SHA256 before optimization. No framework shuffle, sampler, or language
RNG participates.

Warmup is always

```text
floor(0.04*total_steps + 0.5).
```

## Gate 0: CPU semantics

Synthetic CPU tests must establish before real-data or GPU actions:

- strict-positive APPLY, KEEP ties, zero-output identity, and identical-path
  forced zero;
- exact feature dimensions/formulas/dtypes, pools, module defaults, and 38,674
  parameters;
- frozen producer/eval boundary and exact state capture without output change;
- no gradients into producer, canonical inputs, or target embeddings;
- gain-weighted hinge against hand cases, prompt-weight identity, conditional
  collision behavior, neutral zero loss, and randomized regret bounds;
- gain-scaled signed initialization gradients, projection-only first update,
  and later upstream gradient;
- accepted-prefix realization and binary oracle against brute-force token IDs;
- deterministic split counts, block serialization, pass ordering, collision
  fallback, overlap rejection, and all metric denominators;
- finite positive recovery denominator, no clipping, and rejection outside
  `[0,1+1e-6]`;
- forbidden split guards and independent saved-example reconstruction.

Gate 0 does not load the real Phase-3 collection or produce experiment
artifacts.

## Gate 1: same-subset capacity plumbing test

After separately reviewed outcome generation, construct capacity only from
fit. Define:

```text
beneficial:      g>0
harmful:         g<0
changed-neutral: g==0 and p_D!=p_B.
```

In scarcity order harmful, changed-neutral, beneficial, rank blocks by a frozen
SHA256 of protocol ID, serialized block key, producer hashes, canonical hash,
and split-manifest hash. Select 128 harmful, 128 changed-neutral, and 256
beneficial blocks, excluding every already selected prompt. Fail if the exact
prompt-unique composition is unavailable. Capacity is adaptive same-subset
plumbing evidence only.

Optimization is exactly:

```text
batch size                 32
passes / steps per pass    320 / 16
total updates              5,120
optimizer                  AdamW
betas / epsilon            (0.9,0.999) / 1e-8
learning rate              6e-4
weight decay / dropout     0 / 0
gradient clip              1.0
warmup steps               205
post-warmup schedule       cosine to zero
seed                       0
record empirical weight    1/512
```

Evaluate epoch zero and all 320 pass endpoints. Select the earliest exact
full-precision minimum-loss checkpoint among all 321. The selected checkpoint
itself must pass every condition:

- finite values/gradients and zero regret-bound violations;
- mean loss at most 5% of epoch-zero loss;
- benefit recall at least 254/256;
- harm avoidance at least 127/128;
- at least 509/512 zero-regret utility-optimal decisions;
- prompt-balanced binary-oracle recovery at least 0.95;
- at most one harmful APPLY.

First passage and neutral/tie/false-APPLY metrics are diagnostic; no
nonselected checkpoint can rescue failure.

## Gate 2: clean fit, checkpoint selection, and one-shot falsifier

Train one fresh seed-0 sidecar on fit:

```text
batch size                 64
maximum updates            5,120
optimizer                  AdamW
betas / epsilon            (0.9,0.999) / 1e-8
learning rate              6e-4
weight decay / dropout     0 / 0
gradient clip              1.0
warmup                     half-up 4% formula
post-warmup schedule       cosine to zero
```

If `S=ceil(fit_blocks/64)`, run exactly `floor(5120/S)` complete passes, with
no partial pass. Evaluate checkpoint at epoch zero and every complete pass.
Select by strict lexicographic key

```text
(prompt-balanced EAL, -harmed_count, -prompt-weighted_gain_hinge).
```

Exact ties keep the earliest checkpoint. Checkpoint recovery must have finite
positive denominator and lie in `[0,1+1e-6]`; otherwise the checkpoint is
ineligible. Fit behavior, falsifier, and `validation_select` cannot select.

### Frozen scalar comparator

Fit one float64 prompt-weighted ridge model on fit only to predict `g`, with
unpenalized intercept, fit-only standardization, and fixed coefficient `1e-3`.
Its 21 fixed summaries cover change count; sum/mean/min/max of Direct total,
Direct residual, and DFlash margins on changed positions; entropy and retained
mass mean/min/max; and first-position change/margin. Missing changed summaries
are zero. Freeze coefficients/normalizers/hash before checkpoint or falsifier.
Always-KEEP and always-Direct are also fixed comparators. This is a falsifier
baseline, never a second contribution.

### One-shot falsifier

Open the 200-prompt falsifier exactly once with the already selected/frozen
sidecar. Every condition must pass:

- finite data/scores/metrics and zero regret-bound violations;
- finite positive oracle denominator and unclipped recovery in
  `[0,1+1e-6]`;
- recovery at least 0.90;
- prompt-balanced EAL strictly above DFlash and Direct;
- harmed fraction at most 0.05;
- first-token count at most one below Direct;
- recovery at least 0.05 above the best frozen scalar/constant comparator;
- exact split, producer, feature, checkpoint, and native-Direct identities.

Report complete outcome/token mass, APPLY composition, benefit recall, harm
avoidance, neutral APPLY, tie agreement, zero-regret optimality, false APPLY,
oracle regret, score/gain quantiles, domains, and deterministic 10,000-
replicate prompt-cluster intervals.

Failure closes this exact route. No threshold, refit, longer/wider training,
alternate split, extra seed, OPB labels, or validation opening can rescue it.

## Conditional development opening

Only a reviewed falsifier pass can authorize a one-time evaluation on the
already physically isolated 147-prompt / 1,175-block `validation_select`
collection. The frozen model cannot change. Advance requires all:

- `EAL_PROS-EAL_DFlash > 0.28499`;
- `EAL_PROS-EAL_Direct-native >= 0.05`;
- `EAL_PROS-EAL_Direct-one-edit >= 0.05`;
- harmed fraction at most 0.05;
- first-token count at most one below Direct-native;
- finite positive oracle denominator, unclipped valid recovery, exact controls,
  and zero regret-bound violations.

Exact binary-oracle EAL remains `5.430758017492711`. This is development-only
evidence.

## Provenance and failure behavior

Freeze full runtime import closure; split/outcome/capacity/order/comparator
artifacts; producer checkpoint/config/metrics/source hashes; target embedding,
canonical shards, and manifests; native Direct witness; sidecar sources/config;
every evaluated checkpoint and selection key; optimizer/scheduler; prompt sets,
block keys, denominators, environment, and start/end hashes.

Scientific failure must atomically write the complete report before nonzero
exit. No reviewed artifact is overwritten.

## Claims and authorization

Selective prediction, payoff routing, accepted-length heads, binary margins,
and abstention are prior art. The possible contribution is only this
DFlash-specific combination: counterfactual full-path outcomes, exact KEEP
identity, producer-state reuse, utility-aligned supervision, and a strict
producer-out-of-sample stacking/falsification protocol motivated by CAMRS's
observed extreme tail.

Round-3 review authorizes CPU-only implementation and synthetic semantic
tests. It does not authorize real dataset loading, split/outcome/capacity
artifact generation, training, evaluation, GPU execution, formal test, extra
seeds, online rollout, refit, calibration, or latency/efficacy claims. Every
material next gate requires fresh experiment-bridge/code/result review and an
explicit bounded GO.

