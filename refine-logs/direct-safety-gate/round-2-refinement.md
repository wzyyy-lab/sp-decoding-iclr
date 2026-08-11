# Round 2 Refinement: Exact PROS-Gate Protocol Closure

## Status and precedence

This amendment makes the three exact changes required by the Round-2 review.
It changes no method claim, action, loss, model width, validation gate, or
authorization boundary. Where it conflicts with
`round-1-refinement.md`, this document is binding. All other Round-1 terms
remain unchanged.

This document authorizes only a short Round-3 re-review. CPU implementation and
all GPU/data-materialization actions remain closed.

## Revision 1: unique producer-out-of-sample split

The only permitted within-domain prompt split is:

| Domain | Collected | Fit | Checkpoint | Falsifier |
|---|---:|---:|---:|---:|
| chat | 655 | **523** | **66** | **66** |
| code | 665 | **531** | **67** | **67** |
| math | 667 | **533** | **67** | **67** |
| **Total** | **1,987** | **1,587** | **200** | **200** |

For each domain, sort all collected Phase-3 `train` prompt IDs by the frozen
hash already declared in Round 1. Assign the first fit count, the immediately
following checkpoint count, and the final falsifier count. No alternative
rounding, ratio, or table exists. Split generation still precedes every Direct
outcome computation, and all overlap/provenance checks remain binding.

## Revision 2: exact capacity optimizer and binding selected checkpoint

The 512-record prompt-unique capacity manifest uses this one complete
optimization contract:

```text
batch size                 32
passes                     320
steps per pass             16
total updates              5,120
optimizer                  AdamW
betas                      (0.9, 0.999)
epsilon                    1e-8
learning rate              6e-4
weight decay               0
dropout                    0
gradient clip              1.0
warmup steps               floor(0.04*5120 + 0.5) = 205
post-warmup schedule       cosine to zero
seed                       0
capacity empirical weight  1/512 for every prompt-unique record
```

The Direct producer is frozen/eval/detached. The sidecar begins from its exact
zero-output initialization. Evaluate after epoch zero and every complete
16-step pass. After all 320 passes, choose the earliest checkpoint whose
full-precision prompt-weighted capacity loss equals the exact minimum over all
321 evaluated checkpoints.

The **selected checkpoint itself** must pass every conjunctive capacity gate
declared in Round 1. A passing nonselected checkpoint, the first-passage
diagnostic, or a later checkpoint cannot rescue selected-checkpoint failure.
First passage remains descriptive only. Epoch zero remains eligible for
selection but cannot pass the nontrivial behavior gates.

Capacity remains plumbing/memorization evidence only and cannot support the
method claim.

## Revision 3: exact features, modules, warmup, ordering, and recovery

### Fixed scalar definitions

For top-K logits `l_k`, `K=16`, let

```text
q_k = softmax(l)_k
entropy = -sum_k q_k * log(q_k) / log(16)
retained_mass = tanh((logsumexp(l) - base_logsumexp) / 2)
rank_feature = selected_rank_index / 15
position_feature = zero_based_position / 14
change_count = number_of_positions_with_pD_not_equal_0 / 15
```

All `log`, `logsumexp`, `softmax`, reductions, margins, and scalar transforms
are computed in float32. The producer node states are detached in their native
forward dtype and converted to float32 before the sidecar LayerNorm.

The other fixed per-position scalars retain their Round-1 definitions:

```text
direct_total_margin = direct_scores[i,pD[i]] - direct_scores[i,0]
direct_residual_margin = direct_residual[i,pD[i]] - direct_residual[i,0]
dflash_logprob_difference = base_log_probs[i,pD[i]] - base_log_probs[i,0]
change_indicator = 1 if pD[i] != 0 else 0
```

There is no clipping or learned normalization of these eight scalars.

### Exact module attributes

Both LayerNorm modules use `elementwise_affine=True` and `eps=1e-5`. Every
hidden Linear layer uses `bias=True`. The final `Linear(64,1)` uses
`bias=False` and its weight is exactly zero. These choices preserve the exact
38,674 trainable-parameter count:

```text
LayerNorm(200)                           400
Linear(200,64,bias=True)              12,864
Linear(64,64,bias=True)                4,160
LayerNorm(257)                           514
Linear(257,64,bias=True)              16,512
Linear(64,64,bias=True)                4,160
Linear(64,1,bias=False)                   64
total                                  38,674
```

Changed-position mean and max pools are all-zero vectors when no position
changes; the final score is independently multiplied by
`1[number_of_changes > 0]`.

### Exact warmup arithmetic

For both capacity and fit training,

```text
warmup_steps = floor(0.04 * total_steps + 0.5).
```

This is the only rounding rule. Capacity therefore has exactly 205 warmup
steps. Fit `total_steps` is still the largest whole-pass multiple at most
5,120, as declared in Round 1, and its integer warmup is frozen before
training.

### Version-independent pass ordering

Serialize a block key as the UTF-8 byte sequence

```text
sample_id || "\0" || decimal(anchor_offset) || "\0" ||
decimal(context_length)
```

with base-10 integers, no sign for nonnegative values, and no whitespace. For
every zero-based pass index, order records lexicographically by

```text
SHA256("pros-fit-order-v1\0" || decimal(pass_index) || "\0" ||
       serialized_block_key || "\0" || fit_manifest_sha256)
```

and use the unhashed serialized block key as the collision fallback. Capacity
uses the identical construction with its capacity-manifest SHA256 substituted
for `fit_manifest_sha256`. Batch boundaries follow this order exactly; the last
fit batch may be smaller than 64. Every complete pass-order sequence and its
SHA256 are frozen and recorded before optimization. No DataLoader shuffle,
language RNG, library sampler, or nondeterministic tie ordering participates.

### Recovery validity

Every reported recovery has

```text
denominator = EAL_binary_oracle - EAL_DFlash
recovery = (EAL_method - EAL_DFlash) / denominator.
```

The two EAL terms and denominator must be finite, and the denominator must be
strictly positive. Recovery is never clipped. For every gating evaluation,
failure occurs if recovery is nonfinite or outside `[0, 1+1e-6]`; the stricter
capacity/falsifier lower bounds then apply. Checkpoint-split values outside
this numerical interval are recorded and make that checkpoint ineligible for
selection. Exact binary-action reconstruction must make values above the upper
tolerance impossible except for numerical error.

## Closure table

| Round-2 blocker | Exact closure |
|---|---|
| conflicting split tables | unique 523/66/66, 531/67/67, 533/67/67 table |
| incomplete capacity optimizer | all optimizer/schedule values and 320-pass contract frozen |
| nonbinding selected capacity checkpoint | earliest exact minimum itself must pass all gates |
| scalar transform ambiguity | exact entropy, retained mass, rank, position, and change formulas |
| implicit module defaults | affine/epsilon/bias/dtype choices and parameter arithmetic explicit |
| warmup ambiguity | half-up floor formula |
| library-dependent ordering | SHA256 ordering with serialized-key fallback and pretraining hashes |
| invalid recovery handling | finite positive denominator and unclipped numerical interval required |

## Unchanged boundary

PROS-Gate remains a DFlash-specific systems component, not a new general
learning principle. The scalar ridge is only a falsifier baseline; capacity is
only plumbing evidence; calibration, learned thresholds, auxiliary losses,
additional backbones, OPB supervision, formal test, refitting, extra seeds,
online rollout, and latency claims remain forbidden.

A Round-3 score of at least 9.0 with `READY` may authorize CPU implementation
and semantic tests only. All material data preparation and GPU execution still
require subsequent experiment-bridge/code review and explicit bounded GO.

