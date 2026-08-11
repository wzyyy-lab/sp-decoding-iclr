# Research Proposal: First-Miss Action Selection for Frozen DFlash Lattices

## Problem Anchor

- **Bottom-line problem:** Improve DFlash's greedy accepted draft length with a
  fixed-depth selector, while preserving the deployable single-chain setting
  and making measurable progress toward the same-anchor Domino result.
- **Observed bottleneck:** The best axial direct selector improves raw
  prompt-balanced EAL by only `+0.28499`, harms `7.32%` of validation blocks,
  repairs `18.29%` of 984 in-K first-miss opportunities, and recovers only
  `6.18%` of the K16 oracle gap.  Larger or longer-trained direct classifiers
  can lower their classification objective while validation EAL collapses.
- **Non-goals:** No target calls at inference, tree verification, recurrent
  decoder, CRF/Viterbi search, backbone/LoRA training, new candidate data, or
  sealed-test access.  This route does not make an information-ceiling claim.
- **Constraints:** Frozen released DFlash-b16 features and Qwen3-4B target
  embedding; K16 and 15 positions; greedy temperature zero; prompt-disjoint
  OPB train/development artifacts; D64/H4/L1 axial global backbone first; one
  changed mechanism; exact DFlash identity before training.
- **Success condition:** On full OPB-99,356 seed 0, FMAS must exceed a matched
  Candidate-D-PACE direct control in raw prompt-balanced EAL and keep harm at
  or below `5%`.  A lower-harm tie is a safety variant, not a performance win.
  Only a performance win authorizes a three-seed confirmation.

## Evidence That Motivates the Pivot

The current objective asks 240 position/candidate decisions to be correct even
though longest-prefix verification only observes the first wrong deployed
token.  Post-hoc decoder probes on the best historical D64 checkpoint show:

| decoder on the same learned scores | raw EAL delta vs DFlash | harm |
|---|---:|---:|
| unconstrained direct path | `+0.28499` | `7.32%` |
| maximum-margin single edit | `+0.16254` | `3.49%` |
| earliest changed position only | `+0.23312` | `7.32%` |
| top two margin-ranked edits | `+0.23712` | `4.51%` |
| top three margin-ranked edits | `+0.27478` | `5.62%` |

The margin-ranked restrictions reduce harm but do not improve EAL because the
model was not trained to choose a block-level intervention.  A perfect block gate over
the existing direct path would yield `+0.431` EAL, while an oracle that may
make exactly one edit—repair the base path's first miss and retain the base
suffix—raises EAL from `5.11200` to about `6.64407` (`+1.532`).  The one-edit
action space therefore has ample headroom; the missing object is selection of
the correct intervention, not additional edit count.

The feature-capacity result also separates representation from optimization:
the D640 compatibility model memorizes the fixed capacity set, yet repeated
10K training overfits and loses EAL.  The queued fixed-step 99K feature probe
remains a separate positive-only diagnostic.  FMAS does not depend on its
outcome and must not alter that probe's pinned source files.

## Technical Gap

For a base rank-zero path with accepted prefix length `m`, any single edit
before `m` can only harm, any edit after `m` cannot repair the existing first
miss, and the only improving action is selecting the gold in-lattice rank at
position `m`.  If the base path is fully correct or the gold token at `m` is
outside K16, retaining the base path is optimal in the allowed action space.

Per-position candidate CE does not encode this decision.  It rewards suffix
predictions that are irrelevant after a first miss, permits mutually
inconsistent edits at many positions, and never makes KEEP_BASE compete as an
explicit block action.  Policy-reach masking attempted to remove suffix loss,
but its moving support starved hard-candidate learning and failed its capacity
gate.  A static, base-anchored block target avoids both defects.

## Method Thesis

**First-Miss Action Selection (FMAS)** converts the frozen lattice into one
categorical block decision:

```text
A = {KEEP_BASE} union {(position i, non-base rank k)}
```

For L=15 and K=16 this is 226 actions.  The supervised target is:

```text
if base is fully correct:
    KEEP_BASE
elif gold at the base first-miss position is outside K16:
    KEEP_BASE
else:
    (base first-miss position, gold candidate rank)
```

This label is the exact EAL-optimal action within the declared one-edit action
space.  It is computed only during training/evaluation from existing target
labels; inference uses frozen draft-side features alone.

## Proposed Mechanism

### Reused backbone and action logits

Reuse `GlobalDirectCandidateSelector` unchanged.  Let `s[i,k]` be its direct
candidate score, including the frozen DFlash log probability.  Define:

```text
logit(KEEP_BASE) = 0
logit(i, k)      = s[i,k] - s[i,0],  k in {1,...,K-1}
```

Flatten all non-base logits after KEEP_BASE and train ordinary block-level
cross entropy.  The score difference has a direct interpretation: evidence
that one candidate should replace DFlash rank zero at one position.  A fixed
KEEP logit makes confidence comparable across blocks without adding a free
global bias.

At exact zero-residual initialization, all edit logits equal the negative
DFlash top-1 gaps and are non-positive; deterministic tie handling selects
KEEP_BASE.  Thus epoch zero reproduces DFlash exactly.  No modification is
made to the pinned direct-selector head or trainer used by the queued formal
feature probe.

### Decoder

Choose the maximum of the 226 action logits.  KEEP_BASE emits the all-zero
rank path.  An edit action changes exactly one rank and leaves every other
position at rank zero.  This decoder adds no sequential step and has O(LK)
scoring/argmax overhead after the existing fixed-depth backbone.

### Why plain action CE is the first falsifier

- Static labels eliminate policy-conditioned support collapse.
- Every block contributes one loss, so suffix length does not dominate.
- KEEP_BASE and every possible intervention compete in the same normalized
  decision.
- The target is optimal for the deployed action class, not merely correlated
  with per-position accuracy.

No focal weighting, reward weighting, separate gate, temperature, or
multi-edit policy is included in the first screen.  Adding one before testing
plain action CE would obscure whether action-space alignment itself is useful.

## Failure Modes and Required Diagnostics

- **226-way class sparsity:** report target and prediction histograms by
  KEEP/edit, first-miss position, and rank bucket.  A failed capacity gate
  closes D64 FMAS before development; D640 may be used only as an explicitly
  separate capacity diagnostic.
- **Conservative KEEP collapse:** report repair opportunity recall, edit
  precision, repair count, and oracle single-edit gap recovery.  Identity has
  zero harm but cannot pass the EAL gate.
- **Confident harmful edits:** report harmed blocks/fraction, first-token
  accuracy, and margin of selected edit over KEEP.  Calibration is descriptive
  and may not choose the winning checkpoint.
- **Neutral wrong edits after the base first miss:** distinguish action-label
  accuracy from realized EAL; action accuracy is diagnostic, never primary.
- **Historical-control mismatch:** the matched seed-0 direct control must use
  the same full prompt set, steps, backbone, seed, optimizer, target embedding,
  and validation artifact.  The already queued full-data compact cell may be
  reused only if all identities match fail-closed.
- **Single-edit ceiling:** failure does not establish that multi-edit decoding
  is necessary unless FMAS first demonstrates reliable first-miss action
  selection.  It closes this static action parameterization only.

## Claim-Driven Validation Plan

### Gate 0 — deterministic semantics

Unit tests must prove:

1. target encoding returns KEEP for full-correct and out-of-K first misses;
2. it returns exactly the gold action at the base first miss otherwise;
3. action-to-path changes zero or one position only;
4. epoch-zero logits/path exactly reproduce DFlash, including ties;
5. decoded realized prefixes and harm/repair accounting match hand examples;
6. gradients flow through the edit score and not through frozen inputs.

### Gate 1 — fixed capacity probe

Use the existing materialized 512-block capacity artifact and the D64 axial
global additive backbone.  The probe passes only if the selected checkpoint
meets all of:

- action target accuracy at least `0.97`;
- repairable-action recall at least `0.95`;
- single-edit oracle-gap recovery at least `0.95`;
- harmed fraction at most `0.01`;
- all-zero initialization identity and finite gradients.

Selection is by action CE, then action accuracy; capacity EAL cannot select an
epoch.  If D64 fails, one preregistered D640 diagnostic may test whether the
failure is capacity rather than formulation.  It cannot advance directly to
development.

### Gate 2 — full prompt-diverse development

Run D64/H4/L1 axial global, additive encoder, seed 0, K16, batch64, three
epochs, LR `6e-4`, zero dropout/weight decay, warmup `0.04`, and exactly the
same 99,356 prompts / 793,989 blocks / 37,221 optimizer updates as the matched
direct control.  Evaluate only the existing 147-prompt / 1,175-block
`validation_select` split.  Choose the checkpoint by raw prompt-balanced EAL,
then harm, then action accuracy.

FMAS is a **performance win** only if:

- raw prompt-balanced EAL exceeds the matched direct control; and
- harmed fraction is at most `0.05`.

Report paired prompt-cluster bootstrap intervals for FMAS minus direct and
FMAS minus DFlash, plus fixed chat/code/math breakdowns.  A result within
`0.01` EAL of direct with materially lower harm may be retained as a clearly
labeled safety variant, but it does not authorize the paper-performance route.

### Gate 3 — confirmation

Only after Gate 2 passes, freeze the configuration and run seeds 0/1/2 for
FMAS and the matched direct control.  Advance a claim only if FMAS-direct is
positive in every seed and the paired prompt-cluster 95% CI lower bound is
above zero.  The sealed test remains unopened until method selection ends.

## Contribution Scope

- **Potential contribution:** an explicit base-preserving intervention policy
  whose labels and decoder are exactly aligned to first-miss repair under
  longest-prefix speculative verification.
- **Supporting evidence:** action-space oracle headroom, exact DFlash identity,
  matched direct control, prompt-cluster inference, and harm accounting.
- **Not claimed:** generic optimality beyond one-edit paths, new lattice
  features, information sufficiency, or superiority to Domino before matched
  end-to-end evidence exists.

## Compute Estimate

- Semantics/tests: CPU only.
- D64 512-block capacity: under 15 A40 minutes with a generous fail-safe.
- Optional D640 capacity diagnostic: under 20 A40 minutes based on the observed
  27.5M-parameter probe.
- D64 full-data seed-0 screen: about the same model-update budget as the queued
  compact direct control; full-data loading/evaluation may dominate wall time.
- Three-seed confirmation is contingent and is not launched by this proposal.
