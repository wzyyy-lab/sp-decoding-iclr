# Initial Proposal: Cost-Augmented Max-Regret Selection (CAMRS)

## Problem anchor

The end goal is unchanged: improve real accepted draft length over released
DFlash on its frozen K16 lattice without target calls, autoregressive
selection, tree verification, or more than one token edit. The immediate
bottleneck is now sharply localized:

- flat 226-way FMAS CE improved classification while causing `31.7--35.0%`
  harm because it ignored the cost of a wrong action;
- dense signed-value SAVS achieved RMSE `0.006909` and zero selected harm on
  the 512-block capacity set, yet recovered only `0.44546` of the one-edit
  oracle gap because action-average MSE hid rare positive decisions;
- at SAVS initialization, harmful actions dominated the output-head gradient
  by `1,518.6x`; 56/256 beneficial values remained nonpositive at the selected
  checkpoint.

The new route changes exactly the supervised decision objective. It keeps the
same action space, features, residual-difference parameterization, decoder,
model size, and capacity manifest. It is not a weighting or continuation of
the failed MSE route.

## Action utility and scores

For block `x`, let `A(a,x)` be accepted-prefix length after KEEP or one edit,
`L=15`, and

```text
v(a,x) = [A(a,x)-A(KEEP,x)] / L,     v(KEEP,x)=0.
```

The action set has 226 elements. Let `a*(x)` be the true utility-maximizing
action, with KEEP preferred on any tie. Reuse the exact-identity head:

```text
s(KEEP,x) = 0,
s((i,r),x) = rho[i,r]-rho[i,0].
```

The output is now called a decision score, not a calibrated value estimate.
At zero initialization every edit score is exactly zero. Deployment selects
the maximum edit only if its score is strictly positive; otherwise it keeps
the DFlash path.

## Cost-augmented maximum-regret objective

Define the true action regret relative to the one-edit oracle:

```text
Delta(a*,a;x) = v(a*,x)-v(a,x) >= 0.
```

For each block, train one cost-augmented structured hinge:

```text
L_CAMRS(x) = max_a [s(a,x) + Delta(a*,a;x)] - s(a*,x).
L_CAMRS    = mean_x L_CAMRS(x).
```

There are no class weights, focal factors, temperatures, auxiliary losses,
learned KEEP bias, or tunable inference thresholds. Each block contributes
one hardest loss-augmented violation instead of averaging 225 edit errors.

If `a_hat` is the deployed score maximizer with KEEP-preferred ties, then

```text
L_CAMRS(x)
 >= s(a_hat)+v(a*)-v(a_hat)-s(a*)
 >= v(a*)-v(a_hat).
```

Thus the per-block training loss directly upper-bounds the exact normalized
deployed regret. The dense true score `s=v` attains zero loss. At all-zero
initialization, a repairable block pushes its oracle edit upward and its
currently worst cost-augmented competitor downward; a no-benefit block pushes
its worst harmful competitor downward. The gradient on a repairable oracle is
not divided by the 225-action count.

This is a standard loss-augmented structured-margin construction specialized
to the DFlash one-edit action lattice. The generic structured hinge is not a
novel contribution.

## Why this is a new mechanism

Action-uniform SAVS optimizes average score calibration and gives every edit
an equal `1/225` share of a block. CAMRS instead optimizes a pointwise upper
bound on the deployed max-policy regret and exposes exactly one current worst
decision constraint per block. It changes the geometry and active examples,
not their scalar weights. The frozen SAVS checkpoint, optimizer state, and
results are never reused.

## Required diagnostics

Every evaluation must report, in addition to prompt-balanced EAL:

1. mean and maximum CAMRS hinge;
2. mean exact decoded regret and empirical verification that
   `hinge >= decoded regret` for every block within `1e-7`;
3. loss-augmented competitor counts by true utility sign and action type;
4. beneficial strict-positive recall, oracle-action accuracy, harmful
   nonpositive recall, deployed harm, repair recall, edit precision, and
   no-benefit false-edit rate;
5. epoch-zero output-projection gradient norms for repairable-oracle upward
   terms and cost-augmented-competitor downward terms;
6. epoch history maxima for the behavior gates, so checkpoint selection cannot
   hide a passing epoch.

## Gate 0: CPU semantics

Tests must prove:

- dense utilities and oracle actions match brute-force token-ID reconstruction;
- loss-augmented inference and KEEP-preferred ties are exact;
- the hinge is nonnegative, equals zero for `s=v`, and upper-bounds decoded
  regret on exhaustive fixtures and randomized finite tensors;
- zero scores deploy exact DFlash and yield the analytically expected
  oracle/competitor gradient signs;
- a repairable block's oracle gradient is not divided by action count;
- only the zero-initialized output projection receives first-backward
  gradient, upstream parameters receive gradient after one optimizer update,
  and frozen inputs never receive gradients;
- saved example records reconstruct every gate and diagnostic.

## Gate 1: new capacity-only falsifier

Use the already frozen 512-train-block / 459-prompt manifest only as an
adaptive same-subset engineering falsifier. Freeze D64/H4/L1 axial-additive,
K16, batch 32, 320 epochs, seed 0, LR `6e-4`, warmup `0.04`, zero dropout and
weight decay, gradient clip 1.0, and exactly 5,120 fresh updates. Select the
earliest checkpoint attaining the minimum mean CAMRS hinge; no behavior metric
may select a checkpoint.

All conditions are conjunctive:

- empirical `hinge >= decoded regret` for every block;
- mean CAMRS hinge `<= 0.05 * mean one-edit-oracle advantage`;
- beneficial strict-positive recall `>=0.99` (at least 254/256);
- oracle-action accuracy on the 256 repairable blocks `>=0.95`;
- harmful nonpositive recall `>=0.99`;
- decoded one-edit oracle-gap recovery `>=0.95`;
- selected harmed fraction `<=0.01`;
- no-benefit false-edit fraction `<=0.01`;
- exact 256 beneficial actions, finite gradients, and epoch-zero identity.

A failure closes this exact structured objective/parameterization/schedule;
there is no longer training, D640, margin rescaling, extra seed, threshold, or
class-weight rescue. A pass establishes only same-subset decision-objective
capacity and authorizes fresh code/result review before one isolated
development run.

## Conditional development gate

Only after Gate 1 PASS and a new authorization may seed 0 train on the exact
99,356-prompt / 793,989-block collection for three epochs and 37,221 updates.
Checkpointing uses physically isolated `validation_select` and chooses by raw
prompt-balanced EAL, then lower harm, then lower CAMRS hinge, with strict
improvement preserving the earliest tie.

Advance requires all of:

- `EAL_CAMRS - EAL_DFlash > 0.28499`;
- `EAL_CAMRS - max(EAL_Direct-native,EAL_Direct-one-edit) >= 0.05`;
- harmed fraction `<=0.05`;
- first-token accuracy no more than `0.001` below Direct-native.

Development selection is not a paper estimate. Seeds 1/2, formal test,
calibration, threshold sweeps, and rollout remain forbidden until a later
result-to-claim review.

## Novelty boundary and closest work

- Crammer--Singer multiclass margins and structured SVM loss-augmented
  inference establish that cost-augmented max-margin training is prior art:
  <https://www.jmlr.org/papers/v2/crammer01a.html> and
  <https://jmlr.org/papers/v6/tsochantaridis05a.html>.
- SpecDec++ predicts acceptance probability to adapt candidate length:
  <https://openreview.net/forum?id=NnExMNiTHw>.
- Hybrid Verified Decoding predicts accepted-length payoff to select a draft
  source: <https://openreview.net/forum?id=vr5iRoUn0I>.
- AngelSpec/DFly combines expected utility with profiled cost for runtime
  verification allocation: <https://arxiv.org/abs/2607.25852>.

Therefore neither max-margin learning, accepted-length prediction, nor
utility-guided speculative scheduling is novel. The narrow candidate method
claim is the first-miss frozen-lattice formulation: exact counterfactual
one-edit regrets, a KEEP-anchored exact-identity score head, and a
cost-augmented per-block surrogate that provably upper-bounds deployed
one-edit regret. Novelty remains provisional until broader search and positive
held-out evidence.

## Authorization boundary

- This initial proposal authorizes only independent research review.
- `READY >=9.0` authorizes CPU implementation and tests, not GPU.
- A fresh experiment-bridge GO is required for exactly one capacity job.
- No result from the previously inspected capacity subset may be presented as
  confirmatory or submission-facing evidence.
