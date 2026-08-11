# Round-1 Refinement: Signed Action-Value Selection

## Anchor check

The research question is unchanged: improve released DFlash using only its
already-materialized frozen candidate lattice, one fixed-depth parallel head,
and at most one base-preserving edit. The revision changes no data, features,
backbone, action space, target calls, or deployment complexity. It tightens the
statistical claim and the falsification metrics around the single proposed
change: dense signed action-value supervision.

## Changes made in response to review

### 1. Population consistency is no longer conflated with finite-model safety

For visible inference features `X`, action-uniform squared loss has the
population minimizer

```text
v_hat*(a, X) = E[v(a, x) | X].
```

If this population risk is attained with adequate capacity and the inference
features are complete for the conditional decision, choosing the largest
strictly positive conditional mean advantage is Fisher-consistent for expected
incremental accepted length. This statement does **not** imply that empirical
action-average RMSE from a finite shared model controls max-policy regret.

Accordingly, SAVS is claimed to repair the **supervision semantics** of flat
canonical-action CE. Whether it repairs finite-model decision behavior is the
experimental question.

### 2. Max-over-225 behavior now has exact, frozen diagnostics

Let `E(x)` be the 225 edit actions, `v` the true normalized advantage,
`a_hat(x)` the strict-positive deployed edit or KEEP, and
`v*(x)=max(0, max_{a in E(x)} v(a,x))`. Report:

```text
no_benefit_false_edit_rate
  = P(max_a v_hat(a,x) > 0 | max_a v(a,x) <= 0)

selected_action_harm
  = P(v(a_hat(x),x) < 0)

edit_selective_precision
  = P(v(a_hat(x),x) > 0 | a_hat(x) != KEEP)

selected_action_regret
  = E[v*(x) - v(a_hat(x),x)]
```

The first denominator contains only blocks with no beneficial edit. The second
uses all blocks and is identical to the deployed harmed fraction. Regret is
reported in normalized units and in tokens (`L * regret`). Repair recall and
oracle-gap recovery remain behavior metrics. If no block selects an edit,
`edit_selective_precision` is `NA`, never silently zero or one.

For beneficial, neutral, and harmful edit classes, also report count, mean MSE,
SSE fraction, target/prediction sign counts, and mean prediction. At epoch
zero, separately backpropagate each class component
`sum_class(error^2)/(B*225)` through the residual projection and report its
gradient norm and cosine with the total projection gradient. These are frozen
diagnostics, not loss weights.

### 3. Gate-1 conclusions are deliberately narrow

A failure closes only this combination:

```text
D64/H4/L1 axial-additive residual-difference head
+ action-uniform MSE
+ LR 6e-4 / fixed schedule
+ frozen 512-record composition
+ minimum-MSE checkpoint rule.
```

It cannot identify whether the cause is the objective, shared capacity,
optimization dynamics, or feature identifiability. Not launching D640 after a
failure is a preregistered compute-routing decision, not a scientific claim
that dense cardinal-value capacity has already been disproved.

### 4. Gate-1 thresholds now have explicit meaning

The capacity manifest contains exactly 256 blocks with one positive action,
or `256 / (512*225) = 0.2222%` positive edit targets. Beneficial sign recall
`>=0.99` therefore requires at least 254/256 positives to be predicted
strictly positive and permits at most two misses.

All-action normalized RMSE `<=0.02` is only a regression-fidelity engineering
threshold. It is not a policy-safety certificate: if every block has one
action that errs by 0.30 while its remaining 224 actions are exact, the global
RMSE is still `0.30/sqrt(225)=0.02`. The actual capacity decision gates are
decoded one-edit oracle-gap recovery `>=0.95` and harmed fraction `<=0.01`,
jointly with the sign checks and RMSE.

### 5. Closest-work boundary is explicit

- [SpecDec++](https://arxiv.org/abs/2405.19715) predicts conditional token
  acceptance and uses a stopping threshold to adapt candidate length.
- [Hybrid Verified Decoding](https://openreview.net/forum?id=vr5iRoUn0I)
  regresses accepted-length payoff to choose between a cache draft and a
  model-based draft.
- [BASTION](https://openreview.net/forum?id=uqeOxztSIS) uses a drafter-side
  expected-acceptance surrogate plus hardware cost to grow a verification
  tree.

Therefore neither value regression, expected accepted length, nor payoff-guided
selection is a novelty claim. The only plausible method distinction to test is:
complete counterfactual signed prefix-advantage labels for all 225
base-preserving one-edit interventions on a frozen DFlash lattice, coupled to
an exact-identity residual-difference head and strict-positive KEEP policy.

## Revised proposal

### Problem and evidence boundary

Flat FMAS CE is binding negative: on the isolated 1,175-block development set,
CE fell from `4.12094` to `2.53864` while EAL fell by `0.424--0.505` and harm
rose to `31.7--35.0%`. The one-edit oracle still improves DFlash EAL from
`5.11200` to `6.64407`, so the action space remains an open opportunity.

The exhaustive edit distribution is highly asymmetric:

| true edit utility | actions | fraction | mean nonzero utility |
|---|---:|---:|---:|
| beneficial | 984 | 0.3722% | +1.829 tokens |
| harmful | 90,120 | 34.0879% | -5.304 tokens |
| neutral | 173,271 | 65.5399% | 0 |

No target call, recurrence, tree verification, new candidate data, backbone
update, sealed-test access, post-hoc threshold, or calibration is allowed.

### Method: SAVS

For block `x`, let `A(a,x)` be the realized accepted prefix under KEEP or a
single edit, `b(x)=A(KEEP,x)`, and `L=15`. The dense exact target is

```text
v(a,x) = [A(a,x) - b(x)] / L,
v(KEEP,x) = 0.
```

Gold constructs targets only in training/evaluation. Reuse the unchanged
`GlobalDirectCandidateSelector`, but use only its learned residual scores:

```text
v_hat(KEEP) = 0,
v_hat(i,r) = rho[i,r] - rho[i,0],  r=1,...,K-1.
```

Per-position residual mean-centering cancels in this difference. The residual
projection is exactly zero initialized, so every edit value begins at exactly
zero. The sole objective is

```text
L_value = mean_blocks mean_225_edits (v_hat(a,x)-v(a,x))^2.
```

No class rebalance, CE auxiliary, focal term, reward temperature, learned KEEP
bias, or tuned threshold is allowed. Decode the maximum predicted edit only
when its value is strictly greater than zero; otherwise KEEP. This changes at
most one token and exactly reproduces DFlash at initialization.

### Gate 0: CPU semantics

Tests must prove:

1. dense targets equal brute-force one-edit decoding for random and hand-built
   beneficial/neutral/harmful cases;
2. normalization and token-valued advantages are exact;
3. residual-only values initialize to zero and strict-positive decoding keeps
   DFlash on every tie;
4. a decoded path changes zero or one position;
5. MSE includes all 225 edits and remains finite; on the first backward the
   zero-initialized residual projection has nonzero gradient while upstream
   parameter gradients are zero, and after the first optimizer update a second
   backward gives nonzero upstream gradients, with frozen inputs untouched;
6. EAL, prompt/domain balancing, repair, harm, sign, max-policy, regret, and
   per-class loss/gradient diagnostics reconstruct exactly.

### Gate 1: one capacity-only falsifier

Reuse the frozen 512 training-block capacity manifest. Run only
D64/H4/L1 axial-additive, K16, batch32, 320 epochs, seed0, LR `6e-4`, zero
dropout/weight decay, warmup `0.04`, exactly 5,120 steps. Select minimum full
dense-value MSE; exact ties retain the earliest checkpoint.

The selected checkpoint must jointly satisfy:

- all-action normalized RMSE `<=0.02` (engineering fidelity only);
- beneficial-action strict-positive recall `>=0.99` (at least 254/256);
- harmful-action nonpositive recall `>=0.99`;
- decoded one-edit oracle-gap recovery `>=0.95`;
- selected-action harmed fraction `<=0.01`;
- finite gradients and exact epoch-zero identity.

Also report every frozen max-policy and class-contribution diagnostic above.
Failure closes only the exact parameterization/objective/schedule/data/rule
combination. It authorizes no D640 rescue. Passing is same-subset optimization
evidence only and requires a separate review before any prompt-diverse run.

### Gate 2: prompt-diverse development, conditional only

Only after Gate 1 and a new GO may one run use the existing 99,356-prompt,
793,989-block training contract and physically isolated 147-prompt selection
set: batch64, three epochs, 37,221 steps, seed0, LR `6e-4`, warmup `0.04`.
Select lexicographically by raw prompt-balanced SAVS EAL, lower harm, then
lower dense MSE; strict improvement retains the earliest exact tie.

This set is used only to route later experiments, not to estimate publishable
effect size. Advance only if all raw point conditions hold:

- `EAL_SAVS - max(EAL_Direct-native,EAL_Direct-one-edit) >= 0.05`;
- `EAL_SAVS - EAL_DFlash > 0.28499`;
- harmed fraction `<=0.05`;
- first-token accuracy is no more than `0.001` below Direct-native.

No threshold sweep, calibration, seeds 1/2, formal data, or rollout is
authorized. A negative result closes only this exact SAVS route and cannot
establish an information ceiling.

## Compute authorization requested

- After READY: CPU implementation and semantic tests.
- After an independent experiment-bridge GO: exactly one D64 512-block
  capacity job.
- Full-data, additional seeds, formal test, and rollout remain unauthorized.
