# Research Proposal: Signed Action-Value Selection for Frozen DFlash Lattices

## Problem anchor and evidence boundary

The goal remains a deployable fixed-depth, single-chain improvement over
released DFlash, using the frozen DFlash-b16 features and Qwen3-4B target
embedding, K16, and 15 draft positions. No target call, recurrence, tree
verification, backbone update, new candidate data, or sealed-test access is
allowed.

The prior first-miss action space remains well motivated but its flat
canonical-action CE route is now binding negative. Full-data job `10133114`
completed all 37,221 updates and correctly selected epoch 0. Across trained
epochs, held-out action CE improved from `4.12094` to `2.53864` and action
accuracy rose from `16.26%` to about `33%`, while EAL fell by
`0.424–0.505` and harm reached `31.7–35.0%`. This closes flat action CE; no
threshold, D640, longer-training, or multi-seed rescue of that result is
permitted.

It does not close one-edit intervention. On the isolated development lattice,
the one-edit oracle improves prompt-balanced EAL from `5.11200` to `6.64407`.
The D64 model also memorized the frozen 512-block canonical mapping perfectly,
which is only a function-class witness and not an information-sufficiency
claim.

## Failure geometry

For block `x`, let `A(a,x)` be the number of consecutively accepted draft
tokens under KEEP or one edit, and let `b(x)=A(KEEP,x)`. Exhaustive,
independently reviewed enumeration on all 1,175 isolated development blocks
gives 264,375 non-KEEP actions:

| signed utility | actions | fraction | mean nonzero magnitude |
|---|---:|---:|---:|
| beneficial | 984 | 0.3722% | +1.829 tokens |
| harmful | 90,120 | 34.0879% | -5.304 tokens |
| neutral | 173,271 | 65.5399% | 0 |

Flat CE identifies one canonical class but assigns the same error to a neutral
late edit and an early edit losing five tokens on average. Its classification
surrogate is pointwise consistent only at perfect classification; under the
observed errors it is not cost-sensitive to deployed EAL. The new mechanism
must therefore change the supervision, not the action space, data, backbone,
or decoder complexity.

## Method thesis: SAVS

**Signed Action-Value Selection (SAVS)** learns the conditional accepted-prefix
advantage of every one-edit intervention. For each action `a=(i,r)`, define the
exact dense target

```text
v(a,x) = [A(a,x) - b(x)] / L,     L = 15.
v(KEEP,x) = 0.
```

Thus benefits are positive, harmful early edits are negative in proportion to
their actual lost prefix, and neutral actions are zero. Gold tokens construct
these targets only during training/evaluation. Inference remains gold-free.

### Model and identity

Reuse `GlobalDirectCandidateSelector` unchanged, but interpret only its learned
residual score as action value. If `rho[i,r]` is the residual score, define

```text
value_hat(KEEP) = 0
value_hat(i,r)  = rho[i,r] - rho[i,0],    r=1,...,K-1.
```

The residual projection is exactly zero-initialized, so every edit value is
zero before training. Stable argmax puts KEEP at index zero and exactly
reproduces DFlash, including tied candidate logits. Candidate logits and ranks
remain input features, but frozen DFlash log-probability gaps are not
misinterpreted as predicted token-valued utility.

### Objective and decoder

The first falsifier uses one hyperparameter-free, action-uniform squared loss:

```text
L_value = mean_blocks mean_225_edits (value_hat(a,x) - v(a,x))^2.
```

No class rebalance, focal term, CE auxiliary, reward temperature, learned
KEEP bias, or validation-tuned threshold is permitted. Under squared loss,
the population prediction for each action is its conditional mean signed EAL
advantage; selecting the largest positive prediction is the corresponding
plug-in utility decision. Natural action frequencies retain the true asymmetry
rather than manufacturing a balanced classification problem.

At inference choose the maximum predicted edit value. Apply it iff that value
is strictly greater than zero; otherwise KEEP. Exact zero ties therefore stay
with DFlash. The decoder changes at most one token and adds the same O(LK)
argmax overhead as FMAS.

## Why this is a clean mechanistic test

- It supplies dense labels for harmful and neutral alternatives instead of
  collapsing them into undifferentiated non-target classes.
- Harm is supervised with its exact prefix cost, not a binary proxy.
- It changes only the loss/output interpretation that the negative result
  identified; backbone, features, optimizer family, action set, and deployment
  complexity stay fixed.
- Zero-residual initialization gives a stronger identity contract than using
  full DFlash score gaps as action logits.
- Failure of unweighted value regression is interpretable. It does not permit
  a post-hoc positive weighting or threshold sweep; either would be a new
  route.

## Risks and required diagnostics

1. **Rare positive utility:** only one action on a repairable block is
   beneficial. Report positive/zero/negative RMSE and sign accuracy separately,
   but do not rebalance the loss after seeing results.
2. **Max-over-actions noise:** small positive errors among 225 edits can cause
   harmful selection. Report maximum predicted value, edit coverage, harmful
   false-positive rate, and value calibration by predicted-value bins. The
   primary threshold remains zero.
3. **Neutral-action ambiguity:** action accuracy is no longer primary. Report
   realized EAL, harm, repair precision/recall, signed value RMSE, and regret
   to the one-edit oracle.
4. **Capacity versus distribution:** a fixed-subset pass is only an
   optimization witness. A failure closes SAVS-D64; no D640 diagnostic is
   authorized because the predecessor already established D64 function-class
   capacity and the new question is objective behavior.
5. **Control contamination:** no `validation_gate` or formal record may enter
   memory. Continue using the physically isolated `validation_select`
   collection and hash-pinned data helper.

## Gated validation plan

### Gate 0 — CPU semantics

Tests must prove:

1. every dense signed target equals brute-force one-edit decoding;
2. beneficial, neutral, and harmful hand fixtures have exact token-valued
   advantages and normalized values;
3. residual-only action values initialize to exact zero and KEEP wins all ties;
4. decoding changes zero or one position and uses strict-positive edit gating;
5. squared loss is finite, includes all 225 edits per block, and propagates the
   expected two-step gradients without touching frozen inputs;
6. evaluation accounting reconstructs EAL, regret, repair, harm, sign metrics,
   and per-domain prompt-balanced summaries exactly.

### Gate 1 — capacity-only GPU falsifier

Reuse the frozen 512-block capacity manifest solely as an optimization set.
Run D64/H4/L1 axial-additive, K16, batch32, 320 epochs, seed0, LR `6e-4`, zero
dropout/weight decay, warmup `0.04`, and exactly 5,120 steps. Select only by
minimum full dense-value MSE, then earliest epoch on an exact tie.

The selected checkpoint must jointly satisfy:

- all-action normalized RMSE `<=0.02`;
- beneficial-action sign recall `>=0.99`;
- harmful-action nonpositive recall `>=0.99`;
- decoded one-edit oracle-gap recovery `>=0.95`;
- harmed fraction `<=0.01`;
- finite gradients and exact epoch-zero identity.

Failure closes this exact SAVS objective. Passing is still same-subset evidence
and authorizes only a separately reviewed seed-0 development job.

### Gate 2 — prompt-diverse development, conditional only

Only after Gate 1 and a second fresh code/result review, run the exact existing
full OPB contract: 99,356 prompts, 793,989 blocks, batch64, three epochs,
37,221 steps, seed0, LR `6e-4`, warmup `0.04`, and the physically isolated
147-prompt selection set. Select checkpoint lexicographically by raw
prompt-balanced SAVS EAL, then lower harm, then lower dense-value MSE; strict
`>` retains the earliest exact tie.

Advance only if all raw point-estimate conditions hold:

- `EAL_SAVS - max(EAL_Direct-native, EAL_Direct-one-edit) >= 0.05`;
- `EAL_SAVS - EAL_DFlash > 0.28499`;
- harmed fraction `<=0.05`;
- first-token accuracy is no more than `0.001` below Direct-native.

No calibration, threshold sweep, seeds1/2, or formal data is authorized by
this proposal. Those require a positive Gate 2 and another frozen review.

## Claim scope

Potential contribution: dense signed action-value supervision for a
base-preserving one-edit speculative decoder, explicitly pricing longest-prefix
benefit and harm. This is not claimed novel or effective until it beats the
identical Direct checkpoint under both native and one-edit decoding, survives
multiple seeds, and then succeeds on untouched formal data. A negative closes
this exact unweighted value-regression route only; it cannot establish an
information ceiling or generic impossibility.

## Compute authorization requested

- Now: CPU implementation/semantics only after proposal review reaches READY.
- Then: one D64 512-block capacity job only after fresh experiment-bridge GO.
- Full OPB, additional seeds, formal test, and end-to-end rollout remain
  unauthorized.
