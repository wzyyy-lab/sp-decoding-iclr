# Final Proposal: Tie-Safe Cost-Augmented Max-Regret Selection

## Problem and evidence

The goal is to improve accepted draft length over released DFlash on its
frozen K16 lattice with one parallel head and at most one token edit. Flat
FMAS CE was cost-insensitive and harmful; dense SAVS MSE fit aggregate values
but missed sparse positive decisions. On the frozen SAVS capacity probe,
harmful output gradients initially exceeded beneficial gradients by
`1,518.6x`, beneficial-sign recall was `0.78125`, and oracle-gap recovery was
`0.44546` despite RMSE `0.006909`.

CAMRS changes only the training objective. It retains the D64/H4/L1 axial
head, frozen features, 226-action one-edit space, residual-difference scores,
zero initialization, and strict-positive KEEP decoder. It is a fresh route,
not a weighted or continued SAVS checkpoint.

## Exact action semantics

For accepted-prefix length `A`, block length `L=15`, and action set containing
KEEP plus 225 edits:

```text
v(a,x) = [A(a,x)-A(KEEP,x)]/L,    v(KEEP,x)=0.
s(KEEP,x)=0,
s((i,r),x)=rho(i,r)-rho(i,0).
```

Targets use gold only during training/evaluation. Inference is gold-free. The
oracle is KEEP when `max_edit v<=0`; otherwise it is the lowest-index utility
maximizer. Capacity repairable blocks each have exactly one beneficial action.
Deployment keeps DFlash when every edit score is `<=0`; positive score ties
choose the lowest edit index.

## Tie-safe cost-augmented loss

All objective and gate arithmetic is FP32. For each non-oracle action:

```text
m(a) = s(a)+v(a*)-v(a)-s(a*)
c    = lowest-index argmax_{a != a*} m(a)
H(x) = ReLU(m(c)), with d ReLU(0)/d x = 0
H    = uniform mean over blocks.
```

The explicit non-oracle maximum plus zero-gradient ReLU makes `s=v` both
zero-loss and stationary. There are no class weights, focal factors,
temperatures, auxiliary losses, learned KEEP bias, or tunable deployment
thresholds.

If deployed `a_hat != a*`, then `a_hat` is included in the non-oracle maximum
and `s(a_hat)>=s(a*)`, so

```text
H(x) >= s(a_hat)+v(a*)-v(a_hat)-s(a*)
     >= v(a*)-v(a_hat).
```

If `a_hat=a*`, regret is zero. Thus CAMRS pointwise upper-bounds exact
normalized deployed regret for KEEP, beneficial, neutral, and harmful choices.
At initialization, every repairable block supplies an undiluted upward oracle
gradient and a downward gradient for its worst cost-augmented competitor;
no `1/225` positive dilution remains.

## Gate 0: CPU semantics

Tests must prove dense-utility/brute-force equality; deterministic oracle,
deployment, and competitor ties; zero loss/zero score and residual gradient at
`s=v`; upper bounds for every deployed action type; explicit/original loss
value equivalence; randomized minimum slack `>=-1e-6`; residual-coupled
gradient directions; undiluted `-1/B` oracle gradient; exact first/second
backward behavior; and reconstruction of every saved metric.

## Required optimization diagnostics

Every epoch records:

- block-mean hinge, exact decoded regret, bound minimum slack/violation count;
- competitor/deployed equality, raw-score rank, utility sign/regret,
  cost-augmentation-only wins, distinct coverage, churn, and zero-loss blocks;
- oracle-upward/competitor-downward projection norms, cosine, cancellation;
- unclipped gradient norms and clipped-step fraction;
- beneficial sign, utility-optimal action, harmful sign, repair, precision,
  false edit, harm, EAL, and prompt-balanced oracle-gap metrics;
- every gate boolean, `joint_gate_passed`, selection key, and selected flag.

The early competitor will often be the most harmful action rather than the
deployed boundary. This is an explicitly measured finite-optimization risk;
no auxiliary boundary loss is allowed.

## Gate 1: frozen capacity falsifier

Use only the existing adaptive 512-train-block / 459-prompt manifest. Train a
fresh D64/H4/L1 axial-additive model with K16, batch32, seed0, LR `6e-4`,
warmup `0.04`, zero dropout/weight decay, clip1.0, 320 epochs, and exactly
5,120 updates. Select the earliest exact minimum block-mean hinge checkpoint.

The manifest contains 462 oracle-gain tokens, giving

```text
V*_block = 462/(512*15) = 0.06015625
H threshold = 0.05*V*_block = 0.0030078125.
```

The selected checkpoint must jointly meet:

- zero FP32 bound violations beyond `1e-6`, minimum slack `>=-1e-6`;
- block-mean hinge `<=0.0030078125`;
- beneficial strict-positive recall at least `254/256`;
- utility-optimal action accuracy at least `244/256` repairable blocks;
- harmful nonpositive recall `>=0.99` over 57,765 harmful actions;
- prompt-balanced one-edit oracle-gap recovery `>=0.95`;
- selected harm at most `5/512`;
- false edits at most `2/256` no-benefit blocks;
- exact 256 beneficial actions, finite gradients, and epoch-zero identity.

The report separately lists every jointly passing epoch. A nonselected pass
cannot rescue checkpoint-rule failure. Failure closes exactly this objective,
model, optimizer, schedule, manifest, and selection combination; no longer
training, D640, margin scaling, smoothing, thresholds, weighting, or extra
seed is allowed.

## Conditional development

Only a capacity PASS followed by fresh result/code review may authorize one
seed-0 run on 99,356 prompts / 793,989 blocks for 37,221 updates. Validation is
the physically isolated 147-prompt / 1,175-block collection. Before launch,
exact Direct-native and Direct-one-edit artifact paths, hashes,
configurations, and evaluation semantics must be frozen. Selection is raw
prompt-balanced EAL, then lower harm, then lower hinge, preserving earliest
ties.

Advance requires `EAL_CAMRS-EAL_DFlash>0.28499`, at least `+0.05` over both
Direct controls, harm `<=0.05`, and at most one fewer first-token-correct block
than Direct-native on exactly 1,175 blocks. Development is route selection,
not paper evidence; seeds1/2, formal test, calibration, thresholds, and rollout
remain closed.

## Claim and novelty limits

The pointwise bound survives expectation for realized examples, but no Fisher
consistency or calibrated-value claim is made. Conflicting utilities for
indistinguishable inputs can prevent zero population hinge. Same-subset
capacity cannot establish generalization, identifiability, frozen-feature
sufficiency, or safety.

Multiclass margin and structured loss-augmented inference are established
prior art. Acceptance prediction and payoff-guided speculative scheduling are
also established. The provisional contribution is only their task-specific
integration: frozen-lattice one-edit counterfactual prefix regret,
KEEP-anchored identity scores, and deployment-aligned cost-sensitive training.
No algorithmic-hinge or “first” claim is permitted without broader search and
positive held-out evidence.

## Authorization

Round-2 independent review scored this proposal `9.2/10 READY`. This authorizes
CPU implementation and semantic tests only. A fresh experiment-bridge GO is
required before exactly one capacity GPU job.
