# JAPD-16 M1 Experiment Code Review

**Time:** 2026-08-10 15:16:48 CST  
**Scope:** J010 capacity, J011 full-fit, J012 eager profile, and full sidecar  
**Reviewer:** fresh independent GPT-5.6-Sol xhigh agent under `experiment-bridge`

## Verdict

**GO** to execute M1.  J010/J011 remain operationally blocked until the full
R047 sidecar produces a passing GPU replay receipt.  This verdict does not
authorize M2 and is not a performance claim.

## Blocking findings and closure

1. Initial M1 selection did not fail-close on malformed/stale 512-record and
   512-prompt manifests.  The trainer now requires exactly 512 unique capacity
   record keys from 512 prompts, exactly 512 unique full-fit prompts, zero
   capacity/full-fit prompt overlap, and exact train/eval semantic-record-set
   equality.  Effective train/eval sets must also match.
2. Initial checkpoint cadence included a selection-eligible step-1 evaluation,
   contrary to the frozen every-250 recipe.  Formal selection is now step0,
   every 250 updates, and final only; replacement requires strictly higher EAL,
   so exact ties retain the earlier checkpoint.

Both blockers have adversarial regressions and are closed.

## Verified contracts

- J010 real selection: 512 raw records, 510 effective records, 512 prompts.
  The two `h=0` rows are excluded only from the frozen effective objective;
  raw512 remains in evaluation.  Schedule is 8,000 updates; gate is
  J2 `>=99%`, oracle-gap recovery `>=95%`, harm `<=1%`.
- J011 real selection: 512 prompts, 4,096 raw blocks, 4,045 effective blocks;
  six epochs; J2 `>=90%` AND oracle-gap recovery `>=80%`.
- Full sidecar must pass a semantic GPU replay receipt before training can load
  it: source/target/split/path/count plus Top16, stored-logit, scalar, score, and
  selected-token checks all fail closed.
- Online model forward receives only DFlash hidden, candidate embeddings,
  candidate logits, base LSE, and anchor embedding.  Gold, target teacher
  labels, and offline released-Domino `policy_ids` never enter forward.
- J012's complete JAPD callback contains base vocab GEMM, FP32 Top16 and
  logsumexp, candidate/anchor gather, one global non-causal head call, and one
  per-position argmax.  Released Domino uses the same hidden/projection and its
  eager released correction path.  `complete_p50_ratio <= 1.20` is fail-closed.
- The full sidecar writes only to a job-ID-derived directory; canonical rollout
  tensors remain read-only.

## Verification

- Focused tests: `24 passed`.
- Four M1 Slurm launchers: `bash -n` passed.
- Python compilation/AST and CLI import checks passed.
- Real manifest validators reconstructed capacity `512/510/512` and full-fit
  `4096/4045/512` with zero prompt overlap.

## Authorization boundary

Run the full sidecar and J012 profile.  Only a full-sidecar job with exit zero
and a passing replay receipt may unlock J010/J011.  Do not launch M2 from this
review; M1 outcomes must first be adjudicated against the frozen gates.
