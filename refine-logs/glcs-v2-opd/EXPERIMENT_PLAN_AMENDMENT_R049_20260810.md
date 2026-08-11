# R049 Amendment: Multi-Depth Target-Logit Information Probe

**Date:** 2026-08-10  
**Parent:** `EXPERIMENT_PLAN_AMENDMENT_R048B_20260810_052100.md`

R048-B is closed: on the preregistered 64-prompt same-set capacity gate, the
L4 180,224-parameter lens recovered only `144/541 = 26.617%` of the exact
Fast-K64 one-repair reward by step 200, despite zero harmful rewrites.  R049
does not extend that run or sweep its rank, optimizer, or data.

## R049-A: zero-parameter information localization

Run one clean, unsplit target verifier forward per frozen Fast-K64 proposal and
capture post-block, pre-final-norm proposal-path states at target depths
`L={4,8,12,16,24,32,36}`.  Row 0 is the anchor state predicting proposal token
0; row `i` is the state after proposal token `i-1` and predicts token `i`.

For every depth:

1. apply the frozen target final RMSNorm;
2. gather the true tied target LM-head rows for the K64 support;
3. compute FP32-centered candidate scores without trainable parameters;
4. use labels only through the clean target's original first rejection;
5. report both:
   - `R_token`: exact reward recovery when the true frontier is supplied and
     only candidate-token ranking is tested;
   - `R_policy`: exact earliest-one, global-margin recovery after a same-set
     zero-harm threshold sweep.

The full unsplit target remains authoritative.  Correct-frontier repair reward
is measured by an independent clean-cache target rerun; proposal suffix logits
are never used as repaired-path ground truth.

## Controls and hard gates

- Let `epsilon` be the maximum centered-score discrepancy between L36
  gather-dot and authoritative gathered logits.  Rows whose authoritative
  candidate margin is at most `2*epsilon` remain in the oracle denominator but
  are forced to KEEP.  L36 candidate argmax must match on 100% of the remaining
  valid rows.  The raw authoritative target-logit policy
  control must recover 100% of the K64 oracle reward.  Otherwise stop for a
  hook, row-alignment, or numerical-contract bug.
- Only `L<=12` is initially system-feasible.  `L16+` is diagnostic and cannot
  authorize training without a later integrated recompute/throughput proof.
- Both reward-weighted and gain-block recovery must pass the stated gates.  If
  a shallow layer has `R_policy >= 90%`, skip residual training and proceed
  to prompt-disjoint calibration/evaluation with the shallowest passing layer.
- Otherwise, a shallow layer must have `R_token >= 80%` to authorize exactly
  one R049-B residual/gate capacity run.  If all shallow layers are below 80%,
  close the early-target route.

## Conditional R049-B architecture

Only if R049-A authorizes it, use

```text
h' = h + W_up SiLU(W_down RMSNorm(h))
candidate_score = dot(FinalRMSNorm(h'), target_embedding[candidate])
```

with rank 64 (`327,680` residual parameters), zero-initialized `W_up`, and one
shared-down-feature binary frontier/KEEP gate with fewer than 10K additional
parameters and a negative fail-closed bias.  Candidate KL is limited to the
accepted prefix plus first rejection; gate labels are KEEP on accepted rows and
REPAIR on a repairable first rejection; suffix is masked.  The same 64-prompt,
200-step gate still requires zero harm and at least 90% oracle-gain recovery.
No two-repair, global mixer, or depth/rank sweep is authorized.
