# Round-1 External Method Review

- Reviewer: fresh same-family research reviewer
- Calibration: none
- Overall score: **8.2 / 10**
- Verdict: **REFINE**
- Evidence checks passed:
  - the `984 / 90,120 / 173,271` beneficial/harmful/neutral action
    counts reproduce the FMAS Gate-2 evidence;
  - `GlobalDirectCandidateSelector.residual_projection` is exactly zero
    initialized, and `rho[i,r] - rho[i,0]` cancels the per-position residual
    gauge;
  - all 512 capacity records come from the training split, with no
    `validation_gate` or formal-test leakage.

## Scorecard

| Dimension | Score | Assessment |
|---|---:|---|
| Problem anchor | 9.5 | The proposal stays within frozen DFlash, a fixed-depth one-edit chain, and zero target calls. |
| Formula/statistical consistency | 8.8 | The action counts, EAL geometry, and conditional-mean argument are correct. |
| Repair of flat-CE utility mismatch | 8.0 | Signed labels repair target semantics, but finite-model ERM and max-policy behavior remain distinct. |
| Residual-only parameterization | 9.1 | The difference removes the residual gauge and does not misuse DFlash score gaps as utility. |
| Sparse-positive/max-over-225 risk | 6.2 | The risk is named but not yet quantified tightly enough. |
| Identity/gradient semantics | 9.2 | KEEP identity and the two-step zero-head gradient behavior are correct. |
| Capacity gate | 7.3 | Joint behavior thresholds are useful, but RMSE has no standalone policy-safety meaning and the failure scope is too broad. |
| Data leakage discipline | 9.3 | Physical split isolation is sound. |
| Controls/claim boundary | 8.2 | Controls are adequate; development selection and gating must remain routing evidence only. |
| Novelty plausibility | 6.3 | The particular counterfactual lattice construction may be new, but payoff/value-guided selection is not. |

## Strongest counterexamples

1. **Action-average RMSE does not control the max policy.** One `+0.30`
   normalized prediction error and 224 exact predictions give RMSE `0.02`,
   yet the outlier can win the argmax with a fictitious 4.5-token advantage.
2. **The positive supervision is extremely sparse.** On development, positive
   and negative absolute utility mass differ by about 266x. Unweighted MSE can
   learn a safe all-negative predictor before it learns useful repairs.
3. **Frozen-feature collisions remain possible.** Identical visible features
   with opposite utilities force conditional averaging. Memorizing 512 records
   cannot establish held-out information sufficiency.

## Blocking revisions

1. State only the population-risk result: with complete inference features,
   unlimited capacity, and population MSE optimization, each prediction is a
   conditional mean signed utility and positive argmax is Fisher-consistent
   for expected EAL. Do not claim finite-model RMSE controls policy regret.
2. Freeze explicit max-policy diagnostics: no-benefit false-edit rate,
   selected-action harm, edit selective precision, selected-action regret,
   and per-sign loss/initial-gradient contributions.
3. Scope a Gate-1 failure to the exact D64 parameterization, unweighted MSE,
   optimizer/schedule, subset composition, and checkpoint rule. It cannot
   distinguish objective, capacity, optimization, or information causes.
4. Explain that RMSE `<=0.02` is an engineering fidelity threshold, while
   oracle-gap recovery and selected-action harm are the actual decision gates.
   The 512-record manifest has exactly 256 positive actions, so `>=0.99`
   positive sign recall permits at most two misses.
5. Bound novelty against SpecDec++, Hybrid Verified Decoding, and BASTION.
   The potential distinction is the complete 225-action counterfactual signed
   prefix-advantage target plus exact identity residual-difference deployment
   on a frozen DFlash lattice—not generic value regression.

## Verdict

The method is coherent and the identity/data contracts are sound, but it is
not READY until finite-model max-selection risk and claim scope are made
explicit.
