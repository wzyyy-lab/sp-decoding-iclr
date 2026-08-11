# Round 3 Review

<details open>
<summary>Full raw reviewer response</summary>

CALIBRATION: none

## Seven Dimension Scores

1. **Problem Fidelity: 9/10**

   The immutable anchor remains preserved. The revision still attacks released DFlash’s internal decision boundary, retains the single-chain/one-draft-forward/one-verification contract, uses prospective data, and treats Domino only as descriptive gap accounting. Batch-local feasibility and held-out empirical safety are now clearly separated from unseen-data guarantees.

2. **Method Specificity: 8/10**

   The method is now substantially implementation-ready. `L=15`, complete-block reductions, all A/B/C/D objectives, the vacuous branch, stateless restoration, shadow Adam state, zero weight decay, nonlinear acceptance, and failure semantics are explicit. The D-PACE formula matches the official asymmetric-smoothed, detached suffix-sum construction up to the declared minibatch averaging. [D-PACE method](https://arxiv.org/html/2605.18810)

   The half-space correction has the correct sign. For linear residual `u=c+gᵀd>0`, the update `d←d-u g/(‖g‖²+ε)` reduces the same residual to approximately `uε/(‖g‖²+ε)`. Multiple constraints are handled by cyclic sequential projection, and exact nonlinear backtracking prevents accepting a merely linearized-feasible step.

   Two details still require correction:

   - The per-constraint condition appears stale: residuals are computed before sorting, but earlier projections change later residuals. Immediately before each update, recompute `u_j(d)=c_j+g_jᵀd` and project only if that current value exceeds `tau_l`.
   - Adam bias correction needs an accepted-update counter, while equal data/schedule budgets need an outer-minibatch counter. Define `t_adam` for moments and `k_outer` for `lr_k`; skipped task proposals advance `k_outer` but not `t_adam`. Otherwise “identical steps/schedules” across constrained and ordinary arms is ambiguous.

3. **Contribution Quality: 6/10**

   The contribution is now focused and intellectually coherent: full D-PACE supplies uncensored coverage, speculative first-mismatch semantics constructs the repair frontier and protected prefix, and lexicographic feasibility gives an exact fixed-block implication. This is no longer a renamed hard-support mask.

   - **Specific weakness:** The construction remains close to generic base-relative constrained fine-tuning applied to a first-error hard example. D-PACE already adapts learning toward example-specific acceptance bottlenecks; sequential projection, margins, LoRA, and lexicographic constrained optimization are established primitives. Moreover, the claim that the feasible set is “uniquely” induced by verification is too strong: `0.5*gamma_base`, `epsilon_tie`, and the unit coefficient on `L_dynamic` are design choices, not consequences of first-mismatch semantics.
   - **Concrete method-level fix:** Use the minimal verifier-derived constraint `gamma_i(z)≥epsilon_tie` for every frozen-accepted prefix position unless preserving half the frozen margin is separately justified as an empirical design choice. Present only the sign-constrained active/feasible-set construction as derived; describe the margin fraction and task scaling honestly as fixed implementation choices. Keep the current factorial controls—do not add another module or arm.
   - **Priority:** CRITICAL

4. **Frontier Leverage: 8/10**

   Mergeable LoRA and frozen-target supervision are the natural modern primitives under the deployment constraint. The revision correctly excludes replay, learned critics, RL, extra experts, and inference-time machinery. No forced modernization would improve the paper.

5. **Feasibility: 6/10**

   The optimizer is now mathematically auditable, but its practical cost and convergence remain blocking uncertainties.

   - **Specific weakness:** Computing a separate 1.835M-parameter gradient for every protected position can require up to roughly `15N` constraint-gradient evaluations per task step, followed by multiple cyclic sweeps and nonlinear forwards. The stated 40–100 A800 GPU-hour training estimate may be substantially low without a vectorized per-sample Jacobian implementation. Four projection sweeps are a budget, not a proof of infeasibility; therefore `NO_LINEAR_SOLUTION` is also an inaccurate failure name. The numerical implementation accepts `c≤tau_f`, whereas the formal optimization states `c≤0`.
   - **Concrete method-level fix:** Specify batched/vectorized constraint-gradient computation and add a preregistered throughput/memory gate relative to ordinary D-PACE. Rename failure to `PROJECTION_BUDGET_EXHAUSTED`. State the implemented feasible set as `c≤tau_f` and note that `epsilon_tie>tau_f` still implies strictly positive protected margins. If vectorized all-constraint projection cannot meet the compute gate, close the route rather than introducing top-K constraints.
   - **Priority:** CRITICAL

6. **Validation Focus: 8/10**

   The five outcomes are minimal and sufficient for the two component claims. The receipt exception is narrow, component isolation is replayable, all Claim-2 contrasts now have power targets, missing feasible checkpoints fail claims deterministically, safety statistics are explicit, prompt components—not blocks—are the resampling unit, and latency uses the correct 90% TOST interval for an alpha-.05 equivalence test.

   Remaining details are modest but should be frozen before collection:

   - Include the Claim-1 first-token non-inferiority margin in `n_power`; it is also a confirmatory pass condition.
   - Clarify that power refers to rejecting the null at the stated true effect, not an 80% probability of also satisfying the point-estimate threshold at that boundary.
   - Latency pairs from one continuous run are autocorrelated. Form inference units from preregistered timing batches or independent process restarts, then apply TOST to batch-level paired log ratios.

7. **Venue Readiness: 6/10**

   If the optimizer passes and all factorial effects meet their preregistered margins, this could become a sharp empirical-method paper. It is not yet venue-ready because the novelty remains application-specific and the training mechanism may be too costly relative to the small deployed change.

   - **Specific weakness:** A reviewer can still accurately summarize the current method as “D-PACE LoRA plus current-error repair under base-prefix margin constraints.” The fixed-block proposition is correct but elementary, so venue strength must come from a clean verifier-derived formulation, practical optimizer, and decisive factorial separation.
   - **Concrete method-level fix:** Remove “uniquely derived” and any residual representation-causal language; state one contribution only: verifier-induced sign constraints and current-breaker repair for parallel drafter adaptation. Demonstrate in the capacity stage that its vectorized optimizer remains a practical training intervention before treating the formulation as paper-ready.
   - **Priority:** IMPORTANT

WEIGHTED COMPOSITE: 7.35/10

GAP: The revision resolves the former mathematical blockers around projection sign, multiple constraints, restoration monotonicity, vacuous batches, transactional task moments, complete-block D-PACE, control definitions, missing checkpoints, power receipts, safety metrics, and latency equivalence. The remaining 1.65-point gap to READY is not missing exposition; it is the combination of modest novelty and potentially severe all-constraint gradient cost. The proposal must distinguish the verifier-derived sign-constrained construction from arbitrary base-relative safe fine-tuning, correct the remaining residual/counter semantics, and show that exact sequential projection is computationally viable without truncating the constraint set.

## Simplification Opportunities

1. Replace `r=max(epsilon_tie,0.5*gamma_base)` with the minimal positive-margin floor unless the half-margin retention rule is independently justified. Only positivity is required by the proposition.
2. Rename `NO_LINEAR_SOLUTION` to `PROJECTION_BUDGET_EXHAUSTED`; the feasible linear system can exist even when four cyclic sweeps do not reach tolerance.
3. Keep the five sealed outcomes. They are already minimal; deleting B or C requires deleting the corresponding dynamic-frontier or constraint claim.

## Modernization Opportunities

NONE

## Drift Warning

NONE

## Remaining Action Items

1. **CRITICAL — Fix sequential-projection execution semantics.** Recompute each constraint’s current residual immediately before its projection update; use sweep-start residual only for stable ordering.

2. **CRITICAL — Separate optimizer and experiment counters.** Use an outer-minibatch/schedule counter and a committed-Adam-update counter so skips preserve moments without changing equal data budgets.

3. **CRITICAL — Resolve projection feasibility cost.** Specify vectorized per-constraint gradients and preregister throughput, peak-memory, and slowdown gates versus D-PACE before scientific capacity interpretation.

4. **IMPORTANT — Align formal and numerical feasibility.** State `c≤tau_f` as the implemented constraint and explicitly derive `gamma≥epsilon_tie-tau_f>0`.

5. **IMPORTANT — Freeze exact restoration algebra.** For slack target `c+gᵀd≤-tau_r`, write the implemented residual as `c+tau_r+gᵀd`; recompute it sequentially just as in task projection.

6. **IMPORTANT — Narrow the novelty statement.** Only the active/feasible-set construction comes from first-mismatch semantics; the base-margin fraction, hinge scale, projection algorithm, and LoRA are design choices.

7. **IMPORTANT — Complete confirmatory power.** Add first-token non-inferiority and define whether absolute harm gates receive precision targets.

8. **IMPORTANT — Make latency inference robust to autocorrelation.** Apply TOST to independent restart- or timing-batch-level paired log ratios rather than treating 1,000 adjacent iterations as independent.

9. **MINOR — Pin parity tolerances numerically.** Replace “frozen CPU/GPU tolerances” with exact scalar and per-gradient absolute/relative limits and a code/version hash.

Verdict: REVISE

</details>

Assurance: same saved GPT-5.6-Sol/xhigh reviewer; positive judgments remain provisional.

