# Round 2 Review

<details open>
<summary>Full raw reviewer response</summary>

CALIBRATION: none

## Seven Dimension Scores

1. **Problem Fidelity: 9/10**

   The immutable anchor is preserved verbatim and operationally. FBAC still modifies released DFlash internally, retains one parallel draft forward and one target verification, uses prospective data, and avoids selector/tree/causal-head drift. The descriptive Domino-gap recovery calculation closes the earlier anchor gap without making an invalid architectural attribution.

2. **Method Specificity: 6/10**

   The revision resolves most loss-semantic blockers: `L=15` is consistently separated from the known anchor, full-correct and frozen-first-token-break cases are defined, masks are detached and refreshed each forward, ties are deterministic, and the D-PACE smoothing/product/suffix-sum construction structurally matches the official detached-weight objective. [D-PACE equations and algorithm](https://arxiv.org/html/2605.18810)

   - **Specific weakness:** The optimizer is not yet executable without consequential interpretation. `B` in `1/B` is undefined and conflicts with block-size notation; the validity mask affects CE but not prefix products or suffix weights; the parity test refers to an undefined `eta`. The vacuous-constraint case still reaches `gradient(H)` in the pseudocode. `AdamW_proposal` does not specify candidate moment construction, decoupled weight decay, rejected-step moment behavior, or restoration-state behavior. The static and no-constraint arms are named but not given exact objective equations.
   - **Concrete method-level fix:** Define minibatch size as `N`, train only complete 15-position blocks or propagate validity through `p_i` and `w_i`, and replace the undefined `eta=0` test with a named D-PACE-only mode. Add explicit branches for `P_B=∅`, feasible, and infeasible batches. Give equations for all four trained arms and precise optimizer-state transition pseudocode, including which moments are proposed, committed, or left unchanged.
   - **Priority:** CRITICAL

3. **Contribution Quality: 6/10**

   This is now a real mechanism rather than merely renamed weighting: full D-PACE remains intact, the hard breaker supplies an additional task direction, and base-prefix margins define an explicit update-feasibility set. The local proposition is correct and properly scoped to fixed evaluated blocks. The disclosed hard-censoring failure also materially distinguishes FBAC from the failed reachable-support route.

   - **Specific weakness:** The novelty remains a composition of known pieces: D-PACE, current-hard-example repair, base-relative safe fine-tuning, projected optimization, and mergeable LoRA. D-PACE already adapts weights to example-specific acceptance bottlenecks; “use the exact hard breaker” is still modest on its own. The contribution is credible only if the constrained active-set formulation is presented as one derived lexicographic objective, not as D-PACE plus a hinge plus a generic projection.
   - **Concrete method-level fix:** Derive FBAC from a single objective: maximize the current greedy-breaker margin subject to retaining frozen accepted-prefix margins, with full D-PACE as the non-censored coverage term. State exactly which part is application-specific—the construction of the active repair and feasible sets from speculative-prefix semantics—and avoid claiming generic projected optimization or representation adaptation as novelty.
   - **Priority:** CRITICAL

4. **Frontier Leverage: 8/10**

   Mergeable LoRA is the appropriate foundation-model-era carrier under the deployment constraint, and retaining the frozen target as an offline teacher is natural. No LLM critic, RL loop, diffusion component, or sequential expert would improve the paper’s focus. The revision correctly avoids forcing trendier but mismatched machinery.

5. **Feasibility: 6/10**

   The PEFT, data, and merge paths are feasible, but the proposed batch projection may stall or behave differently from the stated algorithm.

   - **Specific weakness:** Projecting against the gradient of only the single worst residual does not handle multiple active or tied constraints. Exact backtracking cannot rescue a direction that points outside another zero-slack constraint for every positive step. On an infeasible new minibatch, halving a restoration step is also directionally inappropriate: if the full restoration step does not reach feasibility, smaller steps generally will not. Sharing AdamW moments between task and restoration gradients can contaminate subsequent task updates, while committing moments after a projected/scaled parameter step is underspecified. Per-minibatch feasibility also does not preserve earlier minibatches, although the proposal appropriately avoids claiming unseen-data safety.
   - **Concrete method-level fix:** At a feasible point, collect all tied or near-active constraints and project the uncommitted AdamW task proposal onto their linearized half-space intersection with a small dual QP or deterministic sequential projection, then exact-backtrack. For an infeasible minibatch, use a bounded stateless restoration loop minimizing maximum violation; do not mutate task moments. Commit task moments exactly once only after an accepted task step, and leave them unchanged on skip/restoration. State explicitly that feasibility is batch-local and empirically audited, not persistent across prior blocks.
   - **Priority:** CRITICAL

6. **Validation Focus: 7/10**

   The five sealed outcomes—released baseline plus four trained arms—are the minimal sufficient causal design if both dynamic-breaker and constraint claims are retained. They identify overall gain, dynamic versus static repair, constraint versus no constraint, and gain beyond ordinary D-PACE LoRA in one opening. Equal tuning budgets, feasible-first selection, component-level grouping, and hash-frozen checkpoints are strong improvements.

   Remaining precision issues prevent a higher score: the power calculation covers DFlash and D-PACE comparisons but not dynamic-versus-static EAL or constraint-versus-no-constraint harm; the exact harm statistic in the latter contrast is unspecified; a no-feasible-checkpoint arm has no deterministic falsifier checkpoint; and using “prior producer-train paired SD” conflicts with the statement that old outcome/model-score artifacts are never read. These should be fixed without adding arms.

7. **Venue Readiness: 6/10**

   The proposal is now focused and falsifiable, and a strong factorial result would be meaningful. It is not yet top-venue ready because the central projected optimizer is underdefined and the novelty over ordinary constrained fine-tuning remains modest.

   - **Specific weakness:** As written, reviewers could summarize FBAC as “D-PACE LoRA with a first-error hinge and trust-region projection.” The fixed-block proposition is correct but elementary and does not itself raise the contribution to a top-venue mechanism.
   - **Concrete method-level fix:** Make the final algorithm mathematically closed, frame the contribution narrowly as speculative-prefix-derived active-set constrained adaptation, and delete broader “faulty representation” claims unless representation-level evidence is actually produced. Do not add models or modules before the prospective mechanism gate passes.
   - **Priority:** IMPORTANT

WEIGHTED COMPOSITE: 6.80/10

GAP: The revision genuinely resolves the earlier anchor, hard-censoring, edge-case, causal-control, and one-shot-isolation blockers. It also turns PROTECT from a soft penalty into an intended feasibility condition. The remaining 2.20-point gap to READY is concentrated in the heart of the method: the projected AdamW procedure is not yet mathematically coherent for multiple active constraints, infeasible new batches, vacuous batches, and optimizer-state commitment. Until that algorithm is closed, feasibility and novelty cannot be judged independently—failure could reflect either the research hypothesis or an ill-defined optimizer. Even after repair, the paper must demonstrate that speculative-prefix construction produces behavior beyond ordinary constrained D-PACE fine-tuning.

## Simplification Opportunities

1. Remove the regression-breaker branch from full FBAC: whenever its batch constraint is satisfied, `m_theta < m_0` is impossible by the stated proposition. Handle infeasible batches solely through restoration; retain the regression branch only where needed for the unconstrained control.
2. Use only complete 15-position training blocks and delete `v_i` unless partial blocks are genuinely required. This removes ambiguity in D-PACE products, suffix sums, and normalization.
3. Delete “internal representation repair” from the core claim. The directly established mechanism concerns output-margin decision boundaries; LoRA merging is only the parameterization and deployment vehicle.

## Modernization Opportunities

NONE

## Drift Warning

NONE

## Remaining Action Items

1. **CRITICAL — Close the projected optimizer mathematics.** Handle all tied/near-active constraints, define exact restoration behavior, add the vacuous branch, and specify proposal/moment/weight-decay/commit semantics.

2. **CRITICAL — Make all objective equations executable.** Resolve `B` versus minibatch-size notation, validity handling, the undefined `eta`, reduction conventions, numeric tolerances, and exact A/B/C/D arm definitions.

3. **CRITICAL — Separate algorithm failure from hypothesis failure.** Unit-test the linearized projection on synthetic multi-constraint cases, exact nonlinear backtracking, infeasible-batch restoration, rejected-step state invariance, and repeated batch changes before interpreting the capacity gate.

4. **IMPORTANT — Complete the sealed power contract.** Power every confirmatory Claim-2 contrast, including dynamic-versus-static EAL and constraint-versus-no-constraint harm, using preregistered minimally meaningful effects.

5. **IMPORTANT — Resolve the old-data contradiction.** Either whitelist and hash-freeze one aggregate variance receipt that contains no row-level outcomes, or use a conservative prespecified variance bound. Do not simultaneously claim that no old outcome artifacts are read.

6. **IMPORTANT — Define every falsifier statistic and missing-arm rule.** Give formulas for block-harm rate, mean harm, and the paired harm contrast; specify prompt clustering; and preassign a deterministic checkpoint for any arm with no safety-feasible selection.

7. **IMPORTANT — Tighten deployment equivalence.** Compare released versus merged operator/kernel traces directly and use a paired confidence interval or equivalence test for the ±2% latency claim, not only a sample median.

8. **MINOR — Keep all five sealed outcomes.** They are minimal for the two-component causal claim; delete an arm only if the corresponding dynamic or constraint claim is also deleted.

Verdict: REVISE

</details>

Assurance: same saved GPT-5.6-Sol/xhigh reviewer; positive judgments remain provisional.

