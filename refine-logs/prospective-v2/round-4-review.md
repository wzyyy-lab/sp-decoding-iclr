# Round 4 Review

<details open>
<summary>Full raw reviewer response</summary>

CALIBRATION: none

## Seven Dimension Scores

1. **Problem Fidelity: 9/10**

   The anchor remains verbatim and operationally intact. The method still modifies released DFlash internally, retains the one-parallel-forward/one-verification graph, uses prospective isolated data, and treats Domino only as descriptive gap accounting. The narrowed claim—verifier-induced prefix feasibility plus current-breaker repair—matches the original bottleneck better than the earlier representation-level language.

2. **Method Specificity: 9/10**

   The algorithm is now mathematically and implementationally closed. The half-space update has the correct sign; every row’s residual is recomputed after prior cyclic updates; restoration uses the correct slack residual `c+tau_r+Gd`; nonlinear backtracking verifies the actual constraints; vacuous, feasible, infeasible, skip, and abort branches are distinct; and task state is transactional.

   The dual counters are also coherent: `k_outer` controls equal data consumption and learning-rate scheduling, while `t_adam` controls moment bias correction and advances only after an accepted task update. Committing full shadow moments after a projected/scaled parameter step is an unusual but explicit optimizer definition, not an ambiguity.

   Complete-block D-PACE is unambiguous and retains the official smoothed, detached suffix-weight construction. [D-PACE method](https://arxiv.org/html/2605.18810) The pinned commit and numeric parity tolerances make the matched baseline auditable.

3. **Contribution Quality: 7/10**

   The revision now presents one focused contribution: first-mismatch verification induces a protected positive-margin set and a current repair frontier, while uncensored D-PACE remains the coverage objective. Removing the half-base-margin rule and “unique derivation” rhetoric makes this substantially cleaner.

   Novelty remains borderline rather than exceptional: positive-margin constraints, current-hard-example repair, projected optimization, and LoRA are known primitives. The defensible novelty is their verifier-specific set construction and exact fixed-block consequence. The factorial controls are sufficient to determine whether that construction adds value beyond ordinary D-PACE and generic constrained fine-tuning.

   One terminology issue remains: the nonlinear feasible set and its affine linearized displacement constraints are not mathematically a cone because they contain nonzero offsets. “Verifier-induced prefix-feasible adaptation” or “sign-feasible adaptation” would be more precise than “sign-cone adaptation.”

4. **Frontier Leverage: 8/10**

   Mergeable PEFT is the correct modern carrier, and the frozen target is used only for existing supervision/context. No critic, RL loop, replay system, expert, or inference-time component is warranted. The proposal is modern without decorative complexity.

5. **Feasibility: 5/10**

   The API is available in the stated local PyTorch version, and the raw Jacobian storage calculation is correct: `60 × 1,835,008 × 4` bytes is approximately 0.41 GiB. The blocking issue is runtime, not API existence or output-tensor storage.

   - **Specific weakness:** `jacrev(..., chunk_size=4)` computes the 60-row reverse-mode Jacobian in roughly 15 vectorized VJP chunks. It does not reduce the derivative work to four baseline backwards. FBAC additionally requires the task backward, frozen reference work, cyclic projection, and nonlinear candidate forwards. Therefore a median D/A wall-time ratio `≤4×` appears unlikely for the proposed complete `K×P` Jacobian. Activation/vmap memory, rather than the 0.41-GiB materialized Jacobian alone, may also dominate. The fail-closed throughput gate cleanly distinguishes engineering failure from scientific failure, but it does not make the current route practically plausible.
   - **Concrete method-level fix:** Replace the `K≤60` projector outputs with one mathematically equivalent blockwise constraint per non-vacuous block:

     `C_n(theta)=max_{i in P_n} c_{n,i}(theta) <= tau_f`.

     This preserves exactly the same feasible set and fixed-block property while reducing the Jacobian to at most `N=4` rows. Use deterministic max/tie semantics, and retain the full per-position vector for exact nonlinear acceptance. If a different position becomes worst after the proposed step, backtracking or the next restoration cycle detects it. Keep the `≤4×`, memory, and no-failure gates unchanged; do not loosen them or introduce sampled/top-K constraints.
   - **Priority:** CRITICAL

6. **Validation Focus: 8/10**

   The five outcomes remain the minimal sufficient design for the two component claims. All confirmatory contrasts, first-token non-inferiority, and absolute harm precision enter the power contract. The old-data exception emits only conservative variance/ICC information. Missing feasible checkpoints fail dependent claims deterministically. Component-cluster inference, independent-restart latency TOST, exact graph traces, and one sealed opening are all strong.

   One statistical sentence remains ambiguous: “first average blocks within each prompt connected-component” should explicitly mean blocks→prompt, prompts→component, then component-cluster resampling. Otherwise components containing prompts with different block counts could inadvertently become block-weighted.

7. **Venue Readiness: 7/10**

   Conditional on practical optimization and decisive sealed contrasts, this is now a plausible focused top-venue method: one verifier-specific construction, no deployment overhead, a negative-result-informed design, and a minimal causal test. It is not READY because the current full-Jacobian implementation likely violates its own complexity budget, and the title overstates the geometry as a cone.

WEIGHTED COMPOSITE: 7.80/10

GAP: The proposal is now technically precise, anchor-faithful, causally testable, and substantially free of contribution sprawl. The remaining 1.20-point gap to READY is concentrated in one issue: the proposed complete 60-row reverse-mode Jacobian is mathematically clean but probably incompatible with the preregistered `≤4×` training-cost gate. The gate appropriately prevents optimizer failure from being misreported as scientific evidence, yet a proposal with an obviously likely gate failure is not implementation-ready. An exactly equivalent four-row blockwise-max constraint would preserve the core claim while making the optimizer commensurate with the stated compute budget.

## Simplification Opportunities

1. Replace all per-position projector rows with one blockwise maximum constraint per block. This preserves the exact feasible set while cutting Jacobian outputs from at most 60 to at most 4.
2. Rename “sign-cone” to “prefix-feasible” or “sign-feasible”; the feasible displacement region uses affine offsets and is not a cone.
3. Keep all five sealed outcomes. No causal arm is redundant while both dynamic-frontier and prefix-feasibility claims remain.

## Modernization Opportunities

NONE

## Drift Warning

NONE

## Remaining Action Items

1. **CRITICAL — Make projection cost plausible.** Adopt the equivalent blockwise-max constraint or another exact `O(N)`-row formulation; retain exact all-position candidate checks and the existing `≤4×` gate.

2. **CRITICAL — Audit blockwise-max ties.** Specify deterministic worst-position selection and include all exact tied maxima or demonstrate that exact nonlinear acceptance safely handles the chosen subgradient.

3. **IMPORTANT — Rename the method geometry.** “Sign cone” is mathematically inaccurate for affine, offset constraints.

4. **IMPORTANT — Pin the throughput benchmark procedure.** Measure A and D in separate clean processes, reset peak-memory statistics, include Jacobian construction and every feasibility forward, and report both absolute time and the preregistered ratio.

5. **IMPORTANT — Clarify hierarchical metric aggregation.** Define blocks→prompt→connected-component explicitly before component-cluster bootstrap.

6. **IMPORTANT — Match power-test sidedness in the receipt.** Superiority tests may use two-sided alpha `.05`; first-token and EAL non-inferiority should use the same one-sided alpha `.05` rule as their final tests, or retain the more conservative two-sided calculation and state that choice.

7. **MINOR — State task-scaling diagnostics.** Record the initial and training-time norm ratio between D-PACE and breaker gradients; keep the unit coefficient fixed and treat severe domination as a capacity failure rather than adding a post-hoc weight.

Verdict: REVISE

</details>

Assurance: same saved GPT-5.6-Sol/xhigh reviewer; positive judgments remain provisional.

