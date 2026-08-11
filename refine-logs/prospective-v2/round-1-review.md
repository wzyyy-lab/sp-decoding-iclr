# Round 1 Review

<details open>
<summary>Full raw reviewer response</summary>

CALIBRATION: none

## Seven Dimension Scores

1. **Problem Fidelity: 8/10**

   The proposal directly attacks the frozen-representation bottleneck inside released DFlash while preserving the anchored single-chain, one-draft-forward, one-target-verification deployment contract. Avoiding Domino-style causal heads and DFlare-style fusion changes is disciplined. The remaining fidelity gap is that success is operationalized mainly as `ΔEAL ≥ +0.30` versus DFlash, whereas the anchor also asks to explain or resolve the Domino gap. The final analysis should report the descriptive fraction of the existing DFlash-to-Domino-parallel gap recovered, without claiming architectural causality.

2. **Method Specificity: 6/10**

   The architecture, frozen/trainable split, data flow, and deployment path are unusually concrete for an early proposal, but the central loss is not yet implementation-complete.

   - **Specific weakness:** The prose promises a stricter regression-breaker branch, but the displayed `L_repair` has only one formula and never uses the frozen margin at the breaker. For `m_theta=L+1`, all positions implicitly become PROTECT and REPAIR vanishes, but this edge case is not stated. `argmax`, the first-break mask, tie handling, detachment, and mask-refresh cadence are unspecified. `L_coverage` is called “D-PACE/teacher-forced block loss” without an exact equation. Finally, a scalar `lambda_protect` does not justify “only allow small degradation”: it permits repair to trade away protected margins.
   - **Concrete method-level fix:** Provide executable pseudocode and explicit cases. Compute `m=stopgrad(first_mismatch(argmax(z),y))` every forward pass with deterministic ties. Use `max_{v≠y}z_i(v)` rather than a selected `a_i` in the hinge. For a regression breaker, require `gamma_m(z) ≥ max(mu_reg, gamma_m(b)-delta_reg)`; for a base-wrong breaker require `gamma_m(z) ≥ mu_fix`; for `m=L+1`, protect all positions and omit repair. Define the exact coverage equation and whether its weights are detached. Most importantly, formulate prefix protection as an active constraint, optimized by a single training-only primal-dual or projected update, rather than an ordinary weighted penalty.
   - **Priority:** CRITICAL

3. **Contribution Quality: 5/10**

   The proposal is admirably focused, but the claimed novelty presently reads as a hard, one-hot version of dynamic position weighting plus safe fine-tuning. D-PACE already derives weights from an accepted-length surrogate and moves training signal toward positions currently limiting acceptance; it also evaluates prefix-reachability masking. Therefore, “the supervised position moves as the model improves” is not by itself a sufficient novelty boundary. [D-PACE primary paper](https://arxiv.org/abs/2605.18810)

   - **Specific weakness:** Dynamic first-break selection, LoRA, suffix-value weighting, margin preservation, curriculum progression, and hard-example mining are individually established ideas. The current scalar combination does not create a qualitatively new optimization mechanism or a defensible safe-policy-improvement guarantee.
   - **Concrete method-level fix:** Make the contribution **first-break active-set constrained adaptation**: repair the currently active greedy breaker subject to explicit base-relative prefix-margin constraints, using one stated constrained optimizer. Derive the active set from hard greedy EAL semantics and explain why D-PACE’s smooth all-position weighting does not enforce these constraints. If the method remains a weighted hinge sum, narrow the claim to an empirical loss heuristic and do not call it safe policy improvement.
   - **Priority:** CRITICAL

4. **Frontier Leverage: 8/10**

   Mergeable PEFT and a frozen target teacher are appropriate modern primitives for the anchored deployment constraint. No extra LLM critic, RL loop, diffusion module, or inference-time search is warranted. Domino’s causal refinement addresses dependency modeling through an added causal head, while this proposal deliberately targets same-graph adaptation; forcing that route would change the intervention and likely the latency contract. [Domino primary paper](https://arxiv.org/abs/2605.29707)

5. **Feasibility: 7/10**

   Last-two-layer LoRA, frozen reference logits, prospective target-generated labels, and post-training merging are implementable within the stated budget. The main uncertainties—whether two layers have enough capacity and whether moving support oscillates—are genuine empirical risks rather than implementation blockers. Deployment neutrality is plausible, provided merged weights retain the identical module/operator graph.

6. **Validation Focus: 6/10**

   The proposal has a commendably small validation program and a genuine one-shot falsifier, but its current control design cannot establish the stated mechanism causally.

   - **Specific weakness:** FBSA versus D-PACE-objective LoRA compares an entire objective package, not dynamic first-break support specifically. The static-position and no-PROTECT arms are confined to fit/checkpoint data, so their evidence can be selected or overfit. Exact prompt hashes do not exclude source-, conversation-, or semantic near-duplicates. “First-token non-inferiority” and “clearly better than D-PACE” lack margins and CI rules. Checkpoint ranking by EAL first and harm second can select an unsafe model.
   - **Concrete method-level fix:** Freeze all causal arms before one falsifier opening and materialize them together: released DFlash, matched D-PACE-objective LoRA, a static-support version with the same base-relative constraint, dynamic FBSA without protection, and full FBSA. Use identical tuning budget and feasible-first checkpoint selection: discard models failing preregistered harm/first-token constraints, then maximize checkpoint EAL. Split by source document/conversation, add cross-split near-duplicate detection, define all non-inferiority margins and paired-CI decisions, and size the falsifier from a preregistered power calculation.
   - **Priority:** IMPORTANT

7. **Venue Readiness: 5/10**

   The proposal has a strong problem narrative and disciplined scope, but its present method claim is too close to dynamic weighting/curriculum and safe PEFT for a top-venue paper. A positive pilot would currently show that a particular fine-tuning loss works, not yet that FBSA is a distinct mechanism.

   - **Specific weakness:** “Representation becomes safely repairable” is stronger than the method establishes; the training signal acts only on output margins, and the safety property is neither constrained nor guaranteed. The novelty story depends heavily on terminology.
   - **Concrete method-level fix:** Reframe around one precise contribution—constrained greedy-prefix active-set adaptation—and state one modest theorem or formal property connecting the active breaker, protected prefix constraints, and hard greedy EAL. Treat LoRA merging and representation language as implementation and interpretation, not parallel contributions. Report only the descriptive fraction of the prior Domino gap recovered until causal evidence supports more.
   - **Priority:** CRITICAL

WEIGHTED COMPOSITE: 6.40/10

GAP: The proposal is 2.60 weighted points below READY. Its strongest qualities are anchor fidelity, parsimony, feasibility, and deployment discipline. The central gap is mechanism identity: D-PACE already adapts training emphasis as acceptance-limiting positions move, while the present FBSA objective is a discrete first-error mask combined with soft margin penalties. The missing step is to turn base-relative PROTECT from a tunable regularizer into an explicit active-set safety constraint, specify every loss edge case, and evaluate the minimal factorial controls in the same sealed falsifier opening. Without that change, even strong EAL results would support “another effective weighting/curriculum” more readily than a distinct safe adaptation principle.

## Simplification Opportunities

1. Remove EMA teacher-of-self and periodic mask refresh as fallback options from the core proposal. Specify one deterministic per-forward mask rule; if it is unstable, close the route or revise before testing.
2. Collapse PROTECT/REPAIR into one constrained active-set update plus one exactly defined small coverage stabilizer. Do not present three losses as three contributions.
3. Remove the optional top-M KL branch from this proposal. It is a separate post-failure route and currently weakens the one-mechanism story.

## Modernization Opportunities

NONE

## Drift Warning

NONE

## Remaining Action Items

1. **CRITICAL — Rewrite the objective as executable mathematics.** Define regression, base-wrong, and full-correct blocks; stop-gradient semantics; tie handling; mask refresh; term normalization; exact coverage loss; and minibatch aggregation.

2. **CRITICAL — Make safety structural rather than rhetorical.** Replace scalar `lambda_protect` tradeoff with a single active-set constrained optimizer or projected update. Otherwise rename the method as a heuristic and substantially narrow the novelty claim.

3. **CRITICAL — Establish the novelty boundary against D-PACE.** State explicitly that D-PACE already moves weight toward current acceptance bottlenecks; identify the proposed distinction as exact greedy reachability plus enforceable base-relative prefix constraints, not merely dynamic weighting.

4. **IMPORTANT — Repair the causal control and falsifier allocation.** Freeze the matched D-PACE, static-support, dynamic-no-PROTECT, and full-FBSA checkpoints before opening one common falsifier. Equalize hyperparameter-search and checkpoint-selection budgets.

5. **IMPORTANT — Strengthen data isolation.** Freeze timestamped manifests before outcomes, group by source/conversation/document, and exclude semantic near-duplicates against the old manifest and across fit/checkpoint/falsifier—not only exact normalized hashes.

6. **IMPORTANT — Make selection and decision rules safety-first.** Define harm and first-token units, non-inferiority margins, paired-CI rules, minimum FBSA-over-D-PACE effect, and feasible-first checkpoint selection.

7. **IMPORTANT — Certify deployment equivalence.** Require identical module/operator graph, draft/target forward counts, kernel path, dtype, and adapter-versus-merged logits within a preregistered tolerance; run paired latency measurements under fixed warmup and clock conditions.

8. **MINOR — Close the anchor loop to Domino.** Report how much of the pre-existing same-anchor DFlash-to-Domino-parallel EAL gap FBSA recovers, explicitly as descriptive evidence because training-data differences prevent architectural attribution.

Verdict: REVISE

</details>

Assurance: fresh same-family GPT-5.6-Sol/xhigh review; positive judgments remain provisional.

