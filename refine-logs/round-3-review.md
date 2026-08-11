# Round 3 Method Review

## Parsed result

- Overall score: **8.05 / 10**
- Rubric verdict: **REVISE**
- Experiment decision: **GO after a short implementation-contract preflight**
- Anchor: preserved
- Interpretation: there is no conceptual blocker for smoke/capacity experiments; the remaining top-venue risk is empirical novelty and end-to-end latency.

<details>
<summary>Full raw reviewer response</summary>

CALIBRATION: anchored  
Weighted composite: **8.05/10**  
Review independence: **same-family**  
Acceptance status: **provisional**  
Rubric verdict: **REVISE**  
Experiment decision: **GO after a short implementation-contract preflight**

## Scorecard

| Dimension | Score | Assessment |
|---|---:|---|
| Problem Fidelity | 10 | The proposal remains exactly on the frozen-DFlash, one-pass, greedy reranking problem. Conditional LoRA and sequential drafting are cleanly excluded. |
| Method Specificity | 8 | The architecture, loss, safety regularizer, dimensions, masks, candidate ordering, and split protocol are substantially frozen. A few numerical and feature-definition details remain. |
| Contribution Quality | 7 | This is now one focused contribution: safe global full-lattice residual reranking. The individual primitives are familiar, so publication strength will depend heavily on the coupled empirical result. |
| Frontier Leverage | 8 | Full-lattice attention over frozen foundation-model representations is a natural modern solution. The proposal correctly avoids unnecessary RL, online teachers, and latent compression. |
| Feasibility | 8 | The 240-node compact model and gated probe are practical. The main uncertainty is latency, which is handled by an appropriate measured gate. |
| Validation Focus | 9 | Objective, representation, scope, safety, capacity, and latency are isolated with strong deletion rules and independent splits. |
| Venue Readiness | 7 | Claim ownership is now honest and defensible, but top-venue novelty still requires substantial global/full-lattice interaction gains and positive end-to-end throughput. |

## Gap and mathematical consistency

There is no remaining conceptual error that should block the smoke or capacity experiments. The mathematical contract is consistent, the method is focused, and the positive-only probe interpretation is sound. The remaining gap is primarily empirical: attention, unsmoothed D-PACE-style reach weighting, and a margin regularizer are individually established primitives. The paper becomes compelling only if their coupling produces a material, statistically stable global gain that simpler axial/local/causal selectors cannot obtain, while satisfying the harm and latency constraints.

For valid positions,

\[
L_{\mathrm{reach}}=1-\frac{1}{BL}\sum_b\sum_t S_{b,t},\qquad
S_{b,t}=\prod_{j\le t}q_{b,j},
\]

implies

\[
\frac{\partial L_{\mathrm{reach}}}{\partial \ell_{b,i}}
=\frac{1}{BL}\sum_{t\ge i}S_{b,t},\qquad
\ell_{b,i}=-\log q_{b,i}.
\]

Gold-not-in-K censoring is correct: setting availability to zero makes the current and all subsequent survival terms zero. Safe gathering avoids evaluating an invalid index or explicitly computing `log(0)`. Length-normalized Candidate-D-PACE at `alpha=0`, with detached suffix weights and identical support, has exactly the same score gradients; it should remain a unit-test equivalence rather than an experimental condition.

The block-balanced base-prefix regularizer is coherent: it protects only the original contiguous rank-one prefix, leaves the first miss repairable, maps an empty prefix to zero, and first averages within blocks. It is a regularizer rather than a harm guarantee; the independent 5% upper-confidence requirement remains the safety criterion.

## Contribution assessment

The focused contribution is:

> A base-anchored, globally informed K-way residual selector that preserves the complete parallel candidate lattice, trains with the unsmoothed accepted-reach gradient on candidate support, and regularizes against shortening the frozen base prefix.

The paper must not claim that expected reach, self-attention, multiplicative compatibility, or margin protection is independently novel. The publishable claim is their coupled design and demonstrated necessity in frozen parallel drafting. At 240 nodes, `D=128`, two layers, and no online teacher, no pre-latency slot compression is needed.

## Required preflight before jobs

1. Freeze exact scalar-feature definitions, signs, scaling, clipping, and the top-K entropy convention.
2. Freeze the 128-block diagnostic architecture, objective, `lambda`, optimizer, and budget.
3. Compute `log_softmax`, gold probabilities, prefix products, utility, and parity checks in float32.
4. Assert candidate zero is the exact released-DFlash greedy action and compare the complete epoch-zero path/scores with the frozen baseline.
5. Define scope invariance by perturbation: disallowed future nodes cannot affect causal outputs and other positions cannot affect local outputs.
6. Make the positive probe reuse the development-selected objective and `lambda`; do not conflate capacity with objective choice.

These are implementation-contract fixes, not reasons for another proposal round.

## Capacity-probe and novelty limitations

- A high held-out probe gain is positive evidence for the tested frozen inputs and function class; a low result is only an engineering stop signal.
- The probe is not a Bayes ceiling and the 70% student-recovery bar is an engineering criterion.
- Full-lattice global attention must beat parameter-matched causal/local and single-mean controls.
- Positive EAL without positive end-to-end throughput supports a selector result but weakens the decoding-system claim.
- KEEP_BASE remains a separately labeled calibrated variant.
- If compatibility/full-lattice is deleted by ablation, objective adaptation alone is unlikely to be a strong top-venue contribution.

## Bottom line

Round 2 is methodologically coherent and ready for implementation once the short preflight is fixed. The rubric remains `REVISE` because `READY` requires demonstrated novelty and evidence, not because another module or redesign is needed. Do not delay the 128-block smoke and capacity program for additional paper ideation.

</details>
