# Round 5 Review

<details open>
<summary>Full raw same-reviewer terminal response</summary>

CALIBRATION: none

## Anchor / Drift Audit

The immutable Problem Anchor is preserved exactly:

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

**Drift: NONE.**

FBPF still adapts released DFlash internally, preserves the exact deployment topology, uses prospective isolated data, and evaluates the originally anchored EAL/safety/Domino-gap questions.

## Concise Proposal Summary

FBPF adds mergeable last-two-layer LoRA to released DFlash. It retains exact, uncensored D-PACE training, adds a hinge at the adapted model’s current first mismatch, and accepts task updates only when every position in the frozen model’s accepted prefix retains a positive gold margin. Per-position constraints are represented exactly by one blockwise maximum constraint per block, giving at most four batched VJP rows. Deployment removes LoRA wrappers and retains the original draft/verification graph.

## Local Implementation Audit

- The pinned D-PACE checkout is exactly commit `f36bad6e6b0f9f5b59e1e6cf405c705b46d2b43f`.
- Its implementation uses the stated asymmetric smoothing, detached prefix/suffix weights, and batch-size reduction. For complete valid blocks, the proposal’s D-PACE equation matches [dflash.py](/hpc2hdd/home/zwang668/dflash-iclr/third_party/D-PACE/specforge/core/dflash.py:234).
- The local project consistently treats accepted draft tokens as excluding the known anchor and adds one for verification advance, as documented in [candidate_ceiling.py](/hpc2hdd/home/zwang668/dflash-iclr/src/sph/candidate_ceiling.py:26). The frozen direct-safety contract is explicitly length 15 in [direct_safety_gate.py](/hpc2hdd/home/zwang668/dflash-iclr/src/sph/direct_safety_gate.py:24).
- The project environment uses PyTorch `2.9.1+cu128`; `torch.autograd.grad` supports `is_grads_batched=True`.

## Axis Scores

| Axis | Score | Weight | Weighted contribution | Reason |
|---|---:|---:|---:|---|
| Problem Fidelity | 9.8 | 0.15 | 1.470 | Exact anchor preservation; method and validation directly address the frozen-selector bottleneck and deployment constraints. |
| Method Specificity | 9.2 | 0.25 | 2.300 | Equations, ties, projection, restoration, counters, optimizer state, controls, data, gates, and deployment are implementation-level concrete. |
| Contribution Quality | 8.2 | 0.25 | 2.050 | One focused verifier-specific construction with an exact fixed-block implication; novelty remains an application-specific composition of known primitives. |
| Frontier Leverage | 8.5 | 0.15 | 1.275 | Mergeable PEFT and frozen-target supervision are appropriate; no forced extra foundation-model machinery. |
| Feasibility | 8.3 | 0.10 | 0.830 | Four batched VJP rows make the cost gate credible, though actual vectorized-kernel performance remains an explicit pre-science risk. |
| Validation Focus | 8.7 | 0.05 | 0.435 | Strong sealed factorial design, signed power hierarchy, empirical safety gates, data isolation, and restart-level latency equivalence. |
| Venue Readiness | 8.1 | 0.05 | 0.405 | Potentially strong if all contrasts pass, but method novelty remains borderline for a top venue until supported by decisive results. |

**Arithmetic:**  
`1.470 + 2.300 + 2.050 + 1.275 + 0.830 + 0.435 + 0.405 = 8.765`

**WEIGHTED COMPOSITE: 8.77/10**

## Technical Audit Findings

### Blockwise feasibility equivalence

Correct:

\[
C_n=\max_{i\in P_n}c_{n,i}\le\tau_f
\iff
\forall i\in P_n,\ c_{n,i}\le\tau_f.
\]

This changes only the computational representation, not the feasible set. Empty protected sets are correctly vacuous.

### Exact ties and active switches

The uniform average of gradients at exact tied maxima is a valid max subgradient. It does not guarantee that every tied branch decreases under the linear proposal, but the proposal does not rely on that guarantee: every nonlinear candidate reevaluates all per-position constraints. Therefore an unsafe tie direction or active-position switch cannot be committed. It can only cause smaller backtracking, a transactional skip, or restoration failure.

### Batched-VJP feasibility

Reducing from up to 60 VJP rows to at most four makes the `≤4×` median and `≤6×` p95 gates credible. A single `is_grads_batched=True` call still incurs vectorized reverse-mode work and activation memory, so passing is not guaranteed, but the route is no longer obviously infeasible. Complete work accounting and independent clean-process comparisons are appropriate.

### Transactional optimizer

The state semantics are coherent:

- `k_outer` advances with consumed data and controls the LR schedule.
- `t_adam` advances only with accepted task updates and controls bias correction.
- Backtracking commits projected/scaled parameters with one full shadow moment update.
- Task skips preserve parameters and Adam state.
- Restoration changes parameters only and leaves moments and `t_adam` untouched.
- Exact nonlinear checks guard every committed constrained update.

### D-PACE and causal controls

The A/B/C/D design is minimal and correctly factored:

- A isolates ordinary D-PACE LoRA.
- B tests a frozen/static frontier under the same feasibility mechanism.
- C tests the dynamic frontier without feasibility.
- D tests the full method.

Thus D–A tests the complete mechanism, D–B tests dynamic versus static repair, and D–C tests prefix feasibility. The signed superiority/non-inferiority rules align with those claims.

### Latency equivalence

Twenty independent process restarts, restart-level paired log ratios, alternating order, and TOST with a 90% CI inside ±2% form a defensible equivalence design. The trace audit independently verifies that any apparent latency equivalence is not hiding a changed runtime graph.

## Strongest Strengths

1. The proposal has one dominant contribution and no inference-time module pile-up.
2. It explicitly absorbs the local hard-support negative result instead of renaming or retrying it.
3. The blockwise-max representation preserves exact safety semantics while making training cost plausible.
4. All optimizer failure modes are separated from scientific falsification.
5. The one-opening factorial design can reject individual dynamic/constraint claims without expanding the method afterward.
6. Data isolation, checkpoint selection, deployment equivalence, and latency inference are unusually well preregistered.

## Remaining Issues

### Fatal issues

**NONE.**

### Major issues

1. **Primary estimand ambiguity.** The proposal defines mean harm “over all blocks,” then specifies blocks→prompt→component aggregation. It also averages prompts within a component before cluster resampling, which produces a component-balanced estimate if component means are subsequently equally weighted. That may differ from the historical prompt-balanced EAL and invalidate direct Domino-gap recovery comparisons.

   **Executable fix:** Freeze one estimand before experiment planning. Recommended:

   - average blocks equally within each prompt;
   - compute point estimates as the equal-weight mean across prompts;
   - resample connected components as clusters, carrying all constituent prompt metrics into each replicate;
   - never globally average raw blocks.

   If component-balanced inference is intentionally preferred, relabel every metric accordingly and recompute all historical comparison quantities under the same estimand.

2. **Top-venue novelty remains result-dependent.** FBPF is now a clean verifier-specific construction, but its primitives are standard. The proposal cannot establish venue-level contribution quality through further prose refinement; D must decisively separate from A, B, and C at the preregistered effect sizes.

   **Executable fix:** No added method component. Preserve the current claims and let the sealed factorial gate determine whether the paper-level mechanism survives.

### Minor issues

1. Clarify that the exact-tie mask and non-gold winner indices are detached before constructing batched VJP masks.
2. In the A/D throughput benchmark, A must execute only the work required by actual D-PACE training. Do not add an otherwise-unused frozen-base branch to A merely to reduce the reported overhead ratio. D must include every required reference, VJP, projection, restoration, and exact-forward cost.
3. Specify whether p95 ratio means the ratio of paired p95 step times or the p95 of paired per-step ratios; use the same definition across all three process pairs.
4. The gradient-norm ratio thresholds `.05` and `20` are fixed engineering gates, not theoretically derived balance guarantees; retain that wording.
5. Record allocated and reserved CUDA peaks, even though only allocated memory currently has a numeric pass threshold.

## Executable Pre-Experiment Fixes

Before any job or data generation:

1. Freeze the prompt-balanced versus component-balanced estimand and make every EAL/harm/power/bootstrap statement consistent.
2. Freeze the detached exact-tie/non-gold mask construction used by the four-row batched VJP.
3. Ensure the throughput baseline contains no artificial reference work.
4. Pin the p95-ratio computation and process-pair aggregation.
5. Add unit tests showing:

   - blockwise maximum feasibility is equivalent to all-position feasibility;
   - averaged tie subgradients cannot bypass exact candidate checks;
   - active-position switching causes rejection/backtracking when unsafe;
   - restoration and task skips preserve Adam state exactly;
   - accepted task steps increment each counter exactly as specified.

## Final Verdict

The proposal has no remaining fatal method flaw and is sufficiently concrete for execution planning. However, under the mandatory canonical weighting, the composite is `8.77 < 9.0`; therefore the formal verdict cannot be READY.

**Verdict: REVISE**

**proceed_to_experiment_plan: yes**

Proceed only after the estimand and throughput-baseline clarifications above are frozen in the experiment plan. The remaining venue-readiness risk must be resolved empirically, not by adding further method complexity.

</details>
