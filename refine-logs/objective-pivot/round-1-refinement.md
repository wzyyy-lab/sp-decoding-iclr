# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** Improve DFlash's greedy single-chain accepted length with a fixed-depth, draft-side selector while ultimately closing meaningful ground against Domino, without tree verification or a sequential GRU.
- **Must-solve bottleneck:** The current axial GCLS sees useful bidirectional candidate-lattice signal, but repairs only 18.29% of first-miss opportunities and recovers 6.18% of the K16 oracle gap; its Candidate-D-PACE training support includes positions that the selector's own greedy path can no longer reach, and training loss can improve while accepted length collapses.
- **Non-goals:** Do not rescue the closed flat full-lattice C1a/C1b route; do not add CRF/Viterbi, recurrent decoding, target calls, tree verification, target-label leakage, a new representation module, or joint/LoRA training in this route.
- **Constraints:** Frozen released DFlash-b16 and Qwen3-4B target embedding; K16, 15 positions, greedy temperature 0; existing prompt-disjoint OPB train and development artifacts; one objective factor changed at a time; sealed test remains unopened; development compute should fit an A40 debug job.
- **Success condition:** On a matched axial D64/L1 seed-0 OPB-25K screen, a policy-reach objective must beat a newly rerun Candidate-D-PACE control in raw prompt-balanced EAL, not increase harm by more than 1 percentage point, and show that direct loss after the current first breaker is exactly zero. Only then may the objective be confirmed on full OPB-99,356 with three seeds and prompt-cluster intervals.

## Anchor Check

- **Original bottleneck:** Fixed gold-coverage supervision continues after the deployed greedy selector has already broken, while accepted length cannot benefit from those independent suffix predictions.
- **Why the revision still addresses it:** The revision changes only the post-break coefficient inside the exact same Candidate-D-PACE weighted objective.
- **Reviewer suggestions rejected as drift:** None. The requested attribution fixes strengthen the original bottleneck test.
- **Terminology correction:** “Policy reach” in the immutable anchor means reach under the current deployed greedy predictions; the method is not differentiable policy optimization. The proposal now uses “prediction-conditioned reachable-prefix support.”

## Simplicity Check

- **Dominant contribution:** A support-only intervention on Candidate-D-PACE.
- **Components removed or merged:** Uniform Head-AUF and a separately normalized coverage auxiliary are deleted; one `L_lambda` covers control and both treatments.
- **Suggestions rejected as unnecessary complexity:** No RL, soft policy gradient, repair/protection margin, teacher, or new architecture.
- **Why this is smallest adequate:** It adds one Boolean helper, one fixed scalar, and diagnostics; parameters and inference remain unchanged.

## Changes Made

### 1. Isolated support from weighting and scale

- **Reviewer said:** The original comparison also changed D-PACE weights and normalization.
- **Action:** All cells now use identical coverage-derived Candidate-D-PACE weights and fixed `1/(B*L)` normalization. Only the coefficient on post-break coverage positions changes.
- **Impact:** `lambda=1` has a required value-and-gradient parity test against historical `candidate_dpace`.

### 2. Fixed evaluation denominators

- **Reviewer said:** Moving support would make candidate accuracy endogenous.
- **Action:** Loss output separately returns fixed coverage positions and prediction-conditioned training positions. Candidate/hard accuracy always uses fixed coverage; true path metrics remain greedy.
- **Impact:** Existing five-part capacity thresholds remain valid and comparable.

### 3. Strengthened routing gate and claim language

- **Reviewer said:** Any positive epsilon is not meaningful; detached argmax is not policy optimization.
- **Action:** Seed-0 gate requires `>=+0.05` raw EAL over control plus safety constraints. Wording is narrowed to hard support alignment and all confirmation remains development-only until an untouched split.
- **Impact:** A small noisy win cannot trigger six full-data runs or a paper claim.

## Revised Proposal

# Research Proposal: Reach-Censored Candidate-D-PACE for Axial GCLS

## Problem Anchor

- **Bottom-line problem:** Improve DFlash's greedy single-chain accepted length with a fixed-depth, draft-side selector while ultimately closing meaningful ground against Domino, without tree verification or a sequential GRU.
- **Must-solve bottleneck:** The current axial GCLS sees useful bidirectional candidate-lattice signal, but repairs only 18.29% of first-miss opportunities and recovers 6.18% of the K16 oracle gap; its Candidate-D-PACE training support includes positions that the selector's own greedy path can no longer reach, and training loss can improve while accepted length collapses.
- **Non-goals:** Do not rescue the closed flat full-lattice C1a/C1b route; do not add CRF/Viterbi, recurrent decoding, target calls, tree verification, target-label leakage, a new representation module, or joint/LoRA training in this route.
- **Constraints:** Frozen released DFlash-b16 and Qwen3-4B target embedding; K16, 15 positions, greedy temperature 0; existing prompt-disjoint OPB train and development artifacts; one objective factor changed at a time; sealed test remains unopened; development compute should fit an A40 debug job.
- **Success condition:** On a matched axial D64/L1 seed-0 OPB-25K screen, a policy-reach objective must beat a newly rerun Candidate-D-PACE control in raw prompt-balanced EAL, not increase harm by more than 1 percentage point, and show that direct loss after the current first breaker is exactly zero. Only then may the objective be confirmed on full OPB-99,356 with three seeds and prompt-cluster intervals.

## Technical Gap

The global axial selector is already the supported structural baseline: at full data it beats matched local and causal controls over three seeds. Flat full-lattice mixing has now failed its binding route. The new route therefore tests only whether the objective's direct-output support matches longest-prefix verification.

Let `m_cov` be the fixed prefix through which every gold token remains in K16. Candidate-D-PACE assigns detached continuation weights to every position in `m_cov`. If the current selector makes an in-K wrong choice earlier, later positions remain representable but cannot affect accepted length for that forward pass. Flat-compatibility demonstrated the associated divergence: objective improved through epoch 9 while raw EAL fell from an epoch-4 peak `+0.1067` to `-0.1068` and harms rose from 90 to 198 blocks.

More data or parameters are not a clean first response: scaling diminishes after 50K, D128/L2 does not reliably beat D64/L1, and the flat route underperforms axial. The minimal falsifier is a support-only intervention with all other loss semantics held fixed.

## Method Thesis and Contribution Focus

- **Thesis:** Downweighting only the independent Candidate-D-PACE losses after the current greedy selector's first breaker should improve longest-prefix EAL if fixed gold coverage is the causal optimization mismatch.
- **Dominant contribution:** Prediction-conditioned reachable-prefix support alignment for an extra residual selector.
- **Supporting ablation:** A fixed 10% post-break coefficient tests whether hard zeroing is too unstable.
- **Non-contributions:** No policy gradient, new weighting formula, architecture, inference rule, trainable component, distillation, or representation adaptation.

## Proposed Method

### Frozen System

Reuse axial-global D64/H4/L1, additive nodes, K16, frozen DFlash hidden/logits, frozen target embeddings, zero-initialized residual, raw greedy argmax, optimizer, scheduler, checkpoint rule, and exact OPB data. No inference code changes.

### Matched Masks and Weights

For gold support `g_i`, gold rank `r_i`, and detached greedy prediction `r_hat_i`:

```text
m_cov[i] = g[i] and product_{j<i} g[j]
correct[i] = g[i] and (detach(r_hat[i]) == r[i])
reach[0] = true
reach[i] = product_{j<i} correct[j]
m_auf[i] = m_cov[i] and reach[i]
m_suffix[i] = m_cov[i] and not m_auf[i]
```

This is off-by-one intentionally: a first wrong in-K breaker remains in `m_auf`; its suffix enters `m_suffix`. A first out-of-K breaker and its suffix are outside `m_cov`.

Compute `w_i` with the unchanged detached Candidate-D-PACE alpha `0.5` rule on `m_cov`. For the single frozen factor `lambda`:

```text
L_lambda = (1 / (B*L)) * sum_i w_i *
           (m_auf[i] + lambda*m_suffix[i]) * CE_i
```

Cells are exactly:

- `lambda=1.0`: control; must match `candidate_dpace` loss and gradients bitwise/tolerance-tight.
- `lambda=0.0`: hard reachable-prefix support; suffix output gradient exactly zero.
- `lambda=0.1`: same treatment with 10% of control's post-break direct gradient retained.

No term is normalized by its moving active count. Reachable positions are not double-counted. `lambda` will not be tuned after results.

### Metrics Contract

The loss output returns:

- `coverage_positions = m_cov` for fixed candidate/hard accuracy and NLL denominators;
- `training_positions = m_auf` for support diagnostics;
- `post_break_positions = m_suffix`;
- component losses before applying lambda, effective position weights, `m_auf/m_cov`, and post-break counts.

Repair, oracle-gap recovery, harm, first-token accuracy, rank buckets, and EAL remain computed from the true greedy path. The method is detached prediction-conditioned supervised censoring, not differentiable policy learning.

### Tests and Capacity Gate

Unit witnesses must establish:

1. `g=[1,1,1,1]`, correct `[1,0,1,1]` gives `m_auf=[1,1,0,0]`.
2. `g=[1,1,0,1]`, correct `[1,1,0,*]` gives fixed coverage and reachable support `[1,1,0,0]`.
3. `lambda=0` has exactly zero gradient on suffix logits.
4. `lambda=1` matches `candidate_dpace` value and all score gradients.
5. Masks require no gradient; fp32 loss remains finite under bf16 scores.

The 128-block same-subset capacity matrix uses all three lambda cells. Candidate accuracy `>=.99` and hard accuracy `>=.97` are always computed on fixed `m_cov`; repair `>=.95`, oracle gap `>=.95`, harm `<=.01` use greedy paths. All five thresholds must pass for every cell before development. Failure is a code/capacity stop, not authorization to tune lambda.

### Development and Confirmation

Development matrix: exact materialized OPB-25K hash `a3d25eba...fa73`, 199,818 blocks, validation-select, axial D64/H4/L1, batch64, 12 epochs, lr `6e-4`, seed0. Code/data/config hashes and initialization must match.

Choose the better of `lambda=0` and `lambda=0.1` by raw prompt-balanced EAL. Enter confirmation only if all hold:

- winner minus `lambda=1` raw EAL `>= +0.05`;
- harm `<= control + 0.01`;
- first-token accuracy `>= control - 0.001`;
- all artifact/provenance checks pass;
- pure `lambda=0` diagnostic reports zero post-break direct component/gradient by construction.

If the gate fails, close this objective route. Do not add repair/protection, soft reach, or new lambda cells.

If it passes, rerun frozen winner and control on all 99,356 prompts for seeds 0/1/2. Require winner-control positive in every seed and paired prompt-cluster 95% development CI lower bound `>0`. This is still development evidence because the selection prompts are repeatedly observed. A paper-facing positive claim additionally requires an untouched test or newly held-out prompt split, frozen calibration, and integrity audit.

## Failure Modes and Diagnostics

- **Conservative identity trap:** detected by unchanged repair and low reachable/coverage ratio despite fixed-mask hard accuracy.
- **Moving-support instability:** detected by oscillating support ratio/EAL across epochs; only the preregistered 0.1 cell addresses it.
- **Apparent gain from evaluator censoring:** prevented by fixed coverage denominators.
- **Gradient-scale confound:** prevented by common D-PACE weights and `B*L` denominator; lambda1 parity test is binding.
- **Small noisy gain:** filtered by the `+0.05` seed-0 route gate.

## Novelty and Elegance Boundary

This post-hoc hypothesis is independent of the closed flat C1a/C1b route: it returns to the previously supported axial baseline and changes no representation component. Its potential scientific contribution is conditional. A matched support-only effect would show that gold candidate availability and deployed greedy reach are empirically distinct training supports for residual speculative selectors. Without a material, replicated EAL gain, it remains a useful negative diagnostic rather than a paper contribution.

## Claim-Driven Validation

### Claim 1 — Support-only causal effect

- **Experiment:** lambda `1/0/0.1`, same model/data/seed/budget.
- **Decisive metric:** winner-control raw EAL `>=+0.05` under safety constraints.
- **Deletion check:** lambda1 parity; no extra loss/module allowed.

### Claim 2 — Replicated development effect

- **Experiment:** winner/control, full data, three seeds, only after Claim 1 gate.
- **Decisive metric:** every seed positive and paired prompt-cluster CI lower bound >0; repair/harm reported.
- **Boundary:** not paper-confirmatory until untouched evaluation.

## Compute and Handoff

- Unit/capacity: minutes on one A40.
- Three-cell OPB-25K array: expected below one aggregate A40 GPU-hour; each cell must fit 30 minutes.
- Full-data confirmation: only after gate, estimated 3–6 GPU-hours.
- Required implementation: one mask helper, one matched loss mode/CLI scalar, dual-mask metrics, tests, capacity/development aggregators, source-hash provenance.
