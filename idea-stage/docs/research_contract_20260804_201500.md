# Research Contract: Safe Global Full-Lattice Reranking

**Frozen on:** 2026-08-04  
**Source proposal:** `refine-logs/FINAL_PROPOSAL.md`  
**Scope:** Qwen3-4B / released DFlash-b16 / greedy temperature 0 / block size 16 / K16

## Problem

The released parallel drafter exposes a complete candidate lattice, but current per-position top-1 selection and the project's previous candidate-prepooled GCLS recover only a small part of the candidate oracle gap. The research question is whether a compact, frozen-feature, one-pass reranker can use cross-position candidate evidence without shortening the already-correct DFlash prefix or erasing DFlash's latency advantage.

## Primary Claims

### C1 — Full-lattice global interaction

A no-prepool, candidate-specific global compatibility mixer improves accepted draft length beyond parameter-matched local/causal controls and the existing axial candidate-summary mixer.

Required evidence:

- same K16 lattice and verifier for all systems;
- matched local/causal/global architecture, initialization, data, seed, and selection rule;
- global−local and global−causal prompt-cluster bootstrap confidence intervals exclude zero;
- flat compatibility beats axial pooling; otherwise C1 is deleted.

### C2 — Reach-aligned, base-preserving training

Unsmoothed candidate-support accepted-reach training with a block-balanced base-prefix margin regularizer improves the repair/harm trade-off over the project's historical smoothed Candidate-D-PACE α=.5 setup.

Required evidence:

- raw prompt-balanced EAL, first-miss repair, harm, first-token accuracy and worst-domain delta;
- fixed `m=.1`, development comparison only over `lambda={0,.1,.25}`;
- untouched-test harm has one-sided 95% cluster-bootstrap upper bound ≤5%;
- if reach or the regularizer is not selected by the frozen development rule, the corresponding claim is deleted.

## Anti-Claims and Boundaries

The work does not claim the first accepted-prefix objective, self-attention, candidate lattice, multiplicative interaction, or margin regularizer. Accepted reach is gradient-equivalent to length-normalized Candidate-D-PACE α=0 under identical detached support and is verified by a unit test, not counted as an independent method. The work does not use tree verification, sequential GRU correction, joint target/draft training, online teachers, RL, or sampling-temperature guarantees.

K16 oracle is an availability reference, not an attainable selector ceiling. A low high-capacity probe is not evidence that frozen features contain no information. KEEP_BASE is a separately calibrated deployment variant and cannot replace the raw primary result.

## Frozen Method

- Inputs: frozen DFlash hidden, K16 IDs/logits, full-vocab logsumexp, anchor ID, frozen target embeddings.
- Compact model: flat compatibility nodes, D128/H8/L2, FF4, dropout0, 1,235,808 parameters; each block includes a zero-initialized trainable per-head relative-position table and a trainable same-position group bias initialized to `log L`, shared identically across scope controls.
- Scores: DFlash log-prob plus zero-initialized candidate residual.
- Objective: float32 accepted-reach utility with gold-not-in-K censoring.
- Regularizer: block-balanced hinge on the contiguous DFlash rank-1 accepted prefix, margin .1.
- Trainable components: selector only; no vocabulary table.
- Primary deployment: raw greedy argmax; exact tie retains candidate0.

## Data and Leakage Contract

- Development train: nested prompt subsets from the 100K Open-PerfectBlend canonical collection.
- Development selection: Phase-3 `validation_select`; no claim-grade inference.
- Capacity probes: same-subset artifacts labeled `capacity_probe`, never held-out evidence.
- Confirmation: prompt-disjoint train, calibration and untouched test; final test opened once after method, λ, checkpoint rule, and KEEP_BASE rule freeze.
- Unit of statistical resampling: prompt, not block.
- Ground truth: stored target continuation IDs; no model output is used as evaluation truth.

## Success and Stop Rules

The project proceeds to system integration only if the final compact raw model:

1. materially exceeds the current development best (+0.285 raw EAL versus DFlash);
2. beats axial/local/causal controls;
3. shows three-seed consistent direction and cluster CI excluding zero;
4. improves repair without unacceptable first-token/domain loss;
5. later satisfies independent-test harm UCB≤5%.

A positive offline result without positive end-to-end TPS supports only an offline selector claim. Failed capacity, objective, representation, scope, or latency gates trigger the deletion/stop rules in `EXPERIMENT_PLAN.md`; they do not authorize adding unplanned rescue modules.
