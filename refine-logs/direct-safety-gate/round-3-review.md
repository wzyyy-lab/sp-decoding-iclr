# Research-Refine Round 3: Final Review

**Verdict:** `READY`  
**Overall score:** **9.1/10**  
**Authorization:** CPU-only implementation and semantic tests on synthetic
fixtures. Real-data loading, artifact materialization, training/evaluation, and
GPU execution remain closed.

## Raw reviewer response

<details>
<summary>Full verbatim response from the independent reviewer</summary>

## Research-refine Round 3 — Final Review

**Verdict: READY — 9.1/10**

All three Round 2 method blockers are closed. PROS-Gate is now sufficiently specified for CPU-only implementation and semantic testing.

### CALIBRATION

- The proposal is evaluated as a narrow, falsifiable safety-gating contribution—not as a general routing or calibration principle.
- The score reflects implementation readiness and evidentiary discipline, not guaranteed venue acceptance.
- No external reviewer feedback was provided for this round.

### Blocker closure

| Round 2 blocker | Assessment |
|---|---|
| Invalid split counts | Closed: chat `523/66/66`, code `531/67/67`, math `533/67/67`; totals `1587/200/200`. |
| Capacity optimizer/checkpoint ambiguity | Closed: exact batch, passes, update count, optimizer, schedule, seed, ordering, evaluation cadence, tie-breaking, and selected-checkpoint gate semantics are binding. |
| Feature/module/recovery underspecification | Closed: scalar transforms, float32 boundary, margins, layer defaults, zero-change behavior, parameter count, half-up warmup, deterministic order, and unclipped recovery validity are explicit. |

Internal arithmetic is consistent:

- Capacity: `512 / 32 = 16` updates/pass; `320 × 16 = 5120` updates; epoch 0 plus 320 passes gives 321 evaluated checkpoints.
- Warmup: `floor(0.04 × 5120 + 0.5) = 205`.
- Sidecar parameters: `17,424 + 21,250 = 38,674`, or approximately `8.92%` of the frozen producer.
- The weighted unit-hinge objective remains Fisher-consistent for the intended conditional expected-gain decision: its optimal sign follows `E[g | x]`.

### Scorecard

| Dimension | Score |
|---|---:|
| Problem formulation | 9.7 |
| Method soundness and exactness | 9.4 |
| Contribution quality | 8.6 |
| Frontier alignment | 9.0 |
| Feasibility | 9.2 |
| Validation and falsification design | 9.7 |
| Venue readiness | 8.6 |
| **Weighted overall** | **9.1** |

### Strengths

- Producer out-of-sample status is supported by the frozen OPB exclusion metadata and independent Phase 3 accounting.
- The checkpoint/falsifier separation prevents adaptive reuse of the final decision split.
- Capacity is correctly demoted to a plumbing test; it cannot rescue the method claim.
- The selected checkpoint—not an earlier passing or later alternative checkpoint—must satisfy every capacity conjunct.
- Deterministic ordering is version-independent and does not rely on framework shuffle behavior.
- Recovery is mathematically bounded by the pointwise oracle; nonfinite values and numerical bound violations fail rather than being clipped.
- Scope discipline is strong: one frozen producer, one small sidecar, one fixed binary decision, one-shot falsification, no post-falsifier refit.

### Remaining risks, not method blockers

- The central empirical risk is contextual separability: scalar ridge already appears weak, and the sidecar may not recover enough of the oracle gap. The sealed falsifier is the correct adjudicator.
- The direct-vs-base oracle headroom is modest, so variance and latency accounting must remain prominent in the eventual claim.
- CPU implementation review should freeze the remaining library-level conventions—especially ridge standardization/linear-solve behavior, bootstrap RNG semantics, and exact producer-state capture hooks—before artifacts are generated. These are implementation reproducibility checks, not reasons to redesign the method.
- Manifest-only overlap checks must remain distinct from loading formal examples.

### Drift, simplification, and modernization

- **Concept drift:** none. The refinement preserves the original claim: predict when the frozen Direct route is safer than KEEP.
- **Complexity drift:** none material. No calibration stage, auxiliary backbone, extra routing action, or retraining of the producer was added.
- **Further simplification:** not recommended before falsification. Removing the capacity test, scalar comparator, or sealed falsifier would weaken diagnosis.
- **Modernization:** adequate. A detached residual sidecar, deterministic artifact contracts, explicit one-shot falsification, and result-to-claim boundaries match current rigorous empirical practice.

### GAP

The remaining gap is empirical rather than conceptual: it is unknown whether the specified 38,674-parameter sidecar can recover the required fraction of oracle improvement while meeting behavioral and latency gates. The experiment is now designed to answer that question without contaminating the final falsifier.

**Authorization:** proceed with **CPU-only implementation and semantic tests using synthetic fixtures**. This does **not** authorize GPU execution, dataset loading, split/outcome/capacity artifact materialization, training, or evaluation on real examples.

</details>
