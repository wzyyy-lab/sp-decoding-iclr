# Problem Anchor: Beat Domino on accepted prefix length

## Bottom line

On Qwen3-4B, greedy decoding, block size 16, and exact single-chain target
verification, produce a method whose prompt-balanced accepted draft length is
strictly greater than the released Domino checkpoint's same-anchor baseline of
7.015792 on the phase-3 validation-select split, then confirm the gain on a
clean held-out or prospective split.  The optimization target is accepted
length itself; 7.5 is the first practical target, not a ceiling.

## Must solve

The frozen parallel draft alone leaves substantial candidate quality unused.
Domino recovers part of it through a causal correction head, but its fixed
correction and teacher-forced training do not fully address rollout exposure,
target-conditioning bandwidth, or domain/position-dependent calibration.

## Non-goals

- Do not return to frozen candidate-selector variants already falsified by the
  GCLS/FMAS/SAVS/CAMRS/PROS experiments.
- Do not spend research cycles on hashes or publication-style provenance
  closure.  Keep only checks that prevent semantic context mismatch or leakage.
- Do not use oracle/gold access at inference, and do not change exact target
  verification.

## Constraints and freedoms

- The target model remains frozen and exact verification preserves its output
  distribution.
- The proposal remains a single chain for the primary Domino comparison.
- Extra causal-head capacity, a second lightweight pass, adaptive fusion, and
  on-policy fine-tuning are allowed.  Latency is measured, but accepted length
  is the primary objective.
- Model selection uses validation-select; final evidence must use data not used
  to choose the method or hyperparameters.

## Success criteria

1. Reproduce released Domino near 7.015792 under the same-anchor metric.
2. Exceed it on validation-select with a meaningful paired gain.
3. Confirm `EAL > Domino` on validation-gate or a new prospective split; the
   preferred evidentiary bar is a positive prompt-clustered 95% interval.
4. Report per-domain behavior and throughput, while ranking methods by EAL.
