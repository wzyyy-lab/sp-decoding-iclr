# Round 2 External Review

**Score:** 8.9/10  
**Verdict:** REFINE

The core method, leakage boundary, controls, novelty scope, and holdout policy
were judged sound.  Two protocol details remained blocking:

1. At zero residual-projection initialization, the first backward pass cannot
   produce upstream backbone gradients.  Gate 0 must require a nonzero output
   projection gradient at identity and an upstream gradient only after one
   optimizer step.
2. Formal seed aggregation and harm inference were not defined, and the online
   causal control omitted Direct-one-edit.  The final estimand, per-seed rule,
   harm UCB, and five rollout paths must be frozen explicitly.

CPU implementation was authorized in new FMAS-only files.  Capacity is
authorized after Gate 0 and a pinned subset manifest; development remains
conditional on the exact D64 capacity pass.

