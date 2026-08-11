# AGENTS.md

## Pipeline Status

- language: zh
- active_workstream: parc16-parallel-global-single-sequence-head-v4
- authoritative_contract: refine-logs/parallel-global-head-v1/USER_CONSTRAINT_CONTRACT.md

## Immutable User Requirements

All agents and reviewers working in this repository must read the authoritative contract before proposing, editing, reviewing, or launching experiments.

The active method must:

1. consume the complete 16-position DFlash block at once;
2. use global **non-causal** mixing so every position sees all 16 positions;
3. predict all 16 token decisions in one parallel forward pass;
4. emit exactly one 16-token sequence;
5. use Top-16 only as an internal per-position candidate axis;
6. never use causal/autoregressive token feedback, Domino GRU rollout, serial target decoding, iterative refinement, beam, tree, trie, forest, or multi-path verification;
7. use no target-model inference beyond offline supervision and the ordinary final verifier;
8. remain lightweight relative to the DFlash draft model and preserve a credible path to at least 1.15x Domino SGLang throughput;
9. reach at least 1.15x same-job Domino fixed and dynamic EAL on disjoint held-out data;
10. prioritize the main accepted-length mechanism before secondary engineering work.
11. never use capacity fitting, same-set replay, or 512/2K training as an efficacy gate; the first scientific run must be meaningful-scale full16 training with prompt-disjoint validation and held-out evaluation against same-job Domino.

A violation of any item is a hard **NO-GO**, regardless of prior artifacts or reviewer recommendations. R050–R056 target-seed/tree/forest work is off-spec evidence only and cannot authorize the active design.

## Required Execution Order

1. Re-read the authoritative contract.
2. Verify proposal compliance before implementation.
3. Verify tensor/dataflow compliance before code review.
4. Use local unit/shape/gradient checks only as fail-fast implementation safeguards; do not submit a standalone GPU smoke or capacity-training stage.
5. Run the real full16 training dataset with prompt-disjoint train/validation/held-out partitions; select only on validation and compare same-job Domino on the same prompts.
6. Require genuine held-out EAL improvement before system polishing.
7. Compare eager-to-eager first; integrate and compare same-stack SGLang only after the mechanism passes.
