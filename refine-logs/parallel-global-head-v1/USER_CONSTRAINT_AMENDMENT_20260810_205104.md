# User Constraint Amendment：Real Training and Independent Validation

**Effective:** 2026-08-10 20:51:04 +0800

This amendment is authoritative and is incorporated into `USER_CONSTRAINT_CONTRACT.md`.

1. Do not run capacity-set or 512/2K training as an efficacy experiment.
2. Do not submit standalone GPU smoke-training jobs.
3. The first scientific run must be meaningful-scale full16 training.
4. Split prompts before training into disjoint train, validation, and held-out sets.
5. Use validation only for checkpoint selection; open held-out only after selection.
6. Compare DFlash, released Domino, and the new head on the exact same validation and held-out sets.
7. Training-set EAL and same-set fitting are diagnostics, never evidence of generalization.
8. Minimal code/unit/shape checks remain allowed solely to avoid wasting the real run on an implementation crash.
