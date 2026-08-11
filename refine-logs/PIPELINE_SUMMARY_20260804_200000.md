# Research-Refine Pipeline Summary

**Anchored problem:** recover more real accepted draft tokens from the already computed DFlash K16 lattice without tree verification, sequential correction, or target adaptation.

**Selected method:** base-anchored global full-lattice compatibility reranker trained with float32 candidate-support accepted reach and a block-balanced base-prefix regularizer.

**Why the old result was weak:** remote candidate hypotheses were collapsed before global interaction; candidate/context interactions were mainly additive; the training support extended far beyond current greedy reach; and corrections were not balanced against damage to the frozen base prefix.

**Artifacts:**

- Method: `refine-logs/FINAL_PROPOSAL.md`
- Claims/data contract: `idea-stage/docs/research_contract.md`
- Plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Review history: `refine-logs/round-1-review.md` through `round-3-review.md`
- Compute contract: `.aris/compute/env-spec.json`, `.aris/compute/slurm.md`

**Immediate execution order:** R001 GPU smoke → R010–R012 128-block capacity → R020–R023 objective/safety screen. Representation and scope experiments are intentionally blocked until the preceding choice is frozen.

**Current status:** implementation complete locally; 78/78 tests pass; experiment-code review and GPU witness pending.

