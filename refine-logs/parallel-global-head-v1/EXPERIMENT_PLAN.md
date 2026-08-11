# PGCF-16 Experiment Plan

Canonical plan: `EXPERIMENT_PLAN_20260810_114625.md`

Immediate sequence:

1. PGCF-001 isolated implementation + CPU compliance tests;
2. fresh read-only code review;
3. PGCF-002 A40 mechanics smoke;
4. PGCF-003 512-block capacity and PGCF-005 fair eager profile;
5. only after those pass, PGCF-006/007 disjoint global-vs-local signal;
6. no 199.8K collection/training before all early gates pass.

All method details inherit `FINAL_PROPOSAL.md`, `round-2-refinement.md`, and the immutable user contract.
