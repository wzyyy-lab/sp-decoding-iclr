# Reachable-Support Experiment Code Review

Date: 2026-08-04
Assurance: same-family Codex review; provisional
Final verdict: **GO**

## Round 1 — NO-GO

The reviewer found three blockers before launch:

1. evaluation divided by zero when fixed candidate coverage was empty;
2. the capacity aggregator accepted truthy string verdicts and did not
   independently verify the five frozen checks and thresholds;
3. the development aggregator did not prove that `post_break_weight` was the
   only treatment difference across cells.

Fixes added empty-coverage handling, strict Boolean and recomputed capacity
checks, complete frozen configurations, cross-cell configuration signatures,
and adversarial regression tests.

## Round 2 — NO-GO

The reviewer found two remaining artifact-integrity blockers:

1. missing provenance fields could compare equal as `None == None`;
2. gated development scalars were not independently reconciled with the
   prompt-level examples and could admit non-finite or inconsistent values.

Fixes made all source/data/target hashes and external identity checks
fail-closed. The objective summary now reconstructs prompt-balanced EAL,
domain EAL, harm, and first-token accuracy from all validation examples and
checks finiteness, ranges, path-length bounds, oracle bounds, and scalar
consistency before applying the gate.

## Round 3 — GO

The reviewer confirmed:

- provenance, target manifests, external collection identity, configuration,
  budget, and cross-cell matching fail closed;
- objective gate metrics and bootstrap values share the same independently
  reconstructed prompt-level source;
- 64 tests plus 3 parameterized subtests pass;
- reconstruction against an existing 1,175-block artifact agrees within
  `8.88e-16`;
- all four Slurm scripts pass shell syntax validation.

The binding launch order remains: three-cell 128-block capacity gate first;
OPB-25K development only if all three capacity cells pass.
