# PGCF-16 G4 Pre-submit Code Review

**Verdict:** SUBMIT GO  
**Review scope:** matched 20k global/local training plus the disjoint Gate2
evaluator  
**Review independence:** same-family provisional secondary Codex reviewer

No blocking finding remained after review.  The reviewer confirmed:

- full16 global non-causal one-call, one-chain semantics remain immutable;
- global/local training differs only in attention visibility and output path;
- the fixed `31*j, j=0..511` train diagnostic is report-only and never enters
  checkpoint selection;
- checkpoint selection remains maximum validation-select prompt-balanced EAL,
  with the earlier checkpoint retained on exact ties;
- the remote diagnostic preserves the recipient anchor and one complete
  `(hidden, candidate IDs, base logits)` triplet, replaces the other 15
  positions with one coherent label-independent donor, and retains only the
  preserved position from each view;
- donor cells are same-domain and same-context-quartile, with no self or
  same-prompt mapping;
- pure base-Top16 oracle, 10,000-draw paired prompt bootstrap, per-domain
  comparisons, raw erasure, and the matched-local negative control are all
  fail-closed;
- Gate2 requires explicit global/local checkpoint paths.

Verification at review time: 34 focused tests, Python compilation, both Slurm
syntax checks, and diff whitespace checks passed.  One additional seeded
global/local initialization-equivalence regression was added afterward; the
final focused suite passed 35 tests without changing the reviewed runtime
implementation.

The review authorized array job `10166898` only.  Gate2 remains dependency
blocked until both array tasks publish their selected checkpoints.
