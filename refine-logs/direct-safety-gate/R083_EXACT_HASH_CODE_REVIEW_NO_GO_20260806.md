# R083 Exact-Hash Code Review — NO-GO

**Date:** 2026-08-06  
**Reviewer:** fresh GPT-5.6-Sol xhigh agent  
**Independence:** same-family; provisional  
**Reviewed wrapper:**
`27c079e2b869f9784b0493bc2c9f382587e3c8b54b05f6563bbe051e8d73f01b`

## Verdict

- `exact_hashes_match`: **yes**
- `submission_authorized`: **no**
- `scientific_semantics_verdict`: **NO-GO**
- `publication_and_wrapper_verdict`: **NO-GO**

All ten requested hashes matched. The reviewed source manifest independently
closed exactly 63 sorted current first-party files, with entries digest
`6d9b994e5937dac293e070f95d707d8b1333bc6072e48b71b57e07925ea85f36`.
All checked frozen scientific input hashes also matched. The falsifier remained
unopened and no Slurm job was submitted.

## Blocking findings

1. `scripts/evaluate_direct_safety_falsifier.py::_write_outcome_bundle`
   compared reloaded and original tensor-bearing mappings with ordinary Python
   equality. Distinct multi-element tensors raise `RuntimeError: Boolean value
   of Tensor with more than one value is ambiguous`. The reviewer reproduced
   this in the project environment; the reviewed evaluator would crash after
   materialization and before gate adjudication or READY.
2. Scientific conjunct failures could become operational crashes instead of
   complete published FAILs:
   - nonfinite PROS or comparator scores raised before `_gate_checks`;
   - a nonpositive oracle denominator made all bootstrap recoveries invalid and
     raised;
   - a nonzero regret-bound violation raised before its gate conjunct.
3. The publication consumer did not verify that the retained pending receipt
   and `READY.json` still shared device/inode, link count two, and mode `0400`.
   A byte-identical copied READY with missing pending could be accepted.
4. Published start/end identities covered canonical metadata but not every
   canonical shard, and did not cover the target config/index/embedding shard.
   This left verify-to-load races and made the documented identity chain
   incomplete.

## Nonblocking findings

1. The in-process saved-record replay is a same-helper consistency replay, not
   an independent R084 adjudication. The intended records otherwise contain
   sufficient fields for a fresh independent replay.
2. The contract said reservation preceded frozen-input validation, whereas the
   reviewed evaluator validated/captured frozen inputs before reservation. No
   falsifier content was opened in that inversion, but documentation and code
   order differed.
3. Regression coverage did not exercise the blockers above.

## Required rescue boundary

Wrapper `27c079e2...` is permanently submission-ineligible. A rescue may alter
only the blocked integrity/publication plumbing and corresponding tests; it may
not change any data assignment, model, checkpoint, comparator, score decision,
seed, bootstrap count, metric equation, threshold, or scientific gate. It must
add:

- tensor-aware exact outcome round-trip verification;
- complete READY-published scientific failure packets for nonfinite score,
  invalid-denominator, and regret-violation cases;
- consumer rejection tests for broken pending/READY hardlink and mode;
- canonical-shard and target config/index/embedding start/end identities plus
  mutation tests;
- a new source closure, wrapper hash, full regression, explicit GPFS replay,
  and a second fresh exact-hash review.

Only a new zero-blocker GO may authorize exactly one submission. Exit `0` must
mean fully published scientific PASS, exit `2` fully published scientific
FAIL, and any other exit an operational failure consuming the opening. Every
completed outcome still requires fresh R084 result-to-claim review.
