# R079 Outcomes Numeric-Portability Failure

**Date:** 2026-08-05  
**Job:** `10135890_[0-1]`  
**Disposition:** both tasks failed closed before atomic bundle publication; no
fit/checkpoint bundle, audit receipt, or temporary directory remains

## Evidence

Both tasks passed exact source closure, canonical metadata, split manifest,
split GO receipt, Direct checkpoint/metrics, and exclusion identity checks.
They loaded the frozen producer and computed their assigned outcomes. Bundle
validation then stopped publication:

- task 0 (`fit`): `saved position feature is inconsistent`; 6/15 values
  differed, maximum absolute difference `5.960464477539063e-08`;
- task 1 (`checkpoint`): `saved rank feature is inconsistent`.

Slurm recorded both as `FAILED 1:0` after 40 and 32 seconds. The writer creates
a hidden temporary directory only after record validation, so no partial or
completed-looking output was exposed.

## Localized cause

The feature producer computes rank as CUDA float32 `direct_path / 15` and
position as CUDA float32 `arange(15) / 14`, then clones the resulting feature
matrix to CPU. Both the primary artifact validator and independent auditor
recompute these two formulas on CPU and require bitwise equality. Division may
differ by one float32 ULP across CPU and CUDA even though the mathematical
feature is identical. Other recomputed continuous scalars already use bounded
absolute tolerances; discrete path/change identities remain exact.

## Current boundary

The fresh reviewer selected an exact adjacent-float32 semantic rule, not a
general absolute tolerance and not persistence-time canonicalization. For rank
and position only, the primary validator and independent auditor accept the
CPU theoretical float32 value or its immediate `nextafter` predecessor or
successor. Expected endpoints 0 and 1 remain bitwise exact, and actual values
must stay in `[0,1]`. Two-ULP, `1e-4`, and endpoint mutations fail closed.

The two implementations share no helper. Existing discrete and continuous
checks are unchanged. Focused tests pass `62` with one CUDA-only skip; the full
CPU suite passes `329`, one skip, and three parameterized subtests.

Because this changes two first-party Python files, the old split receipt cannot
authorize repaired code. The old closure was preserved verbatim as
`R079_SOURCE_CLOSURE_20260805_PRE_NUMERIC_RESCUE.json` (SHA-256
`513ad34d...05a53`). The repaired 57-file closure is
`8e62d261...da3b4`; all wrappers use a separate
`artifacts/pros_gate/r079_numeric_rescue` root and updated pins. A fresh bounded
implementation review must return GO before the identity-only split/audit chain
is re-established. Falsifier, capacity, training, evaluation,
validation/formal data, and alternate resource changes remain closed.

## Second fail-closed observation

After the adjacent-float implementation, new closure/split/audit chain, and a
real A40 rank/position smoke all passed, the separately reviewed new-root
outcomes job `10136583_[0-1]` exposed a second cross-device reconstruction
boundary. Both tasks passed every preflight and then failed before publication
on retained mass:

```text
tanh((logsumexp(candidate_logits) - base_logsumexp) / 2)
```

Each first failing record differed in one of 15 positions by exactly
`1.9073486328125e-06`, exceeding the existing `1e-6` absolute tolerance.
Task 0/1 exited `1:0` after 35/26 seconds. No bundle or temporary directory
exists. This does not invalidate the adjacent-float fix; it proves the prior
single-field CUDA smoke was insufficient to certify all continuous reductions.

The workflow is stopped again with no retry authorization. A different fresh
reviewer is now designing a systematic, operation-aware cross-device policy
and deciding whether a bounded synthetic stress diagnostic or a
fit/checkpoint-input-only CUDA diagnostic is required before any further code
change. No general tolerance widening will be applied from this single first
mismatch.
