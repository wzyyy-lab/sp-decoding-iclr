# R079 Numeric-Portability Rescue Code Review

**Date:** 2026-08-05 15:11 +0800  
**Reviewer:** fresh GPT-5.6-Sol, xhigh  
**Verdict:** **GO**, narrowly bounded

## Authorized sequence

1. Submit exactly one new-root identity-only split job.
2. On success, freeze its SHA-256 and run exactly one independent split audit.
3. Stop after the split GO receipt.

No outcomes, GPU, capacity, falsifier, training, or evaluation is authorized.

## Verified repair

- Columns 195/196 accept only exact float32 or one immediate `nextafter`
  neighbor.
- Expected endpoints 0 and 1 remain exact, and actual values remain in
  `[0,1]`.
- Two-step and `1e-4` tampering fail.
- The independent auditor reimplements the rule without importing the primary
  validator.
- Generator, materializer, schema, formulas, dimensions, paths, outcomes, and
  protocol identifiers are unchanged.

New source closure:
`8e62d261f8e61262804ad9e20f0cbd7f44298488b1173e0e8b97bc652f8da3b4`.
The exact old closure is preserved at
`R079_SOURCE_CLOSURE_20260805_PRE_NUMERIC_RESCUE.json`, SHA-256
`513ad34d8a71cd4bb340eaeda2dd8132be311a38f075d2148af2dadf7ef05a53`.
Old artifacts remain under `r079`; all reviewed wrappers target only
`r079_numeric_rescue`.

Wrapper syntax/source preflight passed; the reviewer's independent focused run
passed 50 tests with one CUDA-only skip. Local full regression passed 329 tests,
one skip, and three subtests.

## Required later CUDA gate

After the new split audit and before any outcomes authorization, run only the
single synthetic CUDA regression
`test_actual_cuda_normalized_division_is_accepted_by_both_validators` in a
separately reviewed tiny GPU smoke. It must explicitly require CUDA and report
one pass with no skip, while opening no datasets or outcomes.
