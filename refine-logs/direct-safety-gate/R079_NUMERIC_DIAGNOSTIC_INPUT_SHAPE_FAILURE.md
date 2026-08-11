# R079 Numeric Diagnostic Input-Shape Failure

**Date:** 2026-08-05  
**Job:** `10137369`  
**Disposition:** failed closed before synthetic scan or model forward; no retry
is currently authorized

## Evidence

Slurm reports `FAILED 1:0`, elapsed 19 seconds.  Stdout is empty (SHA-256
`e3b0c442...b855`); stderr contains only a 12-line traceback (SHA-256
`d0dfc4dd...2fff`).  The first selected record stopped in the reviewed
allowlist extractor:

```text
RuntimeError: canonical candidate IDs do not have shape [15,16]
```

No aggregate JSON was printed.  No file below `artifacts/pros_gate` was
created or modified by the job.

## Localized cause

The frozen canonical metadata declares `draft_positions=15` and `top_k=64`.
The production materializer's reviewed collator accepts the stored K and
selects the Direct-native lattice with `[:, :candidate_k]`, where
`candidate_k=16`.  The diagnostic extractor instead required the unsliced raw
ID and logit tensors to already have shape `[15,16]`.  It therefore failed on
valid frozen storage before copying a numeric batch.

The failure occurred before target/model loading, the synthetic numeric scan,
any Direct forward, or any aggregate comparison.  It neither accessed gold
semantics nor computed outcomes, gains, capacity, falsifier, validation,
reserved, or formal evidence.

## Closed boundary

The reviewed job was consumed and cannot be resubmitted.  A candidate repair
must first receive a fresh failure-rescue verdict.  The anticipated narrow
contract is to require raw ID/logit shapes `[15,K]` with the same `K>=16`, then
copy exactly the first 16 candidates and retain typed `[15,16]` tensors.  Tests
must cover real K=64 slicing, K<16 rejection, mismatched ID/logit K rejection,
and selected-prefix ordering.  Any repair changes the diagnostic source,
tests, 58-file closure, wrapper pins/hash, and therefore requires a new code
review before a reviewer may consider exactly one retry.  Nothing downstream
is open.
