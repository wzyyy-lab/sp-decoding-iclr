# R082 Fit / Checkpoint Code and Launch Review — Initial NO-GO

**Date:** 2026-08-05  
**Workflow:** ARIS `experiment-bridge` fresh code/launch-contract review  
**Reviewer:** fresh `gpt-5.6-sol`, `ultra`, read-only  
**Verdict:** **NO-GO**

The exact wrapper SHA-256
`f6d158873cafa5836a646f0cf0f6149bbea1389264da3c93acf47916febbd141`
was not authorized and was never submitted. The reviewer matched all supplied
hashes, independently replayed the 61-file source closure, passed the 56-test
focused suite and the then-current full suite, and opened no real outcome
records or later-stage data.

## Blocking findings

1. The production partition is `198 × 64 + 14`, but the trainer asserted a
   final batch size of 62 and would deterministically crash after the first
   pass.
2. Non-strict replay still raised for a nonpositive binary-oracle denominator,
   preventing the required recorded-ineligible / atomic scientific-FAIL path.
3. End identities copied the start identities for the source manifest and two
   receipts instead of rehashing them.
4. Directory publication used an existence check followed by `os.rename`,
   which was not race-safe no-clobber.

## Bounded remediation

- Derive and test the exact 12,686/64 production partition, including final
  batch 14.
- In non-strict replay, represent invalid-denominator recovery as `null` and
  make the checkpoint ineligible; preserve strict fail-closed behavior.
- Rehash every authorized input at end and test receipt mutation detection.
- Publish with Linux `renameat2(RENAME_NOREPLACE)` and test a destination race.
- Re-freeze source, wrapper, and test hashes and obtain a fresh re-review.

No GPU launch, retry, falsifier, validation, reserved, formal, threshold,
width, or seed change was authorized by this review.
