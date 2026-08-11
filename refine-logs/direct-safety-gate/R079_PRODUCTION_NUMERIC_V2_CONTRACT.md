# R079 Production Numeric Protocol v2 Contract

**Date:** 2026-08-05  
**Status:** implementation complete; fresh pre-run code review pending  
**Authorization:** no Slurm submission or downstream experiment is authorized by this document

## Why v2 exists

The old outcome validator reconstructed CUDA-produced float32 features on CPU
with a uniform absolute `1e-6` threshold. Rank/position first exposed a
one-neighbor float32 rendering, and retained mass later differed by
`1.9073486328125e-06`. Blindly increasing one threshold was rejected.

The sealed label-blind diagnostic job `10137460` instead scanned 12,686 fit and
1,600 checkpoint inputs plus a preregistered synthetic grid. Its 20-field,
605,839,056-comparison aggregate passed the frozen operation-aware policy with
zero envelope/cap/nonfinite/range violations and rejected 1,343/1,343 negative
mutations. Fresh result review authorized only the production-v2 work below.

## Versioned policy identity

- Policy protocol: `pros-gate-cross-device-numeric-policy-spec-v2`
- Policy ID: `pros-gate-cross-device-numeric-policy-v2`
- Canonical policy SHA-256:
  `cbd80345e7249707931f71b29c65722ec8910263d51b7d649c5dd5c04fc4d4f0`
- Outcome artifact protocol: `pros-gate-direct-outcomes-v2`
- Outcome format version: `2`

The policy spec canonically binds field classes, same-device relation count,
and every numeric constant. `numeric_policy_receipt()` recomputes the digest
on every call and returns a defensive JSON copy. Every outcome record stores
the policy ID and digest; bundle metadata and provenance store the full receipt.
Missing, changed, or internally inconsistent identities fail closed.

## Operation-aware persisted checks

- Direct path, change mask, and change scalar remain bitwise exact.
- State difference, base log probabilities, Direct scores, and three margins
  use float64 reconstruction with two source-scale float32 ULPs and a `2^-14`
  half-width cap.
- Rank/position permit exact or one adjacent interior float32 only; normalized
  endpoints remain exact and values remain in `[0,1]`.
- Entropy uses an independent float64 reference, `[0,1]` range, and `2^-17`
  absolute envelope.
- Retained mass uses `E_lse = 8*ulp32(scale)+2^-20`, monotone interval
  propagation through `tanh`, two outward float32 neighbors, `2^-20` outer
  widening, source-LSE ULP at most `2^-16`, and analytic cap
  `E_lse/2+2^-20+4*2^-23 < 1e-4`.
- Material `1e-4` mutations remain rejected by both implementations.

## Producer same-device invariant

Before any host copy, the materializer independently reconstructs all 15
persisted feature/output relation classes on the producer device and requires
bitwise equality. It covers path, change mask, selected/base state copies,
state difference, base log probabilities, Direct scores, three margins, rank,
position, change scalar, entropy, and retained mass. The provenance witness
records the policy identity, batch count, and exactly `15 * batches` relation
checks. The independent auditor requires those values and the existing
native/hooked/repeat/state-dict witnesses.

## Independent auditor boundary

`scripts/audit_direct_safety_artifacts.py` does not import the production
artifact validator or numeric-policy module. It carries a separate literal
policy spec and separate add/sub, normalized-neighbor, entropy, and retained
implementations. It independently recomputes the canonical digest and
requires exact equality with record, metadata, provenance, and native-witness
bindings.

## New chain identities

- Artifact root: `artifacts/pros_gate/r079_numeric_v2`
- Production source closure: 59 files
- Closure SHA-256:
  `2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4`
- Closure entries SHA-256:
  `cc4a9ecea4c1f9a32e0ceae0e3c5551759e51be141bc4a9b37b6a5d03b88d02a`
- Materializer SHA-256:
  `28545c2edd2c3e8b4a404b50f7c5f6fedbf2348a33349106d2d7feb538d853f4`
- Primary artifact contract SHA-256:
  `852915273e330344c1afe472f9d0c2b4789b564e6636af897051486fbaa3a6d9`
- Numeric-policy implementation SHA-256:
  `3b2f0fbab3a0a683a7f5e142b88f79c96960b6427e48253861c090e85768437c`
- Independent auditor SHA-256:
  `102a87a5ea9dbfedaf7fc6429fac79023bfd7e7ebbee062e332f4ad4ad5a863d`
- Source-closure verifier SHA-256:
  `9fac5193a09c3def52b8e1360c0790d2497597f4daa8f5b67a16df5bbb1a31d4`

Reviewed wrapper candidates:

- split: `ac30d701faf6070f2dbfa6e46bcd27b1d2e989e0981fd24aea21e94ead2b4f86`
- independent audit: `5b19d0bfe423651c99da27172d6f061ce4af0a50bd7be7630950329f9117b8f7`
- numeric-v2 CUDA smoke: `5e664f729fb587ed3f6f61ff337a162569fb5038f570d8a4281093333ff1b106`
- outcomes (not authorized): `505711d3eb12c421f2fd6bd33077bf3fdb0dd870824a1cee5f5f73db05fe3a4d`
- capacity materializer (not authorized): `3aa8bfb42ff9f90ad059112126321c0ed1704b086be36d53801a5993c3a76b76`
- capacity trainer (not authorized): `970c5ca0cd3783797acbce4efa762c842dde1a6d6e4381f5f7d5105ba1823b72`

The exhausted diagnostic wrapper remains unchanged at
`1bedcf8b3418ebff72378d0c02473b4fae9a2ba027e8fd42ad7939996b9fefcb`
and is excluded from current-source pin checks while retained as immutable
historical evidence.

## Verification completed

- Focused production-v2 suite: `69 passed, 2 skipped`.
- Full CPU suite: `358 passed, 2 skipped, 3 subtests passed`.
- The skips are CUDA-only tests in the local non-GPU environment.
- All four changed Python files compile.
- All six production-v2 Slurm wrappers pass `bash -n`.
- The 59-file closure replays exactly.
- Static wrapper-pin and archived-diagnostic-identity tests pass.

## Required staged continuation

Fresh code review must first adjudicate the exact hashes above. At most the
following sequence may then be opened one boundary at a time:

1. create only the new-root identity split;
2. independently audit and bind that split;
3. run one synthetic CUDA v2 same-device + host-portable + independent-auditor
   roundtrip;
4. request a separate outcomes-stage review.

No outcome array, capacity, training, falsifier, validation, reserved, or
formal evaluation is currently authorized.
