+# PROS-Gate R079 Artifact-Stage Code Review — Binding Portability-Rescued Version

**Date:** 2026-08-05 13:04 +0800  
**Workflow:** ARIS `experiment-bridge` + deployment rescue  
**Final verdict:** **GO**  
**Binding source manifest:** `dccf65403ec539af50c380174e0ee5a6d093353338ae664125780e1f1fa2d51f` (57 files)

## Review history

The initial artifact-stage review found five blockers: unfrozen exclusion
identities, incomplete producer-state invariance, live generation of capacity
orders, incomplete first-party source closure, and stale semantic GO-receipts.
All five were remediated, and the focused reviewer returned GO for source
manifest `50e094...c56c`.

Before job submission, deployment preflight established that `jq` is absent
from the cluster environment. The prior GO was therefore suspended without
submitting a job or opening a real artifact payload.

## Portability rescue

A single stdlib-only script,
`scripts/verify_pros_gate_receipt.py`, replaces the three downstream `jq`
predicates. It fails closed on:

- receipt regular-file/non-symlink status and reviewed file SHA-256;
- exact `status == "GO"`;
- current split, fit, checkpoint, and capacity parent hashes;
- canonical metadata, frozen Direct checkpoint/metrics, and source-manifest
  identities;
- nested fit/checkpoint metadata hashes;
- exact exclusion resolved paths, byte sizes, file hashes, and aggregate
  hashes.

The helper is part of the exact conservative closure, which now covers 57
files under `scripts/*.py` and `src/sph/**/*.py`. Every wrapper still runs
the stdlib-only source verifier before any project entrypoint import. No
`jq` reference remains in the five R079 wrappers.

## Current frozen identities

```text
source manifest             dccf65403ec539af50c380174e0ee5a6d093353338ae664125780e1f1fa2d51f
receipt verifier            7152d94f47cf9f2e0dd28bb6bda097062d2f3090a6a2e963afc9784b12efa104
source verifier             18c99279fb301b881db076d63627574b3c77205df8e91112373bd3579bee8a81
artifact contract           0b264a095ce7b27b4c7832aeb086021e555de5ee791cb53bdbfff7f60b2368f2
materializer                3ccebdedd6877d20f6c40983ee41a1bdc1961bb6103862076d6b6e9727b7fbbf
independent auditor         d0307406488b70bbf35b3024ea6539bae76b6eab953b36413b4d5470391bb0b7
capacity trainer            2a8ea20a466096f540fc0d001f9dea0a766dc5c1773d351408f9e12d4b57e2f6
artifact audit wrapper      76c4859f9bfc6ad57642bd46d6e5d056910ec5016b578d5ffa5b974691c4441c
capacity-data wrapper       749a8f3e616746b629504a7ee62b0edb447ac9b442c977c200ff6a9a5b1ce9db
capacity-training wrapper   71b30bc0e30a405cb3edf41b6e676dc319289596b588241d12f2578a1013a426
outcomes wrapper            85119100d17ec3f244352a146fa022e9d5ef4d591b46153f471daf78022730ce
split wrapper               c539df5ef2317b12e124ca7e3d67c4c3751fae3a84fbb7f4eedc268278cc1237
```

## Verification

- Portability/source/contract review suite: **11 passed** independently.
- Local focused artifact suite: **38 passed**.
- Local full CPU suite: **300 passed, 3 subtests passed**.
- Python compilation, source preflight, all five `bash -n` checks, and
  `git diff --check`: passed.
- Real payloads opened: none.
- GPU used / jobs submitted before verdict: none / none.

The fresh portability reviewer reported no blocking and no non-blocking
findings. It independently reconstructed the prior 56-file manifest after
removing only the helper, confirming the rescue introduced no unrelated
first-party source change.

## Binding authorization

Only this sequence is authorized:

`split → independent split audit → fit/checkpoint outcomes → independent outcomes audit → capacity artifact → independent capacity audit → exactly one seed-0 capacity job`.

No falsifier outcomes, validation/formal-test access, clean training,
additional seed/width/job, calibration, refit, rescue, or downstream claim is
authorized.

