# R082 Publication Rescue v1 — Frozen Preparation Contract

**Date:** 2026-08-05  
**Workflow:** ARIS `experiment-bridge` failure rescue  
**Current authorization:** preparation complete; **submission NO-GO pending a
fresh exact-hash review**

## Immutable failure boundary

Job `10138454` and its old identities remain immutable:

- wrapper `059e06cfe90d17a929d6d59999b72eb810ac446c5f6115d936c66e1b2c684e69`;
- trainer `1e95a5dd1eaf6673e49cd2a9b778a708724a1e394b12d2046c3ef449593bfff7`;
- source closure `1d02ebed845df7e7183baaf1741f5a05ca59661fc905faa6974307623efb4206`;
- stdout `e6cd56baea567634b4400182a9cab7331e9aadfe3c14be77f438d0c6106348f7`;
- stderr `94b3d0c5f9db01a5393c6a33f5e85952e6e1f18462f282bea8947d33f52bf169`.

The old one-submission GO is consumed. The exposed pass trajectory is not an
official result, cannot recover a checkpoint, and was not used to alter any
scientific choice.

## Frozen scientific identity

The new version keeps the exact same fit/checkpoint records and metadata,
outcomes receipt, capacity-adjudication receipt, initialization seed `0`,
prompt-balanced gain hinge, architecture, optimizer, batch partition
`198×64+14`, 25 passes / 4,975 updates, learning-rate schedule, ridge
comparator, checkpoint selector, decision threshold, source dependencies, and
absence of later-data CLI surfaces.

The old and new 61-file source closures have no added or removed paths and
exactly one changed entry: `scripts/train_direct_safety_fit.py`. A bytewise old
trainer snapshot with the old SHA is retained at
`/tmp/r082_train_direct_safety_fit_v2_1e95a5dd.py` for the pending review. Its
diff against the new trainer is limited to publication plumbing, wrapper
identity capture, consumer verification, output-root naming, commit calls, and
removal of failure cleanup.

## Publication protocol

`pros-gate-r082-directory-commit-v1` uses only GPFS-verified primitives:

1. The new job-scoped `seed0` is acquired through one directory-fd exclusive
   `mkdir`; existing files, directories, or symlinks fail without overwrite.
2. A read-only `RESERVATION.json` permanently binds job, seed, absolute output,
   trainer, wrapper, and source closure. Missing `READY.json` is the sole
   incomplete state.
3. All payload files and directories are fsynced bottom-up. The exact tree
   manifest records every relative path, byte count, SHA-256, directory, end
   input identity, source-closure end identity, and scientific status.
4. The tree is rescanned after manifest construction, after fsync, and after a
   parent-side read-only READY pending receipt is fsynced.
5. A directory-fd `os.link(..., follow_symlinks=False)` atomically creates
   `seed0/READY.json` without replacement. The output directory is fsynced and
   never mutated afterward; the parent pending hard link is retained.
6. Every failure preserves the reserved directory and all accumulated
   evidence. A published scientific `FAIL` is complete evidence, not a
   scientific GO. Consumers require successful job completion in addition to
   a semantically valid READY tree.

The new wrapper runs the exact publication smoke on the actual job-scoped
`artifacts/training` GPFS before hashing or opening either `records.pt` and
before any GPU update. The smoke bundle and pending receipt are retained.

## Frozen v1 identities

- Trainer: `9bc069a5d70ff6f074e323a076c065eb7ae8ff903f4ed83cd1f6a317b790bff4`
  (`77,027` bytes).
- Wrapper: `1bffe45017da30393f2dbda5bd33b1e14ff1e0ca1a60c0be4fc86f4ad1885c74`
  (`5,366` bytes).
- Source closure:
  `f36291a961ea793dbaa888950bc4312d8b53954fcc5ecdb01a5caad4af97e184`;
  61 files; entries digest
  `7d6f8dfabd30e7968eb8c9b8857492796872d3a9dade1468a89d7f7ba81ad123`.
- Publication tests:
  `0fdaad4ddfbd7e490274659475a4ae1b4cc0fa98de30d12091065472cdb1894c`.
- Slurm-contract tests:
  `3e11f99214c843096d964aa6aea7b707a7364ec78f00579fbde70584b932fb8d`.
- Source-closure tests:
  `04493afc4be3bb2e3730fd6717482906100c4b2b9662cd281b53b39a8913c71d`.
- Artifact-audit tests:
  `039cc4a9feccedf20ee9aef14ed385c4f7a52baafceef7e8c253c2d84fcb1329`.

## Verification evidence

- Focused contract/protocol/audit suite: 93 passed, one CUDA-only skip and one
  explicit real-GPFS integration skip in the default CPU run.
- Full CPU suite: 396 passed, 3 CUDA-only skips, 3 parameterized subtests.
- Real GPFS integration: 1 passed. It covered a nonempty nested payload,
  file/directory fsync, two-process exclusive-mkdir race, two-process READY
  hardlink race, kill before link, and kill after link but before final fsync.
- GPFS JUnit receipt:
  `artifacts/tmp/R082_GPFS_PUBLICATION_INTEGRATION.xml`, SHA-256
  `ee460fb86e33f290a89482cbf85bf44c076d8a070f7aa2c500d53028628dc012`.
- Retained actual-filesystem smoke:
  `artifacts/training/r082_publication_rescue_preflight_local/.pros-gate-publication-smoke.5e27c7b546464378950e50d0c1fa3ee4`.
  Its manifest SHA is
  `9c6cffbb90ea99a1cf750f96aa265d04777c85885bb970dcfd35095d14896231`;
  READY/pending SHA is
  `26d44054d03d2bf0a6ea15ee5733eed52223bbf1f97b50ba97c6c47c80302dd2`.
  READY and pending are mode `0400`, link count `2`, and share GPFS device `59`
  and inode `5100366760`.
- Wrapper syntax and the new source-closure replay pass. No active `pros-fit`
  or `pros-fit-pub` job was observed at freeze time.

## Gate

A fresh, read-only code reviewer inspected the exact hashes above and returned
GO with zero blockers. It authorized exactly one submission of wrapper
`1bffe45017da30393f2dbda5bd33b1e14ff1e0ca1a60c0be4fc86f4ad1885c74`.
Any submission/outcome consumes that authorization; there is no retry. Any
later-data access, model/threshold/seed/hyperparameter change, manual
checkpoint recovery, or old-wrapper reuse remains forbidden. The sole allowed
successor is a hash-bound result-to-claim review of the new job.
