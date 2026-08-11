+# PROS-Gate R079 Artifact-Stage Code Review

**Date:** 2026-08-05 12:46 +0800  
**Workflow:** ARIS `experiment-bridge`  
**Reviewer:** fresh same-family GPT-5.6-Sol xhigh, read-only  
**Final verdict:** **GO**

## Authorized review scope

This review covers the R079 identity split, fit/checkpoint outcome materializer,
independent artifact auditor, exact fit-only capacity artifact, capacity-only
trainer, source closure, Slurm stage wrappers, and synthetic/static tests.

The review did not inspect any real artifact payload, did not use CUDA/GPU, and
did not submit a scheduler job.

## First-pass verdict: NO-GO

The first pass found five deployment blockers:

1. Exclusion role names accepted arbitrary files rather than frozen
   producer-train, validation, and reserved identities.
2. Native/hooked output equality did not prove full producer
   `state_dict()` invariance.
3. The 320 deterministic capacity pass orders were generated inside the live
   loop instead of being persisted before optimization.
4. Runtime pins and the capacity source snapshot did not close the complete
   first-party import surface.
5. Prior GO receipts were hash-checked but not semantically bound to the
   current parent inputs.

The reviewer authorized CPU/synthetic remediation only and explicitly withheld
real-data, artifact, training, evaluation, GPU, and launch permission.

## Remediation

- Froze exact resolved path, byte count, and SHA-256 for all three exclusion
  manifests in both materializer and independent auditor; Slurm repeats the
  byte/hash checks before execution.
- Cloned every `state_dict` tensor and compared it bitwise after native,
  hooked, and repeated-hooked forwards. Persisted before/after semantic hashes,
  key/check cardinalities, and audited every witness. A producer with a
  forward-mutated registered buffer is rejected.
- Persisted and fsynced all 320 × 512 block-key orders before model or optimizer
  construction, independently reloaded every permutation/hash, and trained
  exclusively from the frozen orders. Checkpoints and metrics bind the order
  manifest SHA-256.
- Added a conservative source-closure manifest covering exactly every
  `scripts/*.py` and `src/sph/**/*.py` file. A stdlib-only verifier runs
  directly before any project import in every Slurm wrapper; each entrypoint
  requires the reviewed manifest SHA-256 and verifies start/end identity.
  Capacity snapshots the manifest and all 56 files with preserved paths.
- Added semantic audit receipts. Downstream wrappers bind GO status plus the
  exact current split, fit, checkpoint, capacity, canonical, Direct producer,
  Direct metrics, exclusion, and source-closure identities.
- Preserved the least-privilege boundary: R079 can materialize fit and
  checkpoint outcomes only. No falsifier outcome route exists.

## Frozen implementation identities

```text
R079 source manifest       50e094144aee2d18a01f51c45737a8c099384cffa476e12ac5f9fb961628c56c
source_closure.py          18c99279fb301b881db076d63627574b3c77205df8e91112373bd3579bee8a81
direct_safety_artifacts.py 0b264a095ce7b27b4c7832aeb086021e555de5ee791cb53bdbfff7f60b2368f2
materializer               3ccebdedd6877d20f6c40983ee41a1bdc1961bb6103862076d6b6e9727b7fbbf
independent auditor        d0307406488b70bbf35b3024ea6539bae76b6eab953b36413b4d5470391bb0b7
capacity trainer           2a8ea20a466096f540fc0d001f9dea0a766dc5c1773d351408f9e12d4b57e2f6
Gate-0 sidecar             e3bd6392f7430e60e0eef16217dc904eeb018313ae8d4f543bd089a1943739b6
Gate-0 protocol            bdde815e546993edb039e675e991cf6353477a62c24ad69215f056fb545ee24b
```

Slurm wrapper identities:

```text
artifact audit             827d4a31caa37d9e9896ba6c7feb26554479e618f8d8749b7b79c61f02ebfc66
capacity materialization   55b7a6d65b32f03290d16e40330cb88fbe258cb23256e1f7ef2be4fe4f0ed6a1
capacity training          6640c53fa8f5b1e1a308c4e13e40313a3ccaa5417398afffa22c2f72eb30fc94
outcome materialization    17f7242f94110600dd3ce93c1d5e6780ed3e2807d6eeaefb2149efe8ece2f169
split materialization      38c0aeb3e21e4e4116feb4f767ffa99e9fa4b848581e9f39c52c986a355ef7da
```

## Verification

- Focused synthetic/static suite: **34 passed**.
- Full CPU suite: **296 passed, 3 subtests passed**.
- Source closure: **56 exact files**, with independent implementations
  producing the same summary.
- Python compilation: passed.
- All five Slurm wrappers: `bash -n` passed.
- `git diff --check`: passed.

## Final focused re-review

The reviewer closed all five blockers and returned **GO**. Its binding
authorization is:

1. Create and independently audit the real identity-only split.
2. Only after a hash-bound split GO, materialize physically separate fit and
   checkpoint outcomes—never falsifier—and independently audit them.
3. Only after outcome GO, create and audit the exact 512-record fit-only
   capacity artifact.
4. Only after capacity-artifact GO, run exactly one reviewed seed-0 capacity
   job: 320 passes, 5,120 updates, and 321 checkpoints.

This GO does **not** authorize falsifier outcomes, validation/formal-test
access, clean fit/checkpoint training, additional seeds or widths, calibration,
refits, or any downstream claim.

