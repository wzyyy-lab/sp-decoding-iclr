# R083 One-Shot Falsifier — Frozen Preparation Contract

**Date:** 2026-08-06  
**Workflow:** ARIS `experiment-bridge`  
**Current authorization:** preparation complete; **submission NO-GO pending a
fresh exact-hash code review**

## Authorization and unopened boundary

The fresh R082 result-to-claim review authorizes exactly one unchanged R083
opening after implementation, tests, source closure, wrapper freeze, and a
fresh code-review GO. Preparation has not materialized, loaded, enumerated, or
inspected the 200 falsifier prompt identities or their outcomes. Static source,
schemas, aggregate split-audit counts, paths, byte counts, and cryptographic
identities were inspected; no validation, reserved-test, or formal-evaluation
records were opened.

The opening is consumed by the first submitted job regardless of scientific
result or operational exit. There is no retry, threshold change, refit,
calibration, alternate checkpoint, extra seed, longer training, wider model,
split change, or later-data rescue. A scientific FAIL closes this exact PROS
route after complete evidence publication. An operational failure also closes
the unique opening unless a later ARIS failure audit explicitly establishes a
new authorization; no such rescue is pre-authorized here.

## Frozen scientific identity

- Split: exactly `falsifier`, 200 prompt-disjoint prompts and 1,600 blocks;
  expected domains are chat/code/math `66/67/67`.
- Method: the immutable R082 pass-5/update-995, 38,674-parameter sidecar;
  action is strictly `APPLY iff raw z > 0`.
- Comparators: DFlash/base, frozen Direct, the fit-only 21-scalar ridge,
  always-KEEP, and always-Direct. Nothing is fit or selected in R083.
- Materialization: frozen Direct-native behavior, batch size 32, candidate
  width 16, using only the dedicated falsifier-only function. The historical
  R079 CLI remains unable to expose falsifier outcomes.
- Statistics: exact full token masses/EAL, unclipped recovery, harm,
  first-token counts, outcome/action composition, regret, domain slices,
  exact-linear quantiles, and 10,000 deterministic prompt-cluster bootstrap
  replicates with seed `20260806`. Point estimates, not intervals, bind the
  gate.

Every following check is conjunctive:

1. finite scores and selected-checkpoint gradients, with zero regret-bound
   violations;
2. finite positive oracle denominator and unclipped recovery in
   `[0, 1 + 1e-6]`, with recovery at least `0.90`;
3. PROS EAL strictly exceeds both DFlash and frozen Direct;
4. harmed fraction is at most `0.05`;
5. PROS first-token count is at most one below Direct;
6. all frozen comparator recoveries are valid and PROS recovery is at least
   `0.05` above the best of them;
7. exact source, split, exclusions, Direct, R082, native-state, and publication
   identities hold from start to end.

## Frozen inputs

- Canonical metadata: `0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320`.
- Split manifest: `7a572670867bbf6f811aad58c7ed1365f179083cda8187f5402221d554d1f1c0`.
- Independent split receipt: `3df67764ef6dc7e8a827277c34730233bc7a5155451fe42c6b98b99bf7a7ef76`.
- Direct checkpoint / metrics:
  `9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e` /
  `9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef`.
- R082 READY / publication manifest / metrics:
  `91c51864339321436ffa560470667bb99d1a22e955e7273fcd7a3d5711f3f508` /
  `9c17020a95d42725a097e6847b5023d6c6e7971a0647fb4475727e93d2297a5e` /
  `ba7fd8264813b7baa4927d94d1acd8d697bad5175d1acdfcc62dea5a103491b0`.
- R082 selected bundle / checkpoint / selected records:
  `a0abcfd4e56229647afd1dda5ca2fe861f7dbf21d00c09fe275ba8a66826c142` /
  `f3e7c68dafd93528c03deda9710e3d23cf5b0e9e51a7b2ef66200f08201066dc` /
  `1fd6beb846c8a46e874875628b37ac1094e3ac61eaddbfef3bb9d7b7dbd88749`.
- Ridge model / freeze receipt:
  `2c5a76ca96f9f6afb08d47a116cdb17fdce6de11386b6350c5e3e485732f4f16` /
  `2a4f6457fc1d85a3ad42ec7430ecb4d28d532d4bf57f06371ac019526c3dc809`.
- Producer-train / validation-exclusion / reserved-exclusion manifests:
  `b05087a56e8e717605415026421f7bae23092eb7cb9509361a36932f80260e3a` /
  `e16374068e9c8904214fbf282b4adb6187a0b099db5c37e79660fc46a2801d01` /
  `ae25467fbb52b7091c8d9a5f98776b11ccf76e87e781850b5638734548a53bb4`.

## Frozen implementation identities

- Evaluator: `scripts/evaluate_direct_safety_falsifier.py`, 44,245 bytes,
  SHA-256 `8587ab4a202a95d88f34d00b8b26c445cc8ba16372e4ddecd95adfa7f1cc0ad7`.
- Materializer: `scripts/materialize_direct_safety_artifacts.py`, 35,669
  bytes, SHA-256
  `90001f5b2f0224e79d8d205bdd781876f7dcbed89273fb55d9ca4c3f53d95b2b`.
- Publication module: `src/sph/direct_safety_publication.py`, 21,042 bytes,
  SHA-256 `07e7bd0d04c479dabb00e1c5ad8e27e1bc9cd5deb3ebf4ee1dcf40964856fc10`.
- Sole wrapper: `scripts/slurm/pros_gate_falsifier.sbatch`, 8,725 bytes,
  SHA-256 `27c079e2b869f9784b0493bc2c9f382587e3c8b54b05f6563bbe051e8d73f01b`.
- Source closure: `R083_SOURCE_CLOSURE_V1.json`, 63 files, SHA-256
  `b78da0f9e6203e7b481302a1dd48e2683bf2b8ae8431cf5bf8aaf36ee6e275ba`,
  entries digest
  `6d9b994e5937dac293e070f95d707d8b1333bc6072e48b71b57e07925ea85f36`.
- Falsifier tests:
  `303a49608642b2935d4a9c08dd6f0cc98796934dab80fede27c7907f86b7fbee`.
- Slurm-contract tests:
  `cbd072b14016cce747f012e73e192e601941b1910bed9dc26a5014de465689d4`.
- Materializer tests:
  `37777df157584bcb2a3a3dcd69f511fa63aafd7a5292f2329728a9b80fc00146`.
- Source-closure tests:
  `6f57882faae0ccf52a58ebc42511409f31df6c366b163c55123fa0653446376c`.
- Independent artifact-audit tests:
  `0e254bd915786a8c7a8e342de4c52ece438489aa1132ef72c72e3409efece8ff`.

## Ordering, publication, and exit contract

The wrapper is one non-array, one-GPU, 30-minute job with output fixed to
`artifacts/evaluation/pros_gate_falsifier_${SLURM_JOB_ID}/seed0`. It first
verifies the 63-file source closure, then exercises the exact
`pros-gate-r083-directory-commit-v1` filesystem smoke. Only after that smoke
commits READY may it hash or open the split, canonical collection, exclusions,
Direct checkpoint, or R082 bundle.

The real run exclusively reserves its job-scoped directory, snapshots the
entire verified source closure, validates all frozen inputs, and only then
opens the falsifier once. It publishes outcomes, raw action records for every
system, domain metrics, quantiles, bootstrap distributions, gate receipt,
metrics, complete input start/end identities, and a source snapshot. The
payload tree is fsynced and content-manifested; read-only READY is created by a
no-replace hard link. No failure path removes accumulated evidence.

- Exit `0`: fully published scientific PASS.
- Exit `2`: fully published scientific FAIL.
- Any other exit: operational failure; evidence is retained, but there may be
  no READY. The wrapper never retries.
- For exits `0` and `2`, an independent post-run consumer must replay the READY
  tree before the wrapper returns the original scientific exit code.

## Verification evidence

- Focused source/slurm/falsifier/materializer suite: `25 passed` before the
  expanded publication tests; final focused publication/slurm/source suite:
  `33 passed, 1 explicit GPFS skip`.
- Full CPU suite after the current-source pointer repair: `419 passed, 4
  skipped, 3 subtests passed` in 90.70 seconds.
- Python compilation and `bash -n` both pass.
- Explicit real-GPFS race/crash integration: `1 passed`; JUnit receipt
  `artifacts/tmp/R083_GPFS_PUBLICATION_INTEGRATION.xml`, SHA-256
  `af67ed87f43923dd7fc48a286f89aee9c530a03acc2ce46cce78d90eb6d0d7ea`.
- Retained exact-protocol smoke:
  `artifacts/evaluation/r083_publication_smoke_preflight_v1/.pros-gate-r083-publication-smoke.b20430bfdc6b420fb0456f561cc280cd`.
  Manifest SHA-256 is
  `8cdd72c74f71782c8777a1eda66321e022172bc2bae40061ea4d24acf76a37b3`;
  READY SHA-256 is
  `c79789ee540ddb69aca38765928f015473bf49c277ec75db75c09a8b54466c85`.
  READY and pending are mode `0400`, link count `2`, device `59`, and share
  inode `1170930995`.

## Review gate

A fresh read-only reviewer must audit the exact identities in this document,
including code semantics, split isolation, lack of tuning surfaces, metric
equations, invalid-denominator handling, bootstrap grouping, scientific-FAIL
publication, source snapshot, filesystem races, wrapper order, and no-retry
behavior. Only an exact-hash GO with zero blockers may authorize one `sbatch`
of the wrapper above. Until then, R083 remains submission NO-GO.
