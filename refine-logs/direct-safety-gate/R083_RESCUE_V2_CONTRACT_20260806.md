# R083 One-Shot Falsifier Rescue v2 — Frozen Contract

**Date:** 2026-08-06  
**Workflow:** ARIS `experiment-bridge` blocker rescue  
**Current authorization:** the exactly-one submission was consumed by Slurm
job **10141601** on 2026-08-06 using wrapper
`b20fc9461daac0385b09fbd0840aa5d476dcf71fadd3b6ba3288938b9d124560`.
The fresh exact-hash review is recorded in
`R083_RESCUE_V2_CODE_REVIEW_GO_20260806.md`. No retry is authorized under any
exit condition.

## Immutable v1 rejection

The first reviewed wrapper
`27c079e2b869f9784b0493bc2c9f382587e3c8b54b05f6563bbe051e8d73f01b`
is permanently submission-ineligible. The fresh review in
`R083_EXACT_HASH_CODE_REVIEW_NO_GO_20260806.md` found four blockers before any
falsifier content was opened and before any job was submitted. The original
one-shot authorization therefore remains unconsumed, but it cannot be applied
to v1 or reused without a fresh review of this exact v2 identity.

No scientific choice changed. Split, selected checkpoint, model parameters,
ridge/constants, action `z>0`, seed, batch size, bootstrap count/seed, metric
equations, thresholds, gate conjuncts, input datasets, and no-rescue rule are
byte-for-byte or semantically unchanged.

## Minimal blocker closure

The 63-file v1 and v2 source surfaces have no additions or removals and exactly
two changed files: the evaluator and publication module.

1. Outcome replay now uses recursive tensor-aware exact equality over mappings,
   sequences, dtypes, shapes, and `torch.equal`. A real tensor-bearing outcome
   bundle is saved, reloaded, semantically validated, and compared in tests.
2. Preregistered scientific evaluation failures are fail-closed without score
   substitution:
   - nonfinite PROS/scalar-comparator values and regret-bound violations create
     an explicit `scientific_failure.json`, optional raw score tensors and
     partial records, a conjunctive FAIL receipt, full start/end identities,
     metrics, and durable READY before exit `2`;
   - unrelated exceptions remain operational failures;
   - zero/nonpositive recovery denominators no longer crash diagnostic
     bootstrap. They report zero valid replicates and a null recovery interval,
     while point-estimate recovery conjuncts fail and the normal complete FAIL
     packet is published.
3. Every consumer now requires pending and READY to be regular, non-symlink
   references to the same device/inode, with exactly two hard links and mode
   `0400`. Missing pending, copied READY, a third link, or wrong mode is rejected.
4. Input start/end identities now cover all 72 canonical shard files plus the
   target `config.json`, `model.safetensors.index.json`, and exact embedding
   shard selected by `model.embed_tokens.weight`. Paths, surface, bytes, and
   hashes are checked against frozen canonical metadata at both boundaries.
   The output directory and full source snapshot are reserved before any
   frozen-input validation/capture.

## Frozen v2 identities

- Evaluator: 63,036 bytes,
  `54d430a6d9d92118e2005e6c985c6c04f0424cfcc86afe60c5eecce5f39aa571`.
- Materializer: 35,669 bytes, unchanged,
  `90001f5b2f0224e79d8d205bdd781876f7dcbed89273fb55d9ca4c3f53d95b2b`.
- Publication module: 22,071 bytes,
  `141abdef88320173fb03c438c5c54907118a64bcf0a29932268be389fb4f5f1c`.
- Sole wrapper: 8,739 bytes,
  `b20fc9461daac0385b09fbd0840aa5d476dcf71fadd3b6ba3288938b9d124560`.
- Source closure: `R083_SOURCE_CLOSURE_RESCUE_V2.json`, 63 files,
  `204c025305a9665803e714708dc0eab29394644d5905ad76f1715c7309020878`;
  entries digest
  `fb943ef5be7fd2597e92f8bb230eaef480a4b78f33a1693a105e0f73aadbe796`.
- Falsifier/publication tests:
  `280c85882ff355e55a9e733a96ff1fdf0ec95acca4cf2bbf067ff335610724a3`.
- Slurm-contract tests:
  `b92b66375c9c58297bb7fcd586c9b9d8434f26cc0f7c872e6653f06eb14af7bb`.
- Materializer tests:
  `37777df157584bcb2a3a3dcd69f511fa63aafd7a5292f2329728a9b80fc00146`.
- Source-closure tests:
  `8024a370d851511cc803ab82f64f24f268150033d396ece291cb846d4a18e999`.
- Independent artifact-audit tests:
  `5434e70061c05b826c33cca7d6f1df07d80caf98146b8b02ce7949d6f37d1c4f`.

All scientific input hashes remain exactly those frozen in
`R083_ONE_SHOT_FALSIFIER_CONTRACT_20260806.md`.

## Verification evidence

- Blocker-focused evaluator/materializer tests: `38 passed, 1 explicit GPFS
  skip`.
- Combined falsifier/slurm/source/materializer/independent-audit suite:
  `70 passed, 2 explicit skips`.
- Full CPU suite: `432 passed, 4 skipped, 3 subtests passed` in 97.22 seconds.
- Python compilation, source-closure replay, wrapper static pins, and `bash -n`
  pass.
- Real input-identity preflight, without deserialization or identity listing:
  72 canonical shards / 1,565,176,184 bytes plus 3 target files /
  3,957,934,385 bytes verified in 18.60 seconds; aggregate identity digest
  `4f95342300528e6c5075b40b5e780c936611c67cf590cb7d58a759dd9dcd29d8`.
- Explicit real-GPFS race/crash integration: `1 passed`; JUnit receipt
  `artifacts/tmp/R083_RESCUE_V2_GPFS_PUBLICATION_INTEGRATION.xml`, SHA-256
  `35b966d68e62b24d2c502bda99fea950bc2af22cdd4abeef1c970a7ea5746cc2`.
- Retained exact-protocol v2 smoke:
  `artifacts/evaluation/r083_publication_smoke_preflight_v2/.pros-gate-r083-publication-smoke.76f41ca7e35c4ea5b6f885012dfb800a`.
  Manifest SHA-256 is
  `b0a5816b8b644ad3bee0eab502dfe696250a599833fe7365f68d91d9707db91f`;
  READY SHA-256 is
  `feac2eb61d9933e136bc6d079ef827cefe040048d3e1bc2db41ee8341a3a589b`.
  READY and pending are mode `0400`, link count `2`, device `59`, and share
  inode `318460897`; the strengthened consumer independently accepts them.

## Exact launch gate

A second fresh read-only reviewer verified every v2 hash above, independently
replayed all four blocker closures, confirmed that the two-file source diff
contains no scientific drift, and returned zero blockers. Wrapper
`b20fc946...` may therefore be submitted exactly once. There is no retry:

- exit `0` = fully published scientific PASS;
- exit `2` = fully published scientific FAIL;
- any other exit = operational failure consuming the opening.

No downstream data may be opened until a completed R083 artifact receives a
fresh R084 result-to-claim verdict.

## Submission record

- Slurm job: `10141601`.
- Sole wrapper SHA-256:
  `b20fc9461daac0385b09fbd0840aa5d476dcf71fadd3b6ba3288938b9d124560`.
- Slurm terminal state: **FAILED**, exit `1:0`, elapsed `00:00:30`.
- Failure boundary: `verify_phase3_split_manifest` rejected a reconstructed
  split-manifest difference limited to `provenance`; no official R083 READY
  publication exists.
- Stdout SHA-256:
  `55dc834f6011e5986b996ee30d743f57e07c8087bc56edb9a3091a5a5c15f68f`.
- Stderr SHA-256:
  `189dfeb09c983dabf828e64fcef11efd09e90878ce99e8519201dbea8d1779e5`.
- Authorization state: **consumed by operational failure**.
- Retry state: **forbidden**.
- Downstream state: R084 independently closed the exact route without an
  efficacy verdict. No validation, reserved, or formal data may be opened.
- R084 receipt: `R084_RESULT_TO_CLAIM_JOB10141601_20260806.md`.
