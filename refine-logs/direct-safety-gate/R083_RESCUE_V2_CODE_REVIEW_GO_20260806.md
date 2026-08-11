# R083 Rescue v2 Exact-Hash Code Review — GO

**Date:** 2026-08-06  
**Reviewer:** fresh read-only GPT-5.6-Sol xhigh agent  
**Independence:** same-family; acceptance is provisional  
**Verdict:** **GO**  
**Submission authorized:** exactly one submission of wrapper
`b20fc9461daac0385b09fbd0840aa5d476dcf71fadd3b6ba3288938b9d124560`;
no retry.

## Structured verdict

- `exact_hashes_match`: **yes**
- `v1_to_v2_scope_exact`: **yes**
- `submission_authorized`: **yes**
- `authorized_wrapper_sha256`:
  `b20fc9461daac0385b09fbd0840aa5d476dcf71fadd3b6ba3288938b9d124560`
- `review_independence`: **same-family**
- `acceptance_status`: **provisional**
- `blocking_findings`: **none**

## Closure of the four v1 blockers

1. **Tensor-bearing replay: CLOSED.** The evaluator uses recursive,
   dtype/shape-aware `torch.equal`. The reviewer independently reproduced
   save, load, tensor equality, saved-record replay, and gate adjudication.
2. **Scientific FAIL publication: CLOSED.** Nonfinite PROS, ridge/scalar
   comparator failures, and regret-bound exceptions publish complete
   READY-bound FAIL packets without score substitution; unrelated exceptions
   remain operational. Invalid recovery denominators and zero-valid bootstrap
   replicates produce null recovery values and reach normal fail-closed gate
   publication.
3. **READY consumer integrity: CLOSED.** Missing pending, copied READY,
   wrong inode/device relation, link count other than two, wrong mode,
   symlinks, and payload tampering are rejected. The retained GPFS smoke has
   device 59, inode 318460897, link count 2, mode 0400, and passes current
   consumer replay.
4. **Input identity closure: CLOSED.** Start/end capture covers canonical
   metadata, all 72 declared shards, target config/index/exact embedding shard,
   split/receipt, Direct, R082 bundle, exclusions, and the source manifest.
   The independent preflight reproduced 72 shards / 1,565,176,184 bytes and
   three target files / 3,957,934,385 bytes. Output reservation and the full
   source snapshot precede scientific input capture/load.

## Identity and scope audit

All requested evaluator, materializer, publication, wrapper, manifest, and
five test hashes and sizes matched. The source closure independently verified
63 files, entries digest
`fb943ef5be7fd2597e92f8bb230eaef480a4b78f33a1693a105e0f73aadbe796`,
identical v1/v2 path sets, and exactly two changed entries: evaluator and
publication module.

Reverse reconstruction confirmed that removing only strengthened READY-link
verification recreates the v1 publication hash, and reverting only the three
source pins plus source-manifest filename recreates the rejected v1 wrapper
hash. No existing scientific constant changed: the v2 differences are blocker
plumbing, scientific-failure publication, identity closure, and diagnostics.

## Verification reproduced by the reviewer

- Focused exact suite: **70 passed, 2 skipped**.
- Static Python compilation and `bash -n`: passed.
- GPFS JUnit SHA-256:
  `35b966d68e62b24d2c502bda99fea950bc2af22cdd4abeef1c970a7ea5746cc2`.
- Smoke manifest SHA-256:
  `b0a5816b8b644ad3bee0eab502dfe696250a599833fe7365f68d91d9707db91f`.
- Smoke READY SHA-256:
  `feac2eb61d9933e136bc6d079ef827cefe040048d3e1bc2db41ee8341a3a589b`.

## Nonblocking findings

1. Saved-record replay is an in-process independent adjudicator and does not
   replace mandatory fresh R084 result-to-claim review.
2. For early comparator failures, some named gate booleans retain their
   PROS-only meaning, but the actual comparator failure is explicit in the
   scientific-failure reason/system, diagnostics, and false comparator
   validity/margin conjuncts.
3. The reviewer reproduced the exact 70-test focused subset and retained GPFS
   receipt rather than rerunning the documented 432-test full suite.

## Authorization boundary

Exactly one submission of the authorized wrapper is permitted. Exit 0 means a
fully published scientific PASS; exit 2 means a fully published scientific
FAIL; any other exit is an operational failure that still consumes the
opening. There is no retry. Every completed R083 outcome requires a fresh R084
result-to-claim review before any downstream data access.

The reviewer inspected or changed no Slurm state and did not open the
falsifier.
