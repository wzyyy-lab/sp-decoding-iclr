# PROS-Gate R079 Artifact Review — Binding Split-Filter Rescue

**Date:** 2026-08-05 13:28 +0800  
**Workflow:** ARIS `experiment-bridge` failure rescue  
**Final verdict:** **GO**  
**Binding source manifest:** `513ad34d8a71cd4bb340eaeda2dd8132be311a38f075d2148af2dadf7ef05a53` (57 files)

## Trigger and remediation

The previously reviewed split job `10135740` failed closed before output. The
combined `phase3_development_v3.jsonl` contains train and validation rows, but
the old materializer excluded every row as validation. The repair freezes and
checks per role: exact path/bytes/SHA-256, selected row splits, and the complete
file row census. Producer selects `train`; validation selects only
`validation_gate` and `validation_select`; reserved selects `test`.

The materializer and auditor implement parsing independently. Both persist and
bind the selected splits and census in provenance and semantic hashes. The
stdlib receipt verifier independently pins those enriched identities.

## Binding identities

```text
source manifest             513ad34d8a71cd4bb340eaeda2dd8132be311a38f075d2148af2dadf7ef05a53
materializer                26d81d57e611e3ca2adb62fe7c6aaa7262283726e201f5ca62f5475172073e74
independent auditor         9be5d38fc67a00aa9aec3b10f5951cb4e63f1b84917a6deac979827b8391bec4
receipt verifier            c93325260be458d9904d25503670a97a4d04c277837a77cb4e22f5178db0091a
artifact contract           0b264a095ce7b27b4c7832aeb086021e555de5ee791cb53bdbfff7f60b2368f2
protocol                    bdde815e546993edb039e675e991cf6353477a62c24ad69215f056fb545ee24b
source verifier             18c99279fb301b881db076d63627574b3c77205df8e91112373bd3579bee8a81
capacity trainer            2a8ea20a466096f540fc0d001f9dea0a766dc5c1773d351408f9e12d4b57e2f6
split wrapper               65cf7a59df14af4ff8d6c26f6583e7484105e6d982af24664680bdcd551ed97f
outcomes wrapper            9057b2daa95e48ed1a030ceee19e9d36967a4c2a7d8b57167e792bbfe3bfa916
artifact audit wrapper      9c5bc73a9bbcbed90ae80d9e8703071598dedca1a71b5a90b487128a7fbce61d
capacity-data wrapper       0235c63523d092db54415ff61a1e43b94527505fadece25fe4fa298779669d5f
capacity-training wrapper   38cb0e10fce35e9c327da0f6204eb4d37ff5a0c6db21e72e6a9d95a68e4e2505
```

Semantic exclusion hashes:

```text
producer_train              dcd1decfa63d17b4f4ee180a2d30e774ffb87bc9eed96a956f045a117039b16d
validation                  fc336fa8672140facd82dc6f73be067c02d87e27bb6e276137a290a16cc7ab09
reserved                    94c3e4274af5f310042766f962fcc3bf57b854d9b1f3e99bec950ddb367b4885
```

## Independent evidence

The fresh xhigh reviewer independently reproduced the old all-row bug and
verified the corrected real identity surface:

- Phase-3: 1,987 prompts / 15,886 blocks, domains 655/665/667;
- selected exclusions: 100,000 producer, 300 validation, 600 reserved;
- Phase-3 overlap with those three sets: 0/0/0;
- frozen prompt split: 1,587/200/200 with exact domain counts
  523/66/66, 531/67/67, and 533/67/67;
- block split: 12,686/1,600/1,600;
- focused repaired suite: 20 passed; local full suite: 302 tests plus three
  parameterized subtests passed;
- 57-file closure preflight and all five wrapper syntax/static-pin checks
  passed.

No outcome payload, GPU, scheduler, or job was used during review. There were
no blocking findings. Two non-blocking hardening suggestions were recorded:
post-parse exclusion rehash/same-descriptor reading, and a separate wrapper
manifest. Existing mandatory independent split audit contains the former's
theoretical race; neither suggestion changes this bounded verdict.

## Binding authorization

Authorize exactly one resubmission of `pros_gate_split.sbatch`. After it
completes, freeze the split-manifest SHA-256 and run the already reviewed
independent split audit and receipt verification.

This GO does not directly authorize outcomes, capacity materialization,
training, evaluation, falsifier access, validation/formal access, additional
seeds/widths/jobs, calibration, refit, or downstream claims.

## Stage-boundary addendum after split receipt GO

Split job `10135795` and independent audit `10135872` both completed `0:0`.
Manifest `ae7ea2fb...ed04` and receipt `50d202de...185f` replayed as `BOUND`.
A reviewer follow-up found no blockers and authorized exactly one existing
array submission with tasks `0=fit` and `1=checkpoint`, followed only by the
independent outcomes audit. Task failure requires diagnosis and does not permit
a retry. Capacity, training, falsifier, and evaluation remain closed.

The first submission, job `10135884`, never allocated because the wrapper's
40-minute request exceeded debug's 30-minute hard limit. It was cancelled with
elapsed 0 and no output. A deployment-rescue review proved the corrected
wrapper differs by exactly one byte (`40→30` minutes), current SHA-256
`750b5995...f7dba`, and issued GO for one resubmission. A timeout or failure
does not authorize another retry.
