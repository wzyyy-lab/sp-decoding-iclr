# R080 Capacity Materialization Result (2026-08-05)

## Verdict

`PASS` for the sole authorized capacity artifact materialization. This result
does not authorize the independent capacity audit or capacity training.

## Execution

- Job `10137981` (`pros-cap-data`), wrapper SHA-256
  `3aa8bfb42ff9f90ad059112126321c0ed1704b086be36d53801a5993c3a76b76`.
- Job/batch/extern all `COMPLETED 0:0`; elapsed 67 seconds; 4 CPU / 64 GiB.
- Stdout SHA-256:
  `dff19e67d29b21bdab3401f5b2f4d7ac1933abf143edf8afbd8b8ce0a4674a4d`.
- Stderr empty; SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- All source/split/outcome metadata/outcomes-receipt preflights passed; receipt
  verifier returned `BOUND` before reading the fit bundle.

## Artifact

- Directory contains exactly `metadata.json` and `records.pt`.
- Metadata SHA-256:
  `215215915bc19be4312030f5f8235de75e3ff8718087a1d24a454c70bdc1e7ec`.
- Records SHA-256:
  `2617e374e94b8019ff72f8c42338d202998e2aebf9b999a57df19f88a3b905ad`.
- Protocol/format: `pros-gate-capacity-artifact-v1` / 1.
- Cardinality: 512 blocks from 512 unique prompts.
- Composition: 256 beneficial, 128 harmful, 128 changed-neutral.
- Semantic selection SHA-256:
  `e226170b6e63f5f1f8813343faf5487aaf08e3e3cc0c1b7b0df2648ff063c7c3`,
  exactly matching the fresh pre-execution review prediction.
- Metadata exactly binds fit metadata `061069ed...`, producer checkpoint and
  metrics, canonical metadata, split `7a572670...`, and source closure
  `2bd264d7...` / entries `cc4a9ece...`.
- The production capacity loader fully validated all 512 records, hashes,
  prompt uniqueness, composition, semantic selection, and source receipt.
- No temporary/partial artifact exists under the numeric-v2 root.

## Boundary

A fresh result/stage reviewer must independently replay this selection and
review the capacity-audit invocation before exactly one audit may run.
Capacity training and all clean-fit/falsifier/development/formal stages remain
blocked.
