# R079 Numeric-v2 Outcomes Result (2026-08-05)

## Verdict

`PASS` for fit/checkpoint outcome materialization. This result permits only a
fresh review of the independent outcomes-audit stage; it does not authorize
the audit itself or any capacity/training/evaluation work.

## Frozen inputs

- Array wrapper SHA-256:
  `505711d3eb12c421f2fd6bd33077bf3fdb0dd870824a1cee5f5f73db05fe3a4d`
- Source-closure SHA-256:
  `2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4`
- Split-manifest SHA-256:
  `7a572670867bbf6f811aad58c7ed1365f179083cda8187f5402221d554d1f1c0`
- Split-audit receipt SHA-256:
  `3df67764ef6dc7e8a827277c34730233bc7a5155451fe42c6b98b99bf7a7ef76`
- Numeric-policy SHA-256:
  `cbd80345e7249707931f71b29c65722ec8910263d51b7d649c5dd5c04fc4d4f0`

## Slurm evidence

- Array: `10137837`, exactly tasks 0-1.
- `10137837_0` / raw allocation `10137839` / fit:
  `COMPLETED 0:0`, 67 seconds, one NVIDIA A40.
- `10137837_1` / raw allocation `10137837` / checkpoint:
  `COMPLETED 0:0`, 29 seconds, one NVIDIA A40.
- Fit stdout: SHA-256
  `d4a5ce07b5e2494201220472d2d0fc60eda684d783ef2f7d0b8da6eb6d4940a0`.
- Checkpoint stdout: SHA-256
  `65bf8aba8dc7c40c2c72360004ffb4c26722f76289db4c35f6ae0c16ed26607b`.
- Both stderr files are empty, with SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- All wrapper preflights, source closure checks, split-receipt binding, frozen
  model/data hashes, and target-identity checks passed before GPU compute.

## Atomic artifacts

| Split | Prompts | Blocks | Metadata SHA-256 | Records SHA-256 |
|---|---:|---:|---|---|
| fit | 1,587 | 12,686 | `061069ed644b7fd700d7b65586622c02ef878c611ab6a549968f78bba8425f98` | `645007ec2665e141813b09e4bd1e35c33337b4b32e27655ae088c52c89fbcc6b` |
| checkpoint | 200 | 1,600 | `cdc879ef861608c6f26e004e4dd27826554f4c5631929e68f8dc57e8ea753047` | `203cadf7141684b91b7c41d70fc0222098898ca2f9200825ccd60fc8ecbb93a2` |

Exactly `metadata.json` and `records.pt` exist in each physical split
directory. No temporary bundle residue exists anywhere under
`artifacts/pros_gate/r079_numeric_v2`.

## Metadata and native witness

- Both artifacts use protocol `pros-gate-direct-outcomes-v2` and format 2.
- Both bind the full production numeric-policy receipt and its exact digest.
- Both bind the exact split manifest and unchanged start/end source closure.
- Fit native witness: 397 batches, 12,686 records, 5,955 same-device relation
  checks (`397 * 15`), 1,191 state-dict checks (`397 * 3`).
- Checkpoint native witness: 50 batches, 1,600 records, 750 same-device
  relation checks (`50 * 15`), 150 state-dict checks (`50 * 3`).
- All regular/hooked/repeated-output, repeated-state, numeric-invariant, and
  state-dict-unchanged witness booleans are true; before/after state-dict
  hashes agree.

## Outcome summaries and loader replay

- Fit: 1,411 beneficial, 754 harmful, 10,521 neutral, including 7,592 changed
  neutral blocks; base/direct token mass 63,782/66,428.
- Checkpoint: 174 beneficial, 101 harmful, 1,325 neutral, including 954 changed
  neutral blocks; base/direct token mass 7,867/8,214.
- The production `load_outcome_bundle` fully loaded and validated all
  12,686 fit and 1,600 checkpoint records against the frozen metadata hashes.
  It recomputed tensor shape/dtype/device/finite constraints, the
  operation-aware portable numeric relations, token-derived outcomes, unique
  block identities, and complete summaries without error.

## Boundary

No independent outcomes audit has run. A fresh reviewer must bind the two
metadata hashes above, the current independent auditor/wrapper/closure, and
the no-retry boundary before authorizing exactly one `AUDIT_STAGE=outcomes`
job. Capacity, training, falsifier, validation, reserved, and formal work
remain blocked.
