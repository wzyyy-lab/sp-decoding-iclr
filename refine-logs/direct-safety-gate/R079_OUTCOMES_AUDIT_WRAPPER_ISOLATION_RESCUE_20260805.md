# R079 Outcomes-Audit Wrapper Isolation Rescue (2026-08-05)

## Scope

This rescue responds only to the fresh audit-stage review's sole blocker. No
audit job or audit CLI has run, so this is a pre-execution wrapper correction,
not a retry.

## Exact change

- Exhausted/rejected wrapper SHA-256:
  `5b19d0bfe423651c99da27172d6f061ce4af0a50bd7be7630950329f9117b8f7`.
- Rescued wrapper SHA-256:
  `a5651e99bfdc925167a7dd427b4e34b2275578aeb9a962f86918635e2d15d83b`.
- The six size/hash checks for OPB, Phase-3 development, and Phase-3 reserved
  manifests moved unchanged from the common pre-dispatch prefix into only the
  `split)` branch.
- The `outcomes)` branch and independent auditor invocation are byte-for-byte
  unchanged. No data, artifact, numeric, model, or receipt logic changed.

## Regression and unchanged identities

- Added a static regression proving all three exclusion manifest paths are
  absent from the common prefix and outcomes branch, and present in the split
  branch.
- Updated Slurm-contract test SHA-256:
  `ce76b4b27b1ecb0ac030b024351c428e9d877d54746386d86c13f109b22d6889`.
- Slurm-contract tests: `7 passed`.
- `bash -n scripts/slurm/pros_gate_artifact_audit.sbatch`: PASS.
- Independent auditor remains
  `102a87a5ea9dbfedaf7fc6429fac79023bfd7e7ebbee062e332f4ad4ad5a863d`.
- The unchanged 59-file source closure replays as:
  `2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4`
  with entries digest
  `cc4a9ecea4c1f9a32e0ceae0e3c5551759e51be141bc4a9b37b6a5d03b88d02a`.

## Authorization boundary

This repair authorizes nothing by itself. A different fresh reviewer must
bind the rescued wrapper hash, prove runtime stage isolation and all frozen
inputs again, and explicitly authorize exactly one outcomes audit. Capacity,
training, falsifier, validation, reserved, and formal work remain blocked.
