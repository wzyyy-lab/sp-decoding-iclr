# R079 Outcomes-Audit Isolation Rescue Review (2026-08-05)

## Verdict

`GO` for exactly one numeric-v2 outcomes audit using wrapper SHA-256
`a5651e99bfdc925167a7dd427b4e34b2275578aeb9a962f86918635e2d15d83b`.
The repair occurred before the first outcomes-audit execution and is not a
retry. Blocking findings: none.

## Independent findings

- Moving the six exclusion-manifest preflights back to the common prefix in
  memory exactly reconstructs the rejected wrapper SHA
  `5b19d0bfe423651c99da27172d6f061ce4af0a50bd7be7630950329f9117b8f7`;
  this proves the rescue diff is only the reviewed branch relocation.
- The common prefix and `outcomes)` branch no longer access falsifier,
  validation, reserved, or capacity data/artifacts.
- Slurm-contract test SHA is `ce76b4b2...`; `7 passed`; `bash -n` passed.
- Auditor `102a87a...`, source closure `2bd264d7...`, all split/outcome hashes,
  numeric policy, Slurm completion evidence, empty stderr, record replay,
  identity/domain checks, summaries, and prompt disjointness were independently
  reproduced without production-validator reuse or audit execution.
- No prior numeric-v2 outcomes receipt, partial/tmp output, or concurrent audit
  existed at authorization time. Historical `pros-audit` jobs were split-stage
  audits only.

## Exact authorization

```bash
sbatch --export=ALL,AUDIT_STAGE=outcomes,SPLIT_MANIFEST_SHA256=7a572670867bbf6f811aad58c7ed1365f179083cda8187f5402221d554d1f1c0,FIT_METADATA_SHA256=061069ed644b7fd700d7b65586622c02ef878c611ab6a549968f78bba8425f98,CHECKPOINT_METADATA_SHA256=cdc879ef861608c6f26e004e4dd27826554f4c5631929e68f8dc57e8ea753047 scripts/slurm/pros_gate_artifact_audit.sbatch
```

Any abnormal/nonzero/partial result, temporary residue, or missing receipt
permanently closes the route without retry. Success still requires a fresh
outcomes-result review before capacity. No later stage is authorized.
