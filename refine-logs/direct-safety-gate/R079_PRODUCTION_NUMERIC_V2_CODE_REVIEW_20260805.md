# R079 Production Numeric v2 Fresh Code Review (2026-08-05)

## Verdict

`GO` for exactly one new-root identity split only.

## Blocking findings

None.

## Nonblocking findings

- The local environment has no CUDA; two CUDA-only tests skipped as designed.
  CUDA roundtrip requires a later, separately reviewed job.
- The 15 same-device relations are semantic categories. Eleven position-feature
  categories are jointly checked through exact equality of the complete
  `[B,15,200]` tensor rather than 11 separate calls. Independent per-category
  mutation testing covered all 15/15 categories and every mutation failed closed.
- Split physically deserializes canonical shards but reads only identity,
  domain, and source-split fields; it does not access outcome/gain/capacity or
  evaluation semantics.

## Independent evidence

- Canonical policy digest independently recomputed as
  `cbd80345e7249707931f71b29c65722ec8910263d51b7d649c5dd5c04fc4d4f0`.
- Production and auditor policy receipts match exactly; AST inspection confirms
  the auditor imports neither the production policy nor artifact validator.
- All ten diagnostic/production/auditor frozen constant groups match.
- Retained bounds match on 635 additional random cases; add/sub, normalized,
  entropy, and retained `1e-4` mutations are rejected by both implementations.
- All 15 same-device categories independently fail closed under targeted
  mutation.
- Focused suite independently returns `69 passed, 2 skipped`.
- The 59-file closure and entries replay as `2bd264d7...` / `cc4a9ece...`.
- All six production wrappers pass `bash -n` and match their contract hashes.
- Old wrapper, old production closure, diagnostic closure, and job10137460
  stdout identities remain unchanged.
- New root `artifacts/pros_gate/r079_numeric_v2` is absent and the split writer
  refuses overwrite.

## Exact authorization

Only one submission of `scripts/slurm/pros_gate_split.sbatch`, SHA-256
`ac30d701faf6070f2dbfa6e46bcd27b1d2e989e0981fd24aea21e94ead2b4f86`,
is authorized. After completion, the split hash must be frozen and an
independent split audit must run and be adjudicated. A CUDA numeric-v2 smoke
then needs separate review.

No outcomes, capacity, training, falsifier, validation, reserved, or formal
evaluation is authorized.
