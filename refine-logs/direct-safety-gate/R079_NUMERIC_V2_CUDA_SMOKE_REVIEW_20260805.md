# R079 Numeric-v2 CUDA Smoke Review (2026-08-05)

## Verdict

`GO` for exactly one synthetic CUDA smoke submission.

## Evidence

- Split/audit jobs `10137729` / `10137749` completed `0:0`, with empty stderr.
- Receipt `3df67764...` is GO and independently BOUND to split `7a572670...`
  and closure `2bd264d7...`.
- The 59-file closure replays, and wrapper/test/current source pins match.
- Wrapper SHA is `5e664f729fb587ed3f6f61ff337a162569fb5038f570d8a4281093333ff1b106`;
  test SHA is `b552f16dd8f9afab8df570609391f443571ca199ed6e13b2a00844b8f2c4a4c5`.
- It is a single-node, single-task, one-GPU, non-array, five-minute job.
- CUDA availability is asserted before pytest, and `PROS_REQUIRE_CUDA=1`
  converts an unavailable-CUDA skip into failure.
- The selected test creates only fixed-seed synthetic tensors, performs the
  15-category same-device exact invariant on CUDA, copies to CPU, and then
  invokes both production portable validation and the separate auditor.
- No canonical data, model, outcome, gold, capacity, falsifier, validation,
  reserved, or formal path is opened; only Slurm logs are written.
- The old smoke job used the old closure and is not a numeric-v2 attempt.

## Nonblocking finding

The auditor module defines static real-data/model paths, but this selected
tensor-only call path never dereferences them. Old and new smokes share a job
name, while `%j` log names prevent overwrite.

No outcomes or downstream stage is authorized.
