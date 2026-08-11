# R079 Numeric-v2 CUDA Smoke Result (2026-08-05)

- Job: `10137790`
- Wrapper SHA-256: `5e664f729fb587ed3f6f61ff337a162569fb5038f570d8a4281093333ff1b106`
- Slurm: `COMPLETED 0:0`, 5 seconds, one NVIDIA A40
- Test: `test_cuda_same_device_and_portable_independent_roundtrip`
- Result: `1 passed in 1.80s`, zero skipped
- Stdout: 9 lines / 638 bytes, SHA-256
  `e06b300310fdfe7baed7c64f41fd7053e144b0160f3abfca420f0674168d9a90`
- Stderr: 0 bytes, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- All pinned source, test, and 59-file closure preflights passed.

The smoke constructed fixed-seed tensors on CUDA, passed all 15 same-device
bitwise relation categories, copied the persisted inputs/features to CPU, and
passed both the production operation-aware validator and separately
implemented auditor. It did not access data, model, outcome, gold, capacity,
falsifier, validation, reserved, or formal inputs and wrote no artifact beyond
Slurm logs.

This result permits only a fresh outcomes-stage review. It does not itself
authorize outcome materialization or any later stage.
