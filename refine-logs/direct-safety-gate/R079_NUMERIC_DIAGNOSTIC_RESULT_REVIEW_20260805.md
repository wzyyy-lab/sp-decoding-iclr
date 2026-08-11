# R079 Numeric Diagnostic Result Review (2026-08-05)

## Verdict

`GO` for production numeric protocol v2 design and implementation only.

## Blocking findings

None.

## Independently verified evidence

- The authorized script, tests, wrapper, Slurm-contract test, and 58-file closure hashes match the sealed shape-rescue review.
- The closure replay contains exactly 58 sorted unique paths with all byte counts and hashes correct; the pre-rescue and rescue closures differ only in the diagnostic script.
- Job `10137460` ran after the review and sealed-file mtimes, completed `0:0` on one A40 with 128 GiB in 71 seconds, and was the only new rescue attempt.
- Stdout is exactly one 5,179-byte canonical-JSON line, SHA-256 `54515f7271937cc1fd8ddcda1c762a05868c2d27a36fff62bf6b5fccb2217b3f`; stderr is empty.
- Protocol/status/counts are exact: diagnostic v1, policy pre-scan v1, PASS, 12,686 fit, 1,600 checkpoint, 862 positive synthetic cases, 1,343/1,343 rejected negative cases, and zero forbidden semantic operations.
- The independently derived 20-field census matches every field exactly, including 548,582,400 hidden copies and 605,839,056 total comparisons.
- Every envelope, cap, nonfinite, and range violation count is zero.
- The 1,343 negative cases independently decompose into 1,275 retained subset-boundary predecessors plus 68 fixed first-outside/material mutations.

## Nonblocking scope limits

- Evidence covers only the frozen fit/checkpoint inputs, preregistered synthetic grid, and the A40 / PyTorch 2.9.1+cu128 / CUDA 12.8 environment.
- Diagnostic v1 did not embed the wrapper/source hash or a policy digest in its JSON; production v2 must bind policy ID, canonical digest, and complete provenance.
- `forbidden_semantic_operations_executed=0` is supported by the sealed typed allowlist, AST boundary, and call graph, not dynamic taint tracking. The monolithic canonical pickle physically deserializes `gold_ids`; the valid statement is semantic noninterference, not absence from process memory.

## Authorized boundary

The result supports only this limited statement: the preregistered operation-aware policy accommodates observed CPU/CUDA differences on the frozen scanned inputs and rejects all preregistered negative mutations.

The next authorized work is production numeric protocol v2 implementation with:

1. policy ID plus canonical digest;
2. producer/materializer same-device invariants;
3. an independently implemented portable auditor; and
4. a new production source closure, artifact root, split/audit receipt, and CUDA authorization chain.

No outcome retry, capacity, training, falsifier, validation, reserved, or formal evaluation is authorized by this review.
