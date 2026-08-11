# High-Capacity Frozen-Feature Probe Code Review

Date: 2026-08-04
Assurance: fresh same-family Codex review; provisional
Final verdict: **GO**

## Reviewed launch contract

The reviewer checked the positive-only OPB-10K diagnostic that compares:

- compact reference: axial-additive D64/H4/L1, learning rate `6e-4`;
- high-capacity probe: flat-compatibility D640/H10/L4, learning rate `3e-4`.

Both cells use Candidate-D-PACE with `alpha=0.5`, zero safety weight, batch
size 64, 30 epochs, and the identical nested OPB-10K materialization. The
frozen materialization contains 10,000 prompts and 79,931 blocks with prompt
hash `7cecb2289e172df0642056c3d5cc78f99f10093ec08ba79f7df30a57d89047e9`.
This gives exactly `ceil(79,931 / 64) * 30 = 37,470` optimizer steps per
cell.

## Gate and integrity checks

The launch is authorized only as a positive-only engineering diagnostic. It
passes if the D640 probe obtains either:

1. raw prompt-balanced EAL improvement over DFlash of at least `+0.6`; or
2. oracle-gap recovery of at least `0.15`.

Calibration and bootstrap intervals are descriptive and cannot satisfy the
gate. A negative result stops this probe; it cannot support an
information-ceiling claim. A positive result authorizes the preregistered
100K diagnostic but is not itself a method-effect claim.

The reviewer confirmed that the summary implementation:

- independently reconstructs base, direct, and oracle prompt-balanced EAL,
  block harm, first-token accuracy, and oracle-gap recovery from examples;
- fails closed on inconsistent reported scalars;
- checks source hashes, validation and external-data identity, target
  identity, and the full common configuration across both cells;
- excludes only the preregistered architecture fields, learning rate, and
  output path from the cross-cell equality requirement;
- requires exactly matching prompt keys and performs paired prompt
  resampling for bootstrap intervals;
- keeps the scientific-negative interpretation bounded to an engineering
  stop.

## Verification

- The 512-block D640 capacity witness completed 1,920 steps in 168 seconds on
  an A40 with 2.58 GiB peak allocated memory and passed all five checks.
- Slurm scripts pass `bash -n`.
- Python files pass compilation.
- Targeted verification passes: 73 tests plus 3 parameterized subtests.
- Array run-root labels and the dependent summary job-ID interface agree.

The final review verdict is **GO** for the matched OPB-10K array and its
fail-closed summary only.
