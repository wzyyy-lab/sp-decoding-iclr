# JAPD-16 M1 D256 Conditional Branch Code Review

## Verdict

**SUBMIT GO** for J010-D256, J011-D256 and J012-D256. No blocker was found.
This authorizes only the preregistered conditional M1 branch; M2 remains
blocked until all three D256 gates pass.

Review assurance: fresh same-family Codex reviewer, provisional, pending
external review.

## Contract and recipe checks

- D64 J010 and J011 both have `gate_passed=false`, so the frozen two-failure
  AND condition is satisfied.
- The three launchers change only job/output names, architecture
  `D256/H8/L2`, and exact parameter assertion `4,539,888` relative to D64.
- Data, split, sidecar, JAPD objective, batch size, optimizer, learning-rate
  schedule, seed, evaluation cadence, checkpoint selection and scientific
  gates are unchanged.
- D64/H4/L1 constructs exactly `433,852` parameters; D256/H8/L2 constructs
  exactly `4,539,888`.
- Both architectures consume full `[B,16,*]`, use global axial noncausal
  mixing, emit `[B,16,16]` candidate scores, and select one `[B,16]` chain.
- No causal mask, GRU, selected-token feedback, serial target decoding,
  iteration, beam/tree/forest or multipath path exists in this branch.

## Profile checks

- JAPD complete path includes base vocab GEMM, FP32 Top16 and LSE,
  candidate/anchor gather, global head, argmax and final ID gather.
- Domino uses the same hidden states, target weight and base GEMM followed by
  the released eager correction head through token selection.
- Complete/incremental output parity and cached released-policy parity fail
  closed.
- The latency gate uses the worse of staged and standalone complete p50 ratios.
- Profile checkpoints must match the explicit architecture fields and load
  strictly.

## Verification

- Focused profile/JAPD tests: `26 passed` after the final delta.
- Independent broader review: `56 passed, 3 subtests passed`.
- Python compilation and all six related launchers' `bash -n` checks passed.
- A40 48 GB OOM risk is low; coarse parameter-count extrapolation remains below
  the 30-minute capacity/full-fit limits and profile has a 15-minute limit.

Reviewer trace: `.aris/traces/experiment-bridge/2026-08-10_run34/`.
