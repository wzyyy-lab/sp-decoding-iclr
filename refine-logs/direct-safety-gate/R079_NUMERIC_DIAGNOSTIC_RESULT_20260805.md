# R079 Continuous-Numeric Diagnostic Result (2026-08-05)

## Sealed attempt identity

- Slurm job: `10137460`
- Wrapper SHA-256: `1bedcf8b3418ebff72378d0c02473b4fae9a2ba027e8fd42ad7939996b9fefcb`
- Diagnostic script SHA-256: `d7e3f0d763b35e997b50f532e2a57bc399df26c93c60a70af6b5e76a27f4083d`
- Test SHA-256: `841e214ba39640e305dfe15f340c2319e4bef485bd27219c559e8563a58b3ab2`
- Diagnostic source-closure SHA-256: `34d6f0c37caabf6039675b437ab708f2efe515fdb400bcbecc2fe1604ecf3fc3`
- Prior failed wrapper `4b2178a4...` was not resubmitted.

## Scheduler and stream evidence

- Slurm state: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:01:11`
- Allocation: one NVIDIA A40, 8 CPUs, 128 GiB memory
- Stdout: exactly one newline-terminated canonical-JSON line, 5,179 bytes
- Stdout SHA-256: `54515f7271937cc1fd8ddcda1c762a05868c2d27a36fff62bf6b5fccb2217b3f`
- Stderr: exactly zero bytes
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

The raw evidence is retained at:

- `artifacts/logs/pros-numdiag-10137460.out`
- `artifacts/logs/pros-numdiag-10137460.err`

## Aggregate result

- Protocol: `pros-gate-numeric-portability-diagnostic-v1`
- Numeric-policy ID: `pros-gate-cross-device-numeric-policy-pre-scan-v1`
- Status: `PASS`
- Device/runtime: NVIDIA A40, PyTorch `2.9.1+cu128`, CUDA `12.8`
- Real-input census: 12,686 fit + 1,600 checkpoint blocks
- Synthetic positive cases: 862
- Synthetic negative cases/rejections: 1,343 / 1,343
- Forbidden semantic operations executed: 0
- Field census: exactly 20 expected fields and 605,839,056 comparisons
- Across all 20 fields: zero envelope, cap, nonfinite, and range violations
- Maximum observed retained-mass absolute CPU/CUDA difference: `1.7787804864638864e-06`, accepted by the pre-sealed analytic interval/cap policy
- Maximum observed entropy absolute CPU/CUDA difference: `2.6242010209287514e-07`, below the pre-sealed `2^-17` envelope
- Exact-copy fields had zero exact mismatches. Expected continuous/rank-position representational mismatches were assessed by their pre-sealed operation-aware policies.

## Independent local parser check

A separate stdlib-only parser, without importing the diagnostic implementation, verified:

1. exactly one newline-terminated JSON record;
2. byte-for-byte canonical `sort_keys=True`, compact-separator serialization;
3. protocol, PASS status, input counts, zero forbidden operations, and full negative rejection;
4. the independently recomputed 20-field comparison census from `14,286 * 15` positions, 16 candidates, 64 state channels, and 2,560 hidden dimensions;
5. zero envelope/cap/nonfinite/range violations for every field; and
6. empty stderr.

The parser returned `INDEPENDENT_RESULT_CHECK=PASS` and the same stdout SHA-256.

## Interpretation and authorization boundary

This result diagnoses the frozen Direct feature path as cross-device portable under the predeclared operation-aware policy. It does **not** authorize editing production validators, materializing outcomes, running capacity/training, opening falsifier/validation/reserved data, or making an efficacy claim.

The next permitted action is a fresh independent result review. Any production numeric-policy v2 proposal must preserve the label-blind evidence, explicitly version and digest the policy, bind producer/materializer and independent auditor to that policy, re-seal the production source closure, and regain split/audit/CUDA authorization from the start.
