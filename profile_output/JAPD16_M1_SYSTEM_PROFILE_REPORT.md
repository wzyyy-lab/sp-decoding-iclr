# JAPD-16 M1 A40 Eager System Profile

## Status

- D64/H4/L1: **PASS** for the frozen M1 latency gate.
- D256/H8/L2: **PASS** for the same frozen M1 latency gate.
- This is an eager-to-eager head-path comparison, not an SGLang end-to-end
  throughput claim.

Authoritative D64 artifact:
`profile_output/japd16_eager_10167573.json` (A40, batch 1, full 16 positions,
1,000 measured repetitions).

Authoritative D256 artifact:
`profile_output/japd16_eager_d256_10167608.json` (A40, batch 1, full 16
positions, 1,000 measured repetitions).

## Complete-path latency

| System | Standalone p50 (ms) | Staged same-call p50 (ms) | p90 (ms) |
|---|---:|---:|---:|
| JAPD-16 D64 | 2.783232 | 2.846720 | 2.802848 |
| JAPD-16 D256 | 4.136960 | 4.210688 | 4.171776 |
| Released Domino eager | 4.226048 | 4.227072 | 4.228096 |

The gate uses the worse of the staged and standalone complete-path ratios:
`max(0.673450, 0.658590) = 0.673450`, well below the frozen `1.20x` ceiling.
For D256 the same conservative calculation is
`max(0.999271, 0.982013) = 0.999271`, also below the gate.

Both complete paths include the shared base vocabulary GEMM. JAPD additionally
includes FP32 Top16 and logsumexp, candidate/anchor gathers, the global
noncausal head, per-position argmax, and final candidate-ID gather. Domino
includes the released eager correction head through final token selection.

## JAPD staged breakdown

| Component | p50 (ms) | Share of staged total |
|---|---:|---:|
| Base vocabulary GEMM | 1.204224 | 42.3% |
| FP32 Top16 + logsumexp | 0.293888 | 10.3% |
| Candidate + anchor gather | 0.011264 | 0.4% |
| Global head + argmax | 1.337344 | 47.0% |
| Total | 2.846720 | 100% |

Mean component additivity error is `2.48e-9 ms` for JAPD and `1.90e-7 ms`
for Domino, closing the non-additive attribution issue in the earlier v1 run.

For D256, the staged global-head-plus-argmax p50 is `2.702336 ms` and the
complete p50 is `4.210688 ms`; its mean additivity error is `2.26e-8 ms`.

## Memory and invariants

- JAPD complete incremental peak allocation: `24,315,392` bytes.
- Domino complete incremental peak allocation: `5,491,712` bytes.
- JAPD parameters: `433,852`, or `0.0807%` of the `537,427,968`-parameter
  DFlash backbone reference.
- D256 parameters: `4,539,888`, or `0.8447%` of the same backbone reference.
- Full input geometry is `[1,16,2560]`; output score geometry is `[1,16,16]`.
- Zero-init JAPD exactly reproduces DFlash base Top1 at all 16 positions.
- Released Domino eager exactly reproduces the cached released policy.
- Complete and incremental token outputs match for both systems.

## Bottleneck interpretation

For D64, the global head and base vocabulary GEMM are comparable in cost. D256
roughly doubles global-head time but remains at parity with released Domino for
the entire complete eager path. Therefore M1's binding problem remains model
capacity/optimization rather than latency. This does not yet imply a 1.15x
SGLang end-to-end result; that later claim also requires accepted-length gain
and same-stack integration.

## Instrumentation changelog

- Added `scripts/profile_japd16_head.py` as the claim-scoped batch-1 A40 eager
  profiler.
- Added same-call CUDA event boundaries around every JAPD complete-path stage
  and around Domino's shared GEMM and released head.
- Added independent standalone complete callbacks and made the scientific gate
  use the worse complete p50 ratio.
- Added full16 shape, zero-init identity, released-policy replay, and
  complete-versus-incremental token checks.
- Added peak-allocation measurements for incremental and complete paths.
- Added `scripts/slurm/japd16_m1_profile.sbatch` for the reproducible A40 run.
- Parameterized the same profiler for the frozen D256/H8/L2 branch, added
  strict checkpoint/config matching, and added
  `scripts/slurm/japd16_m1_profile_d256.sbatch`.
- Added CPU regressions for exact D64/D256 parameter counts and profile
  checkpoint architecture mismatch fail-closed.
- The initial v1 artifact `profile_output/japd16_eager_10167551.json` is retained
  as a historical screen but is not authoritative for component attribution.

No production inference code was changed for this profile.
