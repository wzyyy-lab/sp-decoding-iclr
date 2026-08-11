# R052 Exact-Prefix Eager Profile

## Outcome

**FAIL-SYSTEM.** R051's four-token exact target prefix raises clean fixed-B16 EAL from Domino's 7.295675 to 9.060131, but the four serial target-model seed calls dominate latency. This route is closed for deployment even though its acceptance target passes.

The authoritative raw artifact is `r052_exact_prefix_eager_10165070.json` (job 10165070, NVIDIA A40, request batch size 1, HF SDPA eager).

## Fair complete-cycle comparison

Both paths include the base vocabulary GEMM, proposal head, and target verifier. R051 additionally includes its four serial target seed calls. The shared DFlash parallel backbone and serving scheduler are excluded identically.

| Metric | Domino | R051 s4 |
|---|---:|---:|
| Clean fixed-B16 EAL | 7.295675 | 9.060131 |
| Output advance ratio | 1.0000x | 1.212696x |
| Median-record complete non-common p50 | 39.1941 ms | 160.4332 ms |
| Complete-cycle time ratio | 1.0000x | 4.093297x |
| Projected throughput ratio | 1.0000x | 0.296264x |

For a 1.15x throughput target, R051 may use at most a 1.054518x time ratio. It misses this bound by a wide margin. Adding an inferred 2184.6 ms shared path would be required to hide the measured difference, so omitted common work cannot plausibly reverse the decision.

The three profiled contexts (lengths 54, 163, and 296) all fail: projected throughput is 0.3015x, 0.2943x, and 0.2990x respectively.

## Bottleneck at median context length 163

| Component | p50 latency |
|---|---:|
| Shared base vocabulary GEMM | 1.1909 ms |
| Domino eager head, excluding base GEMM | 3.9240 ms |
| R051 Fast-K64 head, excluding base GEMM | 4.6612 ms |
| Domino target verifier, 17 rows | 35.0310 ms |
| R051 four-call serial target seed chain | 120.7808 ms |
| R051 final target verifier, 13 rows | 34.9947 ms |

The extra proposal head cost is only about 0.74 ms. The system failure is caused by rereading the full target model four times serially, not by the lightweight correction head.

Peak allocated/reserved memory was 8.94/9.00 GiB on a 47.40 GiB A40, so memory is not the blocker.

## Correctness and scope

This is a bounded HF eager latency result, not an SGLang throughput claim. R051's HF split path still matches only 300/303 emitted bonus tokens on full-accept blocks, so lossless deployment is independently disallowed. The clean unsplit evaluator remains the sole acceptance authority.

The next architecture must retain one target-model verification invocation per cycle; serial exact-target seeds are no longer eligible.

## Instrumentation changelog

| File | Change | Purpose |
|---|---|---|
| `scripts/profile_r052_exact_prefix.py` | Created | Times both complete non-common eager cycles and their components with CUDA events; records context quantiles, memory, geometry assertions, and throughput gates. |
| `tests/test_r052_profile.py` | Created | Checks output/time-ratio algebra, context selection, and component geometry helpers. |
| `scripts/slurm/r052_exact_prefix_eager_profile.sbatch` | Created | Reproducible A40 batch-1 launcher with fixed assets, reports, warmup, and repeat count. |
| `refine-logs/glcs-v2-opd/EXPERIMENT_PLAN_AMENDMENT_R052_20260810.md` | Created | Pre-registers comparison scope and hard system gate. |
| `refine-logs/glcs-v2-opd/EXPERIMENT_CODE_REVIEW_R052_20260810.md` | Created | Records fresh same-family ARIS review and scope limitations. |

