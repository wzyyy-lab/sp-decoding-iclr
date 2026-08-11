# R052 Amendment: Fair Eager Exact-Prefix Cycle Profile

**Date:** 2026-08-10  
**Parent:** `EXPERIMENT_PLAN_AMENDMENT_R051_20260810.md`

R051 passed the acceptance target with seed length four: clean batch-1 unsplit
EAL is `9.060131195`, versus clean Domino `7.295675413` and the historical
formal Domino value `7.239552964`.  R052 profiles the selected `s=4` mechanism
before any SGLang integration.

The first comparison is deliberately eager for both methods, as requested.  On
one A40 and batch one, profile three deterministic records at the p10, p50, and
p90 context-length quantiles.  Both cycles start from an already materialized
parallel DFlash hidden block and include the shared base-vocabulary GEMM.

```text
Domino non-common cycle:
  base vocab GEMM + released eager Domino correction + target verify 17 rows

R051 non-common cycle:
  four sequential target seed calls + base vocab GEMM
  + forced-prefix Fast-K64 suffix + target verify 13 rows
```

The shared DFlash backbone is excluded from both measured cycles and must be
reported as such.  Report component timings, complete-cycle p10/p50/p90, peak
GPU memory, and the exact common-path latency that would be needed to reach the
1.15x throughput target.  Do not add isolated kernel medians and call the sum a
measurement: the complete dependent callback is the primary timing authority.

Using same-evaluator output advances, the required complete-cycle time ratio is

```text
T_R051 / T_Domino <= ((9.060131195+1)/(7.295675413+1)) / 1.15.
```

Split/unsplit HF numerical parity remains a deployment correctness diagnostic.
The known three emitted-bonus mismatches prevent a lossless deployment claim,
but do not invalidate this bounded eager latency measurement.  SGLang
integration is authorized only after latency analysis identifies a plausible
path and the serving implementation reaches exact non-tie and emitted-bonus
parity.

