# PGCF-003 / PGCF-005 Initial Results

## PGCF-005 fair eager A40 profile — PASS

Job `10166801`, A40, batch 1, full16, BF16 eager, 50 warmups and 1,000
timed iterations:

| Path | PGCF-16 p50 | Released Domino p50 | Ratio |
|---|---:|---:|---:|
| Complete base-vocab-to-token pipeline | 1.820672 ms | 4.224000 ms | 0.431030x |
| Incremental head path | 1.560576 ms | 3.887104 ms | 0.401475x |

The complete PGCF path includes base vocabulary GEMM, FP32 Top-16,
projected-table gather, global head, and argmax.  All cached-lattice,
released-policy, and complete/incremental token checks passed.  The derived
BF16 table is 77,791,232 bytes.  The preregistered `<=1.20x` development gate
passes with substantial margin.

## PGCF-003 first 512-block capacity attempt — PARTIAL / REMEDIATE

Job `10166802`, global PGCF-16, 4,000 curriculum steps:

- base EAL: `5.19140625`;
- released Domino EAL: `6.498046875`;
- pure base16 oracle: `11.021484375`;
- selected EAL: `10.7734375`;
- oracle-gap recovery: `95.7454%` — pass;
- harmed fraction: `0.390625%` — pass;
- all-supported candidate accuracy: `96.9781%` — below 99%;
- supported non-Top1 accuracy: `94.3914%` — below 97%.

The final checkpoints were still improving, so this is an optimization-horizon
failure rather than evidence against head capacity.  The run also exposed a
protocol issue: the independent supported-Domino-action witness cannot be
represented by only the first 10% of a combined curriculum run.  Remediation
therefore keeps the architecture fixed and separates:

1. an 8,000-step curriculum target-capacity witness, gated on the four target
   metrics; and
2. a 4,000-step teacher-only witness, gated on supported Domino-action
   reconstruction.

Both use exactly the same parallel global one-chain head and offline labels.
No held-out efficacy run is authorized until both witnesses pass.
