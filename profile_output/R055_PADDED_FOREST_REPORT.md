# R055 Fixed Padded-Forest Pareto

R055 replaces the dynamic R053 trie and host traversal with one shared anchor
plus fixed independent 16-token chains.  The full 147-prompt / 1,175-block
development run on an NVIDIA A40 selected W8/N129 as the only joint
accuracy/system Pareto point.

| Variant | Clean EAL | Delta vs Domino | Complete cycle p50 | Projected TPS vs Domino | Joint gate |
|---|---:|---:|---:|---:|---|
| Released Domino | 7.285471 | -- | 39.551487 ms | 1.000x | baseline |
| W4/N65 | 8.279640 | +0.994169 | 34.445311 ms | 1.282010x | fail accuracy |
| **W8/N129** | **8.667396** | **+1.381924** | **35.570177 ms** | **1.292935x** | **pass** |
| W16/N257 | 9.051749 | +1.766278 | 53.336065 ms | 0.896667x | fail latency |

W8 improves chat/code/math EAL by `+0.833705/+1.505102/+1.787500` and has
strict canonical output advance `9.634232`, versus Domino `8.285471`.  It gains
1,709 draft tokens and loses 87 across paired blocks (net `+1,622`).  Peak
allocated memory was 9.093 GiB on a 47.404 GiB A40.

The profile includes the base vocabulary GEMM, CUDA-graphed Fast-K64 beam,
static token and mask materialization, one target forest forward, full-vocab
argmax, all-path traversal, and bonus selection.  It remains an HF SDPA
development projection, not an SGLang end-to-end measurement.

Deployment is still blocked on exact SGLang branch/output/bonus parity and a
fair end-to-end throughput comparison against an in-engine released-Domino
baseline.  The frozen final target is a prompt-bootstrap 95% CI lower bound of
at least `1.15x`.

Primary artifact:
`artifacts/analysis/r055_padded_forest_full_10165728.json`.
