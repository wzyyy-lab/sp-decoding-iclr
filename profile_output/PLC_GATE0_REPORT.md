# PLC-Head Gate 0

## Production-shape result

The complete untrained PLC v1 architecture passed the fail-fast latency gate on an NVIDIA A40 (`10158737`). The benchmark uses batch 1, 15 corrected positions, Top-16, four modes, width 128, one width-256 FFN, and vocabulary size 151,936.

| Quantity | Result |
|---|---:|
| PLC trainable parameters | 317,952 |
| PLC active head parameters, including reused `W_h/W_out` | 39,868,928 |
| Released Domino active correction-head parameters | 50,823,168 |
| PLC / Domino active parameters | 0.7845x |
| PLC complete head, eager | 1.5244 ms/block |
| PLC complete head, CUDA Graph | 0.4260 ms/block |
| Released Domino complete head, CUDA Graph | 2.2560 ms/block |
| PLC / Domino graph latency | 0.1888x |
| Required Gate-0 limit | 1.8048 ms/block |
| Gate 0 | **PASS** |

The PLC timing includes input copies, full-vocabulary Top-16, projected lexical-code gathers, the local and global attention paths, route mixing, one full-vocabulary correction GEMM, base-logit addition, argmax, graph replay, and output clone. Random PLC weights are intentional because Gate 0 tests architecture cost before training.

The frozen projected lexical table occupies 38,895,616 bytes, versus 933,494,784 bytes for optimized Domino's precomputed token-to-GRU input table. Graph and eager PLC returned identical tokens for the profiled input.

## Interpretation

PLC v1 is not merely parameter-compatible with Domino: its measured correction path has substantial head-latency and memory headroom. This does not yet establish end-to-end acceleration because acceptance quality is untrained and target-verification frequency dominates the final result. The next hard gate is on-policy Domino imitation, followed by an EAL improvement stage and complete generation benchmark.

Raw result: `profile_output/plc_gate0_10158737.json`.
