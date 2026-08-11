# Domino Correction-Head Profile

## Result

| Quantity | NVIDIA A40 result |
|---|---:|
| Released eager sequential correction | 3.7014 ms/block |
| Released optimized CUDA-graph/Triton correction | 2.1109 ms/block |
| Batched vocabulary projection with precomputed correction codes | 0.1743 ms/block |
| PLC encoder budget at a 0.8x optimized-Domino head target | 1.5145 ms/block |
| Released correction-head parameters | 50.823M |
| Released sequential-head estimated MACs (horizon 15) | 722.534M |
| Batched projection-only estimated MACs | 544.539M |

The real cached released path was reproduced with zero token mismatch by both the eager reference and released graph runner, and the batched projection returned exactly the same tokens when given the same correction codes. The batched value is not projected PLC performance because code prediction is excluded. It is the lower bound used to budget the new encoder. This record has a 15-token cached horizon (one base token plus 14 corrected positions); the production B16 path corrects 15 positions, so the final implementation must also be profiled at the production shape.

Existing end-to-end A800 baseline (`artifacts/baselines/qwen3_4b_10022436.json`, 96 samples) reports released Domino at 128.57 wall tokens/s and DFlash at 118.30 wall tokens/s. PLC must be measured through the same complete generation loop; isolated head latency alone cannot establish final acceleration.

## Instrumentation changelog

| File | Change type | What was added |
|---|---|---|
| `scripts/profile_domino_correction_head.py` | created | Real-record CUDA-event benchmark for released sequential correction and its batched-code projection floor |
| `scripts/slurm/profile_domino_correction_head_debug.sbatch` | created | Reproducible A40 debug launch |
| `profile_output/domino_correction_head_10158697.json` | generated | Structured raw profile result |
| `profile_output/domino_correction_head_10158719.json` | generated | Adds the released optimized CUDA-graph/Triton serving baseline |
