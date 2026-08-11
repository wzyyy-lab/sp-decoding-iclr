# Round 2 Review: Corrected Latency Budget

## Verdict

- **Method verdict:** PASS / READY FOR IMPLEMENTATION
- **Scope of pass:** implementation is authorized only through a fail-fast production-shape latency Gate 0; it does not presume an unfused PyTorch implementation is fast enough.

The corrected A40 comparison is:

- optimized CUDA-graph/Triton Domino head: `2.1109 ms/block`;
- complete PLC limit at `0.8x`: `1.6888 ms/block`;
- same-code batched vocabulary projection: `0.1743 ms/block`;
- remaining budget for Top-16, gather, encoder, routing, base-add, and argmax: about `1.5145 ms/block` at the cached 14-correction shape.

The pass survives this stricter budget subject to all of the following implementation constraints:

1. Reprofile the production shape of 15 corrected positions; the 14-position cached record is not a final system gate.
2. Keep exactly four slots, width 128, one global block, and FFN width 256. No second block, LoRA, or auxiliary deployed head.
3. Batch all local cross-attention across 15 positions; no Python position loop.
4. After training freezes, precompute the 128-D lexical projection of every `W_out[token]` row and gather it at inference. This adds about 39MB in BF16, far below the approximately 0.93GB GRU input-projection table removed from Domino graph mode.
5. Compute `W_h H` once and reuse it for both node features and final correction.
6. Do not compute a full-vocabulary softmax or log-sum-exp for features. Use only Top-16-normalized logit, margin, rank, entropy, and mass features.
7. Use fused SDPA/FFN and fuse or co-schedule node construction, routing, and delta projection where practical.
8. Execute one batched full-vocabulary GEMM followed immediately by base-add/argmax; never fall back to 15 GEMVs.
9. Capture the fixed B1/B16 head in a CUDA graph with no dynamic allocation, CPU synchronization, or host token handling.
10. Run Gate 0 before imitation training. Only a complete production-shape head at or below `1.6888 ms/block` may proceed.

If an implementation satisfying these constraints still misses `1.6888 ms`, the latency gate is failed; the budget must not be relaxed by comparing against eager Domino.
