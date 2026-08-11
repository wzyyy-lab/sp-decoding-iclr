# Problem Anchor: Lightweight Domino Replacement

- **Bottom-line problem:** “完全解决接受长度不高的问题，一定要超过 Domino，越高越好。”
- **Deployment constraint:** “这个新加的架构要足够轻量，要跟 Domino 的 GRU head 差不多，一定不能比它高出很多，可以稍微高一些；最后的端到端加速也要比 Domino 好很多。”
- **Must-solve bottleneck:** the released Domino backbone exposes much higher-quality alternatives than its greedy chain uses, but a draft-only selector must identify them without target-side search and without losing Domino's strong learned causal correction.
- **Non-goals:** no draft tree, beam sent to target, multi-branch verification, early target layers, extra target forward, or multi-pass Jacobi refinement in the deployed path.
- **Success condition:** on the exact Qwen3-4B B16 prompt-balanced evaluation, one frozen single-chain method reaches at least 1.15x released Domino EAL on validation-select and validation-gate and at least 1.15x Domino end-to-end TPS after comparable SGLang integration. Start near a 2%-of-draft encoder budget, but allow evidence-driven capacity/latency increases when the final EAL/TPS pair remains favorable.
