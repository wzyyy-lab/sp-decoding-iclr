# R053 Experiment Code Review

**Verdict:** GO to A40 smoke/full fixed evaluation  
**Review independence:** same-family / provisional  
**Reviewer role:** fresh ARIS experiment-code review, read-only

The reviewer found two initial authority blockers, both fixed before GO:

1. Clean continuation is now generated unconditionally with 17 batch-1 `qlen=1` target calls for every block; the teacher-path pass is diagnostic only.
2. Every node budget now performs a real 4-D tree-mask target forward on every block. Gate EAL is the common prefix between the actually selected tree path and clean AR, rather than structural simulation.

Final checks passed:

- Fast-K64 trunk exactly matches the existing implementation; fixed K16 branch support is canonical DFlash Top15 plus the protected trunk.
- Ordinary cumulative beam pruning and `gamma=.75` max-descendant draft-only tree allocation are correctly isolated from gold.
- N17 retains anchor plus all 16 trunk tokens; tree masks, depth/RoPE positions, parent geometry, KV rows and bonus row are aligned.
- Full-pool, hindsight, structural simulation, raw HF self-acceptance and deployable actual clean-prefix EAL are separately reported.
- Domino and tree complete timings both include base GEMM, head/beam, target forward, full-vocabulary argmax and acceptance/traversal.
- Throughput uses `(EAL+1)` and the pre-registered 1.20x development gate.
- 22 focused tests, `py_compile`, and both Slurm `bash -n` checks pass.

The HF emitted-token/bonus parity remains diagnostic in R053, per the parent research review. `lossless_deployment_claim_allowed=false`; stable non-tie and bonus parity are hard gates only for the conditional SGLang implementation.

Non-blocking profile scope: shared DFlash backbone/scheduler is excluded identically; dynamic trie packing and paged-KV pointer commit are an explicit optimistic zero-cost bound. Peak allocated/reserved memory is recorded.

## Post-review GPU smoke correction

The first four-block smoke reached the real Qwen3 SDPA call and failed before producing metrics because Transformers 4.57 requires a 4-D attention bias to match the BF16 query dtype. `pack_tree_tensors` now takes an explicit `mask_dtype`, and both evaluator call sites bind it to `target_weight.dtype`. This is a mechanics-only correction; tree allocation, accuracy authority and decision gates are unchanged.
