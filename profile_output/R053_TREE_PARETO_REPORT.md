# R053 One-Pass Target-Tree Pareto

## Outcome

**PASS-ACCURACY / FAIL-CURRENT-EAGER-SYSTEM.**  The N64 one-pass target tree is
the first route in this project to clear the main acceptance target by a
material margin: clean autoregressive EAL is `8.483722`, versus `7.285471` for
the same-geometry Domino control and the preregistered gate `8.325486`.
However, the current unfused eager beam and traversal make the complete cycle
too slow (`0.8456x` projected Domino throughput).  This result supports the
multipath mechanism but not an end-to-end or lossless deployment claim.

The authoritative artifact is
`artifacts/analysis/r053_tree_budget_pareto_10165201.json` (job 10165201,
NVIDIA A40, request batch size 1, 1,175 fixed validation-select blocks).

## Acceptance Pareto

The accuracy authority is an unconditional 17-step, batch-1, `qlen=1` target
continuation.  The reported deployable column is the common prefix between the
actual HF tree-selected output and that clean continuation; raw tree
self-acceptance is not used as the score.

| Rows including anchor | Actual clean-prefix EAL | Delta vs clean Domino | Relative EAL | Draft-only simulation | Gold-aware structural upper bound |
|---:|---:|---:|---:|---:|---:|
| 17 | 7.262512 | -0.022959 | -0.32% | 7.322036 | 7.322036 |
| 24 | 7.578110 | +0.292638 | +4.02% | 7.643586 | 8.803450 |
| 32 | 7.829325 | +0.543853 | +7.47% | 7.897352 | 9.125729 |
| 48 | 8.202867 | +0.917396 | +12.59% | 8.278547 | 9.129130 |
| 64 | **8.483722** | **+1.198251** | **+16.45%** | 8.574708 | 9.129130 |

At N64, 316 blocks improve, 13 regress, and the net gain is 1,406 accepted
tokens.  Domain EAL gains versus the clean Domino control are `+0.66964` chat,
`+1.25000` code, and `+1.65500` math.  Relative to the historical Domino EAL
`7.239553`, N64 gains `+1.244169`; its `(EAL+1)` output-advance ratio is
`1.1510x`.

The full W16 path pool has structural EAL `9.129130`.  Its unique prefix trie
uses 99 nodes at p50, 101.1 on average, 152 at p95, and at most 212.  This is
important system headroom, not a deployable score: R053 did not run the actual
full tree through the verifier on every block.

## Current eager system result

The comparison includes base vocabulary GEMM, proposal head, one target
verification, full-vocabulary argmax, and tree traversal on both applicable
paths.  It excludes the shared DFlash backbone and scheduler identically.

| Median-context component | p50 latency |
|---|---:|
| Base vocabulary GEMM | 1.1909 ms |
| W16 Fast beam, excluding base GEMM | 16.9257 ms |
| N64 target-tree forward | 33.6379 ms |
| N64 argmax + traversal | 1.6041 ms |
| N64 optimistic complete cycle | 52.4564 ms |
| Domino complete non-common cycle | 38.7517 ms |

N64's clean output-advance ratio is `1.144621x`, but its complete-cycle time
ratio is `1.353654x`, giving only `0.845579x` projected throughput.  To reach
the user's final `1.15x` throughput target, the N64 complete cycle must be at
most `38.5705 ms`.  Holding the measured base GEMM and target-tree forward
fixed leaves only `3.7417 ms` for beam plus traversal, versus `18.5298 ms`
today.

If an actual full-W16 tree retains approximately the structural `9.129130`
EAL, its output ratio would be `1.222517x` and the corresponding complete-cycle
limit would relax to `41.1954 ms`; using the current N64 target latency as an
optimistic placeholder leaves `6.3666 ms` for beam plus traversal.  The next
experiment must measure, rather than assume, the larger-tree target cost.

## Correctness and claim boundary

- Fast-K64 trunk identity, K16 support, and tree-not-below-trunk controls all
  pass; N17 structurally reproduces the trunk.
- HF tree output differs from the clean `qlen=1` authority on 36/1,175 N64
  blocks, and full-accept bonus parity is 394/398.  These differences reduce
  the reported clean-prefix EAL rather than being hidden.
- No lossless deployment claim is allowed from HF SDPA.  A conditional SGLang
  implementation still needs selected-branch stable non-tie parity and exact
  emitted bonus parity.
- This is one development split.  It supports a bounded system-optimization
  phase, not a generalization or final-paper claim.

## Next bounded system test

1. CUDA-graph the fixed-shape W4/W8/W16 beam and remove all GPU-to-host control
   flow; report graph/eager token parity and measured p50.
2. Compare an actual larger/full W16 trie with a fixed-shape padded beam forest.
   The forest removes data-dependent trie packing entirely; select the smallest
   width/geometry that preserves EAL while meeting the complete-cycle budget.
3. Only a positive acceptance/latency Pareto proceeds to SGLang tree attention
   and its lossless parity gate.  Do not return to frozen selector or OPB12K
   loss/parameter sweeps.

## Instrumentation changelog

| File | Change | Purpose |
|---|---|---|
| `src/sph/r053_tree.py` | Added Fast-K64 protected beam, draft-only prefix-closed allocator, tree packing and fixed-shape traversal | Construct and verify a bounded multipath proposal without extra target calls. |
| `scripts/evaluate_r053_tree_budget.py` | Added clean autoregressive authority, all-budget actual target forwards, accuracy/system gates and component profiling | Produce the claim-bearing R053 Pareto. |
| `tests/test_r053_tree.py` | Added beam/trie/mask/traversal/oracle regressions | Fix tree geometry and acceptance semantics. |
| `scripts/slurm/r053_tree_budget_smoke.sbatch` | Added 4-block A40 mechanics launcher | Catch dtype, geometry and memory failures before the full run. |
| `scripts/slurm/r053_tree_budget_pareto.sbatch` | Added frozen 1,175-block A40 launcher | Reproduce the full accuracy and eager-profile result. |
| `scripts/profile_r053_beam_graph.py` | Added fixed-shape beam CUDA-graph probe | Test whether the measured system bottleneck can fit the N64/full-tree head budget. |
| `scripts/slurm/r053_beam_graph_profile.sbatch` | Added reproducible A40 graph-profile launcher | Run the first post-R053 system optimization gate. |
