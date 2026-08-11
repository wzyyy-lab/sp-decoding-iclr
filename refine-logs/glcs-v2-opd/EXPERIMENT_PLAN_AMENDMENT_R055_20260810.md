# R055 Amendment: Fixed-Shape Padded Beam Forest

**Parent:** `EXPERIMENT_PLAN_AMENDMENT_R053_20260810_072850.md`  
**Trigger:** R053 job 10165201 passed the acceptance gate at N64
(`8.483722`) but failed the current eager system gate because an unfused W16
beam plus traversal cost `18.5298 ms`.

## Claim and scope

R055 tests one claim only: the accuracy-positive R053 multipath mechanism can
be expressed as a fixed CUDA-graphable forest whose complete-cycle latency is
low enough to justify SGLang integration.  It adds no trainable parameters and
does not reopen frozen selectors, OPB12K distillation, serial target seeds, or
repair-comb templates.

R054 repair-comb is closed.  Its gold-optimal fixed-template capacity recovers
only 65.62% of the one-repair reward with 47 extra nodes, below the required
88.67%, and is dominated by R053's measured N64 accuracy.

## Fixed forest geometry

Evaluate beam widths `W={4,8,16}`.  Each forest has

\[
N_W = 1 + 16W \in \{65,129,257\}
\]

rows: one shared anchor plus `W` independent 16-token chains.  Every chain row
attends to the cached target prefix, the shared anchor, itself, and only its own
earlier chain rows.  Position IDs are `prefix_length+1..+16` for every chain.

The frozen Fast-K64/Domino beam uses DFlash Top-15 plus the protected Fast trunk
as its fixed K16 branch support.  Beam, input layout, attention mask, position
IDs and parent metadata all have fixed shapes.  There is no token deduplication,
CPU trie construction, dynamic mask, or GPU-to-host decision in the timed path.

All paths are evaluated independently.  The anchor posterior is compared with
each path's first token; later tokens are compared with the preceding row's
posterior on that same path.  The verifier chooses the path with the longest
accepted prefix, with a deterministic lowest-index tie break.  It must never
choose one duplicate sibling and discard the other before their continuations
are compared.  The last accepted row supplies the next target token; a fully
accepted path uses its row-15 posterior as the bonus.

## Accuracy authority

The sole accuracy authority remains an unconditional 17-step batch-1
`qlen=1` target continuation from the clean cached prefix.  For every W report:

- structural full-pool EAL against that continuation;
- actual HF forest self-acceptance as a diagnostic;
- actual selected/emitted path's common-prefix EAL against clean authority;
- prompt-balanced chat/code/math EAL;
- paired gain/loss blocks and tokens versus same-job clean Domino;
- stable non-tie selected-path parity and full-accept bonus parity.

HF SDPA parity is diagnostic for this bounded experiment.  It cannot authorize
a lossless deployment claim; conditional SGLang must pass exact stable non-tie
selected-path and emitted-bonus parity.

## Fair latency contract

On the same A40 job and median context, remeasure Domino and each W.  Both
complete non-common cycles include:

- base vocabulary GEMM;
- proposal head/beam;
- target verifier including LM head;
- full-vocabulary argmax and acceptance/traversal.

The shared DFlash backbone and scheduler are excluded identically.  Forest
fill/mask copies are included even when static; paged-KV pointer commit may be
reported separately as an optimistic exclusion until SGLang.

For clean Domino EAL `E_D` and complete time `T_D`, a forest with EAL `E_W`
passes the development gate only if

\[
\frac{(E_W+1)/T_W}{(E_D+1)/T_D} \ge 1.20.
\]

The final SGLang target remains a 95% CI lower bound of at least `1.15x`.
Using R053's diagnostic `E_D=7.285471`, `T_D=38.7517 ms`, the final 1.15x
complete-cycle ceilings are approximately:

| EAL | Maximum complete time |
|---:|---:|
| 8.325486 | 37.93 ms |
| 8.483722 | 38.57 ms |
| 9.129130 | 41.20 ms |

The corresponding 1.20x development ceilings are approximately 36.35, 36.96,
and 39.47 ms.

## Execution order and hard routing

1. **Graph-beam gate.** Profile W4/W8/W16 beam-only eager and CUDA-graph
   latency on the R053 median record.  Exact graph/eager beam and protected
   trunk token parity are mandatory.  This is a system diagnostic, not an
   accuracy result.
2. **Small mechanics/latency smoke.** Run actual W4/W8/W16 forests on a fixed
   balanced subset.  Require correct mask/row/bonus indexing, no OOM, and
   same-job component timings before the full run.
3. **Full validation Pareto.** Evaluate all three widths on the complete 147
   prompts / 1,175 blocks.  Select the smallest W satisfying all of:
   - actual clean EAL `>=8.325485909`;
   - no chat/code/math regression versus clean Domino;
   - optimistic graph complete-cycle throughput `>=1.20x` Domino.
4. If no W passes, close fixed forest.  Do not rescue it with a learned
   selector, post-hoc threshold, or more beam widths.
5. If one passes, freeze the smallest passing W and implement SGLang paged-KV
   tree attention.  Final output parity and end-to-end throughput decide the
   deployment claim.

An EAL near 9 is desirable but never compensates for failing the latency gate.

## Conditional late-pruning route

Target-informed pruning at L24/L32 is blocked unless W8 or W16 passes accuracy
but fails solely because N129/N257 target-layer latency exceeds the system
budget.  If opened, it is a zero-parameter probe only: final RMSNorm plus
gathered child-token LM-head scores prune to a protected prefix-closed N32/N64
late tree.  Any depth whose ideal split cycle misses `1.20x` or whose clean EAL
falls below `8.325486` is closed without tuned-lens training.

## Claim boundary

R055 can support only a bounded development claim on Qwen3-4B, fixed B16,
batch 1, A40/HF graph execution.  Generalization, lossless serving and the
paper's final throughput claim require a frozen SGLang implementation and
independent evaluation after W is selected.
