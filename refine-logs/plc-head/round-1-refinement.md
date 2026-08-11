# PLC-Head v1: Parallel Lattice Correction

## Locked Problem and Deployment Contract

The primary objective is to raise B16 accepted length beyond released Domino while also moving the end-to-end speed/quality frontier. The deployed method is **draft-side only** and always emits exactly one chain for exactly one ordinary target verification. It uses no target-layer probe, target-side reranking, tree/beam verification, extra target forward, autoregressive student feedback, or iterative refinement.

For the released Qwen3-4B-Domino-b16 configuration (`shift_label=true`, `pure_draft_prefix_len=1`), the exact path is:

1. Run the released Domino/DFlash parallel backbone once to obtain `H[0:16]` and base logits.
2. Emit position 0 with the existing base top-1 rule.
3. For positions 1--15, construct a Top-16 base-logit lattice without selecting a token path.
4. Predict all 15 correction codes `delta[1:15] in R^256` in one parallel head call.
5. Compute `base_logits + W_out SiLU(W_h H + delta)` for all 15 positions with one batched full-vocabulary projection and take 15 argmaxes.
6. Concatenate the base-prefix token and 15 corrected tokens into one draft chain and call the unchanged target verifier once.

No released Domino token is used as an inference feature: that token would require executing the GRU and would invalidate the speed claim.

## Why This Architecture

Released Domino's correction is

`b_i = W_out SiLU(W_h h_i + W_s s_i)`,

where only the 256-dimensional `W_s s_i` affects the final logits even though `s_i` is a 1024-dimensional sequential GRU state. PLC predicts this 256-dimensional correction sufficient statistic directly from the parallel hidden states and their candidate lattice. It therefore preserves the released 39.55M-parameter `W_h/W_out` lexical correction basis while removing the recurrent state and its 15-position dependency.

This differs fundamentally from the failed GCLS route. GCLS learned an independent scorer from pure-DFlash/random candidate features and recovered only about +0.285 EAL. PLC starts from the released Domino backbone and correction basis, learns from the exact on-policy released GRU trajectory, and is required to recover Domino quality before it is allowed to optimize target acceptance.

## Minimal Mode-Preserving Encoder

The first implementation uses one width-128 global block and four persistent mode slots per corrected position.

### Candidate nodes

For each position `i` and base Top-16 candidate `v_ik`, form a width-128 node from:

- a learned projection of the frozen released lexical code row `W_out[v_ik] in R^256`;
- a learned projection of the reusable released hidden branch `z_i = W_h h_i in R^256`;
- standardized base logit, probability, top-1 margin, cumulative Top-16 mass, and rank embedding;
- position embedding.

The node construction is fully parallel. It gathers 240 rows from the already-resident `W_out` table and never invokes target computation.

### Four local modes

At every position, four learned queries cross-attend to its 16 nodes, producing `m_iq in R^128`, `q=1..4`. The slots are not pooled. All 60 slots plus one anchor/route slot pass through a single bidirectional width-128 self-attention block with a width-256 feed-forward layer.

### Shared block route

The updated route slot `r` is shared by the whole block. It chooses among the four modes separately at every position:

`alpha_iq = softmax_q(r^T U m_iq)`,

`delta_i = sum_q alpha_iq A m_iq`,

where `U in R^(128x128)` and `A in R^(256x128)`. A single shared route state encourages a coherent block-level lexical/semantic mode, while position-specific weights avoid averaging incompatible candidate modes.

There is no second global block, recurrent feedback, or vocabulary LoRA in v1. A second block may be considered only if the exact imitation gate fails and the measured complete-head latency remains below budget; it is not part of the initial search space.

## Exact Size and Compute Budget

Released Domino correction head:

- GRU: 11.010M parameters;
- `W_h/W_s` plus `W_out`: 39.813M parameters;
- total: 50.823M parameters.

PLC retains `W_h` and `W_out` (39.551M) and replaces the GRU plus `W_s` with a lattice encoder budgeted at no more than 0.60M. The implemented v1 encoder has 317,952 trainable parameters, making the exact active head 39.869M parameters: 78.45% of Domino and below the hard 45M ceiling. Frozen parameters still count as active because they are read at inference.

The slot encoder is constrained to one small 61-token block. The dominant vocabulary operation is one `[15,256] x [256,V]` GEMM rather than 15 dependent GEMVs. Speed will not be inferred from parameter count or eager FLOPs: the decisive comparison is against Domino's optimized CUDA-graph/Triton path, including candidate Top-K and gather, lattice encoding, full-vocabulary correction, and argmax.

The first real A40 graph-runner profile (`10158719`) measured the released optimized head at `2.1109 ms` on a cached horizon containing 14 corrected positions, versus `3.7014 ms` for the eager path. The same-code batched vocabulary projection took `0.1743 ms`. Thus the `0.8x` gate leaves approximately `1.5145 ms` for Top-16, gathers, and the entire PLC encoder at this shape. This replaces the earlier overly generous eager-only budget; the final kernel must also pass at the production 15-correction shape.

The inference implementation is correspondingly fixed: batch all 15 local attentions, precompute the frozen 128-D lexical projection table after training, reuse `W_h H`, use no full-vocabulary probability feature, fuse the single SDPA/FFN block, execute exactly one vocabulary GEMM, and capture the fixed B1/B16 path in a CUDA graph. The preprojected lexical table costs about 39MB in BF16, much less than the approximately 0.93GB token-to-GRU input table used by optimized Domino.

Gate 0 subsequently passed on the production 15-correction shape (`10158737`): the complete PLC CUDA-graph path took `0.4260 ms/block`, while the same-shape released Domino graph runner took `2.2560 ms/block` on the same A40. PLC used only `0.1888x` Domino head latency, well below the `0.8x` limit. This benchmark includes Top-16, gather, both attention stages, routing, the vocabulary GEMM, base-add/argmax, graph input copies, and output clone; it establishes implementation feasibility, not acceptance quality.

## Two-Stage Training

### Stage 1: exact on-policy Domino distillation

For every real training anchor, run the frozen released B16 Domino path under the same inference contract. Record the base Top-16 lattice and, for positions 1--15:

- the on-policy teacher prefix and teacher token;
- `teacher_delta_i = W_s s_i` before SiLU;
- teacher corrected logits or their exact full-vocabulary CE targets;
- whether the teacher token matches the target token under a clean target-prefix reach mask.

Train PLC using full-vocabulary teacher CE as the primary loss and normalized correction-code regression as a warm-start auxiliary. The candidate lattice always comes from the deployable base path; clean-gold-prefix GRU states are not teacher inputs. Code loss decays after token imitation stabilizes.

The stop/go gate is strict: token agreement must be near-saturated and validation EAL must fall within 0.10 of released Domino, with within 0.05 preferred. Failure triggers architecture/data repair only; no larger corpus or target-improvement stage is allowed.

### Stage 2: acceptance-frontier improvement

Starting only from a passed Stage-1 checkpoint, optimize the same full-vocabulary output with

`L_improve = sum_i w_i CE(target_i) + lambda_keep sum_i w_i 1[teacher_i=target_i] CE(teacher_i)`.

`w_i` is a detached clean-prefix reach times continuation-utility weight. When Domino is correct, the preservation term makes regression expensive; when Domino is wrong, target CE teaches a replacement from the Top-16-informed representation. The decayed code-regression auxiliary may remain for stability, but there is no separate collection of candidate KL, expected-prefix, and asymmetric penalty hyperparameters.

Concretely, for the student's probability of the clean target token `p_i = p_theta(g_i)`, use

`r_i = stopgrad(prod_{j<i} p_j)`,

`c_i = stopgrad(1 + sum_{t>i} prod_{j=i+1}^t p_j)`,

`w_i = r_i c_i`.

This continuously emphasizes reachable early errors and their lost continuation without an unstable hard-argmax reach mask.

Training uses full-vocabulary logits so the objective matches deployment. If memory requires sampled training, the support must include teacher, target, the full base Top-16, and global hardest negatives, followed by a mandatory full-vocabulary confirmation run.

The existing four Domino-materialized OPB parts provide 25K prompts and 199,800 blocks for the first real run. Expansion toward the 99K-prompt source corpus is conditional on a positive learning curve and requires collecting the missing on-policy Domino states; scale is not a substitute for passing imitation.

## Hard Gates

1. **Imitation gate:** distilled PLC EAL within 0.10 of released Domino on validation-select; within 0.05 preferred. If it fails, do not run Stage 2.
2. **Primary acceptance gate:** one checkpoint selected on validation-select must exceed Domino by at least +0.50 EAL; +1.0 is the preferred research target. The frozen checkpoint must also exceed Domino on validation-gate.
3. **Size gate:** the new trainable PLC encoder is at most about 2% of the 537.427M-parameter headless DFlash draft model; report the complete active head separately.
4. **Prototype latency diagnostic:** compare eager PLC to eager Domino and graph PLC to graph Domino separately. A development head up to roughly 1.2x the corresponding Domino head remains admissible if it is small relative to the draft and does not erase projected system gains.
5. **Acceptance gate:** frozen EAL at least 1.15x Domino on validation-select and validation-gate; the measured Top-16 ceiling is `7.240 -> 10.254` (+3.015), so the optimization target remains much larger than a +0.3 patch.
6. **End-to-end gate:** after comparable SGLang integration, same-hardware, same-prompt, same-generation-length TPS at least 1.15x Domino. Compare both wall and decode-only throughput.

An EAL-only win is rejected. A speed-only result that does not exceed Domino acceptance is also rejected.

## Focused Validation

The development sequence is deliberately narrow:

1. Implement the complete fixed-shape head and run production-shape latency Gate 0 before training. Report eager-to-eager and graph-to-graph comparisons and reject only if the added head cost makes a 1.15x final throughput win implausible.
2. Overfit 1,024 blocks to verify that the parallel representation can reproduce on-policy teacher tokens and codes.
3. Run the 25K-prompt Stage-1 imitation experiment and apply the imitation gate.
4. Run Stage 2 with a small preservation-weight sweep, select once on validation-select, and open validation-gate once.
5. Measure full end-to-end latency against optimized graph-mode Domino before scaling data.

In addition to target EAL, Stage 1 reports first-corrected-position teacher-token agreement, prompt-balanced teacher-path longest-common-prefix, and all-position teacher-token agreement. These are diagnostics; the hard imitation decision remains actual target EAL within 0.10 of Domino.

Only two mechanism deletions are initially necessary: replace four modes with one expected node, and remove on-policy distillation. They isolate mode preservation and inherited causal knowledge without an ablation zoo.

## Highest-Risk Assumption

A prefix-free lattice may not contain enough information to reproduce the acceptance-relevant part of a causal on-policy GRU state. The 1,024-block and 25K-prompt imitation gates test this directly. If they fail under the 0.60M encoder budget, the method is invalid in its lightweight form; target-side search or recurrent student feedback will not be smuggled back into deployment.
