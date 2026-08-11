# Research Proposal: PLC-Head — Parallel Lattice-to-Correction Distillation

## Problem Anchor

- **Bottom-line problem:** “完全解决接受长度不高的问题，一定要超过 Domino，越高越好。”
- **Deployment constraint:** “这个新加的架构要足够轻量，要跟 Domino 的 GRU head 差不多，一定不能比它高出很多，可以稍微高一些；最后的端到端加速也要比 Domino 好很多。”
- **Must-solve bottleneck:** the released Domino backbone exposes much higher-quality alternatives than its greedy chain uses, but a draft-only selector must identify them without target-side search and without losing Domino's strong learned causal correction.
- **Non-goals:** no draft tree, beam sent to target, multi-branch verification, early target layers, extra target forward, or multi-pass Jacobi refinement in the deployed path.
- **Success condition:** on the exact Qwen3-4B B16 prompt-balanced evaluation, one frozen single-chain method exceeds released Domino on validation-select and validation-gate, while its active head is at most approximately Domino-head scale and same-hardware end-to-end decode throughput is materially higher.

## Technical Gap

Released Domino obtains `5.93853 -> 7.01579` EAL from its causal correction, so a new frozen selector cannot discard that learned mechanism and restart from the backbone. The local Top-16/beam diagnostics show substantial candidate headroom, but the earlier small GCLS learns only a weak correction because it has no released-Domino teacher, no Domino-trained lexical code, and no way to preserve the strong correction at initialization. Target-side multi-path scoring proves identifiability is the bottleneck, but violates deployment cost.

Domino's deploy-time correction is also an opportunity. Its bias has the form

`W_out SiLU(W_h h_i + W_s s_i)`,

where `s_i` is a 1024-dimensional GRU state. The GRU and projection together contain about 50.82M parameters and the correction is launched sequentially for the draft positions. The useful object is not the full GRU state; only the 256-dimensional projected code `W_s s_i` affects logits.

## Method Thesis

Replace Domino's sequential GRU with one mode-preserving, block-parallel predictor of the 256-dimensional correction code, initialize/reuse the released Domino lexical projection, distill the released correction before acceptance-aware target fine-tuning, and keep ordinary single-chain target verification.

This is the smallest adequate intervention because it preserves the strongest released component, removes the only head-side sequential dependency, and adds one compact trainable encoder rather than another decoder or verifier.

## Contribution Focus

- **Dominant contribution:** parallel correction-code distillation: compress a sequential causal correction teacher into a one-pass candidate-lattice student whose inference is fully block parallel.
- **Supporting contribution:** acceptance-frontier fine-tuning of the same student; this is a training stage, not a second architecture.
- **Explicit non-contributions:** target early exit, tree verification, iterative refinement, a second full drafter, or a separate safety model.

## Proposed Method

### Complexity Budget

- Reuse the released Domino parallel backbone, `W_h` slice, and `W_out` vocabulary projection.
- Remove the 11.01M-parameter GRU and its 0.262M state-to-code slice from deployment.
- Add at most 1M parameters for the mode-slot lattice encoder in the first implementation.
- Optional vocabulary-projection LoRA is forbidden until the frozen-projection student passes the main development EAL gate; if opened, active head parameters must remain below 45M.
- Deployment has one draft forward, one parallel head forward, one chain, and one target verification.

### System Overview

```text
released Domino parallel backbone
  -> H[1:L], base_logits[1:L,V], base Top-K candidates
  -> candidate nodes using released W_out[token] lexical codes
  -> 4 mode slots per position (candidate-local cross-attention)
  -> two D=128 global blocks over 4L slots + anchor slot
  -> delta_code[1:L,256] in parallel
  -> W_out SiLU(W_h H + delta_code), one batched vocabulary GEMM
  -> base + correction argmax for all positions
  -> exactly one draft chain -> ordinary target verification
```

### Mode-Preserving Lattice Encoder

For base Top-16 candidate `v_{ik}`, construct a node from:

- the released Domino output row `W_out[v_{ik}]` (256 dimensions, reused lexical knowledge);
- standardized base logit, margin, probability mass, and rank;
- a projection of the parallel hidden `H_i`;
- position and rank embeddings.

Four learned slots attend to the 16 nodes at each position. This retains several lexical modes instead of averaging all candidates into one expected embedding. The resulting `4L=60` slots and one anchor slot pass through two width-128 bidirectional blocks. The four updated slots at each position are pooled and mapped to `delta_code_i in R^256`.

The student correction is

`student_bias_i = W_out SiLU(W_h H_i + delta_code_i)`.

`W_h` and `W_out` start from released Domino and remain frozen in the first stage. Thus the new trainable mechanism is small, while the active head retains the 1.42M-example lexical knowledge already encoded by Domino.

### Parameter and Compute Budget

For the local Qwen3-4B checkpoint:

- Domino GRU: 11.010M parameters;
- Domino two-layer correction projection: 39.813M;
- total Domino correction head: 50.823M.

PLC-Head reuses approximately 39.55M of `W_h + W_out` and adds about 0.6M for the initial encoder, for roughly 40–41M active parameters. A rank-16 optional LoRA would still keep it below about 43M.

The dominant 256-to-vocabulary work remains about 583M MACs for 15 positions, but becomes one batched GEMM. PLC removes roughly 165M recurrent GRU MACs and the 15-step launch dependency; the slot encoder is budgeted below 30M MACs. These are design estimates only. Same-process CUDA-event head latency and end-to-end TPS are hard decision metrics.

The first real A40 profile (`10158697`, one cached horizon-15 block, 25 warmups and 200 repeats) reproduced the cached Domino path exactly. Released sequential correction took `3.6796 ms/block`; projecting the same already-computed 256-D codes in one batched GEMM took `0.1744 ms/block`. Therefore a conservative `0.8x Domino` head-latency gate leaves `2.7693 ms/block` for the entire lattice encoder. The batched number is only a lower bound—PLC still has to predict the codes—but it confirms that removing repeated small vocabulary projections creates a large measured latency budget, not merely a FLOP argument.

### Training Recipe

1. **Exact teacher extraction.** On existing Domino canonical training blocks, run the frozen released GRU on the clean target prefix and save only `teacher_code_i = W_s s_i`, plus teacher candidate logits. This is training-only and adds no deployment component.
2. **Correction-code warm start.** Train the student with normalized code regression plus candidate-set KL to the released Domino teacher. Require same-subset imitation before target improvement training.
3. **Acceptance-frontier adaptation.** Add candidate listwise target CE with detached prefix/continuation weights, expected-prefix utility, and an asymmetric penalty for changing a teacher/base-correct reachable token. Censor at the first target token outside the supported candidate set.
4. **Scale in two stages.** The existing four Domino-materialized OPB parts contain 25K prompts and 199,800 blocks, enough for the first real development run. The source 99K-prompt/794K-block DFlash corpus exists, but the remaining Domino-backbone hidden/code materialization must be collected before claiming a 99K Domino-student run. Select a single checkpoint on validation-select EAL; evaluate validation-gate only after method and checkpoint are frozen.

The main loss is

`L = lambda_code L_code + lambda_KD L_teacher_KL + lambda_gold L_frontier_CE + lambda_keep L_reachable_preserve`.

The warm-start schedule decays `lambda_code` rather than deleting teacher preservation abruptly. No gold or target feature is used at inference.

### Inference Path

Candidate construction, slot attention, global mixing, code prediction, and the vocabulary projection are all block parallel. There is no selected-token feedback loop. The output is a single greedy block and uses Domino's existing verifier unchanged.

### Failure Modes

- **Student cannot imitate Domino:** stop before full training; increase slot count/width only within the 45M total-head budget, not by adding passes.
- **Imitates but cannot exceed Domino:** the missing signal is target alignment, so open dense target candidate KL before increasing architecture size.
- **EAL improves but head latency does not:** use candidate-only gathered `W_out` scoring as a serving variant, but only after measuring whether candidate coverage preserves the frozen method.
- **Offline EAL improves but online TPS does not:** reject the method; no paper claim based only on EAL.

## Claim-Driven Validation

### Claim 1: PLC replaces the sequential head without losing Domino quality

- 1,024-block memorization/imitation gate, then the existing 25K-prompt run; expand Domino materialization toward 99K only if the learning curve remains positive.
- Compare released Domino, PLC after teacher distillation, and PLC after acceptance adaptation on identical anchors.
- Required evidence: teacher-distilled PLC is close to Domino before target adaptation; adapted PLC strictly exceeds Domino on prompt-balanced select and frozen gate EAL.

### Claim 2: PLC produces a materially better speed/quality point

- Same checkpoint, same GPU, same prompts, same generation length, warmup, CUDA Graph policy, and target verifier.
- Report backbone, correction head, verification, time/output token, target calls/output token, and total TPS.
- Hard budget: active head <=45M parameters; correction-head latency <=0.8x Domino; final target is >=1.15x Domino TPS, with >=1.20x preferred. A method that raises EAL but misses throughput is rejected.

### Minimal mechanism deletion

Only three cells are needed: no teacher distillation, one expected-embedding slot instead of four mode slots, and the full PLC model. They test whether released correction knowledge and mode preservation are necessary without creating an ablation zoo.

## Highest-Risk Assumptions

1. A prefix-free block lattice contains enough information to approximate the acceptance-relevant projection of Domino's clean-prefix GRU state.
2. Reusing `W_out` transfers lexical correction knowledge rather than constraining the student to the released local optimum.
3. Batched full-vocabulary correction is sufficiently faster than sequential correction in the real serving loop.
