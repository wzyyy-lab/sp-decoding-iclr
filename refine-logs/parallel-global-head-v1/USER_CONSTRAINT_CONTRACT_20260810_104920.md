# Authoritative Contract: Parallel Global Single-Sequence Head

**Status:** immutable user constraint  
**Effective:** 2026-08-10 10:49:20 +0800  
**Supersedes:** any earlier proposal, amendment, experiment, or review that authorizes a causal/autoregressive head, serial target decoding, iterative refinement, beam/tree/forest generation, or multi-path verification.

## Problem Anchor

- **Bottom-line problem:** completely solve the low accepted-length problem and substantially exceed released Domino.
- **Mechanism thesis:** DFlash already produces a 16-position parallel draft lattice. A new lightweight head must globally mix the complete 16-position state so every output position can use information from every other draft position, then select all 16 output tokens in one parallel forward pass.
- **Deployment objective:** improve accepted length enough that, after comparable SGLang integration, end-to-end throughput is at least 1.15x released Domino.

## Non-Negotiable Architecture Invariants

1. **Parallel full-block input.** The head consumes the complete block together, with the canonical feature tensor shaped conceptually as `[batch, 16, feature_dim]`. Per-position DFlash hidden states, base logits, Top-16 candidate IDs/scores, candidate embeddings, and position features may be used.

2. **Global non-causal visibility.** Before token selection, every one of the 16 positions must be able to exchange information with all other 15 positions through a non-causal mixer. No causal/triangular position mask is allowed. Sequential depth across network layers is normal; sequential dependence across output token positions is forbidden.

3. **One parallel prediction.** A single head invocation outputs scores shaped `[batch, 16, K]` (normally `K=16`) or an equivalent full-vocabulary tensor for all positions simultaneously. One argmax per position produces exactly one tensor `[batch, 16]`.

4. **Exactly one sequence.** Top-16 is an internal candidate axis, not a sequence/path axis. No beam, tree, trie, forest, multiple proposal paths, target-side candidate set, or multi-branch verification is allowed.

5. **No autoregressive token feedback.** The selected token at position `i` must not be embedded or otherwise fed into computation for position `i+1` within the head. Domino-style causal GRU rollout, causal lattice decoding, token-by-token loops, Jacobi refinement, and iterative repair are forbidden.

6. **No extra target inference.** The target model may supply offline training supervision and the normal final speculative verifier only. Serial target seeds, early-target routing, extra target forward calls, or target-derived online features unavailable in ordinary DFlash inference are forbidden.

7. **Lightweight, evidence-scalable head.** Start near or below 10.75M new trainable parameters (about 2% of the 537.427M headless DFlash draft model). Capacity may increase only when held-out acceptance evidence justifies it and measured head latency remains compatible with the final throughput goal. Report complete active parameters and eager latency fairly; do not require the isolated head to be smaller than Domino under every prototype.

8. **Primary objective first.** The first gate is genuine held-out accepted-length improvement from the parallel global head. Do not divert effort to hashes, packaging, minor engineering cleanup, tree systems, or unrelated optimizations before the main mechanism works.

9. **No leakage or same-set substitution.** Training/selection/held-out prompt IDs must be disjoint. Target logits/tokens are labels only. Same-set capacity, oracle coverage, and training-fit results are diagnostics and cannot satisfy the acceptance gate.

10. **Fair final comparison.** Prototype comparisons are eager-to-eager on the same A40, batch size, block length, backbone outputs, and precision. The final comparison is same-stack SGLang end-to-end throughput against released Domino with comparable optimization.

## Quantitative Gates

- **Fixed held-out EAL:** at least `1.15 ×` the same-job released Domino EAL; higher is preferred. The measured Top-16 oracle is an ambitious ceiling, not a deployable result.
- **Dynamic rollout EAL:** at least `1.15 ×` the same-job released Domino dynamic EAL.
- **Head cost:** initially target a complete eager head cost no more than roughly `1.2 ×` the corresponding eager Domino head; this is a development guide, not a substitute for the final system result.
- **Final SGLang throughput:** at least `1.15 ×` released Domino end-to-end tokens/s on the same A40 and workload, with accepted length also above Domino.

## Mandatory Pre-Launch Compliance Check

Every new proposal, implementation, and Slurm launcher must answer **YES** to all of the following before GPU submission:

- Does one invocation consume all 16 positions?
- Can each output position see all 16 input positions non-causally?
- Are all 16 token decisions produced simultaneously?
- Is the output exactly one 16-token sequence?
- Is there no selected-token feedback across positions?
- Is there no beam/tree/forest or target-side multi-candidate verification?
- Is there no extra/serial target forward at inference?
- Are only online-available DFlash features used?
- Is held-out evaluation disjoint and same-job Domino the comparator?
- Is the parameter/latency path compatible with the 1.15x SGLang goal?

Any **NO** is a hard stop. Reviewer suggestions that violate an invariant are architectural drift and must be rejected.

