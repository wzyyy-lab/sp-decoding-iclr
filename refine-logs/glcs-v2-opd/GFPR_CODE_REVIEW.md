# Fresh GFPR Implementation Review

**Reviewer:** fresh GPT-5.6-Sol xhigh agent (same-family provisional)  
**Date:** 2026-08-10  
**Initial verdict:** **BLOCKED**

The GFPR math core is mostly sound, but the reviewed workflow could report proof-of-signal from leaked same-set evaluation and could not yet compute the required true adapted-policy dynamic-rollout metric. It was therefore not ready for a claim-bearing GPU screen at review time.

## Blocking findings

### 1. CRITICAL — Capacity overfit could be reported as held-out proof

`gfpr_capacity_smoke.sbatch` used the same rollout and `train` split for training and evaluation, while `train_gfpr_head.py` emitted the full `proof_of_signal_gate` without a sample-overlap check. The observed roughly +2.94/+3.02 results are capacity evidence, not Gate-B evidence.

Required fix: detect train/eval prompt overlap; require disjoint prompts for claim-bearing gates; provide an explicit capacity-only mode that suppresses held-out decisions.

### 2. CRITICAL — Cached dynamic replay was not true adapted-policy rollout

The trainer independently decoded stored anchors/hiddens and never advanced the adapted model by its new `r+1`. A collection marked `dynamic` therefore evaluated a new head on released-policy dynamic anchors, not on its own trajectory. Different trajectories cannot be paired by zipping blocks because their anchors and cycle counts differ.

Required fix: retain fixed-anchor replay for checkpoint selection, but add a separate evaluator that collects released and adapted trajectories on identical prompts/continuation budgets and pairs prompt-level EAL, advances, and cycle counts.

### 3. CRITICAL — v1 refresh and 50/50 replay were not implementable

Training accepted only one rollout, always initialized the released checkpoint, and had no `--initial-adaptation`. A v1 rollout would therefore be trained from a different policy than the one that generated it.

Required fix: support initial-adaptation loading, repeatable rollout inputs, prompt-level source balancing, and checks tying refreshed rollouts to their generating adaptation and policy version.

### 4. HIGH — Gate A was self-consistency and did not fail closed

Collection and training identity used the same helper. The analyzer wrote EAL/oracle booleans but did not return failure. The full fixed artifact exactly matching the historical 7.239552964 and oracle values was encouraging but not enforced.

Required fix: build one overall Gate-A decision from semantics, historical EAL tolerance where available, and oracle headroom; exit nonzero on failure. The user explicitly rejected hash-style formalism, so remediation should focus on behavioral equivalence rather than hashes.

### 5. HIGH — Split sealing and compatibility were not enforced

The collector default included `validation_gate`; trainer metadata checks did not enforce target/Domino compatibility, split disjointness, or policy version. Adaptation checkpoints lacked base-policy provenance.

Required fix: require explicit splits; reject sealed validation unless deliberately enabled after method freeze; validate rollout assets and store lightweight checkpoint provenance.

## Additional efficacy and robustness findings

- **MEDIUM — Prompt accumulation:** microbatch normalization was correct locally, but an incomplete final accumulation was under-scaled and batches with different prompt counts received equal weight. Accumulate prompt-weighted losses over the actual group.
- **MEDIUM — Position-zero efficacy:** alpha-zero identity was tested, but real position-zero repair/harm and alpha gradient were not reported. Add baseline/current position-zero accuracy plus repaired/harmed counts and gradient diagnostics.
- **MEDIUM — Memory/nonfinite:** the teacher path retained an unused FP32 full-vocabulary base tensor; long prompt trajectories could be expensive. Remove the unused return and fail on nonfinite loss/gradients.
- **MEDIUM — Recollection ergonomics:** atomic completion existed but no resume, and the smoke script hardcoded a previous rollout. This was not considered an efficacy blocker for the short 2K collection.

## Verified-correct portions

- GRU timing is correct: consume anchor, score position 0, feed selected position-0 token, then score position 1.
- Teacher-prefix state equals current-policy state through the current first rejection.
- Alpha-zero preserves position-zero base identity; synthetic nonzero alpha has gradient.
- Current-frontier masking, competitor margin, normalized keep loss, `r+1`, and full-accept bonus indexing are logically correct.
- The then-current unit suite passed (8 tests), and reviewed Slurm scripts passed shell syntax.

## Remediation status

The primary agent began remediation immediately after this review. A re-review is required before the claim-bearing 2K screen is accepted.
