# Literature Review: What can actually beat Domino's accepted length?

**Date:** 2026-08-08  
**Scope:** Qwen3 block-parallel speculative drafting, single-chain exact
verification, with the local same-anchor target of beating 7.015792 accepted
draft tokens over 15 positions.

## Bottom line

The evidence rules out a frozen selector or a fixed second expert as the main
route.  The strongest recurring mechanisms are:

1. stronger target-conditioned parallel representations;
2. causal correction conditioned on the realized draft prefix;
3. training credit concentrated at the acceptance frontier;
4. target alignment on draft-induced states; and
5. a small number of whole-block refinement passes when extra drafting latency
   is acceptable.

For this project, the fastest credible route is to retain the released Domino
backbone and adapt its causal correction head on exact local anchors using
acceptance-aligned clean-prefix training.  Target replay on draft-induced
prefixes is the next stage, followed by joint backbone adaptation or parallel
iterative refinement only if head-only adaptation saturates.

## Evidence table

| Work | Mechanism | Relevant result | Direct implication here |
|---|---|---|---|
| [DFlash](https://arxiv.org/abs/2602.06036) | One parallel masked-block backbone conditioned on selected target hidden layers | Qwen3-4B paper average `tau=6.54`, using a plus-one convention unlike our metric | Establishes the backbone and explains why suffix quality decays without causal token dependence |
| [Domino](https://arxiv.org/abs/2605.29707) | Parallel backbone plus GRU prefix state and low-rank vocabulary residual; teacher forcing and base-anchor curriculum | Official Qwen3-4B average `tau=7.08`; local same-anchor decomposition is `5.93853` backbone to `7.01579` full head | The causal head is the dominant local recoverable mechanism; its training implementation is now in [SpecForge](https://github.com/sgl-project/SpecForge/pull/571) |
| [DFlare](https://arxiv.org/abs/2606.02091) | Per-draft-layer target fusion, separate target/draft KV projections, deeper backbone, progressive position loss | Qwen3-4B paper average `tau=7.47` | Richer target conditioning can beat the released Domino range without a GRU, but requires a larger jointly trained backbone and much more data |
| [D-PACE](https://arxiv.org/abs/2605.18810) | Detached dynamic CE weights derived from a smoothed expected-prefix surrogate | Roughly 8.5--10.7% emitted-length gain over DFlash in its setting | Use as an acceptance-frontier objective, but revalidate it for Domino because sequential states violate the paper's one-pass factorization |
| [Spec-AUF](https://arxiv.org/abs/2607.01893) | Train only through the current first mismatch | Improves DFlash and modestly improves Domino when applied to the base branch | A first-failure mask is a strong cheap objective candidate; applying it blindly to Domino final logits may be neutral or harmful |
| [DeLS-Spec](https://arxiv.org/abs/2607.07409) | Frozen long-context DFlash plus independent local GRU product-of-experts | Qwen3-4B `6.04 -> 6.35`, below full Domino `6.45` in its ablation | The missing long/short interaction matters; fixed `alpha=beta=0.3` is not sufficient and has exposure/calibration mismatch |
| [DSpark](https://arxiv.org/abs/2607.05147) | Sequential Markov/RNN correction, target-TV distillation, confidence calibration | Block-15 gains reported around 22--30% by domain over DFlash | Full target-distribution alignment and prefix-conditioned correction are high-value; full-vocabulary target caches are costly |
| [Draft-OPD](https://arxiv.org/abs/2605.29343) | Replay draft-induced prefixes and distill target distributions with different accepted/rejected KL directions | Non-thinking DFlash gains about `6.04 -> 6.60` on Qwen3-4B/8B settings | If clean-prefix fine-tuning plateaus, collect target supervision on actual Domino proposals rather than assigning original gold labels to wrong prefixes |
| [xPress](https://arxiv.org/abs/2608.02438) | Jacobi-style whole-block parallel refinement over fixed DFlash features | Qwen3-8B average `tau=8.02`, above a sequential Markov head at `7.64`; gains plateau near 6--7 passes | Best high-capacity fallback when EAL dominates latency, but no local public implementation was found and it is not the first engineering step |
| [AdaFlash](https://arxiv.org/abs/2607.19223) | Online reverse-KL/hard-CE distillation plus adaptive verification length | Qwen3-8B average `tau=7.28` at concurrency 1 in its setup | Confirms distribution adaptation can move acceptance materially, but its online service stack is unnecessary for the initial local objective |

## Local evidence that changes the decision

The local same-anchor validation-select result provides a controlled target:

- released DFlash top-1: `5.1120019436`;
- released Domino parallel backbone: `5.9385325559`;
- released Domino on-policy correction: `7.0157920311`;
- DFlash K16 oracle: `9.7266763848`.

Thus the available path-quality headroom remains about `+2.71` over Domino,
but previous frozen selectors recover almost none of it.  GCLS, FMAS, SAVS,
CAMRS, high-capacity frozen teachers, and PROS collectively show that this is
not primarily a top-K reranking problem.

Two new diagnostics were run on the exact validation-select anchors:

| Variant | Prompt-balanced EAL | Delta vs same-run Domino | 95% prompt-cluster interval |
|---|---:|---:|---:|
| Released Domino, same A40 run | 7.007653 | 0 | -- |
| Global correction scale 0.9 | 7.026239 | +0.018586 | [-0.021259, 0.061953] |
| Best Domino+DeLS screen | 7.019315 | +0.011662 | [-0.049563, 0.075437] |
| Released DFlash+DeLS recipe | 5.276725 | -1.730928 | not primary |

The best fusion configuration contained no DeLS contribution; every tested
positive DeLS weight reduced EAL.  Therefore static product-of-experts fusion
is closed as a main route.  The weak optimum below correction scale 1 supports
calibration as a secondary component, not as a sufficient solution.

The same-run Domino value differs by about 0.008 from the earlier A800 result,
consistent with a small hardware/kernel numerical path effect.  All decisions
use paired differences within one run; the final success test will rerun both
checkpoints on the same hardware.

## Research decision

### Primary route: Frontier-Aligned Domino Adaptation (FADA)

Reuse the released Domino parallel backbone and initialize from its released
causal head.  Cache exact same-anchor Domino hidden states, then fine-tune the
GRU and low-rank residual using only the train split.  Screen a compact set of
objectives:

- released position-decayed clean-prefix CE;
- D-PACE-style detached frontier weighting;
- AUF first-mismatch truncation;
- a carefully specified AUF/frontier hybrid if the first three expose a clear
  complementary failure mode.

Selection is based only on validation-select on-policy accepted length.  The
validation-gate split remains untouched until the method and checkpoint are
frozen.

### Conditional stage: target replay on draft-induced prefixes

If clean-prefix tuning gains less than 0.20 EAL, materialize target logits or a
high-coverage top-K distribution on proposals made by the current checkpoint.
Use target conditionals on those actual prefixes; never train a wrong-prefix
state toward the original target continuation token as if that were still the
correct conditional label.

### Capacity escalation

If the correction head saturates below 7.5 EAL:

1. jointly adapt the last one or two Domino backbone layers using the official
   SpecForge path with base anchoring;
2. add a learned position/state-dependent scalar gate initialized to one;
3. only then prototype an xPress-style 2/4/6-pass refiner, because the user
   explicitly prioritizes accepted length over minimal drafting latency.

## Claims deliberately not made

- A small validation-select increase does not mean Domino has been beaten.
- Oracle top-K availability does not prove a frozen selector can identify the
  correct path.
- Teacher-forcing exposure after the first error is not itself an acceptance
  bottleneck; only clean-prefix states contribute to the current round's EAL.
- Paper-reported `tau` values often include the target bonus and use different
  data/backend conventions, so they are architectural guidance rather than
  numerically interchangeable baselines.
