# PCLD-16R Literature Boundary Update

This note narrows the contribution claim after checking current primary sources.

## Closest mechanisms

- [DFlash](https://arxiv.org/abs/2602.06036) establishes one-forward block-diffusion drafting conditioned on target context features. PCLD cannot claim one-pass parallel drafting itself.
- [Your LLM Knows the Future](https://arxiv.org/abs/2507.11851) already uses simultaneous future-token prediction, latent consistency, a lightweight sampler, and gated adaptation. PCLD cannot claim predictive-latent alignment itself.
- [Parallel Token Prediction](https://arxiv.org/abs/2512.21323) establishes that a one-call parallel model can represent dependent token distributions. PCLD cannot claim the general possibility of joint parallel prediction.
- [DFlare](https://arxiv.org/abs/2606.02091) enriches DFlash target knowledge through layer-wise target-feature fusion and larger data. PCLD cannot claim richer target conditioning or scaling draft capacity itself.
- [DSpark](https://arxiv.org/abs/2607.05147) explicitly targets acceptance decay from missing intra-block dependency, but solves it with a lightweight sequential module. That route violates the immutable user contract and is not an allowed baseline architecture.

## Defensible contribution only if experiments pass

The narrow contribution is:

> A candidate-conditioned full16 global noncausal head distills 16 clean autoregressive predictive residuals into one simultaneous single-chain Top16 decision, without selected-token feedback, serial target inference, iteration, or multi-path verification; matched local and no-latent controls show that both remote block context and residual supervision causally improve held-out accepted prefix.

The mechanism claim fails if either matched control is statistically equivalent. The overall system claim additionally requires fixed and dynamic EAL plus same-stack SGLang throughput to exceed released Domino by the frozen 1.15x gates.
