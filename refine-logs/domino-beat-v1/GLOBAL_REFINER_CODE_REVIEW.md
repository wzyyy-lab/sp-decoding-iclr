# Global causal refiner implementation review

Experiment-bridge local and fresh-agent reviews found no blocking issue before
GPU smoke.  They confirmed:

- a zero residual exactly preserves released Domino proposals and accepted
  lengths; candidate-only variants deliberately floor logits outside their
  deployable set rather than claiming full-logit equality;
- released GRU state and `[anchor, guesses[:-1]]` token alignment are correct;
- the per-channel mixer is strictly lower triangular while each position keeps
  its direct fused input;
- position zero has a trainable residual and loss path;
- Jacobi updates read the previous whole iterate and update synchronously;
- target, Domino, and the released rank-to-vocabulary readout stay frozen while
  the readout still propagates gradients into the refiner;
- selection is on-policy prompt-balanced EAL on validation-select.

The 64-block GPU smoke completed all baseline, training, checkpoint, and
Jacobi paths.  The main run adds the xPress-style all-position exponentially
decayed CE objective; released-seed gold supervision is reported as such and
is not described as a KL consistency loss.

Non-blocking optimization: evaluating K=1/2/4/6 currently recomputes the
released rollout; this can be fused after the performance path is established.
