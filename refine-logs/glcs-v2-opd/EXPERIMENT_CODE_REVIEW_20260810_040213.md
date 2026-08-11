# R047 Current-Anchor Early-Exit Code Review

**Date:** 2026-08-10 04:02 CST  
**Reviewer:** secondary Codex GPT-5.6-Sol, xhigh  
**Independence:** same-family provisional  
**Verdict:** GO for staged GPU sanity; Phase3 gated on trained-checkpoint KV alignment

## Confirmed properties

- `hidden_states[4][context_length]` is the current anchor after target
  decoder layers 0--3. The anchor index is correct and causal attention cannot
  read later gold tokens.
- The alignment probe correctly truncates the target to four layers and replaces
  the terminal norm with `Identity`; this recovers the full model's unnormalized
  layer-4 intermediate state.
- The full-B16 teacher geometry is correct: the anchor state predicts gold0 and
  gold0--gold14 states predict gold1--gold15.
- Target KL touches only the released accepted prefix and original first
  rejection. Stored wrong-prefix suffixes have zero loss.
- The zero-initialized residual exactly reproduces the released Domino fallback,
  including GRU timing and tie behavior.
- The R047 trainable scope is exactly
  `3584*64 + 2560*64 + 64*256 = 409,600` parameters. Released GRU,
  projections, target embeddings, and candidate basis remain frozen.
- Checkpoint state and target/Domino/train/eval/alignment provenance are saved.
- No evident A40 OOM or numerical-stability blocker was found.
- Verification passed: 24 focused tests, Python compilation, and all four R047
  Slurm launchers passed syntax validation.

## Required execution order

1. Small collection (at most 32 prompts).
2. Numeric full-replay versus batch-1 incremental-KV alignment.
3. Full train and `validation_select` collection.
4. 32-prompt mechanics smoke.
5. Alignment using a genuinely trained `step > 0` checkpoint with nonzero
   `residual_up`.
6. Phase3 only if the checkpoint alignment gate passes.

## Blocking conditions before Phase3

The initial review found three ways a meaningless alignment could pass:

- using a zero-residual step-0 `best_candidate.pt`;
- using an alignment report from a different collection/model;
- treating a smoke checkpoint as proof for the final Phase3 checkpoint.

The implementation now rejects step-0 or zero-`residual_up` checkpoints,
binds report collection/target/Domino/eval-rollout provenance, and makes the
Phase3 launcher require a passed trained-checkpoint report. If Phase3 produces
a promising checkpoint, that final checkpoint must undergo incremental-path
evaluation again, preferably over all of `validation_select`, before any
acceptance claim.

## Final reviewer decision

**GO** for mini collection -> numeric alignment -> full collection -> 32-prompt
smoke -> trained-checkpoint alignment. **NO-GO** for Phase3 until those gates
pass.

