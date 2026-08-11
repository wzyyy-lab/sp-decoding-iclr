# R046 Full-Vocabulary Distillation Code Review

**Date:** 2026-08-10 03:19  
**Reviewer:** secondary Codex GPT-5.6-Sol, xhigh  
**Independence:** same-family provisional  
**Verdict:** GO for sanity smoke

## Blocking findings and resolution

The first review pass identified two blockers, both fixed before this verdict:

1. The initial path still trained the legacy 15-token canonical horizon. R046
   now reconstructs the exact 16th target from the same causal target pass as
   the independent full-B16 evaluator, trains all 16 Domino outputs, selects
   checkpoints on the historical full-16 runtime cache, and permits both the
   legacy `B-1` and full `B` shift-label horizons.
2. Prefix protection originally did not mask rows where replayed target top-1
   disagreed with canonical gold. The protection, frontier, and future masks
   now all obey the declared mismatch contract.

No blocking issue remains.

## Confirmed properties

- Full-B16 gold labels, teacher hidden states, and student positions have no
  off-by-one error.
- Training, checkpoint selection, historical runtime cache, and independent
  evaluator all use the complete B16 metric.
- Target future logits are labels only. Student inputs stop at each anchor's
  real context length, so there is no online future-token leakage.
- Gradients reach the DFlash backbone; positions 1--15 also reach the existing
  Domino GRU/projection, while position 0 correctly trains only the backbone.
- Frontier KL, per-block suffix KL, and prefix protection have compatible
  scales; suffix length does not multiply its aggregate loss.
- Full-backbone/head checkpoint save and reload use the same sorted trainable
  layout.
- The actual Phase3 train and `validation_select` sample IDs are disjoint.
- Nineteen focused tests and both Slurm syntax checks pass.

## Sanity gates

- step-0 token and accepted-length mismatch counts are zero;
- prompt-balanced full-B16 baseline is exactly `7.23955296404276`;
- frontier and future active-position diagnostics are nonzero;
- some frontier gold tokens are outside base Top-16;
- loss and gradient norm remain finite;
- no A40 OOM or 30-minute timeout.

## Non-blocking follow-up

- Treat the smoke as the GPU integration test for full-B16 materialization and
  cache alignment.
- Record peak memory/time from the full-vocabulary KL path.
- A stricter cumulative mask after a rare target replay mismatch is optional;
  the current row-wise mask matches the written experiment contract.

