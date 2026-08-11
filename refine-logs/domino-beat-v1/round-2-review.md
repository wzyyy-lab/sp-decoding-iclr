# Round 2 Research Review (fresh-agent, raw)

**VERDICT: PASS — SCORE: 9.3/10 — BLOCKERS: none.**

The three substantive issues are closed:

- Stage 2 keeps `L_base,0` active, so blocks with a wrong first token propagate
  gradients into an unfrozen backbone.  Later positions retain detached
  clean-prefix reachability and therefore remain aligned with accepted length.
- `9.7267` is correctly identified as the DFlash K16 candidate-path diagnostic,
  not a Domino oracle.
- D-PACE position-zero probability enters only the suffix survival weights;
  head-only training does not optimize position zero, effective suffix weights
  normalize the loss, and frozen-base reachability still masks unreachable
  blocks.

The proposal now directly optimizes same-anchor greedy accepted length.
Teacher-forced states create signal only at on-policy-reachable breakers and
their correct prefixes; no reachability or shift/state misalignment remains.
Proceed to implementation and GPU validation.
