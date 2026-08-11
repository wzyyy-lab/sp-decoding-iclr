# Autonomous Review: Domino Causal Lattice Decoder

Scope: substantive review of the new causal prefix Transformer × full Top-K
lattice decoder.  Provenance hashes and publication formalities are explicitly
out of scope; the operative question is whether the implementation and design
can improve exact on-policy accepted length over released Domino.

## Round 0 — fresh-agent review

1. Score: 7/10

2. Verdict: almost

3. Critical weaknesses, ranked

1. The scheduled 96K matrix cannot identify the claimed mechanism. The two tasks simultaneously change dimension, objective, batch size, and initialization. A win answers “can one configuration improve validation-select?” but not whether frontier loss, lattice capacity, or pretrained features caused it. It also does not provide the clean held-out confirmation required by C1.

   Minimum fix: after this screen, run a matched objective pair at identical dimension/init/batch, then evaluate one frozen winner on `validation_gate`.

2. The optimization measure is not prompt-balanced EAL. `decay_ce` supervises every candidate-covered gold-prefix suffix, including positions beyond the current greedy breaker; both objectives divide by global active-position count. Long/high-coverage blocks therefore dominate, despite selection using prompt-balanced EAL. This is especially risky given the prior 215K head adaptation improved only `+0.029`.

   Minimum fix: normalize weighted loss per block before batch averaging, preferably sample prompts uniformly; make breaker/reach weighting the primary objective and retain decay CE as a control.

3. The crucial teacher/sequential alignment is inspected correctly but not tested. Teacher correction uses `[anchor, gold[:-1]]` with `gru_out[:,1:]`; rollout uses `[anchor, proposals]` and takes the final prefix state. These are aligned on a correct prefix, but the tests only establish zero-residual rollout identity, not teacher/rollout score equality.

   Minimum fix: add a forced-gold-prefix test comparing per-position Domino logits, candidate IDs, and decoder scores between `teacher_forward` and sequential decoding.

4. Candidate rank semantics are aliased. When released Domino’s top-1 is outside base Top-K, it replaces the last base candidate but receives the same rank embedding as genuine base rank K. The model is not explicitly told “released action” versus “base rank K.”

   Minimum fix: add a released-action flag and true base-rank/missing-rank scalar to candidate features.

5. The smoke does not exercise the actual main route: it uses K=16, no selector initialization, and a small batch, whereas main uses K=4 and one initialized branch. Thus it will not expose initialization-route or main-memory failures.

   Minimum fix: add a short K=4 initialized smoke matching task 0 before interpreting the 96K job.

4. Positive findings

- Candidate, prefix, and GRU state shifts are correct by inspection.
- Teacher forcing is not inherently an exposure-bias bug for exact accepted-prefix length: every state that contributes before the first error necessarily has a gold prefix. Post-breaker rollout states do not affect EAL.
- Full future lattice attention is not leakage because the memory contains only inference-available parallel hidden states/base candidates.
- Exact zero-Domino identity is well designed: zero residual initialization plus mandatory released top-1 inclusion, with a hard cached replay abort.
- The five CPU tests pass; syntax and compilation checks pass. Training and validation prompt sets inspected are disjoint.

5. Launch decision

GPU smoke 10152908 may launch unchanged. Main 10152987 may also launch unchanged after smoke passes, but only as an exploratory validation-select screen. It cannot by itself establish C1 or C2; matched controls and frozen `validation_gate` confirmation remain mandatory.

## Immediate implementation response

The two weaknesses that directly affect the primary metric were fixed before GPU launch: loss is now normalized per block and inverse-weighted by prompt block count, and candidate scoring now receives explicit released-action/outside-base-Top-K features. Mechanism-isolation and publication-formality work remains deferred until the method clears the required accepted-length threshold.
