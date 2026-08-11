# R048 Draft-Prefix Early-Verifier Research Review

**Date:** 2026-08-10 04:36 CST  
**Reviewer:** secondary Codex GPT-5.6-Sol, xhigh  
**Independence:** same-family provisional  
**Verdict:** GO for a one-iteration earliest-one falsifier only

## Approved mechanism

1. Generate a frozen proposal.
2. Run target layers 0--3 over exactly
   `[anchor, proposal_0, ..., proposal_14]` using the verified-prefix KV.
3. State 0 is produced by anchor and predicts proposal 0; state i is produced
   by proposal i-1 and predicts proposal i.
4. Find the earliest position whose best candidate beats the proposal token by
   a calibrated margin and change only that token. Keep the suffix unchanged.
5. The full target verifier remains authoritative, so generated output stays
   lossless.

At the true first rejection r, proposal[:r] equals the target prefix, so every
state through r is verifier-path exact. States after r are stale and may not be
used for a decision or a training label.

## Rejected variants

- All-position one-shot correction: rejected because every state after the first
  changed token belongs to the old prefix.
- Two-iteration correction: not yet authorized. It doubles early-target/cache
  complexity without fixing first-pass false positives. Compute only its
  perfect oracle for headroom.
- Retaining the full 2.1109 ms released Domino head as the final system:
  rejected for throughput. It may be used only as an information falsifier.

## Head contract

Use candidate-only Fast-K32 and a target-only tuned residual:

```text
e_i = RMSNorm(early_state_i)
q_i = Linear(2560, 64)(e_i)
r_i = Linear(64, 256)(SiLU(q_i))
delta_ik = dot(r_i, frozen_Domino_basis[token_ik])
```

The up projection is zero-initialized. Trainable parameters are exactly
180,224. No local DFlash/GRU branch, depth sweep, mixer, or second repair is
allowed. The frozen candidate proposal already carries DFlash+GRU information.

## Evidence and gates

The unrestricted exact single-frontier repair oracle is 8.474125364, but a
system must recover 87.96% of that gain to reach 8.325485909. Therefore:

1. Compute candidate-constrained perfect one-repair oracles before training.
   K16 must be discarded if its oracle is below 8.40; evaluate K32.
2. Fast-K32 perfect correction plus the real layer-split cache path must have an
   ideal throughput ceiling of at least 1.20x Domino.
3. A 32--64 prompt same-set capacity run must recover at least 90% of its
   K-oracle gain within 200 steps.
4. On prompt-disjoint Phase3, stop if step50 is below 7.55 or step100 below
   7.80, or if protected harm exceeds 0.05 EAL.
5. No data expansion unless best fixed EAL is at least 8.10. Accuracy GO is
   fixed full-B16 EAL at least 8.325485909.

Training uses centered target candidate KL plus a protected hinge weighted by
potential lost prefix length; frontier examples are weighted by their exact
one-repair reward. The margin threshold is selected only on a train-internal,
prompt-disjoint calibration subset.

## Runtime/cache contract

If no token changes, early-layer KV and layer-4 outputs are reusable by the
verifier. If position j changes, early-layer KV through the input before
proposal_j remains valid; original KV beginning at proposal_j and layer-4 rows
j+1 onward must be dropped. Recompute the corrected token and unchanged suffix
through layers 0--3. Later target layers have not run yet and retain the original
verified-prefix cache.

Training collection must execute the same released/candidate proposal and
partial-target geometry. Gold-path features after the first rejection are not
valid deployment inputs.

