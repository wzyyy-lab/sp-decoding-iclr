# PROS-Gate Refinement Report

The route began from a concrete CAMRS failure: a small positive tail among
90,120 harmful edit actions was enough for max selection to over-edit, even as
the validation hinge improved. The initial response reduced 226 actions to one
KEEP-versus-frozen-Direct decision, but review exposed an objective mismatch,
stacking leakage, weak falsification, and unnecessary duplicate compute.

The final method is substantially different and tighter. It uses a
prompt-balanced, gain-weighted unit-margin hinge whose conditional sign follows
expected accepted-token gain; learns only from Phase-3 prompts excluded from
the Direct producer's OPB training; reuses detached globally contextual Direct
states through a 38,674-parameter sidecar; selects only on a dedicated
checkpoint split; and must pass a one-shot 200-prompt contextual falsifier
before development can open. Capacity is explicitly plumbing-only. Every
split, feature, optimizer step, order, denominator, and failure boundary is
deterministic and preregistered.

Review evolution:

| Round | Main concerns | Change | Result |
|---:|---|---|---|
| 1 | utility, leakage, duplicate model, weak falsifier | new loss/OOS split/sidecar/falsifier | 6.3 → 8.8 |
| 2 | conflicting counts and implicit protocol defaults | exact split/optimizer/features/order/recovery | 8.8 → 9.1 |
| 3 | closure audit | no blockers | READY |

The final proposal is `FINAL_PROPOSAL.md`. Remaining risk is empirical
contextual separability and, after offline success, fused end-to-end latency.

