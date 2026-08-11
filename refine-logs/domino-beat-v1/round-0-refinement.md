# Round 0 Refinement

The first implementation is revised around *reachable acceptance breakers*.
For each block, position zero is produced only by the frozen parallel backbone.
If that token is wrong, every correction-head loss is masked to zero.  At later
positions the clean-prefix mask is detached from the current greedy rollout.

The primary Stage-1 loss is now:

- unit-weight CE at the first current mismatch reachable through a correct
  base token;
- `0.1`-weight CE on the preceding correct prefix to avoid moving an earlier
  decision boundary the wrong way;
- L2-SP (`1e-3` initially) toward released Domino parameters.

DECAY-CE and D-PACE remain controlled alternatives.  Their weights are
multiplied by base reachability and normalized by the effective weight sum.
D-PACE's optimized positions begin at `j>=1`; position zero appears only as a
constant reachability factor.  Plain final-head AUF is demoted from the default
because recent AUF evidence is stronger for a base branch.

The initial trainable scope is the GRU plus the rank/input part of the
projection.  The large tied vocabulary projection is frozen.  Full-head
training is an escalation, not the first screen.  Cached metadata includes the
released prompt-balanced EAL, and training aborts if a vectorized teacher-state
calculation disagrees in argmax with the explicit position-wise calculation.

Target replay remains conditional and uses target conditionals evaluated on
actual draft prefixes.  Wrong-suffix supervision is auxiliary because it
cannot extend the current accepted prefix.  Head-only adaptation is abandoned
quickly if it cannot reach `+0.10`; the route then expands the parallel
backbone/correction interaction, with `>=7.5` still the selection target.
