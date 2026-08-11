# Round 0 Research Review (fresh-agent, raw)

## Verdict

**REVISE — 7.4/10; no fatal flaw.**  The proposal is directionally strong and
the same-anchor objective is appropriate, but the first training stage needs a
more exact reachability definition and a lower-risk trainable scope before it
is ready to spend GPU budget.

## Required revisions

1. Define three masks explicitly: fixed base-first reachability, clean-prefix
   reachability, and head-trainable positions.  If the frozen base first token
   is wrong, a correction-head objective cannot improve this block and its
   head loss must be zero.
2. D-PACE may keep position zero as a constant factor in its reachability
   product, but the optimized suffix starts at `j>=1`; normalize by effective
   nonzero weight rather than by batch size.
3. Plain final-head AUF has weak evidence for this setting.  Add an explicit
   reachable-breaker CE/margin loss at the first current mismatch, plus a small
   prefix-preservation term.  The evidence for AUF is stronger on a base
   branch than on a final correction branch.
4. Training the entire roughly 51M correction head on about 16K blocks is
   risky.  Start with the GRU and first/rank projection while freezing the
   roughly 38.9M vocabulary projection; initialize from released Domino and
   use L2-SP toward those weights.  Expand to the full head only if measured
   under-capacity calls for it.
5. Cache and report prompt-balanced released outcomes, not only a
   round-weighted mean.
6. For target replay, use actual draft-prefix target conditionals.  A loss on
   an already-wrong suffix has no direct current-round greedy EAL contribution,
   so it must not be treated as the primary path to the result.
7. Before training, run a single-block teacher-state alignment test and
   evaluate the released initialization on the same device/code path.
8. Escalate beyond head-only adaptation if the gain is below `+0.10`; retain
   the performance target of at least `7.5`, not a merely positive delta.

## Recommended diagnostics

- Per-position conditional acceptance given that the prefix is still clean.
- Frequency and location of the first reachable breaker.
- Beneficial versus harmful block changes and prompt-balanced paired deltas.

## Assessment

The slot/state alignment is otherwise consistent with released Domino, and
the proposed conditional escalation to joint adaptation or iterative
refinement is reasonable.  Incorporating the items above would make the first
experiment both lower-risk and more directly optimized for accepted length.
