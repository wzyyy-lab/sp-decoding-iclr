# Round 1 Proposal: Reachable-Frontier Domino Adaptation

## Hard outcome and fixed evaluator

Optimize prompt-balanced accepted draft tokens on the exact Qwen3-4B greedy
B16 single-chain anchors.  Released Domino is `7.0157920311/15`; selection aims
for at least `7.5`, followed by one frozen validation-gate confirmation.  Every
candidate and the released initialization are evaluated in the same job and
code path.

## Evidence-backed diagnosis

The released parallel backbone scores `5.9385`, its causal correction reaches
`7.0158`, while the same-candidate oracle is `9.7267`.  A global correction
scale adds only `0.0186` with an interval crossing zero, and every positive
released DeLS fusion weight hurts.  The remaining problem is not static
calibration or independent expert selection; it is state-dependent correction
at the first still-reachable acceptance boundary.

## Stage 1: low-risk correction adaptation

Cache released Domino parallel hidden states on the existing train/select/gate
anchors.  Cache construction reproduces released on-policy proposals and
prompt-balanced EAL and enforces the released `shift_label=true`, pure-prefix
one, block-16/horizon-15 alignment.

The target and parallel backbone stay frozen.  Initially train only the causal
GRU and the rank/input portion of the correction projection, initialized from
released Domino; freeze the large vocabulary projection.  Add L2-SP to the
released parameters.  Full-head training is tried only after a measured
capacity shortfall.

For every block define:

- `m_base`: frozen base position zero equals the target-greedy token;
- `m_clean(j)`: all current greedy tokens before `j` equal the gold prefix;
- `m_train(j)`: `m_base * m_clean(j)` for correction-controlled `j>=1`.

If `m_base=0`, correction-head loss is exactly zero.  The primary loss is
reachable-breaker CE: weight one at the first current reachable mismatch and
weight `0.1` on its preceding correct prefix, plus L2-SP.  Controlled
alternatives are released-style decay CE and D-PACE; both use the same base
mask and effective-weight normalization.  A vectorized/exact-position teacher
state check must agree before optimization begins.

Selection uses on-policy EAL after every epoch.  A head-only route below
`+0.10` over its same-run released initialization is stopped rather than
over-tuned.

## Stage 2: expand the mechanism until the target is met

If low-risk head adaptation is insufficient, expand one bottleneck at a time:

1. unfreeze the full correction head with stronger L2-SP;
2. jointly tune the final one or two released parallel-backbone layers so the
   fixed first token and correction features can move together;
3. add target replay only on actual draft-induced prefix states, emphasizing
   accepted/reachable states rather than an already-wrong suffix;
4. if the sequential head plateaus, introduce a small 2/4/6-pass Jacobi block
   refiner over released parallel features, retaining exact verification.

The first candidate reaching `>=7.5` on validation-select is frozen.  The
validation-gate split is then opened once; success requires a strict same-job
gain over released Domino, with a positive prompt-cluster interval preferred.

## Measurements that drive decisions

Report prompt-balanced EAL, paired delta, full-horizon rate, first-base-token
acceptance, per-position conditional acceptance given a clean prefix, first
reachable-breaker position, per-domain EAL, and beneficial/harmful blocks.
Teacher loss alone never selects a method.  No hash audit or formal provenance
gate is part of this performance route.
