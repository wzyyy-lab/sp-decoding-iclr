# Round 2 Refinement: Final Gradient and Formal-Estimand Contract

This addendum supersedes only the two affected clauses in
`round-1-refinement.md`; every architecture, objective, data, threshold,
control, and stopping rule otherwise remains unchanged.

## Corrected Gate-0 gradient contract

Exact identity requires `residual_projection.weight = 0`.  Consequently the
identity backward pass is expected to stop upstream gradient while still
training the output projection.  Gate 0 now requires this exact sequence:

1. **At identity initialization:** FMAS logits/path exactly reproduce DFlash;
   action CE is finite; `residual_projection.weight` has a finite, nonzero
   gradient; every frozen input has no gradient; no nonzero upstream-backbone
   gradient is required on this first backward pass.
2. **After one optimizer step on the output projection:** a new forward/backward
   pass has finite, nonzero gradient in the residual projection and in at least
   one upstream trainable parameter group; frozen inputs still have no gradient.

The test must fail if the residual projection is detached, if frozen inputs
receive gradients, or if upstream gradients remain identically zero after the
first update.

## Frozen Gate-4 estimand

Seeds are exactly `0,1,2`; there is no checkpoint ensemble and no seed selected
after seeing formal data.  Each seed is evaluated separately with its
development-selected checkpoint.

For each formal prompt and each comparison control, compute the prompt's FMAS
minus control EAL difference separately for all three seeds, then average those
three paired differences within that prompt.  The primary aggregate EAL
estimand is the equal mean of these seed-mean paired prompt differences.
Prompt-cluster bootstrap resamples formal prompt keys and carries all methods,
blocks, and seeds for a sampled prompt together.  Success requires:

- every individual seed's point estimate is positive for FMAS minus
  Direct-native and FMAS minus Direct-one-edit; and
- the aggregate seed-mean paired 95% CI lower bound is above zero for both
  comparisons.

For harm, compute a prompt-cluster one-sided 95% UCB independently for each
FMAS seed's harmed fraction.  The deployment condition is the conservative
worst-seed rule:

```text
max_seed UCB_95(harmed_fraction_seed) <= 0.05
```

The end-to-end formal rollout contains five fixed paths: released DFlash,
Direct-native, **Direct-one-edit from the identical Direct-native checkpoint**,
FMAS, and same-protocol Domino.  It reports per seed where applicable and the
same seed-mean paired prompt estimand.  Direct-one-edit is required online so
that the supervision effect remains causally isolated after on-policy context
shift and selector overhead.

