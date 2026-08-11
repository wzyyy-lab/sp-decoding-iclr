# PCLD-16R P1 support-denominator diagnosis

## Outcome

Slurm job `10168459` stopped before the first optimizer update because the
trainer observed a frozen-support strict-J2 denominator of `314` while the
legacy JAPD manifest stores `411`.  This is a successful fail-closed receipt,
not a capacity result.  The model output directory is empty and no P1 science
gate has been evaluated.

## Read-only decomposition

The diagnostic artifact is
`artifacts/analysis/pcld_capacity_support_v2_10168459.json`.  On the exact 512
capacity records it reports:

| Label/support definition | Strict-J2 denominator |
|---|---:|
| legacy stored `target_top1_ids` | 411 |
| current authoritative BF16 target replay | 403 |
| authoritative + FP32/BF16 top-1 agreement | 402 |
| full frozen support, including margin `> 2 epsilon_num` | 314 |

`epsilon_num=0.24676132202148438`, so the frozen stability threshold is
`0.49352264404296875`.  Relative to authoritative replay, numeric top-1
agreement removes one eligible block and the margin rule removes another 88.
The legacy and authoritative eligible sets also differ in both directions:
23 are legacy-only and 15 are authoritative-only.

## Interpretation boundary

The old `411` is exactly reproducible, but it was generated before PCLD added
its authoritative replay and stable-row contract.  The frozen method requires
authoritative BF16 top-1 equality, FP32/BF16 agreement, and the margin rule for
every loss and J2 row.  The experiment plan nevertheless copied the older
JAPD denominator into PCLD P1.  These two written requirements are mutually
inconsistent.

The independent contract review preserves `411` as the binding strict-J2
population.  The stable continuous prefix is explicitly the support shared by
the trainable latent/safe/KL losses and candidate-agreement metric; it does not
replace the separately frozen historical J2 evaluation population.  The
authoritative `403`, authoritative+numeric `402`, and stable `314` branches
remain mandatory diagnostics.

The approved implementation correction therefore adds a distinct legacy-J2
mask used only by evaluation, while keeping all losses unchanged.  Before
optimizer step 1, a source-bound receipt must independently reproduce all four
denominators, epsilon, threshold, exact eligible semantic keys, and stable
horizons.  GPU training remains blocked until that patch receives fresh
review.

## Architecture invariants

This diagnosis does not modify the method: the production head remains one
full16, globally non-causal, simultaneous single-chain invocation.  It adds no
serial target decode, autoregressive feedback, iteration, beam, tree, or extra
online target computation.
