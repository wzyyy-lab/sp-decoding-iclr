# Round-1 Independent Review: CAMRS

**Overall:** 7.8/10  
**Verdict:** REFINE  
**Authorization:** no implementation or GPU execution.

## Blocking findings

1. **Loss-augmented tie/subgradient semantics were undefined.** The pointwise
   regret bound and zero loss at `s=v` are mathematically correct, but an
   ordinary first-index `torch.max` chooses KEEP in the all-action tie at
   `s=v`. On a repairable block this produces a nonzero
   `e_KEEP-e_oracle` subgradient at zero loss.
2. **The hinge gate mixed averaging measures.** Training hinge is a uniform
   block mean, while the reported EAL gap is prompt-balanced. The capacity
   manifest has block-weighted normalized oracle advantage
   `462/(512*15)=0.06015625`, so the exact matching 5% hinge threshold is
   `0.0030078125`; the prompt-balanced gap must remain separate.
3. **Per-metric epoch maxima cannot prove joint passage.** Every epoch needs
   a single conjunctive gate verdict, along with selected-checkpoint identity
   and a list of jointly passing epochs.
4. **“Not weighting” was too categorical.** CAMRS is not class-reweighted MSE,
   but regret is a cost-sensitive margin and selects active constraints.

## Required revisions

- Implement an explicit non-oracle maximum followed by `ReLU`, with zero
  gradient at zero, deterministic lowest-index non-oracle ties, KEEP oracle
  for zero utility, and FP32 loss/gate arithmetic.
- Test zero-loss stationarity, all deployed action/sign cases, score ties,
  all-neutral/no-benefit blocks, abstract and residual-score gradients, and
  randomized regret-bound slack.
- Freeze the block-weighted hinge threshold at `0.0030078125`; retain the
  prompt-balanced oracle-gap gate as nonredundant behavior evidence.
- Persist all epoch gate values/booleans, `joint_gate_passed`, selection key,
  minimum hinge-bound slack, violation count, and selected flag.
- Diagnose hardest-competitor mismatch: deployed-action equality, raw rank,
  sign/regret, cost-augmentation-only selection, coverage/churn, zero-loss
  blocks, directional gradient norms/cosines/cancellation, and clip frequency.
- State the stochastic-label and capacity-memorization limits explicitly.
- Freeze exact denominators and utility-equivalent tie semantics.
- Retain hashes, snapshots, exact steps, and reconstructible example records;
  physically isolate development validation and pin exact Direct artifacts
  before any development run.
- Frame novelty as frozen-lattice task formulation/integration, not a new
  hinge or a verified “first.”

## Non-blocking assessment

The regret bound is valid, KEEP deployment ties are compatible with it, the
residual head can represent arbitrary per-edit scores pointwise, and a
repairable oracle receives an upward gradient not divided by 225 actions. The
main empirical risk is that early hard-negative work suppresses the most
harmful action rather than the current KEEP/edit boundary. Changed gradient
scale and hard-negative churn are falsifiable risks of the exact frozen
optimizer/schedule combination.

## Scorecard

| Dimension | Score |
|---|---:|
| Problem fidelity | 9.6 |
| Mathematical correctness | 8.1 |
| Method specificity | 7.9 |
| Objective-deployment alignment | 7.6 |
| Contribution quality | 6.6 |
| Frontier leverage | 8.4 |
| Optimization feasibility | 7.7 |
| Validation focus | 7.3 |
| Data/checkpoint discipline | 9.0 |
| Claim discipline | 8.2 |
| Venue readiness | 6.4 |

The route is promising and genuinely different from class-reweighted SAVS,
but it is not READY until tie/autograd semantics, aggregation units, joint
gating, diagnostics, and scope language are frozen.
