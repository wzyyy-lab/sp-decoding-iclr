# GCLS Result Analysis and Pivot Evidence

## Binding representation screen

All values below are raw prompt-balanced EAL on the same 147-prompt development split.

| Variant | Params | Selected epoch | Delta vs DFlash | Harm | First-miss repair | Oracle gap recovered | Hard candidate acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| axial-additive | 1,467,440 | 6 | +0.128280 | 8.00% | 14.13% | 2.78% | 9.30% |
| flat-additive | 1,071,968 | 4 | +0.087464 | 6.47% | 12.60% | 1.90% | 8.21% |
| flat-compatibility | 1,235,808 | 4 | +0.106657 | 7.66% | 14.02% | 2.31% | 8.76% |

Prompt-cluster paired bootstrap, 20,000 repetitions:

| Difference | Estimate | Development 95% CI |
|---|---:|---:|
| flat-additive − axial-additive | -0.040816 | [-0.130952, +0.049320] |
| flat-compatibility − flat-additive | +0.019193 | [-0.059524, +0.099490] |
| flat-compatibility − axial-additive | -0.021623 | [-0.106293, +0.066812] |

## Historical raw comparison

| Train prompts | Axial D64/L1 epochs | Raw delta | Calibrated delta | Harm | Repair | Gap recovered |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 30 | +0.009961 | +0.024660 | 2.47% | 4.57% | 0.22% |
| 25,000 | 12 | +0.107750 | +0.107750 | 5.19% | 10.87% | 2.33% |
| 50,000 | 6 | +0.219266 | +0.221817 | 6.89% | 15.65% | 4.75% |
| 99,356 | 3 | +0.242468 | +0.256074 | 5.79% | 16.26% | 5.25% |
| 99,356 | 9 (epoch 7 selected) | +0.284985 | +0.285836 | 7.32% | 18.29% | 6.18% |

## Numbered findings

1. **C1a fails:** retaining all candidate identities in a flat 240-node mixer does not beat the axial inductive bias under the frozen comparison.
2. **C1b is inconclusive:** the compatibility point estimate is positive but small, single-seed, and its paired interval crosses zero.
3. **The surrogate can optimize the wrong behavior:** flat-compatibility's train objective falls from `0.3743` to `0.3078`, while raw delta falls from its epoch-4 peak `+0.1067` to `-0.1068` at epoch 9 and harm rises to 198/1,175 blocks.
4. **Data is necessary but no longer sufficient:** the learning curve is steep through 50K and then flattens; scaling the same head/loss alone is unlikely to close a `1.6188` EAL gap to Domino.
5. **Hard-token identification is the actionable bottleneck:** the best D64 run repairs only 18.29% of first-miss opportunities and succeeds mostly on rank 2, despite a large K16 oracle ceiling.

## Suggested next experiments

1. Capacity sanity on a tiny fixed subset: Candidate-D-PACE versus Head-AUF; both must memorize without harm.
2. Frozen objective screen on matched axial D64/L1 and OPB-25K: Candidate-D-PACE control, pure Head-AUF, and Head-AUF plus 0.1 coverage CE. Primary metric is raw EAL; report suffix-loss mass, repair, harm, and rank buckets.
3. Only if a Head-AUF cell beats the control, confirm on full OPB-99,356 with three seeds and prompt-cluster intervals.
4. If the objective screen cannot improve materially, run the separately preregistered 20M–50M frozen-feature ceiling diagnostic. A ceiling near `+0.3` routes to DFlash representation adaptation rather than another selector sweep.

Sources: `artifacts/training/gcls_v2_representation_10132458/*/metrics.json`, `artifacts/training/gcls_v1_open_perfectblend_*/*/metrics.json`, and `artifacts/analysis/domino_phase3_validation_select_10129790.json`.
