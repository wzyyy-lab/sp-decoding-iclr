# FMAS Gate-1 Experiment Code Review

**Final verdict:** GO — D64 capacity job authorized.  
**Review mode:** fresh read-only GPT-5.6-Sol xhigh reviewer.  
**Formal/development authorization:** none; development remains conditional on
all four D64 capacity thresholds.

## Initial NO-GO and remediation

The first review found two blockers:

1. Capacity checkpoint selection used an unregistered third tie-break
   (repairable recall) after CE and action accuracy.
2. Gate-0 tests did not explicitly assert second-pass residual-projection
   gradients or exercise improved/harm/neutral accounting; training also needed
   a nonfinite-gradient failure.

The implementation was corrected to use exactly `(-CE, action_accuracy)` with
strict-`>` earliest tie retention, added the missing gradient/accounting
regressions, rejects nonfinite loss, and invokes gradient clipping with
`error_if_nonfinite=True`.  The reviewer then independently reran the focused
checks and returned GO.

## Frozen reviewed identities

| artifact | SHA256 |
|---|---|
| FMAS trainer | `8d8e6da24e1097a4509cb14ac4464f5ed47249cd2de75ad123f661eed986ad0c` |
| FMAS head/wrapper | `332fac8948c61b2287cb861988b700c8a249a91cec6f808c0010d47fe7260cef` |
| capacity helper | `58a92a38572dfca983361e257194a5d76f0a14f7eb9a196b8d7e749f584c7238` |
| Slurm job | `a00716db1a0f09424dd39ce1329d70fdf42e90f41dbc9cde7e31bffdce4d1c2f` |
| capacity manifest file | `d60613a00fc8557f4ff227ec302ced42de6a071d030b7ae7eb9eb5120bf5b67f` |
| pinned direct trainer | `e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110` |
| pinned direct selector | `f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06` |

The manifest reconstructs exactly at 512 blocks with subset SHA256
`1c4f911cb91cc994ecdc7a26f25a9ead0dfd33a126e4f0f74d91ab94e0aec355`:
256 edit targets, 156 full-correct KEEP targets, and 100 out-of-K KEEP targets.

## Verification

- Final local suite: 167 tests passed plus 3 parameterized subtests.
- Reviewer focused suite: 15/15 passed.
- Manifest reconstruction: passed.
- Slurm syntax and 5,120-step arithmetic: passed.
- Four inclusive thresholds and fail-closed exit: passed.
- Gold-free forward boundary, identity/ties, and frozen inputs: passed.

Known nonblocking limitation: the canonical collection is the already accepted
legacy `none_legacy` rank-zero-witness tier.  This is recorded provenance and
does not broaden the capacity result.

