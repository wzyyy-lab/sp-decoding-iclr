# CAMRS Experiment-Bridge Code Review

**Final verdict:** GO  
**Authorization:** exactly one D64/H4/L1 seed-0 512-block capacity job; no
full data, D640, extra seeds, longer training, threshold changes, or formal
evaluation.

## Initial review and remediation

The first review found the mathematics, real-data geometry, training loop,
source pins, and Slurm contract sound, but returned NO-GO because the gate
could accept unreachable malformed summaries such as `-inf` hinge, `+inf`
slack/gap, over-counts, or negative harm counts. The trainer was hardened to
require finite continuous values, nonnegative hinge, bounded sub-counts, gap
`<=1+1e-6`, and exact 512-block/459-prompt cardinalities. Regression tests now
cover every malformed state.

Focused re-review replayed every adversarial mutation; all fail, while the
exact passing boundary remains valid. No blockers remain.

## Independent real-data audit

- All `512*226=115,712` dense utilities match token-ID brute force exactly.
- Geometry: 256 beneficial, 57,179 neutral, 57,765 harmful edits; 256
  repairable blocks, exactly one positive each; 462 oracle-gain tokens; 459
  prompts; no duplicate top-K candidate IDs.
- DFlash prompt-balanced EAL `7.4411764705882355`; one-edit oracle
  `8.356572258533044`.
- Real `1/15` targets at `s=v` yield bit-exact zero hinge and zero gradient.
- Randomized 16,384-block bound audit has no violation.
- Tie behavior, residual gradients, earliest-min checkpointing, manifest
  mismatch rejection, and prompt hash are correct.

## Verification

- Focused suite after remediation: 23 passed.
- Full suite: 211 passed plus 3 subtests.
- Python compilation: passed.
- Slurm syntax: passed.
- Direct trainer/head pins remain
  `e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110`
  and
  `f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06`.

## Frozen hashes

| File | SHA256 |
|---|---|
| final proposal | `4e0ed0f8208de0a7ab2e736e78a57323f72eb54e12b5b3b91063a9ef67137f22` |
| CAMRS head | `1a7a89be1ca525fc1c10ad61d22a006c2f7525210f4358699befa926f7e2d0ce` |
| CAMRS trainer | `ca7e9a30e2f2450fde856ccd313b0b49e8ad31866016882a26fd33da4d980379` |
| selector tests | `e6c85290502a1d831a563562c80ea8cc5ee868951cb6e11bcd7ffb5e5bba5080` |
| training tests | `0a76c3f89735937fb6513defcb616190a45182ef129dd0d54ca5a85e87dc241e` |
| Slurm | `c122f024e17dd704950cc3f148f3302a8f56f06e4407b71759bc8a9484e7f26e` |
| capacity manifest | `d60613a00fc8557f4ff227ec302ced42de6a071d030b7ae7eb9eb5120bf5b67f` |

The mixed-root deserialization exception remains limited to this adaptive
same-subset Gate 1; no validation record enters model or metrics. A capacity
PASS would still require a new result-to-claim/code authorization before any
development work.
