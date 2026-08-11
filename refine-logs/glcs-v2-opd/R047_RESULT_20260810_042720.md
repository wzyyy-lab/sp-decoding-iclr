# R047 Current-Anchor Target Early Exit Result

**Date:** 2026-08-10 04:27 CST  
**Status:** stopped by the pre-registered step-200 kill gate

## Validated mechanics

- Full `validation_select`: 147 prompts / 1,175 blocks.
- Released full-B16 baseline reproduced exactly:
  `7.23955296404276`.
- K16-union all-position oracle: `10.999878522837706`.
- New trainable parameters: exactly 409,600.
- Cached full-replay versus incremental target-KV feature alignment passed.
- A trained step-8 nonzero-residual checkpoint produced zero token-path or EAL
  difference on the 24-block incremental alignment smoke.
- No semantic mismatch, OOM, or non-finite loss/gradient was observed.

## Held-out efficacy

| Step | Fixed full-B16 EAL | Delta vs Domino |
|---:|---:|---:|
| 0 | 7.239552964 | 0 |
| 50 | **7.300777454** | **+0.061224490** |
| 100 | 7.283770651 | +0.044217687 |
| 150 | 7.217444121 | -0.022108844 |
| 200 | 7.241618076 | +0.002065112 |

Best step 50 had 144 gained versus 72 lost accepted tokens and 95% paired
bootstrap interval `[-0.00850, 0.13605]`. Domain deltas were chat
`-0.01563`, code `-0.01276`, and math `+0.20750`.

## Decision

The current-anchor feature has a real signal substantially larger than R046,
but it does not approach the step-200 `7.50` gate and the curve decays after
step 50. Job 10164747 was cancelled after step 200. No rank, learning-rate, or
same-representation data sweep is authorized.

The next mechanism must expose target-side states along the realized draft
prefix so the original first rejection is verifier-exact. A diagnostic on the
same held-out set shows why this is the right scope change:

- exact repair of only the original first rejection, leaving all suffix tokens
  untouched: `8.474125364`;
- with the deployed Top15+released K16 candidate restriction: `8.336127308`;
- two sequential reachable frontier repairs: `8.973517979`.

Thus a one-repair system has almost no error margin over the exact 15% target;
R048 should support up to two confidence-gated iterative repairs.

