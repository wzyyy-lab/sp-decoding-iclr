# SAVS Capacity Result-to-Claim

This immutable snapshot is identical in verdict and scope to
`refine-logs/first-miss-value/CAPACITY_RESULT_TO_CLAIM.md` at
2026-08-05 03:14:37 +08:00.

**Verdict:** `claim_supported = no`; job `10133339` is a scientific
`FAIL-CLOSE` and Gate 2 is forbidden.

The frozen D64/H4/L1 action-uniform-MSE capacity run passed aggregate RMSE
(`0.006909`), harmful-nonpositive recall (`1.0`), selected harm (`0.0`), and
positive-count (`256`) checks, but failed beneficial-sign recall (`0.78125`
versus `>=0.99`) and one-edit oracle-gap recovery (`0.445458` versus
`>=0.95`). No epoch passed: maxima were `0.792969` sign recall and `0.474415`
gap recovery.

The output-head epoch-zero harmful/beneficial gradient-norm ratio is
`1,518.6x`, and beneficial errors contribute `85.06%` of selected-checkpoint
SSE. This supports a positive-gradient-starvation-consistent diagnosis, not a
unique causal attribution. Full data, continuation, D640, post-hoc threshold,
class-weight rescue, and extra seeds are forbidden. The next allowed route
must be a newly preregistered mechanism with a fresh capacity contract and
review. Full evidence and scope restrictions are in the latest file and
`.aris/traces/result-to-claim/2026-08-05_run03/`.
