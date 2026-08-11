# SAVS Capacity Result-to-Claim

**Verdict:** `claim_supported = no`  
**Binding decision:** job `10133339` is a scientific `FAIL-CLOSE`; Gate 2 is forbidden.  
**Confidence:** high  
**Assurance:** same-family independent review, provisional; no formal integrity audit.

## Frozen evidence

- Configuration: D64/H4/L1 axial-additive residual-difference signed-value
  head, action-uniform MSE, seed 0, 320 epochs and exactly 5,120 updates.
- Data: frozen 512-train-block / 459-prompt same-subset capacity manifest;
  prompt-set SHA256
  `1e2be08968b2356f71e9818a5be5b8f3ecdd12ee50299ba6212a035f8a4d2707`.
- Action geometry: 256 beneficial, 57,179 neutral, and 57,765 harmful edit
  actions.
- Selected checkpoint: epoch 307, the unique minimum-MSE epoch, objective
  `4.7738627358739905e-05`; 433,772 finite parameters.
- Runtime: 5,120/5,120 updates in 156.32 seconds on one NVIDIA A40. Slurm
  state `FAILED 1:0` is the declared fail-closed scientific exit after the
  complete artifact was written, not a runtime exception.
- Artifact hashes:
  - metrics: `6c9a2f8676d0c4d44f8ea867a5169f43fb6fc4167f9498a4d31401e2e17d216e`;
  - checkpoint: `844c204b51e62ca95c3ec4d3dab80fce1aa47195ee962288b18c38c0f644c63a`;
  - stdout: `d61c50dfcb9d69732125acc6cfc9db30fb0f9616b1a4ebffabdf099489d414a5`;
  - stderr: `8677c55356cf0f62537785fc55c17bc8ed2800d1b1ee91bb454f322ca5149abc`.

## Binding gate

| Criterion | Result | Frozen threshold | Verdict |
|---|---:|---:|---|
| all-action RMSE | 0.00690931 | <= 0.02 | PASS |
| beneficial sign recall | 200/256 = 0.78125 | >= 0.99 | **FAIL** |
| harmful nonpositive recall | 1.0 | >= 0.99 | PASS |
| one-edit oracle-gap recovery | 0.445458 | >= 0.95 | **FAIL** |
| selected harmed fraction | 0.0 | <= 0.01 | PASS |
| beneficial-action count | 256 | exactly 256 | PASS |

No checkpoint-selection rescue exists within the run: the best beneficial
sign recall over all epochs was only `0.792969` at epoch 265, and the best
gap recovery was only `0.474415` at epoch 282.

## What is supported

On this exact same-subset probe, the frozen configuration fits the dense
action-average target to low aggregate RMSE, predicts every harmful action
nonpositive, selects no harmful edit, and improves prompt-balanced EAL from
`7.441176` to `7.848947` versus the one-edit oracle `8.356572`. It selects 83
beneficial and 272 neutral edits. These are capacity-subset descriptive facts,
not held-out method results.

## Mechanism-consistent diagnosis

Aggregate MSE hides the rare positive decisions that control the max policy.
At epoch zero, the output projection's harmful-class gradient norm is
`0.430526`, versus `0.0002835` for beneficial actions: a ratio of about
`1,518.6x`. The harmful gradient has cosine approximately `1.0` with the
total gradient, while the beneficial cosine is `-0.4979`. At the selected
endpoint, beneficial errors contribute `85.06%` of all SSE; their mean target
is `0.12031`, but their mean prediction is only `0.02129`.

This is strong evidence consistent with positive-gradient starvation and an
action-average-loss/max-selection mismatch. It is not proof that this is the
unique cause: one seed and an epoch-zero decomposition cannot isolate the
objective from representation, optimizer, schedule, or shared-head geometry.

## Unsupported claims

The run does not establish held-out generalization, population safety,
feature-information insufficiency, generic failure of value regression, or
failure of D640 or any untested objective/optimizer. In particular, low
all-action RMSE does not imply low max-policy regret.

## Binding routing

For this frozen route, full-data training, longer continuation, D640 rescue,
post-hoc thresholds/calibration, class-weight rescue, and extra seeds are all
forbidden. A new route may target the sparse-beneficial/max-selection mismatch
with a genuinely different, separately preregistered class-aware or
sign/ranking-sensitive objective. It must restart with CPU semantics, a new
frozen capacity protocol, and fresh review; this negative result alone does
not authorize implementation or GPU execution.

Reviewer trace: `.aris/traces/result-to-claim/2026-08-05_run03/`.
