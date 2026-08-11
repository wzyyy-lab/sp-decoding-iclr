# CAMRS Capacity Result-to-Claim

**Run:** Slurm `10133549`  
**Verdict:** `claim_supported = yes`, `PASS-ADVANCE`  
**Confidence:** high  
**Assurance:** same-family Codex review, provisional  
**Integrity:** unavailable — no formal `EXPERIMENT_AUDIT.json`  
**Deterministic checker:** unavailable; the reviewer reconstructed every cited value directly

## Binding result

The exact seed-0 D64/H4/L1 axial-additive, tie-safe CAMRS procedure passed the
frozen same-subset Gate 1. The earliest exact minimum-hinge checkpoint is epoch
`98` (after `1,568` updates); checkpoint search continued through all `5,120`
updates. The selected checkpoint itself passes, so no later or nonselected
epoch is being used as a rescue.

| Gate item | Frozen threshold | Selected result |
|---|---:|---:|
| finite hinge/slack/gap | required | `0.0 / 0.0 / 1.0` |
| bound violations | `0` | `0` |
| mean block hinge | `<=0.0030078125` | `0.0` |
| beneficial strictly positive | `>=254/256` | `256/256` |
| utility-optimal selected | `>=244/256` | `256/256` |
| harmful nonpositive | `>=57,188/57,765` | `57,629/57,765` |
| prompt-balanced oracle-gap recovery | `>=0.95` | `1.0` |
| selected harmful | `<=5/512` | `0/512` |
| no-benefit false edits | `<=2/256` | `0/256` |
| oracle-gain tokens / blocks / prompts | `462 / 512 / 459` | exact |
| epoch-zero identity | required | exact |

The selected CAMRS prompt-balanced EAL is `8.356572258533044`, exactly equal
to the single-edit oracle on these 512 records, versus base
`7.4411764705882355`. There are 243 jointly passing epochs; epoch 71 is the
first gate pass, while epoch 98 is the earliest exact zero-hinge minimum.

## Integrity and reconstruction

The reviewer independently replayed all 512 saved examples and all 115,200
edit actions. The decomposition is exactly 256 beneficial, 57,179 neutral,
and 57,765 harmful actions. Strict-positive deployment, KEEP preference,
lowest-index ties, oracle actions, competitors, hinge, regret, slack, EAL, and
all twelve `.aris/claims.json` values reconstruct exactly.

`best.pt` contains 36 finite FP32 tensors and 433,772 parameters; its config
equals `metrics.json.config`. All ten declared source hashes match at run start,
run end, source snapshot, and the current files. Slurm completed `0:0` on an
A40 in `00:03:28`; stderr is empty.

Primary artifact hashes:

- metrics: `b40c1640aed08644a23ab1947e7b360dc2e08aa63566ad5d241e863559fbc681`
- checkpoint: `736d1c9c1f13ccb563ba3987ad0b85bd7c3a3d0b8192b529f6f370d231fa3e97`
- stdout: `7daad663e25792643b740d058f3bd6e5fdf3436dd260e57772fb7b0f4138cce2`
- empty stderr: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Supported and unsupported claims

Supported: this exact head, objective, optimizer/schedule, manifest, and
checkpoint rule have sufficient same-subset representational/optimization
capacity to realize the utility-optimal one-edit policy within the frozen
budget. CAMRS operationally removes the SAVS sparse-positive/max-selection
failure on this capacity subset.

Unsupported: held-out or population generalization, frozen-feature sufficiency
for unseen examples, identifiability rather than memorization, safety,
calibration, robustness, seed stability, superiority to external Direct
controls, or paper-level evidence. The capacity artifact's changing internal
`direct_native` diagnostic is not a valid frozen Direct control.

The initial CAMRS oracle-upward projection gradient norm is `0.2035767`, and
all 256 repairable blocks are active, unlike SAVS's diluted positive signal.
This is mechanism-consistent evidence, not a unique causal proof. Shared
projection cancellation remains, and 136 harmful actions have positive scores
at the selected checkpoint even though none wins deployment.

## Binding route

The result authorizes development implementation and a fresh code review only.
Before any CAMRS development GPU launch, exact external Direct-native and
Direct-one-edit artifact paths, hashes, configs, and evaluation semantics must
be frozen and validated. After that precondition and a fresh `GO`, exactly one
seed-0 run may use 99,356 prompts / 793,989 blocks / 37,221 updates and the
physically isolated 147-prompt / 1,175-block validation collection.

Seeds 1/2, repeats, continuation, longer training, D640, changed margins or
weights, calibration, threshold tuning, formal-test access, and rollout remain
forbidden.

Reviewer trace: `.aris/traces/result-to-claim/2026-08-05_run04/`.
