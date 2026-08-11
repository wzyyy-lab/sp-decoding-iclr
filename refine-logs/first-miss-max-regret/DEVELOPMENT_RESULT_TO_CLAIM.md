# CAMRS Development Result-to-Claim

Reviewed 2026-08-05 with a fresh same-family GPT-5.6-Sol xhigh agent under
the ARIS `result-to-claim` gate.

- **claim_supported:** no
- **gate:** FAIL-CLOSE/PIVOT
- **confidence:** high
- **acceptance:** provisional
- **integrity:** unavailable; deterministic artifact reconstruction passed, but
  no independent-model `EXPERIMENT_AUDIT.json` exists.

The exact preregistered CAMRS development procedure does not pass. Its binding
checkpoint is epoch 0 DFlash identity, not a trained checkpoint.

## Frozen gate reconstruction

| Criterion | Required | Reconstructed | Result |
|---|---:|---:|---|
| finite gate values | all finite | all finite | pass |
| CAMRS - DFlash EAL | >0.28499 | 0.0 | **fail** |
| CAMRS - Direct-native EAL | >=0.05 | -0.22266763848396387 | **fail** |
| CAMRS - Direct-one-edit EAL | >=0.05 | -0.10009718172983462 | **fail** |
| harmed fraction | <=0.05 | 0/1175 | pass |
| first-token shortfall vs Direct | <=1 block | 7 (1029 vs 1036) | **fail** |
| validation blocks/prompts | 1175/147 | 1175/147 | pass |
| example identity/order/oracle | exact | zero mismatches | pass |
| frozen source/control identities | exact | exact | pass |

Selected prompt-balanced EAL is `5.112001943634597`; the binding absolute
target was strictly above `5.396991943634597`. Frozen controls are
Direct-native `5.334669582118561`, Direct-one-edit `5.212099125364432`,
and exact single-edit oracle `6.644071914480078`.

## Complete trajectory

| Epoch | Steps | Train hinge | Val hinge | EAL | Delta vs base | beneficial >0 | utility-optimal | harmful >0 | harmed | false edits/no-benefit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | - | 0.4430071104 | 5.1120019436 | 0 | 0/984 | 0/984 | 0/90120 | 0/1175 | 0/191 |
| 1 | 12407 | 0.2139467557 | 0.2011940502 | 5.0371720117 | -0.0748299320 | 79/984 | 34/984 | 637/90120 | 67/1175 | 55/191 |
| 2 | 24814 | 0.1993434878 | 0.2004279790 | 5.0031584062 | -0.1088435374 | 81/984 | 29/984 | 908/90120 | 86/1175 | 80/191 |
| 3 | 37221 | 0.1953414814 | 0.1976001873 | 4.9839650146 | -0.1280369291 | 101/984 | 53/984 | 1326/90120 | 108/1175 | 58/191 |

All epochs retain zero pointwise-bound violations and minimum slack 0. The
lexicographic keys are headed by EAL, so epoch 0 is uniquely best; strict
`key > best_key` also preserves earliest exact ties. The progress log prints
epoch 0 before its post-training `is_selected` flag is assigned; the final
artifact correctly identifies it.

The selected example replay reconstructed the exact 226-action geometry:
984 beneficial, 173,271 neutral, and 90,120 harmful edits. Every selected
action score is zero and every block selects KEEP. Base/CAMRS paths, accepted
lengths, and first-token results match on all 1,175 blocks. Cross-artifact
alignment against the frozen Direct control had zero mismatches.

## Artifact and protocol integrity

- Slurm job `10133649`: expected scientific `FAILED 1:0`, elapsed 19:57,
  NVIDIA A40; metrics were atomically written before the deliberate exit.
- Metrics SHA256:
  `5f856c49be6bfd02e6eb7d7bd7448a04c6c74c1415346fcef41ae5756a10a182`.
- Checkpoint SHA256:
  `4e92ef2d193baf1d42b66eed4171128e880fd4eda4f8d9824093d0f914a45fdc`.
- Stdout SHA256:
  `29023a99af8e78d14509c158a11b3167bb3e0250880d35cabfdb9e908e61cf92`.
- Empty stderr SHA256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- The checkpoint contains 36 finite FP32 tensors and 433,772 parameters.
- All 18 captured source/document/wrapper files satisfy
  start hash = end hash = current hash = source snapshot.
- Direct control, Direct metrics, and Direct checkpoint retain their exact
  frozen hashes at run start and end.
- The eight physically train-only collections contain exactly 793,989 blocks
  and 99,356 prompts, with train prompt hash
  `45471a62f93a488f3f7653c096bebcddb0ddae3773f6c99744bd070e348a9405`.
- Physical validation contains exactly 1,175 blocks / 147 prompts, with prompt
  hash `278c27e266e50c6b81b94a88bd8dbf5dc2645563add738db7536f2489a01edaa`.
  Prompt overlap is zero and every canonical shard passed size/SHA validation.

The external-train metadata hashes were verified before all shards were loaded
and still match, but the artifact lacks separate run-recorded `_at_end` fields
for those eight metadata files. This cannot make the negative result ambiguous.
A second diagnostic-completeness warning is that the development artifact
records aggregate gradient norms/clipping, but not the proposal's per-epoch
oracle/competitor projection norms and cosines. It limits causal interpretation,
not the direct gate verdict.

## Diagnosis

The artifact-backed result is a deployment-boundary/max-tail failure. Validation
hinge falls from 0.4430 to 0.1976, yet EAL decreases monotonically and harmful
positive-score actions rise from 637 to 1,326. Beneficial sign recall remains
only 8.0%-10.3%, while utility-optimal accuracy remains 2.95%-5.39%. With
90,120 harmful actions, even 98.5%-99.3% harmful-nonpositive recall leaves a
large positive tail; max selection turns that tail into over-editing and harm.
The identity checkpoint is safe only because it makes no edits.

This is narrower than generic architectural incapacity: CAMRS exactly fit its
frozen 512-block same-subset capacity probe. It is not clean proof of ordinary
validation overfit either, because online train and validation hinge decline
together. Since no final train endpoint diagnostic was saved, optimization,
full-distribution feature conflict, and held-out generalization cannot be
causally separated. The defensible label is:

> Frozen full-data finite-schedule/deployment-boundary calibration failure,
> with over-editing at trained checkpoints and identity fallback under the
> preregistered selector.

Relative to FMAS, CAMRS reduces trained-checkpoint harm from 31.7%-35.0% to
5.7%-9.2% and EAL degradation from 0.424-0.505 to 0.075-0.128. It mitigates,
but does not solve, the failure. The same-subset SAVS-to-CAMRS improvement does
not transfer to full-data development.

## Claim boundary and routing

Supported:

- this exact seed-0 D64/H4/L1, full-OPB, 37,221-step procedure fails;
- its selected checkpoint is exact DFlash identity with zero gain and zero harm;
- trained checkpoints lower hinge but have negative EAL and harmful/false edits;
- the pointwise regret-bound implementation remains internally consistent.

Unsupported:

- improvement over DFlash or either Direct control;
- generalization from the capacity subset;
- generic feature/architecture impossibility or unique causal attribution;
- seed stability, safety, calibration, robustness, or paper/formal-test efficacy;
- rescue by a trained epoch, threshold, continuation, D640, or capacity result.

This exact route is closed. Read-only postmortem is allowed. A genuinely new
mechanism aimed at the deployment boundary / max-tail problem may proceed only
through new refinement and preregistration, CPU semantic tests, a separately
frozen capacity gate, and fresh code review. Seeds 1/2, repeats, continuation,
longer training, D640, post-hoc thresholds/scaling/smoothing/weights/auxiliaries,
formal-test access, rollout, and paper-facing positive claims remain forbidden.

