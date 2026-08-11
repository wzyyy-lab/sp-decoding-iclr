# CAMRS Development Prelaunch Control Freeze

Frozen at 2026-08-05 04:56:31 +0800, before observing any CAMRS full-data
development outcome. This record authorizes no run by itself; the development
implementation still requires a fresh experiment-bridge code-review `GO`.

## Direct producer

- Slurm array task: `10133585_0` (`camrs-direct-control`), resubmitted from the
  already reviewed D64 task of `scripts/slurm/gcls_v4_feature_100k.sbatch` with
  command-line-only overrides `--partition=debug --time=00:30:00 --array=0`.
- Accounting: `COMPLETED`, exit `0:0`, 2026-08-05 04:32:13--04:54:26 +0800,
  elapsed 00:22:13, NVIDIA A40.
- The original formal D64/D640 array `10132819_[0-1]` and dependent summary
  `10132820` were not modified or cancelled.
- Run directory:
  `artifacts/training/gcls_v4_feature_100k_10133585/compact_axial_additive_d64_full_seed0`
- `metrics.json` SHA256:
  `9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef`
- `best.pt` SHA256:
  `9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e`
- stdout SHA256:
  `867fa686af3aa5af639266f48a24f2b02b3fda7a773fb7769e6ff2da272f196e`
- empty stderr SHA256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

The checkpoint is selected at epoch 2 by the preregistered Direct-native rule.
Its frozen configuration is global axial-additive D64/H4/L1, K=16, seed 0,
batch 64, three epochs, AdamW learning rate 0.0006, zero weight decay, warmup
ratio 0.04, gradient clip 1.0, `candidate_dpace` with alpha 0.5, and 37,221
optimizer steps over all 793,989 blocks / 99,356 prompts in the eight frozen
Open-PerfectBlend parts. Development selection uses the 1,175-block / 147-prompt
`validation_select` split. The source data, external-train prompt set, checkpoint
configuration, source hashes at start/end, and target embedding identity are all
recorded in `metrics.json`.

Pinned Direct implementation hashes:

- `scripts/train_global_direct_selector.py`:
  `e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110`
- `src/sph/global_direct_selector.py`:
  `f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06`

## One-edit control evaluator

- Slurm job: `10133586`, submitted with `afterok:10133585` and
  `DIRECT_JOB_ID=10133585` using `scripts/slurm/evaluate_direct_one_edit.sbatch`.
- Accounting: `COMPLETED`, exit `0:0`, 2026-08-05 04:54:26--04:54:37 +0800,
  elapsed 00:00:11, NVIDIA A40.
- Artifact:
  `artifacts/analysis/fmas_gate2/direct_one_edit_10133585.json`
- Artifact SHA256:
  `b97c003745a96d2f2dde7e24425cf30d8e6c321257b746e922a794696e732b25`
- stdout SHA256:
  `a2cb055f4da17a0aca06623ff76a738bc42927ffce94df3148f8346ce80a5722`
- empty stderr SHA256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

Pinned evaluator inputs:

- evaluator `scripts/evaluate_direct_one_edit.py`:
  `802abd7fd8715e67a6b2cab5f33056a9ce5e17fb9af34723b0dd080850c450fd`
- wrapper `scripts/slurm/evaluate_direct_one_edit.sbatch`:
  `b6a1e13086abfb105f90dba684e0ac7644d0d23fd307b5ac0f46668ba1b569b9`
- FMAS semantics `scripts/train_first_miss_action_selector.py`:
  `2e27b4500fdd6d440078e40378c2aa56c09f09227072d7ffdc48bfaafddbcd10`
- FMAS head `src/sph/first_miss_action_selector.py`:
  `332fac8948c61b2287cb861988b700c8a249a91cec6f808c0010d47fe7260cef`
- canonical loader `src/sph/data.py`:
  `c811701d0ec097afa86e594946857290bcfff80e2cfc2e8638f0bcdaffcc0742`
- physically isolated validation metadata:
  `b63be7bbfd56651aadbee57a819bfe0afb39395b1601b5ea4fc1564cc9f933d7`
- physically isolated selected manifest:
  `1496caa3d71ce64de9cd3fc2c29e40be60e9b636a988c9b400a0712e3ee5e811`

The evaluator reconstructs the exact Direct checkpoint, evaluates the physically
isolated `validation_select` collection, and asserts that its recomputed base and
Direct-native summaries match `metrics.json` field-by-field at absolute tolerance
`1e-12`. The one-edit decoder chooses exactly one action from `KEEP` or the global
maximum Direct residual margin over all position/candidate edits. It is a decoder
control, not a separately trained FMAS model. Per-block examples retain the base,
Direct-native, one-edit, and exact single-edit-oracle realizations.

## Frozen control values

All EAL values below are raw prompt-balanced accepted draft tokens; verification
advance is EAL + 1 and is not substituted into any gate.

| Control | EAL | First-token accuracy |
|---|---:|---:|
| DFlash / base | 5.112001943634597 | 0.8757446808510638 |
| Direct-native | 5.334669582118561 | 0.8817021276595745 (1,036/1,175) |
| Direct one-edit | 5.212099125364432 | 0.8782978723404256 |
| Exact one-edit oracle | 6.644071914480078 | 0.9906382978723405 |

Direct-native improves over DFlash by 0.222667638483964 EAL. The one-edit decoder
improves over DFlash by 0.100097181729835 EAL but changes 896 blocks, improves 78,
is neutral on 789, and harms 29 (2.4681%); it recovers 6.5335% of the exact
single-edit oracle gap. These controls therefore establish that unconstrained
Direct editing has real signal but also substantial neutral editing and safety
cost, precisely the failure mode CAMRS is intended to address.

## Frozen CAMRS development gate

The unique D64/H4/L1 seed-0 CAMRS development run must satisfy every condition:

1. CAMRS EAL - DFlash EAL is strictly greater than `0.28499`, so CAMRS EAL must
   be strictly greater than `5.396991943634597`.
2. CAMRS EAL - Direct-native EAL is at least `0.05`, so CAMRS EAL must be at
   least `5.384669582118561`.
3. CAMRS EAL - Direct-one-edit EAL is at least `0.05`, so CAMRS EAL must be at
   least `5.262099125364432`.
4. Harmed fraction is at most `0.05` (at most 58 of 1,175 blocks).
5. CAMRS first-token-correct count may trail Direct-native by at most one block,
   hence must be at least 1,035/1,175.
6. Counts, prompt identity, per-block order/oracle/base realizations, summaries,
   finiteness, all source hashes, and all three frozen Direct artifact hashes must
   match exactly or the run fails closed.

The binding EAL requirement is condition 1. No seed replication, D640 run,
threshold tuning, calibration, formal-test evaluation, or post-outcome protocol
change is authorized by this freeze.
