# FMAS Capacity Result-to-Claim

## Evidence

- Job: `10133018`, COMPLETED 0:0 in 00:02:23.
- Exact budget: 5,120/5,120 optimizer steps.
- Model: axial-global additive D64/H4/L1, 433,772 parameters.
- Data: frozen 512-block same-subset manifest, file SHA256
  `d60613a00fc8557f4ff227ec302ced42de6a071d030b7ae7eb9eb5120bf5b67f`.
- Checkpoint: epoch 297, selected by minimum action CE then action accuracy.
- Final gate values:
  - action accuracy: `1.0` (threshold `>=0.97`);
  - repairable-action recall: `1.0` (threshold `>=0.95`);
  - single-edit oracle-gap recovery: `1.0` (threshold `>=0.95`);
  - harmed fraction: `0.0` (threshold `<=0.01`).

## Supported claim

**YES:** The reviewed D64 FMAS parameterization and optimizer can represent and
fit the frozen 226-way canonical one-edit action mapping on the preregistered
512-block capacity subset without harm.  The capacity/optimization prerequisite
for seed-0 full-data development is satisfied.

## Unsupported claims

- No prompt-disjoint or held-out action learnability has been established.
- The `+0.9154` same-subset EAL gain is not a generalization result.
- This does not show superiority to Direct-native, Direct-one-edit, DFlash, or
  Domino on development/formal prompts.
- It does not establish an information ceiling, novelty, throughput benefit,
  or a paper-level method claim.

## Binding routing decision

Gate 1 passes and therefore authorizes exactly Gate 2: D64/H4/L1 FMAS seed0 on
the full 99,356-prompt / 793,989-block OPB training collection for three epochs
and 37,221 steps, evaluated only on `validation_select`.  Seeds1/2 and the
reserved formal test remain forbidden until their later gates pass.

