# R081 Capacity Result Evidence (2026-08-05)

## Frozen execution outcome

- Sole training job `10138104`: Slurm `FAILED 2:0`; extern `COMPLETED 0:0`;
  elapsed 117 seconds on one A40.
- Exit 2 is the preregistered scientific-failure path. Stderr is empty and the
  full run directory was atomically published; no retry is authorized.
- Metrics SHA-256:
  `6c5a34c1454a0cc513587e0646615d529b711747da2656d84070e3d84aa707a6`.
- History / selected checkpoint / selected records SHA-256:
  `3cbab50d...e4b16` / `8bc70170...c684` / `1f55087c...5d0ed`.
- Stdout/stderr SHA-256:
  `1796cfccf98c2cc6c89c82eccc26da53521ce0ddafdbf94e640e402bfe76fd44`
  / `e3b0c442...b855`.
- Exactly 321 checkpoints and history rows exist for passes 0 through 320.
  Earliest exact minimum is pass 70 after 1,120 updates.

## Saved verdict versus selected evidence

`metrics.json` records `capacity_gate_passed=false` and
`scientific_status=FAIL`. The selected pass-70 row records:

| Frozen conjunct | Requirement | Selected | Surface verdict |
|---|---:|---:|---|
| finite values/gradients | true | true / true | PASS |
| regret-bound violations | 0 | 0 | PASS |
| loss relative to epoch zero | <=5% of 0.1153645888 | 0.0 | PASS |
| beneficial APPLY | >=254/256 | 256/256 | PASS |
| harmful KEEP | >=127/128 | 128/128 reconstructed | PASS |
| utility optimal | >=509/512 | 512/512 | PASS |
| oracle recovery | [0.95, 1+1e-6] | 1.0 | PASS |
| harmful APPLY | <=1 | 0 | PASS |

The selected row additionally has 512 records/prompts, exactly 256
beneficial / 128 harmful / 128 neutral, method EAL equal to oracle EAL, and
zero decoded regret.

## Deterministic adjudicator failure

`capacity_gate_passes()` requires the key `harmful_keep_count`. Its guarded
lookup converts any missing key to `False`. However,
`reconstruct_saved_gate_evaluation()` does not emit that key; it emits the
same integer only as `harm_avoidance_numerator` while also emitting
`harmful_count` and `harmful_apply_count`.

This is systematic rather than checkpoint-specific:

- all 321 history rows lack `harmful_keep_count`;
- source search finds the key only in the gate adjudicator, never in the
  evaluator output or saved results;
- selected harmful KEEP reconstructs exactly and redundantly as
  `128 - 0 = 128` and `harm_avoidance_numerator = 128`;
- replaying the current adjudicator on the saved selected row returns `False`
  solely because the required key lookup raises `KeyError` before the visible
  conjuncts are evaluated.

No code has been changed and no rerun/retry has occurred. A fresh
`result-to-claim` reviewer must decide whether the capacity claim is truly
unsupported, or whether the frozen run supports the same-subset plumbing claim
despite an instrumentation-only false negative. It must also decide whether a
minimal offline adjudicator repair/replay is permissible; it may not authorize
retraining, threshold changes, a different checkpoint, or later-stage data
access by implication.
