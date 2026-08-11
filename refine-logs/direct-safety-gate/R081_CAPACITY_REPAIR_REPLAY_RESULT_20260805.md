# R081 Capacity Repair Replay Result (2026-08-05)

## Execution

The sole authorized CPU-only offline replay completed successfully on its first and only invocation.

- replay protocol: `pros-gate-capacity-offline-replay-v2`;
- adjudication schema: `pros-capacity-adjudication-v2`;
- output: `artifacts/adjudication/pros_gate_capacity_10138104/capacity_adjudication_v2.json`;
- receipt SHA-256: `17ac807e5b599c45e414958786fade47d7b2a0e1fd5603c3726d531eb143a352`;
- device: CPU;
- training/optimizer steps: 0;
- selected checkpoint: pass 70, update 1,120;
- repaired replay verdict: capacity gate PASS.

No retry occurred or is authorized.

## Immutable inputs

All pinned hashes were identical before and after replay:

| Artifact | SHA-256 |
|---|---|
| original metrics | `6c5a34c1454a0cc513587e0646615d529b711747da2656d84070e3d84aa707a6` |
| history | `3cbab50df060a67880fa8905def457da416925e7050296866bb91124b63e4b16` |
| selected checkpoint | `8bc70170a67dae1b6e2bac74929a5c6fac83debae16eb7cffffc41658716c684` |
| pass-70 checkpoint | `8bc70170a67dae1b6e2bac74929a5c6fac83debae16eb7cffffc41658716c684` |
| selected records | `1f55087ce1854f457050d5a1de5b40f38f73d2f4904ff90a614a556b98a5d0ed` |
| checkpoint manifest | `f8e8f5f579760cd669df3f5c2206420dd9d2c028dcb0145d1a9984caa7e9e3e4` |
| pass diagnostics | `e9904b00b5b2d1991d9e74e97cd1f6e92dbd418345bc89d8b980da9c301b81bf` |
| order manifest | `91427e920d506c68b40d7fee831911cd69d3dae10ddb1d5ef422d056e64eec90` |

The original machine artifact remains `scientific_status=FAIL` and `capacity_gate_passed=false`; it was not overwritten or relabeled.

## Versioned adjudication

The replay added exactly one in-memory legacy alias:

```text
harmful_keep_count
  = harm_avoidance_numerator
  = harmful_count - harmful_apply_count
  = 128.
```

The denominator equals harmful count, and all invariants pass. The repaired selected row reconstructs:

- 512 records / 512 prompts;
- 256 beneficial APPLY out of 256;
- 128 harmful KEEP and 0 harmful APPLY out of 128;
- 512 utility-optimal decisions;
- loss 0 versus epoch-zero 0.11536458879709244;
- oracle recovery 1.0;
- zero regret-bound violations;
- finite values and frozen-run finite-gradient witness.

Neutral APPLY remains 90/128 and is explicitly non-gating/non-calibration evidence.

## Source and publication closure

The explicit 60-file repair closure verified both before and after replay:

- manifest SHA-256: `b072758abf6aabc7f7af39d52db0327d9d749192f1be812c3da7cc5fe735f8f2`;
- entries SHA-256: `77894a782c151bba34c01dfd89e1482313669e104832905f962afca7a4e46f92`.

Receipt publication used a fully written/fsynced same-directory temporary file and atomic no-clobber hard-link publication. The target did not previously exist and is now append-only.

R082 and every later data stage remain blocked pending a fresh `result-to-claim` review of this receipt.
