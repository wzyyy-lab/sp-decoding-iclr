# R081 Capacity Result-to-Claim (2026-08-05)

## Binding verdict

Fresh, capacity-only GPT-5.6-Sol ultra review selected **Route B** with high confidence:

- capacity behavior: `PASS`;
- saved machine verdict: `FAIL`, preserved unchanged;
- failure class: deterministic evaluator/adjudicator schema false-negative;
- claim scope: same-subset capacity/plumbing only;
- independence: same-family, therefore provisional;
- integrity: warning because the deterministic evidence checker is unavailable, raw gradient tensors were not retained, and global no-retry cannot be proven inside the strict allowlist.

The first reviewer independently reached the same arithmetic result but accidentally previewed a few later-split manifest identity fields. Its arithmetic is retained, while its routing authority is invalidated. The binding second reviewer preserved the strict capacity-only boundary.

## Reconstructed selected checkpoint

The frozen job `10138104` contains 321 checkpoints for passes 0 through 320 and 5,120 total updates. Pass 70 at update 1,120 is the earliest exact minimum-loss checkpoint and is byte-identical to `selected.pt` (SHA-256 `8bc70170…c684`). Its saved-record behavior is:

| Criterion | Requirement | Reconstructed | Verdict |
|---|---:|---:|---|
| records / unique prompts | 512 / 512 | 512 / 512 | PASS |
| composition | 256 / 128 / 128 | 256 beneficial / 128 harmful / 128 neutral | PASS |
| loss | <=5% of 0.1153645888 | 0.0 | PASS |
| benefit recall | >=254/256 | 256/256 | PASS |
| harmful KEEP | >=127/128 | 128/128 | PASS |
| utility optimal | >=509/512 | 512/512 | PASS |
| harmful APPLY | <=1 | 0 | PASS |
| oracle recovery | [0.95,1.000001] | 1.0 | PASS |
| regret-bound violations | 0 | 0 | PASS |
| finite values/gradients | true | true | PASS |

Neutral APPLY is `90/128`; this is non-gating because neutral accepted-length regret is zero, but it forbids any neutral-conservatism or calibration claim.

## Exact false-negative

The evaluator computes harmful KEEP but serializes it only as `harm_avoidance_numerator`. The gate independently requires `harmful_keep_count`; the missing lookup is caught and converted to `False` before substantive gate evaluation. All 321 rows show the same omission and satisfy:

```text
harm_avoidance_numerator
  = harmful_count - harmful_apply_count.
```

For pass 70 this value is `128 - 0 = 128`. Adding only that alias in memory changes the gate result from false to true and leaves earliest-minimum selection at pass 70.

## Binding remediation boundary

Authorized now:

1. introduce a new adjudication schema/version;
2. emit `harmful_keep_count` alongside the legacy numerator;
3. fail closed on alias, partition, denominator, or integer-count inconsistency;
4. add compatibility and tamper tests;
5. run exactly one CPU-only deterministic offline replay against pinned job-10138104 artifacts into a separate append-only receipt;
6. obtain a fresh review of patch, tests, and receipt.

Not authorized: any artifact overwrite, GPU retry/retraining, parameter/checkpoint/seed/threshold change, R082, falsifier, validation, reserved/formal access, or C1/C2/generalization claim.

Trace: `.aris/traces/result-to-claim/2026-08-05_run06/`.
