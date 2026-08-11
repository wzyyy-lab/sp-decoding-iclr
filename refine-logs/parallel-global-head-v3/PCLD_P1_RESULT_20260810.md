# PCLD-16R P1 capacity result

## Verdict

PCLD006 completed on A40 as Slurm `10168532`.  The selected checkpoint is step
6000.  The frozen capacity gate failed, so PCLD-16R cannot advance to disjoint
P2 and cannot be rescued with a width, schedule, threshold, or loss sweep.

## Selected result

| Metric | Result | Gate | Verdict |
|---|---:|---:|---|
| Same-set EAL | 9.525390625 | diagnostic | strong fit signal |
| Pure DFlash EAL | 4.8984375 | baseline | — |
| Released Domino EAL | 6.57421875 | baseline | — |
| Teacher candidate agreement | 99.9875746% | >=99% | PASS |
| Base-to-oracle gap recovery | 66.9209040% | >=95% | FAIL |
| Harmed blocks | 6.25% | <=1% | FAIL |
| Legacy strict J2 | 322/411 = 78.3454988% | >=99% | FAIL |
| Stable-support J2 | 314/314 = 100% | diagnostic only | complete stable fit |

All three domains improve over pure DFlash on this same-set capacity group.
This does not constitute held-out evidence.

## Mechanistic reading

The head can memorize the authoritative stable-support decisions: candidate
agreement is effectively exact and stable J2 is exactly 100%.  It also raises
same-set EAL by 4.626953125 over base and 2.951171875 over released Domino.
However, the stable trainable prefix contains 503 blocks/4754 rows, while the
binding historical J2 population contains 411 eligible blocks and includes 97
blocks outside stable J2.  Performance falls to 322/411 on that broader early
error population and produces 6.25% harm.  The failure is therefore not lack of
parallel-head latency or inability to fit the stable labels; it is failure to
learn a sufficiently safe full clean trajectory under the frozen supervision
support/objective.

## Evidence boundary

- Result: `artifacts/models/pcld16_capacity_10168532/report.json`
- Stdout/stderr: `artifacts/logs/pcld-cap-10168532.{out,err}`
- Frozen support receipt:
  `artifacts/manifests/pcld16_capacity_support_10168459.json`
- P0 complete eager profile: `profile_output/pcld16_eager_10168424.json`

The Slurm job exits with code 1 only because `--require-gate` converts the
completed negative science verdict into a nonzero exit.  The report and best
checkpoint are complete; no runtime/OOM/numerical failure occurred.
