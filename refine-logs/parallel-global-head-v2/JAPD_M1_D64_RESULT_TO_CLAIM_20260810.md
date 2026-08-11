# JAPD-16 M1 D64 Result-to-Claim

## Verdict

- `claim_supported: no` for the narrow claim that D64/H4/L1 is sufficient to
  enter M2 unchanged.
- Confidence: high.
- Review: same-family Codex, provisional, pending external review.
- This does **not** adjudicate the unrun held-out JAPD method claim.

## Binding results

| Gate | Model EAL | Domino EAL | J2 | Oracle recovery | Harm | Verdict |
|---|---:|---:|---:|---:|---:|---|
| J010 D64 same-set capacity | 11.314453 | 6.574219 | 100.00% | 92.7966% | 1.3672% | FAIL |
| J011 D64 full-fit diagnostic | 6.312012 | 7.330566 | 17.4518% | 4.0101% | 16.8945% | FAIL |

J010 misses `recovery>=95%` and `harm<=1%`, although it proves substantial
same-set memorization. J011 misses both binding gates by a wide margin, so D64
is not sufficiently optimized/capable on the broader same-set diagnostic.

## Routing

The frozen two-failure AND condition is satisfied. Exactly one conditional
architecture branch is authorized before fresh300:

- D256/H8/L2, exactly 4,539,888 parameters;
- the same full16 global-noncausal one-call one-chain architecture;
- identical data, objective, schedule, selection, and gates;
- all future global/local/DPACE arms use the same D256 size if M1 passes.

D64 may not proceed to M2. D256 must pass J010, J011 and the complete eager J012
profile before M2. Any scientific D256 M1 failure closes JAPD; D512, loss
retuning, schedule sweeps, serial target decoding, GRU, iteration, beam/tree,
forest, or multipath rescue is not authorized.

Primary artifacts:

- `artifacts/models/japd16_capacity_d64_10167565/report.json`
- `artifacts/models/japd16_full_fit_d64_10167566/report.json`
- `profile_output/japd16_eager_10167573.json`

Reviewer trace: `.aris/traces/result-to-claim/2026-08-10_run03/`.
