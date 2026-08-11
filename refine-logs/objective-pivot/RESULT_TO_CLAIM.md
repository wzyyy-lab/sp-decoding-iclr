# Reachable-Support Capacity Result-to-Claim

Date: 2026-08-04
Artifact: `artifacts/training/gcls_v3_reach_capacity_10132646/reach_capacity_summary.json`
Fresh-review assurance: same-family Codex; provisional
Claim supported: **no**
Routing: **close reachable-support objective route**

## Binding result

| Cell | Candidate | Hard candidate | Repair | Oracle gap | Harm | Gate |
|---|---:|---:|---:|---:|---:|---|
| `lambda=1` control | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | pass |
| `lambda=0` hard | 0.989313 | 0.940639 | 1.000000 | 0.949495 | 0.000000 | fail |
| `lambda=0.1` soft | 0.990840 | 0.949772 | 1.000000 | 0.959596 | 0.000000 | fail |

The all-three-cell gate is therefore scientifically negative. The control
shows that the architecture, data path, optimizer, and budget can fit the
same-subset probe. Both reach-censored treatments specifically lose
fixed-coverage hard-candidate classification capacity.

## Supported statement

On this 128-block same-subset capacity probe, weakening direct supervision
after the selector's current breaker preserves repair and observed safety but
starves hard alternatives enough to fail the preregistered capacity gate.

## Unsupported statements

- No causal EAL improvement is established.
- No development, cross-seed, confidence-interval, or test claim is allowed.
- The result does not prove all support-aware objectives are ineffective.
- The OPB-25K comparison and full-data confirmation are gated off and must not
  be run for this route.

## Next route

No rescue, threshold relaxation, or new `lambda` cell is authorized. The
broader program returns to the separately preregistered positive-only
D640/H10/L4 frozen-feature probe to distinguish compact-head capacity from a
frozen-input limitation. That probe is diagnostic and cannot resurrect the
closed reachable-support claim.
