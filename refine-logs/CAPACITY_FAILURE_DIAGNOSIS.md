# Capacity Failure and Rescue Review

**Initial capacity run:** 10132235  
**Aggregate:** scientific negative  
**Reviewer:** fresh second-opinion, read-only

## Observed gate result

| Condition | Accuracy | Hard accuracy | Repair | Oracle-gap recovery | Harm |
|---|---:|---:|---:|---:|---:|
| reach λ=0 | 0.8725 | 0.2511 | 0.5625 | 0.3468 | 0 |
| reach λ=.1 | 0.8710 | 0.2374 | 0.4844 | 0.2963 | 0 |
| reach λ=.25 | 0.8656 | 0.1963 | 0.4688 | 0.2323 | 0 |

All three complete artifacts failed the simultaneous gate. The post-array summary correctly exited 1 as a scientific negative, not an artifact failure.

## Root cause

The evidence points primarily to accepted-reach gradient starvation rather than demonstrated lack of model capacity.

- Of 219 hard active positions, the initial hard gold probability median is 0.0578.
- Only 0.197% of normalized ARR gradient mass reaches hard labels; median hard continuation weight is 3.25e-5 and 72.6% is below 1e-3.
- Smoothed α=.5 raises hard gradient mass to 1.658% and gives about 15.5× larger normalized first-miss weight.
- Soft expected reach rises strongly while unweighted NLL worsens, showing that the optimizer sharpens easy early survival rather than pushing low-probability alternatives across the greedy boundary.
- λ cannot help this subset because harm is already zero; increasing λ only worsens hard metrics.
- Historical additive/axial smoothed-D-PACE or uniform models pass on a nested 512-block superset, so the capacity stop rule is not yet causally identified.

Checkpoint selection and metric mismatch are ruled out: no epoch was close to passing, and repair/gap fail as badly as classification.

## One-shot rescue authorization

Exactly one three-cell capacity screen is permitted. Common budget is 1,280 steps / 40,960 example presentations, matching the historical D128 capacity budget.

- A “compat_arr_budget”: compatibility + accepted reach; budget-only control.
- B “compat_cdpace05”: compatibility + length-normalized Candidate-D-PACE α=.5; smoothing control without 15× legacy scale.
- C “additive_cdpace05”: additive + the same normalized smoothed objective; encoder control.

All other model/data/optimizer settings remain fixed; λ=0. The unchanged simultaneous gate applies at one retained checkpoint. Per-metric epoch cherry-picking is forbidden.

## Binding decision table

- A passes: retain ARR and use the 1,280-step capacity budget.
- A fails, B passes: delete the ARR/safety claim; explicitly re-freeze smoothed D-PACE and retain compatibility/full-lattice route.
- A/B fail, C passes: stop the compatibility/full-lattice thesis.
- All fail: stop the route; no fourth rescue or high-capacity probe.

Implementation: scripts/slurm/gcls_v2_capacity_rescue.sbatch and scripts/summarize_gcls_v2_capacity_rescue.py.

## Rescue outcome and binding method change

Job 10132304 completed all three cells; job 10132307 produced the aggregate decision.

| Cell | Selected epoch | Accuracy | Hard accuracy | Repair | Gap recovery | Harm | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| A compat ARR | 126 | .8847 | .3196 | .625 | .4074 | 0 | no |
| B compat CDPACE .5 | 173 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | yes |
| C additive CDPACE .5 | 201 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | yes |

The binding route is `delete_arr_claim_and_refreeze_smoothed_cdpace`, with diagnosis `unsmoothed_arr_gradient_starvation`. The primary method now uses length-normalized smoothed Candidate-D-PACE `alpha=.5` and safety weight zero. ARR remains only as a reproducible negative diagnostic. B and C both passing means the capacity probe does not select an encoder; that choice is delegated to the matched held-out representation screen.
