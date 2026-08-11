# PGCF-16 Artifact Manifest

| Artifact | Role | Status |
|---|---|---|
| `USER_CONSTRAINT_CONTRACT.md` | Immutable full16 global parallel one-chain requirements | authoritative |
| `PROBLEM_ANCHOR.md` | Bottom-line accepted-length and throughput objective | current |
| `FINAL_PROPOSAL.md` | READY 9.3 PGCF-16 method | current |
| `REFINEMENT_REPORT.md` | ARIS research-refine trajectory | complete |
| `EXPERIMENT_PLAN.md` | Claim-driven ordered experiment plan | current |
| `EXPERIMENT_TRACKER.md` | Live gate ledger | current |
| `EXPERIMENT_CODE_REVIEW_20260810_121346.md` | Fresh Gate0 implementation review | GO for PGCF-002 only |
| `src/sph/parallel_global_candidate_fusion.py` | Full16 global head and offline loss | implemented; 22 focused tests pass |
| `scripts/train_pgcf16.py` | Full16 trainer/evaluator | implemented; smoke authorized |
| `scripts/profile_pgcf16_head.py` | Fair eager PGCF/Domino profile | implemented; gated on smoke |
| `scripts/slurm/pgcf16_gpu_smoke.sbatch` | 32-record A40 mechanics smoke | authorized |
| `artifacts/models/pgcf16_smoke_10166796/report.json` | PGCF-002 A40 mechanics evidence | complete: PASS |
| `PGCF002_SMOKE_RESULT_20260810.md` | Mechanics result and claim boundary | complete |
| `profile_output/pgcf16_eager_10166801.json` | Fair A40 eager PGCF/Domino profile | complete: PASS, 0.4310x |
| `artifacts/models/pgcf16_capacity_10166802/report.json` | First 512-block capacity attempt | partial: gap/harm pass, accuracy shortfall |
| `PGCF003_005_INITIAL_RESULTS_20260810.md` | Quantitative Gate2/3 result record | current |
| `artifacts/models/pgcf16_capacity_target_10166814/report.json` | 8K-step frozen-curriculum capacity witness | partial: near-oracle EAL, token gates fail |
| `artifacts/models/pgcf16_capacity_teacher_10166815/report.json` | Independent teacher-only capacity witness | complete: PASS at 99.813% |
| `PGCF_GATE1_CAPACITY_RESULT_20260810.md` | Honest Gate1 conjunction and mask diagnosis | complete: PASS with bounded interpretation |
| `artifacts/models/pgcf16_capacity_gold_ce_10166838/report.json` | Independent d256 architecture capacity witness | complete: exact oracle, four-way PASS |
| `artifacts/models/pgcf16_capacity_local_gold_ce_10166853/report.json` | Matched-local D0 capacity diagnostic | complete: 98.66% gap, 99.680%/99.476% accuracy |
| `PGCF004_LOCAL_CAPACITY_RESULT_20260810.md` | Local-control optimization diagnosis | complete: PASS, not a held-out claim |
| `scripts/evaluate_pgcf16_gate2.py` | Disjoint global/local + coherent remote diagnostic | complete |
| `scripts/slurm/pgcf16_r047_screen.sbatch` | Matched 20k global/local seed-0 training | complete: job 10166898 |
| `scripts/slurm/pgcf16_gate2_eval.sbatch` | Explicit-checkpoint Gate2 evaluator | complete: scientific FAIL, job 10167001 |
| `PGCF_G4_CODE_REVIEW_20260810.md` | G4 training/evaluator pre-submit review | complete: SUBMIT GO |
| `artifacts/models/pgcf16_r047_screen_10166898/global_seed0/report.json` | G4 global training evidence | complete: best validation EAL 6.10277 |
| `artifacts/models/pgcf16_r047_screen_10166898/local_seed0/report.json` | G4 matched-local evidence | complete: best validation EAL 6.08892 |
| `artifacts/results/pgcf16_gate2_10167001/report.json` | Binding Gate-2 evidence | complete: FAIL |
| `PGCF_GATE2_RESULT_TO_CLAIM_20260810.md` | Result boundary and v1 close decision | complete: claim_supported=no |
| `.aris/traces/result-to-claim/2026-08-10_run02/` | Fresh result-to-claim trace | complete; same-family provisional |
