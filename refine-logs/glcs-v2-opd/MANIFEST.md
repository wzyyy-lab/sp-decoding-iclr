# GFPR / OPAL Artifact Manifest

| Artifact | Role | Status |
|---|---|---|
| `ANCHOR_DISTRIBUTION_DIAGNOSTIC.md` | Evidence that canonical offsets are not deployment anchors and position 0 is unrecoverable in GLCS-v1 | complete |
| `round-0-initial-proposal.md` | Initial OPAL proposal | frozen |
| `round-1-review.md` | Verbatim same-family fresh-agent review | frozen |
| `round-1-refinement.md` | GFPR revision responding to all critical review items | current |
| `round-2-review.md` | Verbatim same-family re-review of GFPR | frozen |
| `round-2-refinement.md` | Unified full-vocabulary GFPR proposal responding to Round 2 | current |
| `round-3-review.md` | Verbatim final same-family review; score 9.20, READY | frozen |
| `FINAL_PROPOSAL_20260809_235057.md` | Timestamped accepted GFPR proposal | frozen |
| `FINAL_PROPOSAL.md` | Fixed final-proposal entrypoint | current |
| `REVIEW_SUMMARY_20260809_235057.md` | Timestamped review trajectory | frozen |
| `REVIEW_SUMMARY.md` | Fixed review summary | current |
| `REFINEMENT_REPORT_20260809_235057.md` | Timestamped refinement report | frozen |
| `REFINEMENT_REPORT.md` | Fixed refinement report | current |
| `EXPERIMENT_PLAN_20260810_000025.md` | Timestamped claim-driven GFPR experiment plan | frozen |
| `EXPERIMENT_PLAN.md` | Fixed executable experiment plan | current |
| `EXPERIMENT_TRACKER_20260810_000025.md` | Timestamped run ledger | frozen |
| `EXPERIMENT_TRACKER.md` | Fixed live run ledger | current |
| `score-history.md` | Review score ledger | current |
| `REFINE_STATE.json` | Machine-readable workflow state | current |
| `src/sph/gfpr.py` | Exact all-position decode, frontier loss, oracle, paired metrics, head checkpoint core | implemented; 8 unit tests pass |
| `scripts/collect_gfpr_rollouts.py` | Single-path fixed/dynamic all-16 rollout collector | Gate A passed |
| `scripts/analyze_gfpr_rollouts.py` | Semantic and Top-16/K16/K17 oracle analysis | Gate A passed |
| `scripts/train_gfpr_head.py` | BF16 deployment-identical / FP32-master GFPR trainer | capacity screen active |
| `scripts/sweep_gfpr_position_zero.py` | Position-zero scale diagnostic | active |
| `artifacts/analysis/gfpr_validation_select_fixed_10163965.json` | Full fixed Gate A result | complete: released 7.23955 / all16 oracle 10.90926 |
| `artifacts/analysis/gfpr_validation_select_dynamic_10163965.json` | Full dynamic Gate A result | complete: released 6.60282 / all16 oracle 10.23480 |
| `artifacts/models/gfpr_capacity_smoke_10163970_{0,1}/report.json` | Same-set Frozen-15 / GFPR-16 capacity test | complete: 8.22578 / 8.31352 |
| `GFPR_CODE_REVIEW.md` | Fresh implementation review and substantive blockers | frozen: BLOCKED before remediation |
| `GFPR_CODE_REREVIEW.md` | Independent review of remediated implementation | current: READY_FOR_GATE_B |
| `scripts/compare_gfpr_dynamic_rollouts.py` | Prompt-paired true adapted-policy trajectory comparison | implemented and identity-tested |
| `EXPERIMENT_PLAN_AMENDMENT_20260810_033957.md` | R047 current-anchor target early-exit contract and hard gates | current |
| `scripts/slurm/r047_anchor_early_collect.sbatch` | R047 train/select feature and teacher collection | ready for review |
| `scripts/slurm/r047_anchor_early_smoke.sbatch` | R047 32-prompt GPU sanity | ready for review |
| `scripts/slurm/r047_anchor_early_phase3.sbatch` | R047 prompt-disjoint Phase3 efficacy gate | gated on sanity |
| `scripts/check_anchor_early_exit_alignment.py` | Cached-replay vs incremental-KV anchor feature/path comparison | ready for review |
| `scripts/slurm/r047_anchor_alignment.sbatch` | R047 numerical and token-path alignment gate | ready for review |
| `EXPERIMENT_CODE_REVIEW_20260810_040213.md` | Fresh secondary R047 code review | frozen: staged-sanity GO, Phase3 gated |
| `EXPERIMENT_CODE_REVIEW_R047.md` | Fixed R047 code-review entrypoint | current |
| `artifacts/analysis/r047_anchor_t4_alignment_10164717.json` | Mini cached-replay/incremental feature alignment | pass |
| `artifacts/canonical/r047_anchor_t4_{train,validation_select}_10164718` | Full R047 train/select collection | complete; select baseline exact |
| `artifacts/models/r047_anchor_t4_smoke_10164721` | 32-prompt mechanics smoke | complete |
| `artifacts/analysis/r047_anchor_t4_alignment_10164722.json` | Trained nonzero-residual incremental token-path alignment | pass |
| `R047_RESULT_20260810_042720.md` | Definitive R047 held-out result and kill decision | frozen |
| `R047_RESULT.md` | Fixed R047 result entrypoint | current |
| `RESEARCH_REVIEW_R048_20260810_043611.md` | Fresh ARIS review of proposal-prefix early verification | frozen |
| `RESEARCH_REVIEW_R048.md` | Fixed R048 review entrypoint | current |
| `EXPERIMENT_PLAN_AMENDMENT_R048_20260810_043611.md` | R048 oracle/capacity/held-out experiment contract | frozen |
| `EXPERIMENT_PLAN_AMENDMENT_R048.md` | Fixed R048 experiment-plan entrypoint | current |
| `EXPERIMENT_PLAN_AMENDMENT_R048K_20260810_045547.md` | K64/K128 reachability, latency, and minimum-K gate | frozen |
| `EXPERIMENT_PLAN_AMENDMENT_R048B_20260810_052100.md` | Clean-unsplit-target 64-prompt information-capacity contract | current |
| `src/sph/fast_r048.py` | Fast-K candidate proposal, 180,224-param tuned lens, graph-safe earliest-one decision | implemented; focused tests pass |
| `src/sph/r048_capacity.py` | Causal capacity loss, exact one-rewrite EAL scorer, exhaustive zero-harm threshold sweep | implemented; fresh review GO |
| `scripts/collect_r048_capacity.py` | Disposable-L4 features plus clean-unsplit baseline/oracle collection | GPU smoke passed |
| `scripts/train_r048_capacity.py` | Maximum-200-step R048-B capacity trainer and 90% recovery gate | optimizer/checkpoint smoke passed |
| `artifacts/analysis/r048_fast_k64_oracle_10164820.json` | Exact batch1 K64 oracle and reward-recovery feasibility | complete: oracle 8.460398445 |
| `artifacts/analysis/r048_fast_candidate_profile_10164871.json` | Complete graph candidate+lens+decision A40 profile | complete: K64 p50 1.327104 ms |
| `artifacts/analysis/r048_layer_split_smoke_10164859.json` | HF DynamicCache/SDPA layer-split falsifier | complete: deployment NO-GO, capacity-only route authorized |
| `artifacts/canonical/r048_capacity_smoke_10164903` | 8-prompt clean-verifier collection mechanics smoke | complete; 64/64 exact invariants |
| `artifacts/models/r048_capacity_train_smoke_10164905` | Two-step optimizer/evaluation/checkpoint smoke | complete |
| `artifacts/canonical/r048_capacity_64_10164907` | 64-prompt clean-unsplit capacity collection | complete: Fast 7.003906 / oracle 8.060547 |
| `artifacts/models/r048_capacity_64_10164909/report.json` | Preregistered L4+180K capacity result | complete: FAIL, best 7.285156 / 26.617% recovery / zero harm |
| `EXPERIMENT_PLAN_AMENDMENT_R049_20260810.md` | Zero-parameter multi-depth target-logit probe and conditional residual/gate contract | current |
| `scripts/analyze_r049_depth_probe.py` | Hook-based zero-parameter target-logit depth probe with exact token/policy metrics | implemented; fresh review GO |
| `scripts/slurm/r049_depth_probe.sbatch` | Official 64-prompt K64 L4/8/12/16/24/32/36 probe launcher | complete: final job 10164979 |
| `artifacts/analysis/r049_depth_probe_10164979.json` | Controlled deployable multi-depth token/policy result | complete: shallow route closed; L36 stable control exact |
| `EXPERIMENT_PLAN_AMENDMENT_R050_20260810.md` | Exact target-seeded Fast-K64 serving geometry and accuracy/system gates | current |
| `scripts/evaluate_r050_target_seeded.py` | Clean batch1 target-seeded Fast-K64 evaluator with full 17-row split/bonus control | implemented; fresh review GO |
| `scripts/slurm/r050_target_seeded_fixed.sbatch` | Full fixed validation_select R050-A launcher | complete: job 10164996 |
| `artifacts/analysis/r050_target_seeded_fixed_10164996.json` | Exact p0-seeded accuracy and 17-row split result | complete: FAIL, EAL 7.719145 and bonus parity 279/281 |
| `EXPERIMENT_PLAN_AMENDMENT_R051_20260810.md` | One-shot exact seed-length 2/3/4 accuracy contract | current |

This route deliberately lives under `refine-logs/glcs-v2-opd/` so it does not overwrite earlier, independently auditable research routes.

| `artifacts/analysis/r053_tree_budget_pareto_10165201.json` | Full 1,175-block clean-authority R053 accuracy and eager system Pareto | complete: N64 PASS accuracy 8.483722 / current eager TPS 0.845579x |
| `profile_output/R053_TREE_PARETO_REPORT.md` | Quantitative R053 result interpretation, claim boundary, latency budget and instrumentation changelog | current |
| `scripts/profile_r053_beam_graph.py` | CUDA-graph W4/W8/W16 beam bottleneck probe | implemented; pending A40 review gate |
| `scripts/slurm/r053_beam_graph_profile.sbatch` | Reproducible median-context A40 graph launcher | implemented; pending review |
| `EXPERIMENT_PLAN_AMENDMENT_R055_20260810.md` | Fixed-shape W4/W8/W16 padded-forest accuracy/system contract | current |
| `src/sph/r055_forest.py` | Fixed shared-anchor forest mask, structural oracle and all-path graph-safe traversal | implemented; CPU tests pass |
| `scripts/evaluate_r055_padded_forest.py` | Clean-authority W4/W8/W16 actual forest evaluator and graph-beam complete-cycle profile | implemented; pending fresh review/smoke |
| `scripts/slurm/r055_padded_forest_smoke.sbatch` | Four-block A40 mechanics/profile gate | implemented; pending review |
| `scripts/slurm/r055_padded_forest_full.sbatch` | Full 1,175-block A40 Pareto launcher | implemented; blocked on smoke |
| `artifacts/analysis/r055_padded_forest_full_10165728.json` | Full 147-prompt / 1,175-block R055 accuracy and complete-cycle Pareto | complete: W8/N129 joint PASS, EAL 8.667396 / projected TPS 1.292935x |
| `profile_output/R055_PADDED_FOREST_REPORT.md` | R055 quantitative result, selected W8 point, latency breakdown and deployment claim boundary | current |
| `RESULT_TO_CLAIM_R055_20260810.md` | ARIS result-to-claim review authorizing bounded W8 SGLang implementation | current; same-family provisional |
| `profile_output/r053_beam_graph_10165436.json` | A40 W4/W8/W16 exact CUDA-graph beam latency | complete: W16 3.584000 ms / five-tensor parity PASS |
| `profile_output/R053_BEAM_GRAPH_REPORT.md` | Beam graph result, budget interpretation and instrumentation changelog | current |
