# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Primary Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | deterministic preflight | CPU tests + py_compile | synthetic/canonical loader | pass/fail | MUST | DONE | 39 targeted tests passed |
| R001 | M0 | GPU/kernel smoke | flat-compat global D128 L2, reach λ=.1 | same 128 train blocks | artifact, finite loss, peak memory | MUST | TODO | launch first |
| R002 | M0 | agent-follows-doc env validation | documented Slurm smoke invocation | same 128 train blocks | witness and doc divergence | MUST | TODO | fresh agent required |
| R010 | M1 | capacity gate | flat-compat global, reach λ=0 | same 128 train blocks | repair, oracle gap, harm | MUST | BLOCKED_BY_R001 | 120 epochs |
| R011 | M1 | capacity gate | flat-compat global, reach λ=.1 | same 128 train blocks | repair, oracle gap, harm | MUST | BLOCKED_BY_R001 | diagnostic default |
| R012 | M1 | capacity gate | flat-compat global, reach λ=.25 | same 128 train blocks | repair, oracle gap, harm | MUST | BLOCKED_BY_R001 | safety trade-off |
| R020 | M2 | historical objective control | axial-additive D64, D-PACE α=.5 | OPB-25K → validation_select | raw prompt-balanced EAL | MUST | BLOCKED_BY_M1 | seed0 |
| R021 | M2 | unsmoothed reach | axial-additive D64, reach λ=0 | OPB-25K → validation_select | EAL, repair, harm | MUST | BLOCKED_BY_M1 | seed0 |
| R022 | M2 | reach + safety | axial-additive D64, reach λ=.1 | OPB-25K → validation_select | EAL, repair, harm | MUST | BLOCKED_BY_M1 | seed0 |
| R023 | M2 | reach + stronger safety | axial-additive D64, reach λ=.25 | OPB-25K → validation_select | EAL, repair, harm | MUST | BLOCKED_BY_M1 | seed0 |
| R030 | M3 | pooled control | selected axial-additive | OPB-25K → validation_select | raw EAL | MUST | BLOCKED_BY_M2 | reuse R02x if identical |
| R031 | M3 | no-prepool isolation | flat-additive D128 L2 | OPB-25K → validation_select | raw EAL, latency | MUST | BLOCKED_BY_M2 | selected λ |
| R032 | M3 | compatibility isolation | flat-compat D128 L2 | OPB-25K → validation_select | raw EAL, latency | MUST | BLOCKED_BY_M2 | selected λ |
| R040–R048 | M4 | matched scope/seed confirmation | local/causal/global × seeds0/1/2 | frozen train/cal/test | delta CI, harm UCB | MUST | BLOCKED_BY_M3 | test once |
| R050–R053 | M5 | end-to-end value | DFlash/Domino/raw/KEEP_BASE | frozen test | TPS, EAL, latency | MUST | BLOCKED_BY_M4 | same hardware/protocol |
| R060–R062 | M6 | positive capacity diagnostic | D640/H10/L4 staged | 512→10K→100K | held-out EAL | CONDITIONAL | NOT_TRIGGERED | selected objective/λ only |

