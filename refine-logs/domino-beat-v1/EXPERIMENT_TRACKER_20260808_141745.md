# Experiment Tracker

| Run ID | Milestone | Purpose | Variant | Split | Primary metric | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | Same-anchor comparator | Released DFlash / Domino / K16 oracle | validation-select | prompt-balanced EAL | MUST | DONE | 5.11200 / 7.01579 / 9.72668 |
| R002 | M0 | Correction calibration | Domino scales 0--2 | validation-select | paired EAL delta | MUST | DONE | scale 0.9: +0.01859, CI crosses 0 |
| R003 | M0 | Independent local expert | Domino + released DeLS grid | validation-select | paired EAL delta | MUST | DONE | all positive DeLS weights hurt; route closed |
| R004 | M1 | Cache released Domino backbone features | semantic same-anchor replay | all phase-3 splits | cached baseline EAL | MUST | DONE | job 10152397; 18,253 blocks; select EAL exactly 7.015792; 320 s |
| R005 | M2 | Official-objective adaptation | DECAY-CE | train/select | on-policy EAL | MUST | RUNNING | job 10152420; released head initialization |
| R006 | M2 | Dynamic frontier adaptation | D-PACE / effective-normalized variant | train/select | on-policy EAL | MUST | RUNNING | job 10152420; detached smoothed weights |
| R007 | M2 | First-break adaptation | reachable-breaker CE / greedy margin | train/select | on-policy EAL | MUST | RUNNING | job 10152420; first reachable mismatch + prefix preservation |
| R008 | M3 | Seed/data scale | best R005--R007 | train/select | mean/std paired EAL | MUST | BLOCKED_BY_R005_R007 | launch only after +0.10 screen |
| R009 | M4 | Clean confirmation | frozen best vs released Domino | validation-gate | paired EAL + CI | MUST | BLOCKED_BY_R008 | no tuning on gate |
| R010 | M5 | Proposal-prefix distillation | Draft-OPD-style replay | train/select | on-policy EAL | CONDITIONAL | BLOCKED_BY_M2 | actual-prefix target labels |
| R011 | M5 | Joint capacity | final backbone layers + head | train/select | on-policy EAL | CONDITIONAL | BLOCKED_BY_R010 | official SpecForge base |
| R012 | M5 | Iterative capacity | 2/4/6-pass block refiner | train/select | EAL and draft latency | CONDITIONAL | BLOCKED_BY_R011 | EAL prioritized over latency |
