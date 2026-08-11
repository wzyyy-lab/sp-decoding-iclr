# Experiment Tracker: prospective-v2

**Route status**：PLANNED_GATED；no GPU/data/falsifier authorization is implied。  
**Old route**：R083 remains CLOSED_OPERATIONAL_FAILURE_EXIT1_OPENING_CONSUMED_NO_RETRY_JOB10141601。

| Run ID | Milestone | Purpose | System / Variant | Split | Decisive Output | Priority | Status | Authorization / Notes |
|---|---|---|---|---|---|---|---|---|
| PV2-001 | M0 | freeze contract/source/metric/power schemas | protocol only | none | PROSPECTIVE_V2_CONTRACT + fresh review | MUST | TODO | G0；no execution |
| PV2-002 | M1 | native LoRA zero-init/count/merge tests | mock + released config | analytic | 1,835,008；exact zero；merge parity | MUST | BLOCKED | opens after PV2-001 GO |
| PV2-003 | M1 | D-PACE/frontier/constraint/tie tests | A/B/C/D pure functions | analytic | scalar/grad parity；max equivalence；active-switch rejection | MUST | BLOCKED | CPU only |
| PV2-004 | M1 | transactional optimizer/statistics/protocol tests | B/D + bootstrap | analytic | counter/moment/restoration exactness；estimand replay | MUST | BLOCKED | full suite + fresh code review |
| PV2-005 | M2 | one real-model synthetic GPU smoke | A then D | synthetic | finite full rows；memory/timing receipt | MUST | BLOCKED | only after G1 fresh review |
| PV2-006 | M2 | three counterbalanced clean-process pairs | A/D × 3 pairs | synthetic | every median/p95/memory gate | MUST | BLOCKED | only after PV2-005 result review |
| PV2-007 | M2 | independent cost adjudication | receipts only | none | ENGINEERING_GO/CLOSE | MUST | BLOCKED | failure closes route |
| PV2-008 | M3 | producer-only power receipt and n_f | aggregate only | producer train | n_f with no means/signs/rows | MUST | BLOCKED | only if PV2-007 GO |
| PV2-009 | M3 | component-disjoint candidate split | protocol builder | OPB remainder | fit/checkpoint/sealed-falsifier manifests | MUST | BLOCKED | no old downstream |
| PV2-010 | M3 | independent source/split replay | independent auditor | manifests only | exact hashes/counts/components | MUST | BLOCKED | opens sequence generation |
| PV2-011 | M4 | generate fit/checkpoint greedy sequences | frozen target | fit/checkpoint | 8k/1k complete prompt records | MUST | BLOCKED | falsifier remains sealed |
| PV2-012 | M5 | 32-block micro-overfit | A/B/C/D | fit only | finite decreasing loss；transactions | MUST | BLOCKED | reviewed wrapper only |
| PV2-013 | M5 | 512-block capacity | A/B/C/D | fit only | feasibility/gradient-ratio/counter receipts | MUST | BLOCKED | failure closes before full train |
| PV2-014 | M5 | capacity result audit | receipts only | none | FULL_TRAIN_GO/CLOSE | MUST | BLOCKED | no checkpoint/falsifier |
| PV2-015 | M6 | matched main training | A/B/C/D × seeds 0/1/2 | fit + checkpoint | 12 selected checkpoint hashes or diagnostic failure | MUST | BLOCKED | 4×3 array only after G6 |
| PV2-016 | M7 | freeze identity closure | 12 trained + released | hashes only | immutable 13-instance receipt | MUST | BLOCKED | no outcomes yet |
| PV2-017 | M7 | single common falsifier generation/evaluation | released/A/B/C/D | sealed falsifier | prompt-level outcome bundle | MUST | BLOCKED | one opening only |
| PV2-018 | M7 | frozen bootstrap and result-to-claim | C1/C2 contrasts | same bundle | scientific PASS/FAIL per claim | MUST | BLOCKED | no model rerun |
| PV2-019 | M8 | merge/trace/output audit | released vs D seed0 | checkpoint/small replay | graph/output exactness receipt | MUST | BLOCKED | only if C1 passes |
| PV2-020 | M8 | restart-level latency TOST | released vs merged D seed0 | fixed latency prompts | 20 restart values + TOST | MUST | BLOCKED | deployment claim only |
| PV2-021 | M8 | descriptive failure figures | frozen outcomes only | falsifier | fixed-bin appendix figures | NICE | BLOCKED | no new inference/jobs |

## Immediate Queue

1. PV2-001：write and independently review the execution contract。
2. PV2-002：implement native LoRA in an isolated new module。
3. PV2-003：implement D-PACE/frontier/blockwise feasibility pure functions and unit tests。

No Slurm run is queued or submitted。
