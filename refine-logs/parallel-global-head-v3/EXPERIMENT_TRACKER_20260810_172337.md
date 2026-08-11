# Experiment Tracker：PCLD-16R

| Run ID | Milestone | Purpose | System / Variant | Split | Decisive metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| PCLD000 | M0 | architecture/API contracts | PCLD global | synthetic | exact params；full16 `[B,16,16]`；one-chain；forbidden scope/fields fail closed | MUST | TODO | CPU only |
| PCLD001 | M0 | loss/support/metric contracts | PCLD objective | synthetic + R047 refs | continuous prefix mask；prompt balance；stable safe/KL/latent；EAL/J2/harm | MUST | TODO | CPU only |
| PCLD002 | M0 | materializer dry-run | PCLD sidecar | 2–4 R047 blocks | row0/15 geometry；schema；source/key alignment | MUST | TODO | CPU/model-free fixtures first |
| PCLD003 | M1 | target teacher GPU mechanics | PCLD sidecar | 32 cross-domain R047 train blocks | batched/manual rows；direct/residual scores；stability calibration | MUST | BLOCKED | independent code review required first |
| PCLD004 | M1 | head GPU mechanics | PCLD global | same 32 blocks | zero identity；remote visibility；zero-init/one-update gradients | MUST | BLOCKED | PCLD003 PASS |
| PCLD005 | M1 | fair eager profile | PCLD vs released Domino | A40 batch1 | complete p50 `<=1.20x`；p90/mean/memory | MUST | BLOCKED | PCLD004 PASS |
| PCLD006 | M1 | 512 same-set capacity | PCLD global seed0 | frozen capacity group | agreement `>=99%`；recovery `>=95%`；harm `<=1%`；J2 `>=99%` | MUST | BLOCKED | PCLD003–005 PASS |
| PCLD020 | M2 | main mechanism arm | global PCLD seeds0/1/2 | fit/select | internal EAL checkpointing | MUST | BLOCKED | PCLD006 all gates PASS |
| PCLD021 | M2 | visibility control | local PCLD seeds0/1/2 | identical | matched metrics | MUST | BLOCKED | same batch order/params |
| PCLD022 | M2 | supervision control | global no-latent seeds0/1/2 | identical | matched metrics | MUST | BLOCKED | only `alpha=0` |
| PCLD023 | M2 | one-shot mechanism adjudication | all nine checkpoints | untouched diagnostic199 | EAL/recovery/J2/harm/domain；two paired CIs | MUST | BLOCKED | checkpoints freeze before open |
| PCLD030 | M3 | 25K scale fail-fast | global PCLD seed0 | OPB25K internal | EAL `>=7.8`；positive slope；controls/domain persist | MUST | BLOCKED | PCLD023 all gates PASS |
| PCLD040 | M3 | conditional 100K sidecar | frozen collector | OPB100K | complete/aligned/support stats | MUST | BLOCKED | PCLD030 PASS |
| PCLD041 | M3 | 100K seed0 | global PCLD | train/select | internal fixed EAL/design target | MUST | BLOCKED | frozen recipe |
| PCLD042 | M3 | 100K seed1 | global PCLD | train/select | same | MUST | BLOCKED | frozen recipe |
| PCLD043 | M3 | 100K seed2 | global PCLD | train/select | same | MUST | BLOCKED | frozen recipe |
| PCLD044 | M3 | freeze deployment seed/checkpoint | internal only | no final outcome | recorded deterministic selection | MUST | BLOCKED | no final seed shopping |
| PCLD045 | M3 | sealed fixed/dynamic result | deployment + all seeds vs Domino | sealed/fresh | both EAL ratios `>=1.15`；domains | MUST | BLOCKED | PCLD041–044 opening PASS |
| PCLD050 | M4 | complete-cycle feasibility | PCLD vs Domino | sealed workload | EAL+1 latency inequality | MUST | BLOCKED | PCLD045 PASS |
| PCLD051 | M4 | same-stack SGLang claim | PCLD vs Domino | paired ABBA A40 | output parity；TPS CI lower `>=1.15` | MUST | BLOCKED | PCLD050 PASS |
| PCLD060 | M5 | derived failure anatomy | frozen outputs | P2/P3 | position/margin/domain/repair buckets | NICE | BLOCKED | no checkpoint influence |

## Transition rules

- `PCLD000–002 + independent code review GO -> PCLD003`。
- `PCLD003 PASS -> PCLD004 -> PCLD005 -> PCLD006`。
- `PCLD006` 任一科学门失败即关闭 frozen PCLD-16R；不得 width/loss/schedule rescue。
- `PCLD006 all PASS -> PCLD020–022 -> freeze all checkpoints -> PCLD023`。
- `PCLD023 all gates PASS -> PCLD030`；否则关闭机制 claim。
- `PCLD030 PASS -> PCLD040–044 -> PCLD045`。
- `PCLD045 fixed/dynamic both PASS -> PCLD050`；联合 cycle 门 PASS 后才运行 `PCLD051`。
- `PCLD060` 永远不能解锁失败阶段或改变主路线。
