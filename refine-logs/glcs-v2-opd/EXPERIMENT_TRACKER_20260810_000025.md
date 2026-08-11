# GFPR Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Primary Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | current-frontier mask/loss unit test | synthetic | unit | active frontier、suffix masked | MUST | TODO | include q=0 and q=16 |
| R001 | M0 | normalized keep/prompt weight test | synthetic | unit | per-block/per-prompt weight sums | MUST | TODO | long prefix cannot dominate |
| R002 | M0 | r+1 and full-accept bonus test | synthetic | unit | next anchor and bonus token | MUST | TODO | explicit r=16 |
| R003 | M0 | all-16 Domino identity test | synthetic tensors | unit | released token equality | MUST | TODO | alpha0=0 |
| R004 | M0 | paired bootstrap/harm summary test | synthetic prompts | unit | CI、gained/lost、harm fraction | MUST | TODO | deterministic seed |
| R010 | M1 | collector smoke | released Domino full16 | 32 train prompts | record invariants、identity | MUST | TODO | one GPU |
| R011 | M1 | all-16 oracle | released/DFlash/K17/K16 | validation_select | EAL oracle、position0 gain | MUST | TODO | do not splice old hiddens |
| R012 | M1 | fixed-control equivalence | released Domino | validation_select | historical fixed EAL | MUST | TODO | same collector code |
| R013 | M1 | position0 latency | 15-step vs16-step full head | A40/debug | eager/graph ms | MUST | TODO | full head cost |
| R020 | M2 | dynamic v0 collection | released Domino | phase3 train 2K | blocks、anchor hist、EAL | MUST | TODO | multi-GPU shards |
| R021 | M2 | dynamic heldout collection | released Domino | validation_select | true rollout baseline | MUST | TODO | no training |
| R022 | M2 | fixed-control train records | released Domino | phase3 train 2K | matching record counts | MUST | TODO | same codepath |
| R030 | M3 | same-set capacity smoke | GFPR-16 | 64–128 blocks | frontier advance to oracle | MUST | TODO | before full training |
| R031 | M3 | static control | Fixed-15 | train / validation_select | EAL、harm | MUST | TODO | one seed |
| R032 | M3 | anchor ablation | Dynamic-15 | train / validation_select | EAL、harm | MUST | TODO | one seed |
| R033 | M3 | main 2K screen | GFPR-16 | train / validation_select | EAL、CI、harm | MUST | TODO | continue gate |
| R034 | M3 | Gate B summary | all three arms | validation_select | paired table | MUST | TODO | no validation_gate |
| R100 | M4 | 16K v0 collection | released Domino | OPB balanced 16K | blocks、anchors | CONDITIONAL | BLOCKED | requires R033 pass |
| R101 | M4 | 16K v0 train | GFPR-16 | OPB / validation_select | fixed+dynamic EAL | CONDITIONAL | BLOCKED | one seed first |
| R102 | M4 | v1 recollection | selected GFPR | OPB 16K | new anchor/frontier hist | CONDITIONAL | BLOCKED | requires v0 signal |
| R103 | M4 | 50/50 refresh train | GFPR v0+v1 | OPB / validation_select | EAL、CI、harm | CONDITIONAL | BLOCKED | hard target |
| R104 | M4 | optional 32K extension | GFPR | OPB 32K | scaling curve | CONDITIONAL | BLOCKED | only if 16K promising |
| R110 | M5 | three-seed confirm | final GFPR | validation_select | mean/spread | CONDITIONAL | BLOCKED | target ≥8.325 |
| R111 | M5 | sealed gate | final GFPR | validation_gate | fixed+dynamic EAL | CONDITIONAL | BLOCKED | run once |
| R112 | M5 | reserved test | final GFPR | phase3 test 600 | fixed+dynamic EAL | CONDITIONAL | BLOCKED | after method freeze |
| R120 | M5 | capacity matched arm | residual or LoRA | same data/loss | EAL/latency | OPTIONAL | BLOCKED | only signal+plateau |
| R200 | M6 | SGLang correctness | final GFPR | smoke | lossless output | CONDITIONAL | BLOCKED | after EAL success |
| R201 | M6 | CUDA graph/head profile | Domino vs GFPR | A40 | head ms | CONDITIONAL | BLOCKED | fair graph setup |
| R202 | M6 | end-to-end throughput | Domino vs GFPR | main workloads | tokens/s、EAL | CONDITIONAL | BLOCKED | target +15% |
| R203 | M6 | system summary | final systems | all | latency breakdown | CONDITIONAL | BLOCKED | paper table |

