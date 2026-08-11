# GFPR Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Primary Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | current-frontier mask/loss unit test | synthetic | unit | active frontier、suffix masked | MUST | PASS | `tests/test_gfpr.py`; 2026-08-10，8 tests total pass |
| R001 | M0 | normalized keep/prompt weight test | synthetic | unit | per-block/per-prompt weight sums | MUST | PASS | normalized prefix protection covered |
| R002 | M0 | r+1 and full-accept bonus test | synthetic | unit | next anchor and bonus token | MUST | PASS | explicit r=16 advancement covered；collector semantic mismatch=0 |
| R003 | M0 | all-16 Domino identity test | synthetic tensors | unit | released token equality | MUST | PASS | alpha0=0 teacher/on-policy identity unit test；GPU step-0 dataset identity pending R030 |
| R004 | M0 | paired bootstrap/harm summary test | synthetic prompts | unit | CI、gained/lost、harm fraction | MUST | PASS | deterministic prompt-cluster summary covered |
| R010 | M1 | collector smoke | released Domino full16 | 8 train prompts | record invariants、identity | MUST | PASS | Slurm 10163938；dynamic 301 blocks/fixed 64；all semantic mismatch counts=0 |
| R011 | M1 | all-16 oracle | released/DFlash/K17/K16 | validation_select | EAL oracle、position0 gain | MUST | PASS | Slurm 10163965_1；dynamic released 6.60282，all16 Top16 oracle 10.23480，5693 blocks，semantic mismatches=0 |
| R012 | M1 | fixed-control equivalence | released Domino | validation_select | historical fixed EAL | MUST | PASS | Slurm 10163965_0；exactly reproduces 7.239552964，all16 oracle 10.90926 |
| R013 | M1 | position0 latency | 15-step vs16-step full head | A40/debug | eager/graph ms | MUST | TODO | full head cost |
| R020 | M2 | dynamic v0 collection | released Domino | phase3 train 2K | blocks、anchor hist、EAL | MUST | PASS | Slurm 10163980_1；1987 prompts / 74873 cycles；EAL 6.54462；oracle 10.38569；semantic mismatches=0 |
| R021 | M2 | dynamic heldout collection | released Domino | validation_select | true rollout baseline | MUST | PASS | Slurm 10163965_1；147 prompts / 5693 cycles，EAL 6.60282 |
| R022 | M2 | fixed-control train records | released Domino | phase3 train 2K | matching record counts | MUST | PASS | Slurm 10163980_0；1987 prompts / 15886 blocks；same codepath；semantic mismatches=0 |
| R030 | M3 | same-set capacity smoke | Frozen-15 / GFPR-16 | 8 prompts / 301 dynamic blocks | frontier advance to oracle | MUST | PASS | Capacity only，非held-out claim；Frozen-15 5.2895→8.2258，GFPR-16→8.3135；overlap gate now forcibly suppressed |
| R031 | M3 | static control | Fixed-15 full-head | train / validation_select | EAL、harm | MUST | FAIL | Slurm 10164027_0；best 7.24393 (+0.00437)，CI/harm fail |
| R032 | M3 | anchor ablation | Dynamic-15 full-head | train / validation_select | EAL、harm | MUST | FAIL | Slurm 10164027_1；best is released step0，训练后held-out下降 |
| R033 | M3 | main 2K screen | GFPR-16 full-head | train / validation_select | EAL、CI、harm | MUST | FAIL | Slurm 10164027_2；best 7.25061 (+0.01105)，pos0仅repair 1 block，门失败 |
| R035 | M3 | transfer-stable screen | Dynamic-15 / GFPR-16 GRU-rank | train / validation_select | EAL、CI、harm | MUST | FAIL | Slurm 10164054；四个arm最佳均为step0或至多+0.0034，冻结vocab projection没有解决迁移 |
| R036 | M3 | lightweight Top-16 selector | candidate-only input/GRU rank | 8-prompt capacity / validation_select | EAL、参数、harm | CONDITIONAL | FAIL | 0.918M input-rank capacity +2.116，11.93M GRU-rank +2.447；2K held-out六arm最高仍为7.22425，低于released |
| R037 | M4 | balanced scale + dense labels | OPB10K candidate-only | 9,999 train / validation_select | EAL、domain transfer | CONDITIONAL | FAIL | Slurm 10164123/10164133；79,918 blocks语义全过、oracle 10.487；dense Candidate-D-PACE四arm均选step0，CE持续误改 |
| R038 | M4 | zero-init lightweight residual | 61K adapter, Top-16 only | capacity / OPB10K held-out | EAL、identity、harm | CONDITIONAL | FAIL | Capacity最高 +3.321；held-out最高7.19643，candidate restriction本身较Domino低0.05187 |
| R039 | M4 | exact Domino fallback union | Top-15 + current Domino + 61K residual | capacity / OPB10K held-out | exact identity、EAL、harm | CONDITIONAL | FAIL | Slurm 10164296 capacity：step0逐token identity，最高5.2895→8.90625 (+3.61674)；held-out 10164321 四个完整 arm 中三者 best=step0，最高仅 frontier lr1e-3 的7.24891（+0.00935），末步转负；候选限制已排除但 transfer failure 不变 |
| R040 | M4 | target-margin teacher collection | released accepted prefix + original first rejection | phase3 train / validation_select | target/gold alignment、dense candidate margins | MUST | PASS | Slurm 10164355；train 1987 prompts/15886 blocks、select 147/1175；reachable target top1 与 canonical gold 一致率 train/select=99.60%/99.65%，首拒 union gold availability86.06%/85.71%，margin 中位数均1.75 |
| R041 | M4 | target-vs-Domino advantage distillation | exact union residual rank64 | phase3 train / validation_select | EAL、harm、soft-margin transfer | MUST | FAIL | Slurm 10164368/10164372；KL T1最佳7.27697（+0.03741），T2最佳7.29397（+0.05442，CI跨0），raw advantage-Huber step100直接−1.434；soft margin有小信号但远低于7.8切换门槛，不再扫loss |
| R042 | M4 | verified target-boundary feature | target layers concat at token before anchor | phase3 train / validation_select | new online information、EAL | MUST | FAIL | Slurm 10164397/10164419；1.065M rank64最佳7.31305（+0.07349，CI跨0），2.130M rank128最佳7.29519（+0.05564，CI跨0）；均远低于7.8生死线，停止selector rank/LR/mixer扫描 |
| R043 | M4 | target-distilled DFlash LoRA | live Top-16 union、accepted prefix+first rejection | OPB6K / validation_select | representation transfer、EAL | MUST | FAIL | 1.835M rank16 LoRA；main 10164463 best step300在旧15位口径仅+0.00170；独立full-B16 10164542精确基线7.239552964→7.238702624（−0.000850，CI[−0.03486,+0.03061]，28 gain/29 loss），关闭LoRA-only |
| R044 | M4 | target-frontier distill existing Domino head + LoRA | no new inference module；adapt released GRU/projection and merge LoRA | phase3 train / validation_select | stronger target-posterior capacity、EAL | MUST | FAIL | 修正版10164541旧15位选择最佳step100为+0.03936；独立full-B16 10164548：7.239552964→7.269436346（+0.029883，CI[−0.02430,+0.08771]），math +0.12但chat −0.0543，远低于7.8门 |
| R045 | M4 | target-frontier full draft adaptation | adapt existing 537M DFlash + 50.8M Domino head weights | phase3 train / validation_select | representation ceiling、EAL | MUST | FAIL | 10164556最佳step100；独立full-B16 10164557：7.239552964→7.241375121（+0.001822，CI[−0.03560,+0.04009]），chat下降；588.25M trainable但same-runtime frontier-only adaptation无迁移 |
| R046 | M4 | full-vocabulary target-posterior adaptation | full 537M DFlash + existing 50.8M Domino head；full-B16 frontier+dense exact-gold-prefix KL | phase3 train / validation_select | missing-Top16 gradient、all-16 EAL、representation ceiling | MUST | RUNNING | 取消Top16监督截断；正确prefix仅安全hinge，首拒full-vocab KL，后续exact gold-prefix低权重KL；训练与checkpoint选择均完整B16，推理图/参数/latency不变；19 tests+独立code review GO；smoke 10164606 |
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
