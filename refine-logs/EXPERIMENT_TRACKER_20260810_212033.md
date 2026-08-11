# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| PARC000 | M0 | 本地实现fail-fast（非实验、无GPU） | PARC core/data/loss/train/eval | synthetic unit fixtures only | shape、global visibility、identity、gradient equivalence、`H<=Hbar` | MUST | TODO | 不产生任何效果结论，不提交Slurm |
| PARC010 | M1 | 在任何新label前固定prompt split | deterministic domain-stratified splitter | raw OPB100K manifest | 90K/5K/remainder、prompt overlap=0、旧formal overlap=0 | MUST | TODO | CPU data-contract step |
| PARC020 | M1 | 正式full16 train/validation materialization | released pure DFlash + frozen target | train + validation only | full16、8 anchors、reference EAL、storage/coverage | MUST | BLOCKED | Depends on PARC000/PARC010；8-way A800 array；不是smoke |
| PARC030 | M1 | 冻结train-only numeric certificate | BF16 production vs FP32 gathered rows | train only | `e_num_cert`、`delta_min`、ambiguity rate | MUST | BLOCKED | validation/held-out严格禁止参与 |
| PARC100 | M2 | 唯一正式科学训练 | global PARC D256/L2 + joint DFlash | 90K train / 5K validation | validation EAL/harm、train losses/dual/support-drop | MUST | BLOCKED | 180K steps；不先跑任何小训练 |
| PARC200 | M3 | 一次sealed held-out效果裁决 | DFlash vs released Domino vs locked PARC | held-out fixed + dynamic | EAL ratio、harm、domains、paired CI、oracle recovery | MUST | BLOCKED | checkpoint锁定后首次打开；same-job；之后禁止训练 |
| PARC300 | M4 | complete eager性能定位 | released Domino vs locked PARC | frozen production prompts | complete p50/p95、memory、cycle feasibility | MUST | BLOCKED | 仅PARC200通过后；A40 batch1 |
| PARC400 | M4 | 最终same-stack serving裁决 | SGLang Domino vs SGLang PARC | paired ABBA | TPS ratio 95% CI、exact output parity | MUST | BLOCKED | 硬门95% CI lower`>=1.15` |
| PARC500 | M5 | 后验机制诊断 | matched-local / constraint deletion | validation或新封存数据 | EAL/harm/global-local delta | NICE | BLOCKED | 仅PARC200/PARC400成功后；不复用主held-out |

