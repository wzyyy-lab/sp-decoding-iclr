# Experiment Plan

**Problem**: 在保持DFlash一次并行生成完整16-token block的前提下，用一个轻量、全局noncausal、一次调用且只输出一条序列的head，把fixed/dynamic接受长度和A40端到端吞吐都提升到released Domino的至少1.15倍。  
**Method Thesis**: PARC-16用完整`16×16` edit-action全局混合，加上immutable-reference conditional accepted-gain与deterministic prefix-harm约束，把梯度集中到真正能延长accepted prefix且不破坏已有正确前缀的编辑。  
**Date**: 2026-08-10

> Authoritative override：本计划不含capacity、same-set、512/2K/25K efficacy run，也不提交独立GPU smoke。M0仅为本地代码级fail-fast检查，不是实验；第一项科学训练直接是90K-prompt full16正式训练。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1：PARC-16的fixed-reference gain-constrained并行纠错能显著提高真实accepted prefix | 这是方法的唯一主贡献；不能再用token accuracy或训练集拟合代替 | 在锁定checkpoint后的一次sealed held-out job中，fixed与dynamic prompt-balanced EAL均`>=1.15x` same-job released Domino，harm`<=1%`，chat/code/math均不退化 | B1 |
| C2：该增益能转化为更高端到端吞吐 | 用户要求最终不仅接受更长，还要真实加速 | 同栈A40 SGLang paired ABBA，PARC/Domino TPS ratio的95% bootstrap CI lower`>=1.15`；完整路径计入vocab GEMM、FP32 Top16、gather、head、argmax、verifier与KV提交 | B2 |

**Anti-claims to rule out**:

- 增益来自自回归GRU、串行target decode、迭代、beam/tree或多路径：online graph逐项断言不存在这些路径。
- 增益来自训练集记忆或held-out回调：90K/5K/remainder按prompt先切分；validation只选checkpoint；held-out只打开一次且之后禁止训练。
- 增益只是更大模型：新增head固定2,438,400参数（DFlash的0.454%），并报告完整延迟与TPS。

## Paper Storyline

- Main paper must prove：B1的sealed held-out fixed/dynamic EAL主结果；B2的same-stack SGLang吞吐结果。
- Appendix can support：train-only numeric certificate、harm envelope诊断、错误类型和可达Top16 oracle gap。
- Experiments intentionally cut：capacity/overfit、小数据screen、LR/width/loss-weight sweep、frozen-first、local-first、候选树、serial seed、任何post-heldout retraining。
- matched-local与constraint deletion只有B1/B2成功后才是exploratory appendix；不得延迟主结果，也不得复用主held-out。

## Frozen Data Contract

- Source：已清洗的`artifacts/manifests/open_perfectblend_100k_v2.jsonl`，从原始prompt重新生成full16数据；旧15-position cache不能作为PARC输入、label、reference或效果证据。
- Split before labels：按domain分层并用固定seed对sample ID排序，固定90,000 train、5,000 validation、remainder held-out；三者零prompt重叠，并继续排除既有development/formal prompts。
- Train/validation materialization：每prompt生成至少129个target-greedy continuation tokens，使用8个固定均匀anchor；每个anchor保存16个gold、released pure-DFlash reference Top16/accepted length和训练所需ordinary DFlash inputs。为支持joint DFlash且避免每step重跑4B target，按prompt保存一次BF16 selected-layer context features，8个anchor共享。
- Train-only certificate：`e_num_cert`、`delta_min`、ambiguous launch gate和train-audit stop只从90K train导出。validation/held-out不进入这些量。
- Held-out seal：训练前只保存held-out prompt manifest，不运行DFlash、Domino或PARC，不产生baseline/statistics。锁定checkpoint后，唯一held-out job从raw prompt共同生成authority并同时评估三系统fixed+dynamic。

## Experiment Blocks

### Block 1: Sealed Held-Out Accepted-Length Result

- Claim tested：C1。
- Why this block exists：直接回答“接受长度是否真正超过Domino很多”，不再用训练集或proxy。
- Dataset / split / task：Open-PerfectBlend；90K train、5K validation、remainder held-out；full16 fixed anchors与同prompt dynamic rollout。
- Compared systems：released pure DFlash、released Domino-b16、P​ARC-16 + joint DFlash；三者在held-out同job同evaluator运行。
- Metrics：primary为prompt-balanced fixed EAL、dynamic EAL、PARC/Domino ratio、actual reference-prefix harm；secondary为chat/code/math EAL、paired bootstrap CI、Top16 oracle gap recovery、support-drop与错误位置分布。训练集EAL只作NaN/优化诊断。
- Setup details：P​ARC D256/H8/L2/FFN512，2,438,400参数；batch8 blocks、无gradient accumulation、180K optimizer steps、head LR`3e-4`、DFlash LR`1e-5`、warmup2K、cosine到初始LR的10%、AdamW、clip1、seed0。每10K仅在5K validation选择checkpoint；在harm`<=1%`者中EAL最大，tie取最早。
- Success criterion：held-out fixed和dynamic EAL均`>=1.15x` same-job Domino；actual harm`<=1%`；三个域在fixed/dynamic均不低于Domino。
- Failure interpretation：任一门失败即关闭当前route；不得扩数据、refresh、改width/loss或再次使用该held-out。
- Table / figure target：Main Table 1（fixed/dynamic整体与分域）；Figure 2（accepted-length CDF和first-reject shift）。
- Priority：MUST-RUN。

### Block 2: Complete A40 Throughput

- Claim tested：C2。
- Why this block exists：更长accepted block只有转化为真实TPS才有系统价值。
- Dataset / split / task：B1通过后使用冻结的production semantics与固定prompt order；A40 batch1 eager开发profile，随后same-stack SGLang服务基准。
- Compared systems：released Domino与冻结PARC checkpoint；相同DFlash backbone、target、dtype、backend、page size和生成长度。
- Metrics：complete cycle p50/p95、component latency、peak memory、tokens/s、paired ratio及95% bootstrap CI。
- Setup details：PARC路径完整计入base vocab GEMM、FP32 Top16、77.8MB projected-table gather、完整global head、argmax、ordinary verifier和KV commit。先eager-to-eager定位，再做same-stack SGLang；不允许用未优化Domino作对手。
- Success criterion：SGLang TPS ratio 95% CI lower`>=1.15`；且`T_PARC/T_Domino <= (EAL_PARC+1)/(1.15*(EAL_Domino+1))`。
- Failure interpretation：若EAL通过但TPS失败，只允许保持weights/architecture/tokens/decision完全不变的kernel、fusion或static-buffer优化；不能改模型后重用held-out。
- Table / figure target：Main Table 2（EAL/TPS/latency/memory）；Figure 3（latency breakdown）。
- Priority：MUST-RUN，严格依赖B1通过。

### Block 3: Exploratory Mechanism Diagnosis

- Claim tested：仅诊断global visibility与constraint的贡献，不支持本轮confirmatory claim。
- Why this block exists：主结果成功后帮助解释增益来源；绝不延迟B1/B2。
- Dataset / split / task：validation或未来新封存数据，禁止复用主held-out。
- Compared systems：成功的global PARC、matched-local attention mask、constraint deletion。
- Metrics：EAL、harm、first-reject repair、global-local delta。
- Setup details：除唯一删除项外完全匹配主recipe；只有B1/B2成功后才计划执行。
- Success criterion：仅探索性报告，不设主门。
- Failure interpretation：不能声称global visibility或constraint是confirmatory因果来源。
- Table / figure target：Appendix。
- Priority：NICE-TO-HAVE。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | 实现级fail-fast，不是实验 | PARC000：本地unit/shape/autograd/identity/harm枚举与CLI静态检查；禁止GPU job | 本地tests/py_compile/bash-n全过即进入M1 | CPU分钟级 | 只防明显实现错误，不能据此判断效果 |
| M1 | 生成真实full16 train/validation并封存held-out | PARC010 split；PARC020 8-way A800 train/validation materialization；PARC030 train-only numeric certificate | 90K/5K/remainder零重叠；full16/8-anchor完整；train ambiguity`<=1%`；validation/held-out未进入证书 | 约110–190 A800 GPU-hours，主要是100K target generation/features | target feature存储约数百GB；按prompt共享并分片写入，支持任务级恢复 |
| M2 | 唯一正式主训练 | PARC100：global PARC + joint DFlash，180K steps，validation每10K | 至少一个validation checkpoint harm`<=1%`；按冻结规则选唯一best | 约60–120 A800 GPU-hours；24h作业需周期checkpoint与原样resume | full-backbone optimizer/验证耗时；保存optimizer/RNG/sampler位置以避免重启改变run |
| M3 | 唯一sealed效果裁决 | PARC200：同一Slurm job/array在held-out共同运行DFlash/Domino/PARC fixed+dynamic并聚合 | B1全部硬门通过才进入M4；失败则route关闭 | 约20–40 A800 GPU-hours | held-out只能打开一次；job必须先验证checkpoint锁定receipt再读取manifest |
| M4 | 端到端性能裁决 | PARC300 complete A40 eager；PARC400 same-stack SGLang paired ABBA | TPS ratio 95% CI lower`>=1.15` | 约4–12 A40 GPU-hours | 只允许语义不变工程优化 |
| M5 | 后验诊断 | PARC500 matched-local/constraint deletion，仅B1/B2成功后 | 不影响主claim | 待定 | 不复用主held-out |

## Compute and Data Budget

- Total estimated GPU-hours for must-run path：约194–362 GPU-hours；数据与训练使用A800，正式profile/serving使用A40。
- Data preparation needs：重新从raw prompt生成129-token continuation、完整16-position DFlash inputs/reference；按prompt共享BF16 target context features；预估数百GB，写入分片且拒绝混用旧15-position cache。
- Human evaluation needs：无。
- Biggest bottleneck：joint DFlash 180K训练与每10K全5K validation；训练脚本必须支持不改变scientific run的exact resume。

## Risks and Mitigations

- 训练成本超过单个24h Slurm时限：每1K step保存latest optimizer/model/RNG/sampler state；只允许同config原样resume，best仍由固定validation规则决定。
- full16 collector最后anchor少一个label：target continuation固定至少129，anchor上界为`len(continuation)-17`，并对每条gold shape `[16]` fail-fast。
- joint DFlash使用stale hidden：训练只读取共享target context features，DFlash hidden/base logits/Top16每step由live DFlash重算；绝不训练在离线parallel hidden上。
- held-out泄漏：held-out manifest路径在训练launcher中不可见；PARC200要求锁定checkpoint receipt且首次创建正式输出目录。
- harm约束被support drop规避：immutable `a_b`不重算；protected gold掉出live Top16时`Hbar=1`且gain mask，base loss负责恢复。
- latency比较不公平：同A40、batch1、相同dtype/backend；完整双方路径；最后以same-stack SGLang TPS为authority。

## Final Checklist

- [x] Main paper tables are covered
- [x] Novelty is isolated tofixed-reference gain-constrained objective
- [x] Simplicity is defended bysingle 2.438M one-call head and no fallback
- [x] Frontier contribution is justified without diffusion/RL/search decoration
- [x] Nice-to-have runs are separated from must-run runs
- [x] No capacity/smoke efficacy stage exists
- [x] Validation and sealed held-out roles are disjoint and irreversible

