# Research Proposal: FBSA-DFlash——面向首个断点的安全表示适配

> 路线身份：`prospective-v2`。这是 R083 已关闭路线之外的一条全新、前瞻式路线；不得把本方案、数据或实验描述成 R083 的重试、修复或下游阶段。

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Technical Gap

### 已有证据，而不是新的猜想

1. **候选可用性不是主瓶颈。** 本项目在同一 DFlash block-16 设置下测得 released DFlash EAL 为 `5.1120019436`，而 K=16 候选 oracle 约为 `9.727`。正确延长路径大量存在，但现有机制无法安全地把它们变成可接受前缀。
2. **冻结表示上的选择器已经出现稳定饱和。** GCLS 的最佳已完成结果约为 `+0.28499 EAL`，但 harm 仍为 `7.32%`，收益几乎集中在 rank-2；FMAS、SAVS 和 CAMRS 即使通过某些 surrogate/capacity 检查，完整训练仍分别出现 identity collapse、约 1,518 倍 harmful-gradient dominance、以及 tail/calibration 失效。更关键的是，27,482,160 参数的 D640 冻结特征 teacher 在 99,356 个 Open-PerfectBlend prompt 上仅得到 `+0.07799 EAL`，且比紧凑 d64 selector 低 `0.15063 EAL`，prompt-bootstrap CI 为 `[-0.23652,-0.06803]`。这不是信息论不可能性证明，但已经是停止继续堆叠冻结外部 selector 的充分工程证据。
3. **Direct/PROS 说明“多 APPLY”不等于安全修复。** PROS R082 的 checkpoint 几乎退化为 Direct：beneficial APPLY `174/174`，但 harmful KEEP 仅 `5/101`，harm `6%`，相对 Direct 只增 `0.003125 EAL`。问题不是门不够大，而是冻结表示没有提供可泛化的“先保住已正确前缀、再修复当前首错位”的更新方向。
4. **DFlash 的可改接口非常集中。** released DFlash 使用五层并行 draft backbone；五个 target hidden layer 先经共享 `fc` 融合，同一融合上下文进入 draft layers。当前外部 selector 只在最终 frozen hidden/logits 上决策，无法改变生成首错位所需的内部表示。
5. **Domino 的差距可分成 backbone 与顺序 head 两部分，但不是受控架构结论。** 本地同锚点 released 结果为 Domino parallel backbone `5.93853`、on-policy GRU `7.01579`；相对 DFlash 分别多 `0.82653` 和 `1.07726`。Domino 使用更大训练量，因此这些数值只定位“表示 + 条件修复”是值得攻击的缺口，不能证明某一架构组件本身优越。

### 前沿边界与避免重复

- [DFlash](https://arxiv.org/abs/2602.06036) 建立了单次并行 block drafter；本方案复用它，不把并行解码本身作为贡献。
- [Domino](https://arxiv.org/abs/2605.29707) 使用 parallel backbone、轻量 prefix-causal correction 与 base-anchored curriculum；本方案不引入顺序/因果 correction head。
- [DFlare](https://arxiv.org/abs/2606.02091) 已把 DFlash 的窄共享条件接口归因为瓶颈，并提出逐 draft-layer target fusion、异构 KV projection 和 progressive position-weighted loss；本方案明确不增加 layer-wise fusion，也不声称“分层融合”新颖。
- [DeLS-Spec](https://arxiv.org/abs/2607.07409) 与 [DSpark](https://arxiv.org/abs/2607.05147) 已覆盖独立顺序 expert、logit fusion 及 confidence/load-aware verification；本方案不增加第二 expert、第二 draft forward 或运行时融合。
- D-PACE、VSD 等已有 acceptance-aware/position-aware 训练。因此“接受率加权”或 LoRA 本身均不是贡献。缺少的是：**在真实的当前 greedy 首断点上，做 base-relative、非对称的 PROTECT/REPAIR 表示更新，并把更新完全并回原并行 drafter。**

### 两条路线的选择

- Route A：继续扩大 frozen hidden/logit selector、teacher 或蒸馏门。优点是复用现有 canonical cache；缺点是已经被 27.5M teacher 的全量反结果直接削弱，而且会继续优化代理分数而非首错位表示。
- Route B：在 DFlash 内部做受约束的参数高效表示适配。优点是直接改变首断点 logits，同时合并后不增加运行时模块；风险是训练稳定性与新数据成本。

选择 **Route B**。这是现有证据下最小但足以触及瓶颈的干预；Route A 在本路线中被停止，不再作为并行主贡献。

## Method Thesis

- One-sentence thesis: 只在 DFlash 最后两层 draft backbone 中学习可合并的低秩更新，并用动态首断点上的 `PROTECT`/`REPAIR` 非对称目标约束更新方向，可以在不增加推理图的前提下把“冻结表示不可选择”转化为“内部表示可安全修复”。
- Why this is the smallest adequate intervention: 不改 target、不改共享 target fusion、不加 selector/RNN/expert/额外 forward；只允许一个训练时目标和一组最终被合并的低秩权重。
- Why this route is timely in the foundation-model era: 它利用 parameter-efficient adaptation 与 teacher-generated exact continuations，把昂贵 target 只放在离线训练/验证阶段；部署仍是原 DFlash 的一次并行 forward + 一次 target verification。

## Contribution Focus

- Dominant contribution: **First-Break Safe Adaptation (FBSA)**——一个随当前模型 greedy 路径动态移动的首断点训练支持集，以及 base-relative 的非对称安全更新：已正确的可达前缀只允许小幅退化，当前首错位获得集中修复。
- Optional supporting contribution: 把 LoRA 合并回最后两层 draft 权重，使训练时的表示适配在部署时不形成新模块；这是工程载体，不单独宣称算法创新。
- Explicit non-contributions: LoRA、D-PACE、位置加权、target distillation、并行 speculative decoding、layer-wise target fusion、顺序 correction head、candidate selection 均不单独声称新颖。

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: Qwen3-4B target 全冻结；released DFlash 的 embedding、LM head、共享 `fc`、前三个 draft layers 及其他原始参数全冻结；greedy T=0 verification 与 block size 16 不变。
- New trainable components: 最后两个 draft layers 的 `q/k/v/o` 与 MLP projection 上 rank-16 LoRA，预计不超过约 2.5M 参数；实现后以精确参数计数为准。训练完成后将增量合并进原权重。
- Tempting additions intentionally not used: 不用外部 selector、校准器、第二 expert、GRU/Transformer correction head、tree/multipath、layer-wise target fusion、运行时阈值、第二次 draft forward 或 post-hoc domain policy。

### System Overview

```text
prospective prompt
  -> frozen target greedy continuation y_1:L + frozen target hidden context
  -> released DFlash logits b_1:L              (no_grad reference)
  -> DFlash + mergeable LoRA logits z_1:L      (trainable)
  -> locate current first mismatch m_theta
       prefix before m_theta: PROTECT
       m_theta: regression-repair or novel-repair
       suffix after m_theta: only a small fixed coverage floor
  -> update LoRA only

deployment:
  merge LoRA into DFlash weights
  -> exactly one ordinary DFlash parallel forward
  -> exactly one ordinary target verification
  -> exact target-equivalent output
```

### Core Mechanism

- Input / output: 对每个长度 `L=16` 的 target-greedy block，输入精确上下文与 gold continuation `y_1:L`；输出 adapted DFlash 的 16 位置 logits `z_1:L`。冻结 released DFlash 在同一输入上给出参考 logits `b_1:L`。
- Architecture or policy: 令 `a_i^theta = argmax z_i`，当前首断点

  `m_theta = min { i | a_i^theta != y_i }`，若全部正确则 `m_theta=L+1`。

  只把 `i <= min(m_theta,L)` 视为当前真实可达支持；后缀不能冒充已实现收益。LoRA 只放在最后两层，以允许改变最终决策边界而不重构 DFlash 的 target-conditioning 接口。
- Training signal / loss: 定义 gold margin

  `gamma_i(z) = z_i(y_i) - max_{v != y_i} z_i(v)`，并记 frozen-base margin 为 `gamma_i(b)`。

  采用每个 block 等权、每项内部归一化的三项损失：

  1. `PROTECT`：对当前首断点之前的可达正确前缀，要求 adapted margin 不低于 `max(gamma_i(b)-delta, mu_keep)`：

     `L_protect = mean_{i < m_theta} [max(gamma_i(b)-delta, mu_keep)-gamma_i(z)]_+`。

  2. `REPAIR`：若 `m_theta <= L`，只修当前 breaker。若 frozen base 在此处正确，则这是适配造成的 regression，使用更严格的 frozen-margin 恢复；若 frozen base 也错误，则对当前错误 winner `a_m^theta` 施加 suffix-value 加权 hinge：

     `L_repair = ((L-m_theta+1)/L) * [mu_fix + z_m(a_m^theta)-z_m(y_m)]_+`。

  3. `COVERAGE`：固定小权重 `lambda_all=0.05` 的原始 D-PACE/teacher-forced block loss，防止动态 breaker 之外完全无梯度；它不能主导训练，也不能在看 falsifier 后调整。

  主目标为 `L = L_repair + lambda_protect * L_protect + 0.05 * L_coverage`。`lambda_protect`、`delta`、`mu_keep`、`mu_fix` 只允许在 capacity/fit 与 checkpoint 上按预注册小网格选择；falsifier 只打开一次。另报告三项的梯度范数与冲突余弦，验证修复信号没有再次被 harmful positions 淹没。
- Why this is the main novelty: 支持集由**当前模型实际会走到的首个错误**动态定义；训练目标不是对所有位置平均加权，也不是在冻结 logits 上选择候选，而是在原并行 drafter 内执行 base-relative safe policy improvement。适配每修好一个 breaker，监督边界才向后推进，因此训练目标与 EAL 的“第一个错误决定整段收益”结构一致。

### Optional Supporting Component

- Only include if truly necessary: 初始 pilot 不增加独立 supporting module。若 capacity gate 显示 gold hinge 信息不足，只允许把 `L_coverage` 换成 frozen-target top-M KL；不得与 hinge 并列叠加，也必须先经新的计划审查。
- Input / output: 不适用；当前方案只使用已有 target greedy label、frozen base logits 与 adapted logits。
- Training signal / loss: 不适用。
- Why it does not create contribution sprawl: target top-M KL 被定义为失败后的互斥替代项，而非默认组件。

### Modern Primitive Usage

- Which LLM / VLM / Diffusion / RL-era primitive is used: 可合并 LoRA/PEFT，以及冻结大模型生成的精确 greedy continuation。
- Exact role in the pipeline: LoRA 是受约束表示更新的低参数载体；target teacher 提供与验证策略一致的 gold 序列，但不参与反向传播。
- Why it is more natural than an old-school alternative: 相比另训一个 selector/RNN，合并式 PEFT 直接改变造成首错位的内部边界，且部署不引入新状态、调度或 kernel 路径。

### Integration into Base Generator / Downstream Pipeline

1. 加载原版 Qwen3-4B target 与 released DFlash；验证 zero-adapter logits/argmax 与原 DFlash 逐元素一致。
2. target 在 `no_grad` 下为前瞻数据产生 hidden context 与 T=0 continuation；原 DFlash 分支在 `no_grad` 下产生 `b`，LoRA 分支产生 `z`。
3. 只更新最后两层 LoRA；checkpoint 选择完全基于独立 checkpoint split 的预注册聚合指标。
4. 选定 checkpoint 后合并 LoRA，删除 adapter 分支；重新加载 merged checkpoint 并检查 logits 与未合并 adapter 版本在预注册容差内一致。
5. falsifier 只评估一次；输出仍由 target verification 保证与 target greedy 精确一致。任何额外 forward、外部 sidecar 或运行时阈值都视为协议违规。

### Training Plan

1. **全新数据契约。** 从 Open-PerfectBlend 的未使用 remainder 预先冻结 9,000 个 prompt：8,000 fit、500 checkpoint、500 one-shot falsifier。所有 prompt 必须不在既有 100k canonical manifest 中，因此也不接触旧 R083 的任何 downstream record。先按规范化 prompt hash 去重并冻结 manifest，再做任何 outcome/model 计算。
2. **Capacity gate。** 仅在 512 个 fit blocks 上验证：zero adapter 复现；所有 loss/gradient 有限；LoRA 能过拟合首断点并提高 realized EAL；protect 项确实降低相对-base regression。失败即停，不进入 pilot。
3. **Matched control。** 用完全相同 rank-16、层位、数据、steps、optimizer 的 LoRA，分别训练：(a) 原始 D-PACE/全块目标；(b) FBSA。唯一主变量是首断点 PROTECT/REPAIR 目标。
4. **Checkpoint selection。** 预注册少量 checkpoint；排序键依次为 checkpoint paired `Delta EAL`、harm、first-token non-inferiority，不允许查看 falsifier 后改权重或阈值。
5. **One-shot falsifier。** 对 released DFlash、matched D-PACE LoRA、FBSA merged model 一次性物化 raw block outcomes；独立脚本从 raw records 重算指标、bootstrap CI 与每域结果。
6. **Scale rule。** pilot 只有在 FBSA 对 DFlash 的 paired prompt-bootstrap 95% CI 下界大于 0、点估计至少 `+0.30 EAL`、harm 不高于 `5%`、first-token 非劣且 FBSA 明显优于 matched D-PACE LoRA 时，才允许用新的未见 prompt 扩到 25k–50k；否则关闭该机制路线。

### Failure Modes and Diagnostics

- Dynamic support oscillation: `m_theta` 在相邻 step 大幅前后跳；按 breaker position 绘制转移矩阵与每位置 repair/protect 成功率。缓解只允许 EMA teacher-of-self 或每若干 step 刷新 mask，不能新增 head。
- Protect dominates and model stays identical: EAL 不变、repair gradient 被压制；检查三项梯度范数/余弦。只在 checkpoint 内调预注册 `lambda_protect` 小网格。
- Repair creates earlier regressions: harm 或 first-token 下降；以 base-relative margin violation 和 earliest-regression histogram 定位，触发安全门失败。
- Capacity passes but prospective generalization fails: fit/checkpoint 提升而 falsifier CI 不正；结论是机制未泛化并关闭，不做 post-hoc threshold rescue。
- Gain comes only from LoRA capacity: matched D-PACE LoRA 与 FBSA 等价；主机制 claim 失败，即便 FBSA 对 DFlash 有小幅提升也只能报告普通 finetuning 结果。
- Latency or graph changes: merged checkpoint 仍残留 adapter kernel/额外 forward；用 profiler 与执行图审计，超过 released DFlash batch-1 median latency `2%` 或改变 forward 次数即失败。
- Domain concentration: aggregate 提升但任一预注册 domain paired estimate 为负；不发布普适 claim，只能缩小 scope 或关闭。

### Novelty and Elegance Argument

本方案不把 LoRA 或 acceptance-aware loss 包装成新颖点。与 DFlare 相比，它不改 target-to-draft fusion；与 Domino/DeLS/DSpark 相比，它不添加顺序模块或第二 expert；与本项目 GCLS/PROS 相比，它不在 frozen output 上选择，而是直接适配产生首错位的表示。最小的机制贡献是：将 speculative decoding 的非可加 EAL 目标改写为一个**随模型推进的 first-break boundary**，在边界前实施 base-relative trust region，在边界处实施集中 repair，并最终合并回单次并行 drafter。若 reviewer 判断这仍只是“换权重的 curriculum”，本路线必须通过 matched D-PACE LoRA 对照和 breaker progression 证据证明因果，否则不得扩张贡献叙事。

## Claim-Driven Validation Sketch

### Claim 1: FBSA 在不增加部署推理图的前提下，安全提高 released DFlash 的真实接受前缀

- Minimal experiment: 9k prospective prompt pilot；在 500-prompt one-shot falsifier 上比较 released DFlash 与 merged FBSA，逐 block 保存 raw accepted length、首错位置、first-token 与 domain。
- Baselines / ablations: released DFlash；zero-adapter roundtrip；同参数/同算力的 D-PACE LoRA；FBSA 去掉 PROTECT；FBSA 把 dynamic breaker 换成静态 position weight。ablation 只在 fit/checkpoint 运行，只有最终冻结的核心比较进入 falsifier。
- Metric: primary 为 paired prompt-cluster `Delta EAL` 及 95% bootstrap CI；safety 为 harm rate/mean harm、first-token acceptance、每域 paired estimate；systems 为 merged latency、峰值显存、draft/target forward 次数和 target-output exactness。
- Expected evidence: pilot 最低继续门为 `Delta EAL >= +0.30` 且 CI 下界 `>0`、harm `<=5%`、first-token 非劣、无预注册 domain 负向点估计；merged latency 相对 released DFlash 在 `±2%` 内且没有新 runtime module。

### Claim 2: 收益来自 first-break 非对称目标，而非仅仅增加 LoRA 容量

- Minimal experiment: 完全 matched 的 D-PACE LoRA 与 FBSA；在不接触 falsifier 的 checkpoint split 上记录 breaker 随训练向后移动的轨迹、earlier-regression 率与三项梯度冲突。
- Baselines / ablations: matched D-PACE LoRA；无 PROTECT；静态 suffix/position weighting。
- Metric: FBSA-vs-control paired `Delta EAL`、首次断点分布的 stochastic dominance、base-correct prefix retention、repair-to-regression ratio、梯度 cosine。
- Expected evidence: FBSA 对 matched D-PACE LoRA 有正的 paired CI，并表现为首断点整体后移而非少数 tail block 拉高均值；去掉 PROTECT 会增加 harm，静态 position weight 不能复现同等边界推进。

## Experiment Handoff Inputs

- Must-prove claims: (C1) same-graph merged FBSA 对 released DFlash 有可泛化且安全的 EAL 提升；(C2) 该提升由 dynamic first-break PROTECT/REPAIR 机制而非 LoRA 容量解释。
- Must-run ablations: matched D-PACE LoRA；no-PROTECT；static-position mask；zero-adapter/merge equivalence。不要把多个 rank 或新增 module 变成论文贡献。
- Critical datasets / metrics: 全新 OPB remainder；8k/500/500 prompt-level split；raw accepted length；paired prompt-cluster bootstrap；harm magnitude/rate；first-token；domain；latency/graph/forward-count；target-output exactness。
- Highest-risk assumptions: 后两层 LoRA 有足够自由度修复 first breaker；dynamic breaker 训练不会振荡；9k prospective 数据足以区分机制；matched control 能公平复现原 D-PACE；数据 remainder 与旧 100k 的 prompt hash 排除可被独立重放。

## Compute & Timeline Estimate

- Estimated GPU-hours: capacity gate `1–3 A800 GPU-h`；9k prospective target collection/outcome materialization `10–20 A800 GPU-h`；两条 matched LoRA pilot 与 checkpoint evaluation `12–24 A800 GPU-h`；one-shot falsifier 与 profiling `3–6 A800 GPU-h`。总计约 `26–53 A800 GPU-h`，未含只有通过后才允许的 scale stage。
- Data / annotation cost: 无人工标注；只使用冻结 Qwen3-4B target 的 T=0 continuation。需要约 9k prompt 的全新 manifest、内容哈希和独立 split receipt。
- Timeline: 代码/CPU 单测与 fresh review 约 1 天；capacity gate 半天；pilot collection/training 1–2 天；one-shot falsifier、审计与结果归因约 1 天。任何 gate 失败立即停止，预计 3–5 天得出可证伪结论。

