# Research Proposal: Reach-Aligned Multimodal Candidate-Lattice Selection for DFlash

## Problem Anchor

- **Bottom-line problem**: 在 Qwen3-4B / DFlash-b16 的 greedy speculative decoding 中，利用一次 DFlash forward 已产生的完整 top-K candidate lattice，提高真实可达的 accepted draft length，并最终转化为吞吐收益；核心不是证明候选“存在”，而是让 head 能可靠地选择并保护正确路径。
- **Must-solve bottleneck**: 当前 GCLS-v1 的 global signal 已被三 seed 和 matched local/causal 对照证实，但相对 DFlash 仅约 `+0.232` calibrated EAL（最佳 development checkpoint raw `+0.285`），只回收约 5%–6% 的 K16 oracle gap；同时存在 first-miss repair 低、harm 不低、远端候选被单均值压缩、candidate/context 交互弱、训练 support 与实际策略 reach 不一致的问题。
- **Non-goals**: 不把 K16 oracle 当成 selector 可实现收益；不靠复制 Domino 的顺序 GRU 或扩大 verification tree 来改题；首轮不声称 `T>0` lossless sampling；不把在同一 validation split 上选 checkpoint 和校准阈值后的数字当 sealed-test 结论；不同时堆 CRF、GRU、tree search 和多个辅助 head。
- **Constraints**: target model 冻结；首轮优先复用 released DFlash 与现有 100K Open-PerfectBlend canonical records；greedy `T=0`、block size 16、K16；最终部署 head 必须保持一次并行 lattice 处理且开销低于收益 break-even；现阶段有 Slurm A800 资源，但 validation_select 只有 147 prompts，必须保留独立 calibration/test 口径。
- **Success condition**: 先用高容量 frozen-feature ceiling 证明当前输入至少支持约 `+0.6` raw EAL 或回收 ≥15% oracle gap；随后紧凑模型回收 ceiling 增益的 ≥70%，三 seed 均优于 DFlash，global−causal/local 的 prompt-cluster CI 排除 0，harm 不高于 5%，并在独立 calibration/test 与真实 latency 下取得正吞吐收益。若 ceiling 明显低于该区间，则成功结论是及时证伪 frozen-head 路线并转向 DFlash representation adaptation，而不是继续扩 head。

## Technical Gap

当前证据已经排除了几个表面解释。GCLS-v1 能在 128-block capacity probe 上近乎完全记忆，100K 三 seed 的 global 增益也稳定为正，因此它不是一个完全无信号或明显实现错误的模型。但原始结果显示：最佳 d64 global checkpoint 在 train 上 first-miss repair 为 23.7%，validation 只有 18.3%；validation harm 为 7.3%，oracle-gap recovery 仅 6.2%。同一个 global checkpoint 在 local mask 下 non-top1 candidate accuracy 反而更高，却造成约 `-0.80` EAL，证明普通 token classification accuracy 与真实 prefix utility 严重错位。

具体缺口有四个：

1. **信息是否可识别尚未被测量**：当前 0.43M/1.47M head 的低收益不能区分 frozen DFlash feature ceiling 与小模型 ceiling。
2. **远端 lattice 表示有不可逆压缩**：axial block 把每个远端位置的 K 个候选压成一个 DFlash-probability-weighted mean，多峰候选身份在跨位置交互前已经丢失。
3. **candidate-specific compatibility 太隐式**：当前位置 hidden、candidate embedding 与 anchor embedding仅做加法，浅层小模型必须自行恢复“哪个候选与当前/全局上下文兼容”的乘性关系。
4. **loss 的 support 与部署决策不一致**：Candidate-D-PACE 以 gold-in-K coverage 构造 active prefix，而不是按当前 selector 的实际 argmax reach；它会训练首个策略错误之后的不可达 suffix，也没有显式区分 first-miss repair 与 base-correct protection。

继续原样扩到 1.42M prompts 不能回答以上问题；直接加 GRU 会把贡献改成 causal refinement；直接加 CRF 会重新引入 teacher-forced transition/decoder 错位。最小充分路线是先测 frozen-feature Bayes-like ceiling，再只保留两项与部署决策直接对应的机制：保真传递多模态候选证据，以及按实际 reach 训练 repair/protection。

## Method Thesis

- **One-sentence thesis**: DFlash 的全局 candidate lattice 只有在候选多模态证据被保真传递、且监督严格落在当前策略可达的 breaker 决策上时，才能从“存在 oracle 候选”变成稳定的 accepted-length 增益。
- **Why this is the smallest adequate intervention**: 不修改 target、不引入顺序 draft rollout、不改变 verifier，只重做 selector 的 candidate/context interface 与其 head-specific objective；高容量 teacher 仅作诊断和蒸馏，不进入部署。
- **Why this route is timely in the foundation-model era**: 复用冻结大模型 embedding/hidden 作为语义空间，以高容量 teacher 测可识别上限，再进行同输入空间的 lightweight distillation；这比在小 head 上盲目做模块搜索更符合现代 representation-first / teacher-to-student 方法论。

## Contribution Focus

- **Dominant contribution**: 一个 reach-aligned full-lattice selector：显式 candidate-context compatibility、保留多个候选模式的全局通信，以及直接针对可达 prefix 的 repair/protection objective。
- **Optional supporting contribution**: 用不受延迟约束的 frozen-feature ceiling teacher 作为路线门禁，并将其 soft candidate scores 蒸馏给紧凑 selector。
- **Explicit non-contributions**: 不把大 teacher 当部署系统；不提出新的 verifier；不以顺序 GRU、CRF 或 tree decoding 作为并列贡献；不在本阶段承诺 `T>0` proposal correction。

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**: Qwen3-4B target、released DFlash、target token embedding/LM-head 语义空间、现有 canonical hidden/logits/top-K records。
- **New trainable components**: ceiling teacher（仅诊断）、紧凑 GCLS-v2 的 compatibility encoder、多槽 lattice mixer、一个 residual score readout；不新增 full-vocabulary trainable table。
- **Tempting additions intentionally not used**: GRU/RNN、CRF/Viterbi、beam/tree verifier、额外 target forward、多个独立 confidence heads、端到端 RL。

### System Overview

```text
verified prefix
    │
    ├─ target context features ──> released DFlash (one parallel pass)
    │                                  │
    │                                  ├─ h_i
    │                                  ├─ top-K ids/logits C_{i,k}, b_{i,k}
    │                                  └─ full-vocab logsumexp
    │
    └────────────────────────────> frozen target embeddings e(C_{i,k})

{h_i, e(C_{i,k}), b_{i,k}, anchor}
    └─> explicit candidate/context compatibility
         └─> full-lattice teacher OR compact multi-slot mixer
              └─> base-anchored residual scores s_{i,k}
                   └─> parallel argmax path
                        └─> unchanged target verification
```

### Core Mechanism

- **Input / output**: 输入与当前 v1 完全相同：`h_i`、候选 token embedding、raw/full-vocab calibrated DFlash logits、anchor embedding、位置与 rank；输出每个位置 K 个 direct scores，不使用 gold、teacher-forced block prefix 或已选择 token。
- **Architecture or policy**:
  1. 将 normalized hidden 与 candidate embedding 分别投影为 `q_i`、`k_{i,k}`；构造 `[q_i, k_{i,k}, q_i⊙k_{i,k}, a⊙k_{i,k}, scalar]`，用小 MLP 得到 candidate node。乘性项让 candidate compatibility 成为显式一阶特征。
  2. ceiling teacher 使用 20M–50M、4-layer full-lattice attention，不做单均值压缩，以测量当前 frozen inputs 的可实现上限。
  3. 部署模型使用每个 source position 的 `R=4` mode summaries，而不是一个 soft mean；candidate query 可读取所有允许位置的 `L×R` summaries。local/causal/global 只改变 visibility mask，参数完全匹配。
  4. 输出保持 `s_{i,k}=b^{logp}_{i,k}+Δ_{i,k}`，`Δ` readout 零初始化，epoch 0 精确等于 DFlash。动态 base scale 只有在 teacher 表明“能识别但无法跨 margin”时才作为单独 ablation，不默认加入。
- **Training signal / loss**: 默认 Head-AUF（Accepted-Utility Frontier）：令 `ŷ_i=argmax_k s_{i,k}`，以 detached indicator 构造 `r_i=∏_{j<i}1[ŷ_j=y_j]`。主 CE 只作用于 `r_i=1` 且 gold-in-K 的位置，包含当前 breaker、排除首个策略错误后的 suffix。再加入小权重 all-position coverage CE、DFlash first-miss repair margin，以及 base-top1-correct protection margin。当前公式精确的 D-PACE 明确改名 Candidate-D-PACE，仅作为 warm-up/ablation；Reach-D-PACE 只在实际 reach support 上单独比较。
- **Why this is the main novelty**: 它不是给 DFlash 再挂一个通用 Transformer，而是同时修正 candidate lattice 的信息接口和 speculative prefix utility 的训练 support；每一项都对应已观测到的 lossy pooling 或 repair/harm 失配。

### Optional Supporting Component

- **Only include if truly necessary**: frozen-feature ceiling teacher 和同输入蒸馏。
- **Input / output**: teacher 与 student 使用完全相同的 deployable inputs，输出 K-way temperature-scaled candidate logits。
- **Training signal / loss**: teacher 先用 Candidate-D-PACE 与 Head-AUF 分开训练；若 teacher ceiling 足够高，student 在 supervised Head-AUF 外加入 candidate KL，只蒸馏 gold-in-K / reachable decision，不引入 target 额外输入。
- **Why it does not create contribution sprawl**: teacher 是诊断工具和压缩上界，不是第二个推理模块；若直接 supervised student 已回收 ≥70% ceiling，则删除蒸馏。

### Modern Primitive Usage

- **Which primitive is used**: frozen foundation-model representation reuse + capacity teacher / lightweight distillation。
- **Exact role in the pipeline**: teacher 充当当前输入空间的可识别性探针与 student 的软监督源，不作为在线 planner、critic 或 generator。
- **Why more natural than an old-school alternative**: 先区分 representation ceiling 与 architecture ceiling，再压缩，是比继续枚举 RNN/CRF/head size 更直接的因果诊断。

### Integration into Base Generator / Downstream Pipeline

推理仍是 target context feature → 一次 DFlash parallel forward → 一次轻量并行 selector → target verifier。target 与首轮 DFlash 参数冻结，selector 不读 verifier 结果或生成中的 block prefix。raw selector score 是主要方法；KEEP_BASE margin 只在独立 calibration split 上确定，并在 sealed test 上固定使用。若 frozen-feature teacher ceiling 低，下一阶段才解冻 DFlash 的低秩 adapter，且仍保持单次 parallel forward；该 pivot 不与 frozen-head 结果混报。

### Training Plan

1. **D0: ceiling gate**：先在 512 blocks 上确认 compatibility/full-attention teacher 可记忆且 epoch-0 identity；再在 10K prompts 做单 seed smoke，最后在 100K 上训练 Candidate-D-PACE teacher 与 Head-AUF teacher。主要看 raw prompt-balanced EAL、repair、harm 与 train/validation gap。
2. **D1: objective isolation**：固定 teacher architecture，只换 Candidate-D-PACE、reach CE、Head-AUF；不同时改数据、K 或 calibration。
3. **D2: compact v2**：固定 Head-AUF，比较 current single-mean axial、R=4 multi-slot axial、full-flat teacher；若 student-ceiling gap 大，再加一次 teacher KL。
4. **D3: confirmatory protocol**：冻结 architecture/hyperparameters；划出独立 calibration split 与 untouched test，跑三 seeds matched local/causal/global，并报告 prompt-cluster bootstrap。
5. **Conditional pivot**：若强 teacher raw gain仍约 `≤+0.4`，停止扩 selector，改做 DFlash LoRA / layer-wise target fusion 的 representation-adaptation track。

建议的 Head-AUF 形式为：

```text
L = L_reach
  + λ_cov L_coverage
  + λ_rep L_first_miss_margin
  + λ_prot L_base_protection_margin
```

初始小网格只搜索 `λ_cov∈{0.05,0.1}`、`λ_rep∈{0.25,0.5}`、`λ_prot∈{0.1,0.25}`；margin 固定为 candidate log-prob 空间的小常数，并以 repair−harm / raw EAL 而非 token accuracy 选型。

### Failure Modes and Diagnostics

- **Frozen-feature ceiling 低**：teacher 充分收敛后仍接近 `+0.3`；检测 teacher train/validation、candidate-only/future-only replacement；缓解是停止扩 head并做 DFlash LoRA/target-layer fusion。
- **Teacher 高、compact student 低**：说明单均值或压缩容量是瓶颈；用 R=1/2/4/8 deletion curve 和 teacher KL 定位，选择最小可保留 ≥70% ceiling 的 R。
- **Repair 上升但 harm 同步上升**：检查 base-correct protection violation、首 token和各 domain；增加 protection 权重或独立校准，不增加新 confidence head。
- **Loss 看似下降但 EAL 不升**：以 actual-reach breaker confusion、reachable-changed blocks、repair/harm 作为首要诊断，拒绝用 aggregate candidate accuracy 替代。
- **延迟吃掉收益**：测真实 kernel latency；按 R、dim、layers 做 deletion，必要时蒸馏到更小 student或回退 DFlash。

### Novelty and Elegance Argument

DFlash 在一次 parallel pass 后按位置独立输出；Domino/DSpark 用顺序 causal head 修复 block 内依赖；DeLS-Spec 用已生成 prefix 的轻量 local expert；DFlare通过 layer-wise target fusion增强 draft backbone。这里的研究对象不同：在不读取生成 prefix、也不重新执行 target 的条件下，直接把**预先可得的整张 candidate lattice**当成决策对象。相对当前 GCLS-v1，关键不是“更多 attention”，而是保留候选多峰证据并把 loss support 对齐实际 speculative reach。高容量 teacher 只决定 frozen route 是否值得继续，避免把不可识别的信息问题误包装成结构创新。

## Claim-Driven Validation Sketch

### Claim 1: 当前 deployable frozen inputs 是否含有显著高于 v1 的可识别信号

- **Minimal experiment**: 20M–50M compatibility + full-lattice teacher，在相同 100K records 和 validation protocol 下与 d64/d128 v1 比较。
- **Baselines / ablations**: v1 axial d64；teacher additive encoder；teacher compatibility encoder；candidate-only / hidden-only replacement。
- **Metric**: raw prompt-balanced EAL delta、oracle gap recovered、first-miss repair、harm、train-validation gap。
- **Expected evidence**: teacher 达到约 `+0.6` 或 ≥15% gap recovery，且 replacement 破坏增益，才支持继续 frozen selector。

### Claim 2: Reach-aligned supervision比 Candidate-D-PACE 更好地平衡 repair 与 protection

- **Minimal experiment**: 固定同一 teacher 或 compact architecture，只替换 Candidate-D-PACE、reach CE、Head-AUF。
- **Baselines / ablations**: 去掉 coverage、去掉 repair margin、去掉 protection margin；不做大范围超参搜索。
- **Metric**: raw EAL、first-miss repairs、harmed blocks、首 token与最差 domain delta；token accuracy只作诊断。
- **Expected evidence**: Head-AUF 提高净 `improved−harmed` 和 EAL，且 protection ablation 显著增加 harm。

### Claim 3: 多槽通信是部署时保留 global signal 的最小结构

- **Minimal experiment**: matched R=1/2/4/8 student与 full-flat teacher，固定参数范围和 Head-AUF。
- **Baselines / ablations**: current single mean、matched local/causal/global、无蒸馏 student。
- **Metric**: teacher gain recovery、global−causal/local bootstrap CI、head latency与端到端 break-even。
- **Expected evidence**: 最小 R 在 global 下保留 ≥70% teacher gain，local/causal 做不到，并保持正吞吐收益。

## Experiment Handoff Inputs

- **Must-prove claims**: frozen input ceiling；actual-reach objective 的净 repair/protection；多槽 global signal 的必要性与延迟可行性。
- **Must-run ablations**: additive vs compatibility；Candidate-D-PACE vs reach CE vs Head-AUF；R=1 vs最小有效 R；local/causal/global；candidate/hidden replacement。
- **Critical datasets / metrics**: 100K Open-PerfectBlend train；prompt-disjoint validation_select；新 calibration/test；prompt-balanced accepted draft tokens、repair/harm、oracle-gap recovery、prompt-cluster CI、真实 latency。
- **Highest-risk assumptions**: frozen DFlash hidden/embedding 足以识别 target candidate；100K 数据覆盖足够；多槽 student 能压缩 teacher；额外 head latency低于约 2%–4% round cost。

## Compute & Timeline Estimate

- **Estimated GPU-hours**: 先用 10K smoke 控制失败成本；ceiling + objective isolation约 20–40 A800 GPU-hours，compact/R 消融约 15–30 GPU-hours，三 seed confirmatory 约 15–25 GPU-hours，总计约 50–95 GPU-hours，按门禁提前停止。
- **Data / annotation cost**: 现阶段零新增人工标注；若需要独立 calibration/test，复用未观察 prompts 重新 canonical collect；只有 representation-adaptation 或 dense target distillation 才新增 target forward 成本。
- **Timeline**: 1 天完成实现与 sanity/capacity；1–2 天完成 ceiling gate；ceiling 通过后 2–4 天完成 objective/compact 消融与三 seed确认；ceiling失败则在第 2 天切换 LoRA 数据与在线训练设计。
