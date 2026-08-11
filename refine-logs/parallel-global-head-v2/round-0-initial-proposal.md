# Research Proposal：APEX-16 — Acceptance-Prefix Axial Exchange Head

## Problem Anchor

- **Bottom-line problem：** 在不改变 DFlash 一次并行生成整块这一核心优势的前提下，设计一个轻量 head，显著解决 accepted length 偏低的问题；固定与动态 EAL 都至少达到同作业 released Domino 的 `1.15x`，最终同栈 SGLang 端到端吞吐也至少达到 Domino 的 `1.15x`。
- **Must-solve bottleneck：** DFlash base Top-1 在当前 full16 disjoint development 上为 `6.0685131195`，Domino 为 `7.2395529640`，目标为 `8.3254859086`。PGCF-v1 虽能同集拟合 Top-16 oracle、且延迟足够低，但 held-out 仅为 `6.1027696793`；它把大量修改浪费在首拒之后，并只修复 `46/946` 个可修首拒。完美只修一次的 oracle 也仅为 `7.4985422741`，而完美修正前两次错误可达 `8.4238338192`。因此新方法必须利用完整16位全局信息，在一次并行输出中学会多位置、相互一致的 clean-prefix 修复，而不是只做首拒 gate 或单点修补。
- **Non-goals：** 不做 Domino/GRU 式自回归，不做 selected-token feedback，不做串行 target seed/decode，不做 Jacobi 或任何迭代 refinement，不做 beam/tree/trie/forest/multipath，不让 Top-16 变成路径维，不增加 ordinary verifier 之外的在线 target inference，也不在 accepted-length 主机制成立前投入 SGLang 小修小补。
- **Constraints：** 单次 head 必须同时消费完整 `[B,16,*]` DFlash online features；每个输出位置必须通过无 causal mask 的全局 mixer 看到全部16位；一次产生 `[B,16,16]` scores 并以一次逐位置 argmax 得到唯一 `[B,16]` 序列。Top-16 只作每位置候选轴。训练/选择/held-out prompt 必须隔离，target 信息只作离线标签。新增参数先控制在 `10.75M` 内，并以同 A40、同 BF16、同 batch/block、eager-to-eager 公平 profile 约束成本。
- **Success condition：** 一个未经 target-feature 泄漏、在新 disjoint held-out 上成立的 full16 global-vs-local 机制信号；固定 EAL 至少 `8.3254859086`、动态 EAL 至少 `1.15x` Domino，三个域不退化；随后同栈 SGLang A40 tokens/s 的 paired 95% CI 下界至少 `1.15x` Domino。任何架构不变量失败均为 hard NO-GO，不能用 oracle、same-set capacity 或 off-spec 系统结果替代。

## Technical Gap

### 当前 pipeline 在哪里失败

PGCF-v1 的 256-node flat self-attention 有足够的记忆容量，却没有形成可泛化的
sequence prior。它在 validation 上改动 2,641 个 token，其中 2,487 个位于
base 首拒之后；756 个被改 block 中有 619 个只改 suffix。v1 的
`-sum_m prod_{i<=m} q_i(gold_i)` loss 还会把位置 `i` 的梯度乘上此前所有
gold 概率：早期位置尚未学好时，恰恰需要共同修好的第二、第三个位置几乎收不到
梯度。这与实测的“训练提升、held-out 下降、只会少量首拒修复”完全一致。

### 为什么朴素修补不够

- **更长训练或更大 flat head：** v1 已能同集拟合 oracle，且晚期 train 上升而
  validation 恶化；容量不是当前短板。
- **只保护首拒或加 KEEP gate：** 完美只修一次的 EAL `7.49854`，从上限上就
  过不了 `8.32549`。
- **uniform dense CE：** 会让大量后部 token 数量上占优，重现 suffix-only
  修改，且没有显式刻画 accepted-prefix 的位置成本。
- **Domino action imitation：** causal policy 不是目标方法的可辨识真值；v1 的
  teacher phase 没有产生 transfer，最终目标也不是复制 Domino。
- **序列搜索或迭代：** 即使可能提高准确率，也违反一次并行单链合同。

### 两条候选路线

**Route A — 最小训练修补：** 保留 v1 flat 256-node head，只加 dropout、早停和
continuation-weighted CE。优点是改动最小；缺点是 flat lattice 在历史 matched
representation test 中弱于 axial，且 v1 的 global-local 仅 `+0.01385`，无法证明
它学到了可迁移的全局结构。

**Route B — APEX-16：** 把无结构的 256-node 全连接 mixing 换成
“位置内候选比较 → 16个候选分布摘要 → 每个候选对全部16位置做无 mask exchange”
的 axial mixer，并用非消失的 all-prefix log-risk 蒸馏。旧 GCLS 的三 seed
证据中，axial global-local 为 `+0.17165`，prompt-bootstrap CI
`[+0.11099,+0.23413]`；这不是目标成功，但说明 axial 的跨位置归纳偏置比 flat
更可信。Route B 改变的是一个主机制：如何在不选 token、不走路径的前提下，
让每个候选读取完整 block 并把监督集中到所有可达前缀。

选择 **Route B**。它仍是一个 head、一次调用、一个序列；新组件更少于 v1
flat attention 的全连接关系，且直接针对已经观察到的结构与梯度失败。

## Method Thesis

- **One-sentence thesis：** 用轻量 axial candidate exchange 在一次无 mask 前向中
  让每个位置的每个 Top-16 候选读取全部16位置的候选分布，再用 all-prefix
  log-risk 蒸馏同时学会前两次及更深的 clean-prefix 修复，从而以唯一并行单链
  超过 causal Domino。
- **Why smallest adequate：** 只新增一个 4.54M head；没有 decoder、gate、
  second pass、path state 或在线 teacher。训练增强不进入推理。
- **Frontier relevance：** 采用 target distribution offline distillation 与
  non-autoregressive denoising-style evidence dropout，但保持 single-pass inference；
  不把 iterative masked decoding 当作部署捷径。

## Contribution Focus

- **Dominant contribution：** acceptance-prefix-aligned axial candidate exchange：
  一个专门为 DFlash full16 lattice 设计的全局非因果、一次并行、单链候选 head。
- **Supporting contribution：** all-prefix log-risk + hard-position evidence dropout，
  用训练方式解决 parallel head 的多位置 gradient starvation，不增加在线组件。
- **Explicit non-contributions：** 不提出新 verifier、tree kernel、target early exit、
  recurrent decoder、iterative denoiser 或新的 DFlash backbone。

## Proposed Method

### Complexity Budget

- **Frozen/reused：** Domino checkpoint 中的 DFlash parallel backbone、共享
  embedding/LM-head、base Top-16 GEMM/gather、ordinary target verifier。
- **New trainable component：** 一个 APEX-16 head；默认
  `d=256, heads=8, layers=2, FFN=4, L=16, K=16`，additive node encoder，
  精确预计参数 `4,539,888`（实现时硬断言）。
- **Intentionally excluded：** flat 256-node full attention、trainable vocabulary
  table、GRU、causal mask、token feedback、sequence DP/Viterbi、beam/tree、
  confidence routing 和额外 target features。

### System Overview

```text
one DFlash parallel forward
  -> H[B,16,2560] + base Top16 ids/logits[B,16,16]
  -> 256 candidate nodes (all online-available features)
  -> per-position K=16 candidate-set attention
  -> 16 soft candidate-distribution summaries
  -> every candidate queries all 16 summaries, no causal mask
  -> repeat 2 axial blocks
  -> residual scores[B,16,16] + base logits
  -> one argmax over K for all positions
  -> exactly one proposal[B,16]
  -> ordinary one-chain target verification
```

### Core Architecture

对位置 `i`、候选 `k`：

\[
x^0_{ik}=\mathrm{LN}(W_h\,\mathrm{RMS}(H_i)
 +W_e\,\mathrm{RMS}(E[c_{ik}])
 +W_a\,\mathrm{RMS}(E[y_0])
 +p_i+r_k+W_\phi\phi_{ik}).
\]

每个 axial block 先只在同一位置的16个候选间做 set attention，得到
`u_ik`。随后以 detached base conditional probability 加一个 learned scalar
pool 形成完整候选分布摘要：

\[
\pi_{ik}=\mathrm{softmax}_k(\log q^{base}_{ik}+g(u_{ik})),\qquad
z_i=\sum_k\pi_{ik}u_{ik}.
\]

每个候选 `u_ik` 都作为 query，对 `z_1,...,z_16` 做带相对位置 bias 的
**完整无 mask attention**：

\[
m_{ik}=\mathrm{Attn}(Q u_{ik},\{Kz_j,Vz_j\}_{j=1}^{16}),
\qquad x'_{ik}=u_{ik}+W_o m_{ik}+\mathrm{FFN}(\cdot).
\]

这里没有 hard candidate、previous token 或已选序列。第二层仍同时处理全部
`[16,16]` nodes。最终：

\[
s_{ik}=b_{ik}+W_{out}\mathrm{LN}(x^2_{ik}),\qquad
\hat y_i=c_{i,\arg\max_k s_{ik}}.
\]

`W_out=0` 初始化，因此 step0 精确恢复 base Top-1。一次 tensor argmax 同时产生
16个 token；候选间只交换 soft evidence，不产生任何位置方向上的执行依赖。

### All-Prefix Log-Risk Distillation

令 `r_i` 为 clean target token 在 base Top-16 中的 rank；`h` 为第一个
out-of-K 或 teacher-geometry mismatch 位置，故 `0,...,h-1` 是可监督的 clean
prefix。令 `q_i=softmax(s_i)`。v1 直接最大化 `sum_m prod_{i<=m}q_i(r_i)`，
其后位梯度会被前缀概率相乘衰减。APEX 改为最小化所有可达前缀的负 log
probability：

\[
\mathcal L_{AP}
=-\frac{1}{Z}\sum_{m=0}^{h-1}\log\prod_{i=0}^{m}q_i(r_i)
=\frac{1}{Z}\sum_{i=0}^{h-1}(h-i)[-\log q_i(r_i)].
\]

这仍然对应 accepted-prefix：位置越早，错误会摧毁越多后续 prefix，权重越大；
但任何 `i<h` 都获得非零直接梯度，因此第二、第三次修复不会等第一处概率先学到
接近1才开始训练。它也不会监督第一个 out-of-K 后的不可实现 suffix。

为降低 hard-label 过拟合，用同一次离线 target forward 已保存的 candidate logits
构造温度 `T=2` 的 soft teacher，目标分布固定为：

\[
\tilde p_i=0.9\,\delta_{r_i}+0.1\,
\mathrm{softmax}(t_i/T).
\]

上式直接替换 `-log q_i(r_i)` 为 `CE(tilde p_i,q_i)`，仍使用相同 `(h-i)`
权重。Target logits 永远只作离线 label；不进入 head 输入或部署。

### Hard-Position Evidence Dropout

训练时另构造一个 counterfactual view，只用于迫使 global exchange 学到可泛化
remote evidence：

1. 在 clean prefix 内找到 base rank0 的前两次错误位置；若不存在则不构造该 view。
2. 从这至多两个位置中均匀选一个，只屏蔽该位置的 `H_i` projection 与 logit
   scalar features；候选 IDs/embeddings、rank和其他15位置保持原样，gold 从未作为
   输入。
3. 用同一 APEX head 和同一 all-prefix loss 训练；masked view 权重固定 `0.25`。

Normal view 在每个 batch 始终存在。该 dropout 不向模型透露正确 token，也不进入
推理；它只是让“从其他15位置读取候选一致性”在训练中成为必要能力。若 global
相对 matched-local 在该机制 probe 上仍无优势，说明 remote information 本身不可
辨识，应尽早停掉而不是加模块。

### Modern Primitive Usage

- **Primitive：** offline knowledge distillation + denoising-style evidence dropout。
- **Role：** target 是只在训练时出现的 teacher/critic；masked local evidence 迫使
  单次双向 head 学 token interdependency。
- **Why natural：** Mask-Predict/GLAT 的核心证据是 non-autoregressive one-pass
  generator 需要显式依赖学习与 teacher distillation；APEX 只借用训练思想，明确
  删除它们的 iterative inference、gold-token glancing input 与多候选 decoding。

### Integration

APEX 接在现有 DFlash full16 hidden 与 base Top-16 之后，替换 Domino causal GRU
correction head。共享 embedding 通过 checkpoint 后预投影表 gather；不复制
537.427M DFlash backbone。正式 online dataflow 只有一次 APEX call。最终 verifier
输入仍是一条普通16-token proposal，SGLang 不需要 tree mask 或多路径 KV commit。

### Training Plan

1. **G0 mechanics/capacity：** 512 train blocks；验证 identity、双向 remote
   visibility、full16、one-call/one-chain、loss finite，以及 normal/masked 两视图能
   拟合。Same-set 只作实现/容量证据。
2. **G1 fresh small-data transfer：** 1,987 train prompts 先按固定 label-independent
   permutation 分成 fit/select/diagnostic，比例80/10/10；`dropout=0.1`、
   AdamW、weight decay `0.01`、最多2 epochs，checkpoint 只看 internal select EAL。
   同初始化训练 global、matched-local、global-no-evidence-dropout 三臂。旧
   `validation_select` 只报告诊断，不选模型。
3. **G2 one-shot new heldout：** 按既有 `validation_gate` 的149个 prompt IDs
   新采严格 R047 full16 labels；冻结 G1 checkpoint 后一次性比较 global/local/base/
   Domino。只有 global-local `>=0.15`、prompt-bootstrap CI lower `>0`、global
   EAL `>=7.55`、三域 `>=base`，才允许 full16 OPB scale-up。
4. **G3 scale-up（严格条件式）：** 仅 G2 通过后收集约25K prompts/~200K full16
   blocks，冻结同一 architecture/loss；三 seed development 要求 EAL
   `>=8.3254859086`，再进入 dynamic 和 SGLang。任何失败不授权 off-spec fallback。

### Failure Modes and Diagnostics

- **同集能拟合、internal与fresh heldout都不升：** remote candidate lattice 对
  target choice 不可辨识或 objective 仍过拟合；关闭 APEX，不加参数。
- **global与local都升、二者差小：** 改善来自 prefix objective而非全局 exchange；
  不能声称 global contribution，也不进入 scale-up。
- **masked probe升、normal EAL不升：** augmentation train-test mismatch；删除
  evidence dropout 的后续路线只能作为新变体重新冻结，不能 post-hoc 选权重。
- **train和internal都欠拟合：** 才允许一次 d384/L2（预计仍 <10.75M）容量分支，
  并重做 latency；heldout overfit 时禁止加参。
- **EAL过门但 latency失败：** 只做语义不变的预投影/fusion；不能换 causal、tree
  或额外 target route。

### Novelty and Elegance Argument

Domino 通过 causal GRU 把已选择 token 逐个反馈；APEX 从不选择中间 token，而是让
所有候选同时读取全块 soft candidate distributions。普通 bidirectional head 只说
“看未来”，APEX 的 axial exchange 明确保留每位置候选竞争，再让每个候选访问16个
完整分布摘要。与旧 GCLS/PGCF 相比，贡献不是再堆一层 attention，而是把 full16
candidate geometry、single-pass constraint 和 accepted-prefix gradient 统一成一个
直接 scorer。没有新 verifier、router 或第二模型，论文叙事仍是一条主线。

## Claim-Driven Validation Sketch

### Claim 1：APEX 的全局 exchange 能在单次单链中产生可泛化的多位置前缀修复

- **Minimal experiment：** G1/G2 的 global、matched-local、global-no-dropout 三臂。
- **Baselines/ablations：** base、released Domino、closed PGCF-v1；local隔离全局
  receptive field，no-dropout隔离训练机制。
- **Metric：** prompt-balanced EAL、前两次 base-error repair recall、prefix harm、
  suffix-only edit fraction、global-local paired bootstrap、per-domain EAL。
- **Expected decisive evidence：** fresh heldout global-local `>=0.15` 且 CI lower
  `>0`，global `>=7.55`，前两次 repair 明显提升而 suffix-only 比例下降。

### Claim 2：收益具有达到最终1.15x EAL与吞吐的可扩展路径

- **Minimal experiment：** 仅 G2 通过后做 frozen 200K recipe、三 seed fixed/dynamic
  EAL；机制成功后才 profile/SGLang。
- **Baselines：** same-job released Domino、base DFlash、PGCF-v1。
- **Metric：** fixed/dynamic EAL、eager complete p50/p90、SGLang paired tokens/s。
- **Expected decisive evidence：** development fixed EAL `>=8.32549`，dynamic
  `>=1.15x` Domino，最终 paired TPS CI lower `>=1.15x`。

## Experiment Handoff Inputs

- **Must-prove claims：** global exchange 的 fresh heldout 净贡献；至少两次并行
  correction 的实际回收；最终 EAL/TPS 双1.15x。
- **Must-run ablations：** matched-local、no-evidence-dropout；不增加 flat/causal/
  tree 菜单。
- **Critical data/metrics：** R047 full16 schema、新 `validation_gate` collection、
  prompt-balanced EAL、per-domain、prefix harm、first/second repair。
- **Highest-risk assumptions：** frozen DFlash full16 lattice 对 target 的第二次及
 以后 correction 含有足够可泛化信息；soft target candidate logits 的 teacher-forced
几何能迁移到 ordinary verifier。

## Compute & Timeline Estimate

- **G0/G1：** 单 A40 约1–2 GPU-hours，含三臂与公平 eager profile。
- **G2 collection/eval：** 单 A40 <1 GPU-hour。
- **G3（仅过门后）：** 约8–20 A40 GPU-hours，取决于 full16 collection throughput
  和三 seed 训练。
- **Timeline：** 当天完成实现、code review、G0/G1；机制门通过后再用1–2天完成
  scale-up。不会在 acceptance 失败时提前做系统集成。
