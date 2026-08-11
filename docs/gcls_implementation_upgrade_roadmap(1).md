# GCLS 全方位实现升级方案

> 从“证明全局候选信息存在”走向“可超过 Domino 的单链并行投机解码系统”  
> 日期：2026-08-04（已纳入 GCLS 额外 head 使用 D-PACE 的训练目标修订）  
> 面向：Qwen3-4B / DFlash-b16 / greedy decoding (`T=0`) 的首轮主线；后续扩展到 `T>0` 与更多模型  
> 依据：项目实验日志、方法设计、失败复盘、结果注册表，以及截至 2026-08-03 的公开论文与官方代码

---

## 0. 如何阅读本文

本文严格区分三类结论：

- **【已有证据】**：来自你当前项目中真实完成的作业、附件中的源码审计或已冻结的实验记录。
- **【外部证据】**：来自公开论文和官方仓库。
- **【建议/推断】**：基于前两类证据提出的实现方案；它们需要实验验证，不能提前写成论文结论。

本文不是建议把所有新模块一次性堆进系统。核心原则是：

> **先定位瓶颈属于信息、表示、架构、目标、优化还是系统开销，再只改一个主要因素。**

否则即使结果变好，也无法形成 ICLR 需要的清晰因果结论；结果变差时，也无法知道应该否定哪一部分。

---

# 1. 一页结论

## 1.1 当前 GCLS-v1 已经证明了什么

【已有证据】你当前的 `d64`、单个 axial block、433,772 参数、K16、`Candidate-D-PACE`（公式与官方 D-PACE 一致，但用于 top-K selector 的语义适配）direct GCLS，已经从“oracle availability 假设”推进到了“selectability 有真实信号”：

| 指标 | 当前最强开发结果 |
|---|---:|
| Raw global − DFlash | `+0.24247` accepted draft tokens |
| Raw global − local | `+0.17383` |
| Raw global − local 95% development interval | `[+0.10544, +0.24356]` |
| Global non-top1 accuracy | `10.00%` |
| Local non-top1 accuracy | `4.13%` |
| Global first-miss repair | `16.26%` |
| Local first-miss repair | `7.11%` |
| First-token accuracy change | `+0.00681` |

而且 raw global-local 在 chat、code、math 三个域都为正。这说明：

> 在已知当前位置的 DFlash hidden、候选和 logits 之后，其他位置的 candidate lattice 确实包含可泛化的增量信息。

因此，当前架构不是“完全没学会”；它已经完成了一个很重要的 **proof of signal**。

## 1.2 当前结果为什么仍然太小

【已有证据 + 推断】在同一 `validation_select` 口径下，DFlash K1 EAL 为约 `5.113`，K16 oracle 为约 `9.723`，oracle gap 约 `4.610`。当前 raw `+0.24247` 只回收约：

\[
\frac{0.24247}{9.723-5.113}\approx 5.26\%
\]

这并不意味着剩余 94.7% 都应该由一个可部署 selector 回收。Oracle 直接知道 target gold，而 selector 只看 draft-side observable features。但它说明当前实现离“充分提取可用信号”还有明显距离。

当前增益小，最可能是下面几类因素共同造成的：

1. **Frozen DFlash representation ceiling**：DFlash hidden 并没有被训练成显式表达 candidate-level 全局一致性。
2. **Head 容量和交互形式不足**：0.43M 参数与单层 d64 更适合作为最小验证模型，不像性能上限模型。
3. **一层信息传播不够直接**：你需要的是 `candidate → global mode → candidate`，单层 mixer 容易只做一次相关性平均。
4. **当前 head loss 存在语义错位**：原始 D-PACE 训练 full-vocabulary drafter；当前实现把同一公式作用于 top-K 条件分类概率，并以 gold-in-K coverage 而不是当前 selector 的真实预测 reach 决定支持集。它仍会给首错后的不可达 suffix 分配梯度，也没有显式区分“修复 DFlash 首错”和“保护 DFlash 已正确位置”。
5. **固定 base logit 系数过于保守**：远 rank 候选必须用 residual 跨过较大 DFlash margin。
6. **当前离线数据与真实 rollout 分布仍有差异**。
7. **数据扩展已出现边际递减**：50K→100K 的 calibrated 增量只有 `+0.03426`，区间跨 0，继续原样扩数据不一定解决问题。

## 1.3 最重要的总建议

不要立即把当前 d64 直接扩到 1.42M 数据，也不要立刻重新加入 CRF、Viterbi 或 GRU。建议按以下顺序推进：

1. **冻结当前 GCLS-v1 作为 scientific baseline**，先完成 seeds、同 checkpoint mask intervention、same-anchor Domino。
2. **训练一个不考虑延迟的 20M–50M frozen-feature ceiling model**，判断当前输入特征到底能支持多大收益。
3. 若 ceiling 高：实现 **GCLS-v2：显式 candidate-context compatibility + local competition + latent global modes + mode-to-candidate feedback**，再蒸馏到轻量模型。
4. 若 ceiling 低：不要继续堆 selector，转向 **DFlash LoRA / joint training / DFlare-style layer-wise target fusion**。
5. 将当前 loss 明确重命名为 **Candidate-D-PACE** 并降为 warm-up/ablation；以 **Head-AUF + 少量 coverage CE + first-miss repair + base protection** 作为下一版默认 head-specific objective，再把 Reach-D-PACE、D-PAKL/TV 与 target distillation 作为独立对照。
6. 最后才做 online replay、adaptive K/length 和 fused serving。

---

# 2. 先把 Domino 的启示理解准确

## 2.1 Domino 实际上不是“冻结 DFlash，只训练一个 GRU”

【外部证据】Domino 论文主实验冻结的是 target model；parallel DFlash-style backbone、GRU causal encoder 和低秩 correction head 都属于 trainable draft module。它使用：

- 5-layer parallel backbone；
- GRU hidden dimension 1024；
- correction bottleneck rank 256；
- 1.42M Open-PerfectBlend prompts；
- responses 由对应 target 重新生成；
- 3 epochs，8×A100-80GB；
- lr `6e-4`、weight decay 0、clip 1.0、cosine、warmup 0.04、bf16/FSDP。

论文与官方代码：

- [Domino paper](https://arxiv.org/abs/2605.29707)
- [Domino official repository](https://github.com/jianuo-huang/Domino)

因此，当前比较混合了多个差异：

| 维度 | 当前 GCLS | Domino 主结果 |
|---|---|---|
| Backbone | released DFlash，完全冻结 | 联合训练的 parallel backbone |
| Head 规模 | 0.433M | 论文报告额外约 56M |
| 输出空间 | top-K candidate scores | full-vocabulary residual |
| 输入依赖 | 全 candidate lattice | realized/gold causal prefix |
| 数据 | 约 100K prompts | 1.42M prompts |
| 训练 | selector only | backbone + head joint |

所以当前 `+0.242` 不能直接被解释为“future candidate evidence 远弱于 causal prefix”。它既可能反映信息差异，也可能反映容量、训练边界和数据规模差异。

## 2.2 Domino head 的精确输入和中间状态

【外部证据】官方源码中：

```python
self.prefix_gru = nn.GRU(
    input_size=config.hidden_size,
    hidden_size=gru_hidden_dim,
    num_layers=1,
    batch_first=True,
    bias=False,
)
```

GRU 输入是已实现 draft token 的 target token embedding。位置 `i` 的 correction 使用：

\[
[H_i; S_{i-1}]
\]

其中：

- `H_i`：parallel backbone 的当前位置 hidden；
- `S_{i-1}`：GRU 对当前 block 已生成 prefix 的压缩状态；
- Qwen3-4B 中大致是 `2560 + 1024 = 3584` 维输入；
- 经过 `3584 → 256 → vocab` 的低秩 logit residual；
- 最终 logits 为 `base_logits + residual`。

推理时它必须按 token 更新 GRU；训练时使用 ground-truth prefix teacher forcing。

## 2.3 Domino 最值得借鉴的不是 GRU，而是四个设计原则

### 原则 A：强而直接的候选特异 lexical correction

Domino 不是只给 hidden 增加一个小 MLP，而是把 `[parallel hidden; causal state]` 映射到 full-vocabulary residual。它对不同 token 有强候选特异表达能力。

**对 GCLS 的启示：** 不必做 full-vocab projection，但应该显式学习：

\[
\text{context representation} \leftrightarrow \text{candidate token representation}
\]

而不是只把共享 `h_i` 和 candidate embedding 相加。

### 原则 B：joint representation adaptation

Domino 的 backbone 会适配 causal head；你的 released DFlash hidden 不会主动暴露 GCLS 所需的 candidate-lattice coherence。

**对 GCLS 的启示：** frozen plug-in 要保留为主科学实验，但性能路线必须增加 LoRA 和 joint training 上界。

### 原则 C：base-anchored curriculum

Domino 发现 clean-prefix correction 会 shortcut backbone，因此同时监督 base 和 final logits，并让：

\[
\mathcal L=(1-\lambda_t)\mathcal L_{final}+\lambda_t\mathcal L_{base},
\quad \lambda_t:1\rightarrow0.
\]

**对 GCLS 的启示：** 一旦解冻 DFlash，必须保护 DFlash base prediction；否则 global head 可能变成唯一有效分支，破坏 fallback、校准和系统鲁棒性。

### 原则 D：训练数据必须 target-on-policy 且足够多样

Domino 使用 1.42M unique prompts，并重新生成 target responses。你的 100K 曲线已经证明 prompt diversity 是真实瓶颈，但也显示原样扩展开始边际递减。

**对 GCLS 的启示：** 先定位 feature ceiling，再决定扩到 400K、1.42M 还是转向 representation adaptation。

## 2.4 不应该从 Domino 复制什么

不建议把 Domino GRU直接接在 GCLS 后作为主方法，因为这样会：

- 重新引入 token-by-token recurrence；
- 让 reviewer 无法判断提升来自 global lattice 还是已知 causal correction；
- 弱化“固定深度并行、单链验证”的核心差异。

GRU 可以作为：

1. **causal upper bound / diagnostic baseline**；
2. **global + causal complementarity 实验**；
3. **GCLS 修第一 token、Domino 生成 suffix 的混合上界**。

但不应替代主线。

---

# 3. 当前实现的完整瓶颈地图

## 3.1 P0：先诊断，不修就不应继续大规模扩数据

### P0-1：不知道 frozen feature ceiling

当前最重要的未知量不是 K16 oracle，而是：

> **只使用推理时可获得的 frozen DFlash features，一个充分强的 global model 在 held-out 上最多能做到多少？**

如果 50M 参数、充分训练、dense target supervision 的模型仍只有 `+0.3`，说明主要瓶颈是信息/表示；如果能到 `+0.8` 或更高，说明当前 0.43M head 才是瓶颈。

### P0-2：candidate-context interaction 过于隐式

当前位置 K 个候选共享同一个 DFlash hidden。如果 node encoder 主要做：

\[
W_h h_i + W_e e_{i,k}+\phi_{i,k},
\]

小模型必须自己学会“这个 hidden 与这个 token 的匹配关系”。建议显式加入：

\[
q_i=W_h h_i,\qquad k_{i,k}=W_e e_{i,k},
\]

\[
q_i\odot k_{i,k},\qquad
\frac{q_i^\top k_{i,k}}{\sqrt d},\qquad
q_i^\top W_b k_{i,k}.
\]

这通常比单纯加法更适合 candidate reranking。

### P0-3：一层 mixer 难以完成两跳推理

你的目标隐含两步：

1. 从整张 lattice 识别 continuation mode；
2. 让 mode 反过来改变每个位置候选分数。

即：

\[
\text{candidate}\rightarrow\text{global mode}\rightarrow\text{candidate}.
\]

单层 self-attention 虽有全局 receptive field，但容易退化为一次平均，尤其当 lattice 同时包含多个合理模式时。

### P0-4：per-position direct score 没有显式处理多模态

Direct logits 是正确的第一主线，但一个单一 pooled/global representation 可能把：

- `of course`
- `no problem`

混成平均模式。需要多个 latent mode slots，而不是强制所有候选通过一个全局均值。

### P0-5：固定 base 权重可能阻碍远 rank 修复

当前：

\[
s_{i,k}=\log p_D(c_{i,k})+\Delta_{i,k}.
\]

安全但保守。首次错误中大量 gold 位于 rank 3–16，corrector 需要跨过较大的 base margin。建议改为受控动态温度：

\[
s_{i,k}=\alpha_i\log p_D(c_{i,k})+\Delta_{i,k},
\]

其中：

\[
\alpha_i=1+a\tanh(g_i),
\]

初始化严格为 1，范围例如 `[0.5,1.5]`。DFlash 确定时保持 base；高 entropy、小 margin 时允许 flatten。

不要恢复旧版 z-score；z-score 会破坏原概率尺度。

### P0-6：当前所谓 “exact D-PACE” 只是公式精确，任务语义已经改变

原始 D-PACE 训练的是完整 draft model：

\[
q_i^{\mathrm{draft}}=p_{\mathrm{draft,full\mbox{-}vocab}}(y_i\mid x),
\]

其动态位置权重把 full-vocabulary draft confidence 当作 acceptance surrogate。

当前 GCLS 使用的是：

\[
q_i^{K}=\operatorname{softmax}_{c\in C_i}(s_{i,c})_{r_i},
\]

即 **gold 已在 top-K 的条件下**，selector 在 K 个候选中的分类概率。二者有三个本质差异：

1. top-K 重新归一化会把 retained mass 之外的概率删除，通常使 \(q_i^K\) 比 full-support probability 更自信；
2. gold 不在 top-K 时 selector 没有对应类别，只能 censor；
3. GCLS 是一个相对 DFlash base action 做 override 的 residual head，而原始 drafter 没有“保留 base 还是覆盖 base”的决策。

因此当前实现应在论文、代码和表格中称为：

> **Candidate-D-PACE：formula-exact D-PACE weighting adapted to top-K candidate reranking.**

现有 100K global-local 结论仍然有效，因为 local/global 使用了完全相同的 Candidate-D-PACE；但它只证明“在该训练目标下 global scope 更好”，不能证明 Candidate-D-PACE 是最适合 GCLS 的 loss。

### P0-7：当前 active mask 表示 candidate coverage，不表示真实 policy reach

当前 mask 主要是：

\[
m_i^{K}=\prod_{j\le i}\mathbf 1[y_j\in C_j],
\]

它回答的是“gold 是否仍可由候选集合表示”。真正决定 speculative verification 是否会到达位置 \(i\) 的却是：

\[
R_i=\prod_{j<i}
\mathbf 1[\hat y_j=y_j],
\qquad
\hat y_j=C_{j,\arg\max_k s_{j,k}}.
\]

若 gold 在所有位置都位于 K16，但当前 GCLS 在位置 3 已经选错，那么位置 4–15 对本轮 accepted length 全部不可达。Candidate-D-PACE 的 smoothing 又故意避免权重快速归零，所以这些 suffix 仍可能得到显著直接 CE 梯度。

这会重新产生旧 SPH 已暴露过的失败模式：

> 完整 path 或 suffix accuracy 在变好，但 first mismatch 与 EAL 几乎不动。

对于 GCLS，正确的主支持集应由当前 selector 的实际 greedy policy 决定，并包含当前第一处 breaker token，但不直接监督 breaker 之后的 suffix。

### P0-8：D-PACE 没有表达额外 head 的两项不对称职责

GCLS 不是从零训练的 drafter，而是 DFlash 上的 selector/corrector。它的两个主要动作并不对称：

1. **DFlash top-1 已正确时：不要覆盖错；**
2. **DFlash 首次错误且 gold-in-K 时：跨过 base margin，把 gold 翻到 top-1。**

Candidate-D-PACE 只根据当前 gold probability 分配位置权重，并不知道某个位置是 base-correct、base-first-miss，还是已经位于不可达 suffix。甚至在 hard first-miss 上，gold rank 较后、\(q_i^K\) 较低时，其权重未必最大；反而已正确、高置信的位置可能继续获得大量梯度。

因此下一版需要显式加入：

- **Head-AUF**：用当前 selector 的实际 first mismatch 决定主 CE support；
- **first-miss repair margin**：专门训练跨过 DFlash top-1；
- **base-correct protection**：惩罚 harmful override；
- 少量 all-position/coverage CE：防止硬 reach 在训练早期过于稀疏。

Candidate-D-PACE 应保留为已有强基线、warm-up 和 ablation，而不再作为唯一默认主损失。

### P0-9：checkpoint、calibration 和 test 协议尚未完全隔离

当前 147-prompt `validation_select` 同时承担 checkpoint selection 和 KEEP_BASE margin calibration，而且已被多轮观察。

必须建立：

- train；
- validation-select；
- calibration；
- formal-test。

Raw GCLS 是主方法结果；KEEP_BASE 是独立 calibration 后的系统结果。

## 3.2 P1：决定最终性能与论文竞争力

### P1-1：缺少 dense target distribution

Hard gold rank 只告诉模型第一名是谁，不告诉它：

- target 对 rank-2 是 near-tie 还是强烈反对；
- chat 中是否有多个近似合理候选；
- DFlash margin 应该覆盖多少。

DSpark 直接使用 target/draft distribution TV，并训练 acceptance confidence；D-PACE 附录还推导了 KL 版本 D-PAKL。

### P1-2：DFlash 表示可能是主瓶颈

DFlare 明确指出 DFlash 将少数 target layers 融成所有 draft layers 共用的表示，形成 conditioning bottleneck；其 layer-wise target fusion 配合更深 drafter和更多数据获得稳定提升。

这支持两个方向：

1. GCLS 直接读取多个 target-layer context tokens；
2. 解冻 DFlash 的 target-feature fusion，使 hidden 主动适配 candidate selection。

### P1-3：canonical offline 与真实 rollout state 不同

Draft-OPD、AdaFlash 等工作都强调 offline target trajectory 与 drafter inference state 的偏移。GCLS 会改变 accepted length，因此也会改变下一轮 anchor 分布。

需要在 architecture 冻结后加入 on-policy replay，而不是只增加随机 canonical anchors。

### P1-4：当前方法没有系统级动态长度策略

DSpark、BlockPilot、AdaFlash 等工作表明：固定验证所有 suffix 在高并发时可能浪费 target capacity；不同样本、domain 和位置的最优 block/verification length 不同。

这不是第一阶段 GCLS 的核心 novelty，但最终系统应加入独立的 length/confidence head，避免较小 EAL 增益被 verification waste 抵消。

### P1-5：在线 overhead 可能吃掉全部增益

以当前开发数值近似：

\[
\tau_0\approx5.113+1=6.113,
\qquad \Delta\tau\approx0.242.
\]

若其他成本不变，GCLS 超过 DFlash 的必要条件约为：

\[
\frac{T_{head}}{T_{round}}<\frac{\Delta\tau}{\tau_0}\approx3.97\%.
\]

因此当前 head 若占一轮 5%–10%，算法上变好、TPS 仍可能下降。建议工程目标：

- **优选目标：head < 2% round latency**；
- **当前硬边界：若 gain 不增加，head > 4% 基本 no-go**。

Domino 论文报告约 2.8% total-round overhead，同时 acceptance length 提升约 16.6%，这才是实际竞争门槛。

---

# 4. 推荐的 GCLS-v2 架构

## 4.1 设计目标

GCLS-v2 应满足：

1. 保持 single-chain target verification；
2. 没有 token-by-token neural recurrence；
3. 每个候选分数真实依赖完整 lattice；
4. 明确建模 candidate-specific compatibility；
5. 能保留多个 continuation modes；
6. epoch 0 严格恢复 DFlash；
7. 可裁剪、蒸馏和融合部署；
8. local/causal/global 可以只改 mask 或 context source，保持 matched control。

## 4.2 输入特征升级

对每个 candidate node `(i,k)`，建议输入：

### A. 当前 DFlash 特征

- `LN(h_i)`；
- raw full-vocab log probability；
- top-K conditional log probability；
- `top1 - candidate` margin；
- retained mass；
- conditional entropy；
- rank；
- position；
- anchor embedding。

### B. 候选 lexical 特征

- frozen target input embedding row；
- 若模型 input embedding 与 LM head untied，再加入 frozen LM-head row 的低维投影；
- 不使用巨大的随机 trainable vocabulary table。

### C. 显式候选匹配特征

```text
q_i       = W_h LN(h_i)
k_i,k     = W_e LN(E[c_i,k])
interaction = [q_i, k_i,k, q_i * k_i,k,
               dot(q_i,k_i,k), scalar_features]
node_i,k  = W_node(interaction) + pos_i + rank_k
```

### D. Target context bridge

若 collector 或在线 runtime 已有 DFlash 所使用的多个 target-layer features，增加 2–8 个 context tokens：

\[
t_m=\sum_{\ell}\beta_{m,\ell}W_{m,\ell}h_T^{(\ell)}.
\]

它们不包含未来 target token，也不需要额外 target forward。它们用于绕过 DFlash hidden 的信息压缩。

## 4.3 三阶段固定深度网络

### Stage 1：Local Candidate Competition

每个位置内部先对 K 个候选做 1–2 层 set attention：

\[
Z_i^{local}=\mathrm{LocalSetEncoder}(Z_{i,1:K}).
\]

目的：明确学习当前位置的 candidate relative ranking，而不是一开始就被 240 个节点淹没。

### Stage 2：Latent Global Mode Extraction

引入 `R=4` 或 `R=8` 个 learned mode slots：

\[
M_{1:R}=\mathrm{CrossAttn}(M_{1:R},Z_{1:L,1:K}).
\]

每个 slot 可以吸收一种 continuation mode，例如不同短语、代码结构、数学格式或 identifier pattern。

### Stage 3：Mode-to-Candidate Feedback

候选节点再读取 mode slots：

\[
Z_{i,k}^{global}=\mathrm{CrossAttn}(Z_{i,k}^{local},M_{1:R}).
\]

这样显式形成：

\[
\text{candidate}\rightarrow\text{mode}\rightarrow\text{candidate}.
\]

可以再加一层轻量 global/full 或 axial block，但不要一开始堆很多层。

## 4.4 输出层

推荐：

\[
\Delta_{i,k}=w_o^\top Z_{i,k}^{global}+u_i^\top v_{i,k},
\]

其中第二项是低秩 lexical compatibility：

\[
u_i=W_u[h_i;g_i],\qquad v_{i,k}=W_vE(c_{i,k}).
\]

最终：

\[
s_{i,k}=\alpha_i\log p_D(c_{i,k})+\Delta_{i,k}.
\]

初始化要求：

- `w_o=0`；
- lexical output scale=0；
- `alpha_i=1`；
- epoch 0 每个位置 argmax 与 DFlash 100% 相同。

## 4.5 KEEP_BASE 的正确位置

主模型仍输出 raw candidate scores，不把 threshold 作为核心结构。

另外训练一个 block/position utility 或 uncertainty head：

\[
\widehat{G}=\widehat{U}(y^{global})-\widehat{U}(y^{base}).
\]

只有在独立 calibration set 上冻结阈值后才做：

```text
if predicted_gain > threshold:
    use GCLS
else:
    keep DFlash
```

报告 raw 和 selective 两套结果。

## 4.6 推荐的容量分层

| 层级 | 目的 | 建议规模 | 建议结构 |
|---|---|---:|---|
| GCLS-v1 | proof of signal | 0.43M | 当前 d64 axial |
| GCLS-v2-small | 部署学生 | 2M–5M | d96/128，local + 4 slots + feedback |
| GCLS-v2-medium | 主性能模型 | 5M–15M | d128/192，2 rounds，rank128/256 lexical |
| Feature-ceiling teacher | 判断信息上限 | 20M–50M | 3–4 full global layers或更强 slot model |

不要为了“轻量”强行限制在 1M 以下。对于 4B target，5M–15M 仍很小；关键是实际 latency，不是参数数字本身。

## 4.7 参考伪代码

```python
class GCLSv2(nn.Module):
    def forward(
        self,
        hidden,              # [B, L, H]
        candidate_ids,       # [B, L, K]
        candidate_logp,      # [B, L, K]
        scalar_features,     # [B, L, K, F]
        anchor_ids,          # [B]
        target_context=None, # [B, C, Ht], optional
    ):
        h = self.hidden_norm(hidden)
        e = self.embed_norm(self.frozen_embed(candidate_ids))

        q = self.hidden_proj(h).unsqueeze(2)       # [B,L,1,Dc]
        k = self.token_proj(e)                     # [B,L,K,Dc]
        q_expand = q.expand_as(k)
        dot = (q_expand * k).sum(-1, keepdim=True) / math.sqrt(k.size(-1))

        node = torch.cat([
            q_expand,
            k,
            q_expand * k,
            dot,
            scalar_features,
        ], dim=-1)
        node = self.node_proj(node)
        node = node + self.pos_emb + self.rank_emb

        # Candidate competition within each position
        node = self.local_encoder(node.reshape(B * L, K, -1))
        node = node.reshape(B, L, K, -1)

        flat = node.reshape(B, L * K, -1)
        slots = self.mode_slots.expand(B, -1, -1)
        slots = self.slot_reads_lattice(slots, flat)
        if target_context is not None:
            slots = self.slot_reads_target(slots, target_context)

        flat = self.candidate_reads_slots(flat, slots)
        node = flat.reshape(B, L, K, -1)

        delta_direct = self.zero_init_score(node).squeeze(-1)
        global_pos = node.mean(dim=2)
        lexical_query = self.lexical_query(torch.cat([h, global_pos], dim=-1))
        lexical_key = self.lexical_key(e)
        delta_lex = (lexical_query.unsqueeze(2) * lexical_key).sum(-1)
        delta_lex = self.zero_init_lex_scale * delta_lex

        alpha = 1.0 + self.alpha_range * torch.tanh(
            self.zero_init_alpha(global_pos)
        )
        scores = alpha.unsqueeze(-1) * candidate_logp + delta_direct + delta_lex
        return scores
```

---

# 5. 训练目标升级：从 Candidate-D-PACE 转向 head-specific reach objective

## 5.1 先把四种 loss 的角色分清楚

下一版实现中必须严格区分：

| 名称 | 训练对象 | 概率/支持 | 正确角色 |
|---|---|---|---|
| Original D-PACE | 完整 draft backbone | full-vocabulary draft probability | 训练更强 DFlash/backbone |
| Candidate-D-PACE | 额外 GCLS head | top-K 条件分类概率 + coverage censor | 现有 baseline、warm-up、ablation |
| Head-AUF | 额外 GCLS head | 当前 selector 实际可达 prefix | 推荐主 selector objective |
| Reach-D-PACE | 额外 GCLS head | Head-AUF support 内的平滑动态权重 | 组合 ablation |

因此不能继续把“使用官方 D-PACE 公式”直接等同于“这个 loss 对额外 head 也是理论正确的”。当前实现是 **公式一致、任务语义适配**。

已有 100K 结果无需作废：它们仍然可靠地说明，在同一 Candidate-D-PACE 下，global scope 比 matched local 更好。需要修正的是后续训练路线和论文表述，而不是历史实验本身。

## 5.2 为什么 Candidate-D-PACE 不宜继续作为唯一主损失

当前定义：

\[
q_i^K=\operatorname{softmax}_{k\in C_i}(s_{i,k})_{r_i}.
\]

它有四个问题：

1. **Support mismatch**：top-K conditional probability 不是原始 D-PACE 的 full-vocabulary draft probability；
2. **Reach mismatch**：active mask 基于 gold-in-K coverage，而非当前 GCLS greedy policy 是否仍正确；
3. **Suffix gradient**：D-PACE smoothing 使当前 first mismatch 后的不可达位置仍有直接 CE；
4. **Action mismatch**：它没有显式建模“修复 base first-miss”与“保护 base-correct”这两个不对称动作。

因此 Candidate-D-PACE 更适合：

- 与现有结果保持可比的 baseline；
- 训练前 10%–30% 的稳定 warm-up；
- 与 Head-AUF/Reach-D-PACE 做因果消融；
- 不应作为唯一默认 final objective。

## 5.3 推荐主损失：Head-AUF

令：

- \(r_i\)：gold 在候选集中的 rank；gold 不在 K 内时记为无效；
- \(g_i=\mathbf 1[y_i\in C_i]\)；
- \(\hat r_i=\arg\max_k s_{i,k}\)；
- \(c_i=\mathbf 1[g_i=1\land \hat r_i=r_i]\)。

位置 \(i\) 在当前 policy 下是否可达：

\[
R_1=1,
\qquad
R_i=\prod_{j<i}c_j.
\]

最终主 CE mask：

\[
M_i=R_i g_i.
\]

Head-AUF loss：

\[
\mathcal L_{\mathrm{Head\mbox{-}AUF}}
=\frac1B\sum_{b=1}^{B}\sum_{i=1}^{L}
M_{b,i}\left[-\log q_{b,i}^{K}\right].
\]

关键性质：

- \(R_i\) 只检查当前位置之前，所以 **当前第一处错误本身仍然参与训练**；
- 第一处错误之后的 direct per-position CE 为 0；
- gold 首次离开 top-K 时当前位置没有 candidate label，后续也停止；
- reach mask 必须 `detach`，不经过 argmax 反向传播。

参考实现：

```python
gold_in_k = gold_rank.ge(0)                     # [B, L]
pred_rank = scores.argmax(dim=-1)               # [B, L]
correct = gold_in_k & pred_rank.eq(gold_rank)

prev_correct = torch.cat(
    [torch.ones_like(correct[:, :1]), correct[:, :-1]],
    dim=1,
)
reach = torch.cumprod(prev_correct.float(), dim=1).bool().detach()
active = reach & gold_in_k

ce = F.cross_entropy(
    scores.float().reshape(-1, K),
    gold_rank.clamp_min(0).reshape(-1),
    reduction="none",
).reshape(B, L)

loss_head_auf = (ce * active.float()).sum() / B
```

### 为什么 Head-AUF 不会阻止模型利用未来候选

即使位置 4–15 没有自己的 direct CE，位置 1–3 的 score 仍由整个 lattice 计算：

\[
s_i=f_\theta(\mathcal L_{1:L,1:K})_i.
\]

所以早期 loss 仍可沿 attention/cross-position pathway 反向传播到未来 candidate nodes。Head-AUF 删除的是：

> suffix 作为独立预测目标的无效监督，

而不是：

> suffix 作为帮助早期选择的输入证据。

这是与 GCLS 核心假设最一致的训练语义。

## 5.4 为避免硬 AUF 过于稀疏，保留少量 coverage CE

训练初期若位置 1 很容易错，纯 Head-AUF 每个 block 可能只训练一两个位置。建议增加小权重辅助项：

\[
m_i^{K}=\prod_{j\le i}g_j,
\]

\[
\mathcal L_{\mathrm{all\mbox{-}candidate}}
=\frac1B\sum_{b,i}m_{b,i}^{K}\,CE_{b,i}.
\]

推荐初始范围：

\[
\lambda_{all}\in\{0.05,0.1,0.2\}.
\]

它只负责维持所有可表示位置的基本分类能力，不应重新成为主梯度来源。必须记录：

- Head-AUF direct gradient 占比；
- all-position auxiliary gradient 占比；
- 当前 first mismatch 后的 direct-output loss 是否严格为 0。

## 5.5 显式 first-miss repair

找到 DFlash base path 的第一处错误：

\[
t_B=\min\{i:C_{i,1}\ne y_i\}.
\]

若 \(y_{t_B}\in C_{t_B}\)，定义：

\[
\mathcal L_{repair}
=\max\left(0,
 m+s_{t_B,base}-s_{t_B,gold}
\right).
\]

这直接要求 GCLS 在最影响 EAL 的位置跨过 DFlash top-1 margin，而不是只把 gold probability 从低值提高一点。

可选地使用 detached continuation potential 加权：修复该位置后，后面仍连续 gold-in-K 的长度越长，repair 权重越大。但这一项必须单独做 ablation，不能未经验证就写成主理论贡献。

建议报告：

- rank 2；
- rank 3–4；
- rank 5–8；
- rank 9–16；
- 各 bucket 的 repair/harm 和净 EAL。

## 5.6 显式 base-correct protection

当 DFlash top-1 已正确且当前位置在当前 head reach 内时：

\[
\mathcal L_{protect}
=\max\left(0,
 m+\max_{k\ne r_i}s_{i,k}-s_{i,r_i}
\right).
\]

其职责是减少：

> DFlash 原本正确，GCLS 因弱全局证据而错误 override。

建议将 protection 约束在当前 selector 可达、base-correct 的位置；不要对 base first-miss 之后所有 suffix 强行保护，否则会压制真正需要的 correction。

KEEP_BASE 仍然作为独立 calibration 后的系统策略；`L_protect` 不能被用来隐藏 raw selector 本身的退化。

## 5.7 Candidate-D-PACE 的保留方式

当前 loss 不删除，改名为：

> `candidate_dpace`

并保留三种用途：

1. **历史可比 baseline**：复现当前 100K global/local 结果；
2. **warm-up**：前 10%–30% steps 提供平滑、全位置基础信号；
3. **正式消融**：与 uniform CE、Head-AUF、Reach-D-PACE 比较。

论文中应写：

> We exactly reproduce the D-PACE weighting formula, but apply it to top-K conditional selector probabilities; we therefore refer to it as Candidate-D-PACE.

不能不加限定地写“GCLS 使用 exact D-PACE”，因为这会让读者误以为其概率语义与 full-vocabulary drafter 完全相同。

## 5.8 Reach-D-PACE：只在实际 policy support 内保留平滑权重

若希望保留 D-PACE continuation credit，可先用 Head-AUF 得到 \(M_i\)，再只在该 support 内计算平滑权重。

一种明确实现是：

\[
P_m^{reach}=
M_m\prod_{j\le m}\tilde q_j,
\qquad
\tilde q_j=(1-\alpha)\operatorname{sg}(q_j^K)+\alpha,
\]

\[
w_i^{reach}
=M_i\operatorname{sg}\left(
\sum_{m=i}^{L}P_m^{reach}
\right),
\]

\[
\mathcal L_{\mathrm{Reach\mbox{-}D\mbox{-}PACE}}
=\frac1B\sum_{b,i}w_{b,i}^{reach}CE_{b,i}.
\]

这不是原始 D-PACE，而是一个新的组合 ablation：

- Head-AUF 决定哪些位置真实可达；
- D-PACE 只在可达 support 内做平滑 credit assignment；
- breaker 之后严格为 0。

优先级上先跑纯 Head-AUF；只有它有效后，才判断平滑 continuation weighting 是否有额外价值。

## 5.9 Top-K 条件概率与 full-support probability 的消融

当前 K-way probability：

\[
q_i^K=
\frac{\exp s_{i,r_i}}{\sum_{k\in C_i}\exp s_{i,k}}
\]

适合 deterministic reranking，但会忽略 top-K 外 mass。若需要更接近原始 D-PACE 的 confidence 或未来支持 `T>0`，可定义：

\[
\ell_i'(c)=b_i(c)+\Delta_i(c),\quad c\in C_i,
\]

\[
\ell_i'(w)=b_i(w),\quad w\notin C_i,
\]

\[
Z_i'=
\sum_{c\in C_i}\exp[b_i(c)+\Delta_i(c)]
+\sum_{w\notin C_i}\exp b_i(w).
\]

对 gold-in-K：

\[
q_i^{full}=\frac{\exp[b_i(y_i)+\Delta_i(y_i)]}{Z_i'}.
\]

利用已保存的 full-vocab logsumexp 和 top-K logits，可在不重算完整词表的情况下恢复 outside mass；实现时必须使用稳定的 log-sub-exp。

注意：

- residual 为 0 时应严格恢复 DFlash full-support probability；
- residual 需 position-wise mean-centering 或其他约束，防止所有 top-K logits 同时抬高但排序不变；
- 对 `T=0` 的主 selector，K-way CE 仍可能更直接；full-support 版本主要用于 confidence、D-PACE weight 与 `T>0` proposal ablation。

## 5.10 Target candidate distillation

Collector 保存：

- target 在 `DFlash top-K ∪ target top-M` 上的 logits；
- target full-vocab logsumexp；
- target greedy token。

定义候选支持上的 target conditional distribution：

\[
p_T^C(c)=\frac{p_T(c)}{\sum_{c'\in C}p_T(c')}.
\]

训练：

\[
\mathcal L_{distill}
=\mathrm{KL}(p_T^C\|q_G^C)
\]

或 TV/L1。它补充 hard rank label 无法表达的 near-tie、chat 多模态和 margin 信息。

D-PAKL 可作为 acceptance-aware dense-distribution baseline，但不要和 Head-AUF、repair/protect 一次性全部打开；每次只回答一个因果问题。

## 5.11 推荐默认训练配方

### Stage 0：identity 与 capacity sanity

- residual/temperature identity init；
- 512–1,024 blocks memorization；
- 不使用 KEEP_BASE 选择结果。

### Stage 1：稳定 warm-up

前 10%–30% optimizer steps 使用：

\[
\mathcal L_{warm}=\mathcal L_{Candidate\mbox{-}D\mbox{-}PACE}
\]

或 uniform candidate CE。二者需要做小规模对照，不预设 Candidate-D-PACE 一定更好。

### Stage 2：推荐主训练

\[
\boxed{
\mathcal L_{head}
=\mathcal L_{Head\mbox{-}AUF}
+\lambda_{all}\mathcal L_{all\mbox{-}candidate}
+\lambda_r\mathcal L_{repair}
+\lambda_p\mathcal L_{protect}
}
\]

建议初始搜索：

- \(\lambda_{all}\in\{0.05,0.1,0.2\}\)；
- \(\lambda_r,\lambda_p\in\{0.05,0.1,0.25\}\)；
- margin \(m\in\{0.1,0.2,0.5\}\) logit units。

### Stage 3：dense supervision / representation adaptation

在 Stage 2 架构和 head objective 冻结后，再加入：

\[
+\beta\mathcal L_{distill},
\qquad
\beta\in\{0.25,0.5,1.0\}.
\]

若进入 LoRA/joint，base branch 使用原始 full-vocabulary drafter loss，selector branch继续使用上述 head-specific objective。

## 5.12 最小而有判别力的 loss 实验矩阵

固定同一 GCLS 架构、同一 DFlash、同一 100K 数据与训练预算：

| ID | Loss | 主要问题 |
|---|---|---|
| L0 | Uniform K-way CE | 最基础 candidate classification |
| L1 | Candidate-D-PACE | 复现当前 baseline |
| L2 | Head-AUF | policy reach 是否关键 |
| L3 | Head-AUF + 0.1 all-CE | 硬 reach 稳定版 |
| L4 | L3 + repair/protect | 是否更符合额外 head 的职责 |
| L5 | Reach-D-PACE | soft credit 在真实 support 内是否有增益 |
| L6 | L4 + target KL/TV | dense supervision 是否是瓶颈 |
| L7 | L4 + full-support confidence | top-K 重新归一化是否造成问题 |

至少三个 seeds。Checkpoint selection 使用 raw selector EAL，并施加 first-token/domain non-inferiority 约束；不能按 KEEP_BASE 后的结果选 checkpoint。

必须报告：

- raw EAL；
- first-token accuracy；
- first-miss repair 与 harm；
- base-correct harmful override；
- improved/harmed blocks；
- rank bucket repair；
- 首错后 direct-output loss/gradient 占比；
- global-local gap；
- chat/code/math；
- calibration 前后的 selective result。

## 5.13 优化配置

建议：

- AdamW；
- lr 从 `3e-4`、`6e-4` 两档起；
- weight decay 0 和 0.01 做小对照，默认优先 wd 0；
- warmup 0.04；
- cosine；
- clip 1.0；
- bf16 forward，softmax/logsumexp/loss 在 fp32；
- checkpoint 选择基于 raw EAL，且有 first-token/domain constraints；
- 大模型不能和小模型强行使用相同 optimizer steps；同时报告：
  - convergence-matched；
  - compute-matched。

当前 d128 结果只能说明在约 37K steps 下 underfit，不能说明大容量无效。

---

# 6. Frozen、LoRA、Joint 三档训练路线

## 6.1 Track A：Frozen DFlash + GCLS

这是最干净的主科学结果：

> 不修改 DFlash，仅利用候选 lattice，就能获得跨位置增益。

必须保留。

## 6.2 Track B：DFlash LoRA + GCLS

建议只在以下位置插 LoRA：

1. target-feature fusion projection；
2. DFlash 最后 1–2 层的 attention/MLP；
3. 可选每层 target fusion mixing weight。

Target model、target embedding、target LM head 全部冻结。

目标是判断少量 representation adaptation 是否足够。

## 6.3 Track C：Full joint DFlash + GCLS

作为性能上界，训练整个 draft backbone 和 GCLS。

采用 Domino 式 base-anchored curriculum，但明确区分两个分支的 loss 语义：

\[
\mathcal L_{joint}=
\lambda_t\mathcal L_{DFlash\mbox{-}base}^{full\mbox{-}vocab}
+(1-\lambda_t)\mathcal L_{head}.
\]

其中：

- `DFlash-base` 分支使用原始 full-vocabulary drafter objective；D-PACE 在这里才保持其原始任务语义；
- `head` 分支默认使用 Head-AUF + all-CE + repair/protect，而不是继续机械套用 Candidate-D-PACE；
- `lambda` 从 1 线性下降；
- 可保留 0.05–0.1 floor，避免 base 完全退化；
- 分别记录 base EAL、raw head EAL 和 selective EAL，避免 joint training 把 backbone 退化隐藏在 final result 中。

## 6.4 DFlare-style representation upgrade

若 joint training 明显有效，再实现：

- 每个 draft layer 有独立 target-layer mixture；
- GCLS 也读取这些 layer-wise context tokens；
- 比较 shared fusion 与 layer-wise fusion。

这样可以回答：

> GCLS 的瓶颈是否来自 DFlash narrow conditioning bottleneck？

---

# 7. Feature-ceiling 实验：现在最重要的一次实验

## 7.1 模型

使用当前完全相同的 deployable inputs，训练：

- 20M–50M 参数；
- d256 左右；
- 3–4 层 full attention 或强 latent-slot model；
- 充分训练至 validation plateau；
- 无 latency 约束；
- 为与历史结果可比，先复现 Candidate-D-PACE；随后使用 Head-AUF + repair/protect，并在架构固定后加入 target candidate distillation。

它不是论文最终方法，而是诊断工具。

## 7.2 结果解释

### 情形 A：ceiling 仍只有 `+0.3` 左右

说明：

- current candidate lattice 对 target greedy choice 的 identifiability 较低；或
- frozen DFlash hidden 没有保留足够信息。

此时停止继续堆 frozen selector，优先 LoRA/joint/target-context bridge。

### 情形 B：ceiling 达到 `+0.6`–`+1.0` 或明显接近 Domino

说明当前小架构/训练目标是主要瓶颈。下一步：

- 训练 GCLS-v2-medium；
- 从 ceiling teacher 蒸馏到 small/medium；
- 以 latency 为约束做 Pareto frontier。

### 情形 C：ceiling 对 global-local 很大，但 global-DFlash 仍不大

说明 global 信息真实，但 base protection/override calibration 是瓶颈。重点优化 dynamic base temperature、repair/protect 和 calibration。

## 7.3 建议预注册判据

以下是建议阈值，不是已有事实：

- global teacher − local teacher 的 prompt-cluster CI 下界 > 0；
- raw global − DFlash ≥ `+0.5`，或回收至少 10% 同 split K16 oracle gap；
- three-domain 不出现显著负增益；
- shuffle 后至少损失 50% 的 global-local 增益。

若全部失败，frozen global selector 主线应降级为小贡献或停止。

---

# 8. 数据管线升级

## 8.1 保留现有优点

你当前协议已经做得很好的部分：

- exact stored context；
- prompt-level split；
- target-regenerated continuation；
- file/config/script hashes；
- top-64 candidates 与 base logsumexp；
- 数据完整性审计；
- 不把 smoke 当 evidence。

这些都应继续保留。

## 8.2 新增字段

建议每个位置新增：

1. target greedy token；
2. target logits on `DFlash top-K ∪ target top-M`；
3. target full-vocab logsumexp；
4. target top-M ids/logits；
5. 可选 target last hidden；
6. DFlash selected token、first mismatch；
7. 当前 GCLS selected token、first mismatch（on-policy collection 时）；
8. exact target-layer features或其可重算索引；
9. runtime domain、context length、anchor offset。

## 8.3 数据规模决策

不要立即采满 1.42M。建议：

1. 100K 做 ceiling；
2. 若 ceiling 高，采 nested `200K / 400K / 800K / 1.42M`；
3. 保持 optimizer updates 和 prompt diversity曲线可解释；
4. anchor 数量不要替代独立 prompt 数量。

## 8.4 Hard-example replay

自然数据与 hard blocks 混合：

- natural target-on-policy blocks；
- DFlash first-miss gold-in-K blocks；
- rank 3–16 blocks；
- base-correct protection blocks；
- high-entropy chat blocks。

若改变 sampling distribution，需要记录 sampling probability，并在主分布评估时避免用过采样指标冒充真实 EAL。

## 8.5 On-policy / DAgger 阶段

架构与 loss 冻结后：

1. GCLS 在线生成；
2. 收集真实 verification boundaries；
3. 从 first-error / rejected state replay；
4. target 标注；
5. 与 canonical data 混合训练。

这对应 Draft-OPD / AdaFlash 所强调的 offline-to-inference mismatch，但不应在第一轮 architecture 诊断中提前混入。

---

# 9. 机制证明：如何真正证明模型使用了 future candidates

## 9.1 训练三个模型还不够

分别训练 local/causal/global 可能混有优化噪声。必须加入同 checkpoint intervention。

## 9.2 必做干预

### A. Inference-time mask intervention

同一个 trained global checkpoint，分别用：

- global mask；
- causal mask；
- local mask。

权重和 margin 完全不变。

### B. Candidate-only replacement

保持：

- 当前 query 位置 node；
- 所有 DFlash hidden；

只打乱其他位置：

- candidate IDs；
- candidate embeddings；
- logits/ranks/margins。

如果收益显著下降，才能证明新增 candidate-level context 有用，而不是只利用 DFlash hidden 中已有的双向信息。

### C. Hidden-only replacement

保持 candidate lattice，打乱其他位置 hidden，测量 hidden 通道贡献。

### D. Future-only / past-only replacement

分别替换 `j>i` 和 `j<i`，定位真正的新信息是否来自 future candidates。

### E. Mode-slot intervention

- 打乱 mode slots；
- 使用单 slot；
- 4/8 slots；
- 只允许 slot 读 logits，不读 token IDs；
- 只读 IDs，不读 hidden。

## 9.3 必报机制指标

- first-token accuracy；
- first-miss repair；
- first-miss harm；
- non-top1 accuracy；
- rank 2 / 3–4 / 5–8 / 9–16；
- reachable edits；
- suffix-only edits；
- global score margin；
- shuffle 后 margin drop；
- chat/code/math；
- base margin/entropy；
- context length。

Attention map 只能作为可视化，不是因果证明；必须与 replacement 干预联合。

---

# 10. 公平 baseline 设计

## 10.1 两条评测轨道

### Track 1：Released-checkpoint same-anchor

- released DFlash；
- released Domino；
- released DeLS-Spec；
- released DSpark（若 checkpoint/配置可复现）；
- frozen GCLS。

全部使用相同 stored contexts、相同 15-position horizon。

### Track 2：Controlled same-data training

在相同 target-regenerated data 上训练：

- DFlash baseline；
- DFlash + D-PACE；
- Domino；
- DeLS-style local head；
- DSpark Markov head；
- GCLS local/causal/global；
- GCLS + LoRA/joint。

这条成本更高，但才能公平隔离 architecture。

## 10.2 参数与延迟双重公平

至少报告：

- parameter-matched；
- training-FLOP-matched；
- convergence-matched；
- latency-matched。

GCLS candidate-only 不需要和 Domino full-vocab head 完全同参数；更重要的是同一硬件上的实际 latency/Pareto。

## 10.3 统一口径

分别报告：

- accepted draft tokens `A`；
- verification advance / emitted length `tau=A+1`；
- full-block acceptance；
- first-token acceptance；
- wall tokens/s；
- decode tokens/s；
- head latency；
- total round latency。

Domino 的 16-position/shift-label 需截断到与 DFlash 完全相同的 15 draft positions再比较。

## 10.4 必须正面对比的工作

- DFlash：parallel baseline；
- Domino：full-prefix causal correction；
- DSpark：Markov/RNN + dense distribution matching + confidence scheduling；
- DeLS-Spec：最直接 frozen-DFlash plug-in baseline；
- DFlare：representation/capacity baseline；
- D-PACE / Spec-AUF：训练目标 baseline；
- Draft-OPD / AdaFlash：on-policy training baseline；
- BlockPilot/DSpark/AdaFlash：adaptive length/verification baseline。

Tree 方法（DominoTree、Bastion、DDTree）不是你的同约束主 baseline，但会抬高系统性能门槛；论文应明确你坚持 single-chain、无 multi-branch verification 的约束。

## 10.5 D-PACE 在论文中的正确定位

D-PACE 不能与 GCLS 作为同层级“架构方法”简单并列。应区分：

1. **Drafter baseline**：`DFlash trained with original D-PACE`，回答不增加推理 head、只改 backbone training 能提高多少；
2. **Head loss ablation**：`Candidate-D-PACE`，回答 D-PACE-style soft position weighting 对 top-K selector 是否有效；
3. **Joint base loss**：联合训练时，原始 full-vocabulary D-PACE 用于 DFlash base branch；
4. **Complementarity test**：比较 `DFlash-D-PACE` 与 `DFlash-D-PACE + GCLS`。

最关键的四格矩阵：

| Backbone | No GCLS | + GCLS |
|---|---:|---:|
| DFlash fixed-decay | A | B |
| DFlash D-PACE | C | D |

需要同时成立：

\[
B-A>0,\qquad D-C>0.
\]

尤其是 \(D-C\)：它决定 GCLS 是与更强 drafter training 正交互补，还是只在修补原始 DFlash loss 的不足。

---

# 11. 系统实现建议

## 11.1 目标执行路径

```text
DFlash parallel backbone
    ↓
LM head / top-K extraction
    ↓
Fused candidate gather + scalar feature kernel
    ↓
GCLS local competition
    ↓
Latent-slot global exchange
    ↓
Candidate scores + optional utility
    ↓
One sequence
    ↓
Target verification
```

## 11.2 优化优先级

1. 避免 Python loop；
2. CUDA Graph 固定 `L,K,R,d`；
3. top-K 与 feature gather 尽量融合；
4. frozen embedding gather 后立即低维投影，避免保留 `[B,L,K,2560]` 大 tensor；
5. local K-attention 和 mode cross-attention 使用 fused kernel；
6. fp16/bf16 node states，score/loss fp32；
7. 预计算 frozen token projection table：
   \[
   \tilde E(v)=W_eE(v)
   \]
   只要 `W_e` 冻结；若 trainable，则周期性刷新或训练后固化；
8. batch/concurrency 下 profile，而不只测 batch 1。

## 11.3 Full attention 与 axial/slot 的策略

- Ceiling teacher：240 nodes 的 full attention，每层只有 `240²=57,600` pairs，优先表达能力；
- Deployable student：local K-attention + R slots，复杂度约 `O(LK² + RLK)`；
- 用 teacher distillation 保留能力。

## 11.4 Break-even 监控

每个 checkpoint 同时记录：

\[
\text{predicted net gain}
=\frac{\tau_{new}}{T_{base}+T_{head}}
-\frac{\tau_{base}}{T_{base}}.
\]

不要等到最后才发现 EAL gain 不足以覆盖 head latency。

## 11.5 Adaptive K / verification length

第二阶段加入：

- K ∈ {8,16,32} adaptive selection；
- verification length head；
- hardware-aware threshold。

但 token selection 与 scheduler 要分开训练和报告。尤其在 `T>0` 下，任何 scheduling decision 都必须满足 non-anticipation。

---

# 12. `T=0` 与 `T>0` 正确性

## 12.1 `T=0`

当前主线最干净：GCLS 只改变 draft proposal，target greedy verifier 最终修正，因此数学 losslessness 成立。

## 12.2 `T>0`

必须向 verifier 暴露 GCLS 实际 proposal：

\[
q_i(c)=\mathrm{softmax}(s_i)_c.
\]

不能继续用原始 DFlash probability 作为 `q`。

若只在 top-K 支持上 sample，`q` 在支持外为 0，verifier 必须用这个真实 truncated proposal 做 acceptance/residual correction。若增加 GRU/pairwise dependence，则必须提供：

\[
q_i(c\mid x_{<i}).
\]

在完成 proposal integration 和 statistical equivalence test 前，不要在主论文中声称完整支持 sampling。

---

# 13. 实验门禁与执行顺序

## Gate 0：代码与数学正确性

必须通过：

- epoch-0 DFlash identity；
- local 对其他位置 replacement 严格 invariant；
- global 对 replacement 有响应；
- no gold leakage；
- target/DFlash frozen tensors 无 gradient；
- Candidate-D-PACE 对官方 D-PACE 公式的逐元素 parity；
- 明确测试 top-K conditional probability 与 full-support probability 的差异；
- Head-AUF first-error support unit test：包含 breaker，breaker 后 direct loss 为 0；
- reach mask 无梯度，且 future candidate nodes 仍可通过 early-query loss 获得梯度；
- Reach-D-PACE 在 breaker 后权重严格为 0；
- D-PAKL/TV target distribution unit test；
- duplicate candidate / OTHER / padding tests；
- bf16 forward、fp32 loss finite gradients；
- `T>0` proposal normalization tests。

## Gate 1：完成当前已提交诊断

先读取：

- `10123109_[0-6]`：full-data seeds；
- `10123112`：same-checkpoint mask intervention；
- `10123118_[0-7]`：d64/d128 optimization budget；
- `10123133`：same-anchor Domino。

在这之前不应大改主架构并打开正式 test。

## Gate 2：Frozen feature ceiling

训练 20M–50M teacher。决定下一条路线：

- ceiling 高 → selector architecture + distillation；
- ceiling 低 → LoRA/joint/target bridge。

## Gate 3：GCLS-v2 architecture

固定 loss/data，比：

- additive node encoder；
- multiplicative/bilinear；
- 2-layer full attention；
- latent slots；
- dynamic base temperature；
- local/causal/global。

## Gate 4：Loss

架构冻结后比较：

- Uniform K-way CE；
- Candidate-D-PACE（现有 baseline）；
- Head-AUF；
- Head-AUF + 0.1 all-position coverage CE；
- 上一项 + first-miss repair / base protection；
- Reach-D-PACE；
- target KL/TV 与 D-PAKL；
- top-K conditional vs full-support confidence。

默认候选应从 Head-AUF 系列中选择；Candidate-D-PACE 不再被预设为最终主 loss。

## Gate 5：Representation adaptation

- frozen；
- LoRA；
- full joint；
- shared fusion；
- DFlare-style layer-wise fusion。

## Gate 6：Data scaling / on-policy

- 100K；
- 200/400/800K；
- 1.42M only if curve justifies；
- canonical vs on-policy replay。

## Gate 7：Formal test

配置、代码、数据、checkpoint、calibration 全冻结后：

- 至少 3 seeds；
- prompt-cluster bootstrap；
- fresh select/calibration；
- reserved formal test只运行一次；
- 不用 formal result 调 threshold。

## Gate 8：Online system

- real rollout；
- batch 1 和 concurrency；
- short/long context；
- latency breakdown；
- fused/CUDA Graph；
- actual TPS and Pareto。

---

# 14. 推荐的最小实验矩阵

| ID | Backbone | Selector | Loss | 目的 |
|---|---|---|---|---|
| A | frozen DFlash | none | — | DFlash baseline |
| B | frozen | local v1 | Candidate-D-PACE | 现有 local rerank baseline |
| C | frozen | global v1 | Candidate-D-PACE | 当前 proof of signal |
| D | frozen | global full-attn teacher | Candidate-D-PACE → Head-AUF | frozen feature ceiling |
| E | frozen | GCLS-v2 local | Head-AUF + all-CE | 新 local control |
| F | frozen | GCLS-v2 global | Head-AUF + all-CE | 架构增益 |
| G | frozen | GCLS-v2 global | + repair/protect | head-specific duty alignment |
| H | frozen | GCLS-v2 global | Reach-D-PACE | 平滑 credit ablation |
| I | frozen | GCLS-v2 global | G + target KL/TV | dense supervision |
| J | LoRA DFlash | GCLS-v2 global | original D-PACE(base) + head objective | representation adaptation |
| K | joint DFlash | GCLS-v2 global | base-anchored joint | 性能上界 |
| L | controlled joint | Domino | original recipe | causal baseline |
| M | frozen | DeLS-style head | NTP | direct plug-in baseline |
| N | controlled joint | DSpark Markov | CE+TV | semi-AR baseline |
| O | DFlash-D-PACE | none / GCLS | original base + head objective | GCLS 与强 drafter training 的互补性 |

主论文不需要把所有组合放主表，但开发阶段必须用它们定位瓶颈。

---

# 15. 代码组织建议

```text
src/gcls_v2/
├── config.py
├── feature_builder.py
├── candidate_compatibility.py
├── local_set_encoder.py
├── latent_mode_mixer.py
├── selector.py
├── losses/
│   ├── candidate_dpace.py
│   ├── head_auf.py
│   ├── reach_dpace.py
│   ├── full_support.py
│   ├── dpakl.py
│   ├── repair_protect.py
│   └── counterfactual.py
├── calibration.py
├── interventions.py
├── rollout.py
└── kernels/
    ├── candidate_gather.py
    └── fused_selector.py

tests/
├── test_identity.py
├── test_scope_invariance.py
├── test_context_replacement.py
├── test_candidate_dpace_parity.py
├── test_head_auf_support.py
├── test_reach_dpace_support.py
├── test_full_support_probability.py
├── test_dpakl.py
├── test_no_gold_leakage.py
├── test_sampling_correctness.py
└── test_kernel_reference.py

scripts/
├── train_feature_ceiling.py
├── train_gcls_v2.py
├── train_joint.py
├── evaluate_same_anchor.py
├── evaluate_interventions.py
├── evaluate_online.py
└── aggregate_formal.py
```

每个产物强制写入：

- evidence tier；
- project commit；
- dirty status；
- script/config/data/checkpoint hashes；
- train/select/calibration/test identifiers；
- allowed claim；
- superseded-by。

当前 `results_registry.json` 仍停留在 2026-07-22，而 experiment log 已到 2026-08-03，应立即统一 authoritative registry。

---

# 16. ICLR 论文应该如何定位

## 16.1 最稳妥的核心 claim

> Parallel block drafters expose a rich candidate lattice but collapse it through independent per-position selection. We show that other-position candidates carry measurable conditional information about early target-consistent choices, and introduce a fixed-depth global candidate selector that extracts this information without autoregressive neural rollout or multi-branch target verification.

中文：

> 并行 drafter 已经产生了丰富的候选格子，但逐位置 top-1 丢弃了其中的跨位置结构。我们证明其他位置候选对早期 target-consistent 决策包含额外条件信息，并用固定深度全局选择器提取该信息，同时保持单链验证、无自回归神经 rollout。

## 16.2 不要声称

- “首次提出 bidirectional draft head”；DFlash/SpecFormer 等已经使用双向结构。
- “解决 Domino 首错后的 state pollution”；首错后的 suffix 不影响本轮 accepted length。
- “首次 acceptance-aware loss”；D-PACE、Spec-AUF、VSD 等已经覆盖。
- “global path optimal”；当前 direct selector 不是结构化 Bayes-optimal path decoder。
- “超过 Domino”；在 same-anchor 和在线结果完成前不能写。

## 16.3 形成 ICLR 级贡献所需的四块证据

1. **现象**：global candidate information 的条件增量价值；
2. **方法**：candidate-specific、multi-mode、固定深度 selector；
3. **机制**：replacement/intervention 后增益消失；
4. **系统**：EAL gain 转化为真实 TPS，并优于强 causal/local baseline。

---

# 17. Kill / Pivot 标准

## 17.1 停止 frozen GCLS 主线

满足以下条件后仍无明显收益：

- 20M–50M ceiling model 充分收敛；
- target distribution supervision；
- 100K+ diverse prompts；
- matched local/global；
- global-local CI 包含 0 或小于系统可用阈值；
- candidate-only shuffle 几乎不影响结果。

这说明 future lattice 的 deployable selectability 很低。

## 17.2 转向 joint representation

若 frozen ceiling 低，但 LoRA/joint 明显提高：

> 核心故事应改为“训练 parallel drafter 显式产生可全局选择的 candidate lattice”，而不是纯 plug-in selector。

## 17.3 转向 hybrid causal/global

若 global-only 明显低于 Domino，但 `Domino + global evidence` 有稳定增益：

> 将论文定位为 global evidence 对 causal correction 的互补机制；但需要重新评估并行性和 novelty。

## 17.4 停止系统化投入

若 formal online 中：

- EAL 提升不足；
- head latency超过 break-even；
- fused 后仍不提高 TPS；
- strong baseline 明显更优；

则不应再靠扩大参数掩盖系统 no-go。

---

# 18. 立即执行的优先级清单

## P0：不改主架构，先拿到关键答案

1. 完成并审计 `10123109 / 10123112 / 10123118 / 10123133`。
2. 更新唯一 authoritative results registry。
3. 新建独立 calibration split；保留仍未观察的 reserved test。
4. 实现 same-checkpoint candidate-only/future-only replacement。

## P1：判断信息上限

5. 实现 full-attention 20M–50M feature-ceiling teacher。
6. 在现有 100K 上训练到真正收敛。
7. 记录 global-local、global-DFlash、rank repair、shuffle drop。

## P2：实现 GCLS-v2

8. 显式 multiplicative/bilinear compatibility。
9. local candidate competition。
10. 4/8 latent mode slots。
11. mode-to-candidate feedback。
12. dynamic base temperature，identity init。
13. 从 teacher 蒸馏到 2M–15M student。

## P3：训练目标

14. 将现有 loss/日志统一重命名为 Candidate-D-PACE，保留历史兼容字段。
15. 实现并单测 Head-AUF：包含 breaker，breaker 后 direct loss 为 0。
16. 实现 `Head-AUF + 0.1 all-CE + repair/protect`，作为下一版默认候选。
17. 实现 Reach-D-PACE 与 top-K/full-support probability 消融。
18. collector 增加 target candidate logits/logsumexp，再比较 D-PAKL/TV。

## P4：表示适配

19. Frozen / LoRA / full joint 三档。
20. Domino-style base curriculum。
21. DFlare-style layer-wise target fusion。

## P5：系统与正式论文

22. on-policy replay。
23. fused kernels/CUDA Graph。
24. online TPS、concurrency、context length。
25. 第二个模型规模与 K/block-size ablation。
26. 配置冻结后只运行一次 formal test。

---

# 19. 最终判断

当前实现不是最优架构，也不存在任何结构能够“保证”学会全局候选一致性，因为可学习性首先取决于 frozen DFlash features 中是否保留了 target choice 所需的信息。

但是，你的 100K 结果已经排除了“全局 candidate lattice 完全没有可用信息”这一悲观解释。global-local 的 raw 增益、三域一致性、non-top1 accuracy 和 first-miss repair 都说明方向值得继续。

现在最需要避免的错误是：

> 在不知道 feature ceiling 的情况下，继续用同一 0.43M 架构单纯扩数据，或同时堆入更多 decoder/loss，使结果失去可解释性。

最推荐的主路线是：

> **用大容量 teacher 测 frozen feature ceiling；若 ceiling 高，推进“显式候选匹配 + local competition + latent global modes + mode feedback”的 GCLS-v2 并蒸馏；若 ceiling 低，转向 target-context bridge 和 DFlash LoRA/joint training。训练上将 Candidate-D-PACE 降为历史基线/warm-up，以 Head-AUF + 少量 coverage CE + repair/protect 作为 head-specific 主候选，再独立比较 Reach-D-PACE、target distillation 与 D-PAKL/TV。**

这条路线既保留你最有价值的核心创新——未来 candidate evidence 在 token commitment 之前帮助早期选择——也修正了把 full-vocabulary drafter loss 直接迁移到 top-K residual head 的语义错位，并正面吸收 Domino、DSpark、DFlare、D-PACE、Spec-AUF、Draft-OPD 和最新 adaptive block drafting 工作带来的实现经验。

---

# 参考资料

## 项目内部材料

1. `experiment_log.md`，截至 2026-08-03 的 authoritative experiment log。
2. `global_head_source_audit_and_v1_design.md`，DFlash/Domino/DSpark/DeLS/D-PACE 源码审计与 GCLS-v1 设计。
3. `method.md`，单链、state pollution、SPH 与全局候选选择的问题定义。
4. `phase3_failure_analysis.md`，旧 SPH/no-mixer 失败复盘。
5. `results_registry.json`，旧版证据注册表；需要与最新日志同步。

## 公开论文与官方代码

1. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
2. [Domino: Decoupling Causal Modeling from Autoregressive Drafting](https://arxiv.org/abs/2605.29707)
3. [Domino official repository](https://github.com/jianuo-huang/Domino)
4. [D-PACE: Dynamic Position-Aware Cross-Entropy](https://arxiv.org/abs/2605.18810)
5. [DSpark: Confidence-Scheduled Speculative Decoding](https://arxiv.org/abs/2607.05147)
6. [DeepSpec official repository](https://github.com/deepseek-ai/DeepSpec)
7. [DeLS-Spec: Decoupled Long-Short Contexts](https://arxiv.org/abs/2607.07409)
8. [DFlare: Scaling Up Draft Capacity](https://arxiv.org/abs/2606.02091)
9. [Draft-OPD: On-Policy Distillation for Speculative Draft Models](https://arxiv.org/abs/2605.29343)
10. [Spec-AUF: Accept-Until-Fail Training](https://arxiv.org/abs/2607.01893)
11. [AdaFlash: Adaptive Speculative Decoding via On-Policy Distilled Diffusion Drafters](https://arxiv.org/abs/2607.19223)
12. [BlockPilot: Instance-Adaptive Policy Learning](https://arxiv.org/abs/2606.31315)
13. [WhiFlash: Token-Level Cross-Paradigm Routing](https://arxiv.org/abs/2606.07710)
14. [DominoTree: Conditional Tree-Structured Drafting](https://arxiv.org/abs/2607.08642)

