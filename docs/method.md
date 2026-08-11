# 单链重审：Domino 的“误差累积”、双向 head 与可投稿方向

> 日期：2026-07-22
> 状态：当前方法定义；早期 ReFlash / PlanDomino 文件已从工作区删除。
> 约束：只输出一条 draft sequence；不做 draft tree、tree attention 或多分支 target verification；新增开销必须显著小于 Domino 的顺序 GRU correction。

## 0. 结论先行

用户对 Domino **执行时序**的描述是正确的：Domino 必须先用 DFlash hidden 和 GRU 自回归地产生整个 draft block，随后 target 才一次验证；所以 GRU 生成第 $i$ 个 token 时，确实不知道前 $i-1$ 个 token 是否会被 target 接受。若早先 token 错了，之后的 GRU state 也确实会被错误 token 污染。

但这里必须区分两个命题：

1. **无条件的后部 token 准确率会因错误 rollout 下降。**正确。
2. **首错后的 state 污染会让本轮接受长度进一步下降。**错误。

原因不是 Domino 在生成时知道了验证结果，而是 longest-prefix verifier 的计分规则：一旦第一个错误已经出现，本轮接受长度已经确定；此后 suffix 再差也不能让它“再少接受”。真正造成第 $i$ 位贡献概率变小的是前缀生存概率的乘积，这对任何单链 speculative decoding 都存在，双向 head 也不能绕过。

因此：

- 原始“让首错后的 Domino suffix 不受污染”不是有效主目标；
- 简单在 Domino 上叠一个 bidirectional module，开销和新颖性都不合格；
- 值得做的新问题是：**在 target 验证之前，用整块候选的全局一致性反过来改善早期 token 的选择，直接优化单链的期望接受前缀。**

本文推荐的 conditional-go 方向是：

> 用一个并行、低秩、全局归一化的 structured head 取代 Domino GRU，在 DFlash top-$K$ 候选上一次性打出相邻转移分数；再通过 acceptance-aligned dynamic programming 选择唯一一条序列。神经计算完全并行，target 仍只验证一条普通链。

工作名暂记为 **Survival Path Head (SPH)**。它不是 PlanDomino，也不是 tree verification。

---

## 1. “前一个 50%，后一个肯定更低”到底哪里对、哪里不对

令 $Y_i$ 为 draft 的第 $i$ 个 token，$C_i$ 表示该 token 在 target 验证时匹配，draft 接受数为

\[
A=\sum_{i=1}^{L}\prod_{j=1}^{i}\mathbf 1[C_j].
\]

于是

\[
\Pr(A\ge i)
=\Pr(C_1,\ldots,C_i)
=\prod_{j=1}^{i}\Pr(C_j\mid C_{<j}),
\]

\[
\mathbb E[A]
=\sum_{i=1}^{L}\Pr(A\ge i).
\]

假设：

- 第 1 个 token 正确概率是 $0.5$；
- 若第 1 个正确，第 2 个正确概率是 $0.8$；
- 若第 1 个错误，因为 GRU state 污染，第 2 个边际正确概率只有 $0.1$。

那么第 2 个 token 对接受长度的贡献概率是

\[
\Pr(A\ge2)=0.5\times0.8=0.4.
\]

现在假设有一个 oracle 完全消除错误分支上的 state 污染，把 $0.1$ 提升到 $0.8$。第 2 位的**无条件 token 准确率**会显著提高，但

\[
\Pr(A\ge2)=0.5\times0.8=0.4
\]

完全不变，因为第 1 位错误的那一半轮次已经在第 1 位停止。

所以用户说的“第 2 位要建立在前面 50% 的基础上，贡献概率更低”完全正确；需要纠正的只是归因：

- $0.5$ 是第 2 位的 **reach/survival bottleneck**；
- 错误分支上的 GRU 污染不是额外的 acceptance bottleneck；
- 能提高接受长度的是提高 $0.5$，或提高正确分支上的 $0.8$，而不是修复已经失败分支上的 $0.1$。

### 1.1 一个更强的点态结论

若两个 drafter 在所有“此前 token 全部正确”的 history 上有相同的下一 token 分布，而只在已经发生错误后的 history 上不同，那么它们的本轮接受长度分布完全相同。

这是逐样本成立的：令首错位置为 $k$，则 $A=k-1$，所有 $Y_{>k}$ 都不再出现在 $A$ 的值里。

这也解释了 Domino 论文 §4.2 为什么 teacher forcing 比 self-generated-prefix training 更好：位置 $i$ 能贡献接受长度的训练事件，恰好就是 clean-prefix regime。论文不是声称推理时 GRU 能看到 target 接受结果，而是在对齐真正有 reward 的条件分布。

---

## 2. 双向 head 能做什么，不能做什么

### 2.1 不能做的事

在 target forward 之前，任何只使用 draft 信息的 head 都不可能知道某个 token **实际上**是否被 target 接受。要得到这个事实，只能：

- 提前调用 target；
- target 与 drafting 交替执行；
- 或验证多个分支以在首错后保留另一条路径。

前两种增加 target/串行成本，第三种就是用户明确排除的 tree/multi-path 模式。因此在“一次 draft、一次单链 target verification”的限制下，无法真正消除 survival product，也无法在首错后继续累计接受长度。

### 2.2 能做的事

双向或全局决策可以在**尚未提交第一个 token 前**利用 suffix coherence 改善更早的选择。例如并行 marginals 同时偏好两个模式：

- `of course`
- `no problem`

逐位置 top-1 可能产生 `of problem`。Domino 从左到右选定 `of` 后再修正下一位；全局单链选择则可以在提交第 1 位前比较整个 `of course` 与 `no problem`。

真正有价值的目标不是“让错误 prefix 后的 token 继续准确”，而是：

> 用未来可解释性提高早期 token 的 conditional hazard，并在所有候选路径中选择预测期望接受长度最高的一条。

### 2.3 为什么一个普通 BiGRU / 双向 attention 层不够

1. DFlash 的 block hidden $H_{1:L}$ 本身已经做块内双向 attention；再加一层主要是增加容量，不天然增加新信息。
2. 双向层最后若仍逐位置独立 argmax，multi-modal collision 仍然存在。
3. 若把 hard top-1 token 双向喂入，训练和推理都要处理离散固定点或迭代 denoising，延迟会上升。
4. SpecFormer 已经明确使用 draft bidirectional attention；“加一个双向层”本身不足以形成 2027 ICLR 级别的新颖性。

因此需要把创新放在 **结构化单链分布 + acceptance-aligned inference**，而不是模块方向名上。

---

## 3. 为什么撤销 PlanDomino 与 DDTree 证据

### 3.1 PlanDomino 不再推荐

早期 PlanDomino 方案是在 Domino causal state 外再加 future-plan adapter。它有三个问题：

1. 仍保留 Domino 的逐 token GRU loop；
2. 增加模块和参数，却没有改变 locally greedy 的选路机制；
3. 与 DFlash 已有 bidirectional hidden、SpecFormer 的双向设计相比，贡献很容易被审稿人归类为 incremental feature fusion。

它可以作为一个 future-information ablation，但不应作为主方法。

### 3.2 DDTree trace 不能回答当前问题

此前 trace 来自 DDTree rollout。它不能作为当前单链 idea 的可行性证据，因为：

- rollout prefix 是 DDTree 的验证/选路策略产生的，不是纯 DFlash 或 Domino 的单链 prefix 分布；
- trace 没有 Domino 的 on-policy GRU state，无法测量 state pollution；
- tree 选择改变了哪些 anchor 会被访问，位置分布和 reach distribution 都不同；
- DDTree top-$K$ candidate recall 不能直接换算为单链 head 能实现的接受收益。

这批数据最多只能测试分析脚本和 trace schema，不能进入当前方法的 kill/go 结论，更不能作为论文主证据。此前围绕它得到的 31,373 轮统计应从主叙事撤下。

---

## 4. 推荐方法：Globally Normalized Survival Path Head

### 4.1 输入与唯一输出

DFlash 一次并行 forward 得到

\[
H_{1:L},\qquad b_i(v)=\operatorname{LMHead}(H_i)_v.
\]

每个位置仅保留 base logit 的 top-$K$ 候选集合 $\mathcal C_i$，推荐先测 $K\in\{4,8,16,32\}$。该候选集合只用于廉价的 draft-side reranking，不送给 target。

最终算法输出

\[
\hat y_{1:L}\in\mathcal C_1\times\cdots\times\mathcal C_L
\]

这一条链，并使用 DFlash 原来的 longest-prefix target verifier。没有 tree attention，没有多分支 target token，没有额外 target forward。

### 4.2 并行低秩转移能量

对 verified anchor $y_0$ 以及每对相邻候选 $u\in\mathcal C_{i-1},v\in\mathcal C_i$，定义

\[
s_i(u,v)
=b_i(v)
+\left\langle
P_LE(u)\odot g_i(H_{1:L}),
P_RE(v)
\right\rangle
+a_i(v).
\]

其中：

- $P_L,P_R\in\mathbb R^{d\times r}$，$r=16\sim64$；
- $g_i$ 是由 DFlash hidden 得到的 rank-$r$ context gate；
- $a_i(v)$ 是可选的 unary residual；
- 所有 $LK^2$ edge scores 用一次 batched einsum 并行计算。

冻结模型后，$P_LE(v)$ 和 $P_RE(v)$ 可以预计算成两个 vocabulary lookup table。以 $L=15,K=16,r=32$ 为例，真正的 pairwise 核心只有约 $15\times16^2\times32=122{,}880$ 个乘加，远小于一次 full-vocabulary LM head，也没有 15 步 GRU hidden update。

### 4.3 吸收式 `OTHER` prefix-CRF

候选集合之外仍有不可忽略的 base probability。若只在完整 top-$K$ 路径上
建立普通 CRF，suffix partition 会在 residual 为零时改变早期概率，无法同时
满足“全局归一化”和“严格恢复 DFlash”。正式模型因此把 `OTHER` 定义为
吸收式失败状态：进入 `OTHER` 表示正确前缀在该位置终止，后续位置不再计入
样本空间或 loss。

令

\[
\ell_i(u,v)=b_i(v)-\operatorname{LSE}(b_i^{full})+r_i(u,v),
\]

\[
\ell_i(u,\bot)=\log p_i^{out}
=\log\sum_{w\notin\mathcal C_i}e^{b_i(w)}
-\operatorname{LSE}(b_i^{full}),
\]

其中 $r_i$ 是 learned pairwise/unary residual，$\bot$ 表示 `OTHER`。样本空间
包含所有“候选前缀后进入 $\bot$”以及长度 $L$ 的完整候选路径。Backward
sum-product 为

\[
B_{L+1}(v)=0,
\]

\[
B_i(u)=\operatorname{LSE}\left(
\ell_i(u,\bot),
\left\{\ell_i(u,v)+B_{i+1}(v)\right\}_{v\in\mathcal C_i}
\right).
\]

诱导的精确条件概率为

\[
\log q_i(v\mid u,H)=\ell_i(u,v)+B_{i+1}(v)-B_i(u),
\]

\[
\log q_i(\bot\mid u,H)=\ell_i(u,\bot)-B_i(u).
\]

候选概率与 `OTHER` 概率在每个 predecessor state 上和为一。当 $r_i=0$
时，候选 base mass 与 outside mass 本来就和为一；由反向归纳可得所有
$B_i(u)=0$，所以模型严格恢复原始 DFlash 分布。这一不变量是正式实现的
单元测试，不是经验近似。

较早位置的条件概率显式包含 suffix partition $B_{i+1}(v)$，因此未来候选
可以在提交第一个 token 前影响选择。旧的 locally normalized Markov 分布和
不含 `OTHER` 的 candidate-only CRF 都保留为消融，但都不是 proposed model。

### 4.4 不用普通 Viterbi：直接优化接受前缀效用

普通 Viterbi 最大化

\[
\prod_i q_i(y_i\mid y_{i-1},H),
\]

即整块全部正确的概率。但 speculative decoding 的 reward 是首错前的 token 数，而不是 all-or-nothing sequence accuracy。

对固定单链 $y$，Markov surrogate 下的预测期望接受数为

\[
U(y)
=\sum_{i=1}^{L}
\prod_{j=1}^{i}q_j(y_j\mid y_{j-1},H).
\]

定义从位置 $i$ 开始、已知前一正确 token 为 $u$ 时的最优 future value：

\[
V_{L+1}(u)=0,
\]

\[
V_i(u)
=\max_{v\in\mathcal C_i}
q_i(v\mid u,H)\left(1+V_{i+1}(v)\right).
\]

该 Bellman recurrence 通过归纳可证明精确最大化 $U(y)$，复杂度为 $O(LK^2)$。回溯后只得到一条 path。

这一目标正面刻画用户指出的乘法效应：较早 token 的概率乘在所有后续收益之前；但若一个稍低概率的早期 token 能带来非常稳定的 suffix，动态规划也允许未来 coherence 反过来改变早期选择。

### 4.5 三个必须保留的 decoding 对照

使用完全相同的 edge scores 比较：

1. **local greedy**：每一步只取当前 $q_i$ 最大者，近似 DSpark Markov head；
2. **Viterbi/MAP**：最大化整块 path probability；
3. **survival DP**：最大化预测 expected accepted prefix。

只有 3 显著优于 1 和 2，才能证明 acceptance-aligned inference 是贡献，而不是 pairwise head 单纯增加了参数。

---

## 5. 训练设计

### 5.1 第一阶段必须冻结 DFlash

先冻结 DFlash backbone、embedding 和 LM head，仅训练 structured head。这样可以：

- 明确测量 head 的独立价值；
- 避免把 backbone 重新训练收益包装成 head 收益；
- 快速完成 feasibility test；
- 若失败，成本有限。

训练 block 必须来自纯 DFlash 的真实 anchor/hidden/logit，不使用 DDTree rollout。

### 5.2 候选覆盖是硬上限

若 target token $z_i\notin\mathcal C_i$，structured head 不可能在该位置恢复它。必须先报告

\[
\operatorname{Coverage}(i)
=\Pr(z_1\in\mathcal C_1,\ldots,z_i\in\mathcal C_i),
\]

并计算 oracle lattice upper bound

\[
\operatorname{EAL}_{oracle}
=\sum_i\operatorname{Coverage}(i).
\]

这个 upper bound 应在**纯 DFlash canonical anchors**上计算。若 $K=16$ 或 $32$ 时 oracle bound 仍没有明显超过 Domino，主线应直接 no-go。

### 5.3 Prefix-censored global NLL

正式训练直接使用上述 absorbing-OTHER 模型，不把 gold 注入 inference
lattice。若 gold 在前 $m$ 个位置位于候选中、在第 $m+1$ 位首次离开，观测
事件就是

\[
z_1,\ldots,z_m,\bot.
\]

loss 为这些全局诱导条件概率的负对数和；进入 $\bot$ 后的 suffix 不可达，
不再监督。若整个 block 都在 lattice 中，则使用完整候选路径概率。这样训练
事件与 longest-prefix verification 完全对齐，也不会把 top-$K$ 外的真实概率
错误地重新分配给候选。

locally normalized prefix-censored NLL 使用相同 edge scorer，作为强 control；
gold injection 只允许用于明确标注的 debugging ablation，不进入主结果。

### 5.4 Acceptance-aligned loss 不能单独作为创新点

可增加

\[
\mathcal L_{surv}
=-\sum_i\prod_{j\le i}q_j(z_j\mid z_{j-1},H),
\]

或其稳定的 log-space surrogate，使早期 transition 自动获得更多梯度。但 D-PACE 和 Variational Speculative Decoding 已经研究 acceptance-aware training；因此它只能是完整方法的一部分，不能单独声称新颖。

推荐总 loss：

\[
\mathcal L
=\mathcal L_{CRF}
+\lambda_{surv}\mathcal L_{surv}
+\lambda_{base}\mathcal L_{base-anchor}.
\]

### 5.5 校准不是可选项

survival DP 使用的是概率绝对值，不只是排序。必须在 held-out set 上报告：

- conditional NLL / Brier score；
- 预测 $U(\hat y)$ 与真实 accepted length 的 reliability curve；
- 每位置 prefix survival calibration；
- temperature scaling 前后结果。

若概率未校准，DP 可能为一个虚假的高 suffix value 牺牲第 1 位准确率。可加入保守系数

\[
V_i(u)=\max_v q_i(v\mid u)(1+\lambda V_{i+1}(v)),
\quad0\le\lambda\le1,
\]

其中 $\lambda=0$ 是 local greedy，$\lambda=1$ 是完整 survival objective；该 sweep 应作为核心消融。

---

## 6. 正确的数据与诊断实验

### 6.1 必须重新采集两类单链数据

1. **Canonical-anchor offline blocks**：沿 target greedy 序列选择相同 anchor，分别运行 DFlash、Domino 和新 head；用于严格的同输入比较。
2. **真实 rollout**：每种方法独立做完整生成；用于最终 latency、throughput 和 acceptance length。

每轮至少保存：

- anchor 的 target token offset；
- DFlash $H_i$、full base logsumexp、top-64 token/logit；
- Domino 的 sampled token、GRU state、corrected top-$K$；
- target continuation 和逐位置 match；
- backbone、LM head、head、DP、verification 的 CUDA event latency。

### 6.2 必须同时画三条曲线

对每个方法报告：

- marginal match：$\Pr(C_i)$；
- reach：$\Pr(C_{<i})$；
- conditional hazard：$\Pr(C_i\mid C_{<i})$。

只画 marginal curve 会再次混淆错误 suffix 与 acceptance-relevant branch。

### 6.3 直接验证“state pollution 存在但不产生额外 EAL 损失”

对 Domino 同一 block 做两条离线轨迹：

- on-policy GRU：喂自己生成的 token；
- teacher-forced GRU：始终喂 target token。

首错之后两条 state/logit 会快速分离，这能证明确实存在 state pollution；但在首错之前两者必须逐位相同，而接受长度也已经由首错决定。这一实验可以把争论拆成两个都可观测的事实，不再靠语言解释。

### 6.4 真正与新方法相关的 probe

需要测：

1. top-$K$ prefix coverage / oracle EAL；
2. target bigram 是否能被 low-rank edge head 从候选中区分；
3. local greedy、Viterbi、survival DP 三者首次选择不同的比例；
4. survival DP 改变第 1 位时，真实第 1 位准确率是升还是降；
5. gain 来自哪些模式：标点、固定搭配、代码语法、数学模板、开放对话；
6. $(K,r,\lambda)$ 对 EAL/latency 的 Pareto frontier。

---

## 7. 截至 2026-07-22 的新颖性边界

简单“轻量双向 head”已经不够。至少需要正面对照以下工作：

- [DFlash](https://arxiv.org/abs/2602.06036)：block hidden 已有 bidirectional attention。
- [Domino](https://arxiv.org/abs/2605.29707)：DFlash + causal GRU + low-rank residual。
- [DSpark](https://arxiv.org/abs/2607.05147)：parallel backbone + Markov/RNN sequential head + prefix-survival scheduling；其方法明确选择 autoregressive factorization，而非 globally normalized energy。
- [DeLS-Spec](https://arxiv.org/abs/2607.07409)：固定 DFlash + 独立训练的 Markov/RNN local expert。
- [SpecFormer](https://arxiv.org/abs/2511.20340)：已经组合 causal context attention 与 draft bidirectional attention。
- [DiffuSpec](https://arxiv.org/abs/2510.02358)：已经在 diffusion token lattice 上用 causal proxy 和 beam search 选一条 path。
- [Accelerating Codec-based Speech Synthesis with MTP and SD](https://arxiv.org/abs/2410.13839)：在 codec speech 中已用 top-$K$ transition matrix 和 Viterbi 选 path。
- [D-PACE](https://arxiv.org/abs/2605.18810) 与 [VSD](https://arxiv.org/abs/2602.05774)：已经把 expected acceptance / path utility 引入训练。
- [PTP](https://arxiv.org/abs/2512.21323)：从辅助随机变量角度研究单次并行的 joint token prediction。

所以可成立的论文 claim 必须收窄为下面的组合，而不能把任一单点包装成首次提出：

1. 面向 DFlash 的、target-conditioned、全局归一化的 candidate transition energy；
2. 一个对单链 longest-prefix utility **精确 Bayes-risk optimal** 的 $O(LK^2)$ decoder，而非 MAP/Viterbi 或 heuristic beam；
3. 无 sequential neural rollout、无 tree verification 的 fused GPU 实现；
4. 证明 global normalization + prefix-risk decoding 在相同 edge head 下各自贡献增益；
5. 在 matched latency/parameter/verification budget 下超过 Domino、DSpark、DeLS-Spec 与 DiffuSpec 式 path search。

这仍只是一个有明确空位的研究假设，不是已经保证新颖。投稿前必须继续做关键词、引用链和 2026 下半年新论文检索。

---

## 8. ICLR 级别的 kill/go 标准

### Phase A：不训练或极小训练的上限测试

在 canonical-anchor 数据上计算 $K=4,8,16,32$ 的 oracle lattice EAL。

- 若 $K\le16$ 的 oracle EAL 不能显著超过 Domino：**no-go**；候选空间没有足够信息。
- 若只有 $K=64+$ 才有空间：大概率系统开销不划算，除非有更好的 adaptive shortlist。

### Phase B：冻结 DFlash 的 structured-head 测试

必须使用相同 edge scores 比较 local greedy / Viterbi / survival DP，并加入：

- DFlash top-1；
- DSpark/DeLS 风格 Markov head；
- Domino；
- 同参数 bidirectional MLP/BiGRU control。

若 survival DP 不能在 held-out canonical blocks 上稳定提高真实 EAL，或提高 suffix 却降低首 token accuracy：**no-go 或重新校准**。

### Phase C：系统测试

通过 eager correctness 后依次实现 Triton、CUDA Graph、fused top-$K$+edge+DP。报告 batch size 1 以及多并发下：

- draft backbone latency；
- full LM head latency；
- structured head + DP latency；
- target verify latency；
- end-to-end tokens/s 和 speedup。

只有接受长度收益转化成 matched-hardware 的吞吐收益，且优于强单链基线，才进入完整 ICLR 实验。论文需要至少两个 target scale、math/code/chat 三类任务、$T=0$ 与正确实现的 $T=1$ 讨论。

---

## 9. 当前原型

张量级实现：

- `src/sph/survival_path_head.py`
- `tests/test_survival_path_head.py`

已实现：

- low-rank context-conditioned $K\times K$ edge scores；
- 保留 full-vocabulary outside mass 的 locally normalized control；
- absorbing-OTHER prefix-CRF backward sum-product 与精确 log-partition；
- residual=0 严格恢复 DFlash 的全局条件概率；
- prefix-censored global NLL；
- candidate-only chain CRF ablation；
- local greedy、ordinary Viterbi、survival-DP 三种 decoder；
- brute-force 枚举验证 DP 最优性。

运行：

```bash
module load anaconda3
PYTHONPATH=src python -m unittest discover -s tests -v
```

当前结果（2026-07-22）：

```text
Ran 27 tests
OK
```

其中包括 variable-length prefix sample space 的 brute-force partition、
prefix-censored NLL、有限梯度、candidate+OTHER 归一化以及 residual=0 精确
恢复 DFlash。训练器已实现 frozen-feature 训练、独立 validation
checkpoint 选择、三 seed 汇总和 reserved-test 封存。两轮小数据
development probe（`10035142/10035188`）已经真正训练，但均未提高
held-out EAL；它们是负结果，不是 plumbing smoke。因此尚没有可声称
有效的 trained-SPH EAL 或 GPU latency 收益。当前排队的 Phase 3 tier-1
作业只是 clean-data learnability gate，不是最终模型或论文主表训练。

---

## 10. 最终建议

1. 不再推进 PlanDomino 主线，也不把 DDTree trace 当当前 idea 的证据。
2. 先用纯 DFlash canonical anchors 做 top-$K$ prefix coverage 与 oracle EAL；这是最便宜、最硬的第一道门。
3. 若存在候选空间，训练冻结 DFlash 的 globally normalized transition head，并严格比较 local greedy / Viterbi / survival DP。
4. 只有 acceptance-aligned global path selection 本身带来增益，才做 fused GPU 实现和联合训练。
5. 论文叙事应是“单链 speculative drafting 的 prefix-risk structured inference”，而不是“修复 Domino 错误状态”。

这个方向尊重用户最重要的三个约束：只输出一条序列、替换而非堆叠 Domino、额外计算远低于顺序 GRU；同时也把“前缀概率乘法下降”从一个模糊的 error accumulation 叙事，转化为可以精确定义、优化和证明的目标。
