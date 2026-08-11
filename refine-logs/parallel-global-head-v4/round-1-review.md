# PARC-16 Round-1 Fresh Method Review

**CALIBRATION: none**

## 总评

PARC 当前在线数据流严格合规，但方法核心尚未成立。最严重的问题不是 head 是否并行，而是：

1. 当前概率风险 `R=1-\prod q_i(0)` 与部署时 deterministic harm 不等价，并在零初始化 hard identity 时也可能给出接近 1 的“风险”；
2. `Top1 + soft Top16 summary` 已被 JAPD 采用，KEEP-relative 分数本质上是 residual logits 的 gauge fixing，而不是新的信息接口；
3. D-PACE、base-prefix safety、bidirectional correction、base-anchored joint training 均已有直接覆盖；
4. 联合更新 DFlash 时，`a_base` 是移动基线，模型可通过削弱 base prefix 让风险约束变得更容易，除非固定 reference base。

因此当前版本不是一个已聚焦但待补细节的 top-venue 方法，而是一个由已有组件组成、且核心约束仍数学错位的方案。

**Verdict：RETHINK**

---

## 1. Immutable Contract / Drift 审查

| 条件 | 结论 |
|---|---|
| 一次消费完整 `[B,16,*]` | PASS |
| 所有位置通过无 causal mask mixer 看全 16 位 | PASS |
| 一次同时产生 `[B,16,16]` | PASS |
| 单次逐位置 argmax 得到唯一 `[B,16]` | PASS |
| Top16 仅为位置内 candidate axis | PASS |
| selected-token feedback | NONE |
| 串行 target seed/decode | NONE |
| GRU/causal rollout | NONE |
| Jacobi/iteration/第二阶段 refinement | NONE |
| beam/tree/trie/forest/multipath | NONE |
| ordinary verifier 外额外 target forward | NONE |

`x0=C[:,:,0]` 是 DFlash 的 provisional Top1，而不是 PARC 自己的 selected token，因此不构成 feedback。DFlash→PARC 是允许的单个 draft-head pipeline，不是第二轮 correction。

**Drift Warning：NONE**

任何后续建议都不应引入串行、迭代、多路径或额外在线 target inference。

---

## 2. 核心方法审查

### 2.1 Top1 + soft Top16 summary：合规，但不是新接口

PARC 的 position mixer 确实能让全部 16 个输出同时读取完整 provisional sequence；不过该信息接口并不比已有方法更强：

- JAPD 已明确使用 `candidate-local attention → soft candidate-set summary per position → global queries`。
- 旧 full-lattice GCLS/PGCF/PCLD 甚至保留了全部 `16×16` candidate nodes；PARC 将远端候选压成一个期望 embedding，信息量反而更弱。
- embedding 加权平均不是可逆的 Top16 表示；不同多峰 candidate sets 可映射到近似相同的 `m_i`。

因此，“整条 Top1 noisy sequence + soft candidate summary”可以是一个简洁实现，但不能承担 novelty，也不能声称解决了旧 selector 的信息瓶颈。它是否足够，只能由 global-local held-out falsifier 证明。

### 2.2 KEEP-relative advantage：数学自洽，但主要是重参数化

\[
A_{i0}=0,\qquad A_{ik}=Z_{ik}-Z_{i0}+d_{ik}
\]

有以下正确性质：

- `d=0` 时与 pure DFlash 排序严格等价；若并列，candidate 0 优先即可保持 identity。
- 远 rank edit 必须显式跨越真实 base margin。
- 固定 KEEP 为零不会丢失决策表达力，因为 softmax/argmax 只依赖相对分数。
- gold=KEEP 时，梯度通过压低 edit advantage；gold=edit 时，梯度抬高对应 edit，校准原则上可行。

但一般 residual selector

\[
s_{ik}=Z_{ik}+\Delta_{ik}
\]

的决策只依赖

\[
(Z_{ik}-Z_{i0})+(\Delta_{ik}-\Delta_{i0}),
\]

故 PARC 的 `d_ik` 只是把 `\Delta_{ik}-\Delta_{i0}` 固定到一个 KEEP gauge。旧 zero-readout residual head 已有相同 zero identity；“跨越 raw logit gap”也不是新能力。

它是好的工程语义和可能有用的 inductive bias，但单独不构成方法级贡献。

### 2.3 当前 harm constraint 是 blocking mathematical error

部署时相对 base 的 deterministic harm 是

\[
H_b=
\mathbf 1\!\left[
\max_{i<a_b,k>0}(A_{bik}-A_{bi0})>0
\right].
\]

当前方案却约束

\[
R_b=1-\prod_{i<a_b}q_{bi}(0).
\]

二者严重错位：

- hard output 全部 KEEP、实际 `H_b=0` 时，若六个 protected positions 各有 `q_i(0)=0.7`，则 `R_b=1-0.7^6=0.882`。
- 一个 edit 刚刚超过 KEEP 时，hard harm 已发生，但 `q_i(0)` 可以接近 `0.5`。因此 `E[R]\le1%` 最多给出约 `E[H]\le2%` 的粗界，而不是 1%。
- 零初始化虽然 deterministic identity、harm 为零，但 restricted-Top16 softmax 通常并不极端尖锐，故 primal-dual 一开始会把大量安全样本判作高风险，驱动所有 edit logits极度负化，使编辑学习饱和或退化。

这不是轻微 calibration 问题，而是当前主贡献的约束对象错误。

最小数学修订应基于 protected-prefix maximum edit margin：

\[
M_b=\max_{i<a_b,k>0}(A_{bik}-A_{bi0}),\qquad
H_b=\mathbf1[M_b>0].
\]

一个逐 block 的光滑上界可写为

\[
\bar H_b=
\frac{\operatorname{softplus}(M_b/\tau)}{\log 2}.
\]

当 `M_b>0` 时，`\bar H_b>1`，所以 `H_b\le\bar H_b`。`max` 几乎处处可微；若用 log-sum-exp 替代，必须保留其对 max 的上界方向。然后约束与绑定评估使用同一个 block/prompt averaging unit：

\[
\operatorname{Mean}_b(\bar H_b)\le0.01.
\]

仍需明确 primal-dual update、dual LR、batch/EMA 估计、是否 cap `λ`，并保留真正 deterministic held-out harm gate；训练分布上的上界不保证 held-out safety。

### 2.4 联合训练产生移动基线退化

若 Top16 和 `a_base` 都来自正在联合更新的 DFlash，则风险约束可被 game：

- DFlash 若把原本正确的 base early prefix 弄错，`a_base` 会缩短；
- 被保护的位置随之消失；
- 当 `a_base=0` 时风险直接置零。

`L_base_DPACE` 是软保护，不能消除这个可行退化。因此“相对 base 的 constrained policy improvement”必须二选一：

1. 训练 PARC 时冻结 DFlash，使 base/reference 和 candidate-0 语义固定；或
2. 固定 step-0 DFlash 为 immutable reference，`a_ref` 永不变化；联合训练时保护 reference-correct gold action。若 live base 把该 gold token移出 Top16，该 block 的 harm surrogate必须 fail closed，而不能缩短 protected set。

否则 primal-dual constraint 没有稳定的 policy-improvement baseline。

### 2.5 更强修订：truncated gain 是正确修复，但仍偏 loss tweak

建议的

\[
U_{\rm gain}
=\sum_{t=a}^{h-1}\prod_{j=0}^{t}q_j(r_j)
\]

确实比当前 all-prefix D-PACE 更准确地表示“超越 base 后新增的 accepted tokens”。其 detached D-PACE coefficient 应为

\[
w_i^{\rm gain}
=\operatorname{sg}\!\left[
\sum_{t=\max(a,i)}^{h-1}
\prod_{j=0}^{t}\tilde q_j(r_j)
\right].
\]

与 protected-prefix max-margin risk 配对后，形成一个数学上合理的“maximize incremental accepted length subject to deterministic-harm surrogate”问题。

但 novelty 判断仍然是：

- tail truncation 是 D-PACE accepted-length surrogate 的 base-conditioned specialization；
- 本地代码已有逐位置 base-prefix ReLU safety hinge；
- 新增部分主要是 summed/fixed-weight hinge → blockwise max-margin/primal-dual constraint。

所以它仍主要是 **incremental objective revision**，不是新的序列建模接口。若能给出清楚的 surrogate/upper-bound 推导，并在多模型或至少强 matched controls 上产生大幅、稳定的 EAL/TPS 增益，可以形成窄而合理的“constrained policy improvement for parallel drafting”贡献；仅在这一单一 DFlash workstream 上换 loss，尚不足以天然达到 NeurIPS/ICML/ICLR 方法新颖性。

---

## 3. Joint training 是否只是旧 adaptation 换名

它不完全等同于失败的 LoRA/full adaptation：

- 旧 R043–R046 使用 target posterior/frontier KL 或既有 Domino correction graph；
- PARC 改变了部署 head、KEEP action semantics 和训练约束；
- 因此作为一次严格 falsifier，科学上仍合理。

但以下内容不能作为新贡献：

- base-anchored joint training 已是 [Domino](https://arxiv.org/html/2605.29707) 的明确机制；
- 588M full adaptation 和 full-vocabulary KL 已显示“解冻更多参数”本身几乎不迁移；
- 当前没有 matched frozen-PARC arm，无法证明收益来自 representation co-adaptation，而不是新 loss/head。

所以 joint training只能是高风险训练配方或待证机制，不能写成 supporting contribution。若保留该 claim，必须有同数据、同 head、同 objective 的 frozen-DFlash control。

---

## 4. Frontier / Novelty Boundary

- **DFlash**：已覆盖一次 blockwise parallel draft；PARC 不能认领。
- **D-PACE**：已推导 expected accepted length 与动态位置权重。当前 `L_gain` 基本是 candidate-head 上的 D-PACE；truncated gain 只是 base-conditioned extension。[D-PACE 原文](https://arxiv.org/html/2605.18810)
- **PTP**：已证明单个模型调用可表达 token 间任意依赖；PARC 不能认领“一次并行但有依赖”的一般能力。[PTP 原文](https://arxiv.org/html/2512.21323)
- **SpecFormer**：已使用 bidirectional draft attention；PARC 不能以 noncausal Transformer 为创新。[SpecFormer 原文](https://arxiv.org/html/2511.20340)
- **Domino**：PARC 与其 causal GRU/selected-prefix feedback 有清晰在线区别；但 base-anchored joint curriculum 不新。
- **PGCF/JAPD/PCLD**：这是最致命的本地 closest work。soft summary、global noncausal selection、base residual、zero identity、accepted-prefix loss和 safety regularization 均已有直接覆盖。
- **Speculative Correction**：最新工作已明确提出“先完成完整 draft，再以 full-sequence bidirectional correction 作为 editable initialization”，且使用多个 diffusion denoising passes。[原文](https://arxiv.org/html/2608.02625) 该方法是 response-level、iterative DLM refinement，使用大型 refiner，不是 strict lossless Top16 one-shot speculative head，因此不覆盖 PARC 最窄的部署接口；但它已经覆盖并显著削弱“把完整 draft 当 noisy sequence 做全局纠错”这一高层叙事。PARC 必须引用它，不能再把 draft-as-noisy-codeword/bidirectional correction 写成主新意。

若要保留 architecture/interface novelty，最低限度应从 lossy soft summary 改为显式 KEEP-relative `16×16` edit-action tensor：每个 action node编码 candidate-vs-Top1 embedding/logit delta，全部 256 actions 做一次 noncausal exchange，再同时输出 advantages。但本地 full-node PGCF/PCLD 已很接近；因此这也需要单独 novelty audit，不能仅换成 delta feature 就宣称新架构。不要为了 novelty 再堆 latent、diffusion、router 或第二个 head。

---

## 5. 参数、延迟与训练可行性

`9.47M` 估算在 hidden size 2560 下数量级可信：

- `W_h + W_e` 约 `2.621M`；
- 三个 D512/FFN1024 Transformer block 约 `6.30M`；
- scorer/projections 约 `0.53M`；
- 总计约 `9.45–9.48M`，取决于 bias/LN/position 参数。

但必须给出逐项 exact ledger 和代码断言。D512 preprojected BF16 lexical table 为

\[
151936\times512\times2
=155{,}582{,}464\ {\rm bytes}
\approx148.38\ {\rm MiB}.
\]

A40 eager latency低于 Domino 是可信假设，因为 mixer 只有16个 position tokens；但尚不是事实。profile 必须计入 base vocab GEMM、FP32 Top16/LSE、gather、完整 PARC、argmax，以及约148 MiB derived table。`W_e` 训练期间会变化，所以 table 只能在最终 freeze 后 materialize并做 raw-vs-table parity。

`10–30 A40 GPU-hours` 对联合优化537M backbone缺少 blocks/epochs/effective batch/checkpointing依据。还需报告：

- 新增在线参数；
- 总优化参数；
- optimizer-state/activation memory；
- `L_base` 与 `L_gain` 的归一化和相对尺度；
- head loss 到 hidden/logits 的精确 gradient routing；
- live TopK rank swap、gold-not-in-K 和 reference-base protection 的处理。

---

## 6. Validation Focus

现有 512 → 25K三臂 → conditional scale/system 的阶段顺序是合理的，但当前三臂只能区分：

- global vs local；
- risk constraint vs no-risk。

它不能区分 joint representation，因为三臂都联合训练。若 joint representation仍是 claim，最小 matched design 实际需要第四个 `global+risk+frozen-DFlash` arm。若不愿增加 arm，应删除 joint representation contribution，只把联合训练称为 recipe。

还需修正：

- capacity 的“完整 clean-prefix recovery≥95%”与正文“oracle-gap recovery≥95%”不是同一指标；
- 25K 要固定 train/select/heldout prompt counts；
- global-local 应预注册最小 effect 与 paired CI，而不仅是“global exceeds”；
- risk gate必须以最终 deterministic accepted-length harm为 binding metric；
- conditional scale 不能由 train loss或 capacity授权。

---

## 七维评分

| 维度 | 分数 | 评语 |
|---|---:|---|
| Problem Fidelity | **9/10** | 完整保持问题与所有禁止项，目标非常清楚。 |
| Method Specificity | **6/10** | 数据流具体，但 risk、移动基线、primal-dual update、loss尺度和gradient routing未闭合。 |
| Contribution Quality | **4/10** | soft-summary global head、residual identity、D-PACE、安全hinge、joint training均有直接先例；当前差异主要是重参数化和约束形式。 |
| Frontier Leverage | **5/10** | 使用 bidirectional one-shot primitive合适但不新，且漏掉最新 Speculative Correction 边界。 |
| Feasibility | **5/10** | 线上结构可实现；当前概率约束会在 identity 点制造巨大虚假风险，联合训练还有 moving-base 退化。 |
| Validation Focus | **6/10** | 阶段化和 stop logic较好，但三臂无法识别 joint representation，若干门未精确定义。 |
| Venue Readiness | **4/10** | 当前最强审稿解读是“已有 global selector + D-PACE + 标准 constrained safety”，尚无足够独立的方法贡献。 |

**Weighted OVERALL：5.6/10**

计算：`0.15×9 + 0.25×6 + 0.25×4 + 0.15×5 + 0.10×5 + 0.05×6 + 0.05×4 = 5.6`。

### GAP

没有项目提供的3好3坏人工 anchors，因此按绝对 top-venue 方法标准评分。距离 READY 不是增加实验数量，而是两个结构性缺口：先把 risk/base-reference 数学闭合；再证明贡献不只是已有 JAPD/PGCF接口上的 D-PACE 与 safety 改写。前者修复后可以进入严格 falsifier，后者若无新的理论边界或强结果仍会阻止 top-venue acceptance。

---

## 对所有低于7分维度的修订要求

### Method Specificity — 6

- **Weakness：** risk 不对应 hard harm；`a_base` 随联合训练移动；primal-dual与loss尺度未定义。
- **Method-level fix：** 使用 truncated accepted-gain、protected-prefix max-margin upper bound；固定 reference base或冻结 DFlash；写明 dual update、averaging unit和gradient routing。
- **Priority：CRITICAL**

### Contribution Quality — 4

- **Weakness：** KEEP-relative是 gauge fixing；soft summary与JAPD重复；D-PACE/safety/joint training均已有。
- **Method-level fix：** 将唯一主贡献收窄为“相对固定base的 deterministic-harm-constrained policy improvement”，给出正式推导；否则必须重新设计真正不同且不丢失 candidate information 的 action interface。
- **Priority：CRITICAL**

### Frontier Leverage — 5

- **Weakness：** “noisy sequence global correction”已被 Speculative Correction高层覆盖；bidirectional Transformer只是标准 primitive。
- **Method-level fix：** 补最新文献并明确 narrow boundary：strict one-pass、lightweight、Top16、single-chain、lossless verifier。不要加入 diffusion/PTP 作为装饰。
- **Priority：IMPORTANT**

### Feasibility — 5

- **Weakness：** 当前 `R` 在 zero-hard-harm 时也可能接近1；联合训练可通过缩短 base prefix规避约束；live TopK使support不连续。
- **Method-level fix：** 先在512 blocks验证 identity点 risk≈0、margin surrogate与hard harm排序一致、reference semantics不随训练变化；失败即关闭。
- **Priority：CRITICAL**

### Validation Focus — 6

- **Weakness：** 三臂不能识别 joint representation；capacity指标表述不一致。
- **Method-level fix：** 加 matched frozen arm，或删除 joint representation claim；固定split、global-local effect、hard-harm gate和capacity metric。
- **Priority：IMPORTANT**

### Venue Readiness — 4

- **Weakness：** 当前最强贡献仍可被审稿人归约为 incremental loss tweak。
- **Method-level fix：** 先解决数学与novelty blockers，再讨论规模化；不要用更大head或更多数据替代贡献。
- **Priority：CRITICAL**

---

## Simplification Opportunities

1. 删除“joint representation”作为 supporting contribution；在有 matched frozen evidence 前只称训练配方。
2. 不把 soft Top16 summary包装成新接口：若追求最小机制，可直接删掉，只保留 Top1 sequence + uncertainty scalars；若认为完整候选交互必要，则一次性改成明确的 full edit-action tensor，不要两者并存。
3. D512/H8/L3 不是 novelty。若更小固定 head 能过 semantics/capacity，就优先缩小；不要做 width 菜单。

## Modernization Opportunities

除正确实现 constrained optimization 外，**NONE**。不应引入 diffusion refinement、PTP latent、RL、router或第二阶段。最新 diffusion work的作用是限定 novelty，不是授权采用其迭代路径。

## 最强拒稿理由

> PARC 将一个已在本项目中实现过的 soft-summary global selector，与 D-PACE、已有 base-prefix safety 和 Domino式 joint training重新组合；KEEP-relative logits只是 residual score的 gauge fixing，而声称控制1% harm的概率乘积既不对应部署argmax harm，又在零伤害 identity 点产生巨大虚假风险。即使修复该公式，剩余差异仍主要是 D-PACE 的 base-conditioned truncation与标准 primal-dual margin constraint，尚不足以构成新的 top-venue sequence-modeling method。

## Blocking Issues

1. `R=1-\prod q_keep` 与 deterministic harm错位。
2. 联合训练下 moving `a_base` 可使约束被规避。
3. soft-summary/KEEP-relative interface缺乏相对 JAPD/PGCF/PCLD 的实质新意。
4. `L_base + L_gain` 的归一化、权重与gradient routing未定义。
5. 三臂设计不能支持 joint representation claim。
6. 最新 Speculative Correction 已覆盖 broad draft-then-global-correct narrative。

## 可保留部分

- full16、noncausal、one-call、one-chain 的在线结构；
- `A_i0=0` 的清晰 action semantics与exact step-0 identity；
- pure-base margin直接进入 edit advantage；
- 512 capacity → disjoint pilot → conditional system 的 stop-first流程；
- 参数预算和 preprojected lexical table方向；
- global/local 与 no-risk 两个核心 falsifier；
- fixed/dynamic/SGLang 三层最终门。

## 最终 Verdict

**RETHINK**

若把 gain 改成固定-reference的 truncated accepted-gain D-PACE，并用 protected-prefix max edit margin的逐block上界做真正 primal-dual constraint，当前方向可升级为一个值得跑的 **REVISE-level falsifier**。但这仍主要是 objective-level贡献；在证明理论边界或提供决定性跨设置结果前，不能称为 top-venue-ready 新方法。
