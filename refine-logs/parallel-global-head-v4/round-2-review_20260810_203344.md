# PARC-16 Round-2 Same-Reviewer Deep Review

**CALIBRATION: none**

## 总结裁决

Round-1 的两个核心数学错误已基本关闭：

- `a_b`、protected prefix、reference margin 均来自 immutable released-DFlash snapshot，joint DFlash 不能再通过缩短 live prefix 逃避约束；
- `Hbar=ReLU(1+M/gamma)` 在明确的非并列、support-in-live-Top16 条件下，确实逐 block 上界 deterministic harm，并在 step-0 identity 精确为零；
- truncated detached weights 确实给出所定义的 conditional incremental-gain surrogate 的精确一阶梯度；
- soft summary 已删除，完整256个 edit-action nodes 保留，线上仍严格 full16、global noncausal、one-call、one-chain。

因此方向已从 Round-1 的 **RETHINK** 提升为一个数学上基本成立、值得实现验证的方案。

但尚有三个阻止 READY 的问题：

1. `gamma_b=delta_b/2` 对极小非零 reference margin 会产生无界大的梯度，当前只处理 exact tie，没有定义数值 ambiguity threshold；
2. `A_{bi,r_ref}` 的下标语义不闭合：joint training后 reference gold 未必仍是 live rank0，必须明确使用 live gold rank；
3. M2 的裁决存在两处错误：用已知数据不足的1589 prompts要求直接超过 Domino可能过早关闭；要求 joint arm 必须胜 frozen arm则会错误拒绝一个更简单、有效的 frozen PARC。

此外，主贡献现在虽然连贯，但仍是 objective-level：base-conditioned D-PACE truncation加 blockwise constrained margin。它需要强结果才能达到顶会贡献强度。

**Verdict：REVISE**

---

## 1. Immutable Contract / Drift

| 合同项 | Round-2结论 |
|---|---|
| 一次消费完整16位 | PASS |
| 每个action node看到全部256 nodes/16 positions | PASS |
| 无 causal mask | PASS |
| 一次输出 `[B,16,16]` | PASS |
| 一次逐位置argmax得到唯一 `[B,16]` | PASS |
| Top16仅为每位置candidate axis | PASS |
| selected-token feedback | NONE |
| GRU/causal rollout | NONE |
| serial target seed/decode | NONE |
| Jacobi/iteration/第二次head | NONE |
| beam/tree/trie/forest/multipath | NONE |
| ordinary verifier外在线target forward | NONE |

256 action nodes之间的连续 self-attention不等于候选路径搜索。网络深度是普通层深度，不是输出位置递归。reference model仅用于离线标签，不进入部署图。

**Drift Warning：NONE**

不得用 diffusion refinement、PTP sequential correction、GRU、tree或target-side refinement解决剩余问题。

---

## 2. Round-1 Blocker Closure

| Round-1 blocker | 状态 | Round-2判断 |
|---|---|---|
| `1-prod q_keep`不对应deterministic harm | **CLOSED，带数值条件** | 新的max-margin上界数学方向正确；仍需极小margin处理。 |
| moving `a_base` 可被joint training规避 | **CLOSED** | `a_b/y_b/delta_b`来自immutable snapshot且不重算；support drop fail-closed。 |
| soft-summary/JAPD接口重复 | **CLOSED** | 删除soft summary，恢复完整256 action carrier；同时正确放弃carrier novelty。 |
| loss normalization/gradient routing不明确 | **MOSTLY CLOSED** | `/16`、prompt sampling、dual update和routing均已写明；需限定梯度等价是piecewise/conditional。 |
| 三臂不能识别joint | **PARTIAL** | frozen control已加入，但当前“必须胜frozen”裁决逻辑错误。 |
| 漏掉Speculative Correction | **CLOSED** | broad draft-then-refine claim已删除，边界清楚。 |

---

## 3. Immutable Reference 是否仍可被 game

当前 reference contract总体正确。

冻结并持久化：

\[
a_b,\quad y_{bi},\quad \delta_b
\]

以后，joint DFlash不能通过下列方式逃逸：

- 缩短live accepted prefix；
- 重算一个更短的`a_b`；
- 把protected位置从loss中移除；
- 让gold掉出Top16后把block当作无风险。

若任一protected gold掉出live Top16，`Hbar_b=1`，因此该block仍消耗完整风险预算。这保证了约束语义不会fail-open。

### 剩余问题：fail-closed分支没有constraint gradient

support drop时`Hbar=1`是常数，constraint本身无法对掉出Top16的gold提供恢复梯度；恢复完全依赖`L_base`。这不是可gaming漏洞，但可能形成不可行状态：

- `lambda`持续升高；
- constraint项对该block仍无梯度；
- 只有full-vocabulary base loss尝试恢复support。

可接受的最小处理是：明确将这种情况称为“constraint infeasibility diagnostic”。若support-drop rate不快速回落且`lambda`触顶，训练判失败，不能声称primal-dual已控制风险。

---

## 4. Truncated Gain 梯度检查

定义

\[
P_{bt}=\prod_{j=a_b}^{t}q_{bj}(r_{bj}),\qquad
G_b=\sum_{t=a_b}^{h_b-1}P_{bt}.
\]

则

\[
\nabla_\theta\left(-G_b\right)
=
-\sum_{i=a_b}^{h_b-1}
\left(\sum_{t=i}^{h_b-1}P_{bt}\right)
\nabla_\theta\log q_{bi}(r_{bi}).
\]

令

\[
w_{bi}=\operatorname{sg}\left(\sum_{t=i}^{h_b-1}P_{bt}\right),
\]

则

\[
\nabla_\theta
\sum_iw_{bi}\left[-\log q_{bi}(r_{bi})\right]
=
\nabla_\theta[-G_b].
\]

因此正文的`L_gain`在以下条件下确实等于`-mean(G_b)/16`的一阶梯度：

- live candidate IDs和`h_b`在当前局部参数区域固定；
- 不对TopK/rank/support选择求梯度；
- `w`完全detach；
- 不使用`0.5q+0.5` smoothing；
- 所有乘积使用同一未平滑`q`。

共享参数同时影响多个位置不破坏该推导。

### 必须收窄的表述

它不是无条件 total incremental EAL 的精确relaxation，因为乘积从`a_b`而非0开始。它是：

> 在reference protected prefix由单独hard-risk constraint保持的条件下，新增accepted suffix length的product surrogate。

此外，该等价是**固定live support内的piecewise exact gradient**。TopK换位或gold进出Top16时，`G_b`本身不连续，不能声称全局可微等价。

这是表述修订，不是方法否定。

---

## 5. `gamma`与Hard-Harm Upper Bound逐边界检查

对protected位置，必须令`r^{live}_{bi}`表示`y_{bi}`在当前live Top16中的rank：

\[
m_{bi}=
\max_{k:C_{bik}\neq y_{bi}}
A_{bik}
-
A_{bi,r^{live}_{bi}},
\qquad
M_b=\max_{i<a_b}m_{bi}.
\]

当前稿中的`A_{bi,r_ref}`容易被实现成reference rank0，这是错误的。joint DFlash后gold可能位于live rank`k>0`。必须统一改为`r^{live}_{bi}`。

### 正常非并列情况

令每个reference protected位置的正确margin为`δ_bi>0`，且

\[
\delta_b=\min_{i<a_b}\delta_{bi},\qquad
\gamma_b=\delta_b/2.
\]

step 0时：

\[
m_{bi}=-\delta_{bi},\qquad
M_b=-\min_i\delta_{bi}=-\delta_b.
\]

所以

\[
\bar H_b
=
\operatorname{ReLU}\left(1-\frac{\delta_b}{\delta_b/2}\right)
=0.
\]

identity closure正确。

任何`M_b>=0`时：

\[
\bar H_b
=
\operatorname{ReLU}(1+M_b/\gamma_b)
\ge1.
\]

用保守tie定义`H_b=1[M_b>=0]`，有逐block

\[
H_b\le\bar H_b.
\]

因此相同非负prompt权重下：

\[
\operatorname{PromptMean}(H)
\le
\operatorname{PromptMean}(\bar H).
\]

上界结论正确。

### Tie、support drop与空prefix

- `a_b=0`：不存在可被破坏的reference prefix，`Hbar=0`正确。
- protected gold掉出live Top16：输出空间不可能重现reference prefix，实际harm必为1，设`Hbar=1`正确。
- live score tie：用`M>=0`标harm比真实argmax更保守，仍是合法上界。
- reference exact tie：`delta=0`导致gamma无定义；将其标ambiguous并设`Hbar=1`正确。

### 新的数值blocker

只处理“exact FP32 tie”不足。若

\[
0<\delta_b\ll1,
\]

则constraint梯度比例为`1/gamma_b=2/delta_b`，可能极大，造成单个近并列block主导batch、BF16路径溢出或Adam状态爆炸。

必须在M0前冻结一个数值ambiguity阈值，例如：

\[
\delta_b\le\delta_{\min}
\Rightarrow \text{ambiguous},\quad\bar H_b=1.
\]

`delta_min`应由FP32/BF16 replay误差上界决定，而非用结果调参。必须先报告ambiguous prompt/block比例；若其prompt mean本身超过1%，当前1% surrogate constraint先验不可行。

另需修改“光滑上界”措辞：`max + ReLU`只是几乎处处可微的分段线性上界，不是smooth upper bound。

---

## 6. Loss Normalization、Dual Update与Gradient Routing

### 已闭合部分

- `L_base`和`L_gain`均按block length 16归一化；
- prompt均匀采样、prompt内均匀采block使batch mean成为prompt-balanced estimator；
- `L_base → DFlash only`；
- `L_gain/Hbar → PARC + DFlash H/Z`；
- target、reference quantities、TopK IDs均stop-gradient；
- `lambda_0=0`、EMA violation、projected ascent和clip均明确；
- 三项原始loss、gradient norm、lambda、support drop和actual harm均报告。

这些已足够实现第一版。

### 仍需写死的两点

1. 初始化`c_0=0`。
2. 若`lambda=100`触顶而EMA constraint violation仍为正，必须判为constraint infeasible并停止，不能继续训练后按actual harm挑checkpoint。

固定dual LR `0.05`和EMA `0.95`是工程超参数，不是理论保证。可以不做grid，但不能把一次未收敛解释成objective无效；M0/M1应先证明这一固定更新能在synthetic/capacity上满足约束。

---

## 7. Full 256 Edit-Action Carrier与参数/成本

### 参数账本

新公式仍与PGCF D256/L2账本相容：

- shared `W_h + W_e`：`1,310,720`；
- position/rank embeddings：`8,192`；
- five-scalar projection：`1,536`；
- `W_c`：`65,536`；
- input LN：`512`；
- two full-node Transformer blocks：`1,051,136`；
- output LN + scorer：`768`；
- 总计：`2,438,400`。

增加两个相同block得到：

\[
2{,}438{,}400+2(525{,}568)=3{,}489{,}536,
\]

所以L4数字也一致。BF16 projected lexical table：

\[
151936\times256\times2
=77{,}791{,}232\text{ bytes}.
\]

参数与memory claim可信。

但最终规范仍应列出bias、LayerNorm affine、dropout和attention relative-bias flags，否则“精确2,438,400”只是在引用旧实现，而不是由本稿自足定义。

### 在线成本

256-token、D256、两层full attention的计算量与已profile PGCF family同量级，A40路径可信；但新的delta-node构造和joint-trained checkpoint仍须重新测complete path。旧profile可证明可行性，不能直接作为PARC latency结果。

### 简化判断

预注册L4 fallback没有方法价值，并接近Problem Anchor禁止的width rescue。既然：

- PGCF已证明D256/L2可精确拟合512 blocks；
- 本稿的唯一贡献是objective；
- transfer failure不能授权扩容；

建议删除L4 fallback。D256/L2 capacity失败时只修实现错误，否则关闭objective。

---

## 8. Novelty与贡献强度

修订后主张已明显更诚实：

- 不认领bidirectional attention；
- 不认领full-node carrier；
- 不认领KEEP gauge；
- 不认领joint training；
- 不认领draft-then-refine；
- 只认领fixed-reference constrained incremental acceptance。

这与以下工作存在可陈述差异：

- D-PACE优化total accepted-length surrogate，没有固定已部署policy的hard-harm约束；
- 旧PGCF/JAPD/PCLD没有将immutable baseline、conditional incremental gain和pointwise block-harm upper bound组成一个constrained policy-improvement目标；
- Speculative Correction是多pass、大型diffusion refiner和response-level quality/latency路线，不是strict speculative single-chain head。

但贡献仍偏窄。最强归约仍然是：

> 截断D-PACE的tail sum，并把已有per-position fixed-weight hinge改成reference-normalized block-max hinge与primal-dual update。

这比Round-1连贯得多，足以成为一个严谨falsifier，也可能在强结果下成为方法贡献；但在没有结果时还不足以声称顶会级实质新意。不要再增加architecture模块补新颖性。论文强度应来自：

- 清楚的pointwise upper-bound与conditional-gradient推导；
- 显著跨过旧global-head长期约`+0.24 EAL`上限；
- actual fixed/dynamic EAL与同栈TPS硬结果。

---

## 9. Main-Goal-First Validation与Stop Rules

阶段化思路正确，但有两处会错误裁决。

### 9.1 1589-prompt main arm不应作为最终科学关闭门

现有证据已表明约2K prompt上的global selector和full adaptation转移很弱。当前方案原本将25K视为最小pilot，却在修订后要求1589-prompt arm直接：

- 超过Domino；
- 相对reference `+0.75`；
- 三域不退化；
- harm≤1%。

这可能把“已知数据规模不足”误判成objective失败。它测试的是极强low-data sample efficiency，而不是最终机制是否成立。

最小修订：

- 1589/199/199只作为nonbinding smoke：要求训练稳定、actual harm≤1%、heldout方向为正且无显著域崩溃；
- 唯一binding main arm直接放在25K prompt上；
- 25K失败才关闭，不需要增加实验菜单。

若坚持2K关闭门，至少不能要求超过Domino；应只要求预注册的正增益与global signal。

### 9.2 Frozen control的决策规则错误

当前要求：

> global-joint相对local-joint和global-frozen均`>=0.15 EAL`。

相对local的门是必要的，因为用户要求有效global context。

但相对frozen不是主贡献门。若global-frozen与global-joint相当甚至更好，这意味着joint recipe可删除，得到更简单、更可信的PARC；不应判方法失败。

正确规则：

- `best(global-joint, global-frozen) - matched-local >=0.15`且CI lower>0；
- 若frozen不差于joint，冻结DFlash并删除joint recipe；
- 只有在joint明显胜出时才保留joint；
- frozen control是recipe selection/deletion test，不是paper claim gate。

### 9.3 Constraint deletion

仅在main成功后跑constraint deletion是合理的，不会浪费主线资源。它用于证明risk component必要，不应在机制尚未产生gain时扩展菜单。

### 9.4 M1 metric

Problem Anchor要求“完整clean-prefix recovery≥95%”，而M1仍写“oracle-gap recovery≥95%”。应固定一个唯一公式。建议明确：

\[
\text{recovery}
=
\frac{\operatorname{EAL}_{PARC}-\operatorname{EAL}_{ref}}
{\operatorname{EAL}_{Top16\ oracle}-\operatorname{EAL}_{ref}}.
\]

同时报告complete-prefix success/J2，但只保留一个binding recovery定义，避免再次用token accuracy替代prefix utility。

---

## 七维评分

| 维度 | Round-1 | Round-2 | 评语 |
|---|---:|---:|---|
| Problem Fidelity | 9 | **9** | Anchor完整保留，线上合同无漂移。 |
| Method Specificity | 6 | **8** | reference、objective、upper bound、dual和routing已具体；剩余是live-rank和数值margin边界。 |
| Contribution Quality | 4 | **6** | 已形成一个连贯的fixed-reference constrained objective，但仍可被归约为D-PACE+margin的窄扩展。 |
| Frontier Leverage | 5 | **7** | 正确处理Speculative Correction/PTP/SpecFormer边界，没有强加现代装饰。 |
| Feasibility | 5 | **7** | carrier、参数和部署路径已有依据；极小gamma与support-drop infeasibility仍需处理。 |
| Validation Focus | 6 | **6** | main-first合理，但2K hard close与必须胜frozen会错误裁决。 |
| Venue Readiness | 4 | **6** | 已达到值得实验的proposal水平；尚缺足够贡献宽度和决定性证据。 |

**Weighted OVERALL：7.2/10**

\[
0.15(9)+0.25(8)+0.25(6)+0.15(7)+0.10(7)+0.05(6)+0.05(6)=7.2.
\]

### GAP

没有项目提供的3好3坏人工anchors，因此继续使用绝对顶会方法标尺。Round-2已跨过“数学对象错误”的主要鸿沟，距离READY的差距不再是增加模块，而是：处理极小reference margin和live-rank边界；修正会误杀方法的validation规则；随后用真实disjoint EAL证明这个窄objective不是旧global-head loss的微小改写。即使实现完全正确，没有显著结果时Contribution Quality仍不会达到9。

---

## 所有低于7分维度的修订要求

### Contribution Quality — 6

- **Weakness：** 核心仍可归约为base-conditioned D-PACE truncation加block-max constrained hinge。
- **Method-level fix：** 不增加架构；把conditional gradient与pointwise harm bound写成正式命题，限定适用条件，并让main结果直接验证旧loss misallocation假设。
- **Priority：IMPORTANT**

### Validation Focus — 6

- **Weakness：** 2K hard close可能数据不足；“必须胜frozen”会拒绝更简单的成功方法。
- **Method-level fix：** 2K降为nonbinding smoke、25K作为唯一pilot gate；frozen arm用于删除/选择joint recipe，仅global-vs-local是机制硬门。
- **Priority：CRITICAL**

### Venue Readiness — 6

- **Weakness：** 尚无证据表明该目标能突破本地长期弱transfer上限。
- **Method-level fix：** 完成修订后的最小main pilot；不扩论文claim，不在通过accepted-length gate前做系统包装。
- **Priority：IMPORTANT**

---

## Blocking Issues

1. **CRITICAL：** 将风险式中的`r_ref`改为reference gold在当前live Top16中的`r^{live}_{bi}`，并为实现写单测。
2. **CRITICAL：** 冻结基于数值replay误差的`delta_min`；`delta<=delta_min`视为ambiguous/fail-closed，并在launch前证明其prompt mean不使1%约束先验不可行。
3. **CRITICAL：** 修正M2：不再以1589 prompts直接超过Domino作为唯一关闭门。
4. **CRITICAL：** frozen arm若匹配或胜joint，应删除joint recipe，而不是关闭PARC。
5. **IMPORTANT：** 明确梯度等价仅对固定live support成立，且是conditional incremental suffix surrogate，不是无条件total EAL。
6. **IMPORTANT：** 若`lambda`触顶而constraint仍违反，声明infeasible并停止。
7. **IMPORTANT：** 统一M1的prefix-recovery绑定公式。

---

## Simplification Opportunities

1. 删除D256/L4 fallback；carrier容量已被既有512-block证据支持，objective失败不能用width rescue。
2. 将joint training设为可删除recipe：frozen匹配时直接采用frozen版本。
3. 保留主臂→local/frozen→constraint deletion的条件顺序，不增加其他loss或architecture消融。

## Modernization Opportunities

**NONE。** 当前primal-dual fixed-reference formulation已经是合适的现代工具。不要加入diffusion、RL、latent variable、router或第二次head。

## 最强拒稿理由

> PARC现在的安全约束和fixed-reference语义是正确且有价值的，但核心方法仍是D-PACE tail-sum的条件截断，加上对已有base-prefix hinge的block-max、margin-normalized primal-dual重写。论文尚未证明这一目标级变化能解决PGCF/JAPD/PCLD已经暴露的跨prompt信息不可辨识性；当前2K门还可能把数据不足误写成方法否定，而joint必须胜frozen的规则会拒绝更简单的成功解。

## 可保留部分

- immutable reference contract；
- full256 KEEP-relative edit-action carrier；
- D256/L2、2,438,400参数与77.8MB table；
- conditional incremental-gain gradient；
- pointwise deterministic-harm upper bound；
- prompt-balanced constrained optimization；
- main-arm-first与系统工作后置；
- local visibility control和frozen recipe deletion test；
- fixed→dynamic→same-stack SGLang硬门。

## 最终 Verdict

**REVISE — 7.2/10**

当前方案已经值得实现M0/M1并进入一个正确规模的main pilot，但仍不满足READY：数值margin边界和live-rank语义尚未闭合，validation有两个会错误关闭路线的规则，且objective-level novelty必须由决定性accepted-length结果支撑。
