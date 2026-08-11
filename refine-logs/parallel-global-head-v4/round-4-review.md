# PARC-16 Round-4 Final Readiness Audit

**CALIBRATION: none**

## 总体裁决

Round-3的全部方法级blocker基本关闭：

- `q=softmax(A.float())`、temperature 1和FP32 log-softmax/product已写死；
- BF16-live与FP32-ref的identity证明已改成正确的不等式；
- `g_max=64`给出了非零且有限的margin-gradient上界；
- support-drop窗口和patience可复现；
- matched-local只改变跨位置可见性；
- on-policy refresh不再更新immutable reference；
- 在线架构、参数账本及no-smoke要求均合规。

但最新严格数据合同尚未闭合。当前稿在训练前使用validation和held-out records计算`e_num_cert/delta_min`及ambiguous比例，提前物化held-out Domino/DFlash baseline，并允许根据首次held-out结果扩到1.42M或在dynamic失败后refresh再评估。这使validation不再“只用于checkpoint选择”，held-out也不再“选择后一次打开”。

这是数据协议blocker，不是方法或架构blocker。修复不需要新增训练臂：把所有训练阈值只从90K train导出，并规定第一次打开held-out之后绝不再训练或调节主模型即可。

**Verdict：REVISE**

---

## 1. Round-3 Blocker Closure

| Round-3项目 | 状态 | Round-4判断 |
|---|---|---|
| `q`、temperature、FP32路径 | **CLOSED** | `q_bi(k)=softmax(A.float())`、temperature 1及FP32概率路径明确。 |
| detached gain gradient | **CLOSED** | fixed support内与`-Mean(G)/16`的梯度等价成立。 |
| BF16-live vs FP32-ref identity | **CLOSED** | 已正确使用`M(0)=-delta_live`及误差不等式。 |
| `e_num=0` | **CLOSED** | `2/g_max=0.03125`提供正下界。 |
| margin-gradient上界 | **CLOSED** | stable branch有`2/delta_b<64`；这是对margin的导数界，不应扩写成完整参数梯度界。 |
| ambiguity fail-closed | **数学CLOSED，数据范围BLOCKED** | `Hbar=1`正确，但不能用validation/held-out ambiguity决定训练是否启动。 |
| support-drop stop | **CLOSED** | cadence、起始step、四窗口、阈值及80%规则均明确。 |
| matched-local唯一区别 | **CLOSED** | 除cross-position attention mask外，参数、数据、优化与选择规则一致。 |
| on-policy refresh保持reference | **CLOSED** | 始终保留最初released snapshot。 |
| 参数账本 | **CLOSED** | 2,438,400参数及77,791,232-byte table自洽。 |
| capacity/512/2K/25K训练 | **CLOSED** | 当前执行计划中均已删除。 |
| standalone GPU smoke | **CLOSED** | R0仅为本地unit/shape/gradient safeguards。 |
| validation/held-out公平性 | **BLOCKED** | certificate、baseline物化和post-heldout retraining违反最新合同。 |

---

## 2. Immutable Architecture Audit

在线图保持完全合规：

| 合同项 | 结论 |
|---|---|
| 一次消费full16 | PASS |
| 256 actions全局noncausal互见 | PASS |
| 一次输出`[B,16,16]` | PASS |
| 一次argmax产生唯一`[B,16]` | PASS |
| Top16仅为position-local candidate axis | PASS |
| selected-token feedback | NONE |
| GRU/causal rollout | NONE |
| iteration/Jacobi/second head | NONE |
| beam/tree/trie/forest/multipath | NONE |
| extra online target forward | NONE |
| online reference model | NONE |

**Architecture Drift Warning：NONE。**

R0本地autograd、shape和枚举检查属于允许的fail-fast safeguards，不是capacity或GPU efficacy实验。100K数据收集及同一正式launcher中的preflight也不构成独立smoke。

---

## 3. `q`与Conditional Gain

概率分布现已完整定义：

\[
q_{bi}(k)=\operatorname{softmax}_k(A_{bi:}^{FP32})_k.
\]

在candidate IDs、gold rank与`h_b`固定的局部区域，detached权重满足：

\[
\nabla_\theta L_{\rm gain}
=
\nabla_\theta[-\operatorname{Mean}(G_b)/16].
\]

以下边界也已明确：

- temperature固定为1；
- 不对TopK/support变化求梯度；
- 不使用概率smoothing；
- 不声称跨support边界全局可微；
- 不声称是无条件total-EAL梯度。

唯一小问题是符号冲突：architecture中`q_i`已经表示hidden query，随后`q_bi(k)`又表示candidate probability。建议把前者改为`u_i`，但这不是方法blocker。

---

## 4. Identity、Numeric Certificate与`g_max`

对stable block，当前证明正确：

\[
M_b^{(0)}=-\delta_{\mathrm{live},b},
\]

且若

\[
|\delta_{\mathrm{live},b}-\delta_{\mathrm{ref},b}|
\le e_{\mathrm{num}},
\qquad
\delta_{\mathrm{ref},b}>2e_{\mathrm{num}},
\]

则

\[
\delta_{\mathrm{live},b}>
\delta_{\mathrm{ref},b}/2=\gamma_b,
\]

所以`\bar H_b=0`。

阈值

\[
\delta_{\min}
=
\max(2e_{\rm num}^{cert},2/64,2^{-14})
\]

也确实保证：

\[
0<2/\delta_b<2/\delta_{\min}\le64.
\]

因此`e_num=0`、极小margin和除零问题均已关闭。`H<=Hbar`仍逐block成立。

### 数据协议问题

当前`e_num_cert`对train、validation和held-out全部records取最大值。由于`delta_min`直接改变：

- 哪些训练block被标ambiguous；
- 哪些block产生constraint gradient；
- 训练是否因先验不可行而启动；

这等于让validation和held-out参与训练目标定义。它与“validation只用于checkpoint选择、held-out选择后才打开”的最新合同冲突。

**最小修订：**

- `e_num_cert`、`delta_min`和pretraining ambiguity-feasibility只由90K train records计算；
- validation只用于checkpoint选择，不参与任何loss阈值或launch决策；
- held-out的numeric replay只在checkpoint完全锁定后审计并报告，不能回写`delta_min`、重新分类训练数据或触发重训；
- 若希望真正跨split的输入无关证书，必须来自固定算子的保守解析界，而不能从validation/held-out样本统计导出。

Pointwise harm envelope本身不依赖该证书，因此这一修订不会削弱安全命题。

---

## 5. Loss、Dual与Support-Drop

Loss normalization和routing已经可实现：

- `L_base`、`L_gain`均`/16`；
- risk是block-level，不再错误`/16`；
- prompt先均匀、prompt内再均匀采block；
- `L_base → DFlash`；
- gain/constraint经过live `H/Z → PARC + DFlash`；
- reference与TopK IDs stop-gradient；
- `lambda_0=c_0=0`；
- EMA、projected ascent、cap和gradient clip均固定。

support-drop停止规则现在有唯一解释：

- validation cadence：10K steps；
- 从20K开始；
- 连续四个窗口均`PromptMean(support_drop)>1%`；
- 第四个值仍高于第一个的80%；
- 满足即判infeasible。

这关闭了Round-3的可复现性blocker。

需要注意，`lambda=100`且EMA violation为正是预注册的工程停止规则，不是primal-dual收敛定理；论文不得将其描述为约束问题不可行的数学证明。

---

## 6. 90K Joint Main Recipe与数据公平性

### 已闭合部分

- 新label生成前先按prompt/domain切分；
- `90K train / 5K validation / remainder≈5K held-out`；
- prompt-disjoint，并排除旧development/formal重叠；
- 旧15-position cache禁止使用；
- 第一项科学训练直接是full16、90K-prompt joint PARC；
- 不先跑frozen、local、capacity或small-data arm；
- checkpoint只按validation EAL和actual harm选择；
- 完全tie取更早step；
- held-out不用于checkpoint选择；
- DFlash、Domino、PARC使用同一prompt和evaluator。

### Blocking protocol violations

#### 6.1 提前物化held-out baselines

R1写明在训练前物化validation和held-out的Domino/DFlash baseline。即使不立刻展示汇总值，也没有必要让held-out model outputs提前进入工作流，而且不能保证最终的“same-job”比较。

应改为：

- R1只生成并封存held-out raw prompt/context/label authority；
- validation baseline可用于validation比较；
- 选定PARC checkpoint后，在一个冻结的held-out evaluation job中同时运行DFlash、released Domino和PARC；
- 该job产生唯一正式held-out结果。

#### 6.2 根据held-out结果扩1.42M

当前规则允许：

> 100K held-out高于Domino但低于1.15x，若validation曲线仍上升则扩到1.42M。

一旦100K held-out已打开，该结果就不能再决定训练数据规模，然后继续在相同held-out上评估。否则held-out已经成为model-selection signal。

允许的两种最小处理：

1. 最简单：100K held-out未达1.15x即关闭route；
2. 若必须保留1.42M：只能在首次打开held-out前，根据validation预注册并完成scale decision，锁定最终模型后才打开held-out一次。

#### 6.3 Dynamic failure后的refresh

保留最初immutable reference是正确的，但若dynamic held-out失败触发训练refresh，再回到相同dynamic held-out评估，仍然是held-out reuse。

最小规则应是：

> 第一次正式held-out打开后，任何训练、loss、数据规模或policy更新均停止。

若未来确需on-policy refresh，必须使用训练prompt生成refresh data、由dynamic validation选择，并保留一个从未打开的新final dynamic holdout。当前最简方案应在本计划中把held-out-triggered refresh删除。

---

## 7. Matched-Local Control

局部对照本身已精确：

- 256 nodes、D256/L2；
- 相同inputs、loss和joint DFlash状态；
- 相同optimizer、batch、steps、cadence和checkpoint rule；
- 唯一区别是同一position的16 candidates可互见，禁止跨position attention。

这足以把global-local差异归因于cross-position visibility。

但R4在主held-out已经打开后才训练local。为满足严格的一次held-out合同，建议：

- 先训练并锁定global checkpoint；
- 只有global validation达到预注册主门时，才自动触发matched-local与constraint-deletion；
- 两个control均只在validation上选择；
- 所有模型锁定后，一次打开held-out并同时评估。

这仍然是main-goal-first：global主模型先完成且validation成功，control不参与global选择，也不会浪费在明显失败的主模型上。

若坚持必须等global held-out成功后才训练local，则local held-out结果只能标为后验exploratory，不能作为同一次严格confirmatory global-visibility证据。

---

## 8. 参数、成本与训练规模

参数账本仍正确：

\[
1{,}310{,}720+8{,}192+1{,}536+65{,}536+512
+1{,}051{,}136+768
=2{,}438{,}400.
\]

BF16 table：

\[
151936\times256\times2
=77{,}791{,}232\text{ bytes}.
\]

D256/L2 full-node attention在A40上有可信可行性，且没有L4 fallback。旧PGCF profile只能支持“成本有希望”，不能替代最终complete-path测量；稿件已正确遵守这一边界。

训练曝光量文字需要修正。若batch 8表示8个block：

\[
180000\times8=1.44\text{M block draws},
\]

对于最多720K records约为两次block-level epoch，而不是“最多720K block exposure”。应写清楚batch单位、prompt内block采样和是否有gradient accumulation。这是实现规范问题，不需要增加实验。

---

## 9. Novelty与最强归约

当前主贡献保持单一：

> fixed-reference conditional accepted-gain与pointwise deterministic prefix-harm envelope组成的安全并行策略改进目标。

相对D-PACE、旧prefix hinge、PGCF/JAPD/PCLD和Speculative Correction的边界仍然可信。

但贡献上限没有因本轮规范修订而变化。最强归约仍然是：

> D-PACE tail truncation，加上已有base-prefix hinge的block-max、reference-normalized primal-dual重写。

两个命题是正确但直接的product-rule和indicator-envelope结果。方法可以成为严谨且有价值的falsifier；要达到顶会贡献强度，必须由干净的100K held-out结果证明它显著突破旧global-head弱transfer上限。不能用数据协议不干净的结果弥补窄novelty。

---

## 七维评分

| 维度 | 分数 | 评语 |
|---|---:|---|
| Problem Fidelity | **9** | 架构与主目标完全保真，但执行稿仍与最新held-out合同冲突。 |
| Method Specificity | **9** | 概率、数值界、loss、stop和control已接近实现级完整。 |
| Contribution Quality | **7** | 单一且严谨，仍属窄objective-level创新。 |
| Frontier Leverage | **8** | constrained formulation使用自然，无现代primitive装饰。 |
| Feasibility | **8** | 参数与100K路径可信；训练曝光量和数据封存细节待修正。 |
| Validation Focus | **6** | pretraining使用held-out统计及post-heldout retraining破坏confirmatory裁决。 |
| Venue Readiness | **7** | 方法可实现，但当前验证协议不能支持可信顶会主张。 |

**Weighted OVERALL：8.0/10**

\[
0.15(9)+0.25(9)+0.25(7)+0.15(8)+0.10(8)+0.05(6)+0.05(7)=8.0.
\]

### 低于7分维度修订

#### Validation Focus — 6

- **Weakness：** validation/held-out参与`delta_min`，held-out baseline提前生成，并允许held-out触发scale/refresh和再次评估。
- **Method-level fix：** 所有训练阈值只由90K train导出；global及预注册controls均在validation完成选择后，一次性冻结并同时打开held-out；首次held-out后禁止任何训练或数据规模更新。
- **Priority：CRITICAL**

### GAP

没有项目提供的3好3坏人工anchors。方法数学和online graph已经跨过readiness所需的大部分距离；当前差距集中在一个非常具体的confirmatory-data contract，而不是模型结构。修复后，它将成为可直接实现的高质量falsifier。距离顶会9分的另一部分仍是不可由文字关闭的贡献风险：只有干净、同集、same-job的100K结果显著超过Domino，才能证明这个窄objective不是旧loss的增量改写。

---

## Blocking Issues

1. **CRITICAL：** `e_num_cert/delta_min/ambiguity launch gate`只能使用90K train，不得使用validation或held-out。
2. **CRITICAL：** held-out DFlash/Domino/PARC必须在checkpoint锁定后于同一正式job首次共同运行；删除训练前held-out baseline物化。
3. **CRITICAL：** 删除任何由held-out结果触发的1.42M expansion、on-policy refresh或主模型重训；scale decision若保留，只能在首次held-out前由validation决定。
4. **IMPORTANT：** global/local/constraint-deletion若要形成confirmatory attribution，应在global validation成功后预注册运行、全部锁定，再一次打开held-out。
5. **IMPORTANT：** 明确最新contract supersedes Problem Anchor中旧的same-set-capacity success clause，防止下游重新生成被禁止的阶段。
6. **MINOR：** 修正1.44M block draws/约两次720K-record epoch的训练曝光账本，并消除`q_i`符号复用。

---

## Drift Warning

**DATA-PROTOCOL DRIFT。** 当前没有architecture drift，但存在与最新immutable amendment冲突的validation/held-out使用方式。该漂移会直接削弱主EAL证据的可信度，必须在实验规划前关闭。

---

## Simplification Opportunities

1. 删除100K held-out失败后的1.42M与refresh fallback；一次干净裁决比多轮test-set适配更有价值。
2. 将numeric certificate严格收窄为full 90K training corpus；held-out只作事后audit。
3. 保持唯一joint global main run；frozen、width和小数据arm继续删除。

## Modernization Opportunities

**NONE。** 不应加入diffusion、RL、router、search或第二次head。当前primal-dual目标已经是合适工具。

## 最强拒稿理由

> PARC的方法定义现已相当严谨，但其confirmatory协议并不独立：held-out records参与训练阈值，held-out baselines在选择前物化，且held-out结果还能触发更大规模训练或on-policy refresh。对于一个本就可被归约为D-PACE truncation加constrained hinge的窄贡献，一旦主EAL证据存在test-set reuse，论文将同时失去新颖性与实证可信度。

## 可保留部分

- immutable released reference及live `r_live`；
- `q=softmax(A.float())`与piecewise gradient命题；
- BF16/FP32 identity不等式；
- `g_max=64`、ambiguous fail-closed和pointwise harm envelope；
- 固定support-drop窗口；
- full256 D256/L2 carrier及参数账本；
- 90K joint main recipe；
- matched-local唯一visibility差异；
- no-capacity/no-small-set/no-standalone-smoke执行原则；
- fixed→dynamic→same-stack SGLang硬门。

## 最终 Verdict

**REVISE — 8.0/10**

当前不满足`overall>=9且无blocking issue`，因此**不授权进入`experiment-plan`**。只需修正数据封存、held-out一次打开及post-heldout no-retraining规则；不要改变方法、架构或增加实验菜单。
