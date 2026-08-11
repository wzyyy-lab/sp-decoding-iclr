# PARC-16 Round-5 Final Readiness Review

**CALIBRATION: none**

## 总体裁决

Round-4 refinement已经关闭此前全部blocking issues。最新可执行方案满足：

- 所有影响loss、训练分类、dual、launch gate和stop的量只来自90K train；
- 5K validation只选择checkpoint，不参与`delta_min`、ambiguity gate或停止；
- held-out在checkpoint、weights、数据、loss、architecture和config全部锁定后首次打开；
- DFlash、released Domino和PARC在同一个正式job、同一约5K prompts、同一evaluator中同时完成fixed与dynamic评估；
- held-out打开后永久禁止训练、扩数据、refresh、改loss、改width或重新选择模型；
- 无capacity、same-set、512/2K/25K efficacy stage，也无独立GPU smoke；
- online graph仍严格full16、global noncausal、one-call、one-chain。

方法本身也已达到实现级闭环：概率定义、conditional-gain梯度、BF16/FP32 identity条件、hard-harm envelope、数值梯度界、support-drop stop、参数账本和系统门均可直接转成交付规范。

剩余最大风险是经验风险而非proposal blocker：核心贡献仍是一个窄但连贯的objective-level创新，必须由100K disjoint held-out上的大幅accepted-length提升证明其价值。当前计划已经给出干净且不可回调的裁决方式，失败会直接关闭route。

**Verdict：READY**

---

## 1. Round-4 Blocking-Issue Closure

| Round-4 blocker | Round-5状态 | 判断 |
|---|---|---|
| `e_num_cert/delta_min`使用validation或held-out | **CLOSED** | 现在只由完整90K train records计算。 |
| ambiguity launch gate污染非训练split | **CLOSED** | 只报告train的prompt-balanced ambiguity并决定是否启动。 |
| validation参与stop或loss阈值 | **CLOSED** | stop使用train-batch EMA与冻结train-audit subset；validation只选checkpoint。 |
| 提前物化held-out baselines | **CLOSED** | held-out阶段只封存manifest；不运行DFlash、Domino或PARC，不产生统计。 |
| held-out不是same-job比较 | **CLOSED** | 三个系统的fixed和dynamic均在唯一正式held-out job共同运行。 |
| held-out触发1.42M expansion | **CLOSED** | fallback已删除；未过门立即关闭route。 |
| held-out触发on-policy refresh | **CLOSED** | refresh已删除；首次打开后禁止任何训练更新。 |
| matched-local复用主held-out | **CLOSED** | 仅作后验exploratory control，不复用主held-out，不支持本轮confirmatory claim。 |
| stale capacity success condition | **CLOSED in executable proposal** | 最新Success condition已完全改成100K prompt-disjoint正式训练；旧anchor文字被authoritative amendment覆盖。 |
| 训练曝光量歧义 | **CLOSED** | batch 8 blocks、无gradient accumulation、1.44M draws、约两次最多720K-record epoch。 |
| query/probability符号冲突 | **CLOSED** | hidden query已改为`u_i`，概率保留`q_bi(k)`。 |

---

## 2. Immutable Architecture / Drift Audit

| 合同项 | 结论 |
|---|---|
| 一次输入完整`[B,16,*]` | PASS |
| 256 edit-action nodes同时构造 | PASS |
| 所有position/candidate无causal mask全局互见 | PASS |
| 一次head invocation | PASS |
| 一次产生`[B,16,16]` | PASS |
| 一次逐位置argmax | PASS |
| 输出唯一`[B,16]` chain | PASS |
| Top16仅为每位置candidate axis | PASS |
| selected-token feedback | NONE |
| GRU/causal/Markov rollout | NONE |
| Jacobi/iteration/second pass | NONE |
| beam/tree/trie/forest/multipath | NONE |
| serial或extra online target forward | NONE |
| online reference/teacher tensor | NONE |

连续Transformer层只是普通网络深度，不是输出位置迭代。完整256-node attention也没有形成候选路径或sequence search。

**Drift Warning：NONE。**

latest amendment对旧`PROBLEM_ANCHOR.md`中capacity-first文字具有明确优先级；最新proposal的有效Success condition和R0–R5执行图均不再包含该阶段。

---

## 3. Probability、Gain与Gradient

训练分布已唯一规定为：

\[
q_{bi}(k)=\operatorname{softmax}_k(A_{bi:}^{FP32})_k,
\]

temperature固定为1，log-softmax、gold probability、prefix product和detached coefficient均用FP32。

在candidate IDs、gold live ranks和support horizon固定的局部区域内：

\[
G_b=\sum_{t=a_b}^{h_b-1}\prod_{j=a_b}^{t}q_{bj}(r_{bj}),
\]

且

\[
\nabla_\theta L_{\rm gain}
=
\nabla_\theta[-\operatorname{Mean}(G_b)/16].
\]

稿件正确限定：

- 这是reference prefix被安全约束保持条件下的incremental suffix utility；
- 不是无条件total EAL；
- fixed live support内piecewise exact；
- 跨TopK换位和support drop不声称全局可微；
- 不使用会改变目标的概率smoothing。

Proposition 1成立，且实现单测可直接用autograd比较两种梯度。

---

## 4. Immutable Reference、Identity与Hard-Harm Envelope

`a_b/y_bi/delta_b`由最初released DFlash reference离线冻结，joint DFlash之后不重算。protected gold使用其当前live Top16中的`r_live`；support drop直接`Hbar=1`并mask gain，因此live model不能通过缩短prefix或移除candidate来game约束。

step-0证明现已正确使用：

\[
M_b^{(0)}=-\delta_{\mathrm{live},b},
\]

以及train-corpus certificate：

\[
|\delta_{mathrm{live},b}-\delta_{mathrm{ref},b}|
\le e_{mathrm{num}}^{cert}.
\]

对stable train block，`delta_ref>delta_min>=2e_num_cert`推出`delta_live>delta_ref/2=gamma`，故step-0有`Hbar=0`。

阈值

\[
\delta_{min}
=
\max(2e_{m num}^{cert},2/64,2^{-14})
\]

同时处理`e_num=0`和tiny margin，并保证stable branch：

\[
2/\delta_b<2/\delta_{min}\le64.
\]

这是对constraint margin derivative的明确界；完整参数梯度仍由gradient clip 1控制，稿件没有过度声称理论收敛。

对任何stable block，若deterministic proposal破坏reference accepted prefix，则`M_b>=0`并有：

\[
H_b=1\le\operatorname{ReLU}(1+M_b/\gamma_b)=\bar H_b.
\]

ambiguous/support-drop取1，empty prefix取0，因此Proposition 2及任意非负prompt weighting下的`PromptMean(H)<=PromptMean(Hbar)`成立。

证书只覆盖90K train是正确的数据隔离选择。最终held-out numeric audit只能只读报告，不能改变训练；pointwise harm envelope本身不依赖该经验certificate。

---

## 5. Loss、Dual、Stop与Gradient Routing

当前训练规范已足够直接实现：

- `L_base`和`L_gain`均按block length 16归一化；
- risk保持block-level，不额外`/16`；
- batch先均匀采prompt、再均匀采该prompt的block；
- `L_base → DFlash only`；
- `L_gain/Hbar → PARC + live DFlash H/Z`；
- target、reference和TopK IDs stop-gradient；
- `lambda_0=c_0=0`；
- dual EMA、LR、projection、cap和primal gradient clip均冻结；
- 不做loss-weight、temperature、threshold或lambda grid。

所有stop均只依赖train：

1. `lambda=100`且train-batch EMA violation仍为正时停止；
2. 从step 20K后，连续四个固定train-audit窗口的support drop均大于1%，且第四个仍高于第一个的80%时停止。

5K validation只计算EAL/harm并按唯一规则选checkpoint，不进入dual、stop、launch gate或loss定义。该数据路由现在完全符合amendment。

---

## 6. 100K Main Run与Sealed Held-Out

### 数据切分

在任何新label生成前固定：

- 90,000 train prompts；
- 5,000 validation prompts；
- remainder约5,000 held-out prompts。

按prompt/domain分层，三者prompt-disjoint，并排除旧development/formal重叠。旧15-position cache明确禁止用于训练。

### 唯一科学训练

第一项科学run直接训练`global PARC D256/L2 + joint DFlash`：

- batch 8 blocks；
- 无gradient accumulation；
- 180,000 optimizer steps；
- 1.44M block draws；
- head LR`3e-4`、DFlash LR`1e-5`；
- warmup 2K、cosine到10%；
- AdamW、gradient clip 1。

不存在frozen-first、local-first、capacity、512、2K、25K或独立GPU smoke。

### Checkpoint selection

每10K steps只在固定validation上评估。在`PromptMean(H)<=1%`的checkpoint中最大化validation prompt-balanced EAL，完全tie取更早step；没有合格checkpoint即训练失败。

### 唯一held-out job

checkpoint选定后冻结全部weights/config/data/loss/architecture，再首次打开held-out。同一个正式job在相同prompt和evaluator中共同运行：

- released DFlash fixed/dynamic；
- released Domino fixed/dynamic；
- PARC fixed/dynamic。

binding gates为：

- fixed EAL至少`1.15x` same-job Domino；
- dynamic EAL至少`1.15x` same-job Domino；
- `PromptMean(H)<=1%`；
- chat/code/math在fixed和dynamic均不低于Domino。

任一失败立即关闭route。held-out结果不再触发1.42M、refresh、width/loss修改或第二次评估。这是严格、可证伪且符合用户最新合同的主裁决。

---

## 7. Controls与System Gate

matched-local已精确定义为除cross-position attention mask外完全相同，但只在主held-out成功后作为后验exploratory机制研究，并明确：

- 不复用已打开的主held-out；
- 只在validation或未来新封存数据上报告；
- 不支持本轮confirmatory global-visibility claim；
- 不延迟主效果裁决。

这诚实地牺牲即时归因强度，换取用户要求的main-goal-first与sealed held-out。由于论文主贡献是fixed-reference gain-constrained objective，而不是“发明global attention”，该处理可接受。

R3 fixed和dynamic均通过后才允许系统工作。系统阶段只能进行不改变weights、architecture、tokens或decision的kernel/fusion/static-buffer优化；D256/L1等模型变体也已删除。

cycle必要条件：

\[
\frac{T_{PARC}}{T_{Domino}}
\le
\frac{EAL_{PARC}+1}{1.15(EAL_{Domino}+1)}
\]

在给定最低EAL门处得到`0.98417`正确。最终authority仍是same-stack A40 SGLang paired ABBA bootstrap TPS ratio的95% CI lower`>=1.15`。

---

## 8. 参数与在线可行性

新增参数账本继续精确成立：

\[
1{,}310{,}720+8{,}192+1{,}536+65{,}536+512
+1{,}051{,}136+768
=2{,}438{,}400.
\]

约占537.427M DFlash的0.454%。BF16 projected lexical table：

\[
151936\times256\times2=77{,}791{,}232\text{ bytes}.
\]

D256/L2、256 nodes、FFN512和两层full attention具有可信A40路径；旧PGCF profile只作为可行性先验，最终完整head、table gather、base vocab GEMM、Top16和argmax均重新公平测量。无L4或其他capacity rescue。

---

## 9. Contribution与Novelty边界

主贡献已经保持单一：

> immutable-reference conditional accepted-gain，加上pointwise deterministic prefix-harm envelope下的constrained one-pass parallel policy improvement。

稿件没有再认领：

- full-node attention本身；
- Top16 carrier；
- KEEP gauge；
- joint training；
- broad draft-then-refine；
- diffusion或sequence search。

相对D-PACE，新增的是固定reference下“保已有prefix/只奖励新增suffix”的objective分解；相对旧safety hinge，新增的是block-max、reference-normalized pointwise envelope与prompt-level primal-dual约束。相对PGCF/JAPD/PCLD和Speculative Correction的边界也清楚。

最强归约仍然是D-PACE tail truncation加constrained blockwise hinge。它限制了纯理论新颖性，但方案已经把这个窄贡献推到最清晰、最小且可证伪的形式。若100K held-out强结果成立，它足以形成一篇聚焦的方法/推理系统论文；若结果不成立，当前stop rule会诚实关闭，而不是堆模块救援。

---

## 七维评分

| 维度 | 分数 | 评语 |
|---|---:|---|
| Problem Fidelity | **10** | 完整保留full16全局并行单链、1.15x EAL/TPS与主目标优先。 |
| Method Specificity | **10** | architecture、概率、loss、数值边界、routing、stop和数据流程均达到实现级。 |
| Contribution Quality | **8** | 单一、简洁、数学自洽；仍是窄objective-level贡献，强依赖结果。 |
| Frontier Leverage | **8** | noncausal full-action mixing与primal-dual使用自然，无现代primitive装饰。 |
| Feasibility | **9** | 2.438M head、既有carrier经验和明确100K recipe构成可信执行路径。 |
| Validation Focus | **10** | 一个正式main run、validation选择、一次sealed held-out及硬stop，最小且充分。 |
| Venue Readiness | **8** | proposal已ready；最终顶会强度取决于是否显著突破旧transfer上限。 |

**Weighted OVERALL：9.0/10**

\[
0.15(10)+0.25(10)+0.25(8)+0.15(8)+0.10(9)+0.05(10)+0.05(8)=9.0.
\]

没有低于7分的维度。

### GAP

没有项目提供的3好3坏人工anchors，因此使用绝对顶会方法标尺。此前的数学、架构和数据协议鸿沟均已关闭；当前唯一剩余gap是无法由proposal文字消除的经验贡献风险。PARC必须在一次干净的100K流程中证明fixed/dynamic EAL、domain和最终TPS硬门，否则其贡献会退化为D-PACE与safety hinge的增量重写。research-refine的职责是给出一个严谨、最小、可执行且不可事后game的falsifier；最新方案已经达到该标准。

---

## Blocking Issues

**NONE。**

以下仅是非blocking执行注意事项：

1. identity的`Hbar=0`证书表述保持限定在90K train certificate覆盖范围；held-out只报告实际deterministic harm。
2. experiment-plan必须继承authoritative contract和最新Success condition，绝不能从旧anchor恢复capacity-first阶段。
3. R4 controls保持exploratory，除非未来使用新封存数据形成独立confirmatory证据。

---

## Simplification Opportunities

**NONE。** 当前唯一joint main run、无fallback、后置exploratory controls和只读系统优化已经是最小充分路径。

## Modernization Opportunities

**NONE。** 不应加入diffusion、RL、router、search、causal state或第二次head；这些都会增加复杂度或直接drift。

## 最强拒稿理由

> 即使实现完全正确，PARC仍可能被归约为D-PACE suffix truncation加已有prefix hinge的block-max primal-dual重写；两个命题本身较直接，且旧100K global evidence只有约`+0.24 EAL`。若本次sealed held-out不能显著超过Domino并达到1.15x硬门，方法贡献不足以支撑顶会接收。

这是结果风险，不是当前proposal blocker；最新计划已经以唯一、干净且不可回调的实验直接裁决它。

## Drift Warning

**NONE。**

## 最终 Verdict

**READY — 9.0/10**

当前方案满足`overall>=9`、零blocking、无drift且只有一个主贡献。

**明确授权进入`experiment-plan`。** 后续计划必须原样继承：第一科学训练直接90K train；validation只选checkpoint；held-out仅在锁定后同一job首次运行三系统fixed+dynamic；held-out后绝不训练、扩数据、refresh或修改模型。
