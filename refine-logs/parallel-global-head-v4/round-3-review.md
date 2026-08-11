# PARC-16 Round-3 Readiness Review

**CALIBRATION: none**

## 总体裁决

Round-2 refinement 已经关闭大部分实质性问题。当前方法确实是：

- 一次消费完整 full16；
- 256 个 edit-action node 无 causal mask 地全局交互；
- 一次输出 `[B,16,16]`；
- 一次逐位置 argmax 得到唯一 `[B,16]`；
- 无 token feedback、迭代 refinement、多路径搜索或额外在线 target forward。

Immutable reference 也已基本不能被 joint DFlash game：`a_b/y_bi/delta_b`冻结，protected gold 使用当前 live Top16 的`r_live`，support drop fail-closed。

不过，当前版本仍有三个阻止 READY 的规范缺口：

1. `q_bi`从未被正式定义；Proposition 1只有在明确规定`q_bi=softmax_k(A_bik)`及固定温度后才是完整命题。
2. FP32 reference与BF16 live replay下，step-0实际是`M_b=-delta_live,b`，不是文中声称的`M_b=-delta_b`。`delta_min=2e_num`可以在额外条件下推出`Hbar=0`，但目前512样本上的经验最大误差不是全数据数值证书，且`e_num=0`或极小时也不能证明所谓“stable gradient bound”在工程上可接受。
3. support-drop“不回落”及matched-local control仍没有可复现的精确定义；它们会影响停止和global-context主张。

这些都是可直接修订的规范问题，不需要新模块或扩实验菜单。方法已经值得进入实现准备，但尚不应直接授权`experiment-plan`。

**Verdict：REVISE**

---

## 1. Round-2 Blocker Closure

| Round-2 blocker | Round-3状态 | 判断 |
|---|---|---|
| protected score误用reference rank0 | **CLOSED** | 已明确使用gold在当前candidate set中的`r_live`；掉出support单独fail-closed。 |
| tiny margin与`delta_min` | **PARTIAL** | 有有限上界形式，但512-record经验误差不构成全数据证书，也未排除`e_num=0`或上界过大。 |
| ambiguous比例及先验不可行 | **CLOSED，依赖统一聚合** | `PromptMean(ambiguous)>1%`时停止是正确必要条件；所有报告必须使用同一prompt-balanced estimator。 |
| conditional/piecewise gradient措辞 | **MOSTLY CLOSED** | 已正确限定fixed support与conditional suffix；但`q=softmax(A)`缺失。 |
| support-drop / `lambda=100` stop | **PARTIAL** | cap violation stop已明确；“support-drop不回落”没有窗口、容忍度或评估频率。 |
| exact M1 recovery | **CLOSED** | 唯一公式为oracle-gap EAL recovery，未再用token/J2代替。 |
| 删除L4 | **CLOSED** | D256/L2冻结，无width rescue。 |
| 2K nonbinding / 25K binding | **CLOSED** | 2K只排机械故障，25K才首次科学裁决。 |
| joint/frozen recipe选择 | **CLOSED** | 选择只看select，tie选frozen；frozen胜出时删joint而非拒绝PARC。 |
| matched local规则 | **PARTIAL** | 决策逻辑正确，但“local”的可见性mask、训练预算和checkpoint规则未写死。 |
| system cycle公式 | **CLOSED** | 在每cycle产生`EAL+1` token的标准口径下代数正确。 |

---

## 2. Immutable Architecture Drift Audit

| 项目 | 结论 |
|---|---|
| full `[B,16,*]`一次输入 | PASS |
| 256 action nodes同时存在 | PASS |
| 所有位置无mask全局可见 | PASS |
| 一次head call | PASS |
| 一次`[B,16,16]`输出 | PASS |
| 一次argmax、唯一chain | PASS |
| selected-token feedback | NONE |
| causal/GRU rollout | NONE |
| Jacobi/iteration/second stage | NONE |
| beam/tree/trie/forest/multipath | NONE |
| extra online target forward | NONE |
| reference进入部署图 | NONE |

256-node Transformer的层深不是序列迭代；candidate轴也没有变成路径轴。

**Drift Warning：当前架构无drift。**  
但M5的“on-policy full16 data refresh”必须明确继续使用原始released immutable `pi_ref`，不得在refresh时把当前joint checkpoint重新定义为reference。否则会重新引入moving-base gaming。

---

## 3. Immutable Reference与`r_live`

这一部分已真正闭合。

对于任意protected位置，风险margin使用：

\[
m_{bi}
=
\max_{k:C_{bik}\ne y_{bi}}A_{bik}
-
A_{bi,r^{live}_{bi}}.
\]

因此joint DFlash不能通过以下方式缩小约束范围：

- 缩短自己的accepted prefix；
- 把reference gold从rank0移到其他rank；
- 重算更短的`a_b`；
- 让gold掉出Top16后mask该block。

support drop时`Hbar=1`，所以约束不会fail-open。该分支虽无constraint gradient，但已由`L_base`恢复support，并被视为可行性诊断，语义是正确的。

---

## 4. Conditional Gain Proposition

令

\[
q_{bi}(k)=\operatorname{softmax}_k(A_{bi:})_k
\]

且温度固定为1，则在candidate IDs、gold ranks和`h_b`固定的局部区域内：

\[
G_b=\sum_{t=a_b}^{h_b-1}\prod_{j=a_b}^{t}q_{bj}(r_{bj})
\]

以及detached权重

\[
w_{bi}
=
\operatorname{sg}
\left(
\sum_{t=i}^{h_b-1}
\prod_{j=a_b}^{t}q_{bj}(r_{bj})
\right)
\]

确实满足

\[
\nabla L_{\rm gain}
=
\nabla[-\operatorname{Mean}(G_b)/16].
\]

共享参数跨位置传播不会破坏该等式。

当前稿正确地不再声称：

- 跨TopK边界全局可微；
- 对无条件total EAL给出精确梯度；
- protected prefix本身由该概率乘积建模。

剩余问题只是规范中没有定义`q`。如果实现者选择对`d`而不是`A`做softmax、加入温度或重新归一化，命题会改变。

**最小修订：** 在architecture后立即写死`q=softmax(A)`、temperature=1、FP32 product/log-softmax，并将其加入M0梯度单测。

---

## 5. Harm Envelope、Identity与数值边界

### Pointwise upper bound

只要`gamma_b>0`：

\[
\bar H_b=\operatorname{ReLU}(1+M_b/\gamma_b)
\]

在任何`M_b\ge0`时均满足`\bar H_b>=1`，因此：

\[
H_b=\mathbf 1[M_b\ge0]\le\bar H_b.
\]

ambiguous和support-drop直接取1、空prefix取0，也保持逐block上界。这个命题是正确的，不依赖概率校准。

### Step-0 identity的当前错误表述

稿件同时使用：

- FP32 reference margin`delta_ref,b`；
- production BF16 live path。

因此step-0实际是：

\[
M_b^{(0)}=-\delta_{\rm live,b},
\]

而不一定是`-\delta_ref,b`。

若确有统一数值误差证书

\[
|\delta_{\rm live,b}-\delta_{\rm ref,b}|\le e_{\rm num}
\]

且

\[
\delta_{\rm ref,b}>2e_{\rm num},
\]

则

\[
\delta_{\rm live,b}>\delta_{\rm ref,b}/2=\gamma_b,
\]

从而仍能推出`\bar H_b=0`。结论可救，但证明必须按这个不等式写，不能继续声称两个margin相等。

### “Stable gradient bound”尚未完全闭合

stable branch上确有：

\[
\left|\partial\bar H/\partial M\right|
=2/\delta_b
<2/\delta_{\min}.
\]

但这只说明有限，不自动说明稳定：

- `e_num`可能测得为0；
- `e_num`可能很小，使`2/delta_min`仍极大；
- 512-record上的最大误差不能自动上界25K/100K所有输入的BF16 replay误差。

**最小修订：**

\[
\delta_{\min}
=
\max(2e_{\rm num}^{cert},\,2/g_{\max},\,\epsilon_{\rm positive}),
\]

其中`g_max`是训练前冻结的最大允许constraint logit-gradient，不做grid；`e_num^{cert}`必须来自覆盖实际算子路径的保守证书或逐数据集preflight，而不只是把512样本最大值称为全局上界。launch前报告`e_num`、`delta_min`、`2/delta_min`、ambiguous PromptMean。

这不会削弱`H<=Hbar`，只是把“identity与稳定性”从经验希望变成可检验合同。

---

## 6. Loss、Dual与Gradient Routing

已闭合的部分：

- `L_base`、`L_gain`均按16归一化；
- block风险本身不应再除16；
- 两阶段prompt/block sampling给出prompt-balanced estimator；
- `L_base → DFlash`；
- gain与stable constraint经`H/Z → PARC + DFlash`；
- reference、target和TopK IDs stop-gradient；
- `lambda_0=0`、`c_0=0`、EMA、projection与cap明确；
- `lambda=100`且EMA violation仍正时判infeasible。

剩余问题是：

> “support-drop rate不回落”不是可执行stop rule。

必须写死evaluation interval、warm-up和patience，例如连续`K`个固定评估窗口不下降且高于预注册阈值才停止。否则不同实现者可以任意提前或延后关闭joint arm。

此外，M1/M2/M3中的“actual harmed-block fraction”应统一写成`PromptMean(H)`；不要一处按block加权、一处按prompt加权。

---

## 7. 参数账本与在线成本

参数算术自洽：

\[
1{,}310{,}720+8{,}192+1{,}536+65{,}536+512
+1{,}051{,}136+768
=2{,}438{,}400.
\]

BF16 lexical table也正确：

\[
151936\times256\times2
=77{,}791{,}232\text{ bytes},
\]

即77.79 MB或约74.19 MiB。

精确参数claim依赖以下实现合同，稿件应维持：

- `RMS(H/E)`是parameter-free；
- `W_h/W_e/W_c`无bias；
- QKV/out/FFN无bias；
- 两个pre-norm affine LayerNorm；
- relative-position和same-position bias合计每层256参数；
- scorer无bias。

256 nodes、D256、L2、FFN512在A40上是可信的轻量规模，且旧PGCF profile提供了可行性证据。但77.8 MB gather、delta-node构造和joint checkpoint仍必须按complete path重测；旧profile不能转写成P​ARC latency结果。

L4已正确删除。最终TPS失败后允许一次L1 deletion属于固定语义下的系统简化，不是方法容量救援。

---

## 8. Validation与Stop Rules

### M1

oracle-gap recovery公式已经唯一且正确：

\[
\frac{EAL_{\rm PARC}-EAL_{\rm ref}}
{EAL_{\rm Top16\ oracle}-EAL_{\rm ref}}\ge95\%.
\]

应补一句oracle和harm均使用同一immutable-reference block authority，并把harm明确为`PromptMean(H)`。complete-prefix/J2和token accuracy只作诊断，不能替代该门。

### M2/M3

2K nonbinding、25K binding的修订正确，避免把已知low-data不足误判为方法失败。

joint/frozen选择也正确：

- 只在select上选recipe；
- tie选frozen；
- heldout只开一次；
- frozen匹配或胜joint时删除joint；
- joint不再是论文贡献。

matched local仍需写死为：

- 相同256 nodes、D256/L2和参数；
- 相同inputs、loss、optimizer、训练预算和checkpoint规则；
- 唯一差异是禁止跨position attention，仅保留同一position的16 candidate互见；
- joint global对应joint local，frozen global对应frozen local。

否则`global-local>=0.15`无法唯一归因于global visibility。

### System formula

在每个speculative cycle平均产生`EAL+1` token的口径下：

\[
\frac{T_{\rm PARC}}{T_{\rm Domino}}
\le
\frac{EAL_{\rm PARC}+1}
{1.15(EAL_{\rm Domino}+1)}
\]

是正确的必要条件。在最低EAL门处得到约`0.98417`也正确。`T`必须定义为相同batch、调度和完整verifier路径下的长期平均cycle time；最终paired SGLang TPS仍是唯一系统裁决，不能以该公式替代实测。

---

## 9. Novelty Boundary

主贡献现在足够单一：

> immutable-reference conditional accepted-gain，加上pointwise deterministic prefix-harm envelope下的constrained policy improvement。

它没有再错误认领full-node attention、KEEP gauge、joint training或draft correction本身。

相对边界也基本准确：

- 相对D-PACE：从total accepted-length allocation改为fixed-reference conditional suffix gain；
- 相对旧GCLS/PGCF safety hinge：从逐位置固定权重hinge改为block-max、reference-normalized、pointwise envelope与prompt-level constraint；
- 相对JAPD/PCLD：不依赖teacher sidecar或filtered support；
- 相对Speculative Correction：不主张广义noisy-sequence refinement，只保留strict one-pass Top16 lossless single-chain setting。

但顶会最强归约仍然成立：

> 这是D-PACE tail truncation与既有prefix hinge的blockwise primal-dual重写，而不是新的inference architecture。

两个proposition都正确但较初等，无法单独提供顶会贡献强度。只有在25K及后续disjoint结果显著突破旧global-head约`+0.24 EAL`上限时，这个窄objective才可能升格为实质贡献。

---

## 七维评分

| 维度 | 分数 | 评语 |
|---|---:|---|
| Problem Fidelity | **10** | 问题、线上合同、硬门与非目标完全保真。 |
| Method Specificity | **8** | 主体可实现；`q`、数值证书和stop/local细节仍缺。 |
| Contribution Quality | **7** | 单一且连贯，但仍是窄objective-level贡献，强依赖结果。 |
| Frontier Leverage | **8** | 正确吸收constrained optimization并诚实处理Speculative Correction边界。 |
| Feasibility | **8** | 参数与carrier可信；数值梯度和complete-path latency待preflight。 |
| Validation Focus | **8** | 2K/25K及recipe deletion合理；matched local和停止窗口需写死。 |
| Venue Readiness | **7** | 已接近可执行方法计划，但离无blocker顶会ready仍有距离。 |

**Weighted OVERALL：8.0/10**

\[
0.15(10)+0.25(8)+0.25(7)+0.15(8)+0.10(8)+0.05(8)+0.05(7)=8.0.
\]

没有低于7分的维度，因此无强制的“<7维度”修订项；下列blockers仍阻止READY。

### GAP

没有项目提供的3好3坏人工anchors。相对绝对顶会方法标尺，P​ARC已经从错误风险proxy和moving baseline，收敛为一个可证、可实现、可证伪的安全增量策略目标。距离READY的差距不是再加模块，而是把混合精度identity/gradient合同、概率定义和control/stop细节写成无歧义规范。更大的学术差距只能由disjoint accepted-length结果弥合，不能靠形式包装弥合。

---

## Blocking Issues

1. **CRITICAL：** 明确定义`q=softmax(A)`、temperature和FP32计算路径，并加入detached-gradient数值单测。
2. **CRITICAL：** 修正step-0证明为`M=-delta_live`；为`e_num`给出覆盖实际路径的证书范围，并处理`e_num=0`及`2/delta_min`过大的情况。
3. **IMPORTANT：** 将support-drop“不回落”改成有固定窗口、阈值和patience的可执行stop rule。
4. **IMPORTANT：** 精确定义matched local mask、训练预算与checkpoint选择；global/local除visibility外不得变化。
5. **IMPORTANT：** 所有binding harm统一为`PromptMean(H)`；on-policy refresh明确不得更新immutable reference。

---

## Simplification Opportunities

- 保持D256/L2，不恢复L4。
- frozen不差于joint时直接删除joint及其论文叙事。
- M2只保留一个nonbinding smoke；不新增loss、threshold或width菜单。
- constraint deletion只在主结果通过后执行。
- relative bias、compatibility分支和full-node carrier均作为继承实现，不为它们安排独立novelty claims。

## Modernization Opportunities

**NONE。** 当前fixed-reference primal-dual formulation已经足够现代。不要加入diffusion、RL、differentiable search、router、第二次head或任何迭代primitive。

## 最强拒稿理由

> PARC的安全语义已经比旧global-head loss严谨，但方法本质仍可归约为D-PACE tail-sum截断，加上已有prefix hinge的block-max、margin-normalized primal-dual版本；两个命题是直接的product-rule和indicator-envelope结果。稿件尚未证明这一objective改写能解决PGCF/JAPD/PCLD暴露的跨prompt可辨识性与transfer失败，而且当前混合精度identity证书、概率定义和local control还没有完全闭合。

## 可保留部分

- immutable released-DFlash reference；
- live `r_live`与support-drop fail-closed；
- full256 global edit-action carrier；
- D256/L2与2,438,400参数账本；
- conditional incremental-gain梯度；
- pointwise deterministic-harm envelope；
- prompt-balanced primal-dual训练；
- 2K nonbinding、25K first binding；
- best joint/frozen recipe deletion逻辑；
- fixed→dynamic→same-stack SGLang硬门；
- 正确的cycle-time必要条件。

## 最终 Verdict

**REVISE — 8.0/10**

Round-2的主要方法错误已经关闭，但上述规范blockers尚未全部消除，因此本轮不满足`overall>=9且无blocking issue`的READY条件。

**当前不授权进入`experiment-plan`。** 只需完成上述最小规范修订后再做一次短readiness audit；不要改变架构，也不要扩展实验菜单。
