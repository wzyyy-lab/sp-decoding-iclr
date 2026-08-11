# JAPD-16 Round 2 Fresh Re-Review

## 结论先行

**VERDICT: REVISE**

相较 Round 1，方案已经显著收敛：problem anchor 完整保留，axial head 被正确降为
复用载体，gold-conditioned dropout 已删除，贡献集中为一个 offline objective，
三臂与 fresh/final split 也基本形成可辨识证据链。

当前仍有三个 blocking mathematical/specification issues：

1. normalized smooth-min 会把单个错误 margin 按 `|P_b|` 稀释，不能支持“最弱边界”的声明；
2. `J2_b` 的 `0:e2` 写法按通常切片语义不包含第二处错误；
3. 当前 minibatch 内按随机权重和归一化的实现并不精确等于 prompt-balanced mean，
   且 `h_b=0` 后使用总 block 数作为权重会使 prompt 权重失衡。

这些都可通过局部公式和协议修改解决，不需要改变方法方向。

## 七轴独立评分

| 维度 | 得分 | 理由 |
|---|---:|---|
| `problem_fit` | 10.0 | 完整攻击 full16 多位置 clean-prefix 修复；固定、动态 EAL 与同栈 TPS 双门均未降级。 |
| `specification` | 7.5 | 数据流、optimizer、support、teacher 与实验协议已很具体，但 smooth-min、J2 端点和 prompt mean 有实质公式错误；D64→D256 条件也前后不一致。 |
| `comparative_clarity` | 8.5 | `global-JAPD / local-JAPD / global-D-PACE` 是干净的最小三臂；fresh300/untouched600 明显优于旧方案。仍需冻结 capacity boolean、25K 决策角色及同作业相对门槛。 |
| `contribution_clarity` | 8.5 | 已不再重新包装 GCLS architecture，主贡献明确为 JAPD。当前 AP 与 J2 的关系仍略像两个标准 loss 的组合，必须依靠正确的 joint certificate 与 matched result 才能形成机制贡献。 |
| `frontier_awareness` | 9.0 | offline candidate-distribution distillation 使用自然，target 严格限于标签，无强行加入 RL、迭代 denoising 或在线 teacher。 |
| `experimental_feasibility` | 8.0 | D64 起步、100K/三 seed、300→600 的成本与执行顺序可行；门槛非常激进但可作为诚实 falsifier。容量分支和25K stop rule尚未完全冻结。 |
| `venue_fit` | 8.0 | 若 joint objective 的数学定义修正且最终全部硬门成立，故事足够集中；在此之前仍可能被审稿人视为 weighted CE + structured margin 的增量组合。 |

**七轴均值：8.5/10。** READY 规则要求七轴全部 `>=9` 且无 blocker，当前未满足。

## Invariant Audit

| 不变量 | 结论 | 审查 |
|---|---|---|
| full16 一次输入 | PASS | 单次读取 `H/base Top16/embeddings[B,16,*]`。 |
| global non-causal，每位置见全16 | PASS | global axial arm 中每个 candidate query 全部16个 position summaries，无 causal mask。 |
| 一次并行输出全部16位 | PASS | 单次产生 `[B,16,16]` scores。 |
| 唯一一条输出链 | PASS | 每位置一次 argmax，得到唯一 `[B,16]`。 |
| Top-16 仅候选轴 | PASS | 没有 path/sequence candidate 轴。 |
| 无 selected-token feedback | PASS | 没有中间 argmax 或 token embedding 回馈。 |
| 无 GRU/autoregression/serial target | PASS | 在线 head 是一张并行图。 |
| 无 iteration/beam/tree/trie/forest/multipath | PASS | 方法与验证路径均无这些结构。 |
| 无额外在线 target inference | PASS | teacher token/logits 仅离线监督；`base_logsumexp` 来自 DFlash 已有 base logits。 |
| 仅在线可用特征 | PASS，需术语修订 | `base_logsumexp` 是同一 DFlash full-vocab tensor 的 reduction，不是 target hidden/logit feature。 |
| 轻量 | PASS | 已核验 D64/H4/L1/full16 配置确为 `433,852` trainable parameters。 |
| 数据隔离 | PARTIAL | 300/600 设计正确，但必须明确100K train 也排除二者的 IDs/near duplicates。 |
| 公平成本比较 | PLANNED | reduction 与 sidecar replay 成本须计入 complete profile。 |

**架构 Drift Warning：NONE。**

## JAPD 数学审查

### Clean/support horizon 与 soft teacher

`h_b=min{i:r_bi<0 or not g_bi}` 作为 clean-prefix support 合理；`h_b=0` 不训练可
避免伪造可实现标签。需明确 `y_bi == gold_ids_bi`。候选条件 soft teacher 设计
合规，但只有其 `0.9 delta` hard component 可解释为 clean-prefix NLL；`0.1` soft
component 是 candidate-conditional distillation regularizer。

### All-prefix normalizer

代数展开
`sum_{m<h} sum_{i<=m} CE_i = sum_{i<h}(h-i)CE_i` 正确，但逐 block 使用
`Z_b=h_b(h_b+1)/2` 会让 `h=1` 与 `h=16` block 总权重相同，弱化长 clean horizon
的 accepted-prefix utility。建议用固定 full16 normalizer `Z=136`；若保留 `Z_b`，
则只能称为 horizon-normalized credit allocation。

### Joint smooth-min

当前 normalized mean 会稀释单点错误。例如 `n=16`、一个 margin `-1`、其余15个
为 `5` 时，当前 `M≈1.74`，存在错误 token 却得到正 certificate。最小修复：

`M_joint = -log sum_{i in P} exp(-d_i)`，
`L_J2 = softplus(-M_joint)`。

等价地可保留 normalized `M` 而使用 `softplus(log|P|-M)`。此外，若贡献明确为
two-frontier，应只在 `|E_b|>=2` 时激活 joint term；0或1处错误由 `L_AP` 负责。

### J2 metric endpoint

`0:e2` 不包含第二处错误。必须改成
`J2_b = 1[forall i<=e2, yhat_bi=y_bi]`，代码使用 `0:e2+1`，并在新实验前做历史
计数 parity audit。

### Prompt-balanced mean

随机 minibatch 内以权重和作分母是 ratio estimator，并不等于全局 prompt mean；
`h=0` block 还会导致 prompt 总权重失衡。精确定义应为：

`L = 1/|P+| sum_p 1/|B_p+| sum_{b in B_p+} L_b`，
`B_p+={b:h_b>0}`。

可用 prompt-uniform sampling，再从有效 blocks 均匀抽样；或 block shuffle 配固定
全局缩放，禁止随机 batch-weight denominator。

## Sidecar、公平性与实验协议审查

`base_logsumexp` sidecar 总体合规，不是额外 target inference。需要把“target
LM-head”改成“DFlash 在线实际使用的共享 frozen vocabulary projection”，增加
sidecar-vs-online `base_logsumexp`、scalar-channel、selected-token parity，并把FP32
reduction计入complete latency。

D64→D256 条件前后冲突，必须冻结唯一 AND boolean：只有 D64 同时失败 same-set
capacity 与预注册 internal-fit optimization gate，且在读取 fresh300 前决定，才可
让全部三臂统一升级 D256。三臂本身设计干净且充分，无需第四个主臂。

后续25K/100K manifests必须排除 fresh300、untouched600 的 IDs/near duplicates。
25K必须明确为无决策 diagnostic，或预注册精确 stop threshold。100K internal
advancement 应要求
`EAL_JAPD >= max(8.3254859086, 1.15 * EAL_Domino,same-job)`。

## Blocking Issues

1. normalized smooth-min 不能保证最弱 margin。
2. `J2` 指标漏掉第二处错误。
3. prompt-balanced minibatch estimator 不等于声明目标。
4. `Z_b` 改变 all-prefix estimand。
5. D64→D256 的 AND/OR 条件冲突。
6. 100K 对300/600的排除未冻结。
7. 25K fail-fast没有预注册决策门。

## 最小修改清单

1. J2改为unnormalized soft-min，并只对`|E_b|>=2`激活。
2. J2 metric改成inclusive `0:e2+1`并做reference parity。
3. AP normalizer改为固定136。
4. 用有效block定义精确prompt mean，禁止随机batch分母。
5. 修订sidecar术语并增加logsumexp/scalar/token replay parity。
6. 冻结D256 AND条件，三臂统一容量且fresh300前决定。
7. 明确25K hard gate并排除100K与300/600 overlap。
8. 100K advancement同时使用absolute与same-job relative门。

完成这些局部修改后可进入下一轮；无需增加任何 serial、tree、multipath 或 iterative
组件。
