# JAPD-16 Round 3 Fresh Re-Review

## Verdict

**READY**

七轴均达到 `>=9`，未发现仍未关闭的 blocking issue。READY只表示方案具备进入
`experiment-plan -> implementation -> gated experiments` 的条件，不代表JAPD已经
获得性能、可扩展性或论文claim。

## Seven-axis scores

| Axis | Score | Finding |
|---|---:|---|
| problem_fit | 10.0 | 完整保持full16 global non-causal one-call one-chain，直接攻击multi-position clean-prefix repair。 |
| specification | 9.4 | horizon、teacher、固定Z、joint certificate、prompt estimator、sidecar、capacity与gates达到实现级精度。 |
| comparative_clarity | 9.5 | 三臂隔离global visibility与JAPD objective；300/600/25K/100K角色与污染边界冻结。 |
| contribution_clarity | 9.1 | architecture明确复用，dominant contribution为AP distillation与conservative two-frontier certificate。 |
| frontier_awareness | 9.3 | target distribution仅离线使用，无online teacher或推理漂移。 |
| experimental_feasibility | 9.0 | D64起步、条件D256、stop gates与20–60 A40 GPU-hour路线可执行。 |
| venue_fit | 9.0 | 若matched/fresh/final gates成立，故事简单、机制明确且系统目标强。 |

Mean：`9.33/10`。

## Blocker closure

- normalized smooth-min：CLOSED；改为unnormalized `-logsumexp(-d)`。
- J2 endpoint：CLOSED；使用inclusive `0:e2+1`。
- historical J2 parity：CLOSED；strict horizon denominator/global/local/Domino
  为 `745/15/0/207`，旧`771/15/0/208`废止。
- prompt ratio bias：CLOSED；有效blocks加固定全局缩放。
- horizon-dependent normalizer：CLOSED；固定`Z=136`。
- sidecar target-feature ambiguity：CLOSED；明确为DFlash/shared vocab projection，
  并加入logsumexp/scalar/token replay parity。
- D64→D256 conflict：CLOSED；两个D64门同时失败才在fresh300前统一切换三臂。
- 25K decision：CLOSED；absolute/relative/domain/J2 hard gate已冻结。
- 100K contamination：CLOSED；排除fresh300/final600 IDs与near duplicates。
- 100K opening：CLOSED；使用`max(8.3254859086,1.15x same-job Domino)`。

## Mathematical verdict

对于

`M_b=-logsumexp_i(-d_bi)`，令`m=min_i d_bi`与`n=|P_b|`，有
`m-log(n) <= M_b <= m`。所以`M_b>0`是整段argmax正确的充分条件；任一margin非正
时certificate必非正。其梯度集中于最弱位置且不乘此前prefix probability。固定
`Z=136`的AP项保持all-prefix相对utility；prompt estimator
`N+/(B P+) sum_b L_b/|B_p+|`对精确prompt mean无偏。

## Immutable architecture audit

full16一次输入、global non-causal全16可见、一次输出`[B,16,16]`、一次argmax唯一
`[B,16]`链、Top16仅候选轴、无selected-token feedback、无GRU/causal/autoregressive、
无serial target、无iteration/Jacobi、无beam/tree/trie/forest/multipath、无额外
online target feature/forward、ordinary one-chain verifier不变：**全部PASS**。

Drift warning：**NONE**。

## Authorization boundary

授权进入experiment-plan，实现D64 JAPD、三臂、sidecar与parity tests，随后按顺序
运行mechanics/capacity/latency、small-data/fresh300、25K、100K/final600、SGLang。
任何后一阶段都只能在前一阶段全部hard gates通过后打开。

未授权在看到outcome后修改`T=2`、`0.9/0.1`、`Z=136`、J2、loss比例、support、
capacity/checkpoint rules；不得以D256 rescue transfer failure；不得提前读取final600；
不得用capacity/oracle/J2替代EAL；不得引入任何causal、GRU、serial、iterative、tree、
multipath或额外target-online路径。
