# PARC-16 grounding and route decision

## 已有本地证据

1. DFlash 的 pure-base Top16 oracle 足够高，但 availability 不等于
   selectability；部署目标需要回收 base-to-oracle gap 的约 `46.62%`。
2. 旧 L15 axial GCLS 在约 100K prompts 上曾给出 global-local
   `+0.17383` 的真实信号，但 absolute gain 只有 `+0.24247`，说明完整 block
   信息有用但提取不足。
3. PGCF-16 证明 2.44M full-node global head 有 same-set capacity 和低延迟，
   但 disjoint EAL 仅 `6.10277`。它的 base-residual direct classifier 与
   soft-prefix objective 没有转移。
4. JAPD-D256 证明扩大同类 selector 并提高平均 candidate accuracy 不足以控制
   accepted-prefix tail：capacity 接近门，但 broader same-set recovery 只有
   `5.51%`，harm `18.16%`。
5. PCLD-16R 证明 target-hidden supervision 能完全拟合过滤后的 stable support，
   但完整 early-error population 只有 `78.35%` J2，harm `6.25%`。这把问题
   定位到完整前缀监督与编辑风险，而不是 hidden regression 本身。
6. rank-16 LoRA、588M full adaptation 和 full-vocabulary KL 都只有接近零的
   disjoint 增益。单独扩大可训练范围或重复 target posterior distillation 不再是
   合理主线。

## 当前文献边界

- [DFlash](https://arxiv.org/abs/2602.06036) 已建立一次 forward 的 blockwise
  parallel drafter；PARC 不能把 parallel block drafting 当作贡献。
- [Domino](https://arxiv.org/abs/2605.29707) 用 prefix-dependent causal head
  补 intra-block dependency，并用 base-anchored curriculum 联合训练 backbone；
  PARC 必须明确区别于它的 token-by-token prefix feedback。
- [D-PACE](https://arxiv.org/abs/2605.18810) 已从 expected accepted length
  推导动态位置权重；PARC 不能声称首次优化 acceptance surrogate。它的新问题是
  selector 相对 base 的 edit/KEEP 风险约束。
- [Parallel Token Prediction](https://arxiv.org/abs/2512.21323) 证明一次调用
  可以联合表示 dependent future tokens；PARC 不声称 parallel dependency 的一般
  可表示性，而是在 DFlash Top16 单链接口下做轻量 deterministic correction。
- [SpecFormer](https://arxiv.org/abs/2511.20340) 已使用 bidirectional attention
  做 non-autoregressive forecasting；PARC 不能把双向 attention 本身当创新。
- DSpark/DeepSpec 的 Markov/RNN 路线仍依赖 previous selected token，违反当前
  immutable contract，只能作为效果边界而不是可采用机制。

## 两条候选路线

### Route A：冻结 DFlash 的更强 global selector

仅把 PCLD/JAPD 换成 position-level transformer，再加 harm loss。优点是实现小，
但 PGCF、JAPD、PCLD 与旧 100K GCLS 已多次显示 frozen observable transfer 很弱。
它很可能只改善 capacity，不足以补 `+2.257` EAL。

### Route B：KEEP-relative、risk-constrained one-shot block corrector，并与 DFlash 联合训练

把 DFlash Top1 明确当作 provisional noisy sequence；candidate 0 是 KEEP，其他
15 个 candidate 是相对编辑动作。16-position non-causal head 一次读取整条 provisional
sequence 与每位置 soft Top16 summary，同时产生全部 edit advantages。训练同时：

1. 用 D-PACE 保护并提升 base drafter；
2. 用 head-side dynamic prefix loss提高完整 accepted prefix；
3. 用 primal-dual constraint把破坏 base 已接受前缀的概率限制到 `1%`；
4. 联合更新已有 DFlash backbone，使其表示为这个并行纠错任务暴露所需信息。

联合训练不增加线上层数或 forward 次数；新在线组件仍只有一个轻量 head。

## 决策

选择 Route B。Route A 保留为 frozen diagnostic，不是主方法。选择依据不是“更大
模型更好”，而是 Route B 同时改变了此前失败的两个关键条件：编辑决策的风险语义，
以及 DFlash representation 与 global corrector 的协同训练。它仍严格满足 full16、
global non-causal、one-call、one-chain、Top16-only、无 selected-token feedback、
无 extra target inference 的全部在线合同。

