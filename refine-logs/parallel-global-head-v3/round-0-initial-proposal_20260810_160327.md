# Research Proposal：PCLD-16 — Parallel Clean-Trajectory Latent Distillation

> 状态：ARIS research-refine Round 0 初稿；尚未授权实现或 GPU 主实验。  
> 方向：把离线 clean sequential teacher 的状态压进一次 full16 全局非因果 head；线上仍只输出一条 16-token chain。

## Problem Anchor

- **Bottom-line problem：** 在不改变 DFlash 一次并行生成整块这一核心优势的前提下，设计一个轻量 head，显著解决 accepted length 偏低的问题；固定与动态 EAL 都至少达到同作业 released Domino 的 `1.15x`，最终同栈 SGLang 端到端吞吐也至少达到 Domino 的 `1.15x`。
- **Must-solve bottleneck：** 当前 full16 disjoint development 上，DFlash base EAL 为 `6.0685131195`，released Domino 为 `7.2395529640`，目标为 `8.3254859086`，pure-base Top16 oracle 为 `10.90925656`。JAPD-v2 的 D256/H8/L2 global head 已证明延迟不是主瓶颈（完整 eager head 为 Domino 的 `0.9993x`，新增参数 `4.54M`），并能在 512-block capacity set 上达到 `99.8627%` candidate accuracy、`99.7033%` hard accuracy、`94.2090%` oracle-gap recovery，但仅 6 个 harmful blocks 就使 EAL/harm 硬门失败；在 512 个完整 prompt 的同集 full-fit 上仍只有 `5.5133%` recovery、`24.6136%` J2 和 `18.1641%` harm。PGCF/JAPD 两代 frozen-feature global selector 因此共同表明：Top16 覆盖、head 延迟和单块记忆容量都足够，真正未解决的是 frozen DFlash 表示对 clean parallel trajectory 的跨-prompt 可辨识性，以及平均 candidate loss 无法控制少数 early-prefix mistakes 的尾部风险。新方案必须在保持一次 full16 全局并行单链推理的同时改变这一信息/优化条件，而不能继续只扩 selector 或调同一 loss。
- **Non-goals：** 不做 Domino/GRU 式自回归，不做 selected-token feedback，不做串行 target seed/decode，不做 Jacobi 或任何迭代 refinement，不做 beam/tree/trie/forest/multipath，不让 Top16 变成路径维，不增加 ordinary verifier 之外的在线 target inference，不把 D512、同一 JAPD loss 调权或 schedule sweep 当作补救，也不在 accepted-length 主机制成立前投入 SGLang 小修小补。
- **Constraints：** 单次 head 必须同时消费完整 `[B,16,*]` DFlash online features；每个输出位置必须通过无 causal mask 的全局 mixer 看到全部16位；一次产生 `[B,16,16]` scores 并以一次逐位置 argmax 得到唯一 `[B,16]` 序列。Top16 只作每位置候选轴。在线只能复用正常 DFlash/target-prefix 特征，target clean continuation 只作离线监督；不得增加 draft/head/target 的位置循环或二次调用。新增在线参数原则上不超过 `10.75M`，训练时允许改变 DFlash 表示的方案必须在部署前合并、不得增加推理层数或 forward 次数，并以同 A40、同 BF16、同 batch/block、eager-to-eager 公平 profile 约束成本。
- **Success condition：** 在严格 disjoint prompt 上证明 full16 global head 的跨-prompt机制增益，fixed EAL 至少 `8.3254859086`，dynamic EAL 至少同作业 Domino 的 `1.15x`，三个域均不退化；随后同栈 SGLang A40 tokens/s 的 paired 95% CI 下界至少 `1.15x` Domino。任何架构不变量失败、仅 same-set capacity 成功、仅 oracle 成功、或使用串行/迭代/多路径结果，都不能替代该成功条件。

## 1. Evidence synthesis

### 1.1 已排除的简单解释

| 证据 | 结果 | 含义 |
|---|---:|---|
| pure-base Top16 oracle | 10.9093 | candidate availability 足够，但不证明可选择 |
| PGCF-16 D256 capacity | exact/near-exact same-set | full-node global head 有记忆容量 |
| PGCF-16 disjoint | 6.1028，global-local +0.0138 | 普通 frozen full-node scorer 不泛化 |
| JAPD D256 capacity | 99.86% candidate acc，EAL 11.4121 | 小集上 token selection 可拟合 |
| JAPD D256 full-fit 6 epochs | EAL 6.3862，recovery 5.51% | 当前 recipe 在更广同集上严重不足 |
| JAPD D256 complete eager | 0.9993x Domino | latency 不是科学失败原因 |
| rank16 DFlash LoRA | full-B16 -0.00085 EAL | 单纯 LoRA + frontier distill 无效 |
| 537M/588M full adaptation | +0.0018 / +0.0043 EAL | 单纯扩大 trainable scope 或 full-vocab KL 无效 |
| PLC parallel correction-code head | 最好约 6.0934 | 15位、mode-compressed、on-policy code imitation 不足 |

一个必须单独记录的 confound：JAPD capacity 用 8,000 updates / 512 records，
约等于 125 次样本暴露；full-fit 只有 1,518 updates / 4,045 effective records，
约 6 epochs，而且 EAL 到 step 1,500 仍在上升。故 full-fit 负结果足以关闭冻结
JAPD recipe，却不能被写成 frozen features 的信息论上限。v3 会用一次固定预算的
exposure diagnostic 区分 under-training 与机制失败，但该诊断不能复活 JAPD-v2。

### 1.2 文献给出的可用原则

- DFlash（arXiv:2602.06036）证明一次 block-parallel hidden 已能承载大量 future
  信息，并用 early-position weighting 适配 accepted-prefix utility；因此不应再加
  在线串行步骤。
- Domino（arXiv:2605.29707）说明 clean teacher-forced prefix state 比 self-generated
  prefix 更适合训练 acceptance-relevant conditional distribution；但其在线 GRU
  是本项目必须去除的串行部分。
- *Your LLM Knows the Future*（arXiv:2507.11851）表明把 parallel/MTP latent
  对齐到 detached autoregressive predictive latent 可显著改善多 token prediction；
  关键是 latent consistency，而不只是 hard token CE。
- PTP（ICLR 2026, arXiv:2512.21323）说明单次并行函数在原则上能表达 token 间
  联合依赖；本项目不采用其多样化采样或 quadratic expansion，只借用“把 sequence
  dependency 蒸馏进一次并行映射”的观点。
- GLAT/DA-Transformer 的 multi-modality 结论提醒：逐位置平均 CE 易混合不同序列
  mode；本方案不引入 DAG/path decoder，而用一个 clean trajectory latent 作为同一
  block 的 dense coordination target。

## 2. Method thesis

**PCLD-16 的核心假设：** frozen DFlash lattice 中存在足以决定 clean proposal 的
信息，但 hard rank/CE 只给每位一个稀疏标签，无法让一个小 global head 学到
“此前 clean tokens 会把 Domino/target predictive state 推到哪里”。把 frozen
Domino 在 clean gold prefix 上的 256-d correction sufficient statistic 同时蒸馏到
16 个全局 student states，再以 target prefix certificate 微调，可把顺序知识压进
一次并行 head，并显著降低 early-prefix 的少数灾难性错误。

这不是在线模仿 Domino 的 rollout。Domino GRU 只在**离线 label materialization**
中 teacher-force 一次；production graph 中没有 GRU、没有 selected token、没有
位置循环。

## 3. PCLD-16 online architecture

### 3.1 输入与唯一输出

一次 released DFlash forward 已有：

- `H in R[B,16,2560]`；
- full-vocabulary base logits `Z`，从同一 tensor 取 pure-base `Top16`
  candidate IDs `C` 和 logits `B`；
- anchor ID；
- frozen target token embedding rows `E[C]`；
- frozen released Domino correction output-basis rows `W_out[C]`。

head 一次返回 `S in R[B,16,16]`，随后一次 tensor argmax 返回唯一
`proposal in N[B,16]`。没有中间 token selection。

### 3.2 Candidate-specific full-block encoder

每个 `(i,k)` candidate node 编码：

[
x_{ik}=LN(W_h RMS(H_i)+W_e RMS(E[C_{ik}])
W_c RMS(W_{out}[C_{ik}])+W_s\phi(B_{i,:},k)+p_i+r_k).
]

首版固定 `d=256, heads=8, layers=2`。全部 `16x16=256` nodes 进入两层
**无 causal mask** 的 full-node self-attention。为保持 candidate comparison，
每层带同位置 bias，但不屏蔽任何远端位置。每个位置的 learned code query 随后
一次 cross-attend 全部 256 nodes，得到 `g_i`；所以任意位置的 correction code
都显式依赖完整 block，而不是只读 pooled block token。

[
hat\delta_i=W_\delta g_i\in\mathbb R^{256},qquad
\lambda_i=\sigma(w_g^Tg_i+b_g).
]

其中 `W_delta` 和 gate 最后一层 zero-init，step0 精确回退 pure DFlash。

### 3.3 Candidate-only correction basis

把 released Domino correction MLP 第一层拆成
`W_h^D H_i + W_s^D s_i`。PCLD 不预测 `s_i`，而直接预测其对 logits 足够的
256-d statistic `delta_i = W_s^D s_i`：

[
u_i=SiLU(W_h^D H_i+lambda_ihat\delta_i),
]
[
S_{ik}=B_{ik}+langle W_{out}^D[C_{ik}],u_iangle.
]

只 gather 16 个 `W_out` rows并做 batched `16x16x256` dot；不做 Domino 的
16 次 GRU update，也不做新的 full-vocabulary correction GEMM。active frozen basis
约 39.55M，与 released Domino 已有权重相同；新增 trainable student 预计 3--5M，
必须实现后精确计数。完整 active head 目标 `<45M`，约为 537.427M DFlash 的
8.4% 以下。

### 3.4 架构 hard invariants

1. `forward` 不接受 gold、target future hidden、selected IDs 或 GRU state。
2. 所有16个 `delta_i` 在一次 invocation 中同时产生。
3. attention mask 只能是 `None` / full visibility；任何 causal/local production
   scope hard fail。
4. Python/CUDA graph 中没有 position loop、head rerun 或 data-dependent iteration。
5. Top16 只作每位置 candidate axis；没有 path/beam dimension。
6. 最终只验证一个普通 `[B,16]` proposal。

## 4. Offline clean-trajectory teacher

对 canonical clean tokens `y_0...y_15`，离线把 frozen released Domino
`prefix_gru` teacher-force 在：

[
[anchor,y_0,ldots,y_{14}].
]

一次 batched teacher pass 产生预测每个 `y_i` 前的 clean state `s_i^*`，并存：

[
\delta_i^*=W_s^D s_i^*.
]

同时存 target 在完全相同 clean prefix geometry 上的 Top16 candidate logits和
top1。所有这些字段只进入 loss；production batch whitelist 明确拒绝它们。

与旧 PLC 的差异不是“换宽度”：旧 PLC 主要拟合 released **self-generated/on-policy**
correction code、只覆盖15 correction positions并先压成 modes；PCLD 使用 full16
candidate-specific nodes和**clean teacher-forced trajectory statistic**，其 teacher
正好对应 accepted-prefix 仍有 reward 的条件分布。

## 5. Training objective

### 5.1 Stage A：clean latent parallelization

在 clean support `i<h_b` 上：

[
L_{latent}=\frac1{|M|}\sum_{bi\in M}
SmoothL1(RMS(hat\delta_{bi}),RMS(\delta^*_{bi})).
]

另以 frozen Domino clean-teacher candidate distribution 做 Top16 KL。Stage A
只回答 student 是否能同时重建 clean sequential correction state；不以同集 token
accuracy声称 accepted-length成功。

### 5.2 Stage B：worst-prefix certificate

令 target gold rank 为 `r_{bi}`，final candidate margin：

[
m_{bi}=S_{bi,r_{bi}}-\max_{k\ne r_{bi}}S_{bik}.
]

对 strict clean support 的每个 block 使用：

[
L_{cert}^{(b)}
=\tau\log\sum_{i<h_b}\exp((\gamma-m_{bi})/\tau).
]

该项由当前 block 最弱的 early-prefix decision 主导，不像 position-micro CE 被大量
easy suffix 稀释，也不只盯第二个 frontier。base 正确位置和可修 base 错误位置都在
同一 certificate 中：前者要求 preserve，后者要求 repair。再加小权重 target
candidate KL；不再叠 JAPD 的 all-prefix + J2 package。

总目标：

[
L=L_{cert}+\alpha(t)L_{latent}+0.1L_{targetKL},
]

其中 `alpha(t)` 在前30% updates由1线性降到0.1，之后固定0.1。只有这一条固定
schedule；不做 loss-weight grid。

### 5.3 Gate calibration

`lambda_i` 是连续、一次并行输出的一部分，不做 autoregressive decision。
internal select 只允许冻结一个全局 gate bias offset，使 `harm<=1%`；offset 冻结后
不得在 diagnostic/final 上改。若为了低 harm 必须关掉绝大部分 repair并导致
recovery不足，则机制直接失败。

## 6. Minimal falsification sequence

### D0：一次 exposure audit（诊断，不复活 JAPD）

在现有 J011 4,045 effective records 上，把 exact D256 JAPD baseline 固定训练到
20,000 updates；其他数据、loss、optimizer不变。只回答：

- 当前6-epoch结果是否主要是 under-training；
- position-micro accuracy、EAL、harm是否仍严重解耦。

无论结果如何，exact JAPD-v2 M2仍关闭；不再做第二个 steps/LR/width点。

### P0：PCLD mechanics / latency

- CPU shape、full visibility、remote intervention、zero identity；
- exact full16 clean-state row alignment，尤其 position15；
- A40 complete eager：base vocab GEMM、Top16、两个 frozen-table gather、PCLD、
  candidate dot、argmax，全计时；
- 与同 hidden/base GEMM 的 released eager Domino比较；development gate
  `PCLD complete <=1.20x Domino`。

### P1：512-block capacity

固定 D256/H8/L2、8,000 updates。必须同时：

- clean-code cosine `>=0.98`；
- supported candidate accuracy `>=99%`；
- oracle-gap recovery `>=95%`；
- harm `<=1%`；
- strict multi-repair success `>=99%`。

失败先判定 index/optimization；只允许修实现 bug，不允许 D512 或 loss grid。

### P2：cross-fitted small-data mechanism gate

使用 R047 的 `1589 train / 199 internal-select / 199 untouched diagnostic`
prompt split；global PCLD 与 parameter-matched local-mask control 使用完全相同
data/order/updates。bias只在 select 冻结，diagnostic只开一次。

GO 必须全部满足：

- global EAL至少 `base +0.30`；
- global-local EAL `>=+0.15` 且 prompt bootstrap 95% CI lower `>0`；
- harm `<=1%`；
- oracle-gap recovery `>=20%`、strict multi-repair `>=40%`；
- chat/code/math均 `>=base`；
- latent ablation（去掉 `L_latent`、其余完全 matched）显著更差。

这些只是大数据许可，不是主结果。

### P3：full16 scale

只有 P2 GO 才把 OPB 25K prompts 重采为约200K个严格 full16 blocks。旧L15 cache
只定位 prompt，不训练。先25K；若 validation scaling curve单调且至少达到
`7.8`，扩到100K prompts，匹配历史 frozen-head数据多样性。若25K低于 `7.55`
或 global-local无信号，关闭 PCLD，不用更多数据掩盖机制失败。

主 fixed gate始终为 `8.3254859086`，不是 `7.8`。

### P4：claim-bearing fixed / dynamic / systems

冻结 architecture、weights、gate bias与checkpoint后：

1. untouched fixed prompts：PCLD EAL `>=1.15x` same-job Domino，三域不退；
2. new dynamic rollout：EAL `>=1.15x` dynamic Domino；
3. 同栈 SGLang A40 batch1 paired TPS ratio的95% CI lower `>=1.15`。

任一失败都不能由串行 target、iteration或multipath结果补。

## 7. Required baselines and ablations

只保留能回答机制的对照：

1. pure DFlash；
2. released Domino；
3. exact D256 JAPD exposure audit（diagnostic only）；
4. PCLD global；
5. parameter-matched local-mask PCLD；
6. PCLD without latent distillation。

不做宽度、head数、层数、loss权重的大网格。若 global 与 local无差异，full-block
claim失败；若 no-latent等价，clean-trajectory latent mechanism claim失败。

## 8. Novelty and failure interpretation

主贡献若成立，应限定为：

> 将 teacher-forced clean sequential correction sufficient statistics 蒸馏到一个
> candidate-specific full16 noncausal head，使16个位置在一次调用中共同预测并只输出
> 一条 proposal；用 worst-prefix certificate 把少数 early errors直接纳入训练。

不声称 attention、Top16、Domino basis、knowledge distillation或 margin loss各自新颖。

明确失败分支：

- D0提升、PCLD不提升：优化预算解释JAPD部分失败，但 latent机制无增量；不写新方法。
- P1过、P2不过：same-set memorization仍不能转移；关闭 frozen-feature PCLD family。
- P2过、P3不过：数据规模/分布不足或 latent teacher不具可扩展性；关闭，不转 causal/tree。
- fixed过、dynamic不过：on-policy shift；仅允许用同一 PCLD 重新收集并训练一次，
  architecture不变。
- efficacy过、TPS不过：只做 kernel/fusion优化；不能用减少验证正确性换速度。

## 9. Highest-risk assumptions

1. clean Domino correction statistic比 hard rank提供更可泛化的 dense target；
2. full16 lattice足以预测这个 clean statistic，而不需要在线 clean prefix；
3. frozen Domino output basis在pure-base Top16上仍是合适的低成本 lexical basis；
4. worst-prefix certificate能把harm压到1%而不把repair recall压没；
5. 25K--100K full16 prompt diversity能跨越旧2K/16K训练规模的 transfer gap。

P1/P2分别最便宜地证伪1--4；P3只在这些机制证据成立后检验5。
