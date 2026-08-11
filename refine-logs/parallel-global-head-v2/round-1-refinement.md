# Round 1 Refinement：JAPD-16 Joint Acceptance-Prefix Distillation

## Problem Anchor

- **Bottom-line problem：** 在不改变 DFlash 一次并行生成整块这一核心优势的前提下，设计一个轻量 head，显著解决 accepted length 偏低的问题；固定与动态 EAL 都至少达到同作业 released Domino 的 `1.15x`，最终同栈 SGLang 端到端吞吐也至少达到 Domino 的 `1.15x`。
- **Must-solve bottleneck：** DFlash base Top-1 在当前 full16 disjoint development 上为 `6.0685131195`，Domino 为 `7.2395529640`，目标为 `8.3254859086`。PGCF-v1 虽能同集拟合 Top-16 oracle、且延迟足够低，但 held-out 仅为 `6.1027696793`；它把大量修改浪费在首拒之后，并只修复 `46/946` 个可修首拒。完美只修一次的 oracle 也仅为 `7.4985422741`，而完美修正前两次错误可达 `8.4238338192`。因此新方法必须利用完整16位全局信息，在一次并行输出中学会多位置、相互一致的 clean-prefix 修复，而不是只做首拒 gate 或单点修补。
- **Non-goals：** 不做 Domino/GRU 式自回归，不做 selected-token feedback，不做串行 target seed/decode，不做 Jacobi 或任何迭代 refinement，不做 beam/tree/trie/forest/multipath，不让 Top-16 变成路径维，不增加 ordinary verifier 之外的在线 target inference，也不在 accepted-length 主机制成立前投入 SGLang 小修小补。
- **Constraints：** 单次 head 必须同时消费完整 `[B,16,*]` DFlash online features；每个输出位置必须通过无 causal mask 的全局 mixer 看到全部16位；一次产生 `[B,16,16]` scores 并以一次逐位置 argmax 得到唯一 `[B,16]` 序列。Top-16 只作每位置候选轴。训练/选择/held-out prompt 必须隔离，target 信息只作离线标签。新增参数先控制在 `10.75M` 内，并以同 A40、同 BF16、同 batch/block、eager-to-eager 公平 profile 约束成本。
- **Success condition：** 一个未经 target-feature 泄漏、在新 disjoint held-out 上成立的 full16 global-vs-local 机制信号；固定 EAL 至少 `8.3254859086`、动态 EAL 至少 `1.15x` Domino，三个域不退化；随后同栈 SGLang A40 tokens/s 的 paired 95% CI 下界至少 `1.15x` Domino。任何架构不变量失败均为 hard NO-GO，不能用 oracle、same-set capacity 或 off-spec 系统结果替代。

## Anchor Check

- **Original bottleneck：** 一个 full16 block 往往要同时修好不止一处 token；v1
  expected-prefix-product 在第二处 correction 上发生严重 gradient starvation。
- **Why revised method still addresses it：** 推理仍是 full16 global non-causal
  one-call one-chain；唯一改变是把训练目标直接改成“所有可达 prefix 的非消失监督
  + 前两处 correction 之前的联合最弱边界”。
- **Reviewer suggestions rejected as drift：** 不接受任何 causal/serial/iterative/
  multipath 建议；本轮 reviewer 没提出这些。Reviewer 建议为 manifest 设 hash gate，
  但用户明确要求不要把精力浪费在形式化 hash 检查。本轮采用 append-only prompt
  manifest、sample-ID/text overlap exclusion 和采 label 前冻结，不设置运行时 hash 门。

## Simplicity Check

- **Dominant contribution after revision：** `JAPD`，一个为并行 speculative
  accepted prefix 设计的 joint distillation objective。
- **Components removed/merged：** 删除 gold-conditioned evidence dropout；删除
  axial architecture 的新颖性声明；删除 D256 默认配置；不再并列包装 architecture、
  augmentation 与 curriculum 三个贡献。
- **Unnecessary complexity rejected：** 不加 RL、denoising loop、router、confidence
  gate 或额外网络。
- **Why smallest adequate：** 复用已有、已测过 global visibility 的 axial direct
  selector；默认 head 仅433,852参数。JAPD 只改变离线 loss，在线图完全不变。

## Changes Made

### 1. 将 architecture 从“新贡献”降为复用的合规载体

- **Reviewer said：** 初稿 axial exchange 与现有 `GlobalDirectCandidateSelector`
  基本相同。
- **Action：** 接受。方法改名 `JAPD-16`；architecture 明确写成 reused axial
  full16 global head。默认从历史证据支持的 D64/H4/L1 开始。
- **Impact：** 贡献变成一个可严格 matched 的 objective claim，避免换名包装。

### 2. 删除 oracle-conditioned corruption

- **Reviewer said：** 用 gold 选择 mask 位置泄漏 error-location bit。
- **Action：** 完全删除 evidence dropout，不保留 label-independent 版本。
- **Impact：** 训练与推理输入一致；新旧 objective 成为唯一实验因子。

### 3. 增加真正联合的 two-frontier bottleneck term

- **Reviewer said：** all-prefix weighted CE 可分离，不能直接支持“两处同时修好”。
- **Action：** 保留 non-vanishing all-prefix distillation，并新增一个通过 smooth
  prefix-min margin 耦合所有位置的 `L_joint`；两个部分合称单一 JAPD objective。
- **Impact：** loss 和 hard metric 都直接要求同一输出在第二处 base error 之前
  全部正确，不再用两个边际 recall 冒充 joint repair。

### 4. 修复 heldout freshness 与 scale 口径

- **Reviewer said：** 历史 validation_gate 不新鲜；25K prompts 小于旧99K负证据。
- **Action：** 旧 validation_select/gate 都只作 diagnostic。新建300-prompt
  mechanism gate和600-prompt untouched final manifest，二者从所有历史 manifest
  外按 domain 平衡选取并在采 label 前冻结。scale-up 至少100K prompts；25K只作
  fail-fast curve point。
- **Impact：** 小门回答机制，100K回答可扩展性，600-prompt set只在最终冻结后开。

## Revised Proposal

# Research Proposal：JAPD-16 — Joint Acceptance-Prefix Distillation for a Full-Block Global Head

## Problem Anchor

- **Bottom-line problem：** 在不改变 DFlash 一次并行生成整块这一核心优势的前提下，设计一个轻量 head，显著解决 accepted length 偏低的问题；固定与动态 EAL 都至少达到同作业 released Domino 的 `1.15x`，最终同栈 SGLang 端到端吞吐也至少达到 Domino 的 `1.15x`。
- **Must-solve bottleneck：** DFlash base Top-1 在当前 full16 disjoint development 上为 `6.0685131195`，Domino 为 `7.2395529640`，目标为 `8.3254859086`。PGCF-v1 虽能同集拟合 Top-16 oracle、且延迟足够低，但 held-out 仅为 `6.1027696793`；它把大量修改浪费在首拒之后，并只修复 `46/946` 个可修首拒。完美只修一次的 oracle 也仅为 `7.4985422741`，而完美修正前两次错误可达 `8.4238338192`。因此新方法必须利用完整16位全局信息，在一次并行输出中学会多位置、相互一致的 clean-prefix 修复，而不是只做首拒 gate 或单点修补。
- **Non-goals：** 不做 Domino/GRU 式自回归，不做 selected-token feedback，不做串行 target seed/decode，不做 Jacobi 或任何迭代 refinement，不做 beam/tree/trie/forest/multipath，不让 Top-16 变成路径维，不增加 ordinary verifier 之外的在线 target inference，也不在 accepted-length 主机制成立前投入 SGLang 小修小补。
- **Constraints：** 单次 head 必须同时消费完整 `[B,16,*]` DFlash online features；每个输出位置必须通过无 causal mask 的全局 mixer 看到全部16位；一次产生 `[B,16,16]` scores 并以一次逐位置 argmax 得到唯一 `[B,16]` 序列。Top-16 只作每位置候选轴。训练/选择/held-out prompt 必须隔离，target 信息只作离线标签。新增参数先控制在 `10.75M` 内，并以同 A40、同 BF16、同 batch/block、eager-to-eager 公平 profile 约束成本。
- **Success condition：** 一个未经 target-feature 泄漏、在新 disjoint held-out 上成立的 full16 global-vs-local 机制信号；固定 EAL 至少 `8.3254859086`、动态 EAL 至少 `1.15x` Domino，三个域不退化；随后同栈 SGLang A40 tokens/s 的 paired 95% CI 下界至少 `1.15x` Domino。任何架构不变量失败均为 hard NO-GO，不能用 oracle、same-set capacity 或 off-spec 系统结果替代。

## Technical Gap

当前 full16 数据显示，1,175个 block 中771个在 pure-base Top-16 可达前缀内
至少需要两次 non-top1 correction。perfect-1 oracle只有 `7.49854`，perfect-2
为 `8.42383`，说明至少两次 joint repair 是到达主门的必要条件，但后者只比目标
高 `0.09835`，并不证明 selectability。

PGCF-v1 的 expected-prefix-product loss 对771个 multi-repair block 的第二次
correction平均 credit为 `0.001067`，仅为第一次的 `8.66%`；588个低于
`1e-3`。这是一个明确的优化错配：single-pass head 同时输出所有位置，loss 却在
第一处没学好时把第二处的训练信号乘没。

已有 axial global head 本身不是新贡献；历史100K规模也证明普通 Candidate-D-PACE
不足以获得目标增益。缺失机制是：在保持同一 full16 global scorer 不变时，如何
给所有可达位置非消失监督，并直接提升“第二处错误之前整段 prefix 的最弱
decision boundary”。

## Method Thesis and Contribution

- **Thesis：** JAPD 把 target candidate distribution 蒸馏成 all-prefix
  non-vanishing token risk，并用 joint prefix-bottleneck margin 耦合前两处
  correction及其保护前缀，使一个全局非因果 head 在一次并行输出中学习多处共同
  正确，而不增加任何在线步骤。
- **Dominant contribution：** 一个 loss-level mechanism，JAPD；architecture
  明确复用，不声称为新发明。
- **Non-contributions：** 不提出新 backbone、verifier、decoder、router、tree、
  iteration 或在线 teacher。

## Reused Online Head

默认直接复用 `GlobalDirectCandidateSelector` 的 axial implementation，冻结为：

- `max_positions=16, K=16, scope=global, mixer=axial`；
- `d=64, heads=4, layers=1, additive node encoder, dropout=0.1`；
- trainable parameters精确 `433,852`；
- D256/H8/L2 `4,539,888` 只有在D64同时失败预注册same-set capacity gate与
  full-fit optimization gate时才允许一次；该boolean必须在读取fresh300 outcome
  前决定，三条matched arm必须整体使用同一容量；
- target/DFlash backbone、embedding和LM head全部冻结。

Dataflow：

```text
H[B,16,2560] + base Top16 ids/logits[B,16,16] + anchor/candidate embeddings
  -> candidate-local attention within each position
  -> soft candidate-set summary per position
  -> every candidate queries all 16 summaries with no causal mask
  -> residual scores[B,16,16] + base candidate log-prob
  -> one argmax over K for every position
  -> exactly one proposal[B,16]
```

`W_out=0`，所以 step0 scores与base条件log-prob完全相同。没有中间argmax、
selected-token embedding、position loop或第二次head call。Position summary是
一个有损soft summary；正文不再称其为“完整分布的可逆表示”。

当前 R047 cache 缺 `base_logsumexp`，本轮冻结为唯一实现：从同一个 frozen
DFlash full-vocabulary logits离线派生并补存 FP32 `base_logsumexp` sidecar。派生时
使用 cache 中的 `parallel_hidden`、DFlash online path 实际使用的共享 frozen
vocabulary projection weight 和与采集相同的 BF16 `F.linear` geometry；重新计算的
Top-16 IDs必须与原记录完全一致，重算Top-16 logits转成存储dtype后也必须逐元素
一致，否则该记录fail closed。sidecar以
`(sample_id, anchor_offset, context_length)` 对齐，不改写 canonical rollout。

正式 online path 中，DFlash 本来就先产生 full-vocabulary base logits再做Top-16；
因此直接在该同一 tensor 上做 FP32 logsumexp reduction，不增加 target model
forward、第二条proposal或额外vocab GEMM。global-JAPD、local-JAPD和
global-Candidate-D-PACE三臂必须读取同一份值并使用完全相同的5个scalar channels；
禁止Top-16 conditional/full-vocabulary normalization混用。GPU mechanics还必须
逐位置检查sidecar-vs-online-replay `base_logsumexp` 的 `atol=1e-5, rtol=1e-6`、
5个scalar channels的同容差一致性及最终selected-token完全一致；complete profile
必须把FP32 reduction算入延迟。

## JAPD Objective

### 1. Clean/support horizon

对 block `b`、位置 `i`：

- `C_bi` 是 pure base Top-16 candidate IDs，要求16个ID互异；
- `y_bi` 明确定义为 `gold_ids_bi`，即 canonical clean target token；
- `r_bi` 是 `y_bi` 在 `C_bi` 中的唯一 rank，不存在则为 `-1`；
- `g_bi = [target_top1_ids_bi == gold_ids_bi]`，其中两者来自同一
  `Qwen3-4B` checkpoint、同一 canonical gold prefix、BF16/SDPA teacher pass；
- target logits只在当前 `C_bi` 的16个ID上gather，不输入head。

定义第一个不可监督位置：

\[
h_b=\min\{i:r_{bi}<0\ \lor\ \neg g_{bi}\},
\]

若集合为空则 `h_b=16`。`i>=h_b` 无任何 loss。若 `h_b=0`，该 block 的loss为0
且不进入loss分母。所有softmax、margin、累计和与normalization均为FP32。

### 2. Candidate-conditional soft teacher

令 `t_b(c_bik)` 为同一 teacher pass 在candidate ID上的logit，固定 `T=2`：

\[
\bar p_{bik}=\frac{\exp(t_b(c_{bik})/T)}
{\sum_{\ell=1}^{16}\exp(t_b(c_{bi\ell})/T)},\qquad
\tilde p_{bi}=0.9\delta_{r_{bi}}+0.1\bar p_{bi}.
\]

这是K16内重新归一化的offline label，不是full-vocabulary distribution，也不在
推理时计算。下文all-prefix解释只对其中 `0.9 delta` hard component称为clean-prefix
log loss；`0.1 pbar` 是使用同一位置权重的candidate-conditional distillation
regularizer。

### 3. All-prefix non-vanishing distillation

令 `q_bi=softmax(scores_bi)`。固定full16 normalizer为：

\[
Z=\sum_{m=1}^{16}m=16\cdot17/2=136.
\]

\[
\mathcal L_{AP}^{(b)}=\frac1{Z}
\sum_{i=0}^{h_b-1}(h_b-i)
\mathrm{CE}(\tilde p_{bi},q_{bi}).
\]

hard-label部分等于所有可达prefix
log loss之和再除以full16常数；soft部分是同权重的distillation regularizer。
因此长clean horizon保留更大的accepted-prefix utility，而第二、第三correction
仍不再乘此前预测概率。

### 4. Joint two-frontier bottleneck

在 `i<h_b` 中定义base错误集合 `E_b={i:r_bi!=0}`。仅当 `|E_b|>=2` 时激活
joint项；令 `e_b` 为第二小元素，联合保护集合为 `P_b={0,...,e_b}`。只有0或1处
base错误的block由 `L_AP` 负责repair/protection，`L_J2=0`，不占用two-frontier
gradient budget。

每位置hard target margin为：

\[
d_{bi}=s_{bi,r_{bi}}-\max_{k\ne r_{bi}}s_{bik}.
\]

用不除以 `|P_b|` 的保守 smooth minimum 表示整段prefix的联合边界：

\[
M_b^{joint}=-\log\left(\sum_{i\in P_b}\exp(-d_{bi})\right),
\qquad
\mathcal L_{J2}^{(b)}=\mathrm{softplus}(-M_b^{joint}).
\]

由于 `M_b^{joint} <= min_{i in P_b} d_bi`，任意位置margin不为正时联合certificate
也不可能为正；梯度按 `softmax(-d)` 自动集中到当前最弱位置。这避免了normalized
mean对单个错误的 `|P_b|` 稀释，因此该项不是两个独立repair recall的相加。它
不把gold/error位置输入head，标签只参与离线loss index。所有logsumexp用稳定的
FP32实现。

最终唯一loss为：

\[
\mathcal L_{JAPD}=\operatorname{PromptBalancedMean}_b
[\mathcal L_{AP}^{(b)}+\mathcal L_{J2}^{(b)}].
\]

精确prompt-balanced目标定义为：

\[
\mathcal L=\frac1{|\mathcal P^+|}\sum_{p\in\mathcal P^+}
\frac1{|B_p^+|}\sum_{b\in B_p^+}
[\mathcal L_{AP}^{(b)}+\mathcal L_{J2}^{(b)}],
\quad B_p^+=\{b:h_b>0\}.
\]

训练继续对全部有效block逐epoch shuffle，但使用固定全局缩放。记有效block总数
为 `N+`、有效prompt数为 `P+`、当前batch实际大小为 `B`，实现固定为
`N+/(B*P+) * sum_b L_b/|B_p^+|`；绝不除以随机batch weight sum。它在uniform
block shuffle下是上式prompt mean的无偏梯度估计，且每个epoch仍完整访问全部有效
block。没有另一个loss weight、teacher curriculum、evidence dropout或base safety
auxiliary。

## Joint Metric

对至少有两处可达base错误的block，记前两处为 `e1,e2`：

\[
J2_b=\mathbf1[\forall i\le e_2,\ \hat y_{bi}=y_{bi}].
\]

这一个条件同时表示：第一处和第二处都修好，二者之前/之间所有原本正确token都
未被破坏。实现必须使用inclusive `0:e2+1`，并在新实验前以逐block reference
loop复核向量化实现。报告prompt-balanced `J2`，并另报分子/分母、Domino、old D-PACE、
matched-local。逐block reference audit已完成：旧 `771/208` 只按candidate coverage
截断，未应用本方案的teacher-geometry clean horizon，故废止。严格同时应用
`r>=0`、`target_top1==gold` 与inclusive `0:e2+1` 后，multi-error分母为745；
released Domino为 `207/745=27.7852%`，PGCF-v1 global为 `15/745=2.0134%`，
local为 `0/745`。向量化实现必须精确复现这四个整数；旧数只说明口径差异，不能
再用于定标。

## Training and Data Protocol

### Small-data frozen recipe

- R047 train的1,987个prompt先按固定、label-independent permutation分成
  fit/select/diagnostic=`80/10/10`；三个集合prompt ID严格不交叠；
- batch size16；AdamW `lr=3e-4, weight_decay=1e-2`；
- 200-update linear warmup，随后cosine decay到 `3e-5`；
- maximum 6 epochs，eval every250 updates；
- seed0；checkpoint只按internal-select prompt-balanced EAL最大选，完全相同取早；
- diagnostic和历史validation_select/validation_gate均不参与选择；
- 三臂同初始化、同fit/select、同batch order和预算。

D64 full-fit optimization gate也在读取fresh300前冻结：在fit prompt上另取一个
label-independent、与capacity512不重叠的512-prompt train-diagnostic manifest；
full-fit gate精确定义为
`pass = (J2>=90%) AND (oracle-gap recovery>=80%)`；任一子项低于门槛即记为
internal-fit失败。只有它与same-set capacity gate都失败才允许全三臂共同切换
D256；capacity通过但heldout差一律解释为transfer failure，禁止靠加参rescue。

冻结三臂：

1. `global-JAPD`；
2. `local-JAPD`（只关闭cross-position visibility，参数完全匹配）；
3. `global-Candidate-D-PACE`（现有exact objective，`alpha=0.5`）。

这分别回答global information净价值和JAPD相对旧objective的净价值。PGCF-v1与
released Domino只作外部baseline，不作为第四个训练臂。

### Fresh manifests

在采任何新target label前，从Open-PerfectBlend原始池建立：

- 300 prompts（chat/code/math各100）的mechanism gate；
- 600 prompts（各200）的untouched final fixed/dynamic/system set；
- 两者以及fit/select必须排除所有历史train/select/gate/formal manifest中的
  sample IDs和normalized-text near duplicates；二者彼此也排除；
- 后续25K和100K training/internal-select manifests也必须显式排除这300与600个
  prompts的IDs及normalized-text near duplicates；
- manifest内容、抽样seed、排除计数和sample IDs append-only落盘；遵从用户要求，
  不设置无助于主结果的运行时cryptographic-hash gate。

300-prompt gate一旦用于决定scale-up就永久降为development；600-prompt set在
architecture、loss、100K recipe、seeds和primary checkpoint冻结前不采/不读outcome。

## Claim-Driven Validation

### Experiment 1 — Mechanics/capacity/latency

- 512 same-set blocks，先跑D64；要求finite、identity、full16双向global visibility、
  one-call/one-chain、J2≥99%、oracle-gap recovery≥95%、harm≤1%；
- 只有D64同时失败same-set capacity和上述full-fit optimization gate，才允许全部
  三臂统一切换一次D256/L2；任一门通过均保持D64，heldout overfit时禁止加参；
- 同A40/BF16/batch1测 complete path（base vocab GEMM→Top16→gather→head→argmax）
  与incremental head p50/p90/mean、active/peak memory；complete path明确包含FP32
  base-logsumexp reduction，且complete≤1.20x eager Domino。

### Experiment 2 — Fresh mechanism gate

三臂在300个新prompts上只评估一次。全部硬门：

- `global-JAPD - local-JAPD >=0.15 EAL`，10K paired prompt-bootstrap CI lower `>0`；
- `global-JAPD - global-D-PACE >=0.15 EAL`，CI lower `>0`；
- global-JAPD EAL `>=7.55`，三域各自`>=base`；
- prompt-balanced J2 `>=` same-job released Domino J2，并且相对local和D-PACE
  至少各提高10 percentage points；
- prefix-harm fraction `<=` Domino；报告suffix-only edit fraction，但不拿它替代EAL。

任一失败关闭exact JAPD route；不加数据、参数或post-hoc权重。

### Experiment 3 — Scale and final objective

仅Experiment2通过：

- 25K prompts只作冻结recipe learning-curve fail-fast；
- 25K仅训练global-JAPD seed0；其internal-select advancement hard gate为
  `EAL >= max(7.80, 1.075 * same-job Domino EAL)`、每域不低于same-job base且
  `J2>=same-job Domino J2`。任一失败关闭路线；全部通过才进入100K，禁止看曲线后
  解释“趋势足够”；
- 最终训练多样性至少100K prompts，约800K full16 blocks，与历史axial baseline
  同量级；三seeds `{0,1,2}`；
- checkpoint/primary selection只看100K内部select，不看600-prompt set；
- 进入final前100K internal development fixed EAL必须
  `>=max(8.3254859086, 1.15 * same-job Domino EAL)`；
- 600-prompt一次性fixed和dynamic均`>=1.15x`同作业Domino且每域不退化；
- 通过后才做同栈SGLang，paired A40 tokens/s ratio的95% CI lower `>=1.15`。

## Failure Interpretation

- **D64 same-set capacity与full-fit optimization同时失败、D256通过：** 证明D64
  underfit，允许全三臂统一D256；不等于heldout成功。只有一个D64门失败不加参。
- **JAPD胜D-PACE但不胜local：** objective有效，global contribution不成立；不能
  作为用户要求的方法继续scale。
- **胜local但不胜D-PACE：** global information存在，JAPD机制不成立；关闭JAPD。
- **J2升但EAL不升：** 其它早期harm或第三处之后损失抵消；主目标失败。
- **fresh mechanism gate失败：** frozen-lattice selectability不足或objective不泛化；
  关闭exact route，不转向serial/tree。
- **fixed成功、dynamic失败：** 训练/rollout shift未解决；不进入SGLang claim。

## Novelty and Claim Boundary

JAPD不把既有axial head重新命名为新architecture。其核心区别是对 speculative
accepted-prefix 的两级优化：all-prefix log transform修复后位gradient starvation，
joint bottleneck明确优化同一输出中前两处correction及其保护prefix的最弱margin。
它只在offline loss中用target labels，在线仍是一张全并行图。

Perfect-two oracle只证明多纠错必要性；它不证明JAPD充分、不证明draft features可
识别gold，也不证明dynamic成功。只有fresh matched三臂、100K scale和final
fixed/dynamic/SGLang证据才能逐级支持这些claim。

## Compute and Timeline

- mechanics + 三臂small-data + 300-prompt collection/eval：约2–4 A40 GPU-hours；
- 25K fail-fast：约2–5 A40 GPU-hours；
- 100K full16三seed：约20–60 A40 GPU-hours，collection与训练并行分片；
- system工作只在fixed/dynamic过门后开始。
