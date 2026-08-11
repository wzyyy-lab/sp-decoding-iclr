# Phase 3 Tier-1 失败复盘与下一版全局候选选择器

> 日期：2026-07-24  
> 复盘对象：数据作业 `10035436`、六次训练作业 `10035437`  
> 结论等级：development analysis；不能作为正式测试结果  
> 原始产物：
> - `artifacts/canonical/qwen3_4b_phase3_tier1_10035436/`
> - `artifacts/analysis/candidate_ceiling_phase3_tier1_10035436.json`
> - `artifacts/training/phase3_tier1_10035437/`

## 1. 结论先行

这轮实验否定的是当前的 **rank-32、no-mixer、absorbing-prefix-CRF SPH
实现**，还没有否定“在冻结 DFlash 后用轻量全局模块选择 top-K token”这个
更大的方向。

原因非常明确：

1. 六次训练全部使用 `--head-type no_mixer`。神经 scorer 在位置 `i`
   只读取 `h_i`、前一候选和当前候选；它从未直接读取其他位置的候选集合。
   所谓 `global` 主要来自一阶 CRF 的 suffix partition，而不是一个真正处理
   整个 `L×K` candidate lattice 的神经网络。
2. gate 上 K16 oracle 相对 DFlash 有 `+4.799` EAL，最好的 learned
   global MAP 只提高约 `+0.065`，仅追回 oracle 缺口的 `1.36%`。
3. 在首次 DFlash top-1 错误处，`42.0%` 的正确答案位于 rank 3–16；
   但六个模型成功修复中 `81.6%–91.3%` 只选择 rank 2。当前模型事实上是
   一个 top1/top2 flipper，没有学会使用 top-16。
4. 约六成到七成的路径改动发生在本轮已经不可达的 suffix。训练目标虽然
   censor 了首次 `OTHER` 之后的 token，却没有充分把梯度集中到真正限制
   accepted length 的位置。
5. global 相对 matched local 六个 seed 都有正增益，说明跨位置结构不是
   完全没有信号；但相对 DFlash 的增益小、置信区间多数跨 0、首 token
   准确率反而下降，而且收益完全由 math 提供。
6. 当前模型在训练集本身也只追回 oracle gap 的约 `2.4%–6.8%`。因此首要
   问题不是 held-out 数据太少导致的泛化，而是表达能力、参数化和目标函数
   连训练数据都拟合不了。

所以正确决策不是继续 sweep 当前 CRF 的 loss weight，也不是马上判死
“全局选择”这个想法，而是：

> 停止当前 SPH 参数化；先做 information/capacity probe，再实现一个真正把
> `L×K` 候选节点共同输入的 Global Candidate-Lattice Selector（GCLS）。

## 2. 数据与候选上限

`10035436` 正常完成，没有采集错误：

| split | prompts | blocks |
|---|---:|---:|
| train | 1,987 | 15,886 |
| validation-select | 147 | 1,175 |
| validation-gate | 149 | 1,192 |
| 合计 | 2,283 | 18,253 |

每个 block 有 15 个 draft positions，保存 DFlash top-64、完整词表
`logsumexp`、DFlash hidden、target greedy gold。train/select/gate 在 prompt
级别隔离；target embedding 与采集 checkpoint 的哈希也被绑定。

### 2.1 候选存在性很强，但不是可选择性的证明

| 集合 | DFlash K1 EAL | K16 oracle EAL | oracle gap |
|---|---:|---:|---:|
| overall | 5.034 | 9.868 | +4.834 |
| validation-select | 5.113 | 9.723 | +4.609 |
| validation-gate | 5.117 | 9.916 | +4.799 |

gate 上 K16：

- 第一位置 gold recall：`98.57%`；
- 完整 15-token block coverage：`37.42%`；
- 平均保留 DFlash full-vocab mass：`88.71%`。

按 domain 的 overall oracle gap：

| domain | K1 | K16 oracle | gap |
|---|---:|---:|---:|
| chat | 2.611 | 6.549 | +3.938 |
| code | 6.094 | 11.214 | +5.121 |
| math | 6.349 | 11.774 | +5.425 |

这证明每个 domain 都有候选空间，却不证明 frozen DFlash feature 能识别
哪个候选是 target token。这里必须明确区分：

- **availability**：gold 是否在 top-K；
- **identifiability/selectability**：不看 target 时，draft-side feature
  是否足以把 gold 从 K 个候选中分出来。

当前项目只严格证明了前者。

### 2.2 首次错误处的 rank 分布

validation-gate 的 1,192 个 blocks 中，1,074 个存在 DFlash top-1 错误。
在首次错误处：

| gold rank | blocks | 占首次错误比例 |
|---|---:|---:|
| rank 2 | 540 | 50.28% |
| rank 3–16 | 451 | 41.99% |
| rank 17–64 | 57 | 5.31% |
| top-64 外 | 26 | 2.42% |

因此 top1/top2 修正最多只能覆盖一半首次错误；要大幅提高接受长度，模型必须
真正学会跨过 base logit margin，选择 rank 3–16。

首次错误处的平均 `top1 - gold` logit gap 随 rank 快速增大：

| gold rank | mean gap |
|---|---:|
| 2 | 1.106 |
| 3–4 | 1.923 |
| 5–8 | 2.869 |
| 9–16 | 3.793 |

当前大多数 checkpoint 的平均 residual range 只有约 `1.86–2.42`，自然主要
只能翻转 rank 2；能产生更大 residual 的 seed 同时带来更多 harmful flips。

## 3. 六次训练到底测了什么

六次训练不是六种架构，而是完全相同的模型：

- `head_type=no_mixer`
- `normalization=absorbing_crf`
- `candidate_k=16`
- `rank=32`
- 327,713 trainable parameters
- 10 epochs，AdamW，lr `3e-4`

唯一实验因子是：

- survival auxiliary weight `0.0` 或 `0.1`；
- seed `0/1/2`。

因此它只能回答“当前一阶 low-rank CRF 加不加这个 auxiliary 是否有效”，
不能回答：

- bidirectional/global candidate mixer 是否有效；
- 更大的 lexical transition capacity 是否有效；
- candidate-only discriminative objective 是否有效；
- confidence-gated fallback 是否有效；
- rank 3–16 hard-example training 是否有效。

### 3.1 gate 聚合结果

| 配置 | local-survival | global-MAP | global-survival | global-survival − base |
|---|---:|---:|---:|---:|
| survival weight 0 | 5.114 | 5.162 | 5.160 | +0.042 |
| survival weight 0.1 | 5.092 | 5.183 | 5.173 | +0.056 |

共同的 DFlash base 为 `5.117`。

有三个重要结论：

1. `global-survival - local-survival` 在六个 seed 上全部为正，均值分别为
   `+0.046` 和 `+0.081`。跨位置结构有可重复的小信号。
2. `global-survival - global-MAP` 均值为 `-0.003` 和 `-0.010`。当前
   survival decoder 没有独立贡献，反而略差。
3. 最好配置仍只追回 oracle gap 的约 `1.17%`；global MAP 约 `1.36%`。

按 prompt 做 paired bootstrap 后：

- weight 0 的三 seed 平均增益 95% CI 约为 `[-0.021, 0.111]`；
- weight 0.1 的三 seed 平均增益 95% CI 约为 `[-0.020, 0.138]`。

因此不能声称方法已经稳定优于 DFlash。

### 3.2 收益来自 math，chat/code 都退化

| 配置 | chat Δ | code Δ | math Δ |
|---|---:|---:|---:|
| survival weight 0 | -0.087 | -0.016 | +0.227 |
| survival weight 0.1 | -0.072 | -0.021 | +0.258 |

math 的局部模式和格式更确定，弱 transition scorer 也能利用；chat 的可选
表达更多、base margin 更难解释，当前 head 发生过度修正。这个结果说明下一版
必须有显式的 abstain/KEEP_BASE 机制，不能强制每个 block 都 rerank。

### 3.3 首 token 变差

base 第一 token accuracy 为 `86.41%`；global-survival 平均约
`85.54%–85.57%`，各 seed 下降 `0.50–1.34` 个百分点。

接受长度的微小正增益来自少数较长修复抵消更多早期破坏，而不是一个安全的
统一改善。对 speculative decoding 来说，这是一种危险的 Pareto trade-off。

checkpoint 选择不应只最大化 aggregate EAL，至少要加入：

1. first-token non-inferiority；
2. harmful override cost；
3. domain-wise no-harm；
4. selective risk/coverage。

### 3.4 模型只会修 rank 2

六个 global-survival checkpoint 的成功修复中：

- rank 2：占 `81.6%–91.3%`；
- rank 3：大多只有个位数到二十余个 blocks；
- rank 4 及以后：几乎没有。

与之相对，真实首次错误中 rank 3–16 有 451 个 blocks。候选上限和实际模型
之间的最大断层就在这里。

### 3.5 大多数路径变化不可达

每个 checkpoint 修改了 874–969 个 gate paths，但只有 249–392 个变化发生
在 base 首次错误之前或该位置；577–625 个是纯 suffix-only 变化。

这解释了为什么 path disagreement 很大，而 EAL 几乎不动。global
survival 与 global MAP 也会在大量完整路径上不同，但真正到达差异位置的
blocks 很少。

### 3.6 连训练集都没有拟合

train 上：

- base EAL 约 `5.020`；
- K16 oracle 约 `9.875`；
- selected checkpoints 的 global-survival 只有 `5.137–5.349`。

即使在训练数据上，也只追回约 `2.4%–6.8%` 的 recoverable gap。这是
capacity/parameterization/objective 问题的直接证据。单纯增加相同分布的数据
不能解决一个连 512-block memorization test 都尚未通过的模型。

### 3.7 NLL 与 EAL 明显错位

六个 run 中，validation NLL 从 selected epoch 到 epoch 10 都继续下降，但
global-survival EAL 全部下降。例如 seed 0、weight 0：

- epoch 5：NLL `12.366`，EAL `5.194`；
- epoch 10：NLL `12.206`，EAL `5.091`。

当前 likelihood 在奖励“整体概率拟合”，而 EAL 只关心第一个限制位置。
`gold_prefix_survival_loss` 只是提高 gold prefix 自身的模型概率，没有直接
比较 selected wrong path、base path 和 gold，也没有对 harmful override
施加足够代价。weight 0.1 几乎没有改变失败模式。

## 4. 根因排序

### P0：当前实现没有真正测试用户的全局选择假设

`SurvivalPathHead` 的 edge score 在位置 `i` 只依赖：

- 当前 DFlash hidden `h_i`；
- previous candidate embedding；
- current candidate embedding；
- 当前位置 base logits。

CRF backward partition 能让未来 edge mass 影响早期条件概率，但这不等于
神经网络看到了其他位置的候选 identities、ranks、logits 和多模态组合。
它仍是一个一阶 Markov energy model。

仓库里虽有 `BidirectionalBlockMixer`，本轮完全没有运行；而且它只把每个位置
的 top-K 压缩成一个 soft expected embedding，`of course` / `no problem`
这类多模态候选可能在平均后丢失。

### P0：参数化过小且 lexical capacity 不足

当前 head 只有 0.328M 参数，rank 32，所有 token 语义都来自 frozen target
embedding，没有 trainable token-specific compatibility table。

作为对照，DSpark 的公开 Qwen3-4B 配置使用 rank 256 的
vocabulary-specific Markov factors。对约 152K vocab，两个
`V×256` table 本身约 77.8M 参数，仍只占 4B target 的约 2%。当前方案比这个
lexical correction 小约 237 倍。

“外挂轻量”不应被误解为“必须小于一百万参数”。10M–30M 参数依然非常轻，
并且 candidate-only head 不需要再做一次 full-vocab projection。

### P0：base-logit residual 形式阻碍远 rank 修复

当前分数是：

```text
adjusted = base_logit + scalar_scale * low_rank_residual
```

正确 token 在 rank 9–16 时平均需要跨过 3.79 logit gap。residual regularizer、
rank 32 和一个全局 scalar scale 共同把模型限制成小幅校正器。

下一版应把任务定义成 top-K **listwise classification/reranking**，而不是要求
一个受正则约束的小 residual 先克服未经校准的 full-vocab logit 尺度。

另一个优化问题是 `residual_scale` 从精确 0 初始化：第一步只有这个 scalar
有梯度，其余 scorer 参数梯度为 0；后续才开始共同学习，而且 scalar 与内部
权重存在缩放不辨识。它不是主要失败原因，但应改成小 gate 或 zero-init
final projection，并做专门的 optimization sanity test。

### P0：训练目标没有对齐“首次限制位置”

当前 censored NLL 对 oracle-lattice prefix 内每个事件做 likelihood，无法区分：

- 修复 base 首错位置带来的大收益；
- 修改不可达 suffix 的零收益；
- 把原本正确的早期 token 改错造成的高损失。

需要动态 reach/continuation weighting，而不是固定 CE 或当前
`-gold_prefix_utility`。D-PACE 的核心启示就是用当前 prefix confidence 与
后续 continuation value 动态决定每个位置的 CE 权重。

### P0：没有 KEEP_BASE / abstention

当前 decoder 总会输出 learned path。chat 上弱证据也会触发 override，导致
早期 harm。应把 DFlash 路径视作强 baseline action：

```text
if predicted_gain(global_path, base_path) > threshold:
    use global_path
else:
    keep base path
```

threshold 只在 validation-select 上校准，并报告 risk-coverage curve。

### P1：CRF 概率语义与 greedy agreement 不完全匹配

DFlash softmax probability 是 drafter 的 token distribution，不天然等于
“该 token 会与 target greedy argmax 相同”的概率。gate 的简单 10-bin
诊断中：

- DFlash top-1 平均 confidence：`57.74%`；
- 实际 target agreement：`46.26%`；
- 10-bin ECE：约 `0.115`。

absorbing CRF 把 base candidate/outside mass 当概率锚点，并用它做
survival-DP。当前 global MAP 反而稳定优于 survival-DP，说明这些概率还不够
calibrated，不能承担 Bayes-risk decoding。

建议先把 candidate scorer 做对，以 global MAP 作为主 decoder；单独训练并
校准 hazard/confidence 后，再决定是否恢复 survival decoder。

### P1：数据量和多样性不足，但不是第一故障

当前只有 1,987 个独立训练 prompts，8 个 anchors/prompt 之间高度相关。
Domino 使用约 1.42M target-regenerated examples；DSpark 公开 recipe 使用
Open-PerfectBlend、10 epochs；DeLS-Spec 的轻量 local head 也使用约 100K
文本样本。

当前数据相对这些工作少约 50–700 倍。chat 还只来自 ShareGPT，code/math
来源也较窄。

但是由于 train EAL 本身离 oracle 很远，正确顺序应是：

1. 小集 memorization/capacity sanity；
2. matched local/global probe；
3. 通过后再扩到至少 100K diverse prompts。

### P1：canonical anchor 与在线 anchor 分布不同

当前每个 target continuation 取 8 个固定 anchors。在线 speculative decoding
的下一 anchor 由上轮 accepted length 决定；新 selector 又会改变这些边界。

这不是当前离线失败的主因，但最终需要 DAgger/replay：

1. 用当前 selector 在线跑；
2. 收集实际 verification boundary 和首次拒绝位置；
3. target 标注后混回训练；
4. 保留随机 canonical anchors 防止过拟合单一 rollout。

Draft-OPD 关于 verification-exposed error states 的结果支持这一方向。

### P2：实验与工程卫生问题

1. `validation_gate` 已被本次复盘详细查看，下一版架构不能再把它当全新
   sealed gate；应将其降为 development-analysis，并从新数据建立 fresh gate。
2. reserved formal test 仍不应打开。
3. collection 在 dirty worktree 上启动，虽然保存了精确 source hashes，
   下一轮正式证据仍应绑定 clean commit。
4. 当前工作区单测为 `29 passed, 1 failed`。失败来自正在修改的
   `tests/test_data.py` 没有导入 `CanonicalBlockDataset`，与已完成 job
   结果无关，但下一次提交 GPU job 前必须修复。
5. `PLAN.md` / `docs/experiment_log.md` 仍把 `10035436/10035437` 写成
   executing，需要在方案冻结后更新，避免状态误读。

## 5. 下一版：Global Candidate-Lattice Selector（GCLS）

### 5.1 设计原则

GCLS 必须满足：

1. 冻结 DFlash；
2. 一次性读取整个 `L×K` lattice；
3. 神经计算没有 token-by-token recurrence；
4. 最终只输出一条 sequence；
5. target 仍做普通单链 verification；
6. base path 可随时无损 fallback。

### 5.2 候选节点，而不是 soft candidate mean

对每个位置 `i`、候选 `k` 建立节点：

```text
x[i,k] = [
    Proj(dflash_hidden[i]),
    Proj(frozen_target_embedding[token[i,k]]),
    trainable_token_embedding[token[i,k]],
    position_embedding[i],
    rank_embedding[k],
    normalized_base_logit[i,k],
    top1_margin[i],
    retained_mass[i],
    candidate_entropy[i],
]
```

推荐首版：

- K=16，L=15，共 240 个节点；
- model dim 128；
- 2 个 full-attention 或 axial candidate-lattice layers；
- 4 heads；
- trainable token embedding dim 64；
- relative-position bias。

240-node 两层 attention 的计算量很小。一个 `152K×64` token table 约 9.7M
参数；总 head 可以控制在约 10M–25M，仍远小于 DFlash/target。

### 5.3 全局 unary + 可精确解码的局部 transition

令 lattice transformer 为每个候选输出全局 unary：

\[
\phi_i(k;\mathcal L),
\]

其中 `L` 是完整候选 lattice。再加入可在大文本上预训练的 lexical
transition：

\[
\psi_i(j,k)
= a(v_{i-1,j})^\top R_{\Delta i} b(v_{i,k}).
\]

路径分数：

\[
S(y)=\sum_i \phi_i(y_i;\mathcal L)
    +\sum_{i>0}\psi_i(y_{i-1},y_i).
\]

因为 `phi` 已经看过整个 lattice，而 path-dependent 部分保持一阶，仍可用
`O(LK²)` DP 精确选择一条 global MAP path。它比当前 SPH 多出的关键信息是：

- 每个候选节点直接看到其他位置的候选 identities 和 logits；
- 不把 K 个 mode 压成一个均值；
- lexical transition 有 trainable token-specific capacity；
- 仍没有 Domino 的 sequential GRU rollout。

### 5.4 先用 MAP，暂时移除 absorbing-CRF 主叙事

当前数据已经显示 survival-DP 不优于 MAP。建议第一版按如下顺序：

1. candidate-only listwise scorer；
2. global MAP / Viterbi；
3. calibrated KEEP_BASE gate；
4. 只有当 calibration 和 validation 都证明有增益时，再加入
   expected-prefix-risk decoder。

`OTHER` 不再作为固定 base-mass 的生成 action，而是单独训练一个
`gold_outside_topK / prefix hazard` head。它用于 confidence、adaptive K
或提前截断，不参与候选 token 的 listwise normalization。

## 6. 如何训练 draft head 选择正确 token

### Stage A：必须先通过 memorization test

从 train 固定抽 512 或 1,024 blocks：

1. 关闭 residual regularization 和 dropout；
2. 反复训练到收敛；
3. 只评该训练子集；
4. 看 first-base-error candidate accuracy 和 EAL。

建议通过线：

- gold-in-K 位置的 candidate accuracy >95%；
- 首次 DFlash 错误且 gold-in-K 的修复率 >90%；
- 至少追回该小集 K16 oracle gap 的 60%。

达不到就不要扩数据：这是架构/实现/优化故障。

### Stage B：candidate-level supervised pretraining

主任务用 `K+1` 分类：

- 类别 `0..K-1`：gold 在候选中的 rank；
- 类别 `K`：gold 在 top-K 外。

loss 组成：

1. candidate-only listwise CE；
2. 单独 outside/hazard BCE；
3. base-first-error 的 pairwise margin loss；
4. rank-group balanced sampling，确保 rank 3–16 有足够梯度。

不要直接用未经标准化的 full-vocab base logit 作为不可跨越的主分数。可用：

```text
score = learnable_temperature * standardized_base_logit
        + global_candidate_score
```

base logit、margin、mass 仍作为强 feature，而不是硬概率锚。

### Stage C：acceptance-aware finetuning

用 D-PACE 风格动态权重：

\[
w_i = \operatorname{stopgrad}
\left(
  \text{prefix-survival}_{<i}
  \times
  \text{continuation-value}_{i}
\right).
\]

这会自动把梯度集中到当前真正限制 EAL 的位置。额外加入：

- base-correct 位置 false override 的不对称高代价；
- base 首错位置成功修复的 continuation reward；
- block-level `global path` vs `base path` pairwise preference。

在 CE 稳定后，可以对 scorer 产生的 top-M paths 做 minimum-risk finetuning：

\[
\mathcal L_{\mathrm{risk}}
=-\sum_{y\in \mathcal B}
 \operatorname{softmax}(S(y)/\tau)\,A(y),
\]

其中 `A(y)` 可由已有 gold 精确计算。它直接奖励 accepted prefix，而不是只
提高 gold path 自身概率。

### Stage D：dense target distillation

下一版 collector 应保存 canonical gold-prefix 条件下：

- target 在 DFlash top-K candidates 上的 logits/probabilities；
- target full-vocab logsumexp；
- target greedy label。

在 hard label CE 外加入 candidate-set KL/TV。DSpark 的公开训练代码以
CE + target-distribution L1/TV 为主要目标，说明 dense target distribution
是很重要的监督，不应只保留 argmax label。

### Stage E：大文本 lexical/lattice pretraining

有两条互补路线：

1. 像 DeLS-Spec 一样，在普通文本上预训练 short-context lexical head；
2. 像 LT-LM 一样，从普通文本构造 synthetic candidate lattices，训练
   single-shot lattice transformer 找回 gold arcs。

第二条尤其符合本项目：可以用大量廉价文本预训练“如何从一组局部候选中找
全局一致路径”，再用 100K target/DFlash canonical examples 适配 target
agreement。

## 7. 必须加入的安全决策

### 7.1 block-level fallback

同时计算：

- base path 的 calibrated predicted utility；
- global path 的 calibrated predicted utility；
- 两者差值和 uncertainty。

只有超过 validation-select 上冻结的阈值才 override。报告：

- override coverage；
- override precision；
- improved/harmed blocks；
- gross token gain/gross token harm；
- risk-coverage curve。

### 7.2 checkpoint 选择约束

建议 lexicographic selection：

1. first-token accuracy 相对 base 不下降超过 0.1 pp；
2. chat/code/math 各域不显著为负；
3. harmful override rate 低于冻结阈值；
4. 在满足上述条件的 checkpoint 中最大化 prompt-balanced EAL。

## 8. 最小而有判别力的实验矩阵

不要再做只改一个 auxiliary weight 的三 seed 重复。先做下面的因果消融：

| ID | scorer 输入 | decoder/loss | 回答的问题 |
|---|---|---|---|
| A | base logits | top1 | DFlash base |
| B | local node MLP | listwise CE | 单位置 feature 能识别多少 |
| C | local + trainable lexical transition | MAP | 强 local/Markov baseline |
| D | full `L×K` lattice transformer | independent argmax | 全局 feature 本身是否有用 |
| E | full lattice transformer + transition | global MAP | 推荐 GCLS |
| F | E + D-PACE | global MAP | acceptance weighting 是否有用 |
| G | F + KEEP_BASE | selective MAP | 是否消除 chat/首 token harm |
| H | current CRF | current loss | 已失败控制 |

关键 negative controls：

1. 把其他位置候选在 batch 内随机打乱；若收益不掉，模型没有使用 global
   candidate coherence。
2. 对 global attention 加 causal mask；full-bidirectional 必须在 matched
   params/latency 下优于 causal/local，才能支持“global information”主张。
3. 保留完整 hidden，只打乱 suffix candidate identities/logits，隔离
   DFlash hidden 自带双向信息与新增 lattice 信息。
4. local/global 使用相同参数预算，避免把容量收益写成全局收益。

核心诊断必须按以下条件分桶：

- base correct vs base wrong；
- first-error gold rank：2、3–4、5–8、9–16、outside；
- draft position；
- domain；
- base margin/entropy；
- reachable change vs suffix-only change。

## 9. 分阶段 go/no-go

### Gate A：实现/容量

在 512–1,024 blocks memorization test 上通过第 6 节阈值。未通过则只修模型，
不收更多数据。

### Gate B：现有 2K 数据的方向性

在当前 select + 已公开 gate 组成的 development pool 上：

- GCLS 相对 matched local head ≥ `+0.15` EAL；
- 相对 DFlash ≥ `+0.20` EAL；
- first-token drop ≤ `0.1 pp`；
- 三域没有明显负增益；
- rank 3–16 修复占比显著高于当前模型；
- shuffled-global negative control 的增益消失。

只在这个 gate 通过后扩数据。

### Gate C：100K fresh data

建议：

- 至少 100K 独立、target-regenerated prompts；
- 每 prompt 先取 2–4 anchors，而不是继续用 8 个高度相关 anchors；
- domain/source 更广；
- hard-example replay 与随机 canonical anchors 混合；
- 新建 validation-select 和 fresh validation-gate；
- 保持 reserved formal test 封存。

如果 100K 后 full global 仍不能稳定优于 matched local/DeLS-style head
`≥0.15–0.20` EAL，就应停止“全局选择器”主线。此时说明 top-K 的大量 oracle
gap 在 draft-side feature 下不可识别，或全局信号不足以抵消成本。

### Gate D：在线与系统

离线 gate 通过后才做：

1. online lossless rollout；
2. DAgger/replay；
3. eager latency；
4. fused/CUDA Graph；
5. matched TPS，而不只报告 EAL。

## 10. 最新相关工作的影响

### DeLS-Spec

[DeLS-Spec](https://arxiv.org/abs/2607.07409) 已经提出“冻结 DFlash + 独立训练
轻量 local head + product-of-experts”。它是本项目最直接的新颖性威胁，也是
必须复现的强 baseline。

本项目能保留的新贡献必须是：

- 非顺序的 full candidate-lattice neural processing；
- future candidate evidence 对早期选择的直接收益；
- matched latency 下优于 DeLS/local Markov/RNN head；
- single-chain、普通 target verification。

### DSpark

[DSpark 论文](https://arxiv.org/abs/2607.05147) 和
[公开代码](https://github.com/deepseek-ai/DeepSpec) 提供了强 Markov/RNN
head、confidence head、target-distribution matching 和大规模训练 recipe。
它说明“很小的 sequential dependency repair”已经是强基线，也说明当前
rank-32、0.328M scorer 的容量比较不公平。

### D-PACE

[D-PACE](https://arxiv.org/abs/2605.18810) 直接针对 parallel draft 的
accepted-length objective 推导动态位置权重。下一版应优先移植其 weighting
思想，而不是继续调当前 gold-survival auxiliary 的系数。

### DiffuSpec

[DiffuSpec](https://arxiv.org/abs/2510.02358) 已经用 causal-consistency score
在 diffusion candidate lattice 上做 path search。它支持“逐位置 top1
不等于一致路径”的动机，但也要求本项目证明 learned global lattice scorer
相对 tiny causal LM/beam 的独立价值。

### Draft-OPD

[Draft-OPD](https://arxiv.org/abs/2605.29343) 指出固定 teacher trajectory
训练会遗漏 verification 暴露的错误状态，并使用 replay/on-policy data。
它对应本项目最终的 anchor-boundary DAgger，而不是当前第一优先级。

### 非投机解码方向

- [LT-LM](https://arxiv.org/abs/2104.02526) 一次处理完整 ASR lattice 并给
  每条 arc 重打分，还用 synthetic lattices 扩充文本训练；它几乎是 GCLS
  最直接的跨领域架构先例。
- [Mask-Predict](https://aclanthology.org/D19-1633/) 说明并行生成中的
  multi-modality 常需要全局/迭代 refinement。GCLS 可借鉴一到两层 soft
  belief refinement，但不应重新调用 DFlash。
- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) 的
  risk-coverage 视角适合实现 KEEP_BASE：不确定时保持强 baseline，而不是
  强制 override。

## 11. 推荐执行顺序

1. 修复当前单测 import，冻结新的 clean commit。
2. 把 `10035436/10035437` 状态和本报告结论登记到 experiment log。
3. 增加统一 failure-analysis 脚本，自动输出 first-error rank、reachable
   edits、repair/harm、domain、bootstrap CI。
4. 对当前模型和现有 bidirectional mixer 做 512-block memorization test。
5. 实现 local listwise MLP；这是 GCLS 的 matched control。
6. 实现 240-node GCLS + global MAP，不先加入 CRF/survival。
7. 做 local/full/shuffled/causal 四个最关键对照。
8. 加 D-PACE weights 和 KEEP_BASE gate。
9. 通过现有 development gate 后，再采集 100K fresh data 和 target logits。
10. 冻结模型后才建立 fresh gate、运行 formal test 和在线 latency。

## 12. 最终判断

### 关于当前实现

应停止。继续调整 `survival_loss_weight`、rank 32 residual regularization 或
CRF decoder，不太可能把 `+0.05` 变成大幅增益。

### 关于全局选择 idea

仍值得做，但必须用可证伪的方式重新测试。当前结果甚至提供了一点正面迹象：
global normalization 相对 matched local 六个 seed 全正；问题是信号被极弱
scorer、错误的概率锚、无 abstention 和不对齐的 loss 吞掉了。

### 最大的科学风险

top-16 oracle 只说明 target token 在集合里。若 DFlash hidden、candidate
identities/logits 和廉价 lexical prior 仍不能把它识别出来，那么单链外挂
不可能追回大部分 oracle gap；只有 target-side 多分支/tree verification
才能直接利用候选覆盖。

因此下一轮最重要的不是再报一个 EAL 数，而是回答：

> 在 matched capacity 下，完整 future candidate lattice 是否真的提高了
> “首次 DFlash 错误处”的 gold-rank classification，且这种提升在打乱 future
> candidates 后消失？

这个实验一旦为正，idea 才真正成立；若在 100K 数据后仍为负，就应果断转向
DeLS/DSpark-style local correction 或允许修改 DFlash backbone。
