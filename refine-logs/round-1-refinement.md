# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**: 在 Qwen3-4B / DFlash-b16 的 greedy speculative decoding 中，利用一次 DFlash forward 已产生的完整 top-K candidate lattice，提高真实可达的 accepted draft length，并最终转化为吞吐收益；核心不是证明候选“存在”，而是让 head 能可靠地选择并保护正确路径。
- **Must-solve bottleneck**: 当前 GCLS-v1 的 global signal 已被三 seed 和 matched local/causal 对照证实，但相对 DFlash 仅约 `+0.232` calibrated EAL（最佳 development checkpoint raw `+0.285`），只回收约 5%–6% 的 K16 oracle gap；同时存在 first-miss repair 低、harm 不低、远端候选被单均值压缩、candidate/context 交互弱、训练 support 与实际策略 reach 不一致的问题。
- **Non-goals**: 不把 K16 oracle 当成 selector 可实现收益；不靠复制 Domino 的顺序 GRU 或扩大 verification tree 来改题；首轮不声称 `T>0` lossless sampling；不把在同一 validation split 上选 checkpoint 和校准阈值后的数字当 sealed-test 结论；不同时堆 CRF、GRU、tree search 和多个辅助 head。
- **Constraints**: target model 冻结；首轮优先复用 released DFlash 与现有 100K Open-PerfectBlend canonical records；greedy `T=0`、block size 16、K16；最终部署 head 必须保持一次并行 lattice 处理且开销低于收益 break-even；现阶段有 Slurm A800 资源，但 validation_select 只有 147 prompts，必须保留独立 calibration/test 口径。
- **Success condition**: 高容量 frozen-feature probe 若达到约 `+0.6` raw EAL 或回收 ≥15% oracle gap，则构成当前输入充分性的正证据；紧凑模型需回收 probe 增益的 ≥70%，三 seed 均优于 DFlash，global−causal/local 的 prompt-cluster CI 排除 0，harm 的单侧 95% 上界不高于 5%，并在独立 calibration/test 与真实 latency 下取得正吞吐收益。低 probe 结果只触发工程上的 stop/pivot，不被表述为 frozen feature 不含信息的科学证伪。

## Anchor Check

- **Original bottleneck**: 当前 selector 能利用 global lattice，但只修复少量 first miss，并会破坏原本正确的 DFlash prefix；必须把 candidate evidence 与 accepted-prefix utility 对齐，而不是继续追逐普通 token accuracy。
- **Why the revised method still addresses it**: 新版本直接以 soft expected accepted-prefix length 为目标，且只增加一个 DFlash base-prefix safety constraint；结构上取消远端候选的单均值压缩，但不改变一次 parallel DFlash + selector + verifier 的部署路径。
- **Reviewer suggestions rejected as drift**: 不把 LoRA/layer-wise target fusion并入 frozen-selector 主方法；该路线改变“固定 DFlash 后 lattice 是否可选择”的科学问题，只保留为另一个项目分支。

## Simplicity Check

- **Dominant contribution after revision**: **Accepted-Reach Risk Minimization (ARR)** over a non-pooled full candidate lattice，配一个明确的 base-safety constraint。
- **Components removed or merged**: 删除 hard-reach Head-AUF、first-miss repair margin、all-position coverage 联合项、multi-slot 默认结构、默认蒸馏、动态 base scale；teacher 改为不参与部署的 empirical sufficiency probe。
- **Reviewer suggestions rejected as unnecessary complexity**: 不引入 inducing-point/slot attention，除非真实 full-lattice latency 已经证明无法 break even；不为 safety 再训练 confidence head。
- **Why the remaining mechanism is still the smallest adequate route**: 当前 flat lattice block 已存在，新增部分只是一种显式 candidate/context node interface、一个可微 prefix utility 与一个 base-prefix hinge constraint。

## Changes Made

### 1. Hard reach mask → differentiable accepted-reach utility

- **Reviewer said**: detached argmax reach 只是 on-policy support filter，不能给早期修复带来的 continuation value 分配 credit。
- **Action**: 取消 hard-reach training mask，定义 soft prefix utility `U=sum_t prod_{i<=t} q_i`，直接最小化 `1-U/L`；hard greedy reach 只用于 evaluation 和 breaker diagnostics。
- **Reasoning**: 这与 accepted draft tokens 的结构精确同形。进一步地，`d(1-U/L)/d(-log q_i)` 正比于 `sum_{t>=i} prod_{j<=t} q_j`，即 D-PACE continuation-value 权重在 `alpha=0` 时的梯度形式。当前 `alpha=0.5` Candidate-D-PACE 是平滑后的优化启发式，不等于该 candidate-support utility 的精确梯度。
- **Impact on core method**: loss 从四项 patch 收紧为一个主 utility；修复早期 breaker 会自然获得其可能解锁 suffix 的价值。

### 2. Multi-slot default → compact full-lattice default

- **Reviewer said**: 在批评单均值压缩后马上加 slot compression 自相矛盾，且 slot operator 未定义。
- **Action**: 部署候选首先使用 1–2 层 compact full-lattice attention；保留所有 `L×K` nodes 到 readout。slot/inducing compression 完全移出主方法，只有 latency 失败后才考虑。
- **Reasoning**: K16、L15 只有 240 nodes，是否必须压缩应由实测而非直觉决定。
- **Impact on core method**: 主结构更简单，global/local/causal 仍通过参数匹配的 mask 比较。

### 3. 四项 Head-AUF → ARR + 单一 base-safety constraint

- **Reviewer said**: coverage、repair、protection 四项权重使贡献分散。
- **Action**: repair 完全交给 ARR；只对原始 DFlash 可接受 prefix 上的 rank-1 建立一个 greedy margin constraint。
- **Reasoning**: ARR 已同时奖励 base-correct token 与 first-miss gold；唯一未显式控制的是“用概率收益换取少量灾难性 early harm”，因此保留 safety constraint 有直接决策含义。
- **Impact on core method**: 只剩 `L_total=L_ARR+lambda_safe L_safe`，无独立 repair/coverage head。

### 4. “Bayes ceiling” → positive-only empirical sufficiency probe

- **Reviewer said**: 弱 teacher 不能证明 feature 不含信息，可能只是优化/样本/函数类失败。
- **Action**: 全部改称 high-capacity empirical sufficiency probe 或 capacity probe。
- **Reasoning**: 高结果能证明同输入存在可实现信号；低结果只能支持有限预算下停止，不支持信息缺失结论。
- **Impact on core method**: probe 不再是贡献或负证据，distillation 也不默认存在。

### 5. 加入现有结果的 support-mismatch 定量证据

- **Reviewer concern addressed**: 需要证明 loss support 问题不是纯概念猜测。
- **Action**: 从当前最佳 d64-e9 checkpoint 的 1,175 validation blocks 计算：Candidate-D-PACE 的 coverage-active positions 平均 9.72/block，而模型实际 greedy reach 加 breaker 仅 6.21/block；36.1% 的 coverage-active positions 位于当前 breaker 之后，chat 为 43.8%。
- **Reasoning**: 这解释了为什么 aggregate candidate accuracy 能升而 EAL 不一定升；但 revised ARR 不把这些位置硬删除，而是用 soft prefix probability 给它们连续、可解释的 reach credit。
- **Impact on core method**: ARR 同时避免 `alpha=0.5` 对远 suffix 的过强平滑和 hard mask 的不连续性。

## Revised Proposal

# Research Proposal: Accepted-Reach Risk Minimization over the DFlash Candidate Lattice

## Problem Anchor

- **Bottom-line problem**: 在 Qwen3-4B / DFlash-b16 的 greedy speculative decoding 中，利用一次 DFlash forward 已产生的完整 top-K candidate lattice，提高真实可达的 accepted draft length，并最终转化为吞吐收益；核心不是证明候选“存在”，而是让 head 能可靠地选择并保护正确路径。
- **Must-solve bottleneck**: 当前 GCLS-v1 的 global signal 已被三 seed 和 matched local/causal 对照证实，但相对 DFlash 仅约 `+0.232` calibrated EAL（最佳 development checkpoint raw `+0.285`），只回收约 5%–6% 的 K16 oracle gap；同时存在 first-miss repair 低、harm 不低、远端候选被单均值压缩、candidate/context 交互弱、训练 support 与实际策略 reach 不一致的问题。
- **Non-goals**: 不把 K16 oracle 当成 selector 可实现收益；不靠复制 Domino 的顺序 GRU 或扩大 verification tree 来改题；首轮不声称 `T>0` lossless sampling；不把在同一 validation split 上选 checkpoint 和校准阈值后的数字当 sealed-test 结论；不同时堆 CRF、GRU、tree search 和多个辅助 head。
- **Constraints**: target model 冻结；首轮优先复用 released DFlash 与现有 100K Open-PerfectBlend canonical records；greedy `T=0`、block size 16、K16；最终部署 head 必须保持一次并行 lattice 处理且开销低于收益 break-even；现阶段有 Slurm A800 资源，但 validation_select 只有 147 prompts，必须保留独立 calibration/test 口径。
- **Success condition**: 高容量 frozen-feature probe 若达到约 `+0.6` raw EAL 或回收 ≥15% oracle gap，则构成当前输入充分性的正证据；紧凑模型需回收 probe 增益的 ≥70%，三 seed 均优于 DFlash，global−causal/local 的 prompt-cluster CI 排除 0，harm 的单侧 95% 上界不高于 5%，并在独立 calibration/test 与真实 latency 下取得正吞吐收益。低 probe 结果只触发工程上的 stop/pivot，不被表述为 frozen feature 不含信息的科学证伪。

## Technical Gap

GCLS-v1 已证明 other-position lattice 有增量信息，但它的训练与部署效用没有完全对齐。最佳 checkpoint 的 coverage-active positions 为 9.72/block，而当前 greedy policy 真正到达的 prefix 加 breaker 只有 6.21/block；36.1% 的 active positions 位于当前 first breaker 后，chat 达 43.8%。另一方面，直接 hard-mask 这些位置会丢失“修复早期 token 可解锁后续多个 token”的 continuation credit。

结构上，当前 axial mixer 在跨位置交互前把每个位置的 K 个 candidate nodes 压成一个 soft mean；candidate identity 的多假设结构不可逆丢失。输入编码又只把 hidden、candidate embedding、anchor 做加法，candidate/context compatibility 必须由一个 0.43M 浅层 head 间接恢复。

因此缺失的最小机制不是更多 module，而是：

1. 一个不在全局交互前压缩 candidate hypotheses 的 full-lattice interface；
2. 一个与 accepted-prefix 乘积结构同形的 differentiable risk；
3. 一个阻止 selector 破坏 DFlash 已正确 prefix 的单一 safety constraint。

## Method Thesis

- **One-sentence thesis**: 在完整保留 DFlash candidate lattice 的条件下，直接最小化 soft accepted-reach risk，并约束原始 DFlash 可接受 prefix 不被翻转，可以把 global candidate signal 转化为净 EAL，而无需顺序 causal head。
- **Smallest adequate intervention**: target/DFlash/verifier 全部不变；复用现有 flat lattice block，只增加显式 compatibility node、ARR objective 和一个 safety hinge。
- **Timeliness**: 方法复用 frozen foundation-model semantic space，但不为了“现代化”引入 RL 或额外 teacher；精确监督和可微 prefix utility 已足够。

## Contribution Focus

- **Dominant contribution**: Accepted-Reach Risk Minimization (ARR) for direct top-K lattice selection，包含其与 expected accepted length 的形式化对应和 base-prefix safety constraint。
- **Supporting implementation choice**: no-prepool full-lattice candidate/context mixer，用于忠实承载 ARR；它不是并列算法贡献。
- **Explicit non-contributions**: capacity probe、distillation、slot compression、LoRA、dynamic base scale均不属于主方法。

## Proposed Method

### Inputs and Candidate Nodes

对位置 `i`、候选 `k`，使用当前 deployable inputs：DFlash hidden `h_i`、frozen target candidate embedding `e_{i,k}`、anchor embedding `a`、DFlash full-vocab log-prob和top-1 gap等 scalar。投影后构造：

```text
z_{i,k} = MLP([q(h_i), k(e_{i,k}), q(h_i) * k(e_{i,k}), k(a) * k(e_{i,k})])
          + position_i + rank_k + scalar_features_{i,k}.
```

乘性项只把已有 candidate-specific relation 显式化；没有 trainable vocabulary table。

### Full-Lattice Mixer and Output

将全部 `L×K` nodes 输入 1–2 层 compact full attention。local/causal/global 只改变 attention visibility：同位置、过去位置、全部位置；其余参数严格相同。主实验不做 candidate pooling。

输出保持 base identity：

```text
s_{i,k} = log p_DFlash(C_{i,k}) + Delta_{i,k},
```

其中 residual readout 零初始化，因此 epoch 0 的 greedy path 逐元素等于 released DFlash。

### Accepted-Reach Risk

令 `a_i` 表示 target gold 是否在位置 `i` 的 K 个候选中，`g_i` 为其 candidate index，

```text
q_i = a_i * softmax(s_i)[g_i].
S_t = product_{i=1..t} q_i.
U_ARR = sum_{t=1..L} S_t.
L_ARR = 1 - mean(U_ARR / L).
```

`U_ARR` 是在每位置按 selector categorical distribution 取样、并遇到第一个错误停止时的 expected accepted tokens。部署仍用 greedy，因此它是平滑训练 surrogate，最终只以 greedy EAL 判定。

它的 continuation credit 不是手工设计：若 `ell_i=-log q_i`，则

```text
d L_ARR / d ell_i = (1/L) * sum_{t>=i} S_t.
```

因此修复早期 position 会按其可解锁的所有 suffix survival 获得更大梯度。Candidate-D-PACE 在 `alpha=0`、detach weights 时给出相同的一阶梯度形式；当前 `alpha=0.5` 是额外 smoothing baseline，而不是 ARR 的定义。

### One Base-Safety Constraint

令 `b_i=1` 当且仅当 DFlash rank-1 在从位置 1 到 `i` 都正确，即该位置属于原始 DFlash accepted prefix。定义：

```text
L_safe = mean_{b_i=1} relu(m + max_{k>0} s_{i,k} - s_{i,0}).
L_total = L_ARR + lambda_safe * L_safe.
```

这直接保护任何会缩短 base accepted prefix 的 greedy flip。ARR 自身负责 repair，不再使用 separate repair margin；不训练额外 safety head。主网格只比较 `lambda_safe in {0, 0.1, 0.25}`，margin `m` 固定。

### Inference

verified prefix → target context features → released DFlash one-pass lattice → compact full-lattice selector → per-position parallel argmax → unchanged target verification。raw model 为主要报告；如需 KEEP_BASE threshold，只在独立 calibration split 选择一次并冻结到 test。

### Empirical Sufficiency Probe

一个 d640、4-layer、约 20M–30M 的 full-lattice model使用完全相同输入，作为 positive-only capacity probe。高收益说明 frozen inputs 对某个该函数类足够；低收益只触发工程 stop，不证明信息不存在。probe 不进入推理，也不默认用于蒸馏。

## Training Plan

1. **Implementation sanity**: ARR 手算/finite-difference；`alpha=0` Candidate-D-PACE gradient parity；epoch-0 DFlash identity；local/causal/global invariance；所有 frozen inputs 无 gradient。
2. **Objective isolation**: 在同一 current axial checkpoint family 上只比较 Candidate-D-PACE `alpha=0.5`、ARR、ARR+safety，先确定 objective 是否降低 harm 并提高 raw EAL。
3. **Representation isolation**: 固定 ARR+safety，比较 additive node vs explicit compatibility，以及 axial single-mean vs compact full-lattice；不同时改 data/K。
4. **Capacity probe**: 512-block memorization → 10K prompt smoke → 100K positive sufficiency probe；只有高 probe−student gap 才考虑蒸馏。
5. **Confirmatory**: architecture/hyperparameters冻结后，独立 checkpoint-selection、calibration 和 test；三 seed matched scopes，prompt-cluster bootstrap。

## Failure Modes and Decisions

- **ARR 数值/优化不稳**: 检查 survival underflow、finite-difference 与 gradient parity；用 log-space cumprod，不回退 hard mask。
- **ARR repair提高但 harm超限**: 只调 `lambda_safe`，不新增 repair/confidence module。
- **Full-lattice 没有超过 axial**: 删除 compatibility/full-lattice architecture claim，保留 objective contribution；这是预先指定的 deletion check。
- **Capacity probe 高、student低**: 才允许一次 conditional distillation experiment。
- **Capacity probe低**: 报告为 inconclusive/practical stop；LoRA route另立研究问题。
- **Latency不 break even**: 先缩 dim/layers；只有实测仍失败才研究 inducing compression。

## Novelty and Elegance Argument

DFlash 的 per-position output缺少显式 lattice decision；Domino/DSpark和DeLS依赖已生成 prefix 的 sequential causal correction；DFlare改变 draft backbone representation；D-PACE为 parallel drafter提供 position-aware CE。ARR-GCLS 的差异是：在 frozen DFlash、无生成 prefix 的 direct K-way selector上，将 full candidate lattice 的表示与 accepted-prefix risk显式耦合，并用一个 base-prefix constraint处理冻结强 baseline 的 asymmetric harm。结构、objective和部署路径围绕同一个“accepted reach”命题，不形成模块堆叠。

## Claim-Driven Validation Sketch

### Claim 1: ARR 比 smoothed Candidate-D-PACE 更对齐 greedy EAL

- **Minimal experiment**: 固定 current architecture/data/steps，比较 Candidate-D-PACE `alpha=0.5`、`alpha=0`、direct ARR、ARR+safety。
- **Metric**: raw prompt-balanced EAL；improved/harmed blocks；first-miss repair；最差 domain和首 token；soft ARR 与 greedy EAL 的 epoch rank correlation。
- **Expected evidence**: ARR 提高 raw EAL；safety 以很小 repair代价显著降低 harm。若仅 `alpha=0` 已等价且最好，则删除 direct-loss实现并把贡献收缩到 candidate-support adaptation+safety。

### Claim 2: 不预池化的 full lattice 才能保留可泛化 global gain

- **Minimal experiment**: 固定 ARR+safety，matched additive/compatibility与 axial/full-lattice；local/causal/global 参数匹配。
- **Metric**: global−causal/local prompt-cluster CI、oracle-gap recovery、candidate/hidden replacement、latency。
- **Expected evidence**: full global 显著优于 full causal/local与 single-mean axial，并达到 break-even。若没有，删除该结构 claim。

### Diagnostic gate: frozen inputs 的正向 sufficiency witness

- **Minimal experiment**: d640/l4 probe与 compact student同输入、同数据、充分收敛。
- **Interpretation**: 高 probe支持 compact/蒸馏路线；低 probe不作信息缺失结论。

## Validation Power and Split Discipline

checkpoint selection、safety calibration、final test使用三个 prompt-disjoint集合。根据 pilot 的 prompt内相关性预先计算 cluster-bootstrap power；在此之前保守目标为至少 1,000 calibration prompts和2,000 untouched test prompts。最终要求 test 上 harm rate 的单侧95% prompt-cluster上界 ≤5%，而不是只看点估计；seed重复不能替代新 prompts。

## Experiment Handoff Inputs

- **Must-prove**: ARR/gradient公式正确；ARR+safety净增 EAL；full-lattice global增量与 latency。
- **Must-run ablations**: Candidate-D-PACE `alpha=.5/.0`、ARR、ARR+safety；additive/compatibility；axial/full；local/causal/global。
- **Defer unless triggered**: slots、distillation、LoRA、dynamic base scale、dense target logits。
- **Critical metrics**: raw prompt-balanced EAL、repair/harm、one-sided harm bound、oracle-gap recovery、epoch EAL correlation、prompt-cluster CI、head/round latency。

## Compute & Timeline

- 实现和 sanity：1 天；objective isolation：约 8–16 A800 GPU-hours；compact representation isolation：约 12–24 GPU-hours；capacity probe按10K门禁后约 15–35 GPU-hours；confirmatory三 seed约15–25 GPU-hours。
- 先做 objective 与 compact full-lattice，不把 kernel压缩计入第一天；总预算约50–90 GPU-hours，任何 gate失败即停止后续分支。
