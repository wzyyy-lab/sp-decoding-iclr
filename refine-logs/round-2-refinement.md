# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem**: 在 Qwen3-4B / DFlash-b16 的 greedy speculative decoding 中，利用一次 DFlash forward 已产生的完整 top-K candidate lattice，提高真实可达的 accepted draft length，并最终转化为吞吐收益；核心不是证明候选“存在”，而是让 head 能可靠地选择并保护正确路径。
- **Must-solve bottleneck**: 当前 GCLS-v1 的 global signal 已被三 seed 和 matched local/causal 对照证实，但相对 DFlash 仅约 `+0.232` calibrated EAL（最佳 development checkpoint raw `+0.285`），只回收约 5%–6% 的 K16 oracle gap；同时存在 first-miss repair 低、harm 不低、远端候选被单均值压缩、candidate/context 交互弱、训练 support 与实际策略 reach 不一致的问题。
- **Non-goals**: 不把 K16 oracle 当成 selector 可实现收益；不靠复制 Domino 的顺序 GRU 或扩大 verification tree 来改题；首轮不声称 `T>0` lossless sampling；不把在同一 validation split 上选 checkpoint 和校准阈值后的数字当 sealed-test 结论；不同时堆 CRF、GRU、tree search 和多个辅助 head。
- **Constraints**: target model 冻结；首轮优先复用 released DFlash 与现有 100K Open-PerfectBlend canonical records；greedy `T=0`、block size 16、K16；最终部署 head 必须保持一次并行 lattice 处理且开销低于收益 break-even；现阶段有 Slurm A800 资源，但 validation_select 只有 147 prompts，必须保留独立 calibration/test 口径。
- **Success condition**: 高容量 frozen-feature probe 若达到约 `+0.6` raw EAL 或回收 ≥15% oracle gap，则构成当前输入充分性的正证据；紧凑模型需回收 probe 增益的 ≥70%，三 seed 均优于 DFlash，global−causal/local 的 prompt-cluster CI 排除 0，harm 的单侧 95% 上界不高于 5%，并在独立 calibration/test 与真实 latency 下取得正吞吐收益。低 probe 结果只触发工程上的 stop/pivot，不被表述为 frozen feature 不含信息的科学证伪。

## Anchor Check

- 原始问题仍是 frozen DFlash lattice 的并行、安全、可达 reranking；没有改成 joint backbone training 或 causal drafting。
- LoRA、DFlare-style fusion 和 Domino GRU 均不进入本提案。
- Reviewer 要求收窄 novelty ownership 不构成 drift，反而防止把 D-PACE 已有 continuation-value idea 重命名。

## Simplicity Check

- **Dominant contribution**: safe global full-lattice residual reranking for a frozen parallel drafter。
- **Coupled elements**: no-prepool candidate representation、unsmoothed candidate-support accepted-reach objective、block-balanced base-prefix margin regularizer；它们共同解决“global repair 与 frozen-base harm”的一个问题。
- **Removed**: ARR 独立 optimizer claim、ARR vs normalized alpha-zero CDP重复实验、slot、默认 distillation、动态 base scale、repair/coverage多项 loss。
- **Conditional only**: capacity probe、KEEP_BASE deployment calibration、未来压缩。

## Changes Made

### 1. Honest objective ownership

- Reviewer确认 ARR 的数学推导正确，但它与 length-normalized Candidate-D-PACE `alpha=0` 梯度严格相同。
- 代码把 `candidate_dpace` 定义为 top-K adaptation 的 length-normalized版本；历史 `dpace` alias保留官方 per-block scale以复现实验。
- 新增 gradient-parity unit test，ARR 与 normalized `candidate_dpace(alpha=0)` 的 score gradients逐元素一致。
- 论文不再声称首创 accepted-prefix utility；只声称在 frozen K-way residual selector中的 candidate-support specialization、global interface和base preservation。

### 2. Safety wording and block balance

- 固定权重 hinge 改称 **base-prefix margin regularizer**，不宣称训练时保证 5% harm。
- regularizer先在每个 block 的 DFlash accepted prefix内部取均值，再对 blocks等权平均；base EAL=0 的 block贡献0，避免长 prefix block主导。
- 5% 只作为独立 test 上的单侧置信上界要求；KEEP_BASE另列 calibrated variant。

### 3. Frozen implementation contract

- `B=16` 含1个 anchor和固定 `L=15` draft positions；`K=16`；canonical records无padding。
- candidate index 0必须是 descending DFlash logits的rank-1；collate阶段新增断言。`torch.argmax` tie时返回最低index，因此精确tie保留base。
- safe gather先将缺失gold index夹到合法范围，再由availability置 `q_i=0`；cumprod在首次gold-not-in-K处及之后为0，不计算 `log(0)`。
- compact默认：`d_model=128`、8 heads、2 flat pre-norm layers、FF multiplier 4、dropout 0、LayerNorm；总参数1,235,808。
- hidden使用无affine LN与独立projection；candidate/anchor共享无affine embedding LN和token projection；五个scalar features经 `5→128→128` projection。
- compatibility node使用 `[h,e,h*e,a*e]→256→128`；residual readout零初始化。
- 主 margin固定 `m=0.1`；只在development比较 `lambda∈{0,0.1,0.25}`，冻结后进入confirmatory。
- high-capacity positive probe固定 `d_model=640`、10 heads、4 flat layers、27,482,160参数。

## Revised Proposal

# Research Proposal: Safe Global Full-Lattice Reranking for Frozen DFlash

## Problem and Evidence

DFlash 一次并行预测15个 draft positions，成本低但最终仍按位置独立选 token。现有 GCLS-v1 的三-seed matched experiment证明其他位置 lattice有真实增量信息：global相对DFlash calibrated `+0.232` EAL，global−local `+0.172`、global−causal `+0.087`，两个prompt-cluster CI均排除0。但最佳checkpoint只修复18.3%的first-miss opportunities、伤害7.3%的blocks，回收6.2%的K16 oracle gap；与Domino on-policy仍差约1.62 EAL。

进一步诊断显示，当前coverage-based训练平均覆盖9.72 positions/block，而最佳greedy selector的实际prefix加breaker仅6.21；36.1%的positions位于当前breaker之后。更关键的是，同一global checkpoint换成local mask后non-top1 accuracy更高，EAL却下降约0.80，说明普通candidate classification不是正确优化对象。

## Narrow Technical Claim

在不修改target、released DFlash或verifier的前提下，若全局交互前保留全部candidate hypotheses，并用unsmoothed candidate-support accepted-reach gradient训练direct residual selector，同时对DFlash原始accepted prefix施加block-balanced margin regularization，则可以提高global repair的净EAL而不依赖顺序causal head。

本工作不声称首创expected acceptance objective。D-PACE已经给出continuation-value weighting；这里的区别严格限定于：

1. frozen parallel drafter后的K-way residual selector；
2. gold-not-in-K时的candidate-support censoring；
3. 无candidate预池化的global lattice interface；
4. frozen强baseline下的asymmetric base-prefix preservation。

## Method

### Fixed inputs

`B=16`由1个verified anchor和`L=15` draft positions组成；每位置取descending DFlash top-`K=16`。输入为DFlash hidden `h_i`、candidate IDs/logits、full-vocab logsumexp、anchor ID和frozen Qwen3 target embeddings。canonical collate断言candidate 0始终为DFlash top-1，无padding或deduplication。

### Candidate-context nodes

hidden经过无affine LayerNorm和独立projection；candidate与anchor embeddings共享无affine LayerNorm和token projection。对每个node：

```text
c_ik = MLP_4D_to_2D_to_D([h_i, e_ik, h_i * e_ik, a * e_ik])
z_ik = LN(c_ik + position_i + rank_k + scalar_MLP(phi_ik)).
```

`phi`包含full-vocab log-prob、K-way conditional log-prob、top1 gap、retained mass和entropy，先投影到model dimension。不存在trainable vocabulary table。

### Compact no-prepool full-lattice mixer

固定主模型为`D=128`、8 heads、2层flat pre-norm self-attention、FF ratio 4、dropout 0，共1,235,808参数。全部240个nodes一直保留到readout。matched local/causal/global只改变visibility mask，参数和初始化相同；Transformer block本身不作为架构新颖性claim。

### Base-anchored direct scores

```text
s_ik = log p_DFlash(C_ik) + Delta_ik.
```

readout零初始化，故epoch 0严格复制DFlash；per-position common residual offset被移除。greedy `argmax`的tie-breaking选择最小index，因此精确tie保留candidate 0。

### Unsmoothed candidate-support accepted reach

安全地gather gold candidate probability；若gold不在K内则置0：

```text
q_i = availability_i * softmax(s_i)[safe_gold_i]
S_t = product_{i<=t} q_i
U = sum_t S_t
L_reach = 1 - mean(U / 15).
```

首次gold-not-in-K令当前及后续`S_t=0`，实现candidate-support censoring。对batch内 `ell_i=-log q_i` 的有效位置：

```text
dL_reach / d ell_i = (1 / (batch_size * 15)) * sum_{t>=i} S_t.
```

这正是length-normalized Candidate-D-PACE在`alpha=0`和detached weights下的parameter gradient。实现以unit test固定等价性；二者不作为两个实验条件。历史主线的`alpha=0.5` smoothed D-PACE是唯一objective baseline。

### Block-balanced base-prefix margin regularizer

对每个block，定义DFlash从position 1起连续rank-1正确的accepted-prefix mask `b_i`。固定`m=0.1`：

```text
R_block = mean_{i:b_i=1} relu(0.1 + max_{k>0}s_ik - s_i0),
R_block = 0 if the base prefix is empty,
L = L_reach + lambda * mean_blocks(R_block).
```

它是regularizer而非harm guarantee。development只比较`lambda=0,0.1,0.25`；raw model为primary，KEEP_BASE threshold是单独的calibrated deployment variant。

## Training and Gates

1. **Deterministic sanity**: manual utility、finite difference、ARR/CDP gradient parity、identity init、scope invariance、frozen-input gradients、sorted top1 invariant。
2. **GPU smoke/capacity**: 128-block same-subset必须重新达到>95% repair和<1% harm；失败则不跑大数据。
3. **Objective isolation**: 固定current axial-additive模型，比较历史smoothed D-PACE与unsmoothed reach、reach+margin；不重复alpha-zero/ARR。
4. **Representation isolation**: 固定reach+margin，比较single-mean axial、flat additive、flat compatibility；matched scopes。
5. **Positive-only capacity probe**: d640/h10/l4、27.48M参数；512-block→10K→100K门禁。高结果是tested function class的sufficiency witness；低结果只作工程stop。
6. **Confirmatory**: 固定d128/h8/l2和lambda；checkpoint selection、KEEP_BASE calibration、untouched test三组prompt-disjoint；三seeds matched local/causal/global。

## Claim-Driven Experiments

### Claim A: unsmoothed reach + base margin改善net utility

- 历史smoothed D-PACE vs unsmoothed reach vs unsmoothed reach+margin。
- primary为raw prompt-balanced EAL；同时报告improved/harmed、first-miss repair、首token、最差domain、soft-U与greedy EAL跨epoch相关性。
- margin有效的判据是harm下降且EAL净增；不要求token accuracy上升。

### Claim B: global gain需要保留candidate hypotheses

- 固定loss，比较axial single-mean、flat additive、flat compatibility；每个结构做matched local/causal/global。
- primary为global−causal/local prompt-cluster CI、oracle-gap recovery、candidate/hidden replacement和真实latency。
- 若full lattice不优于axial，删除representation claim，绝不再加slot补救。

### Diagnostic: empirical sufficiency probe

- 只回答tested frozen inputs+function class能否出现显著高于compact model的held-out gain。
- 高probe−student gap才触发distillation；低probe不支持information-absence结论。

## Safety and Statistical Protocol

以prompt为bootstrap cluster。根据pilot的anchors/prompt相关性做正式power calculation；在结果出来前保守预留至少1,000 calibration prompts和2,000 untouched test prompts。最终5% harm要求是test单侧95% cluster-bootstrap upper bound，不是训练regularizer的保证，也不能由同一147-prompt development split或多seed替代。

## Failure Decisions

- reach+margin不优于smoothed baseline：删除objective adaptation，保留历史GCLS结论。
- full lattice不优于axial：删除architecture claim。
- compact增益高但latency失败：先减layers/dim；只有此时研究inducing compression。
- positive probe高而compact低：单独允许distillation；probe低则停止，不宣称证伪。
- frozen route最终仍远低于Domino：如实归因joint backbone/causal prefix/data confound；LoRA另立项目。

## Novelty Boundary

本提案的可发表性不依赖把ARR重新命名为新loss，而依赖一个可证伪的coupled claim：对frozen DFlash，global candidate evidence必须在pooling前保留，accepted-reach gradient必须按candidate support解释，且强base的harm必须被非对称保护。若任一claim的必要性消失，相应组件和论文claim一并删除。

## Compute

objective screen约8–16 A800 GPU-hours；compact representation screen约12–24；positive probe按门禁约15–35；三seed confirmatory约15–25。总预算50–90 GPU-hours，capacity/10K gate失败即停止后续。
