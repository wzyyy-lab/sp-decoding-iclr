# Research Proposal: Signed Action-Value Selection for Frozen DFlash Lattices

## Problem Anchor

- **Bottom-line problem**：在不增加 target call、顺序解码或 verification
  tree 的前提下，利用 released DFlash 已生成的 frozen K16 candidate
  lattice，提高 Qwen3-4B greedy speculative decoding 的真实 accepted
  draft length。
- **Must-solve bottleneck**：已有 Direct selector 只能获得约 `+0.285`
  development EAL；flat FMAS action CE 虽降低分类损失，却造成
  `31.7--35.0%` harm 和 `-0.424--0.505` EAL，说明监督目标没有表达错误
  编辑的非对称 prefix 代价。
- **Non-goals**：不修改 DFlash/target backbone；不引入 GRU、CRF、tree
  search、LoRA、额外候选、target inference、post-hoc threshold 或 sealed
  data；不把 capacity memorization 当作泛化或信息充分性证据。
- **Constraints**：固定 `T=0`、15 个 draft positions、K16、一次并行
  lattice head、最多修改一个 token；开发阶段只使用物理隔离的
  `validation_select`。
- **Success condition**：先通过严格的 512-block capacity falsifier；之后
  只有在 prompt-diverse seed-0 同时超过 DFlash 和匹配 Direct controls、
  harm 不超过 5% 时，才允许多 seed 或 formal evaluation。

## Evidence and Technical Gap

在 1,175 个隔离 development blocks 上，全部 264,375 个非 KEEP 动作的
真实效用为：

| signed utility | actions | fraction | mean nonzero magnitude |
|---|---:|---:|---:|
| beneficial | 984 | 0.3722% | +1.829 tokens |
| harmful | 90,120 | 34.0879% | -5.304 tokens |
| neutral | 173,271 | 65.5399% | 0 |

Flat 226-way CE 只区分 canonical class，不区分 neutral mistake 和会损失
多个 prefix tokens 的 harmful mistake。新方案只改变监督语义和输出解释，
保持 action space、可见特征、主干和部署复杂度不变。

## Method: SAVS

对 block `x`，令 `A(a,x)` 为 KEEP 或一次编辑 `a` 的真实 accepted-prefix
length，`b(x)=A(KEEP,x)`，`L=15`。构造所有 225 个编辑的 dense target：

```text
v(a,x) = [A(a,x)-b(x)] / L,
v(KEEP,x) = 0.
```

Gold token 只在训练和评估时构造 target。复用未修改的
`GlobalDirectCandidateSelector`，但仅解释其 learned residual：

```text
v_hat(KEEP) = 0,
v_hat(i,r) = rho[i,r]-rho[i,0],  r=1,...,K-1.
```

per-position residual mean centering 在差分中消去。residual projection
精确零初始化，因此所有 edit value 在 epoch 0 都精确为零。

唯一训练目标是 action-uniform MSE：

```text
L_value = mean_blocks mean_225_edits (v_hat(a,x)-v(a,x))^2.
```

不使用 class rebalance、focal loss、CE auxiliary、reward temperature、
learned KEEP bias 或可调阈值。推理时只在最大 edit value **严格大于零**
时应用该编辑，否则 KEEP；因此初始模型逐样本精确等于 DFlash，且路径只
改变零或一个位置。

## Statistical Claim Boundary

在 inference features 给定、容量充分且 population MSE risk 被最优化时，
逐 action 的 Bayes 解是 conditional mean signed utility；选择最大严格正
conditional mean 对 expected incremental accepted length 是
Fisher-consistent。该结论只说明 target semantics 正确，不声称有限共享
模型的 action-average RMSE 能控制 max-policy regret、泛化或信息充分性。

## Frozen Decision Diagnostics

令 `E(x)` 为 225 个 edit actions，`a_hat` 为部署动作，
`v*(x)=max(0,max_a v(a,x))`。必须固定报告：

```text
P(max_a v_hat(a,x)>0 | max_a v(a,x)<=0)       # no-benefit false edit
P(v(a_hat,x)<0)                               # deployed harm
P(v(a_hat,x)>0 | a_hat!=KEEP)                 # edit precision, no edits => NA
E[v*(x)-v(a_hat,x)]                           # selected-action regret
```

同时报告 positive/neutral/harmful 三类的 count、mean MSE、SSE fraction、
prediction signs、mean prediction，以及 epoch-0 各类对 output projection
gradient 的 norm/cosine。RMSE 仅为 fidelity 指标：若每个 block 都只有
一个 `0.30` outlier，其余 224 个 action 精确，RMSE 仍为 `0.02`。

## Gate 0: CPU Semantics

单测必须证明：

1. dense targets 与 brute-force one-edit decoding 完全一致；
2. beneficial/neutral/harmful fixtures 的 token 与 normalized utility 精确；
3. epoch-0 residual values 全零，所有 tie KEEP；
4. decoder 只改变零或一个位置并使用 strict-positive gate；
5. 第一次 backward 只有 zero-initialized residual projection 获得非零
   gradient；更新一次后第二次 backward 才传播到 upstream，frozen inputs
   始终无 gradient；
6. EAL、domain/prompt balancing、repair、harm、sign、regret、class loss 和
   gradient diagnostics 可由 example records 精确重建。

## Gate 1: Capacity-Only Falsifier

使用 frozen 512 training-block manifest：D64/H4/L1 axial-additive、K16、
batch32、320 epochs、seed0、LR `6e-4`、warmup `0.04`、zero dropout/weight
decay、恰好 5,120 steps。checkpoint 只按 minimum dense-value MSE 选择，
完全相等时保留最早 epoch。

联合通过条件：

- normalized all-action RMSE `<=0.02`；
- beneficial strict-positive recall `>=0.99`；manifest 恰有 256 positives，
  即至少 254 个正确、最多 2 个 miss；
- harmful nonpositive recall `>=0.99`；
- decoded one-edit oracle-gap recovery `>=0.95`；
- selected-action harm `<=0.01`；
- finite gradients 和 exact epoch-zero identity。

RMSE 是工程 fidelity threshold；oracle-gap recovery 和 harm 才是行为门禁。
失败只关闭这套 D64 parameterization、unweighted MSE、optimizer/schedule、
subset composition 和 checkpoint rule 的组合，不能区分 objective、容量、
优化或 identifiability。无论失败原因如何，都不追加 D640 rescue。

## Gate 2: Conditional Development

只有 Gate 1 通过且新的独立 code/result review 给出 GO 后，才允许一个
seed-0 full-data run：99,356 prompts、793,989 blocks、batch64、3 epochs、
37,221 steps、LR `6e-4`，selection 使用物理隔离的 147 prompts。
checkpoint 依次按 raw prompt-balanced EAL、lower harm、lower dense MSE；
严格改进才替换最早 checkpoint。

该 selection set 只作后续路由，不作论文效果估计。全部条件同时满足才
advance：

- `EAL_SAVS-max(EAL_Direct-native,EAL_Direct-one-edit) >= 0.05`；
- `EAL_SAVS-EAL_DFlash > 0.28499`；
- harmed fraction `<=0.05`；
- first-token accuracy 相对 Direct-native 下降不超过 `0.001`。

不授权 calibration、threshold sweep、seeds 1/2、formal test 或 rollout。

## Novelty and Claim Scope

SpecDec++ 已预测 acceptance probability 并做 stopping；Hybrid Verified
Decoding 已回归 accepted-length payoff 选择 draft source；BASTION 已用
expected-acceptance surrogate 分配 tree budget。因此 value regression、
accepted-length prediction 和 payoff-guided selection 本身都不是新颖性。

唯一待实验验证的差异是：对 frozen DFlash lattice 的全部 225 个
base-preserving one-edit interventions 构造 counterfactual signed
prefix-advantage supervision，并以 exact-identity residual-difference head
和 strict-positive KEEP policy 部署。负结果只关闭这个精确路线，不能证明
frozen features 存在信息上限。

## Authorization Boundary

- READY 后允许：CPU implementation 与 semantic tests。
- fresh experiment-bridge GO 后允许：一个 D64 512-block capacity job。
- full-data、多 seed、formal test 与真实 rollout 均需后续独立授权。
