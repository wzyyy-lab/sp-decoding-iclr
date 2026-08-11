# Research Proposal：PARC-16 — Parallel Acceptance-Risk Correction

## Problem Anchor

- **Bottom-line problem：** 在不改变 DFlash 一次并行生成完整 16-token block 的前提下，设计一个轻量 head，让完整草稿中的每个位置在最终选 token 前都能看到其余全部 15 个位置，并一次同时输出唯一一条 16-token 序列；fixed EAL、dynamic EAL 和同栈 A40 SGLang 端到端吞吐都必须至少达到 same-job released Domino 的 `1.15x`，越高越好。
- **Must-solve bottleneck：** 当前 full16 disjoint development 的 pure DFlash EAL 为 `6.0685131195`，released Domino 为 `7.2395529640`，硬目标为 `8.3254859086`，pure-base Top16 oracle 为 `10.9092565598`。已有证据排除了“候选不存在、head 太慢、same-set 容量太小、只需增加训练步数或扩大 selector”这些解释：PGCF 在 512 blocks 可精确拟合但 disjoint EAL 仅 `6.10277`；JAPD-D256 在容量集达到 `99.8627%` candidate accuracy 但 broader same-set recovery 仅 `5.5133%`、harm `18.1641%`；PCLD-16R 在稳定训练 support 上达到 candidate agreement `99.9876%`、J2 `314/314` 和 EAL `9.52539`，但完整 early-error population 只有 `322/411=78.3455%`、oracle recovery `66.9209%`、harm `6.25%`。同时，rank-16 LoRA、588M full adaptation 和 full-vocabulary KL 的 held-out 增益接近零。新方法必须直接解决两个相连问题：把整条 DFlash 并行草稿作为一个 noisy sequence 做全局一次性纠错；并以完整 accepted-prefix 收益和破坏 base 正确前缀的风险共同决定 `KEEP` 还是编辑，而不是继续在过滤 support 上拟合平均 token loss。
- **Non-goals：** 不做 Domino/GRU/Markov 式自回归，不做 selected-token feedback，不做串行 target seed/decode，不做 Jacobi 或任何迭代 refinement，不做 Viterbi/DP sequence decoding，不做 beam/tree/trie/forest/multipath，不让 Top16 变成路径维，不增加 ordinary verifier 之外的在线 target inference，不复活 PCLD/JAPD/PGCF 的 width、LR、schedule、loss-weight 或 threshold sweep，不以 R050–R056 的 off-spec 结果授权主线。
- **Constraints：** 单次 head 必须同时消费完整 `[B,16,*]` DFlash online features；每个输出位置必须通过无 causal mask 的全局 mixer 看到完整 16-position provisional sequence；一次产生 `[B,16,16]` scores，并以一次逐位置 argmax 得到唯一 `[B,16]`。Top16 只作每位置候选轴；最终被选择的 token 在 argmax 前不得反馈给任何位置。线上只能使用 ordinary DFlash hidden/base logits/Top16 IDs、冻结 embedding/LM-head rows、anchor/context feature；target clean continuation 只作离线监督。新增在线参数首版不超过 `10.75M`，必须 eager-to-eager 公平 profile，且 accepted-length 主机制过 disjoint gate 前不做 SGLang 工程。
- **Success condition：** 新方法先在 same-set capacity 上同时达到完整 clean-prefix recovery `>=95%` 与 harm `<=1%`，随后在严格 prompt-disjoint held-out 上证明全局机制而非记忆：fixed EAL 至少 `8.3254859086`，dynamic EAL 至少 same-job Domino 的 `1.15x`，三个域均不退化；最后同栈 SGLang A40 tokens/s paired 95% CI 下界至少 `1.15x` Domino。任何串行、迭代、多路径、额外在线 target、same-set 替代 held-out、或只提高 suffix/token accuracy而未提高 accepted prefix 的结果都不算成功。

## Technical Gap

现有 global heads 把任务主要写成“在每个位置从 Top16 做分类”。即便 mixer 有
full-block receptive field，输出仍以 base-logit residual 和平均 candidate loss 为
中心：远 rank 修复需要跨越 raw logit gap，base 已正确的 early prefix 与可编辑
frontier 没有显式不同的 action semantics，过滤后的 stable support 还会移除最难的
early-error events。结果是 capacity 上可以记忆，但在更完整的数据上发生两种错误：
大量 suffix-only edit 不贡献 EAL，少量 early false edit 却立即造成 harm。

单独加大模型、解冻 backbone 或蒸馏更多 target logits 不足以修复这个决策问题。
D-PACE 已经说明 prefix confidence 和 continuation value应动态决定训练权重，但它
没有处理“相对一个已有 base proposal，什么时候应该 KEEP”的约束。Domino 通过
已生成 prefix 获得强信息，但串行依赖违反本项目约束。最小缺失机制因此不是另一种
teacher hidden，而是一个把整条 base proposal 当作 noisy codeword 的一次性纠错器，
其 action space 和 objective 都直接区分有收益 edit 与破坏正确前缀的 edit。

## Method Thesis

- **One-sentence thesis：** 把 DFlash 的 16-token Top1 block 显式作为 provisional noisy sequence，用一个轻量 16-position global non-causal head 同时预测每位置相对 `KEEP` 的 Top16 edit advantage，并在联合训练中最大化 accepted-prefix utility、以 primal-dual constraint 控制 base-prefix harm，可在不引入任何 token feedback 的情况下学到比 Domino 更长的唯一并行草稿。
- **Why this is the smallest adequate intervention：** 线上只增加一个 position-level Transformer 和 candidate scorer；不增加 teacher branch、latent code、第二次 head、decoder、search 或 router。`KEEP` 就是 candidate rank 0，不新增输出空间。
- **Why this route is timely：** PTP 与 SpecFormer 已表明 dependent token 的单次并行建模是可行方向，D-PACE 已表明 acceptance-aligned training有效；PARC 把两者收缩为 DFlash Top16 单链 correction 的安全 edit policy，而不复制 causal prefix rollout。

## Contribution Focus

- **Dominant contribution：** risk-constrained one-shot block correction：一个以整条 provisional sequence 为输入、以 `KEEP`-relative edit 为输出、直接受 accepted-prefix gain/harm 约束的并行单链 head。
- **Optional supporting contribution：** base-anchored joint training，使已有 DFlash backbone 对同一个 global correction objective暴露可迁移表示；它不增加部署组件。
- **Explicit non-contributions：** 不声称发明 bidirectional attention、TopK reranking、expected-acceptance weighting、target distillation或 parallel token prediction；不改变 verifier。

## Proposed Method

### Complexity Budget

- **Frozen / reused：** target model、target embedding/LM-head rows、ordinary verifier、DFlash 现有 5-layer inference graph、base vocabulary GEMM/Top16。
- **New trainable component：** 一个 PARC head，首版 `D512/H8/L3/FFN1024`，预计 `9.47M` 参数；M0 必须精确断言且不得超过 `10.75M`。
- **Existing weights updated in training：** DFlash backbone 可在 main training 中联合更新，但没有新增层；target 与 shared lexical table保持冻结。
- **Intentionally excluded：** target hidden sidecar、Domino GRU/code、LoRA rescue、multiple experts、hard threshold router、candidate tree、iteration、第二次 head。

### System Overview

```text
ordinary target-prefix feature/cache
             |
one DFlash non-causal block forward
             |
 H[16] + base logits -> pure Top16 C,Z -> provisional x0=C[:,:,0]
             |                         |
             |              soft candidate summary per position
             +-------------------------+
                         |
       3-layer full-attention Transformer over exactly 16 positions
                         |
      all-position states z[16] + local candidate rows E[C]
                         |
           KEEP-relative edit advantages A[16,16]
                         |
        one tensor argmax -> exactly one proposal[16]
                         |
                ordinary single-chain verifier
```

### Core Mechanism

#### Input representation

For position `i`, let `C_ik` and `Z_ik` be pure-base Top16 IDs/logits, with
`C_i0` the base Top1 token. Let

`p_ik = softmax_k(Z_i)_k`

and let `E` be the frozen shared lexical row. The soft candidate summary is

`m_i = sum_k p_ik RMS(E[C_ik])`.

The position input is

`u_i = LN(W_h RMS(H_i) + W_e RMS(E[C_i0]) + W_e m_i + W_e RMS(E_anchor) + W_phi phi_i + p_i)`.

`phi_i` contains only bounded online base statistics: Top16 entropy, retained
mass, Top1/Top2 margin, normalized position and base confidence. Thus every
position exposes both a concrete provisional token and the uncertainty of its
complete candidate set.

#### Global one-shot denoiser

`u[0:16]` passes through three pre-norm Transformer blocks with no attention
mask. All 16 states are computed together; every output state reads all 16
provisional tokens and all 16 soft candidate summaries. Hidden depth is
ordinary feed-forward depth, not position recurrence. There is no intermediate
argmax or selected-token embedding.

#### KEEP-relative candidate interface

The same frozen/preprojected `W_e E[C_ik]` rows form candidate features. For
`k>0`:

`d_ik = w_o^T SiLU(W_z z_i + e_ik + W_mul(z_i * e_ik) + W_s psi_ik)`.

`w_o` and its bias are zero initialized. The final relative scores are

`A_i0 = 0`,

`A_ik = (Z_ik - Z_i0) + d_ik`, for `k=1..15`.

Rank 0 is therefore an immutable explicit `KEEP` reference. At initialization,
the complete score ordering and selected tokens are exactly pure DFlash. An
edit must provide enough global evidence to overcome the actual base margin;
the head cannot silently move the KEEP baseline. All 16 rows are produced in
one tensor and one `argmax_k A` yields the unique final chain.

#### Prefix utility and harm constraint

For a target-generated clean continuation, let `r_i` be the gold rank in
Top16. The active horizon `h` is the first position whose gold is outside
Top16; it and its suffix are not selectable and are excluded. Let

`q_i = softmax(A_i)` and `s_i=q_i(r_i)`.

The head-side dynamic prefix loss uses the D-PACE form on this live policy:

`qtilde_i = 0.5 * stopgrad(s_i) + 0.5`,

`w_i = stopgrad(sum_{t=i}^{h-1} prod_{j=0}^{t} qtilde_j)`,

`L_gain = sum_{i<h} w_i * (-log s_i)`.

Let `a_base` be the number of consecutive rank-0 gold actions before the base
first rejection. A model has lower accepted length than base if it breaks any
of these protected positions. Its differentiable per-block risk proxy is

`R = 1 - prod_{i<a_base} q_i(0)`, with `R=0` when `a_base=0`.

Training solves

`min L_base_DPACE + L_gain`, subject to `prompt_mean(R) <= 0.01`,

using projected primal-dual updates for a nonnegative Lagrange multiplier.
There is no manually tuned harm-loss weight and no post-hoc inference
threshold. Binding evaluation still uses actual deterministic accepted length
and actual harmed-block fraction; the probability constraint is a training
surrogate, not a formal guarantee.

### Modern Primitive Usage

- **Primitive：** large target as offline data generator plus one-shot
  bidirectional sequence denoising and constrained policy optimization.
- **Role：** target-generated responses supply clean token labels only; the
  global head learns to correct a noisy DFlash codeword. No target hidden,
  logits, critic, or forward is used online.
- **Why natural：** the project already has a full block proposal before final
  selection; treating it as a noisy sequence directly matches the information
  available at inference, unlike constructing an autoregressive state that is
  forbidden online.

### Integration into DFlash

During main training, Top16 is recomputed live from the current DFlash logits.
The target embedding/LM head remains frozen. Gradients from `L_gain` reach the
PARC head and DFlash hidden states; full-vocabulary `L_base_DPACE` continues to
train/protect the base branch. Candidate IDs are discrete TopK selections, as
in ordinary candidate reranking; no gradient is claimed through the IDs.

At deployment, lexical projection is materialized once as a frozen BF16
lookup table. Runtime executes exactly one DFlash block, one base vocab GEMM /
Top16, one PARC forward and one argmax before the ordinary verifier. Joint
training changes weights but not graph depth or call count.

### Training Plan

1. **M0/M1 semantics and capacity：** use 512 fixed target-generated blocks;
   first freeze DFlash to prove full16 geometry, exact identity, gradients and
   that the head/loss can reach `>=95%` oracle-gap recovery with `<=1%` harm.
   Same-set evidence is never treated as generalization.
2. **25K prompt pilot：** train live Top16 with DFlash+PARC jointly on disjoint
   target-generated prompts. Use prompt-balanced batches, base D-PACE and
   head constrained loss from step 1; select checkpoint only by disjoint fixed
   EAL under the frozen harm gate.
3. **Conditional scale：** only if the 25K pilot shows positive global and risk
   effects, scale the identical method/data recipe toward 100K and then the
   available Domino-comparable corpus. Do not introduce target hidden/KL or
   architecture rescue during scale-up.

### Failure Modes and Diagnostics

- **Capacity fails：** the KEEP-relative interface/objective is not
  optimizable; fix only proven implementation errors, otherwise close PARC.
- **Capacity passes but global equals local on disjoint prompts：** remote block
  context is not supplying useful correction evidence; stop instead of adding
  recurrence or tree search.
- **EAL rises but harm exceeds 1%：** the constrained surrogate is not
  controlling deterministic tail risk; do not hide this with a post-hoc
  threshold.
- **Frozen head works but joint training collapses base：** base D-PACE or
  gradient routing is broken; inspect base EAL and candidate coverage before
  changing architecture.
- **Latency misses the eager guide：** test a deletion-only D384/L3 version
  with identical semantics; do not trade away global visibility or add a
  multi-stage decoder.

### Novelty and Elegance Argument

The proposal does not rely on “another bidirectional Transformer” as novelty.
Its focused object is a correction policy relative to an already competent
parallel proposal: `KEEP` is an explicit reference action, every edit is
conditioned on the entire provisional sequence, and training is the constrained
optimization of accepted-prefix gain versus base-prefix damage. D-PACE has no
base-relative edit constraint; Domino/DSpark obtain dependency from selected
prefix tokens; PTP/SpecFormer do not define this DFlash Top16 risk-controlled
correction interface. The architecture and objective are one mechanism rather
than a pile of auxiliary teachers.

## Claim-Driven Validation Sketch

### Claim 1：PARC can safely optimize full-prefix correction

- **Minimal experiment：** 512-block capacity followed by a 25K-prompt
  disjoint pilot.
- **Baselines / ablations：** PARC-global; parameter-matched local-attention
  control; global without the primal-dual risk constraint.
- **Metric：** prompt-balanced EAL, base-to-Top16-oracle recovery, harmed-block
  rate, first/second-frontier success, global-local paired bootstrap.
- **Expected evidence：** capacity recovery `>=95%`, harm `<=1%`; on disjoint
  data global exceeds both controls without domain regression.

### Claim 2：the accepted-length gain can translate into system advantage

- **Minimal experiment：** only after Claim 1, freeze one checkpoint and measure
  fixed/dynamic EAL, complete eager head cost, then paired same-stack SGLang.
- **Baselines / ablations：** released Domino and pure DFlash under identical
  backend, precision, block and workload.
- **Metric：** fixed/dynamic EAL ratio and SGLang tokens/s ratio with paired
  prompt confidence interval.
- **Expected evidence：** both EAL ratios `>=1.15x` Domino, three domains
  non-regressing, SGLang TPS 95% CI lower `>=1.15x`.

## Experiment Handoff Inputs

- **Must-prove claims：** useful global provisional-sequence evidence; risk
  constraint controls actual harm; final EAL and throughput both exceed Domino.
- **Must-run ablations：** matched local visibility and no-risk constraint only.
- **Critical data / metrics：** target-generated prompt-disjoint blocks;
  prompt-balanced accepted length, harm, frontier repair, domain split,
  complete eager and same-stack TPS.
- **Highest-risk assumptions：** full-block DFlash observables contain enough
  transferable evidence; primal-dual probability risk predicts deterministic
  harm; 25K-to-larger-scale joint training improves rather than shortcuts base.

## Compute & Timeline Estimate

- **M0/M1：** one A40, under 0.5 GPU-hour including profile.
- **25K pilot：** approximately 10–30 A40 GPU-hours depending on live backbone
  recomputation and gradient checkpointing; three matched arms can run in
  parallel if allocation permits.
- **Conditional 100K scale：** approximately 40–120 A40 GPU-hours; no scale-up
  if the 25K mechanism gates fail.
- **Timeline：** 0.5 day for semantics/capacity and review, 1–2 days for 25K
  pilot, then conditional multi-day scale and SGLang integration.

