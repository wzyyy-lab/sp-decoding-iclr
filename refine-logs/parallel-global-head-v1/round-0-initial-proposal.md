# Research Proposal：PGMF-16 并行全局模式反馈 Head

> 状态：ARIS research-refine Round 0 初稿  
> 唯一有效约束：`USER_CONSTRAINT_CONTRACT.md`  
> 目标：一次读取完整 16-position DFlash block，非因果全局混合，一次并行输出唯一 16-token 序列

## Problem Anchor

- **Bottom-line problem：** 完全解决 DFlash 接受长度不高的问题；新方法的 held-out accepted length 必须显著超过 released Domino，并保持最终端到端吞吐优势。
- **Must-solve bottleneck：** DFlash 在 16 个并行位置的 Top-16 候选中包含大量正确 token，但现有逐位置或因果选择器不能利用完整 draft block 的全局一致性来同时选对这些 token。
- **Required mechanism：** 一个轻量、非因果的全局 head，一次读取全部 16 个 DFlash 位置，使每个位置都能看到整段 draft，然后并行输出唯一一条 16-token 序列。
- **Non-goals：** 不允许自回归/因果 token feedback、Domino GRU rollout、串行 target decode、target seed、迭代修复、beam、tree、trie、forest、多路径验证或额外 target inference。
- **Constraints：** Top-16 只能作为每个位置内部的候选维；线上只使用 DFlash 可得特征；初始新参数预算约 10.75M，可在 held-out 收益与 latency 证据支持下增加；先公平 eager 比较，最终同栈集成 SGLang。
- **Success condition：** fixed held-out EAL 和 dynamic EAL 均至少达到同次 released Domino 的 1.15x，同时最终 A40 SGLang end-to-end TPS 至少达到 Domino 的 1.15x。

## 1. 已知事实与真正缺口

当前 full-16 `validation_select` 的同锚点事实为：

| 系统/上界 | prompt-balanced EAL |
|---|---:|
| released Domino-DFlash backbone Top-1 | 6.068513 |
| released Domino | 7.239553 |
| 成功门槛 `1.15 × Domino` | 8.325486 |
| deployable full-16 base-Top-16 oracle | 10.909257 |
| base-Top-15 + released causal-Domino-action union（仅离线诊断，线上禁用） | 10.999879 |

PGMF 线上只能使用 `TopK(Z,16)`，不能把 released causal Domino action 混入候选集，
所以唯一合规的机制上界是 `10.909257`。因此新 head 需要从 backbone Top-1 到该上界的
`4.840743` 缺口中回收至少 `2.256973`，即 `46.6245%`；等价地，需要回收
Domino 到 deployable oracle 缺口的 `29.5918%`。这不是靠阈值或一次小修能完成的任务，必须让候选特异
表示真正利用完整 block。

已有合规或近似合规路线暴露了四个具体问题：

1. **GCLS-v1 有全局信号，但表示和容量不足。** 在较弱的 pure-DFlash backbone 上，
   全局模型相对 local/causal 有稳定增益，最佳 raw EAL 约 `+0.285`，证明未来位置的
   lattice 信息并非纯噪声；但它只回收少量 oracle gap。
2. **高容量 frozen selector 仍不够。** 27.48M 的 D640 frozen-feature 模型在 99,356
   prompts 上仅 `+0.07799`，说明单纯扩大同类 frozen scorer 不是主解。
3. **PLC 过早压缩且 teacher warm-start 数据不足。** PLC 虽是并行单链，但只预测
   15 个 correction positions；每位置候选先压成少量 modes，global mixing 后没有
   candidate-specific feedback。其 8.55M 版本的 teacher imitation 只用 1,024 blocks，
   随后的 acceptance stage 使用完整 15,886 train blocks，四个配置的 held-out 最好
   仍只有 `6.09342`，低于 Domino `7.23955`。
4. **GFPR/R043--R046 不检验本命题。** 它们主要适配 released causal head 或局部
   residual，缺少一个让全部 16×16 candidate nodes 双向交互后直接输出全序列的
   head；2K/10K 上的 transfer failure 不能替代这个结构和大规模 teacher curriculum。

核心判断是：冻结的轻量 scorer 已经不足，但“全局信息无效”也已被对照实验排除。
下一次主实验应把 **强 backbone、candidate-specific 两跳全局交互、大规模顺序教师
并行化 curriculum、target accepted-prefix 目标** 同时对齐，同时保持推理严格并行。

## 2. 方法：Parallel Global Mode-Feedback Head（PGMF-16）

### 2.1 线上输入

一次 Domino-DFlash parallel backbone forward 已产生：

- `H ∈ R[B,16,2560]`：全部 16 个位置的 parallel hidden；
- `Z ∈ R[B,16,V]`：backbone base logits；
- `C, B = TopK(Z, K=16)`：candidate IDs 与对应 logits；
- 当前 anchor token ID；
- frozen shared target embedding / LM-head rows `E[C]`。

这些张量全部在选择前可得。head 不读取任何 target future hidden/logit、gold token、
已选 token 或 verifier 结果。

当前 R047 records 已存 Top-16 logits/IDs，但没有存 `base_logsumexp`。首版 scorer 只使用
Top-16 内的 raw/conditional logits、gap 和 entropy；若需要 retained full-vocabulary mass，
必须用同一个 Domino-DFlash hidden 与 shared LM head 离线补存，严禁用
`target_logsumexp` 冒充 deployable base feature。

### 2.2 Candidate-specific node encoder

对每个位置 `i`、候选 `k`：

```text
q_i  = W_h RMS(H_i)
e_ik = W_e RMS(E[C_ik])
x_ik = LN(q_i + e_ik
          + W_mul(q_i * e_ik)
          + W_scalar(phi_ik)
          + pos_i + rank_k)
```

`phi_ik` 包含 raw/conditional log-prob、Top-1 gap、retained mass、entropy、rank。
乘法兼容项显式回答“该 hidden 与该候选 token 是否匹配”，避免旧 additive encoder
把 candidate identity 当成弱偏置。

### 2.3 固定深度的两跳全局交互

每个 feedback round 只有并行张量操作：

1. **Local candidate competition：** 每个位置的 K=16 candidate nodes 做无 mask
   local attention；4 个 learned mode queries 从该位置完整候选集合提取多模态状态。
2. **Global non-causal mixing：** 将 16×4 个 position-mode tokens、anchor token 和
   一个 block token 展平，运行无 causal mask 的双向 Transformer。任一 mode 都能
   attend 全部 16 个位置。
3. **Mode-to-candidate feedback：** 全部 16×16 candidate nodes 作为 queries，
   cross-attend 全部 global modes，并用 gated residual 更新自身。因此每个最终
   candidate score 都显式依赖整张 16-position lattice，而不是只依赖一个共享 pooled
   vector。

首版使用 2 个 feedback rounds、`d=384`、6 heads、4 modes/position、FFN ratio 2。
预计新 trainable parameters 约 8--11M；frozen token embedding 不计为新增参数，但
完整 active footprint 单独报告。若 held-out 同时显示 train/validation 欠拟合，才升至
`d=512` 或 3 rounds；若 train 高、held-out 低，则不以加参数掩盖迁移问题。

### 2.4 一次并行直接输出

最终每个 candidate node 产生 residual 和受控 base temperature：

```text
alpha_i = 1 + 0.5 * tanh(g_i)          # 初始化为 1
delta_ik = W_score(x_ik)               # 初始化为 0
s_ik = alpha_i * logp_base_ik + delta_ik
proposal_i = C_i[argmax_k s_ik]
```

一次 head invocation 输出 `S ∈ R[B,16,16]`，随后一个张量 argmax 得到唯一
`proposal ∈ N[B,16]`。没有 path score、Viterbi、DP、GRU、token loop、第二次 head、
迭代 refinement 或 target-side branching。

### 2.5 为什么它不同于失败实现

- 相对 GCLS：使用更强的 released Domino-DFlash backbone；不是单次 flat averaging，
  而是 `candidate → multiple global modes → same candidate` 的显式两跳反馈；用大规模
  Domino sequence curriculum，而不是仅靠 hard per-position gold CE。
- 相对 PLC：不把候选先压缩后直接生成一个共享 correction code；global modes 会反馈
  到每个 candidate node，输出是直接 `[16,K]`；同时覆盖 position 0 和完整 16 positions。
- 相对 Domino：训练时可以蒸馏其顺序知识，但推理没有 GRU rollout 或 selected-token
  feedback，16 个位置完全并行。
- 相对 R050--R056：没有额外 target decode，也没有树、多链或宽 verifier；target 仍只
  执行普通单链 verifier。

## 3. 训练：把顺序教师知识并行化，再直接优化 target prefix

### 3.1 Stage A：大规模 full-16 parallelization warm start

已有 25K prompts / 199.8K blocks 的
`qwen3_4b_domino_opb100k_half_part{000..003}_10157195` 只含 15 个 draft positions，
不能直接作为 claim-bearing PGMF 训练数据。先用相同四个 OPB prompt shards 和 released
Domino-DFlash backbone，重采为严格 full-16 schema（约 25K prompts / 200K blocks）：

- 每条输入是一次 non-causal released Domino-DFlash forward 得到的 `H[16]`；
- 用 shared LM head 重建线上 exact Top-16；
- 离线记录 16 位 released policy action、gold、target candidate posterior；
- target early hidden/feature 不得写入在线输入；
- 从数据收集、训练、checkpoint 选择到 evaluator 始终使用完全相同的 L=16 几何。

先审计 released action 与 gold 的 Top-16 coverage。unsupported row 不伪造 label，
只参与可用的 base/protection 项。curriculum 前 20% updates 以 released sequence
imitation 稳定学习，随后线性衰减 teacher 权重并转向 target prefix utility。

现有 full-16 R047 的先验审计表明该 curriculum 可行：train/validation 的 released
policy action 位于 base Top-16 的比例分别为 `91.590%/91.213%`，position 0 为
`100%/100%`；gold support 为 `83.709%/83.106%`，position 0 为
`99.433%/98.894%`。Stage-A 199.8K blocks 仍需独立重算同一统计。

Stage A 的意义不是把 Domino 当最终上界，而是将其 1.42M-prompt 训练得到的 causal
sequence knowledge压入一个 non-causal parallel student；gold objective 负责超过 teacher。

### 3.2 Stage B：full-16 target posterior adaptation

使用 `r047_anchor_t4_train_10164718` 的 1,987 disjoint train prompts / 15,886 blocks：

- 完整 `H[16]`、Top-16、target candidate logits/advantages、gold、released policy；
- Stage A checkpoint 只作初始化；
- 低学习率训练全部 PGMF head；
- 首轮不解冻 DFlash，先判断 head 机制本身；
- 若 Stage A 已达到 Domino 附近而 Stage B 明显欠拟合，可在后续只对 DFlash 最后两层
  加可合并 LoRA，并与 head 一起用大规模 raw-context 重算训练。LoRA 不增加推理 forward，
  但未经 held-out 证据不提前加入首个 falsifier。

### 3.3 主目标函数

设 head 的 candidate distribution 为 `q_i`，gold 在 support 时的概率为 `q_i(y_i)`。
主项直接最大化 soft accepted-prefix utility：

```text
U(q) = sum_t exp(sum_{j<=t} log(clamp(q_j(y_j))))
L_prefix = -mean_block U(q) / 16
```

其梯度天然优先作用于早期 breaker。为保证 future lattice 本身能学到语言模式，保留
小权重 all-position target candidate KL/CE；在 released policy 的真实正确 prefix 上
使用 margin protection，防止一个错误 override 摧毁长 prefix。总目标为：

```text
L = L_prefix
    + lambda_dense * L_target_candidate_KL
    + lambda_protect * L_released_correct_prefix_hinge
    + lambda_teacher(t) * L_domino_sequence_imitation
```

其中 teacher 项在 Stage A 衰减，Stage B 为零或极小；dense 项不得主导 checkpoint
selection。checkpoint 只按 held-out full-16 prompt-balanced EAL 选择，不能用 token CE
或 same-set capacity 代替。

## 4. 数据、泛化与评测

### 4.1 数据边界

- Stage A：OPB part000--003 重采后的 full-16 train，25K prompts；
- Stage B：Phase3 train，1,987 prompts；
- selection：147-prompt `validation_select`；
- sealed efficacy：`validation_gate`，只在 architecture、loss、checkpoint rule 冻结后打开；
- dynamic：用同一 frozen checkpoint 重新 rollout，不用 fixed cache 代替；
- 训练/selection/gate 按 sample ID 强制 disjoint。

若 25K Stage A 显示清晰 scaling 而尚未达到 Domino，可补齐现有 OPB 其余 75K 的
Domino-backbone full-16 cache；这是同一方案的数据扩展，不改变架构，也不允许看 sealed
gate 后再调方法。

### 4.2 机制 sanity（先小后大）

在任何 full run 前必须通过：

1. `forward(H[16], ...) -> scores[B,16,16] -> one proposal[B,16]`；
2. 修改任一远端 position 的合法输入，其他 position scores 会改变；
3. 无 causal mask、无 selected token 参数、无 token-position Python loop；
4. zero output projection 精确复现 backbone Top-1；
5. 512-block same-set capacity 能大幅回收 oracle gap，仅作为优化/索引检查；
6. train/validation IDs 无交集；target tensor只出现在 loss，不进入 forward signature。

### 4.3 决策门

- **Stage-A parallelization gate：** full-16 validation EAL 至少接近 released Domino，
  且相对 backbone Top-1 有大幅、全域提升；否则先检查 teacher support 和表示，再决定
  d384→d512，绝不转 causal/tree。
- **fixed efficacy：** `EAL >= 8.325486`，即同 job Domino 的 1.15x；三域均不退化。
- **dynamic efficacy：** `EAL >= 1.15 ×` 同 job dynamic Domino。
- **system：** 机制过门后才做 eager-to-eager完整 head profile；最后同栈 SGLang A40
  tokens/s 至少是 released Domino 的 1.15x。

## 5. 速度路径

PGMF 不做 full-vocab correction GEMM；DFlash 已有 base logits，新增路径只有：

```text
Top-16/gather
→ 256 candidate node projection
→ local modes
→ 65-token左右的双向 global mixer
→ 256×mode feedback
→ [16,16] score/argmax
```

所有 shape 固定、无数据依赖循环，适合 BF16、SDPA/Triton 与 CUDA graph。首版新参数
约 2% DFlash draft，完整 eager latency 与 Domino eager 公平比较；不能用参数量代替实测，
也不能用未优化 Domino 对比最终 optimized head。

## 6. 风险与预先响应

1. **Teacher action 不在 Top-16：** 先报告 coverage；不扩大 target-side verifier。必要时
   训练时 K=32、部署 K=16 只能作为 representation pretraining，最终 efficacy 必须 K=16。
2. **Stage A 模仿 Domino 但不能超越：** 检查 gold-prefix objective 与 target posterior；
   增加完整 OPB full-16 target training，而不是增加 serial target inference。
3. **train fit 高、held-out 低：** 说明数据/表示迁移问题；优先扩 prompt diversity、
   shared lexical projection、可合并 backbone LoRA，不继续堆 frozen head。
4. **head 欠拟合：** 只在 train 和 validation 同向欠拟合时增加 width/rounds；每次同时
   测参数和 latency。
5. **固定 EAL 过门但 dynamic 失败：** 重新收集同一 policy 的 on-policy blocks并继续
   训练；仍保持一次 parallel head 和唯一单链。
6. **latency 吞掉收益：** 缩 modes/width、融合 gather/projection/feedback；不牺牲成树验证。

## 7. 贡献边界

若成功，主贡献是：

> 将 causal sequence teacher 的一致性知识蒸馏到一个 candidate-specific、固定深度、
> 全局非因果的 parallel lattice head，通过 candidate→global modes→candidate feedback
> 在一次 forward 中直接生成唯一 draft sequence，并将 Top-16 availability 转化为实际
> accepted length 和端到端吞吐。

不声称 Top-16 oracle 可直接实现，不声称 attention/LoRA/蒸馏单独新颖，也不把任何
causal、serial-target 或 tree 结果归入本方法。

## 8. 不可变合规证明

| 用户要求 | PGMF-16 |
|---|---|
| 一次读取全部 16 positions | 是，`H/Z/C` 均为 `[B,16,...]` |
| 每位置看到整段 | 是，无 mask global modes + candidate feedback |
| 一次并行预测全部位置 | 是，单次 `[B,16,K]` 输出 |
| 唯一一条序列 | 是，一个 per-position argmax tensor |
| 无 selected-token feedback | 是，forward signature 无 selected tokens |
| 无自回归/GRU/DP/迭代 | 是 |
| 无树/beam/多路径 verifier | 是 |
| 无额外 target inference | 是，target 只作离线 labels 和普通 verifier |
| 轻量且可融合 | 首版约 8--11M，固定 shape |
| 主目标优先 | 先 held-out EAL，过门后再系统集成 |
