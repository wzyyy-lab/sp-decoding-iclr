# Round 2 Refinement: GFPR — Unified Full-Vocabulary Greedy-Frontier Policy Replay

## Problem Anchor

- **Bottom-line problem:** “最重要的目标是完全解决接受长度不高的问题，一定要超过 Domino，越高越好”；主目标不是小修小补，而是让 Top-16 中已有的正确 token 真正转化成更长的可接受前缀。
- **Must-solve bottleneck:** 当前 released Domino 在 exact B16 runtime held-out 上的 prompt-balanced EAL 是 `7.23955`，DFlash Top-16 oracle 是 `10.25449`，但现有 GLCS 训练后仍只有 `7.15–7.22`。必须解决“同集合可记到 oracle、换新 prompt 完全不增益”的泛化失败。
- **Non-goals:** 不做 tree verification，不增加 target verification 分支，不以哈希/形式化检查代替效果，不把 `+0.1/+0.3` 的小幅开发收益包装成问题已经解决，也不先做 SGLang 外围工程来掩盖接受长度失败。
- **Constraints:** 新 head 相对 537.4M DFlash/Domino draft 必须轻量；允许比 Domino head 稍大、稍慢，也允许效果不足时适当扩容，但最终端到端吞吐必须明显领先 Domino。原型阶段允许 eager 对 eager，最终必须在 SGLang/CUDA Graph 下公平比较。
- **Success condition:** 第一阶段在完全未参与梯度更新的 exact-runtime held-out prompts 上达到至少 `8.325` EAL（`1.15 × 7.23955`），且不是以大量 harmful overrides 换来的；之后在 SGLang 中实现端到端吞吐相对 Domino 至少约 `+15%`。越接近 held-out Top-16 oracle `10.25449` 越好。

## Anchor Check

**PASS.** `7.55` 仍只是扩展许可，最终接受长度门仍是 `8.325`，并新增真实 policy-dependent rollout 上相对 Domino `≥1.15×` 的共同门。SGLang 仍置于接受长度达标以后。

## Simplicity Check

**PASS.** Stages A–C 只适配现有 Domino GRU/correction head，并把同一 head 的调用延伸到位置 0；没有第二个 scorer、candidate attention、独立 safety gate或 suffix loss。该路线恰好隔离唯一尚未被既有失败实验检验的变量：真实 deployment anchors 上的 reachable-frontier supervision。

## Empirical Motivation

本项目已经做过的 static frontier 路线不能替代 GFPR：

- 1,986 train prompts、7,944 固定 blocks、全 537M backbone + 50.8M causal head 的 `topk_frontier_curriculum`，validation-select 最佳只从 `7.01579` 到 `7.06706`（`+0.0513`）。
- head-only static frontier run 在中途 validation 已降到 `6.99` 左右。
- 这些 runs 的 anchors 仍来自固定 target-clean offsets，且 released interface 的位置 0 没有共享 causal correction。

因此本轮不是重跑“frontier loss”，而是同时修复 policy state distribution 和 position-0 blind spot；static arm只作为严格控制组。

## One Unified Policy Contract for Stages A–C

Stages A–C 明确使用 **direct full-vocabulary adapted Domino policy**：

\[
d_i^\theta=\arg\max_{v\in V}s_i^\theta(v).
\]

- 位置 1–15 的 `s_i^θ` 就是现有 Domino base full-vocabulary logits 加 GRU correction logits。
- 位置 0 的 GRU 在每个 block 开始时清零，先消费 anchor 得到 `s_0^{GRU}`；复用同一个 correction MLP 得到 `b_0`，使用 `base_0 + α_0 b_0`，其中 `α_0=0` 初始化。
- 加载 released GRU/head 权重且 `α_0=0` 时，16 位 action 必须逐 token 精确复现 released Domino。训练后直接由适配后的 full-vocabulary scores argmax；推理不保留第二套 frozen head。
- target top-1 在 full vocabulary 中永远可表示，因此 Stages A–C 没有 `gold unavailable` mask，也没有 K16 candidate-renormalized loss。

新增参数只有 `α_0`；可训练主体是已有的约 50.8M Domino causal head，仍与 537.4M draft 同阶且不新增独立网络。位置 0 的实际代价是一整次额外 full-vocabulary correction-head application，不把它淡化成“一个标量”的代价；从 15 次变成 16 次，算术工作约增加 `1/15`，必须独立 profile。

## Top-16 Is an Oracle and Optional Contraction Diagnostic

Top-16/K17/K16 在 Stages A–C **不定义 policy action space**，只承担两个作用：确认用户所指出的候选 headroom，以及决定后续是否值得做 candidate-only deployment contraction。

令 `B_i` 为 DFlash Top-16，`a_i^D` 为 released Domino action：

- K17 oracle: `B_i ∪ {a_i^D}`；
- K16 oracle: 若 action 已在 `B_i` 则使用 `B_i`，否则使用 `{a_i^D} ∪ B_i[:15]`。

Gate A 同时报告 frozen-position-0、all-16 DFlash Top-16、K17和K16 oracle。由于 A–C 的最终 action space 是 full vocabulary，K16 不限制其可达性；如果 Stage D 最终选择 K16 candidate residual作为部署模型，则 **all-position K16 oracle 必须至少达到 `8.825`（主目标上方 `+0.5` headroom）**，否则该 contraction 不得成为最终模型。

## Exact Rollout and Target Semantics

对真实 verification cycle `m`，verified context为 `c_m`，末 token 为 anchor。policy 顺序产生 `d_{m,0:15}`。不引入独立的“target continuation”符号，而直接定义 draft-prefix-conditioned target greedy ID：

\[
g_{m,i}=\arg\max_v T(v\mid c_m,d_{m,<i}),\qquad
r_m=\min\{i:d_{m,i}\neq g_{m,i}\},
\]

全对时 `r_m=16`。对于 `i<r_m` 和首错 `i=r_m`，draft prefix 与 target greedy prefix相同，因而是 clean reachable states；首错后的 `g` 不进入 loss，也无需物化。

每个 block 的 Domino GRU 都重新清零，先消费 anchor；每次位置 `i` 作出选择后，再消费实际 selected `d_i`，供位置 `i+1` 使用。绝不把首错后的 gold suffix送入部署 GRU。

runtime 接受 `r_m` 个 draft tokens，再追加 target correction/bonus token，因此

\[
o_{m+1}=o_m+r_m+1.
\]

当 `r_m=16` 时，target 在完整 16-token draft prefix后产生 position-16 bonus token；它被追加到 context，并作为下一 block anchor。collector 与 evaluator都必须实现这一分支。

## Data Collector

phase3 canonical records 已保存 prompt和256-token target-greedy continuation。每个 prompt可由最长 nested record还原完整序列；一次 target causal pass提供所有 clean-prefix context features和 target logits。随后从 offset 0 开始，用 exact released/current Domino drafting code按 `r+1` 动态前进，直到剩余 continuation不足17 tokens。

每个 record 保存：

- sample/domain/split、prompt length、policy version、actual anchor offset；
- exact context IDs或可还原索引、anchor、gold/clean continuation 17 tokens；
- full 16 parallel hiddens、policy selected tokens、accepted length；
- 可选的 target frontier logit/margin诊断。

必须检查 stored `r` 与重新计算的 first mismatch一致、accepted tokens逐位等于对应 `g_i`、下一 offset严格等于 `r+1`。这些是语义正确性检查；不做无助于效果的哈希审查。

同一个 collector提供两种模式：

- `dynamic`: actual policy anchors，按 `r+1` 前进；
- `fixed-control`: 使用原固定 offsets，但调用完全相同的 draft/head/label代码。

因此 Gate B 的 static/dynamic 差异只有 anchor distribution。

## Current-Frontier Training Objective

在一个已缓存的 clean block上，用当前 `θ` 的 teacher-prefix head logits寻找当前首错 `q_θ`。这不会泄漏：在 `q_θ` 以前当前 action均等于 target gold，所以 teacher-prefix GRU state与真实 current-policy state完全相同；在首错之后不计算训练损失。随着首错被修复，同一 block下一次 forward 会自动暴露更后的 frontier。

令 `Δ_i=s_i^θ(g_i)-max_{v≠g_i}s_i^θ(v)`。主损失为

\[
L_{block}=
\lambda_{break}\mathbf 1[q_\theta<16]
[m_{break}-\Delta_{q_\theta}]_+
+\frac{\lambda_{keep}}{\max(q_\theta,1)}
\sum_{i<q_\theta}[m_{keep}-\Delta_i]_+
+\lambda_{wd}\lVert\theta-\theta_0\rVert_2^2.
\]

- `λ_break > λ_keep`，起点为 `1.0` 与 `0.1`；prefix预算按 block归一化，15个已对 token不会压过一个首错。
- 对 `q=0`，keep项为空；对 `q=16`，没有repair项，16个位置的保护总预算仍被cap。
- prompt内所有 cycles权重和归一为1，避免低接受率prompt因产生更多cycles而过度加权。
- 不训练 wrong-prefix suffix；不使用 candidate-truncated KL；可选 target-logit辅助权重最多0.1且只能在 reachable positions启用。

## Policy-Versioned Replay

1. `v0`: released Domino在训练 prompts上的actual rollout。
2. 训练并通过 Gate B 后，使用选定 `v1` checkpoint重新采集真实 anchors。
3. 扩展训练以prompt为单位混合50% v0和50% v1 records。
4. 一次 refresh为必需步骤，因为修复一个首错后才会暴露新的后续frontier；但不做无限 DAgger。

## Evaluation Contracts

同时保留两套互补但不能互相替代的 held-out 指标：

1. **Fixed exact-runtime comparison:** 在现有完全未训练的 `validation_select` exact-runtime语义上复现 released `7.23955`，目标 `≥8.325`，便于与历史所有结果公平比较。
2. **True rollout comparison:** 在同一 held-out prompts、同一 continuation budget上分别运行 released和current policy的动态 anchors；以每prompt的cycle EAL作paired比较。最终要求 current/released `≥1.15×`。

checkpoint只由 `validation_select` 选择；路线和超参冻结后，`validation_gate`只运行一次确认。训练loss、固定同集合容量和teacher-forced token accuracy均不能替代这两项EAL。

## Statistical and Harm Gates

每次关键评估按prompt配对，报告10,000次prompt-cluster bootstrap的95%区间，并报告：总 gained accepted tokens、总 lost accepted tokens、harmful prompt fraction、各domain delta。

- **Proof-of-signal:** raw fixed EAL `≥7.55`、delta `≥+0.30`，且paired 95% bootstrap lower bound `>0`。
- **Harm condition:** lost accepted tokens `≤0.5 × gained accepted tokens`，且 harmful prompts `≤20%`。Gate B、Gate C和最终 `8.325` 都必须满足。
- **Final method success:** fixed exact-runtime EAL `≥8.325`，true rollout ratio `≥1.15×`，paired lower bound `>0`，并满足harm condition。

如果某domain受损，必须单独报告；不以overall平均掩盖。domain-balanced训练权重只在预先定义的arm中使用，不做事后threshold sweep。

## Experiment Gates

### Gate A — read-only feasibility

- 复现 full-16 released policy和`7.23955`；
- 计算 all-16 DFlash Top-16、K17、K16 oracle以及 position-0单独增量；
- 验证 dynamic collector的`r+1`、`r=16` bonus、GRU reset和first mismatch；
- profile位置0额外full-vocabulary head调用。

### Gate B — 2K-prompt causal screen

在相同prompt、steps、optimizer下比较：

1. fixed-control anchors、只修位置1–15；
2. dynamic anchors、只修位置1–15；
3. dynamic anchors、all-16 GFPR。

只训练现有 Domino GRU/correction head和`α_0`。arm 3必须达到proof-of-signal及harm gate才扩展；`≤7.40`或alignment正确但持续降低EAL则终止当前head/objective假设。

### Gate C — 16K–32K and one refresh

仅在 Gate B 通过后扩展。完成v0训练、v1 recollection和50/50 replay。`≥8.325`才算方法成功；`7.55–8.324`仍是未解决主目标。

### Gate D — effect-driven capacity escalation

只有GFPR已有显著held-out增益但refresh后仍`<8.325`，才比较：

- 更宽/更深但仍轻量的candidate-conditioned lattice residual；
- draft最后一层或低秩LoRA。

若打开candidate residual，先冻结选定的Stage-C full-vocabulary scorer作为唯一base scorer，用其action与DFlash Top-16形成K17/K16，再加zero-init residual；推理只运行这一个base scorer加residual，latency包含两部分。若打开backbone/LoRA，cached hiddens立即失效，必须保留raw contexts并在线重算或按checkpoint重新采集，不能复用旧hidden。

### Gate E — SGLang system result

仅在EAL门通过后集成。用同硬件、同batch、同workload比较EAL、draft/head/verify latency与tokens/s；目标是相对Domino端到端约`+15%`或更高。

## Contribution Focus

- **Dominant contribution:** exact-greedy block-parallel drafts应在真实policy产生的verification anchors上，只对reachable prefix和当前first rejection训练，并覆盖纯parallel的第0位。
- **Conditional supporting contribution:** 只有matched direct-head结果证明表示容量不足时，candidate-conditioned lattice residual才成为方法组成。
- **Explicit non-contributions:** 不把DAgger、margin、GRU、Top-K、zero-init或Draft-OPD本身宣称为新颖；不声称当前greedy objective支持sampling。

## Reviewer Re-check

1. Stages A–C 是否已成为单一、无frozen-reference歧义的full-vocabulary policy？
2. target/GRU/bonus语义是否覆盖所有off-by-one和wrong-prefix风险？
3. keep normalization、paired uncertainty和harm gate是否足以保证首错修复主导且结果不是靠大量回退换来？
4. candidate contraction与LoRA是否已被正确隔离到有证据后的Stage D？
