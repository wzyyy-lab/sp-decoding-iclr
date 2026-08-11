# Round 1 Refinement: GFPR — Greedy-Frontier Policy Replay for Domino

## Problem Anchor

- **Bottom-line problem:** “最重要的目标是完全解决接受长度不高的问题，一定要超过 Domino，越高越好”；主目标不是小修小补，而是让 Top-16 中已有的正确 token 真正转化成更长的可接受前缀。
- **Must-solve bottleneck:** 当前 released Domino 在 exact B16 runtime held-out 上的 prompt-balanced EAL 是 `7.23955`，DFlash Top-16 oracle 是 `10.25449`，但现有 GLCS 训练后仍只有 `7.15–7.22`。必须解决“同集合可记到 oracle、换新 prompt 完全不增益”的泛化失败。
- **Non-goals:** 不做 tree verification，不增加 target verification 分支，不以哈希/形式化检查代替效果，不把 `+0.1/+0.3` 的小幅开发收益包装成问题已经解决，也不先做 SGLang 外围工程来掩盖接受长度失败。
- **Constraints:** 新 head 相对 537.4M DFlash/Domino draft 必须轻量；允许比 Domino head 稍大、稍慢，也允许效果不足时适当扩容，但最终端到端吞吐必须明显领先 Domino。原型阶段允许 eager 对 eager，最终必须在 SGLang/CUDA Graph 下公平比较。
- **Success condition:** 第一阶段在完全未参与梯度更新的 exact-runtime held-out prompts 上达到至少 `8.325` EAL（`1.15 × 7.23955`），且不是以大量 harmful overrides 换来的；之后在 SGLang 中实现端到端吞吐相对 Domino 至少约 `+15%`。越接近 held-out Top-16 oracle `10.25449` 越好。

## Anchor Check

**PASS.** 本轮没有把目标降为“证明有一点信号”。`7.55` 只允许进入扩展实验，`8.325` 才是方法成功；SGLang 集成继续置于接受长度门后。方法从 candidate-KL/lookahead 收缩到 policy-induced greedy frontier，直接针对 held-out 不泛化和位置 0 不可修两个已证实瓶颈。

## Simplicity Check

**PASS, with an explicit escalation ladder.** 第一实现不增加 12–30M cross-attention adapter，不训练首错后的 suffix，不增加独立 gate。先给 released Domino 增加一个共享的 position-0 调用接口并在同一 causal head 上做 frontier adaptation；只有该监督在未见 prompt 上给出真实增益但容量触顶，才加入 candidate-conditioned lattice readout。

## Revised Method Thesis

**One-sentence thesis:** 对 exact-greedy speculative decoding，只在真实 policy rollout 到达的 block anchors 上保护已接受前缀并修复首个 rejection，且覆盖 block 的全部 16 个位置；这比在固定 target-clean anchors 上拟合所有 token 更贴近 EAL 的因果边界。

方法名改为 **GFPR (Greedy-Frontier Policy Replay)**。OPAL 的 candidate-conditioned global adapter 不再是前置假设，而是有正信号后的容量升级。

## Exact Deployment and Record Semantics

### Verification cycle

给定当前 verified context `c_m`，其最后一个 token 是 anchor `x_m`。draft 一次并行前向产生 16 个 hidden 和 deployed policy 顺序选择的 block

\[
d_m=(d_{m,0},\ldots,d_{m,15}).
\]

target greedy continuation 记为 `y_m`。accepted draft length

\[
r_m=\min\{i:d_{m,i}\ne y_{m,i}\},
\]

若 16 位全对则 `r_m=16`。真实 runtime 接受 `r_m` 个 draft token，再写入一个 target bonus token，因此下一轮 anchor offset 必须满足

\[
o_{m+1}=o_m+r_m+1.
\]

这正是现有 Domino generation 中 `start += acceptance_length + 1` 的语义。collector 必须按这个递推采样，不能再使用 `0,16,...` 或均匀固定 offsets。

### Target alignment

target 在位置 `i` 的决策分布是

\[
T(c_m,d_{m,<i}).
\]

对所有 `i<r_m`，以及首错 `i=r_m`，`d_{m,<i}=y_{m,<i}`，所以它们都是 clean、reachable prefixes；只有首错之后是 wrong-prefix suffix。本方法初始阶段完全不训练 suffix，因此一条 target greedy continuation 的 causal pass（或 verification 已产生的 logits）即可无歧义地提供所需 target top-1 和 candidate logits。

每个 cycle record 至少保存：prompt/sample ID、domain/split、exact context/anchor identifier、anchor offset、policy version、16 个 parallel hiddens、policy-selected `d_m`、`r_m`、每位置 candidate IDs 与 released score、target top-1 ID/logit、candidate target logits、以及 target log-normalizer（仅在启用低权重 auxiliary 时需要）。必须满足：

- `d[m,i] == target_top1[m,i]` for every `i < r_m`；
- `d[m,r_m] != target_top1[m,r_m]` when `r_m < 16`；
- stored `r_m` 等于从这两个 token 序列重新计算的 first mismatch；
- GRU state 始终消费 anchor 和 deployed/logged selected tokens，不消费首错后的 gold token。

这些是防止 off-by-one 或 teacher-forcing 泄漏的语义检查，不做与效果无关的哈希仪式。

## Candidate and Score Contract

对位置 `i`，令 `B_i` 为 DFlash base Top-16，`a_i^D` 为 released Domino 的 exact deployed action。

- **Prototype/ceiling path:** 使用至多 17 个候选 `C_i^{17}=B_i \cup \{a_i^D\}`，避免候选压缩混淆数据假设。
- **Final K=16 path:** 若 `a_i^D∈B_i`，则 `C_i=B_i`；否则 `C_i={a_i^D}∪B_i[:15]`。去重后固定顺序为 Domino action 优先、其余按 DFlash rank。
- 每个候选的 base score必须是 released policy 对该 token 的 exact score：位置 0 是 DFlash full-vocabulary logit；位置 1–15 是 DFlash logit加 released Domino correction。不能拿原始 DFlash rank 当 Domino score。
- residual 为零时，candidate argmax 必须复现 released action。若 17-candidate prototype 与 K=16 contraction 的 oracle 差异显著，先报告并定位被删的 rank-16 token，不能静默更换 ceiling。

## All-16 Minimal Domino Adaptation

released Domino 当前只修正位置 1–15，位置 0 直接使用 base top-1。GFPR 强制覆盖全部 16 位：

1. 在位置 0，GRU 只消费 anchor，得到 `s_0`；复用同一个 Domino correction MLP 计算 `b_0=f([h_0,s_0])`。
2. 位置 0 score 为 `base_0 + g_0 b_0`，其中标量 `g_0` zero-init，故 step 0 完全保留 DFlash top-1。
3. 选择位置 0 后，将其送入 frozen/current GRU state；位置 1–15 沿用 released Domino 的 selected-token causal rollout。
4. Stage 1 直接微调 Domino GRU/correction head（以及 `g_0`），不引入另一个网络。训练前和每次评估都以 step-0 exact reproduction 为基线。

这一接口仅多一次与 Domino 相同的轻量 correction step，结构和参数规模仍与 Domino head 同阶。若 full-vocabulary output 成为瓶颈，部署时把末层只 gather 到 `C_i`，但原型先保证 score 语义公平。

## Greedy-Frontier Objective

对一个 policy record，accepted positions `i<r` 只需避免被破坏；若 `r<16` 且 target top-1 在 `C_r` 内，只训练首错位置。首错之后全部 mask；若 gold 不在 `C_r`，不把“最不错误候选”当 teacher，而只施加 identity regularization。

令 `s_i(g)` 为 gold candidate score，`s_i^-` 为最高非-gold score，`Δ_i=s_i(g)-s_i^-`：

\[
L=\lambda_{keep}\sum_{i<r}[m_{keep}-\Delta_i]_+
+\lambda_{break}\mathbf 1[y_r\in C_r][m_{break}-\Delta_r]_+
+\lambda_{id}\lVert s-s^{released}\rVert^2_{inactive/unavailable}.
\]

其中 `λ_break > λ_keep`；默认起点 `λ_break=1`, `λ_keep=0.1`, `λ_id=0.01`，margin 从 released gold/competitor margin 的训练分位数校准，而非大网格搜索。每个 prompt 总权重归一为 1，防止短 acceptance 造成更多 cycles 的 prompt 获得更大权重。

可选的 target candidate-logit margin/温度蒸馏只允许作用在 accepted positions 和可修首错，权重不超过主 frontier loss 的 0.1；它不是主要 claim，也不在 target top-1 缺失时启用。T=0 路线不声称支持 sampling。

## Policy Replay and Refresh

- **Policy v0:** released Domino actual rollout，严格按 `r+1` 前进；训练 prompt 与 held-out prompt 完全分离。
- **Policy v1:** 仅当 v0 screen 有正信号，使用选定 checkpoint 重新 rollout 同一训练池，采集新 frontier。
- v1 训练混合 `50% v0 + 50% v1` prompt-balanced records；保留 version ID，分别报告旧/new-policy frontier repair。
- 不做无限 DAgger。一次 refresh 后如果仍小于 `7.8`，先比较模型容量路线，而不是继续堆同类数据。

## Staged Experiments and Hard Gates

### Gate A — exact feasibility before training

在 untouched exact-runtime split 上计算：released identity、现有 frozen-position-0 oracle、全 16 位 DFlash Top-16 oracle、Domino-union K17/K16 oracle。必须满足：

- released identity EAL 复现 `7.23955`；
- stored first mismatch 100% 对齐重新计算值；
- all-16 deployed-union oracle `>8.325`，否则当前 candidate restriction不可能达到主目标，立即改候选/允许更强 draft，而不是训练。

### Gate B — 2K-prompt causal screen

同一 prompts、steps、optimizer 下比较：

1. fixed/static anchor + frontier loss；
2. actual rollout anchor + frontier loss；
3. actual rollout + all-16 correction（GFPR）。

主模型只训练现有 Domino GRU/correction head和 `g_0`。untouched heldout `≥7.55` 且相对 released `≥+0.30` 才视为 proof-of-signal；`≤7.40` 或 alignment 正确但训练持续降低 EAL，则停止该 head/objective 路线并诊断 representation，而不包装成成功。

### Gate C — 16K–32K prompts and one refresh

只有 Gate B 通过才扩大。先 v0 全量训练，再采 v1 并混合训练。最终方法门仍是 `≥8.325`。如果 refresh 后 `<7.8`，在完全相同数据和 frontier loss 下比较：

- direct Domino-head adaptation；
- position-0 + lightweight candidate-conditioned lattice residual；
- draft final-backbone-layer或低秩 LoRA adaptation。

### Gate D — effect-driven capacity escalation

只有 direct head 已有可靠未见 prompt 增益但低于 `8.325`，才恢复 OPAL 中的 candidate-conditioned lattice readout。先 `D=256`, 1–2 layers，预计 12–30M 参数；只有训练/heldout 都欠拟合才升 `D=512`。若 representation 需要改变，优先 LoRA/最后一层而不是无界增大额外 head。

### Gate E — system validation

仅在 heldout `≥8.325` 后进行 SGLang/CUDA Graph 集成。报告同硬件、同 batch/workload 的 EAL、draft/head/verify latency、tokens/s；最终目标是相对 Domino约 `+15%` 吞吐，不能只以 head microbenchmark 代替。

## Decisive Metrics

- prompt-balanced exact greedy accepted draft tokens（主指标）；
- EAL gap recovery `(model-Domino)/(oracle-Domino)`；
- first-token acceptance、first-frontier repair、harmful prefix regressions；
- gold-in-candidate frontier coverage及按 domain/context length 分解；
- 参数量、eager/graph head latency，最终端到端 tokens/s。

开发 checkpoint 只由 `validation_select` EAL 选择；`validation_gate` 在路线冻结后一次性确认。训练 loss 或 teacher-forced accuracy不能代替 held-out EAL。

## Contribution Focus

- **Dominant contribution:** 对 block-parallel greedy speculative decoding，训练分布应由真实 policy cycles 和首个可达 rejection 定义，并覆盖纯 parallel 的位置 0。
- **Conditional supporting contribution:** 若 matched-data 实验证明需要，candidate-conditioned lattice readout进一步恢复 Top-K oracle gap。
- **Explicit non-contributions:** 不把 DAgger、margin loss、GRU、Top-K gather、zero-init gate或 Draft-OPD 单独宣称为新颖；不把 wrong-prefix suffix distillation用于 greedy claim。

## Why This Revision Addresses the Review

1. 删除 candidate-renormalized mixed KL 和 later suffix，目标与 T=0 EAL 一致。
2. 明确定义 target prefix、cycle record、`r+1` anchor advance和 GRU消费的 token。
3. 明确定义 K17/K16 union与 exact released score，zero residual恢复 identity。
4. 强制修正位置 0，同时只增加一个共享 head 调用和 zero-init scalar gate。
5. 把新 global adapter从主假设降为 signal-after-plateau escalation，先用现有 Domino head隔离数据/目标因果效应。
6. 把 `7.55` 定义为继续门而不是成功，唯一方法成功线仍为 `8.325`。

## Reviewer Re-check Questions

1. 数据、prefix和位置 0 语义是否已足以避免另一次 static/teacher-forced伪实验？
2. frontier objective 是否严格对应 deterministic greedy accepted-prefix utility？
3. 先复用 Domino head、再按效果扩容的路线是否足够简洁且能检验核心 thesis？
4. Gate A/B/C 是否能在大量训练前快速证伪，同时保持用户要求的 `≥8.325` 主目标？
