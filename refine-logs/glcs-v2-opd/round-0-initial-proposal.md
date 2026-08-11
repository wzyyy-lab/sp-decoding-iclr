# Research Proposal: OPAL — On-Policy Advantage Lookahead over Domino

## Problem Anchor

- **Bottom-line problem:** “最重要的目标是完全解决接受长度不高的问题，一定要超过 Domino，越高越好”；主目标不是小修小补，而是让 Top-16 中已有的正确 token 真正转化成更长的可接受前缀。
- **Must-solve bottleneck:** 当前 released Domino 在 exact B16 runtime held-out 上的 prompt-balanced EAL 是 `7.23955`，DFlash Top-16 oracle 是 `10.25449`，但现有 GLCS 训练后仍只有 `7.15–7.22`。必须解决“同集合可记到 oracle、换新 prompt 完全不增益”的泛化失败。
- **Non-goals:** 不做 tree verification，不增加 target verification 分支，不以哈希/形式化检查代替效果，不把 `+0.1/+0.3` 的小幅开发收益包装成问题已经解决，也不先做 SGLang 外围工程来掩盖接受长度失败。
- **Constraints:** 新 head 相对 537.4M DFlash/Domino draft 必须轻量；允许比 Domino head 稍大、稍慢，也允许效果不足时适当扩容，但最终端到端吞吐必须明显领先 Domino。原型阶段允许 eager 对 eager，最终必须在 SGLang/CUDA Graph 下公平比较。
- **Success condition:** 第一阶段在完全未参与梯度更新的 exact-runtime held-out prompts 上达到至少 `8.325` EAL（`1.15 × 7.23955`），且不是以大量 harmful overrides 换来的；之后在 SGLang 中实现端到端吞吐相对 Domino 至少约 `+15%`。越接近 held-out Top-16 oracle `10.25449` 越好。

## Technical Gap

### 当前 pipeline 在哪里失败

现有 GLCS-v1 已经排除了 function-class capacity 不足：61.1M 参数模型在 1024 个同集合 blocks 上达到 `10.46289`，等于该集合 oracle，相比 Domino 的 `6.9375` 高 `+3.5254`。但 15,886-block exact-runtime 训练和 295,604-block / 32,039-prompt 大训练都把训练损失与 teacher-forced candidate accuracy 做好，却让 held-out EAL 从初始化 `7.22279` 降到 `7.15–7.19`。因此核心不是“再加一点宽度”，而是训练分布和接受长度决策不一致。

三个具体错配共同存在：

1. **随机/均匀 canonical anchor 不是部署错误状态。** 现有训练在 target-clean、预选 anchor 上做 hard-label imitation，没有集中采样 Domino/当前 policy 真正发生首错的位置。项目本地 Draft-OPD 论文的关键证据正是 error-position replay 显著优于随机 anchors。
2. **hard one-hot CE 没有提供 candidate advantage。** 每个 Top-16 位置只有一个 gold ID，模型不知道错误候选相对 target distribution 的风险大小。大量不可达 suffix 和容易的已正确 token 会淹没真正决定 EAL 的首个 rejected decision；训练 loss 下降不等价于减少 first-error regret。
3. **GLCS-v1 过早压缩 candidate identity。** 每位置 16 个候选先压成 4 个 mode，之后一个共享 correction code 再通过低秩 vocabulary basis 给所有候选打分。该结构能记忆，却没有显式回答“当前候选 k 与整块未来 lattice 是否兼容”，这正是泛化所需的 candidate-specific evidence。

### 为什么朴素修补不足

- 仅扩大 GLCS-v1：同集合容量已经到 oracle，不能修复错误监督。
- 继续 static hard CE：15.9k 和 295.6k 两个尺度都失败，更多同类型数据只会强化同一错配。
- 只加 override threshold：可以减少 harm，却不能学会找出更多正确候选，最多得到小幅保守收益。
- 只微调 Domino GRU/LoRA：项目已有多条静态 CE、reachable、frontier、head/LoRA 路线，held-out 增益大多为 `0–0.05`，没有接近所需的 `+1.09`。

## Method Thesis

- **One-sentence thesis:** 保留 released Domino 作为 exact-identity base policy，用 Draft-OPD 式真实错误位置 replay 和 target candidate-distribution distillation，训练一个 candidate-specific global-lookahead residual，使每次 override 预测的是相对 Domino 的 target advantage，而不是从随机 anchor 的 hard token 标签中记忆 rank。
- **Why this is the smallest adequate intervention:** 不重训 537M draft，不替换 target，不增加 verification；只把 GLCS 的共享 candidate code 改成 candidate-conditioned residual，并把静态 one-hot 训练改为真正匹配部署错误状态的 distillation。
- **Why timely:** Draft-OPD 已证明 on-policy error replay 和 accepted/rejected 非对称 KL 能提高 speculative draft；本方法把这一 primitive 首次约束到 Domino 的 Top-16 单链 correction head，并结合全局 future lattice，而不是再训练一个完整 draft model。

## Contribution Focus

- **Dominant contribution:** 面向 block-parallel Domino 的 on-policy candidate-advantage lookahead：在不增加 target verification 的前提下，把 target 在真实 draft-error states 上的 dense distribution 蒸馏到轻量 candidate-only residual。
- **Supporting contribution:** candidate-conditioned global-lattice scoring，使未来候选证据对当前每个 candidate 分别产生作用，同时保留 Domino 的 selected-token causal feedback。
- **Explicit non-contributions:** 不声称 KL、cross-attention、Top-K gather、GRU、on-policy distillation 或 zero-init residual 单独新颖。

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone:** released Qwen3-4B-Domino-b16 parallel backbone、target embedding/LM head、Domino GRU 和低秩 correction head均作为 exact base；第一阶段全部冻结。
- **New trainable component:** 一个约 12–30M 参数的 OPAL residual adapter。默认 `D=256`, 2 lattice blocks, 4 global modes/position, candidate-conditioned cross-attention；未过效果门再升至 `D=512` 或 3–4 层。
- **Intentionally excluded initially:** 第二个 draft transformer、full-vocabulary sequential GEMM、tree search、RL、额外 verifier、post-hoc 大型 safety model。

### System Overview

```text
accepted target prefix
        │
        ├── released Domino parallel backbone ──> H[1:L], base Top-16 lattice
        │                                             │
        └── selected draft tokens ──> frozen GRU state S_i
                                                      │
Top-16 candidate nodes (token, rank, logits, H_i) ──> compact global modes
                                                      │
each candidate query Q(i,k) = f(candidate node, H_i, S_i)
                 cross-attends to all block modes ──> residual advantage r(i,k)
                                                      │
score(i,k) = released-Domino score(i,k) + g_i · r(i,k)
                                                      │
                         argmax one token, feed it to frozen GRU, continue
```

### Core Mechanism

#### Candidate-preserving lookahead

沿用 GLCS-v1 的低成本 `L × M` global modes 作为 memory，但不再先选一个共享 mode 再生成统一 correction code。对每个候选 `(i,k)` 构造 query：

```text
q_ik = LN(Pe(e_ik) + Ph(H_i) + Ps(S_i) + position_i + rank_k + Pφ(φ_ik))
c_ik = CrossAttention(q_ik, global_modes[all positions])
r_ik = MLP([q_ik, c_ik, q_ik*c_ik])
```

因此未来 lattice 对 candidate A 和 candidate B 可以给出不同证据。输出只包含 `L×K` 标量 residual，不做新的 full-vocabulary GEMM。`r` 的最后投影和部署 gate 均 zero-init，epoch 0 精确恢复 candidate set 内的 released Domino；candidate set 使用 DFlash Top-16 与 Domino 当前 action 的固定 16-slot union。

#### On-policy error-position replay

不再只从均匀 canonical offsets 取训练 block。用 frozen Domino 以及后续当前 OPAL policy 在训练 prompts 上做真实 speculative rollout：

1. target verification 保留 target-quality continuation；
2. 每轮记录真实 draft anchor、候选 block、accepted length 和第一 rejected 位置；
3. 对同一个 draft-generated prefix replay target，一次性计算 target logits；
4. 只保存候选集上的 target log-probabilities、Domino logits、lattice features 和 acceptance mask。

先采一轮 Domino-policy errors，训练后再采一轮 current-policy errors；混合两轮数据，防止只会修旧 policy 的错误。

#### Acceptance-aware candidate distillation

令 `p_ik` 是 target 在真实 replay prefix 上、限制到 candidate union 后的概率，`q_ik` 是 OPAL 分布：

- accepted prefix：forward KL `KL(p || q)`，覆盖 target modes 并保护当前正确决策；
- first rejected 及其短 suffix：reverse KL `KL(q || p)`，直接压低 draft 自己的错误高概率 mode；
- rejected suffix 权重为 `γ^(k-r)`，默认 `γ=0.5`，首错最大；
- 每个 block 先归一化，再按 prompt 逆 block 数加权，匹配 prompt-balanced EAL；
- 增加一个仅作用于 first rejected position 的 greedy gold-vs-best margin，作为 `T=0` 决策辅助，不让 hard CE 重新主导训练。

损失保持单一 thesis：dense target advantage + acceptance-aware weighting。任何 keep/override 风险控制都由相同 target advantage 产生，不另加独立 gate 模型。

### Integration

训练阶段 target 只作为离线 teacher；推理阶段没有 target replay。运行时仍是一次 Domino/DFlash parallel forward、一次并行 Top-16、一个轻量 global memory，以及 15 次只在 16 candidates 上的 scalar scoring。selected token 继续进入 frozen Domino GRU，因而保持正确的 causal feedback。

### Training Plan

1. **Stage A — alignment/diagnostic:** 在现有 1,987 train prompts 上物化真实 Domino error anchors 和 target candidate logits；验证 target candidate argmax 在可覆盖前缀上精确复现 Top-16 oracle。
2. **Stage B — compact OPAL:** `D=256`, 2 layers，训练 1–2 epochs；每 250 steps 在 exact-runtime `validation_select` 上测 raw EAL、improved/harmed、first-correction accuracy。必须从 exact Domino identity 起步。
3. **Stage C — on-policy refresh:** 用 Stage-B checkpoint 在 16k–32k prompts 上重新 rollout/replay，再混合 Domino-policy 与 current-policy errors 训练。
4. **Stage D — effect-driven scaling:** 只有 Stage C 明确超过 Domino 才扩大到已有 32,039 prompts / 295,604 blocks或补齐 100k prompts；若 objective 已有效但欠拟合，再升 `D=512`。若 adapter 已饱和但仍低于 `8.325`，再联合训练 Domino low-rank head或 backbone LoRA，推理参数量增加为零。

### Failure Modes and Diagnostics

- **Target replay alignment错误:** candidate target argmax 无法在 gold-in-K 位置复现 gold；立即停下修数据，不训练。
- **loss下降但首错不改善:** 对 accepted/first-rejected/later-rejected 分开报告 KL、greedy accuracy 和 EAL contribution；若只改善 suffix，缩短 rejected suffix而不是堆参数。
- **override harm过高:** 检查 predicted target advantage 的 calibration；保持 exact zero residual并增强 accepted-prefix forward KL，不用事后 threshold sweep代替学习。
- **compact adapter欠拟合:** 同集合和 train-policy held-out 都低；允许加宽/加层。
- **train-policy有效、new-policy失效:** 执行一次 current-policy refresh，而不是继续离线重复 epochs。
- **canonical gain不转化到 rollout:** 最终必须在真实 end-to-end prompts 复测；canonical 只用于快速模型选择。

## Novelty and Elegance Argument

Domino 只用 teacher-forced GRU + low-rank full-vocab residual，并在静态 SFT 数据上训练；Draft-OPD 对完整 draft model做 on-policy distillation，但不利用一个现成的 block Top-K lattice，也不设计 candidate-only lookahead head。OPAL 的焦点是两者之间的缺口：用真实 Domino rejection states 监督一个只在候选集内工作的全局 lookahead residual。方法没有平行模块堆叠，全部新增机制都服务于同一件事——预测“哪个 Top-16 candidate 相对 released Domino 真正更接近 target distribution”。

## Claim-Driven Validation Sketch

### Claim 1: on-policy target-advantage distillation解决 static hard-label 泛化失败

- **Minimal experiment:** 同一 `D=256` candidate-specific adapter、同一训练 prompts，比较 static hard CE 与 OPAL error-replay mixed-KL。
- **Baselines:** exact Domino identity、当前 GLCS-v1 loss、OPAL without policy refresh。
- **Metric:** exact-runtime held-out prompt-balanced EAL、first-error repair、harm、oracle-gap recovery。
- **Expected decisive evidence:** OPAL 明显超过 `7.23955`，且 improvement 不是仅来自 suffix candidate accuracy；最终目标 `≥8.325`。

### Claim 2: candidate-specific future evidence优于共享 correction code

- **Minimal experiment:** 在完全相同 replay data/loss/budget 下比较 GLCS-v1 shared-code 与 OPAL candidate-conditioned scoring。
- **Metric:** held-out EAL、首错 target-rank accuracy、参数量和 CUDA-graph latency。
- **Expected decisive evidence:** candidate-specific 结构同时提高首错 repair 和 EAL；若不优，删除结构 claim并保留更简单 scorer。

### Claim 3: 收益能转化成系统吞吐

- **Minimal experiment:** 只有 EAL 达标后，在同一 A40/SGLang 配置下比较 Domino 与 OPAL 的真实 rollout、CUDA Graph head latency和端到端 tok/s。
- **Metric:** acceptance length、draft/verify/head latency、tok/s。
- **Expected decisive evidence:** 相对 Domino 端到端吞吐约 `+15%` 或更高。

## Experiment Handoff Inputs

- **Must-prove claims:** error-position dense distillation提升未见 prompt 的可接受前缀；candidate-specific lookahead贡献独立收益。
- **Must-run ablations:** hard CE vs mixed KL；random/static anchors vs error anchors；shared-code vs candidate-conditioned；Domino-policy-only vs one次 policy refresh。
- **Critical data:** target replay logits、真实 accepted/rejected masks、exact runtime phase3 train/select。
- **Highest-risk assumptions:** 32k–100k error anchors是否足以学到 target advantage；candidate-only target distribution是否保持足够的排序信息；A40 graph latency是否保持在可转化吞吐的范围。

## Compute & Timeline Estimate

- **Data/replay:** 4×A40/A800 并行，首轮 2k sanity 预计 <1 GPU-hour；16k–32k error replay预计 2–6 GPU-hours，取决于 target replay长度。
- **Training:** compact adapter每个配置约 0.2–1 GPU-hour；最多并行 3 个核心 ablation，不做大网格。
- **Scaling:** 仅正结果后补齐；100k replay/大 adapter预计再增加 8–20 GPU-hours。
- **Timeline:** alignment和首轮结果 1天内；一次 on-policy refresh及扩容 1–3天；通过 EAL gate 后再做 SGLang集成。
