# Round 1 Review

<details open>
<summary>GPT-5.6-Sol xhigh raw review</summary>

# APEX-16 Phase-2 Research-Refine Review

## 总评

APEX-16 完整保留了 full16、global non-causal、one-call、one-chain 的问题锚点，且 all-prefix 目标确实针对 PGCF-v1 已观测到的后位梯度饥饿。方向值得继续做一个严格 falsifier。

但当前版本有五个实质问题：

1. 所谓 axial candidate exchange 与现有 `GlobalDirectCandidateSelector` 的 `AxialGlobalDirectBlock` 基本相同，不能作为新的 dominant contribution。
2. all-prefix log-risk 的代数推导正确，但它本质是可分离的 prefix-weighted CE；它没有直接保证两次错误被联合修复。
3. hard-position evidence dropout 使用 gold 判断错误位置，向网络泄漏了“此处必错”的 oracle bit，并造成明显训练—推理错配。
4. G2 复用历史上已查看的 `validation_gate` prompt IDs；重新采标签不能恢复统计新鲜性。
5. 现有负证据与训练规模不支持性能承诺：旧 axial GCLS 用 99,356 prompts 仍仅提升约 `+0.22~+0.285 EAL`，而 APEX 计划用约 25K prompts 跨越约 `+2.257 EAL`。

方向可修，但当前不能作为 paper-ready proposal。

## Calibration 与评分

**CALIBRATION: none**

未发现由技能或项目预先指定的三份 known-good 与三份 known-bad proposal anchors，因此不能自行挑选历史提案冒充 curated anchors。

| 维度 | 权重 | 得分 |
|---|---:|---:|
| 1. Problem Fidelity | 15% | 9/10 |
| 2. Method Specificity | 25% | 6/10 |
| 3. Contribution Quality | 25% | 4/10 |
| 4. Frontier Leverage | 15% | 6/10 |
| 5. Feasibility | 10% | 4/10 |
| 6. Validation Focus | 5% | 4/10 |
| 7. Venue Readiness | 5% | 3/10 |

**WEIGHTED COMPOSITE: 5.50/10**

**GAP：** 由于没有 curated anchors，本段只能锚定 READY=`>=9` 的明确门槛。APEX-16 距离 READY 的主要差距不在描述完整度，而在贡献归属与证据链：架构几乎复用了既有 GCLS，新增训练目标尚未证明能产生联合两次修复，augmentation 带有 oracle 选位泄漏，形式 held-out 也不新鲜。即使实现无误，目前仍更像“已有 axial GCLS 上的一个有动机但未隔离的 loss/augmentation package”，而不是一个已具备顶会机制新意与可信扩展路径的单一贡献。

## 分维度审查

### 1. Problem Fidelity — 9/10

问题锚点保存得很好。方法明确满足：

- 完整16位同时输入；
- 所有位置全局无 causal mask 可见；
- `[B,16,16]` 一次并行输出；
- 每位置一次 argmax，唯一 `[B,16]` 链；
- 无 selected-token feedback、GRU rollout、serial target、iteration 或 multipath；
- target 仅用于离线监督与普通 verifier。

唯一保留意见是实验 freshness，不是架构漂移。

### 2. Method Specificity — 6/10

**具体弱点：**

- `Z` 未定义。若它是每 block 的 `h(h+1)/2`，短/长 clean prefix 的样本权重会发生特定变化；若是 batch 级常数，长 prefix block 会支配训练。
- “teacher-geometry mismatch” 没有可执行定义。
- `softmax(t_i/T)` 没说明是 full vocabulary 还是仅对当前16个 candidate logits 重新归一化。
- 未明确 target logits 是否在与普通 verifier 完全相同的 target checkpoint、token prefix 和 candidate IDs 上采集。
- G1 缺 batch size、LR、scheduler、seed protocol、prompt-balanced weighting；“最多2 epochs”相对既有小数据 GCLS 的训练预算可能明显不足。
- “完整候选分布摘要”表述过强：`z_i=sum_k pi_ik u_ik` 是一个向量摘要，不是保留完整候选分布的可逆表示。

**具体修复：**

明确写成逐 block 损失：

`h_b=min{i:y_bi notin C_bi or not g_bi}`，若集合为空则 `h_b=16`；

`p_bar_bik=exp(t_b(c_bik)/T)/sum_l exp(t_b(c_bil)/T)`；

`Z_b=sum_{i<h_b}(h_b-i)`；

`L_b=(1/Z_b) sum_{i<h_b}(h_b-i) CE(0.9 delta_rbi + 0.1 p_bar_bi, q_bi)`。

同时定义 `g_bi`、prompt/block weighting、FP32 normalization、duplicate/missing candidate 行为以及所有 optimizer 配置。

**Priority: CRITICAL**

### 3. Contribution Quality — 4/10

**具体弱点：**

现有 `GlobalDirectCandidateSelector` 已经实现了几乎相同的机制：

1. 每位置 candidate-local attention；
2. detached base conditional probability 加 learned pool score；
3. soft position summary；
4. 每个 candidate query 全部 position summaries；
5. relative-position bias；
6. residual scores 加 base log-prob；
7. zero-output identity initialization。

因此 APEX 的 axial architecture 不是“相对旧 GCLS 的新 dominant contribution”。主要差异只是 D256/L2、all-prefix loss 和 evidence dropout。当前提案同时改容量、loss、distillation 与 augmentation，也无法把收益归因于单一机制。

**具体修复：**

把论文重构为“在现有 axial GCLS 上的 acceptance-prefix distillation objective”，明确 architecture 是 reused baseline。删去 axial exchange 的架构新颖性声明，并先删除 evidence dropout。最小三臂应为：

- axial-global + 新 all-prefix objective；
- matched axial-local + 新 objective；
- axial-global + 既有 Candidate-D-PACE。

三臂分别识别 global information 与新 objective 的增量价值。

**Priority: CRITICAL**

### 4. Frontier Leverage — 6/10

**具体弱点：**

Offline target-distribution distillation 与 single-pass inference 是相容的，本身没有在线 target drift；但当前 denoising view 不是干净的 self-supervised corruption。mask 位置由 gold 指定，网络可从被置零的通道识别“这是 base rank0 错误且属于前两次错误之一”。这虽不泄漏正确 token ID，却泄漏了一个强 oracle error-location bit。

此外，最终 base logits `b_ik`、candidate IDs/embeddings 和 rank 仍保留，所谓“remote evidence 成为必要能力”并不成立。局部语义和输出 base margin 仍可解决 masked task。

**具体修复：**

优先从主方法删除 hard-position evidence dropout。若一定保留，mask 位置必须仅由部署时可用且 label-independent 的规则选择，例如固定 Bernoulli 或 base entropy/margin strata，并预注册 corruption 分布；不得使用 gold correctness、first/second error rank 选择输入 mask。

**Priority: CRITICAL**

### 5. Feasibility — 4/10

**具体弱点：**

参数计算是可信的：现有 selector 配置 `d=256,H=8,L=2,max_positions=16,K=16` 的确为 **4,539,888** 个 trainable parameters，低于 10.75M。

但性能扩展假设极弱：

- 历史 axial GCLS 在 99,356 prompts / 793,989 blocks 上，三 seed mean raw gain 约 `+0.2215`，最好约 `+0.285`；
- APEX 从 base `6.0685` 到目标 `8.3255` 需要约 `+2.257`；
- PGCF-v1 在 1,987 prompts 上几乎没有 transfer；
- APEX 的 G1 实际 fit prompts 仅约1,590，却要求 G2 达到 `7.55`，即约 `+1.48`；
- G3 约25K prompts 反而小于已有 99K axial 负证据规模；
- D128/L2 历史上未稳定优于 D64/L1，PGCF-v1 也证明容量不是首要阻塞，因此从 D256/L2 起步不符合“smallest adequate”。

**具体修复：**

复用 evidence-supported 的 D64/H4/L1 axial head作为主 falsifier，只改变 objective。只有 internal fit 明确欠拟合时才升级容量。最终扩展预算至少应与历史 99,356-prompt axial baseline 匹配；25K/200K blocks 只能是中间 screen，不能被描述为可信的最终扩展规模。

G0 必须实测完整 APEX head，包括 Top-16、gather、预投影表和 active memory 的 A40 BF16 p50/p90，而不能从 PGCF-v1 的 `1.8207 ms` 外推。

**Priority: CRITICAL**

### 6. Validation Focus — 4/10

**具体弱点：**

- 历史文档明确记载 `validation_gate` 已被查看，不能再称为 fresh/sealed heldout。换成 R047 新标签并不会让既有 prompt IDs 重新独立。
- G2 没有直接 gate “前两次错误被同一输出联合修复”。
- 分别报告 first/second repair recall 不等价于两个错误同时修复且此前无 harm。
- `global-no-dropout` 不能隔离 all-prefix objective 相对既有 GCLS 的作用。
- G2 一旦用于是否 scale-up，就已经是 development set；G3 仍需要独立 formal heldout。
- G2 的 `EAL>=7.55` 可以作为 advancement gate，但不能支持通往 `8.3255` 的 claim。

**具体修复：**

建立全新 prompt manifest，排除所有历史 train/select/gate/formal IDs 及近重复，并在采标签前冻结 hash。至少区分：

- internal select；
- fresh mechanism gate；
- untouched final fixed/dynamic/system set。

加入联合多纠错指标：

`P(first two reachable base errors both repaired AND no earlier correct token harmed)`，

并以 global-new 对 global-old、global-new 对 matched-local 的 paired prompt-bootstrap CI 作为两项硬门。

**Priority: CRITICAL**

### 7. Venue Readiness — 3/10

**具体弱点：**

当前 paper story 把已有 GCLS architecture 重新命名为 APEX，并把一个 weighted CE、soft distillation 与 label-conditioned dropout并列包装。尚无证据表明它能填补约 `2.26 EAL` 的性能差，更无 dynamic EAL 或 same-stack SGLang TPS 证据。

**具体修复：**

先把贡献缩成一个可证伪命题：“prefix-log transformation 是否在同一 axial selector 上把 second-correction credit 转化为 held-out joint repair”。只有这一 matched claim 成立后，才扩展到最终性能与系统论文叙事。

**Priority: IMPORTANT**

## 对 perfect-two-correction oracle 与训练目标的专项判断

新增证据非常重要，但只能支持“必要性和梯度诊断”，不能支持 selectability：

- 1,175 blocks 中有 771 个在 pure-base-Top16 reachable prefix 内需要至少两次 non-top1 correction；
- `k=1` oracle EAL=`7.49854`，低于目标 `8.32549`；
- `k=2` oracle EAL=`8.42383`，刚超过目标；
- v1 expected-prefix-product loss 在第二次纠错处的平均 credit 为 `0.001067`，第一次为 `0.011278`，平均 ratio=`0.08657`、median=`0.05075`；
- normalized all-prefix 权重在第一次/第二次纠错处分别为 `0.14710/0.09775`，ratio 约 `0.665`。

这有力证明 v1 对第二次纠错存在 gradient starvation，也证明 log transform 能显著恢复第二纠错位置的直接梯度。

但它不证明 all-prefix CE 是充分目标，原因有四：

1. perfect-k oracle 使用 gold error identity 和 gold candidate，模型没有；
2. `k=2` 只比最终门槛高约 `0.09835 EAL`，几乎没有容错余量；
3. all-prefix 损失可分解为独立加权 CE，没有显式联合修复项；
4. 该 oracle 是固定 block 诊断，不能推出 dynamic rollout。

因此必须把“joint two-correction without earlier harm”设成决定性证据，而不是只看边际 second-repair recall。

## Simplification Opportunities

1. 直接复用现有 `GlobalDirectCandidateSelector` axial implementation，不把它包装成新 architecture。
2. 删除 hard-position evidence dropout，将 all-prefix objective 作为唯一新机制。
3. 以 D64/H4/L1 起步；只有 label-independent internal underfit 才升级到 D256/L2。

## Modernization Opportunities

1. 保留 offline target candidate-distribution distillation，但明确它是训练标签，并严格限定在当前 K=16 candidate support 上。
2. 若需要 denoising regularization，只使用 label-independent、部署特征可定义的 corruption；不得用 gold 选 mask 位置。
3. 不需要再增加 RL、iterative denoising 或其他 trendy 组件。

## Drift Warning

**NONE（架构层面）**。APEX-16 没有引入 causal、serial、iterative 或 multipath 路径。

但 `validation_gate` 的历史污染是证据协议硬阻塞，不能被“新采标签”掩盖。

## 不变量合规表

| 不变量 | 状态 | 判断 |
|---|---|---|
| 完整16位同时输入 | PASS | `[B,16,*]` 一次进入 head |
| 全局 non-causal visibility | PASS | candidate query 全部16个 summaries，无 triangular mask |
| 一次输出全部16位 | PASS | `[B,16,16]` 同时产生 |
| exactly one sequence | PASS | 单次逐位置 argmax 得到唯一 `[B,16]` |
| Top-16 仅候选轴 | PASS | 未构造路径轴 |
| 无 selected-token feedback | PASS | 无 token embedding 回馈到后位 |
| 无 GRU/autoregression/serial target | PASS | 推理图无顺序 target 依赖 |
| 无 iteration/beam/tree/trie/forest/multipath | PASS | 均明确排除 |
| 无额外在线 target inference | PASS | teacher logits 仅离线使用 |
| 仅使用在线 DFlash features | PASS，待代码审计 | 架构描述合规 |
| 新增参数接近或低于10.75M | PASS | 4,539,888 trainable params |
| 延迟路径兼容 TPS 目标 | PARTIAL | 参数可行，但 APEX 完整 latency 未测 |
| train/select/heldout 证据独立 | FAIL | G2 prompt IDs 历史已被查看 |
| same-job Domino fixed/dynamic/SGLang comparator | PLANNED | 尚无结果 |

## Blocking Issues

1. **B1 — 架构新颖性不成立：** axial exchange 已存在于 GCLS。
2. **B2 — Oracle-conditioned corruption：** evidence dropout 泄漏 error-location bit。
3. **B3 — G2 不新鲜：** 历史 `validation_gate` 不能作为 fresh heldout。
4. **B4 — 多纠错 claim 未被 objective 或 gate 直接支持：** all-prefix CE 仍是可分离损失。
5. **B5 — 经验扩展路径不可信：** 25K prompts 远小于已有 99K axial 负证据规模，却要求近十倍以上 EAL gain。
6. **B6 — 缺少相对现有 GCLS 的 matched objective control。**
7. **B7 — 缺独立 final fixed/dynamic/SGLang heldout。**

## Exact Revision Requests

1. 将 dominant contribution 改为“existing axial GCLS 上的 acceptance-prefix distillation objective”，删除 axial architecture 新颖性声明。
2. 主模型改为 D64/H4/L1；D256/L2 仅作为 internal underfit 后的预注册容量分支。
3. 删除 hard-position evidence dropout；若保留，必须改成 label-independent mask。
4. 完整定义 `h_b`、teacher geometry、candidate-conditional soft target、`Z_b`、prompt weighting 与 optimizer 配置。
5. 将三臂冻结为 `global-new-loss / local-new-loss / global-Candidate-D-PACE`，同初始化、同数据、同预算。
6. 用全新 prompt IDs 构建 mechanism gate，并另留 untouched final set；禁止再把旧 `validation_gate` 称为 fresh。
7. 把 joint-two-repair-without-earlier-harm 设为硬门，并同时约束 prefix harm 与 suffix-only edit。
8. G3 至少匹配已有 99,356-prompt axial baseline 的训练多样性；25K 只能作 fail-fast screen。
9. 在 G0 测完整 head latency、active memory、p50/p90 和 identity parity；最终仍必须通过 fresh dynamic EAL 与 same-stack SGLang paired TPS CI。
10. 在正文中明确：perfect-two oracle 只证明多纠错的必要性，不证明 frozen-lattice selectability、all-prefix sufficiency 或 dynamic success。

## Verdict

**VERDICT: REVISE**

APEX-16 的问题锚点与梯度诊断是正确的，all-prefix transformation 也值得做严格 matched falsification；但在新颖性、label leakage、heldout freshness、多纠错因果证据和扩展可行性修正前，不能进入 READY。

</details>
