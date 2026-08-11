# Round 1 Refinement：PGCF-16 全节点并行全局 Candidate Head

> 状态：ARIS research-refine Round 1 修订稿  
> 唯一有效约束：`USER_CONSTRAINT_CONTRACT.md`  
> 主变化：删除 4-mode 压缩和多项补丁式 loss，改成完整 256 candidate nodes 的固定深度非因果全局混合

## Anchor Check

- **核心要求没有变化：** 一次读取完整 `H[16]`，每个输出位置在选 token 前都能看到全部 16 个位置，一次得到 `[B,16,16]` scores，并用逐位置张量 argmax 输出唯一 `[B,16]` 序列。
- **严格禁止：** selected-token feedback、GRU rollout、causal/triangular mask、位置 decode loop、第二次 head、serial target decode、target seed、Jacobi/迭代修复、Viterbi/DP、beam/tree/trie/forest、多路径 verifier。
- **线上 target 权限：** 只有普通的最终单链 verifier；所有 target posterior/gold 只作离线 label。
- **主目标：** disjoint held-out fixed 与 dynamic EAL 都至少为同 job released Domino 的 `1.15x`，随后同栈 A40 SGLang TPS 至少 `1.15x`。

## 对 Round-0 审查的逐项处理

### 1. 修正不可部署 oracle

- 主上界改为纯 `base_topk_ids[..., :16]` oracle：`10.909256560`。
- `10.999878523` 是 `base Top-15 + released causal Domino action` union，只保留为离线诊断，不能进入 head 输入、candidate coverage、gap recovery 或任何 GO gate。
- 达到 `8.325485909` 需要相对 base Top-1 增加 `2.256972789` EAL，回收 deployable gap 的 `46.6245%`。
- teacher action 不在 base Top-16 的行只 mask；绝不把 action 注入在线 candidate set。

### 2. 删除 L15→L16 claim-bearing 迁移

- 旧 25K/199.8K OPB cache 只用于定位原始 prompt/anchor，不直接训练最终 head。
- 先在现有 R047 full16 上做 512-block mechanics/capacity。
- 通过后，用同一 25K OPB prompts 重采 `H[16]`、纯 base Top-16、gold、released policy、target candidate labels；数据收集、训练、evaluation 全程 L=16。
- 不保存或读取 `target_anchor_early_feature` 等在线不可得特征。

### 3. 删除 4-mode 信息瓶颈

- Round-0 的 `16 candidates → 4 modes` 是非单射压缩，现从主方法完全删除。
- 新主方法直接把全部 `16 positions × 16 candidates = 256` nodes 放入无 mask self-attention；任一候选可直接读取任一位置的任一候选。
- 因此不再需要 R4/R8 结构选择，也不声称压缩无损。若未来为了 kernel 压缩 modes，只能在主方法过效力门以后作为系统 ablation，不能取代当前主机制。

### 4. 收缩 loss 与容量扩展

- 删除 dynamic temperature、protection hinge 和首轮 LoRA。
- 首版只保留一个 FP32 log-space soft-prefix 主项、一个小权重 target-candidate KL、一个仅 warm-start 使用并归零的 teacher CE。
- 只有 train/capacity 同时欠拟合而非 held-out transfer failure 时，才允许增加宽度/深度；不能用参数量掩盖泛化失败。

### 5. 修正 evidence split

- 历史 `validation_gate` 已被 Phase-3 复盘查看，降级为 development-only，不再称 sealed。
- `validation_select` 只用于开发期 checkpoint/recipe 选择。
- 最终 claim 使用仍未打开的 600-prompt reserved formal test；方法、seed aggregation、checkpoint rule 和 evaluator 在收集/读取它之前冻结。
- 按用户要求不设置哈希式形式化 gate；只做有意义的 sample-ID/anchor overlap、tensor geometry 和 evaluator identity 检查。

---

# Revised Proposal：Parallel Global Candidate Fusion Head（PGCF-16）

## 1. 定量目标与机制假设

R047 full16 `validation_select` 的 prompt-balanced 事实：

| 系统/上界 | EAL |
|---|---:|
| Domino-DFlash base Top-1 | 6.068513120 |
| released Domino | 7.239552964 |
| required `1.15 × Domino` | 8.325485909 |
| deployable base Top-16 oracle | 10.909256560 |
| supported-policy + base fallback teacher ceiling | 7.180272109 |

候选可用性足够，但实现目标要求回收 base→oracle gap 的 `46.6245%`。本方法检验的唯一主假设是：

> released Domino-DFlash 的 full16 parallel hidden 与纯 base-Top16 lattice 中，存在能够由一个轻量、固定深度、全节点非因果网络直接解码的跨位置一致性；用大规模 Domino action warm-start 建立稳定初始化，再用 target accepted-prefix objective，可把这种一致性变成唯一并行 proposal 的长前缀收益。

历史 weak-backbone/L15 direct selector 的小增益和 high-capacity transfer failure说明“只扩 frozen MLP”无效，但没有检验 `strong Domino-DFlash backbone + full16 + 199.8K full16 teacher warm-start + direct full-node global mixer` 的合取。

## 2. 推理架构

### 2.1 唯一线上输入

一次 non-causal Domino-DFlash backbone forward 产生：

- `H ∈ R[B,16,2560]`；
- base candidate IDs `C ∈ N[B,16,16]`；
- 对应 base logits `B ∈ R[B,16,16]`；
- 当前 anchor token embedding；
- frozen shared embedding rows `E[C]`。

所有 scalar features 只从当前 Top-16 内计算：centered logit、Top16 conditional log-prob、Top1 gap、normalized rank、Top16 entropy。无需 full-vocabulary logsumexp，更不使用 target logsumexp。

### 2.2 Candidate-specific encoder

默认 `d=256`。对位置 `i`、候选 `k`：

```text
q_i   = W_h RMS(H_i) + W_e RMS(E_anchor)
e_ik  = W_e RMS(E[C_ik])
x0_ik = LN(q_i + e_ik
            + W_mul(q_i ⊙ e_ik)
            + W_phi(phi_ik)
            + p_i + r_k)
```

`q_i ⊙ e_ik` 保留 hidden-token compatibility，但只加一个 `d×d` 投影，不再使用 Round-0 的大 compatibility MLP。

训练时 `W_e` 从 gathered frozen embeddings 接收梯度。checkpoint 冻结后，将
`RMS(E_vocab) W_e` 一次性预投影成 BF16 table；线上只 gather `256×d` rows，不执行
`256×2560×d` projection。Qwen3 vocab 151,936 时该 derived table 为 77.8 MB
（74.2 MiB），是 frozen resident buffer，不是第二个模型或 trainable vocabulary。

### 2.3 完整 256-node non-causal fusion

把 `x0` reshape 为 `X0 ∈ R[B,256,256]`。运行两层不共享参数的 pre-norm Transformer：

```text
A_l       = FullSelfAttention(LN(X_l))     # attention mask = None
X'_l      = X_l + A_l
X_{l+1}   = X'_l + FFN_2x(LN(X'_l))
```

- `l ∈ {0,1}`，8 heads，FFN ratio 2，dropout 0；
- attention matrix 是完整 `256×256`，没有 causal、triangular 或 path mask；
- 每层只加按 draft-position distance 的 learned relative bias 与 same-position bias；它们改变权重但不屏蔽任何 node；
- `X_l` 始终是连续 candidate states；层间没有 argmax、token ID 选择或 selected-token embedding；
- 网络层的固定顺序不是 token-position autoregression，16 个位置没有串行数据依赖。

### 2.4 一次输出唯一序列

```text
delta_ik = W_out LN(X2_ik)          # W_out zero-init
score_ik = B_ik + delta_ik
proposal = gather(C, argmax_k(score))
```

一个 invocation 同时输出 `[B,16,16]`，一个张量 argmax 得到唯一 `[B,16]`。不存在第二次 head、path score、序列搜索或多分支 verifier。zero-init 时 scores 与 base logits 完全相同，必须逐 token 复现 full16 base Top-1。

## 3. 精确首版成本合同

首版 module 的 trainable parameter budget 预注册如下，并由实现测试精确断言：

| component | parameters |
|---|---:|
| `W_h` + shared `W_e` (`2560→256`, no bias) | 1,310,720 |
| position + rank embeddings | 8,192 |
| five-scalar projection | 1,536 |
| `W_mul (256→256)` | 65,536 |
| input LayerNorm | 512 |
| two full-node attention/FFN2x blocks | 1,051,136 |
| output LayerNorm + zero-init scorer | 768 |
| **total trainable** | **2,438,400** |

该数是 537.427M headless DFlash draft 的约 `0.454%`。预投影 vocab table额外驻留 77.8 MB BF16。batch1/L16/K16 的 dense online matrix MAC 约 `0.365G`，不含 shared base vocab GEMM/Top16；最终不以公式代替实测。

在 199.8K full training 前，A40 eager batch1 profile必须覆盖完整公平路径：base vocab GEMM、Top16、projected-table gather、PGCF、argmax。相同输入/precision/warmup 下与 released eager Domino完整 head比较，development指导线为 `PGCF pipeline ≤1.2× Domino pipeline`。失败则停止大训练并先优化/缩小固定 head；不能用未优化 Domino 作为对手。

## 4. Full16 数据合同

### 4.1 Mechanics/capacity 数据

从 `r047_anchor_t4_train_10164718` 固定抽 512 blocks。字段：

- online features：`parallel_hidden[16]`、纯 `base_topk_ids/logits[16,16]`、anchor/candidate embeddings；
- offline labels：`gold_ids[16]`、`policy_ids[16]`、`target_candidate_logits[16,16]`、`target_top1_ids`；
- prohibited input：任何 `target_*feature/hidden/context`、gold、policy、accepted length。

### 4.2 Full16 OPB Stage A

capacity 和 latency 都通过后，用 OPB part000–003 的 25K disjoint prompts重采约 199.8K blocks：

1. released Domino-DFlash 以 `is_causal=False` 一次产生 full `H[16]`；
2. shared LM head生成纯 base Top-16；
3. released Domino action仅作离线 teacher label；
4. clean target verifier产生 full16 gold 和同候选 target logits；
5. 第16位 label、hidden、candidate、loss 与 evaluator geometry显式一致；
6. 不记录 target early feature，不把 causal action并入 Top-16。

OPB、Phase3 train、development和 formal test 用 sample ID + anchor offset 检查交集为零。此检查服务真实数据边界，不做无意义的 artifact hash gating。

## 5. 目标函数与 curriculum

令 `s_i` 是 16-way scores，`q_i=softmax(s_i)`，`y_i` 是 gold 在 base Top-16 中的 rank。令 `a_i=1[y_i exists]`，`m_t=∏_{j≤t} a_j`。主项用 FP32 log-space计算：

```text
log_q_gold[i] = log_softmax(s.float(), dim=-1)[y_i]
log_survival[t] = sum_{j<=t} log_q_gold[j]
U_prefix = sum_t m_t * exp(log_survival[t])
L_prefix = -mean_batch(U_prefix / 16)
```

不存在 `clamp` 或 detached greedy reach；首个 out-of-K position及其 suffix由 `m_t=0` 自然截断。该目标直接把早位正确选择产生的 continuation value分配给早位概率。

target dense auxiliary：

- `pT_i = softmax(target_candidate_logits_i / T)`，首版 `T=1`；
- 只有 clean target top1 与 gold 连续一致到当前位置且 gold in-K 的 rows 有效，首个 replay mismatch 后 suffix 全部 mask；
- `L_KL = mean_valid KL(pT_i || q_i)`，固定权重 `0.05`。

teacher warm-start：

- `policy_id ∈ base Top-16` 的 rows 才有 CE；unsupported rows 为零；
- 前 10% updates 为 teacher CE warm-start；接下来 20% 将 teacher weight从 1 线性降到 0，同时开启 prefix/KL；剩余 70% 完全 target-only；
- teacher action reconstruction、teacher/no-teacher ablation只用于判断 warm-start贡献，不将 causal action带到推理。

最终首版只有：

```text
L = L_prefix + 0.05 * L_KL + lambda_teacher(t) * L_teacher_CE
```

无 protection hinge、dynamic temperature、LoRA 或其它 repair loss。

## 6. 逐级实验与硬门

### Gate 0：代码/合规

必须同时通过：

1. forward signature只含 online tensors，输出 `[B,16,16]` 与唯一 `[B,16]`；
2. `scope=global, L=16, K=16` 硬编码在 claim-bearing launcher，不能切 causal；
3. 无 position decode loop、GRU、selected-token tensor、第二次 invocation；
4. zero-init逐 token复现 base Top-1，prompt-balanced EAL `6.068513120`；
5. 在一个固定 deterministic nonzero scorer witness 上，修改 position 15 的 hidden/candidate/logit会改变 position 0 score，反向同样成立（默认 zero scorer因 identity contract不用于该影响测试）；
6. target/gold/policy tensor不在 forward graph；
7. parameter count精确为 `2,438,400`；
8. pure base16 oracle精确为 `10.909256560`。

### Gate 1：512-block full16 capacity

same-set capacity只证明实现/优化，不算泛化。固定 d256/L2 full-node必须达到：

- gold-in-K candidate accuracy `≥99%`；
- hard/non-Top1 accuracy `≥97%`；
- deployable base16 oracle-gap recovery `≥95%`；
- harmed blocks `≤1%`；
- supported Domino action reconstruction `≥99%` 的独立 warm-start capacity witness。

matched local control使用完全相同参数，仅把 attention限制到同一 position。它不参与主结构选择；其作用是证明 global computation可被干净隔离。若 full-node capacity失败，先修实现/优化；禁止转 causal/tree。

### Gate 2：A40 mechanics 与小型 disjoint signal

- 完整 eager head pipeline `≤1.2×` eager Domino development指导线；
- 在 15,886 train / 1,175 `validation_select` 的 disjoint split上，global必须显著优于 matched local和 base；
- remote-position shuffle/intervention必须消除大部分 global增益，而不改变本位置 candidate multiset；
- 若 train capacity高而 held-out无增益，判 transfer failure，不增加参数；
- 若 train与held-out都呈一致欠拟合，才允许一次预注册扩容至 d384/L2（仍远低于10.75M），并重新 profile。

只有 global signal与成本同时过门才启动25K full16重采/正式训练。

### Gate 3：25K full16 Stage A

- 先报告 gold/policy base-Top16 support；
- teacher-only warm-start需在 supported rows 重构 action，并接近精确 support-constrained teacher ceiling `7.180272109`（unsupported teacher action固定回退 base Top-1）；
- target stage的 `validation_select` EAL至少超过 released Domino，并达到 `≥7.55`，三域均高于 base；
- checkpoint只由 prompt-balanced fixed EAL选择，不用 token CE、same-set capacity或 oracle选择。

未过门时只做 loss/fit/representation诊断；不进入系统小修或 off-spec机制。

### Gate 4：in-domain adaptation 与开发冻结

用 frozen结构继续在 full16 OPB:Phase3 train按 `3:1` block比例混合，低学习率 target-only adaptation；不解冻 DFlash。architecture、loss、schedule、checkpoint rule在此阶段结束后冻结。

开发门：

- fixed `validation_select` EAL `≥8.325485909`；
- 相对 same-job Domino paired prompt bootstrap方向稳定；
- chat/code/math均无实质退化；
- 不能用已被查看的 `validation_gate` 作为新证据。

### Gate 5：fresh formal、dynamic 与系统

方法冻结后才收集/读取 600-prompt reserved formal test 的 full16 labels：

- fixed formal EAL `≥1.15×` same-job released Domino；
- frozen checkpoint重新 dynamic rollout，dynamic EAL `≥1.15×` same-job Domino；
- fixed/dynamic都通过后才集成 SGLang；
- 最终同 A40、batch/workload/precision/stack 的 end-to-end TPS `≥1.15×` released Domino，并同时保留 accepted-length优势。

## 7. 失败判因与允许的优化

| 观察 | 结论 | 允许动作 |
|---|---|---|
| 512 blocks拟合失败 | implementation/optimization/capacity | 修索引、loss、数值；一次 d384扩容 |
| train高、disjoint低 | representation transfer failure | 不加参数；审计 online信息与标签分布 |
| global≈local，shuffle无影响 | 未利用跨位置信息 | 关闭当前机制，不转自回归/tree |
| EAL好但 latency超线 | kernel/width问题 | 固定输出语义下融合、预投影、CUDA graph或缩 d；重新测EAL |
| fixed过门、dynamic失败 | canonical→online boundary shift | 用同一并行 head做 on-policy DAgger重采；仍无串行/树 |
| EAL与TPS都过门 | 成功 | 再做 necessity ablation与论文证据 |

## 8. Claim boundary

若全部门通过，可主张：

> PGCF-16 是一个轻量、固定深度、全块非因果、candidate-specific 的 direct selector；它让完整 16×16 DFlash lattice 在一次并行前向中全节点交互，并在没有 selected-token feedback、序列搜索或多路径验证的情况下输出唯一16-token proposal。

不声称 attention、Top-K、distillation或prefix loss单独新颖；不把 `10.999879` 当部署上界；不引用 R050–R056 的 target-seed/tree结果支持本方法；未过 fresh formal/dynamic/TPS 三门前不声称最终成功。
