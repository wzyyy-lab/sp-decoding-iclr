# PGMF-16 Round 0 独立方法审查

`review_independence: same-family`  
`acceptance_status: provisional`  
`VERDICT: REFINE`  
`OVERALL SCORE: 7.1 / 10`

PGMF-16 的推理图与 Problem Anchor 基本一致：完整读取 16 个位置、无因果 mask、一次并行输出 `[B,16,16]`、最终只产生一条序列，也没有 selected-token feedback、额外 target inference 或多路径验证。方向值得继续，但当前存在三项 blocker，尚不能进入 199.8K 正式训练，更不能判定 READY。

## 1. Anchor alignment

| 硬约束 | 审查结果 |
|---|---|
| 一次读取完整 16-position block | 推理设计满足；现有 Stage A 数据不满足 |
| 每个位置非因果看到全部 16 positions | 结构依赖满足 |
| 16 个决策一次并行产生 | 满足 |
| 只输出一条序列 | 满足 |
| 无 selected-token feedback | 满足 |
| 无 GRU/串行 decode/迭代修正 | 满足；建议把“feedback rounds”改称“fixed-depth feedback layers” |
| 无 tree/beam/multipath | 满足 |
| 无额外 target inference | 满足 |
| 轻量、可达吞吐目标 | 尚未由精确参数和 latency 证明 |
| fixed/dynamic 均达到 Domino 1.15x | 仅为计划，无结果 |

`candidate → global modes → candidate` 确实使每个最终候选具有通往全部 16 个位置的计算路径，因此满足“跨位置可见性”。但它不保证看到全部原始 candidate 信息：每个位置的 16 个 candidate nodes 在跨位置交换前被压缩成 4 个 modes，这是非单射的信息瓶颈。两层固定神经反馈可以缓解，但不会自动使压缩无损。

## 2. 定量口径重算

在 R047 `validation_select` 的 1,175 blocks / 147 prompts 上：

| 指标 | 正确值 |
|---|---:|
| Domino-DFlash base Top-1 | 6.068513120 |
| released Domino | 7.239552964 |
| `1.15 × Domino` | 8.325485909 |
| 纯 deployable base Top-16 oracle | 10.909256560 |
| Top-15 + released Domino action union oracle | 10.999878523 |

初稿使用的 `10.999878523` 不是 PGMF 可部署的纯 Top-16 oracle。它在 released action 不属于 base Top-16 时，用该 causal Domino action 替换 rank-16；PGMF 推理没有这个 action。主表和门槛必须改用 `10.909256560`。

相应地：

- 相对 base Top-1 必须增加 `2.256972789` EAL。
- 相对 Domino 必须增加 `1.085932945` EAL。
- 必须回收 deployable base→oracle gap 的 `46.6245%`。
- 等价地，必须回收 Domino→deployable-oracle gap 的 `29.5918%`。
- 达到门槛后距离 deployable oracle仍有 `2.583770651` EAL。

## 3. 仓库历史审计

“仓库从未执行过 199.8K Domino-action warm-start + full16 noncausal direct `[B,16,16]` scorer”这一判断成立，但需要准确描述已有的相邻实验：

- `global_direct_selector.py` 做过无因果 direct candidate scoring；最大实验使用 793,989 blocks、99,356 prompts、L=15、27.48M 参数，但来自较弱 pure-DFlash backbone，没有 Domino-action warm-start，也不是 full16；held-out 只增加约 `0.07799`。
- `GlobalLookaheadCausalSelector` 使用过包括上述 199.8K OPB blocks 在内的 295,604 blocks，也有 global modes，但保留 GRU selected-token feedback 和串行 decode，属于当前合同明确禁止的路线。
- PLC 是并行单链，但只预测 15 个 correction positions、position 0 固定；candidate 在全局交换前压缩成 modes，之后没有 candidate-specific feedback。其 imitation 只有 1,024 blocks，acceptance stage 才使用 15,886 blocks，最好 EAL 为 `6.093415938`。
- 因此，“强 Domino-DFlash backbone + full16 direct scorer + 199.8K teacher warm-start + candidate feedback”的合取确实没有被检验过。但每个组件并非全新，不能把整个模块列表直接当作 novelty。

## 4. 最大三项 blocker

### Blocker 1：主 oracle 使用了推理时不可得的 causal Domino action

这不仅是小数误差，而是候选集合定义和架构合同不一致。若论文用 `10.999879` 计算回收比例，会把 off-spec teacher union 当作 PGMF 的部署上界。

具体修订：

1. 主上界统一改为 `all16_base16 = 10.909256560`。
2. `all16_k16 = 10.999878523` 只能标为“含 released causal action 的离线诊断”，不能进入 PGMF 主回收比例。
3. 所有 candidate coverage、oracle recovery 和 capacity gate 均以纯 `base_topk_ids[:,:,:16]` 为准。
4. Teacher action 不在 base Top-16 的行只能 mask，不允许在线加入该 action。

### Blocker 2：Stage A 的 L15→L16 迁移与 teacher 语义没有闭合

现有四个 OPB cache 共 25K prompts / 199,800 blocks，但只有 15 位。把模型容量设为 max-L=16 并不能训练第 16 个 position embedding，也不能训练 64-mode full16 mixing 分布。

建议的确定路线是：

1. 先在现有 R047 full16 train 中抽 512 blocks 完成实现与 capacity gate。
2. capacity 通过后，立即用现有 `collect_gfpr_rollouts.py` 把 OPB 25K prompts 重采成 full16；不要在旧 L15 cache 上做 claim-bearing Stage A。
3. 新 cache 保存 `H[16]`、纯 base Top-16、target candidate logits/advantages、target logsumexp、gold、released policy 和 support masks。
4. Teacher CE 只作用于 `released_action ∈ base Top-16` 的行。released-correct-prefix protection 只能覆盖 teacher 与 gold 一致的真实 accepted prefix，不能保护 teacher 首错后的 suffix。
5. hard Domino action 只表述为 warm-start；是否携带可泛化顺序知识必须由 ablation 和 action-reconstruction 证据支持。

### Blocker 3：4-mode 压缩、loss 堆叠和 latency 都尚未被最小证据验证

当前同时引入太多未区分组件。只有 candidate-specific full-block feedback 是清晰主贡献。

具体修订：

1. 把固定层写成 `X^r → M_local^r → M_global^r → X^{r+1}`。`X` 始终是连续 candidate states；任何层中都不得出现 argmax token 或 selected-token embedding。
2. 首个 capacity 对照包括 PGMF R=4、PGMF R=8、1–2 层 full-node noncausal attention ceiling、matched local/no-global control。
3. full-node 只作为无压缩诊断；若 full-node capacity 通过而 R=4 失败，才增加 modes 或改用 full-node；若两者都失败，停止堆结构。
4. 不宣称 R=4 无损，只说每个候选读取由全部位置产生的多模态摘要。
5. 给出精确 trainable parameters、MACs、resident buffers；候选 embedding projection 应离线预投影为 frozen vocabulary table，并计入约 117MB BF16 驻留内存。
6. 199.8K 训练前做完整 A40 eager profile，覆盖 Top-16、gather、mixer、feedback、score、argmax；指导线约不高于 eager Domino head 的 1.2x。
7. 首版 loss 只保留 FP32 log-space soft prefix utility、小权重 target candidate KL/CE、warm-start 衰减 teacher CE。protection hinge 留作 necessity ablation，LoRA 不进首个 falsifier。

## 5. 修订后的 gates

### Gate 0：合规与数学

- 单次输出 `[B,16,16]` 和唯一 `[B,16]` proposal。
- 无 causal mask、selected-token input、位置 decode loop或第二次 head。
- zero residual 精确复现 full16 base Top-1，EAL 为 `6.068513120`。
- 修改远端位置的 candidate ID/logit/hidden 会改变其他位置 score。
- target tensors只进入 loss，不在 forward signature。
- pure base Top-16 oracle精确复现 `10.909256560`。
- prompt/anchor overlap 为零。

### Gate 1：512-block full16 capacity

- gold-in-K candidate accuracy ≥99%
- hard/non-top1 accuracy ≥97%
- deployable base16 oracle-gap recovery ≥95%
- harmed blocks ≤1%

若 R=4 失败而 full-node通过，说明是 mode compression blocker；若 full-node也失败，说明实现、优化或 online representation不足，不能扩数据。

### Gate 2：结构与成本选择

同数据、loss、budget 比较 local、R4、R8、full-node ceiling：global 稳定优于 local；远端 candidate shuffle 显著破坏 global gain；R4/R8 保留 full-node 大部分 held-out gain；完整 eager head约不超过 eager Domino head 1.2x。只冻结一个结构进入 Stage A。

### Gate 3：full16 Stage A

使用重采后的 OPB 25K/199.8K；报告 action/gold support；teacher warm-start 回收绝大部分 support-constrained teacher gain；三域均不能由单一域支撑。

### Gate 4：Stage B fixed efficacy

- gate 前冻结 recipe、architecture、loss、checkpoint rule。
- `validation_select` 只选 checkpoint；sealed `validation_gate` 只读一次。
- fixed EAL ≥ `8.325485909`，三域无实质退化，paired prompt bootstrap方向稳定。

### Gate 5：dynamic 与系统

- 冻结 checkpoint 重新 dynamic rollout。
- 若同 job Domino dynamic baseline仍为 `6.602820376`，门槛为 `7.593243432`。
- fixed/dynamic 都通过后才进入 SGLang；最终同 A40、同 workload、同栈 TPS ≥1.15x Domino。

## 6. Novelty 与 claim 边界

当前可以保留的主张是：

> 一个固定深度、全块非因果、candidate-specific 的 parallel selector，通过多模态全局摘要反馈，在没有 selected-token feedback 或多路径验证的情况下，将完整 DFlash candidate lattice映射成唯一序列。

只有 matched local/no-feedback、remote-candidate intervention、full16 held-out fixed/dynamic/TPS 和 teacher/mode ablation 全部成立后，该主张才具有论文强度。

## 7. 评分

| 维度 | 分数 | 评价 |
|---|---:|---|
| Problem Fidelity | 9.5 | 推理架构严格对齐 anchor |
| Method Specificity | 6.5 | 主图清楚，但 feedback、loss masks、参数和 inference cache未闭合 |
| Contribution Quality | 7.0 | candidate feedback可形成单一主贡献，但当前仍混入过多未经证实组件 |
| Frontier Leverage | 7.5 | 离线 teacher distillation使用自然，但其作用被过度表述 |
| Feasibility | 6.0 | 参数预算表面可行，L15→L16和投影 latency尚未解决 |
| Validation Focus | 6.5 | 有 gate 思路，但缺少压缩 ceiling、matched control 和精确阈值 |
| Venue Readiness | 5.5 | 若达到定量门槛会很强，目前仍属于高风险可证伪方案 |
| **加权总分** | **7.1** | **REFINE** |

最终结论：PGMF-16 是当前合同下合理、值得做一次严格 falsifier 的主路线，但必须先纠正 deployable oracle，完成 full16 OPB 重采，并用 full-node ceiling证明 4-mode compression不是新的信息瓶颈。三个 blocker 解决前不应启动 199.8K 正式训练。
