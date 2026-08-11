# Round 1 Refinement

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Anchor Check

- 上述五行与 round 0 逐字一致。
- 方法仍直接修改 released DFlash 内部表示，部署仍是一条 greedy 链、一次 draft parallel forward、一次 target verification；没有改成更容易的外部 reranking、tree decoding 或顺序 correction。
- Domino 只用于报告既有 same-anchor parallel-backbone gap 的描述性回收比例；训练量不同，绝不做架构因果归因，也不把“击败 Domino”偷偷改成成功门槛。
- 数据与 R083 完全隔离；旧 downstream outcome 不参与训练、选择或评估。

## Simplicity Check

- 删除 optional top-M KL、EMA self-teacher、周期性 mask refresh、post-hoc fallback 和 rank grid。
- 删除 hard reachable-support censoring；它在本项目已被 capacity gate 证伪，会饿死 hard-position 梯度。
- 核心只剩一个可合并 LoRA 参数化和一个约束问题：完整 D-PACE 维持全位置学习，dynamic breaker 提供额外修复方向，projected update 保护 frozen accepted prefix。
- 四个训练对照不是四个模块，而是同一目标的最小 `dynamic × constraint` 因果分解；所有对照在同一次 sealed falsifier opening 中物化。

## Changes Made

### 1. 从软惩罚改为 first-break active-set constrained adaptation

- 将方法重命名为 **FBAC-DFlash (First-Break Active-Set Constrained Adaptation)**，不再用未被保证的 “safe policy improvement” 表述。
- 用 frozen base first-break 定义不可回退的 prefix constraint；用 current adapted first-break 定义 repair active set。
- 用 projected AdamW proposal + exact batch-feasibility backtracking 代替 `lambda_protect`。若当前 batch 已不可行，先执行 restoration；不能恢复则 fail closed。
- 补全 full-correct、regression breaker、base-wrong breaker、detached argmax、tie、归一化和 mask 刷新语义。

### 2. 正面吸收旧 reachable-support 负结果

- 旧 `lambda=0/0.1` hard reachable-support capacity cells 分别只有 `0.940639/0.949772` hard-candidate accuracy，而 full Candidate-D-PACE control 为 `1.0`；因此新版绝不删除 breaker 后的监督。
- 完整 full-vocabulary D-PACE `alpha=.5` 在每个有效 draft 位置保留；FBAC 是受约束的增量修复，而不是旧路线改名。
- 修正 DFlash block 语义：block size 16 包含一个已知 anchor，真正预测与计分的是 `L=15` 个 draft positions；EAL verification advance 为 `1 + accepted_draft_tokens`。

### 3. 收紧新颖性与验证因果链

- 明示 D-PACE 已提供 acceptance-aware 动态权重；FBAC 的区别仅是 hard greedy active breaker 加上 enforceable base-prefix update constraint。
- 加入一个精确但有限的命题：若 frozen accepted prefix 上的 adapted gold margin 保持正，则该 block 不会比 frozen DFlash 更早失败；若 frozen breaker margin 也转正，则至少多接受一个 draft token。该命题不外推到未见数据。
- 预冻结 released、matched D-PACE LoRA、static-break constrained、dynamic-no-constraint、full FBAC 五个 arms，在一个共同 falsifier opening 评估。
- 数据按 near-duplicate connected component 分组，selection 改为 safety-feasible-first，并给出所有 CI/margin/latency 规则。

## Revised Proposal

# Research Proposal: FBAC-DFlash——面向首断点的主动集约束适配

> 路线身份：`prospective-v2`。这是 R083 已关闭路线之外的一条全新、前瞻式路线；不得把本方案、数据或实验描述成 R083 的重试、修复或下游阶段。

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Technical Gap

### 证据链

1. Released DFlash 的本地同锚点 EAL 为 `5.1120019436`，K=16 candidate oracle 约为 `9.727`：候选可用，实际首错位没有被安全修复。
2. GCLS 最佳约 `+0.28499 EAL`、harm `7.32%`；FMAS/SAVS/CAMRS 的完整训练分别出现 identity collapse、约 1,518 倍 harmful-gradient dominance、tail/calibration 失效。27,482,160 参数 D640 frozen-feature teacher 在 99,356 prompts 上仅 `+0.07799 EAL`，比 compact d64 低 `0.15063`，prompt-bootstrap CI `[-0.23652,-0.06803]`。因此不再扩大 frozen selector。
3. PROS R082 几乎等于 Direct：beneficial APPLY `174/174`，harmful KEEP `5/101`，harm `6%`，相对 Direct 仅 `+0.003125 EAL`。选择更多动作没有解决内部表征的错误边界。
4. 旧 prediction-conditioned reachable-support 路线已经给出必要的反例：同一 128-block capacity probe 中，完整 Candidate-D-PACE control 的 hard-candidate accuracy 为 `1.0`，hard/soft reachable censoring 只有 `0.940639/0.949772`，路线被正式关闭。结论不是“首断点无用”，而是**不能通过砍掉不可达 suffix 的直接监督来学习首断点**。
5. released DFlash 有 5 个 parallel draft layers、hidden size 2560、block size 16；已知 anchor 占 position 0，真实预测 `L=15` 个位置。五个 target layers `[1,9,17,25,33]` 经共享 `fc` 融合；外部 selector 无法改变最终首错位，而 merged weight update 可以。
6. 本地 Domino parallel backbone EAL `5.93853`、on-policy GRU `7.01579`；DFlash 到 Domino-parallel 的描述性 gap 是 `0.82653`。训练数据量不匹配，因此 gap 只能用于量化回收比例，不能归因为某个结构。

### 前沿边界

- [DFlash](https://arxiv.org/abs/2602.06036) 是复用的并行 drafter，不是贡献。
- [D-PACE](https://arxiv.org/abs/2605.18810) 已对 accepted-length surrogate 做动态 position weighting；FBAC 不声称首次 acceptance-aware training。
- [Domino](https://arxiv.org/abs/2605.29707) 使用 prefix-causal correction；FBAC 不加顺序 head。
- [DFlare](https://arxiv.org/abs/2606.02091) 使用逐层 target fusion、异构 KV projection 与 progressive loss；FBAC 不改 fusion 接口。
- [DeLS-Spec](https://arxiv.org/abs/2607.07409) 与 [DSpark](https://arxiv.org/abs/2607.05147) 使用顺序/局部 expert 或运行时融合；FBAC 不加第二 expert 或第二 draft forward。

缺失的机制不是又一种位置权重，而是：**在完整 D-PACE 全位置梯度仍存在时，把 current hard-greedy breaker 作为 repair active set，并把 frozen accepted prefix 变成每次训练更新必须满足的显式可行域。**

## Method Thesis

- One-sentence thesis: 在完整 D-PACE 训练上，对 current first breaker 施加增量 margin repair，并把每个 AdamW proposal 投影回 frozen accepted-prefix margin 可行域，可以直接适配 DFlash 内部决策边界，同时保持 merged 后的单次并行推理图。
- Why this is the smallest adequate intervention: 不改 target/fusion/decoder topology，不加推理模块；只增加约 1.835M 个最终合并的低秩训练参数和一个 training-only projected optimizer。
- Why this route is timely: merged PEFT 让 representation adaptation 不产生部署 sidecar；约束优化把 foundation-model finetuning 中常见的“平均收益换局部遗忘”改写为与 speculative prefix 语义一致的 active-set 更新。

## Contribution Focus

- Dominant contribution: **First-Break Active-Set Constrained Adaptation**——full D-PACE 保留所有位置可学习性，current breaker 决定增量 repair，frozen accepted prefix 决定不可回退的 update constraint。
- Optional supporting contribution: 无。LoRA merging 是实现载体。
- Explicit non-contributions: LoRA、D-PACE、margin loss、projected gradient、curriculum、parallel decoding 均不单独声称新颖；不声称全局/未见数据安全保证。

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: Qwen3-4B target 全冻结；released DFlash 的 embedding、LM head、共享 `fc`、前三层 draft、norm 与原权重冻结。
- New trainable components: 最后两层的 `q/k/v/o` 与 `gate/up/down` projections 上 rank-16、alpha-16、dropout-0 LoRA，精确参数数为 `1,835,008`；训练后 float32 merge，再保存为部署 dtype。
- Training-only state: frozen-base reference branch和 projected AdamW 的 moment/state；均不保存到部署执行图。
- Intentionally excluded: selector、RNN、第二 expert、layer-wise fusion、top-M KL、EMA teacher、tree/multipath、runtime threshold、第二 draft forward。

### System Overview

```text
new prospective target-generated sequence
  -> frozen target context features
  -> frozen released DFlash: b, m_0, protected prefix P
  -> DFlash + LoRA: z, current breaker m_theta
  -> full D-PACE on all 15 valid draft positions
  -> breaker-margin task gradient
  -> projected AdamW proposal constrained by P
  -> merge LoRA into the original two layer weights

deployment = original DFlash graph, original one parallel draft forward,
             original one target verification, no adapter operator
```

### Core Mechanism

#### Definitions and deterministic semantics

For each valid block `b`, let `L=15`, gold target-greedy draft tokens be `y_{b,1:L}`, frozen DFlash logits be `b_{b,i}`, and adapted logits be `z_{b,i}`. All decision arithmetic is float32.

- `argmax` uses PyTorch deterministic lowest-vocabulary-index tie breaking.
- `hat_y_{b,i}=argmax_v z_{b,i}(v)` and `hat_y^0_{b,i}=argmax_v b_{b,i}(v)`.
- `m_theta=min{i:hat_y_{b,i}!=y_{b,i}}`; if none, `m_theta=L+1`. `m_0` is defined identically from frozen logits.
- Both indices and every Boolean active mask are `stop_gradient` and recomputed once on every forward; there is no EMA or periodic refresh.
- Gold margin is `gamma_i(z)=z_i(y_i)-max_{v!=y_i}z_i(v)`. The non-gold maximum is explicit; the wrong winner is not cached across updates.
- Frozen protected set is `P_b={i:i<m_0}`; if frozen is fully correct, `P_b={1,...,L}`; if it fails at position 1, `P_b` is empty.

#### Full D-PACE coverage term: never censored

For every valid draft position,

`q_i=softmax(z_i)[y_i]`, `s_i=0.5*q_i+0.5`, `p_i=prod_{j<=i}s_j`, and `w_i=sum_{k=i}^L p_k`.

`w_i` is computed in float32 and detached exactly as official D-PACE. With validity mask `v_i`,

`L_DPACE=(1/B) * sum_b sum_i v_{b,i} w_{b,i} CE(z_{b,i},y_{b,i})`.

There is no moving denominator and no zeroing after `m_theta`. An `eta=0` parity test must match the pinned D-PACE value and every LoRA gradient within numeric tolerance.

#### Current-breaker repair active set

Let `epsilon_tie=1e-4`. For every block with `m_theta<=L`, define target margin

```text
tau_b = max(epsilon_tie, 0.5*gamma^0_{b,m_theta})  if m_theta < m_0
        epsilon_tie                                if m_theta >= m_0
```

The first branch is an adapted regression inside the frozen accepted prefix; the second is the current extension frontier. Fully correct blocks have no repair term. The block-balanced loss is

`L_break = sum_b 1[m_theta<=L] * [tau_b-gamma_{b,m_theta}(z)]_+ / max(sum_b 1[m_theta<=L],1)`.

The unconstrained task objective is `L_task=L_DPACE+L_break`; its coefficient is fixed to 1 before data, not grid-searched.

#### Frozen-prefix feasible set and projected update

For every `i in P_b`, define the retained reference margin

`r_{b,i}=max(epsilon_tie,0.5*gamma_i(b))`.

The batch constraint is the worst protected-position residual

`H_B(theta)=max_{b,i in P_b}(r_{b,i}-gamma_i(z_theta)) <= 0`.

If the batch has no protected position, the constraint is vacuous. Zero-init LoRA is feasible by construction except an audited exact-tie edge case, which must enter restoration before any task update.

One **projected AdamW** step is:

```text
m_theta, L_task, H = forward(theta)
if H > 0:
    d0 = AdamW_proposal(gradient(H))                 # restoration only
else:
    d0 = AdamW_proposal(gradient(L_task))
    gH = gradient(H)                                 # active worst constraint
    if H + dot(gH,d0) > 0:
        d0 = d0 - (H + dot(gH,d0))/(||gH||^2+1e-12) * gH

for alpha in [1, 1/2, 1/4, ..., 1/128]:
    if exact_forward_H(theta + alpha*d0) <= 0:
        commit theta + alpha*d0 and the single AdamW moment update
        break
else:
    skip parameter commit; log infeasible_step
```

The same frozen target features/base logits are reused during line search; only the five-layer draft forward repeats. The capacity gate fails if restoration cannot reach feasibility, if more than 1% of proposals are skipped, or if median accepted `alpha<1/4` after warmup. This is a training-batch constraint, not a theorem about unseen prompts.

#### Exact local property

For any fixed block, if `gamma_i(z)>0` for every `i<m_0`, deterministic greedy prediction matches gold on the entire frozen accepted prefix, so `m_theta>=m_0` and adapted accepted length cannot be shorter on that block. If additionally `m_0<=L` and `gamma_{m_0}(z)>0`, then `m_theta>=m_0+1`, so verification accepts at least one extra draft token. This follows directly from the first-mismatch definition. FBAC uses the first statement as its feasible set and the second as its repair target; D-PACE weights positions but does not enforce either condition.

### Optional Supporting Component

NONE. Any target-logit/hidden distillation, EMA mask, extra rank or runtime gate is a separate future route and cannot be added after seeing the falsifier.

### Modern Primitive Usage

- Primitive: mergeable LoRA/PEFT over a frozen foundation-model drafter.
- Role: low-dimensional parameter subspace for constrained internal adaptation; target only supplies existing training-time hidden context and T=0 gold continuation.
- Fit: unlike an external classifier or recurrent expert, merged PEFT changes the faulty representation without changing deployment topology.

### Integration into Base Generator / Downstream Pipeline

1. Load pinned Qwen3-4B target and released DFlash. Zero adapter must be bitwise-equal in argmax/accepted length and tolerance-equal in logits.
2. Run target in `no_grad` for training context features; run frozen and adapted draft branches on the same sampled anchors.
3. Update only rank-16 LoRA by the exact projected optimizer; log current/frozen breaker, constraint residual, projection norm, line-search alpha and skip/restoration events.
4. Merge LoRA in float32 into original weights, remove every adapter wrapper, save deployment dtype.
5. Require identical module/operator graph, dtype, attention/kernel path, draft/target forward counts and block size. Unmerged-vs-merged logits must satisfy `atol=0.02, rtol=0.02`, with exact argmax and accepted-length equality on the merge audit set.
6. Output tokens remain target-exact because ordinary target verification is unchanged.

### Training Plan

1. **Prospective manifest before outcomes.** Select 10,500 new Open-PerfectBlend remainder prompts: 8,000 fit, 1,000 checkpoint, 1,500 one-shot falsifier. Exclude exact and 8-gram-Jaccard `>=0.5` overlap against the old 100k and hash-only prior exclusion index; never read old outcome/model-score artifacts.
2. **Group isolation.** Build near-duplicate connected components using normalized full conversation/document fingerprints and 8-gram overlap; assign whole components, never rows, to fit/checkpoint/falsifier while stratifying domain. An independent script replays component disjointness and all hashes before any target generation.
3. **Power contract.** Before freezing counts, use only prior producer-train prompt aggregates to verify that 1,500 falsifier prompts provide at least 80% paired power at two-sided alpha .05 for `+0.30` EAL versus DFlash and `+0.10` versus matched D-PACE under the conservative observed paired SD. If not, increase falsifier size before collection; never decrease it after outcomes.
4. **Capacity gate on 512 fit blocks.** Verify 15-position indexing; official D-PACE loss/all-gradient parity; nonzero suffix gradients; deterministic tie/full-correct/regression cases; zero-init feasibility; projection/backtracking; overfit repair without base-prefix regression; exact merge path. Any failure closes implementation.
5. **Equal-budget arms.** Freeze one rank/layer placement, optimizer, data order, step count and checkpoint schedule for: (A) matched D-PACE LoRA; (B) static `m_0` repair + constraint; (C) dynamic breaker without constraint; (D) full FBAC. Released DFlash is immutable baseline. Hyperparameter-search count is identical; no arm receives a private rescue.
6. **Feasible-first checkpoint selection.** On checkpoint only, discard a checkpoint unless block-harm rate `<=5%`, mean harm `<=0.10` draft tokens/block, and the one-sided 95% paired lower bound for first-draft-token accuracy is `>=-0.005` versus DFlash. Among feasible checkpoints maximize prompt-balanced EAL, then choose the earliest step on exact ties. If no feasible checkpoint exists, that arm is frozen as no-selection and still disclosed.
7. **One common falsifier opening.** Hash-freeze all selected checkpoints and code first. In a single read, materialize raw outcomes for released, A, B, C and D. No later arm, seed, threshold or checkpoint may be added.

### Failure Modes and Diagnostics

- Old hard-reach starvation recurs: D-PACE suffix gradient parity/nonzero test fails; stop immediately—no `lambda` rescue.
- Constraint stalls learning: line-search alpha median `<1/4`, skipped steps `>1%`, or D-PACE loss cannot fall on capacity; close projected route rather than weakening protection post hoc.
- Minibatch-safe but held-out harmful: checkpoint/falsifier harm or first-token gate fails; constraint does not generalize, so safety claim fails.
- Gain is ordinary LoRA finetuning: full FBAC does not beat matched D-PACE by the frozen margin/CI; boundary-mechanism claim fails.
- Dynamic frontier is unnecessary: static constrained arm matches FBAC; report constrained adaptation only, not dynamic active-set novelty.
- Constraint is unnecessary: no-constraint arm matches harm and EAL; delete protection claim.
- Domain concentration: any of math/code/chat has negative paired point estimate versus DFlash; no universal claim and scale is blocked.
- Merge/runtime mismatch: extra operator/kernel/forward, accepted-length mismatch, or latency outside tolerance; systems claim fails even if EAL improves.

### Novelty and Elegance Argument

D-PACE already supplies smooth acceptance-aware weights, and the project already showed that hard reachable-prefix censoring loses capacity. FBAC therefore neither renames D-PACE nor retries the censored objective. Its single mechanism is an **active-set constrained parameter update**: full D-PACE continues to learn every position; the actual current breaker identifies the one extra repair constraint; the frozen accepted prefix defines a projection feasible set whose satisfaction has an exact per-block no-shortening implication. LoRA merely makes that update mergeable. Relative to DFlare it changes no fusion architecture；relative to Domino/DeLS/DSpark it introduces no causal/runtime expert；relative to GCLS/PROS it adapts the drafter rather than choosing over frozen outputs. If the static/no-constraint factorial controls do not separate these roles, the paper-level mechanism claim is rejected rather than expanded。

## Claim-Driven Validation Sketch

### Claim 1: FBAC improves released DFlash while retaining the deployment graph and bounded empirical harm

- Minimal experiment: 1,500-prompt (or power-increased) prospective one-shot falsifier with released DFlash and all four frozen training arms.
- Baselines / ablations: released；matched D-PACE LoRA；static constrained；dynamic no-constraint；full FBAC。
- Primary metric: prompt-balanced paired `Delta EAL` (`EAL=1+accepted_draft_tokens`) and 95% prompt-cluster bootstrap CI。
- Safety/system metrics: block harm rate and mean harm magnitude；first-draft-token paired accuracy；math/code/chat estimates；output exactness；operator graph；kernel path；forward count；peak memory；paired latency。
- Pass evidence: FBAC-vs-DFlash `Delta EAL>=+0.30` and two-sided 95% CI lower bound `>0`; harm `<=5%`; mean harm `<=0.10`; first-token one-sided 95% lower bound `>=-0.005`; all three domain point estimates `>=0`; merged median latency within `±2%` under 200 warmups + 1,000 paired timed iterations on fixed exclusive A800 clocks。

### Claim 2: 改进来自 dynamic active breaker + prefix constraint，而不只是 LoRA/D-PACE 容量

- Minimal experiment: 同一次 falsifier 中的 A/B/C/D factorial contrast；所有 arm 同 rank、层位、数据、steps、checkpoint budget。
- Metric: FBAC-vs-matched-D-PACE paired `Delta EAL>=+0.10` 且 95% CI lower `>0`；FBAC-vs-static constrained 的 paired EAL CI lower `>0`；FBAC harm 相对 dynamic-no-constraint 的 paired difference CI upper `<0`，且 EAL 不劣（one-sided lower `>=-0.05`）。
- Mechanism diagnostics: current breaker 向后移动分布、frozen-prefix constraint satisfaction、projection frequency/norm、restoration/skip、per-block proposition witness。
- Expected evidence: dynamic 对 static 提供额外修复，constraint 对 no-constraint 显著降 harm，matched D-PACE 不能解释全部增益。任一必要 contrast 不成立，就删除相应机制 claim。

### Descriptive Domino gap accounting

报告 `FBAC Delta EAL / (5.93853-5.11200)` 作为 DFlash-to-Domino-parallel gap 的描述性回收比例；明确训练量和模型来源不同，不做架构因果检验，也不把 Domino full GRU 当作本 pilot 的 pass gate。

## Experiment Handoff Inputs

- Must-prove claims: C1 same-graph、安全门内的 prospective EAL 提升；C2 dynamic active-set 与 constraint 相对 matched controls 的独立作用。
- Must-run controls: official-D-PACE parity；released；matched D-PACE LoRA；static constrained；dynamic no-constraint；full FBAC；zero-adapter/merge audit。
- Critical data/metrics: 8k/1k/1.5k component-isolated manifest；raw 15-position outcomes；paired prompt bootstrap；harm/first token/domain；constraint traces；graph/kernel/forward/latency。
- Highest-risk assumptions: two-layer LoRA capacity；projection不会使 AdamW 停滞；constraint 能跨 minibatch 泛化；near-duplicate clustering 可重放；1,500 prompt 对 `+0.10` mechanism contrast 有足够 power。

## Compute & Timeline Estimate

- Estimated GPU-hours: manifest/CPU audit `<4 CPU-h`；capacity `3–6 A800 GPU-h`；10.5k target collection `11–24 A800 GPU-h`；四个 equal-budget arms（含 frozen branch 与 line search）`30–70 A800 GPU-h`；共同 falsifier/profiling `6–12 A800 GPU-h`。总计约 `50–112 A800 GPU-h`，未含通过后 scale。
- Data / annotation cost: 无人工标注；只用 frozen Qwen3-4B T=0 continuation。所有 split、component、source/code/checkpoint identities 在生成前冻结。
- Timeline: 实现与 CPU tests 1–2 天；capacity 半天；collection/training 2–3 天；共同 falsifier/审计 1 天。预计 4–6 天；任一 gate 失败即关闭，不做观察后救援。

