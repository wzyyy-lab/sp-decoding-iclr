# Round 1 Review

<details open>
<summary>Full raw reviewer response</summary>

# Round-1 Senior Review

结论先行：核心 pivot 值得继续，但不应按当前公式直接跑 OPB-25K。先修正“只改变 support”的归因、loss 尺度以及 capacity evaluator；这些都是一轮短实现可完成的修改。修正后值得立刻跑 unit test、128-block capacity probe，再决定是否进入最小开发筛选。

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9/10 |
| Method Specificity | 6/10 |
| Contribution Quality | 6/10 |
| Frontier Leverage | 7/10 |
| Feasibility | 8/10 |
| Validation Focus | 6/10 |
| Venue Readiness | 5/10 |

**OVERALL SCORE: 6.75/10**

计算：`0.15×9 + 0.25×6 + 0.25×6 + 0.15×7 + 0.10×8 + 0.05×6 + 0.05×5 = 6.75`。

**CALIBRATION: none**

项目与技能目录没有提供人工筛选的 3 个 good / 3 个 bad proposal anchors，因此只能按显式 rubric 评分。

**GAP:** 该提案距离 READY 的主要差距不在问题选择或工程成本，而在因果归因和 claim 强度。它准确抓住了“固定 gold coverage 对最长正确前缀不匹配”的 bottleneck，也提出了极小的实现改动；但当前公式同时改变 support、D-PACE continuation weighting 和总体梯度尺度，无法把胜负归因于 Head-AUF。Moving hard mask 也只是 detached、prediction-conditioned supervised censoring，不是可微 policy optimization。再加上 endogenous capacity metrics 和仅要求 `> control` 的 seed-0 gate，当前结果即使为正也不足以支持机制主张。

## 最关键的 blocker：当前不是“只改变 support”

提案声称只改变一个 detached support mask，但现有 Candidate-D-PACE 使用 detached continuation weights，并以 `B×L` 归一化；提议的 Head-AUF 则是 uniform CE、仅以 `B` 归一化。因此当前比较同时改变 active support、continuation weights 和 loss scale，是 CRITICAL attribution blocker。

建议统一为：`m_cov` 是 gold-in-K prefix，`m_auf` 是 prediction-conditioned reachable prefix，`m_suffix=m_cov & ~m_auf`，`w` 是按 `m_cov` 算出的原 Candidate-D-PACE detached weights：

`L_lambda = (1 / BL) sum w * (m_auf + lambda*m_suffix) * CE`。

预注册 `lambda=1` 精确还原 control，`lambda=0` pure reachable-prefix，`lambda=0.1` 保留 10% post-break coverage。

## 必答 blocker

1. **Off-by-one:** 当前定义正确保留 breaker。`reach_i` 只检查 `j<i`；若首错 gold-in-K，breaker active、其后 suffix inactive；若首错 gold-out-K，则 breaker 无合法标签且 inactive。必须测试 `[1,1,0,0]` 两种 witness，并测试 suffix logits 梯度严格为零。
2. **Loss scale:** 必须使用共同固定分母 `B×L`，且 auxiliary 仅作用于 `m_cov & ~m_auf`，不能重复加权 reachable positions。不要按 moving active count 分别归一化。
3. **Capacity gate:** 现有 evaluator 若用 moving active mask 计算 accuracy 会产生 endogenous denominator。candidate/hard accuracy 必须固定在 `m_cov`；repair/gap/harm 继续按真实 greedy path；五项原阈值可保留。
4. **Seed0 effect:** 不能只要求正 epsilon。预注册 winner-control raw prompt-balanced EAL `>= +0.05`，harm `<= control+0.01`，first-token `>= control-0.001`。三种子 full-data 确认才要求同号与 paired CI lower bound >0。
5. **路线独立性:** 它不是 C1a/C1b rescue，因为回到已支持 axial D64/L1 并只改 training support；但它是失败结果触发的 post-hoc hypothesis generation，必须如实标注并在任何新结果前冻结代码 hash、三 cell、门槛和停止规则。

## 低于 7 分维度的修复项

- **Method Specificity 6:** 使用统一 `L_lambda`；loss output 分开 training-support 与 evaluation-coverage mask。CRITICAL。
- **Contribution Quality 6:** 不得把 detached argmax mask 称作 policy optimization；改称 prediction-conditioned reachable-prefix supervision / hard support alignment。IMPORTANT。
- **Validation Focus 6:** 使用 `+0.05` 门；capacity classification 固定 `m_cov`；最终 claim 保留 untouched split。CRITICAL。
- **Venue Readiness 5:** 先证明 support-only 因果效应；若只带来微小开发增益，单独不足以成为主贡献。IMPORTANT。

## Simplification Opportunities

1. 用单一 `L_lambda` 统一三格。
2. 复用现有 coverage mask 与 Candidate-D-PACE weights，仅新增 reachable-mask helper 和双 mask diagnostics。
3. 20M–50M teacher 移出正文，只保留失败 routing sentence。

## Modernization Opportunities

NONE。不要加入 RL、soft policy gradient、distillation，也不要把 hard mask 包装成可微 policy optimization。

## Drift Warning

NONE。技术目标没有漂移；只有“policy”措辞可能造成概念漂移。

## Verdict

**REVISE**。修复统一 loss、固定-mask capacity metrics 和 `+0.05` gate 后，建议立即实现并运行最小筛选；原样实现则不建议开跑。

**Assurance:** same-family Codex review；所有正面判断均为 provisional。

</details>
