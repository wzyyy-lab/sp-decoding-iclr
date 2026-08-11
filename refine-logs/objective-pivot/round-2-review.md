# Round 2 Review

<details open>
<summary>Full raw reviewer response</summary>

# Round-2 Senior Review

修订已经消除了上一轮的归因 blocker。`λ=1/0/0.1` 现在构成严格的 support-coefficient 对照，mask 定义完整，dual-mask evaluator 也解决了 endogenous accuracy。方案已达到实现并运行最小筛选的标准。

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 10/10 |
| Method Specificity | 10/10 |
| Contribution Quality | 8/10 |
| Frontier Leverage | 8/10 |
| Feasibility | 10/10 |
| Validation Focus | 10/10 |
| Venue Readiness | 8/10 |

**OVERALL SCORE: 9.10/10**

计算：`0.15×10 + 0.25×10 + 0.25×8 + 0.15×8 + 0.10×10 + 0.05×10 + 0.05×8 = 9.10`。

**CALIBRATION: none** — 没有人工筛选的 good/bad proposal anchors。

**GAP:** 剩余差距是证据而非方案设计。Hard reachable-prefix censoring 的 paper-level 意义仍取决于 material、跨种子、untouched-split 的 EAL 效应；不应添加模块、理论包装或 benchmark 来追求概念分数。

## 数学与实现审计

1. **`lambda=1` 数学上严格等价 historical `candidate_dpace`。** `m_auf` 与 `m_suffix` 不相交且完整划分 `m_cov`，因此相同 coverage、alpha=.5 detached weights、fp32 CE、`1/(B*L)` normalization。浮点顺序可能不 bitwise，使用 tolerance-tight value/all-score-gradient parity；固定 `base_safety_weight=0`。
2. **Masks 完整无矛盾。** `m_cov` 包含当前位置 gold-in-K；`reach` 只检查先前预测；in-K breaker 保留；out-of-K breaker 不进入 coverage。实现应断言 subset、disjoint、union invariants。
3. **Dual-mask evaluator 消除 endogenous metrics。** Candidate/hard accuracy、NLL、by-position classification 使用 coverage；training mask 只做诊断；EAL/repair/harm/gap 来自真实 greedy path。必须检查所有旧 `active_positions` 消费者。
4. **无代码实现前 blocker。** 验收：四类 parity 输入、lambda0 suffix score gradient zero、coverage empty 安全分母、aggregate `>=.05` 包含边界、alpha=.5/safety0/fp32/provenance 固定。

## Simplification Opportunities

NONE。一个 Boolean helper、一个 lambda、dual-mask diagnostics 已是足够小的实现。

## Modernization Opportunities

NONE。RL、straight-through、soft reach 或 teacher 会削弱归因。

## Drift Warning

NONE。没有复活 flat C1a/C1b，也没有漂移成 policy optimization。

## Remaining Action Items

- 实现 mask invariants、parity 和 zero-suffix-gradient tests。
- evaluator 所有 classification denominator 使用 fixed coverage。
- 先跑三 cell 128-block capacity matrix；任一失败即停止。
- 全过后按 `+0.05` gate 跑 OPB-25K；失败即关闭路线。

## Verdict

**READY** — 只表示提案可执行，不表示方法 claim 已被实验支持。

**Assurance:** same-family Codex positive review，provisional。

</details>
