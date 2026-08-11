# Review Summary

**Problem**：在保持 released DFlash 单链、一次 target verification 与原推理拓扑的前提下，解决 frozen selector 的安全收益上限，提升未见 prompt 的接受前缀。  
**Initial Approach**：从 DFlash/Domino 差距与现有失败结果出发，允许重新设计训练机制，但禁止重开旧 R083。  
**Date**：2026-08-06  
**Rounds**：5 / 5  
**Final Score**：8.77 / 10  
**Final Verdict**：REVISE  
**Execution Handoff**：reviewer 明确允许进入 experiment-plan；必须先冻结 estimand、tie mask 与 throughput 口径。

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | Main Resolution | Solved? | Remaining Risk |
|---:|---|---|---|---|
| 1 | soft protection 只是另一种 weighting，机制身份不清 | 用 frozen accepted prefix 定义 hard active feasibility；保留 full D-PACE；冻结 A/B/C/D controls | yes | 多约束 optimizer 尚未闭合 |
| 2 | projection 对多个 constraints、infeasible/vacuous batch、Adam state 不完整 | 加入 cyclic projection、stateless restoration、双计数器与 transactional commit | yes | 约束梯度成本不明 |
| 3 | 60-row complete Jacobian 可能过慢；residual/counter 仍需精确 | 补全 exact nonlinear check、cost gate、failure semantics | partial | 60 reverse-mode rows 很可能超过 4× |
| 4 | 完整 60-row VJP 与 compute budget 冲突 | 用严格等价 Cn=max ci 聚合为每 block 一行，K≤4；clean-process benchmark | yes | 实际 kernel 性能仍须 gate |
| 5 | point estimand 与 component bootstrap 混淆；baseline/p95/tie detach 未冻结 | 最终提案固定 prompt-balanced point estimate、cluster bootstrap、detached masks、真实 A baseline、quantile ratio | yes | novelty 必须由结果证明 |

## Overall Evolution

- 问题从“再训练一个 frozen selector”收紧为“直接改变 draft representation，同时保护 frozen accepted prefix”。
- soft safety penalty 被删除，替换为 verifier-induced exact feasible set。
- 多约束成本从最多 60 行降为严格等价的最多 4 行，没有 sampling 或 constraint truncation。
- 部署只保留 merge 后的原 DFlash 图，不增加 runtime head、fusion 或 sequential path。
- 对 suffix censoring、旧 R083 rescue、post-hoc threshold 与私有 rescue 都设置了明确禁止边界。
- 四臂 factorial 让普通 D-PACE LoRA、static frontier、dynamic repair、prefix feasibility 分别可证伪。

## Final Status

- Anchor status：preserved；五轮 drift 均为 NONE。
- Focus status：tight；唯一主贡献是 verifier-specific first-break prefix-feasible adaptation。
- Modernity status：appropriately frontier-aware；使用 mergeable PEFT，但不堆叠额外 foundation-model 模块。
- Strongest parts：exact blockwise equivalence、transactional optimizer、sealed controls、one-opening prospective evaluation、deployment trace/latency audit。
- Remaining weakness：top-venue novelty 依赖 D 对 A/B/C 的预注册 effect-size separation；cost gate 尚未在真实 A800 上通过。
- Formal reason for REVISE：canonical weighted score 8.77 低于 READY threshold 9.0；不是 fatal flaw。
