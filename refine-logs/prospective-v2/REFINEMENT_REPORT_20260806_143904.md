# Refinement Report

**Problem**：提高 released DFlash 的真实未见 prompt 接受前缀，同时保持单链、一次 target verification、输出 exactness 与原 runtime topology。  
**Initial Approach**：分析 docs、代码与现有结果后自由重设计 DFlash/Domino gap 的解决方案。  
**Date**：2026-08-06  
**Rounds**：5 / 5  
**Final Score**：8.77 / 10  
**Final Verdict**：REVISE  
**Drift**：NONE  
**Handoff**：proceed_to_experiment_plan = yes，先冻结五项 pre-experiment clarification。

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Output Files

- Review summary: refine-logs/prospective-v2/REVIEW_SUMMARY.md
- Final proposal: refine-logs/prospective-v2/FINAL_PROPOSAL.md
- Score history: refine-logs/prospective-v2/score-history.md
- Raw reviews: refine-logs/prospective-v2/round-1-review.md through round-5-review.md

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 6 | 5 | 8 | 7 | 6 | 5 | 6.40 | REVISE |
| 2 | 9 | 6 | 6 | 8 | 6 | 7 | 6 | 6.80 | REVISE |
| 3 | 9 | 8 | 6 | 8 | 6 | 8 | 6 | 7.35 | REVISE |
| 4 | 9 | 9 | 7 | 8 | 5 | 8 | 7 | 7.80 | REVISE |
| 5 | 9.8 | 9.2 | 8.2 | 8.5 | 8.3 | 8.7 | 8.1 | 8.77 | REVISE |

Canonical round-5 weighting is 15/25/25/15/10/5/5；weighted arithmetic is 1.470+2.300+2.050+1.275+0.830+0.435+0.405=8.765，reported as 8.77。

## Round-by-Round Review Record

| Round | Main Concern | Change | Result |
|---:|---|---|---|
| 1 | method looked like soft weighting plus LoRA | hard frozen-prefix feasible set；uncensored suffix；factorial controls | resolved |
| 2 | multi-constraint optimizer undefined | cyclic projection、restoration、transactional Adam state | resolved |
| 3 | complete constraints computationally heavy | exact checks、failure contract、cost gate | partial |
| 4 | 60-row VJP likely violates cost gate | exact blockwise max，K≤4 batched rows | resolved pending benchmark |
| 5 | estimand and benchmark details ambiguous | prompt-balanced estimand、cluster bootstrap、detached masks、honest A baseline、fixed p95 ratio | resolved in final proposal |

## Final Proposal Snapshot

- 对 frozen accepted prefix 强制 exact positive-margin feasibility；
- 对 adapted current first mismatch 使用 dynamic hinge；
- 保留 full D-PACE 覆盖全部 15 positions；
- 用每 block 一个 exact maximum constraint 将 VJP rows 降至 K≤4；
- 以 A/B/C/D sealed factorial 和 one-opening falsifier判断机制是否成立；
- merge LoRA 后删除 wrappers，验证 runtime graph 与 latency equivalence。

## Method Evolution Highlights

1. 删除 frozen selector 与 soft safety weighting，改为 drafter 内部 representation adaptation。
2. 把可疑的逐位置投影变成严格等价、可运行的 blockwise-max transactional optimizer。
3. 拒绝通过 RNN、fusion、tree 或后验阈值增加表面 novelty；让预注册因果 controls 决定 paper contribution。

## Pushback / Drift Log

| Reviewer/Local Evidence | Tempting Change | Author Response | Outcome |
|---|---|---|---|
| reachable-support censoring capacity 低于 full D-PACE | 只训练 reachable suffix | 拒绝；full D-PACE 保留全部 15 positions | accepted |
| Domino/DFlare/DeLS/DSpark 提供 sequential/fusion primitives | 添加 runtime head 或 fusion | 拒绝；违反单链与原图 anchor | accepted |
| 60-row exact Jacobian 过慢 | sample/top-K constraints | 拒绝 approximation；采用 exact max equivalence | accepted |
| novelty 仍 borderline | 继续加模块 | 拒绝 contribution sprawl；以 A/B/C/D 结果判定 | accepted |
| component-aware leakage control | component-balanced metric | 拒绝改变 primary estimand；component 只作 cluster bootstrap unit | accepted |

## Remaining Weaknesses

- 8.77 未达到形式 READY threshold；顶会 novelty 仍为 empirical risk。
- at-most-four batched VJP 的真实 wall time 与 activation memory 尚未经过独占 A800 gate。
- batch-local feasibility 不构成 unseen theorem；只能支持预注册的 empirical harm claim。
- unit dynamic-loss scale 可能梯度失衡；只允许按预注册 gate 关闭路线，不允许事后调权。

## Raw Reviewer Responses

完整原文已经逐轮、不可变地保存在以下 artifacts；每个文件都包含 full raw response，而不是摘要：

1. [Round 1](round-1-review.md)
2. [Round 2](round-2-review.md)
3. [Round 3](round-3-review.md)
4. [Round 4](round-4-review.md)
5. [Round 5](round-5-review.md)

## Pre-Experiment Clarifications Frozen in Final Proposal

1. prompt-balanced point estimand；component 仅作为不可拆分 bootstrap cluster；
2. non-gold winner 与 exact-tie masks detach；
3. A benchmark 不执行人为 reference work，D 计入全部真实开销；
4. p95 ratio 固定为 pair 内 Q95(TD)/Q95(TA)，三个 pairs 均须 pass；
5. equivalence、ties、active switches、restoration、skip 与 counters 必须有 unit tests。

## Next Steps

虽然 formal verdict 是 REVISE，terminal reviewer 明确判断无 fatal flaw且允许进入 experiment-plan。下一阶段只做 claim-driven、gate-first roadmap：先 CPU unit/parity 与独立 code review，再做一次 authorized A800 cost gate；gate 失败则关闭，不进入科学训练。
