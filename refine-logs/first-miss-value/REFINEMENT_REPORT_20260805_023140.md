# Refinement Report

**Problem**：修复 DFlash/Direct 低收益以及 flat FMAS CE 的高 harm。  
**Initial approach**：对 one-edit action 学习 signed accepted-prefix advantage。  
**Date**：2026-08-05  
**Rounds**：2 / 5  
**Final score**：9.2 / 10  
**Final verdict**：READY

## Output Files

- Review summary：`refine-logs/first-miss-value/REVIEW_SUMMARY.md`
- Final proposal：`refine-logs/first-miss-value/FINAL_PROPOSAL.md`
- Score history：`refine-logs/first-miss-value/score-history.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.5 | 8.5 | 7.5 | 7.8 | 8.2 | 8.2 | 7.5 | 8.2 | REFINE |
| 2 | 9.7 | 9.4 | 9.0 | 8.8 | 9.2 | 9.3 | 8.8 | 9.2 | READY |

## Method Evolution Highlights

1. 把 flat 226-way classification 改为全部 225 edits 的精确 signed
   counterfactual prefix utility。
2. 把平均 regression fidelity 与部署决策安全分开，冻结 max-policy harm、
   false-edit、precision 和 regret 指标。
3. 保留 exact DFlash identity 与单一 unweighted MSE，拒绝 class weights、
   threshold sweep、D640 rescue 等会模糊因果解释的追加组件。

## Pushback / Drift Log

| Round | Reviewer concern | Response | Outcome |
|---:|---|---|---|
| 1 | natural frequency 可能先学成全 KEEP；RMSE 不控制 argmax | 不事后重加权；以明确 decision diagnostics 和联合 behavior gate 证伪 | accepted |
| 1 | capacity failure 可否说明 loss 失败 | 收窄为整个 D64+MSE+schedule+subset 配置失败 | corrected |
| 1 | payoff regression novelty 过宽 | 与 SpecDec++、Hybrid Verified Decoding、BASTION 划界 | corrected |

## Remaining Weaknesses

- positive edit 仅占 capacity actions 的 0.2222%，unweighted ERM 可能得到
  安全但无 repair 的解。
- 512-block memorization 不能证明 held-out features 可辨识。
- development selection set 只允许作 routing，不允许作 publishable effect
  estimate。

## Next Steps

1. 实现新的 SAVS module/trainer/tests，不改动冻结 Direct/FM​​AS 文件。
2. 完成全部 Gate-0 CPU tests 与 full-suite regression。
3. 经过 fresh experiment-bridge code review 后，最多提交一个 D64 capacity
   job；其结果再经 result-to-claim 决定是否允许 full-data seed-0。
