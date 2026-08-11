# Review Summary

**Problem**：在 frozen DFlash lattice 上以固定深度、一次编辑提高真实 EAL。  
**Initial approach**：用 dense signed action utility 替代失败的 flat action CE。  
**Date**：2026-08-05  
**Rounds**：2 / 5  
**Final score**：9.2 / 10  
**Final verdict**：READY

## Problem Anchor

保持 released DFlash、target model、K16 candidate lattice 和一次并行 head
不变；只允许 KEEP 或修改一个 token。要解决的是 Direct 增益小和 FMAS CE
高 harm 的 objective/utility mismatch，不以更大 tree、顺序模型或更多
target compute 改题。

## Round-by-Round Resolution Log

| Round | Main concern | Resolution | Status | Remaining risk |
|---:|---|---|---|---|
| 1 | unweighted MSE 的稀有正例与 max-over-225 风险；Gate-1 失败归因过宽；novelty 边界不足 | 限定 population consistency；冻结四个 decision metrics；收窄失败结论；加入 closest-work 边界 | resolved | finite-model utility prediction 仍需实验 |
| 2 | 核对公式、分母、256-positive 算术、identity/gradient 语义 | 全部核对通过；补充 no-edit precision=NA、per-block RMSE 反例与 two-backward test | READY | 仅剩 capacity/generalization 的真实不确定性 |

## Overall Evolution

- 方法保持单一变化：canonical action CE → dense signed action-value MSE。
- 将“Bayes 一致”严格限定到 population conditional decision，不再外推到
  finite-model safety。
- 用 deployed harm、no-benefit false edit、edit precision 和 selected regret
  直接约束 max policy，而非依赖平均 RMSE。
- 不引入 class weighting、阈值调参或更大模型作为事后救援。
- 将潜在新意收窄到 frozen DFlash 的完整 one-edit counterfactual target 与
  exact-identity residual deployment。

## Final Status

- Anchor status：preserved
- Focus status：tight
- Modernity status：appropriately frontier-aware，且有意不引入无必要组件
- Strongest parts：精确 signed utility；epoch-0 DFlash identity；可证伪 gates
- Remaining weakness：极稀疏正效用在有限共享模型下可能仍学不到，必须由
  capacity gate 和随后 prompt-diverse gate 实证决定。
