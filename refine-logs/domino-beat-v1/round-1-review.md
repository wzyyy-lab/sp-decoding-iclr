# Round 1 Research Review (fresh-agent, raw)

**VERDICT: REVISE — SCORE: 8.7/10**

上一轮问题大部分已实质关闭：

- Stage 1 的 `m_base / m_clean / m_train` 正确描述了 head-only reachability。
- teacher-forced state 只使用到首个 mismatch；因此这些状态与真实
  on-policy state 在 breaker 之前完全一致，没有 exposure/state 错位。
- reachable-breaker CE、prefix preservation、L2-SP、冻结 vocab projection
  和 `<+0.10` 快速升级均直接服务 same-anchor EAL。
- replay 已降为辅助，且使用实际 draft-prefix target conditional。

## Remaining blocker

Stage 2 解冻 backbone 后不能继续原样使用 Stage 1 的
`m_base=0 => full-block loss=0`。否则首 token 错误的 block 仍无梯度，
“让 fixed first token 移动”实际上不会发生。

最短必要修改是明确 joint loss：

`L_joint = L_base,0(y0) + sum_{j>=1} 1[pred_<j = y_<j] L_final,j
           + lambda_anchor L_released-base`.

- position 0 始终有 base CE/margin，允许 backbone 修复首 token；
- `j>=1` 继续使用 detached clean-prefix reachability；
- 用 released-base anchoring 防止为修首 token而破坏原 backbone。

D-PACE 中“position 0 constant factor”最好再写成明确公式，但它只是
对照项，不阻塞主 Stage 1。

## Most valuable enhancement

把 breaker CE 换成或加入 `gold-vs-best-competitor` margin loss，例如
`softplus((max_other_logit - gold_logit)/T)`。主指标是 greedy argmax
accepted length，直接推动 breaker 越过当前决策边界通常比单纯降低
全词表 CE 更对齐，且几乎零额外成本。

另一个非阻塞修正：`9.7267` 是 DFlash K16 oracle，不应称作 Domino
“same-candidate oracle”。
