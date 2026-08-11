# FINAL PROPOSAL：JAPD-16

Status：**READY for experiment-plan and gated implementation**（Round 3 score
`9.33/10`; no blocking issue; no performance claim yet）。

## Canonical sources

- 完整方法、公式、dataflow、训练与gates：
  [round-1-refinement.md](round-1-refinement.md)
- Round-2严格数学与protocol修订：
  [round-2-refinement.md](round-2-refinement.md)
- 最终独立复审与授权边界：
  [round-3-review.md](round-3-review.md)
- 用户不可变约束：
  [USER_CONSTRAINT_CONTRACT.md](../parallel-global-head-v1/USER_CONSTRAINT_CONTRACT.md)

上述文件共同构成实现合同；若摘要与完整方法冲突，以完整方法中经Round-2修订的
更保守定义为准。

## Frozen method in one paragraph

JAPD-16复用433,852参数的D64/H4/L1 axial direct selector：完整16个DFlash位置和
每位置pure-base Top16候选一次输入，每个candidate通过无causal mask的global mixer
读取全部16个position summaries，一次输出`[B,16,16]`并逐位置argmax成唯一
`[B,16]` proposal。offline objective为固定`Z=136`的all-prefix hard/soft
distillation，加仅在至少两处base错误时激活的保守joint certificate
`softplus(logsumexp(-margin_prefix))`。online没有GRU、selected-token feedback、
serial target seed、iteration、beam/tree/multipath或额外target inference。

## Frozen success sequence

`mechanics/capacity/latency -> matched small-data + fresh300 -> 25K hard gate ->
100K three seeds -> untouched final600 fixed/dynamic -> same-stack SGLang`。

每一箭头都要求前一阶段所有hard gates通过。最终要求fixed与dynamic EAL均至少
`1.15x` same-job Domino，每域不退化；SGLang A40 paired tokens/s ratio的95% CI
lower bound至少`1.15`。
