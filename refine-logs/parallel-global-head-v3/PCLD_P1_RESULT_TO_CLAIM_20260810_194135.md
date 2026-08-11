# PCLD-16R P1 result-to-claim

## 结论

`claim_supported = no`，置信度高；同模型家族审查，结论为 provisional。
冻结的 PCLD-16R 在 PCLD006 结束，不能进入 disjoint P2。

## 证据判断

step 6000 的 EAL 为 `9.525390625`，teacher candidate agreement 为
`99.9875746%`，说明该并行头能拟合稳定监督子集，并且同集合平均接受长度很高。
但三个绑定门失败：oracle-gap recovery `66.9209% < 95%`，harm
`6.25% > 1%`，legacy strict J2 `322/411 = 78.3455% < 99%`。

稳定 J2 的 `314/314` 只覆盖经过权威 replay、数值一致性和 margin 过滤的
loss-aligned 子集。receipt 中 legacy/stable J2 交集为 313，legacy-only 为
98，stable-only 为 1；因此稳定子集 100% 不能替代完整 411 个早期错误位置的
绑定评估。

## 可支持的窄结论

PCLD-16R 证明了三个事实：full16 全局非因果单链并行头具备稳定子集拟合能力；
它在同集合上显著提高平均 EAL；它的 A40 完整 eager head 路径开销低。它没有
证明完整 same-set capacity、安全前缀控制、held-out 泛化、fixed/dynamic
优势或 SGLang 吞吐优势。

## 路由

冻结方法关闭，不做训练步数、学习率、宽度、loss 权重、温度或阈值补救。
后续只能作为一个单独命名、重新预注册的新方法继续，并继续严格满足 full16、
全局非因果、一次调用、同时输出唯一一条 `[B,16]` 链、每位置 Top16，以及
禁止串行 target decode、自回归反馈、迭代和候选树的全部约束。

Primary evidence:
`artifacts/models/pcld16_capacity_10168532/report.json`。
Reviewer trace:
`.aris/traces/result-to-claim/2026-08-10_run05/`。

