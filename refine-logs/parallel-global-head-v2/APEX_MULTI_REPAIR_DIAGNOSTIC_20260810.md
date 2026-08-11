# APEX-16 多位置修复与梯度诊断

**Data:** `r047_anchor_t4_validation_select_10164718`，147 prompts / 1,175 blocks  
**Scope:** read-only development diagnosis；不是 learned-method result

## 多次修复是主目标的必要条件

在每个 block 的 pure-base Top-16 连续可达前缀内，统计 gold rank 非0的
位置数。1,175个 block 中946个至少需要一次 correction，771个至少需要两次。

| 所需 correction 数 | blocks |
|---:|---:|
| 0 | 229 |
| 1 | 175 |
| 2 | 168 |
| 3 | 138 |
| 4 | 93 |
| 5 | 88 |
| 6 | 73 |
| 7 | 51 |
| 8 | 35 |
| 9 | 33 |
| 10 | 25 |
| 11 | 22 |
| 12 | 20 |
| 13 | 13 |
| 14 | 8 |
| 15 | 2 |
| 16 | 2 |

平均需要 `3.56085` 次 correction 才能走完整个 Top-16 oracle prefix。

一个只允许完美修正前 `k` 次 base 错误、其余位置保持 base 的 oracle 为：

| perfect correction budget | prompt-balanced EAL |
|---:|---:|
| 0 | 6.0685131195 |
| 1 | 7.4985422741 |
| 2 | 8.4238338192 |
| 3 | 9.0838192420 |
| 4 | 9.5549076774 |
| 5 | 9.9204324587 |

因此只做首拒修复在上限上就低于 `8.3254859086`；而前两次 correction 的
oracle 已越过目标。这不证明 draft-side 信息能识别两个正确 token，但严格证明
训练目标不能只围绕一次 edit。

## v1 loss 对第二次修复的梯度饥饿

在771个至少需要两次 correction 的 block 上，以 base Top-16 conditional
probability 作为 step0 student，计算 v1 expected-prefix-product loss 对
`log q_i(gold)` 的 credit：

| quantity | result |
|---|---:|
| 第一次 correction 平均 credit | 0.011277665 |
| 第二次 correction 平均 credit | 0.001067145 |
| 第二/第一次平均比 | 0.0865694 |
| 第二/第一次中位比 | 0.0507524 |
| 第二次 credit `<1e-3` | 588 / 771 |
| 第二次 credit `<1e-4` | 292 / 771 |

同一 block 内归一化的 all-prefix log-risk 权重，第一次/第二次 correction 平均
为 `0.147104/0.097752`。它不会因为此前 token 概率低而把后续 correction 的
直接监督乘没。

## Claim boundary

这些数字支持“v2 需要 non-vanishing multi-position prefix supervision”。它们
不支持 APEX 一定能泛化，也不证明第二个 gold candidate 可由 full16 DFlash
features 识别。后者必须由 matched-global/local 和新 disjoint held-out 实验回答。
