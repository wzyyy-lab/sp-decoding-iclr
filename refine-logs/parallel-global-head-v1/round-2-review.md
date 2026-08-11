# Round 2 独立方法复审

- `review_independence`: `same-family`
- `acceptance_status`: `provisional`
- `CALIBRATION`: `none`
- 综合评分：**8.8 / 10**
- 结论：**REFINE**

这版已经完成了实质性收敛：三个 Round 1 核心 blocker 在方法层面均已关闭，PGCF-16 与问题锚点和硬约束高度一致。当前不需要重新设计架构，但仍有三处会导致实现或 gate 无法唯一执行的规格问题，因此尚不能给 READY。

## 1. Anchor alignment

| 硬约束 | 结论 | 审查意见 |
|---|---|---|
| 完整 16-position 输入 | 满足 | OPB 重采明确生成 `H[:,16]`、`[16,16]` 候选及第 16 位标签 |
| 所有位置 global noncausal | 满足 | 256 个 candidate nodes 使用无 mask full self-attention；第一层中任一节点可直接访问其余 255 节点 |
| 一次 forward 输出 `[B,16,16]` | 满足 | 两个固定连续层属于单个前向图 |
| 仅输出一条 `[B,16]` chain | 满足 | 每位置一次 argmax，无多路径结构 |
| 无 selected-token feedback | 满足 | 层间仅传连续隐藏状态，没有 token/action 回灌 |
| 无额外 target 推理 | 满足 | target logits、policy action 只出现在离线标签中 |
| 无串行或循环修正 | 满足 | 两层固定 attention 不是推理时迭代 |
| 轻量约束 | 参数和 MAC 上满足，时延待证 | 约 2.44M 参数、约 0.365G MAC 很有希望，但仍须公平 profile |

删除 4-mode 后的 full-node 路径关闭了原先的结构性信息瓶颈。每个 output candidate 通过完整候选晶格直接交换信息。

## 2. Round 1 blocker 关闭情况

1. **oracle、门槛和回收比例错误：已关闭。** 主 oracle `10.909256560`，门槛 `8.325485909`，所需 gap recovery `46.6245%` 均正确。
2. **Stage A L15→L16 错位：方案中已关闭。** R047 full16 capacity后按旧 OPB prompt身份重采 full16；第16位明确进入 candidate、labels、loss与EAL。
3. **4-mode压缩与模块堆叠：已关闭。** 256-node full attention保留直接全局通信，loss已收缩为prefix/KL/短warm-start。

## 3. 新 blocker 与最小修复

### Blocker A：identity/visibility测试与 out-of-K loss

- zero-init `W_out` 时 final scores必然不响应 hidden变化。必须拆为 identity test与 deterministic nonzero probe / `X²` remote sensitivity test。
- `gold ∉ Top16` 不能非法 gather 后再乘零；须先用 safe rank 0 gather，再 `where(support, value, 0)`，FP32 cumsum/cumprod。
- KL valid set为空时batch项定义为零。
- teacher-only前10%明确 `lambda_prefix=lambda_KL=0`。

### Blocker B：精确参数bias规格

- 文档应逐项固定 Q/K/V/O、FFN、scalar、`W_mul`、scorer bias。
- `sum(p.numel())` 为Gate0硬断言，文档数字服从实现。
- profile同时报告完整pipeline、base-Top16后的增量head、p50/p90、显存与buffer。

### Blocker C：自然语言gate与formal protocol

- Gate2量化 global-local、bootstrap与shuffle消融比例。
- Gate3 teacher ceiling“接近”量化。
- Gate4/5冻结domain tolerance、seed、checkpoint、formal一次性规则。
- formal prompts曾用于manifest排除审计；准确说法只能是其full16 labels/outcomes尚未读取。
- 不同ID namespace下需做normalized prompt exact/near-duplicate audit，而非只看sample ID。

## 4. 建议冻结数字

- global-local `Delta EAL >=0.15` 且 paired prompt-bootstrap 95% CI下界 `>0`；
- remote shuffle消除至少50% global-local增益，并联合打乱远端 `H/C/B`；
- teacher-only EAL `>=7.080272109`，距 exact ceiling不超过0.10；
- formal前冻结模型、seed、checkpoint、evaluator、dynamic生成与same-job Domino命令；formal失败后不能继续称同一600 prompts为fresh test。

## 5. Gate readiness

| Gate | 状态 |
|---|---|
| Gate0 | 需修safe gather、remote test与bias flags |
| Gate1 | 修safe gather后可落地 |
| Gate2 | 需冻结数值阈值与shuffle实现 |
| Gate3 | 主阈值清晰，teacher阈值需量化 |
| Gate4 | 需冻结domain/seed/checkpoint |
| Gate5 | 需完整一次性formal receipt |

最终结论：核心架构已经通过方法审查；完成三个最小规格修复后，可直接转 experiment-plan，并有条件达到 READY。
