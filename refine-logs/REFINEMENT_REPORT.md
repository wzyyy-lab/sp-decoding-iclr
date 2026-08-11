# PARC-16 Refinement Report

## Outcome

PARC-16经过五轮同一reviewer的ARIS research-refine，最终得到`READY — 9.0/10`且无blocking issue。该判定为same-family provisional；它授权进入实验规划与实现，不代表实验结果已经成功。

## 冻结方法

PARC-16一次消费DFlash完整16位置和每位置pure-base Top16候选，把`16×16=256`个edit-action nodes放入两层D256/H8、无causal mask的全局self-attention。一次调用产生`[B,16,16]`，一次逐位置argmax产生唯一`[B,16]`序列。新增参数2,438,400，占537.427M DFlash约0.454%。

训练目标相对immutable released DFlash reference定义：只奖励reference已接受前缀之后的条件增量accepted length，并用reference-margin-normalized blockwise envelope约束任何会破坏reference正确前缀的deterministic edit。主recipe联合训练PARC与DFlash；target/reference/TopK IDs均stop-gradient。

## 最终执行协议

1. 在任何新full16 label生成前按prompt/domain固定90K train、5K validation与remainder held-out。
2. 只允许本地unit/shape/gradient safeguards；不得提交GPU smoke或capacity训练。
3. 唯一科学训练：batch8 blocks、180K steps、head LR 3e-4、DFlash LR 1e-5、warmup2K、cosine到10%、AdamW、clip1。
4. 所有loss阈值、numeric certificate、launch gate、dual与stop只来自train。
5. validation每10K steps只按`harm<=1%`后最大EAL选checkpoint，tie取最早。
6. weights/config锁定后首次打开held-out；同一job共同运行DFlash、released Domino和PARC的fixed/dynamic evaluator。
7. fixed与dynamic EAL均须`>=1.15x Domino`，三个域均不退化且harm`<=1%`。
8. held-out打开后禁止任何训练、扩数据、refresh或模型修改；未过门立即关闭路线。
9. EAL过门后才做complete eager profile与same-stack A40 SGLang，TPS ratio paired 95% CI lower必须`>=1.15`。

## 审查材料

- `PROBLEM_ANCHOR.md`
- `round-1-review.md`至`round-5-review.md`
- `round-4-refinement_20260810_210630.md`
- `.aris/traces/research-refine/2026-08-10_run06/`

