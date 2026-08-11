# ARIS Research Refinement Report：PCLD-16R

## Outcome

- final method：PCLD-16R Predictive Clean-Latent Distillation Head
- final verdict：`READY`
- final score：`9.34/10`
- rounds：3
- review independence：same-family provisional
- authorization：仅进入 gated P0；没有 student EAL 或系统正结果 claim

## Convergence

| Round | Score | Verdict | Decisive change |
|---:|---:|---|---|
| 1 | 6.15 | REFINE | 删除 Domino correction-code basis；修复 zero identity、latent magnitude 与 unbounded risk |
| 2 | 8.25 | REFINE | 冻结 LM-head/numerical authority、连续 support、stable losses、精确架构与联合系统门 |
| 3 | 9.34 | READY | 八类 blocker 全部关闭；无 drift；方法足够具体，可进入 P0→P1→P2 证伪 |

## Material influence

ARIS 审查把初稿从“借 Domino code 预测 target hidden”的含混设计，收缩为一个严格合规的 3,826,688 参数 PCLD-16R：只用 ordinary full16 DFlash 在线特征，256 candidate nodes 做完整非因果通信，16 queries 一次输出唯一 chain。审查还纠正了 LM-head 权威、BF16/FP32 near-tie support、zero-init 梯度动力学、prompt-balanced reducer，以及 EAL 与 cycle cost 必须联合判定的问题。

## Strongest unresolved risk

teacher ceiling `10.5971817298` 只证明 clean target candidate score 有空间，不证明在线 DFlash lattice 能在 disjoint prompts 上预测 `T-H`。P1 capacity 和 P2 global/local/no-latent 是必要 falsifiers；任一硬门失败都应停止，而不是用更宽 head、调 loss、加数据、serial decode 或 tree 路线救援。

## Next handoff

`experiment-plan → experiment-bridge`：先实现 P0 mechanics/sidecar/fair eager profile，再做 512-block P1；独立代码审查通过前不提交 GPU，P1 通过前不启动 P2。
