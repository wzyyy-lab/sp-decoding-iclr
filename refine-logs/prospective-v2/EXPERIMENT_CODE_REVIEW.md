# FBPF G1 实验代码审查

**日期**：2026-08-08  
**流程**：ARIS experiment-bridge  
**结论**：`G1 GO`，仅开放一次 `sdpa` synthetic real-model GPU smoke。

## 已关闭的阻断问题

1. synthetic fixture 原先由 `torch.inference_mode()` 产生，LoRA backward 无法保存这些 tensor；已改为 `torch.no_grad()`。
2. cyclic projection 四轮预算耗尽后原先仍可能经小 alpha 提交；现有任何 residual 超过 `tau_linear` 都抛出 `ProjectionBudgetExhausted`，禁止静默 commit。
3. 增加 functional flat-LoRA 参数到 D-PACE loss 的独立 scalar/完整梯度 parity 测试。

## 验证结果

- FBPF CPU suite：31 passed。
- pinned D-PACE suite：9 passed。
- LoRA-only functional forward、float32 adapter branch、D-PACE reduction、four-row VJP、commit/skip/restoration/rollback 均无剩余 blocker。

## 非阻断约束

- 本次 smoke 必须使用并记录 `sdpa`。
- 后续 variable-context padding/attention mask 必须在 capacity wrapper 单独验证；本次等长 synthetic context 不覆盖它。

