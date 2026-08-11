# GFPR Transfer Failure Research Review

`review_independence: same-family`  
`acceptance_status: provisional`

## Verdict

冻结 DFlash/Domino 表征上的 selector 家族应判定 **NO-GO**。Top-16 oracle 与同集拟合只证明正确 token 可用、模型有记忆容量，并不证明 held-out 上存在可预测的在线选择信号。主路线必须转为 target-distilled DFlash representation adaptation；任何 `+0.1/+0.3` 都只能作为诊断，不能作为完成。

硬门槛保持不变：fixed EAL 至少 `8.325`（Domino `7.239552964` 的 +15%），dynamic EAL 至少 `7.59324`（Domino `6.602820376` 的 +15%），之后还须在 SGLang 端到端吞吐上显著领先。

## Evidence

| Route | Capacity / teacher signal | Held-out best | Decision |
|---|---:|---:|---|
| 61K exact-union residual | same-set `+3.6167` | `7.24891`, `+0.00935` | fail |
| target KL, T=1 | target top1 agreement `99.65%` | `7.27697`, `+0.03741` | fail |
| target KL, T=2 | first-rejection gold available `85.71%` | `7.29397`, `+0.05442` | fail |
| raw advantage Huber | dense target margins | `5.8053`, `-1.434` | numerically invalid |
| target-boundary rank64, 1.065M | deployable 12,800-d prefix feature | `7.31305`, `+0.07349`, CI crosses 0 | fail |
| target-boundary rank128, 2.130M | larger residual | `7.29519`, `+0.05564`, CI crosses 0 | fail |

Exact fallback removed the Top-16 restriction confound but did not restore transfer. Increasing selector capacity or adding the verified prefix boundary also stayed two orders of magnitude below the required `+1.08545` fixed gain. Therefore another selector/LR/loss/mixer grid is not justified.

## Root cause

1. The frozen draft representation lacks enough conditional information to predict the target-specific residual reliably on unseen prompts.
2. Frontier repair has asymmetric utility: one false override can destroy a long accepted prefix, while one repair helps only if later causal decisions remain correct.
3. `stored_frontier` is exact only on the released accepted prefix and original first rejection. After a repair moves the frontier, stored suffix actions come from a stale wrong prefix.
4. Target soft logits improve supervision but do not add online information; their small positive signal is insufficient by itself.
5. Raw target and Domino logit scales are not interchangeable. Unnormalized Huber directly caused the observed collapse.

## Reviewed lightweight boundary design

The minimal valid module conditions the candidate-specific residual on the already-available target prefix state:

```text
u = per-layer-RMS(target_boundary[5,2560]) -> [12800]
x_i = concat(parallel_hidden_i, causal_GRU_state_i)
z_i = W_local(x_i) + W_boundary(u)
r_i = W_up(SiLU(z_i))
delta_ik = dot(r_i, frozen_Domino_basis[token_ik])
```

At rank64 this has 1,064,960 trainable parameters, preserves exact Domino at zero-initialized `W_up`, and adds only one cached boundary projection per block. No global future mixer is needed because DFlash `parallel_hidden` already contains the block lattice. The measured rank64/rank128 failures close this frozen-backbone branch.

## Required target-teacher semantics

- Store centered FP32 target/Domino candidate advantages, not absolute FP16 logits followed by subtraction.
- Mask rows where a fresh target replay top1 differs from canonical gold unless a verifier-shaped tie envelope proves equivalence.
- Train only the currently accepted prefix and first rejection; never train on stale stored suffixes.
- Rebuild DFlash Top-K, current Domino fallback and `parallel_hidden` after every LoRA update.
- Keep future target logits as training labels only. Online input may use only target prefix states already produced for DFlash.

## Next authorized gate

Train only the existing rank16 DFlash LoRA on layers 3/4 (`1,835,008` parameters), with a live dynamic Top-16 union and target KL at `T=2`. Do not jointly train GRU/head, run a hyperparameter grid, add a global mixer, or refresh on-policy rollouts before the fixed gate passes.

Stop after OPB6K if fixed EAL is below `7.8` or harm exceeds gain (`lost/gained > 0.5`). Only a result at or above `7.8` authorizes larger data and adapted-policy recollection; final success still requires fixed `8.325`, dynamic `7.59324`, and the SGLang throughput gate.

If target-distilled LoRA also remains below `7.8`, the evidence says the +15% target is not attainable with the current 537M draft and zero additional target forward. The next scientific change would need a larger draft or partial target computation, not another lightweight selector patch.
