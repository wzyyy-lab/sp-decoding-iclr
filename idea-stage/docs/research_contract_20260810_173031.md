# Research Contract：PCLD-16R Parallel Global One-Chain Head

**Frozen on:** 2026-08-10 after research-refine Round 3 READY (`9.34/10`)  
**Source:** `refine-logs/parallel-global-head-v3/FINAL_PROPOSAL.md`  
**Plan:** `refine-logs/parallel-global-head-v3/EXPERIMENT_PLAN.md`  
**Scope:** Qwen3-4B / released DFlash-Domino-b16 / greedy temperature 0 / B16 / pure-base K16

## Problem and Primary Claims

### C1 — Full-block predictive residual mechanism

一个精确 `3,826,688` 参数的 full16 global noncausal head，仅用 ordinary DFlash online features，在一次并行唯一链输出中预测 clean target hidden residual；相对 parameter-matched local-mask 与 matched no-latent 两个 control，显著提高 disjoint prompt-balanced EAL，同时保持低 harm。

最低证据：P2 每个 global seed EAL `>=base+0.30`；aggregate oracle-gap recovery `>=20%`、strict J2 `>=40%`；每 seed harm `<=1%`；各域不低于 base；global-local 与 global-no-latent 各 `ΔEAL>=0.15` 且 prompt-paired bootstrap 95% CI lower `>0`。

### C2 — Acceptance and system outcome

冻结 checkpoint 在 sealed fixed 与 fresh dynamic 上的 EAL 均至少达到 same-job released Domino 的 `1.15x`，并在同栈 A40 SGLang batch1 workload 上取得 paired TPS ratio 95% CI lower `>=1.15`。

## Frozen Method

- input：完整 `H[B,16,2560]`、pure-base Top16 IDs/logits、`target.lm_head.weight` 的 candidate rows 与 5 个 base-score scalars；
- nodes：`16×16=256` candidate nodes；参数自由 RMS 后做 biased hidden/lexical projections、scalar projection、position/rank embeddings 与 affine LN；
- mixer：两层 `D256/H8/FFN1024/GELU/dropout0` pre-norm encoder，production mask 必须为 `None`；
- readout：16 learned queries 各自 cross-attend 全部 256 nodes；无 query FFN；
- residual：zero-init `U:256→2560`，`S=original_base_top16+<lm_head_row,Ug>`；
- output：一次 `S[B,16,16]`，一次逐位置 argmax 得到唯一 `[B,16]` chain；
- offline teacher：target rows `T` 来自 `context+anchor+gold[0:15]`；直接 target LM-head gather 是 score authority；`T-H` 只作离线 label；
- support：gold-in-K、authoritative target full-vocab top1==gold、BF16/FP32 stable 的连续前缀；所有 loss 共用同一 mask；
- objective：`L_safe + alpha(t)L_latent + 0.1 L_KL(T=2)`，`alpha 1.0→0.1` during first 30%，随后 0.1。

## Immutable Anti-Claims / Prohibitions

禁止 causal/autoregressive selected-token feedback、GRU、serial target seed/decode、Jacobi/iteration、beam/tree/trie/forest/multipath、Top16 path axis、额外 online target feature/forward、same-set/oracle 替代 held-out EAL。任何生产 `forward` 接收 gold/target hidden/teacher score 都是合同违规。

## Data and Selection Contract

- P1 capacity 仅作 same-set mechanics/capacity receipt，不支持泛化 claim；
- P2 固定 R047 `1589 fit / 199 select / 199 untouched diagnostic` prompt split；checkpoint 仅看 select，三臂九个 checkpoint 冻结后共同一次打开 diagnostic；
- P3 25K/100K 与 sealed fixed/dynamic 必须排除前述 prompts、IDs 与 normalized-text near duplicates；
- deployment seed/checkpoint 在 final outcome 前按 internal-select 冻结；
- ground truth 是 canonical dataset tokens 与 ordinary verifier acceptance，不是另一 student 模型输出。

## Stop Rules

`P0 mechanics/profile -> P1 capacity -> P2 disjoint controls -> P3 25K/100K -> sealed fixed/dynamic -> joint cycle -> SGLang`。前一阶段全部硬门通过才授权后一阶段。P1 任一 `agreement>=99% / recovery>=95% / harm<=1% / J2>=99%` 失败即关闭 frozen PCLD-16R，不允许 D512、loss/temperature/schedule sweep、serial/iteration/tree 救援。

## Joint System Gate

必须满足 `T_P/T_D <= (EAL_P+1)/(1.15*(EAL_D+1))`。形式 EAL floor 为 `8.3254859086`，进入 SGLang 前设计目标为 `>=9.0`；最终 fixed、dynamic 与 paired TPS 三门都不可互相替代。

## Claim Boundary

Round 3 READY 只授权实现和逐级证伪。teacher ceiling `10.5971817298` 不是 student result；在 P2 matched controls、sealed acceptance 和 SGLang system evidence 实际通过并经 result-to-claim 审查前，C1/C2 均不得写成已成立。
