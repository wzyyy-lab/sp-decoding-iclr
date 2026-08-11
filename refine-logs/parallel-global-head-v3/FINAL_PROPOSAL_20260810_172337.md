# FINAL PROPOSAL：PCLD-16R

> ARIS `research-refine`：`READY 9.34/10`  
> 用户硬约束：`../parallel-global-head-v1/USER_CONSTRAINT_CONTRACT.md`  
> 完整可执行规范：`round-2-refinement_20260810_170615.md`

## 问题与目标

在不改变 DFlash full16 一次并行生成方式的前提下，用轻量 head 解决 accepted length 低的问题。当前 development pure-base EAL 为 `6.0685131195`，released Domino 为 `7.2395529640`，pure-base Top16 oracle 为 `10.9092565598`。最终 fixed EAL、dynamic EAL 和同栈 A40 SGLang throughput 都必须至少达到 same-job released Domino 的 `1.15x`。

## 不可变在线合同

- 一次读取 ordinary DFlash 的完整 `H[B,16,2560]` 与 base logits；
- 每个输出通过无 causal mask 的全局 mixer 看到全部 16 位；
- Top16 仅是每位置 candidate axis；
- 一次 head invocation 产生唯一 `S[B,16,16]`；
- 一次 tensor argmax 同时得到唯一 `[B,16]` token chain；
- 禁止 GRU、causal/autoregressive selected-token feedback、serial target decode、Jacobi/iteration、beam/tree/trie/forest/multipath；
- target model 只提供离线监督，线上除普通最终 verifier 外不增加任何 target inference。

## 冻结架构

1. `C=Top16(base_logits.float())`，lexical authority 固定为 `target.lm_head.weight[C]`。
2. 每个 position-candidate 形成一个 node；node 融合参数自由 RMS-normalized DFlash hidden、LM-head candidate row、5 个有界 base-score scalar、position embedding 和 rank embedding。
3. 256 nodes 进入两层 `D256/H8/FFN1024/GELU/dropout0` pre-norm encoder，production attention mask 永远为 `None`。
4. 16 个 learned queries 各自对全部 256 nodes 做一次 cross-attention；无 query FFN。
5. `U:256→2560` 的 weight/bias 全零初始化，预测 target hidden residual：

   `score = original_base_top16_score + <lm_head_row, U(global_query)>`。

6. 精确可训练参数为 `3,826,688`；禁止增加 anchor projection、第二 head、gate、expert 或其它救援模块。

## 离线教师与损失

- 一次离线 teacher-forced target pass 使用 `context + anchor + gold[0:15]`，得到精确对应 16 个 draft token 的 target hidden rows `T`；线上不使用 `T`。
- latent label 为 `T-H`；teacher candidate score 由真实 `target.lm_head(T)` 后 gather 得到。
- 所有 loss 共用同一个连续 clean-prefix support：gold 在 base Top16、authoritative full-vocab target top1 等于 gold、BF16/FP32 稳定；首次失败后整个 suffix mask 掉。
- prompt 均衡：先均匀采 prompt，再在其有效 blocks 内均匀采样；评估先 block 后 prompt。
- 冻结 objective：`L_safe + alpha(t)L_latent + 0.1 L_KL(T=2)`；`alpha` 在前 30% updates 从 `1.0` 线性降到 `0.1`，其后保持 `0.1`。

## 分阶段证伪门

- **P0 mechanics/profile**：row0/row15、teacher geometry、zero identity、remote visibility、zero-init 梯度动力学、mask fail-closed、精确参数、完整 A40 eager path `<=1.20x` released eager Domino。
- **P1 capacity**：512-block same-set、8,000 updates；candidate agreement `>=99%`、oracle-gap recovery `>=95%`、harm `<=1%`、strict J2 `>=99%`。失败只允许修已证明的实现/数值错误，不允许 sweep 救援。
- **P2 mechanism**：固定 `1589/199/199` prompt split，global/local/no-latent 三臂、seeds `0/1/2`；每个 global seed EAL `>=base+0.30`，aggregate recovery `>=20%`、J2 `>=40%`、harm `<=1%`、三域不退化；global-local 与 global-no-latent 均 `ΔEAL>=0.15` 且 prompt-paired bootstrap 95% CI lower `>0`。
- **P3 scale**：仅 P2 全过后收集 25K；仅 held-out EAL `>=7.8`、正 slope、两个 control CI 仍正、无域退化后进入 100K；25K `<7.55` 立即停止。
- **P4 claims**：sealed fixed/dynamic EAL 各 `>=1.15x` Domino；同栈 A40 SGLang paired TPS ratio 95% CI lower `>=1.15`。

## 系统联合门

每轮 useful output 按 `EAL+1` 计。要达到 `1.15x` Domino throughput，必须满足：

`T_PCLD/T_Domino <= (EAL_PCLD+1)/(1.15*(EAL_Domino+1))`。

在 Domino EAL `7.2395529640` 下，等 cycle time 至少需要 PCLD EAL `8.4754859087`。因此 `8.3254859086` 是不可降低的形式 EAL 门，`9.0` 是进入 SGLang 前的设计目标；EAL=9.0 时完整 cycle 最多只能是 Domino 的 `1.0553548490x`。

## 允许的论文主张边界

只有 global/local、latent/no-latent、sealed fixed/dynamic 和最终同栈 TPS 全部过门后，才可主张：在严格禁止 sequential feedback 的 full16 drafter 中，candidate-conditioned full-block noncausal student 能一次蒸馏并输出单条 clean trajectory，并同时改善 accepted length 与端到端吞吐。当前 READY 只授权实现与证伪，不代表该经验主张已经成立。
