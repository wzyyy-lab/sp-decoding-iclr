# Round 2 Refinement：JAPD-16 Conservative Joint Certificate

## Problem Anchor（verbatim）

- **Bottom-line problem：** 在不改变 DFlash 一次并行生成整块这一核心优势的前提下，设计一个轻量 head，显著解决 accepted length 偏低的问题；固定与动态 EAL 都至少达到同作业 released Domino 的 `1.15x`，最终同栈 SGLang 端到端吞吐也至少达到 Domino 的 `1.15x`。
- **Must-solve bottleneck：** DFlash base Top-1 在当前 full16 disjoint development 上为 `6.0685131195`，Domino 为 `7.2395529640`，目标为 `8.3254859086`。PGCF-v1 虽能同集拟合 Top-16 oracle、且延迟足够低，但 held-out 仅为 `6.1027696793`；它把大量修改浪费在首拒之后，并只修复 `46/946` 个可修首拒。完美只修一次的 oracle 也仅为 `7.4985422741`，而完美修正前两次错误可达 `8.4238338192`。因此新方法必须利用完整16位全局信息，在一次并行输出中学会多位置、相互一致的 clean-prefix 修复，而不是只做首拒 gate 或单点修补。
- **Non-goals：** 不做 Domino/GRU 式自回归，不做 selected-token feedback，不做串行 target seed/decode，不做 Jacobi 或任何迭代 refinement，不做 beam/tree/trie/forest/multipath，不让 Top-16 变成路径维，不增加 ordinary verifier 之外的在线 target inference，也不在 accepted-length 主机制成立前投入 SGLang 小修小补。
- **Constraints：** 单次 head 必须同时消费完整 `[B,16,*]` DFlash online features；每个输出位置必须通过无 causal mask 的全局 mixer 看到全部16位；一次产生 `[B,16,16]` scores 并以一次逐位置 argmax 得到唯一 `[B,16]` 序列。Top-16 只作每位置候选轴。训练/选择/held-out prompt 必须隔离，target 信息只作离线标签。新增参数先控制在 `10.75M` 内，并以同 A40、同 BF16、同 batch/block、eager-to-eager 公平 profile 约束成本。
- **Success condition：** 一个未经 target-feature 泄漏、在新 disjoint held-out 上成立的 full16 global-vs-local 机制信号；固定 EAL 至少 `8.3254859086`、动态 EAL 至少 `1.15x` Domino，三个域不退化；随后同栈 SGLang A40 tokens/s 的 paired 95% CI 下界至少 `1.15x` Domino。任何架构不变量失败均为 hard NO-GO，不能用 oracle、same-set capacity 或 off-spec 系统结果替代。

## Anchor Check

本轮只修正 objective 数学、数据采样与 gate boolean；online head仍是full16 global
non-causal one-call one-chain。没有新增serial、causal、iterative、tree、multipath、
selected-token feedback或target online feature。

## Changes Made

1. **Conservative joint certificate：** normalized mean soft-min改为
   `M_joint=-logsumexp(-d_i)`，保证 `M_joint<=min_i d_i`；任何一个prefix token
   margin非正时certificate不可能为正。
2. **Two-frontier scope：** `L_J2`只在前缀内至少有两处base错误时激活；其它block
   只由all-prefix项训练。
3. **Inclusive hard metric：** `J2`固定检查 `0:e2+1`，第二处修复必须真的正确；
   vectorized结果必须与逐block reference loop一致。历史1175-block audit在严格
   clean horizon下得到denominator/global/local/Domino=`745/15/0/207`，废止旧的
   candidate-only `771/15/0/208` 口径。
4. **Utility-preserving normalizer：** AP改用固定full16 `Z=136`，不再把长、短
   horizon强行归一到相同总权重。
5. **Exact prompt objective：** 只对 `h>0` 的有效block定义prompt mean；训练用
   全有效block shuffle、`1/|B_p^+|` 权重和固定全局缩放，不使用随机batch权重分母。
6. **Sidecar freeze：** 唯一使用DFlash/shared vocabulary projection派生的full-vocab
   `base_logsumexp`；增加logsumexp/scalar/token replay parity，并将reduction计入profile。
7. **Capacity boolean：** 只有D64同时失败same-set capacity和预注册full-fit gate，
   才在fresh300前让三臂统一升级D256；transfer失败禁止加参。
8. **Scale decisions：** 25K seed0 hard gate冻结为
   `max(7.80,1.075x Domino)`加domain/J2门；100K final-opening门冻结为
   `max(8.3254859086,1.15x same-job Domino)`；25K/100K都显式排除fresh300/final600。

## Revised Proposal Entry Point

完整、已就地修订的proposal为 [round-1-refinement.md](round-1-refinement.md)。本文件
记录Round 2 delta；二者共同构成Round 3 re-review的唯一输入。若二者冲突，以本文件
及 `round-1-refinement.md` 中较新的保守定义为准。

## Simplicity Check

- online参数仍为D64/H4/L1的`433,852`；
- online dataflow、调用次数与输出形状均未改变；
- loss仍只有一个JAPD package，`L_AP+L_J2`无可调混合权重；
- 新增内容全部是离线监督定义、严格采样或实验门，不增加推理成本。
