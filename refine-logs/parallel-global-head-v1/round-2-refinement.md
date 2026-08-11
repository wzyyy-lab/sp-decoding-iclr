# Round 2 Refinement：PGCF-16 可执行规格闭合

> 本文件是 `round-1-refinement.md` 的规范性增量；冲突处以本文件为准。核心架构、数据流和用户硬约束不变，不增加任何模块。

## 1. Anchor 与结构保持不变

主方法仍是：full `H[16]` 与纯 base `Top16[16,16]` → 256 candidate nodes → 两层完整无 mask non-causal self-attention → 一次 `[B,16,16]` scores → 一次逐位置 argmax → 唯一 `[B,16]` proposal。

没有 causal mask、selected-token feedback、GRU、位置 decode loop、第二次 head、target seed、额外 target inference、迭代修复、beam/tree/trie/forest 或多路径 verification。

## 2. Gate 0 测试拆分

两个测试不再共享同一 scorer 状态：

1. **Identity test：** production init 的 `W_out=0`；逐元素断言 `scores == base_topk_logits`，argmax proposal逐 token等于 base Top-1，full validation EAL复现 `6.068513120`。
2. **Global visibility test：** 不改变 production init；在测试副本上将 `W_out` 装入固定 deterministic nonzero probe，或直接检查 `X²`。保持 position 0 的 `H/C/B` 不变，只扰动 position 15 的完整一致 triplet，要求 `X²[position0]` 与 probe score发生非零变化；反向测试 position15读取position0。local matched control在同一扰动下不得跨位置变化。

这样 identity fallback 与 global可见性分别验证，不再逻辑冲突。

## 3. Safe FP32 prefix/KL/teacher loss

### 3.1 Prefix utility

`gold_rank=-1` 表示 gold不在base Top16。实现必须先safe gather，不能对 `-1` 或越界rank gather后再乘零：

```text
support      = (gold_rank >= 0)                         # [B,16]
safe_rank    = where(support, gold_rank, 0)
log_q        = log_softmax(scores.float(), dim=-1)
ell_raw      = gather(log_q, safe_rank)
ell          = where(support, ell_raw, 0.0)
support_pref = cumprod(support.float(), dim=-1)
log_survival = cumsum(ell, dim=-1)
U            = sum(support_pref * exp(log_survival), dim=-1)
L_prefix     = -mean(U / 16)
```

所有累加、softmax和指数为FP32。首个out-of-K及其suffix被 `support_pref=0` 截断；`ell=0` 只保证数值安全，不重新放行suffix。

### 3.2 Target candidate KL

```text
target_match = (target_top1_ids == gold_ids)
clean_prefix = cumprod(target_match.float(), dim=-1).bool()
valid_kl     = support & clean_prefix
kl_rows      = KL(softmax(target_candidate_logits.float()),
                  softmax(scores.float()))
L_KL         = sum(where(valid_kl, kl_rows, 0)) /
               clamp_min(sum(valid_kl), 1)
```

若一个batch没有valid row，分子为0、分母为1，`L_KL=0`；无 CUDA tensor→Python bool分支。首个target replay mismatch后的suffix全部无KL梯度。

### 3.3 Teacher CE 与显式时间权重

Teacher rank同样用safe rank与support mask，空集合项为0。令训练进度 `tau=completed_updates/total_updates`：

| progress | `lambda_teacher` | `lambda_prefix` | `lambda_KL` |
|---|---:|---:|---:|
| `0 <= tau < 0.10` | 1 | 0 | 0 |
| `0.10 <= tau < 0.30` | `1-u` | `u` | `0.05u` |
| `0.30 <= tau <= 1` | 0 | 1 | 0.05 |

其中 `u=(tau-0.10)/0.20`。实际 loss 唯一为：

```text
L = lambda_prefix(tau)*L_prefix
  + lambda_KL(tau)*L_KL
  + lambda_teacher(tau)*L_teacher
```

## 4. 唯一可复现的参数规格

默认 `d=256, heads=8, layers=2, FFN ratio=2, L=16, K=16, dropout=0`。

Bias/affine flags冻结如下：

- `W_h`, shared `W_e`, `W_mul`, combined `W_qkv`, `W_o`, FFN up/down, final scorer：`bias=False`；
- five-scalar `Linear(5,256)`：`bias=True`；
- 所有 LayerNorm：`elementwise_affine=True`；hidden/embedding输入RMS归一化：`elementwise_affine=False`；
- 每个attention block有 trainable relative draft-position table `[31,8]` 和 same-position bias `[8]`；两者只加score bias，不mask node。

逐项参数：

| component | count |
|---|---:|
| `W_h + W_e` | 1,310,720 |
| position/rank embeddings | 8,192 |
| scalar projection incl. bias | 1,536 |
| `W_mul` | 65,536 |
| input LN | 512 |
| each block: 2 LN + QKV/O + FFN + relative/same bias | 525,568 |
| two blocks | 1,051,136 |
| output LN + scorer | 768 |
| **total** | **2,438,400** |

每block的 `525,568 = 1,024 + 196,608 + 65,536 + 262,144 + 248 + 8`。实现Gate 0用 `sum(p.numel() for p in model.parameters()) == 2_438_400` 硬断言；若真实实现不同，禁止launch并同步修文档，不能为了对数字而遗漏参数。

checkpoint后derived BF16 table精确为 `151936×256×2 = 77,791,232 bytes = 74.19 MiB`。

## 5. Profile 规格

同一 A40、batch1、L16、BF16、相同warmup/同步点，eager-to-eager分别报告：

1. **complete pipeline**：shared/base vocab GEMM → Top16 → gather → PGCF/Domino head → argmax；
2. **incremental head**：从已经生成的base Top16 IDs/logits起计，到唯一proposal止；
3. p50、p90、mean，至少1000 timed iterations；
4. peak allocated/reserved memory与77.8MB projected table；
5. eager/compiled/CUDA-graph状态必须两侧相同并明确记录。

Gate仍为 complete eager PGCF pipeline `<=1.20×` complete eager released Domino pipeline；公共base开销不能替代incremental latency报告。

## 6. Gate 2 数字与 remote intervention

在 15,886 train / 1,175 disjoint `validation_select` 上固定比较 full-global与matched-local：

- `EAL_global - EAL_local >= 0.15`；
- 以prompt为cluster做10,000次paired bootstrap，固定seed 20260810，95% CI下界 `>0`；
- global相对base必须为正，三个domain各自不低于base。

Remote intervention不是把单个tensor随意打乱。对每个待评位置 `i` 单独构造一次诊断输入：

1. 保留原block的anchor和position `i`完整 `(H_i,C_i,B_i)`；
2. 其余15个位置从同domain、同context-length quartile的固定deranged donor block取完整 `(H_j,C_j,B_j)`，candidate embedding由donor `C_j`一致gather；
3. 运行同一global head，只取position `i`输出；16次结果拼成一个诊断proposal；
4. donor mapping按稳定排序后循环移一位生成，不读取label。

令 `Delta=EAL_global-EAL_local`，shuffle erasure定义为

```text
erasure = 1 - (EAL_remote_shuffle - EAL_local) / Delta
```

要求 `erasure >=0.50`。该16-pass流程只作离线机制诊断，绝不进入部署。

## 7. Gate 3/4 数字冻结

### Gate 3 full16 OPB

- teacher-only EAL `>=7.080272109`，即距exact supported-policy ceiling `7.180272109` 不超过0.10；
- target Stage-A `validation_select` EAL `>=7.55` 且高于released Domino；
- 每个domain EAL `>=` 同domain base Top1；
- teacher unsupported rows始终fallback base，不扩candidate set。

### Gate 4 development freeze

固定recipe训练seeds `{0,1,2}`。每个seed的checkpoint按 `validation_select` prompt-balanced EAL最大选择；完全相同时取更早step。最终deployment/formal checkpoint在formal之前按三者中最高 `validation_select` EAL选择；完全相同时取更小seed、再取更早step。

development总门：

- selected checkpoint overall fixed EAL `>=8.325485909`；
- 每个domain EAL至少为same-job Domino domain EAL `-0.05`；
- architecture、loss、schedule、seed、checkpoint rule到此冻结。

## 8. Formal data receipt 与一次性规则

准确表述：reserved formal 600 prompts的manifest曾用于exclusion审计，但其PGCF full16 labels、head outputs、EAL和dynamic outcomes尚未生成/读取。因此“outcomes fresh”，不是“prompt identities从未看过”。

在任何formal label生成前执行一次不依赖label的去重receipt：

- sample-ID交集；
- tokenizer prompt-token IDs exact match；
- Unicode NFKC + whitespace collapse后的字符13-gram Jaccard，`>=0.80` 定义为near duplicate。

exact/near-duplicate formal rows在label生成前排除，冻结剩余N和domain counts；之后不能按结果替换样本。该检查比较实际内容，不设置artifact hash门。

Formal前冻结并写入experiment plan：target/draft checkpoint、BF16、batch1、B16、greedy temperature0、max64 new tokens、EOS规则、fixed evaluator、dynamic block boundary生成、same-job released Domino命令、A40型号、bootstrap seed与metric脚本。

Formal一次性执行：

- primary checkpoint是上述pre-formal选定的单一checkpoint；formal结果不能反向选seed或step；
- primary fixed EAL `>=1.15×` same-job Domino；
- primary每个domain点估计 `>=` same-job Domino对应domain；
- 三个已冻结seed可作为一次性sensitivity同时报告，但不改变primary选择；
- dynamic EAL `>=1.15×` same-job Domino；
- fixed/dynamic通过后才做同栈SGLang，TPS `>=1.15×` Domino。

若formal失败，该600-prompt outcome立即降级为development evidence；任何后续改法都必须使用新的外部heldout，不能再次称其为fresh confirmatory test。

## 9. READY 条件

本轮只闭合规格，不改变方法。若reviewer确认：

- immutable architecture constraints全部满足；
- safe loss无undefined rows；
- 2,438,400参数可唯一复现；
- profile/global-use/teacher/domain/formal gates全部量化；

则直接进入 `experiment-plan → experiment-bridge`，先实现Gate0/512 capacity/A40 eager profile，不提前启动199.8K训练。
