# PARC-16 Implementation and Trace Guide

本文档对应当前可执行版本。方法的数学定义以
`refine-logs/parallel-global-head-v4/FINAL_PROPOSAL.md` 为准；用户不可变架构约束以
`refine-logs/parallel-global-head-v1/USER_CONSTRAINT_CONTRACT.md` 为准。

## 1. 在线数据流

1. Pure DFlash 对完整 block 做一次并行 forward。当前 released pure checkpoint 是
   non-shift 模型，因此 full16 扩展几何固定为 `[anchor] + 16 masks -> raw17 ->
   rows[1:17]`。实现位于 `nonshift_full16_prediction_hidden()`。
2. 冻结 target LM head 对 16 个 hidden rows 做 base vocabulary GEMM；每位置取
   FP32 Top-16，得到 `candidate_ids/logits [B,16,16]`。BF16 并列最大值时，
   `greedy_first_topk()` 强制普通 vocabulary `argmax` 位于 rank 0，同时保持候选
   唯一；collector 与 trainer 共用该实现，zero-residual identity 不依赖
   `torch.topk` 的未定义 tie ordering。
3. `PARC16Head` 对每个 `(position, candidate)` 建立一个 action node，共 256 个。
   Node 包含当前位置 hidden、candidate 与 rank-0 embedding delta、anchor、position、
   rank，以及 centered logit/log-prob/gap/rank/entropy 五个在线标量。
4. 256 nodes 同时通过两层 D256/H8/FFN512 的无 causal mask self-attention。
   `FullCandidateFusionBlock(local_control=False)` 保证跨位置全局可见。
5. Zero-initialized residual scorer 输出 `[B,16,16]` correction advantages；rank-0
   gauge 固定为 0。一次逐位置 argmax 产生唯一 `[B,16]` proposal。

不存在被选 token 的回灌、位置循环、GRU state、target seed、第二次 head forward、
候选路径或 tree verifier。

## 2. 参数账本

冻结的 D256/L2 head 共 2,438,400 个 trainable parameters：

- hidden/token projections：1,310,720；
- position/rank embeddings：8,192；
- five-scalar projection：1,536；
- compatibility projection：65,536；
- input normalization：512；
- two global fusion blocks：1,051,136；
- output normalization/scorer：768。

`assert_frozen_architecture()` 会在训练前 fail closed。

## 3. 离线 trace schema

`scripts/collect_parc16_data.py` 每个 retained prompt 保存：

- `sample_id/domain/source/split`；
- `target_context_features [context,12800]`（BF16 selected-layer features，8 个
  anchor 共享）；
- 8 个均匀 full16 anchors，每个 anchor 包含：
  - `anchor_offset/context_length/anchor_token_id`；
  - `gold_ids [16]`；
  - `reference_topk_ids [16,16]`；
  - `reference_topk_logits [16,16]`；
  - `reference_proposal_ids [16]`；
  - `reference_accepted_length`；
  - `reference_delta_fp32` 与 `numeric_margin_error`；
  - validation 上额外保存 released Domino proposal/accepted length。

只有 target greedy generation 在 EOS 前实际产生完整 129 tokens 的 prompt 才能进入
数据集。每个 part 的 train/validation/domain 配额必须精确填满，否则整个 part 不发布；
不会静默缩小 90K/5K。

## 4. Prompt reserve 与数据隔离

`scripts/build_open_perfectblend_manifest.py` 从本地 Open-PerfectBlend parquet 生成去重、
去既有 development/formal overlap 的 270K prompt source（每域 90K）。然后：

```bash
PYTHONPATH=src:scripts python scripts/build_parc16_split.py \
  --source artifacts/manifests/open_perfectblend_270k_parc_reserve_v1.jsonl \
  --output artifacts/manifests/parc16_opb270k_reserve_v2 \
  --seed 20260810 \
  --parts 16
```

标签生成前即固定：

- train candidates：240K，最终 retained 90K（chat/code/math 各 30K）；
- validation candidates：15K，最终 retained 5K（1667/1666/1667）；
- held-out candidates：15K，最终首次评测时按冻结顺序取 5K eligible prompts。

三个 candidate pools 的 prompt IDs 互斥。M1/M2 完全不读取 held-out labels、baseline
或 statistics。

## 5. 收集 full16 traces

集群入口：

```bash
sbatch scripts/slurm/parc16_full_data.sbatch
```

该 launcher 是 16-way one-A800 array。部署到其他集群前修改文件顶部的
`PROJECT/ASSETS/PYTHON/MANIFEST_ROOT`。单个 part 的等价命令为：

```bash
PYTHONPATH=src:. python scripts/collect_parc16_data.py \
  --target /path/to/Qwen3-4B \
  --draft /path/to/Qwen3-4B-DFlash-b16 \
  --domino-draft /path/to/Qwen3-4B-Domino-b16 \
  --manifest /path/to/train_validation_parts/part-000.jsonl \
  --continuation-tokens 129 \
  --anchors-per-prompt 8 \
  --shard-prompts 16 \
  --attn-implementation sdpa \
  --output /path/to/output/part-000
```

Collector 强制 pure checkpoint 为 non-shift B16、Domino 为 shift-label B16；旧
15-position cache 和 incomplete output 都会被拒绝。

## 6. 正式训练

训练入口：

```bash
PARC_DATA_ROOT=/path/to/completed/16-part-data \
PARC_OUTPUT=/path/to/one-formal-output \
sbatch scripts/slurm/parc16_joint_train.sbatch
```

冻结 recipe：batch 8 blocks、180K updates、head LR 3e-4、DFlash LR 1e-5、2K
warmup、cosine decay 到 10%、AdamW、clip 1、seed 0。DFlash 的 537,427,200
parameters 与 PARC 联合训练；target lexical table 冻结。

每 10K steps 在完整 5K validation 上同时报告 PARC、extended full16 pure DFlash
reference 和 released Domino。只有 actual prefix harm <=1% 的 checkpoint 可参与选择，
再按 prompt-balanced validation EAL 取严格最大值；完全相同取更早 step。Step 0 只在
固定 5K train-audit 上做 identity parity，不读取 validation，也不能被选中。

每 1K step 保存 model、FP32 master optimizer、head optimizer、dual state、sampler、
Python/CPU/CUDA RNG。只有 scheduler interruption 可原样 resume；scientific infeasibility
或 complete 状态禁止 resume。

## 7. Objective 与验证含义

- Base D-PACE loss 更新 live DFlash。
- Conditional gain loss 只奖励 immutable reference accepted prefix 之后、仍在 live
  Top-16 support 中的连续 gold suffix。
- Blockwise harm envelope 保护 immutable reference 已接受 prefix；protected gold 掉出
  live Top-16 时 fail closed 为 harm upper bound 1，并 mask gain。
- `delta_min` 和 numeric certificate 只来自 90K train；validation 不影响 loss、launch、
  dual 或 stop。
- 训练集 EAL 仅诊断。论文效果必须来自锁定 checkpoint 后的一次 sealed held-out
  fixed/dynamic same-job DFlash/Domino/PARC evaluation。

## 8. 测试

```bash
PYTHONPATH=src:scripts python -m pytest -q \
  tests/test_parc.py \
  tests/test_collect_parc16_data.py \
  tests/test_parc_training.py \
  tests/test_build_parc16_split.py \
  tests/test_open_perfectblend_manifest.py
```

测试覆盖 zero-init identity、跨位置 noncausal gradient、gain gradient 等价、harm
upper bound、raw17/full16 slicing、reserve manifest、deterministic sampler/resume、
prompt-balanced metrics 和 terminal-state semantics。

## 9. 当前状态

截至 2026-08-12：focused tests 22 passed；独立 code review 对 M1/M2 均为 GO。
首投 `10169014` 暴露并列 Top-16 ordering 错误且没有发布数据；修复后的真实模型
检查 `10186345` 以 12 prompts/96 blocks 完成并记录 94 个 tie rows。正式数据 job
`10186352` 等待 A800 priority；正式训练 job `10186353` 以
`afterok:10186352` 依赖排队。尚无正式 validation/held-out EAL。

## 10. 发布范围

本仓库只发布 PARC-16 当前方案及其直接依赖。已经证伪或淘汰的 PGCF、JAPD、PCLD、
GFPR、PLC、R048–R056 代码与中间审查记录不在当前版本中，避免把历史探索误认为
可用方法。正式效果仍只能由冻结 checkpoint 后的 validation/held-out 结果支持。
