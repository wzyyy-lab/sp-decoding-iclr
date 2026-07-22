# DFlash Survival Path Head：从可行性验证到 ICLR 投稿的完整执行计划

> 更新日期：2026-07-22
> 项目工作名：Survival Path Head（SPH）
> 目标：在不使用 draft tree、tree attention 或多分支 target verification 的前提下，用并行结构化 head 替换 Domino 的顺序 GRU head，提高 DFlash 单链的接受长度和端到端吞吐。

## 0. 当前证据状态与执行纪律

截至 2026-07-22，只有两个真实研究实验：

- `10022338`：候选上限，支持 Gate 1a（candidate availability）通过；
- `10022436`：DFlash/Domino eager 基线，属于真实但 development-grade 的
  基线结果，不能当最终论文 timing。

其他已有 job 都是环境、collector、诊断或反向传播 plumbing smoke，不是
方法实验。任何文档、表格和 checkpoint 必须携带下列 evidence tier 之一：

- `plumbing_smoke`：只能证明程序能运行；
- `development`：可用于内部 gate/hyperparameter 选择，不能支持论文 claim；
- `formal`：冻结协议、数据、代码 revision 后的论文证据。

当前阶段不是“Gate 1 全部通过”，而是：

1. Gate 1a 已通过；
2. Gate 0 数值/缓存正确性尚待闭环；
3. Gate 1b 同锚点 Domino 对照尚待完成；
4. 还没有任何 trained-SPH 有效性结果。

## 1. 最终交付物

项目完成时必须同时具备以下结果，缺少任何一项都不能称为完成：

1. 一个可训练的 SPH 模型实现；
2. 一个与官方 DFlash Hugging Face 推理兼容的单链生成实现；
3. 一个经过正确性测试的 eager 版本；
4. 一个经过 CUDA Graph/Triton 优化的低延迟版本；
5. 纯 DFlash、Domino 和 SPH 的统一 benchmark harness；
6. Qwen3-4B 和 Qwen3-8B 上 math/code/chat 的完整结果；
7. local greedy、普通双向 head、Viterbi、survival-DP 等关键消融；
8. top-$K$ candidate ceiling、hazard/reach、校准和失败案例分析；
9. 可复现配置、固定 prompt manifests、checkpoint 和运行命令；
10. 一篇论点与实验闭环的 ICLR 稿件。

## 2. 不可改变的研究约束

以下约束贯穿全部实现和实验：

- 最终只输出一条 draft sequence；
- target 每轮只做普通的单链 parallel verification；
- 不使用 tree construction、tree attention 或多路径 target verification；
- 不把 DDTree rollout trace 用作当前方法的训练或主证据；
- 第一阶段冻结 DFlash，只训练新 head；
- 与 baseline 比较时固定 prompt、target、tokenizer、block size、temperature、硬件和计时方式；
- 任何接受长度收益都必须同时报告新增 drafting latency；
- 不把“首错后 GRU state pollution”作为接受长度收益的因果解释；
- 不宣称普通 bidirectional layer、candidate lattice、Viterbi 或 acceptance-aware loss 是首次提出。

## 3. 当前工作区与已知环境

### 3.1 已有资产

- DFlash 论文：`papers/dflash_2602.06036.pdf`
- Domino 论文：`papers/domino_2605.29707.pdf`
- 其他 related-work PDF：`papers/`，全部保留
- 官方 DFlash 源码：`third_party/dflash/`
- DFlash git commit：`94e4abc`
- 官方 Domino 源码：`third_party/Domino/`（commit `930e5cd82`）
- 当前 tensor prototype：`src/sph/survival_path_head.py`
- 当前单元测试：`tests/test_survival_path_head.py`
- 方法推导：`docs/method.md`

### 3.2 已缓存模型

当前 Hugging Face cache 已发现：

- `Qwen/Qwen3-4B`
- `z-lab/Qwen3-4B-DFlash-b16`
- `z-lab/Qwen3-8B-DFlash-b16`

当前本地已具备 Qwen3-4B/8B target、对应 DFlash checkpoints，以及
`Qwen3-4B-Domino-b16`。第一阶段使用 4B；8B Domino checkpoint 在扩展实验前下载。

### 3.3 当前计算限制

当前登录节点 `nvidia-smi` 无法连接 NVIDIA driver，因此这里只做 CPU 测试、数据结构和代码开发。GPU 作业通过 Slurm 提交：A800 使用
`i64m1tga800u`，A40 使用 `i64m1tga40u`，30 分钟 smoke 优先使用 A40
`debug`。A40 smoke job `10022278` 已验证环境、target、DFlash 和 Domino 均可运行。

## 4. 目标目录结构

随着实施推进，项目保持以下结构：

```text
dflash-iclr/
├── README.md                 # 项目入口和当前状态
├── PLAN.md                   # 本执行计划
├── pyproject.toml            # SPH Python package
├── docs/
│   ├── method.md             # 数学动机与方法定义
│   ├── experiment_log.md     # 每次正式实验的结论日志（后续创建）
│   └── paper_outline.md      # 论文提纲（进入写作阶段创建）
├── src/sph/
│   ├── survival_path_head.py # 当前 tensor 原型
│   ├── data.py               # canonical block dataset（后续）
│   ├── losses.py             # CRF/survival/calibration losses（后续）
│   ├── integration.py        # DFlash 推理集成（后续）
│   └── kernels/              # Triton kernels（通过 gate 后创建）
├── tests/                    # CPU/GPU correctness tests
├── scripts/                  # 采集、训练、评测和作图入口（后续）
├── configs/                  # YAML 配置（后续）
├── papers/                   # 所有论文 PDF；不删除
└── third_party/dflash/       # 官方源码，保持独立 git 历史
```

生成数据、checkpoint、profile 和 benchmark 输出进入 `.gitignore` 中的 `artifacts/`、`checkpoints/`、`outputs/`，不与源码混放。

## 5. 总体里程碑与停止门

| 里程碑 | 预计时间 | 交付物 | 继续条件 |
|---|---:|---|---|
| G0 正确性 | 1–2 天 | 固定序列下 full/cache/block/backend logit 误差报告 | 同 shape 可复现；跨 shape 分歧均落在实测误差包络内 |
| G1a Candidate availability | 已完成 | `10022338` top-$K$ coverage 与 oracle EAL | **通过**，K16 为主配置 |
| G1b Same-anchor ceiling | 1 天 | 同 anchor 的 DFlash/oracle/Domino 配对结果 | K16 oracle 显著超过 matched-horizon Domino |
| M2 Development probe | 1–2 天 | no-mixer local/global 三 seed 小样本结果 | 只决定是否扩数据，不形成 claim |
| M3 正式数据与训练 | 1–2 周 | 去污染训练集、数百 prompt test、学习曲线 | global 与 survival 在 held-out 上有独立增益 |
| G2 离线可行性 | M3 后 | 三 seed、cluster CI、跨域真实 EAL | 不牺牲首 token，主要域一致 |
| M4 Eager 集成 | 2–4 天 | 完整单链生成与组件计时 | correctness gate 保持通过 |
| G3 端到端价值 | M4 后 | 2048-token matched EAL/TPS/latency | 吞吐 CI 优于最强单链 baseline |
| M5 GPU 优化 | 1–2 周 | fused/CUDA Graph 版本 | head 开销低于收益 |
| M6 完整实验 | 2 周 | 主表、消融、分析 | 两个模型规模结果稳定 |
| M7 论文 | 1–2 周 | 完整稿、附录、复现包 | 所有 claim 有直接证据 |

总周期按 6–8 周规划。第一周和第二周各有一个强制停止门，避免在没有候选上限或没有吞吐收益时继续投入。

---

## 6. Phase 0：冻结环境与实验协议

### 6.1 建立项目版本记录

要做：

1. 在项目根目录初始化独立 git repository；
2. 将当前清理后的代码作为 `clean-sph-start` 初始 commit；
3. 记录 DFlash commit `94e4abc` 和 Domino commit；
4. 每次正式实验配置与代码 commit 绑定。

为什么：根目录此前没有 git，历史文件删除后不能恢复；正式研究必须保证每个结果可追踪。

### 6.2 创建运行环境

在 GPU 节点执行：

```bash
module load anaconda3
module load cuda/12.4
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e third_party/dflash[transformers]
```

若现有 Domino 环境已经验证可用，优先复制其 package lock，而不是混装多个不兼容的 torch/transformers 版本。

保存以下信息到 `artifacts/manifests/environment.json`：

- hostname、GPU 名称和数量；
- CUDA driver/runtime；
- Python、PyTorch、Transformers、Triton 版本；
- target/draft checkpoint revision；
- DFlash/Domino/SPH commit；
- dtype、TF32、attention backend。

### 6.3 固定评测协议

第一轮统一设置：

- target：Qwen3-4B，thinking disabled；
- draft：Qwen3-4B-DFlash-b16；
- block size：16（15 个 draft tokens，按实现对齐）；
- temperature：0；
- dtype：checkpoint 默认 bf16/fp16；
- batch/concurrency：先 1；
- max new tokens：开发阶段 256，正式阶段按论文设置；
- seed：0、1、2；
- timed prompts 与 warmup prompts 分离。

输出一个固定的 `artifacts/manifests/prompts_v1.jsonl`。以后所有方法读取同一 manifest，不能各自 shuffle 数据集。

### 6.4 Phase 0 验收

- `PYTHONPATH=src python -m unittest discover -s tests -v` 全部通过；
- DFlash 对 10 个 prompts 能完整生成；
- 两次运行 target greedy 输出 token 完全一致；
- 记录 draft、verify、总 decode latency；
- 连续 3 次 benchmark 的 TPS 波动小于 3%。

---

## 7. Phase 1：建立纯单链 baseline

### 7.1 DFlash baseline

基于 `third_party/dflash/dflash/benchmark.py` 建立统一 wrapper：

```bash
python -m dflash.benchmark \
  --backend transformers \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 128 \
  --temperature 0
```

需要额外记录每轮：

- accepted draft count 与 bonus token 是否计入；
- draft backbone、LM head、verification 单独 latency；
- peak memory；
- 输出 token 数和总 wall time。

### 7.2 Domino baseline

下载或定位 `Huang2020/Qwen3-4B-Domino-b16`，使用本机 Domino reviewer code 运行同一 prompt manifest。禁止直接引用论文表格作为最终 baseline。

输出与 DFlash 完全相同的 JSON schema，并额外记录：

- GRU/correction head latency；
- 每位置 corrected token/logit；
- on-policy GRU state norm；
- 是否启用 CUDA Graph/fused kernel。

需要分别报告：

1. Domino eager，便于方法正确性比较；
2. Domino 官方优化版，作为最终系统 baseline。

### 7.3 Baseline 验收

- DFlash 与 Domino 最终生成结果均与 target greedy 完全一致；
- acceptance length 的计数口径统一；
- 同一方法三次运行的均值和标准差已保存；
- 输出 `outputs/baselines/qwen3_4b_t0.json`。

---

## 8. Phase 2：采集 canonical-anchor 数据

这是整个项目最关键的数据步骤，目的是消除不同 speculative 方法产生不同 anchor 分布的混淆。

### 8.1 Canonical anchor 的定义

对每个 prompt：

1. target 使用 greedy decoding 生成一条固定 continuation；
2. 在该 target continuation 上按 token offset 选择 anchor；
3. 对每个相同 anchor，独立运行一次 DFlash parallel backbone；
4. gold block 是该 anchor 后真实 target continuation；
5. DFlash、Domino、BiHead、SPH 都在这些相同 block 上离线评估。

这不是任何方法的 rollout，因此所有 head 收到相同 context、相同 target 和相同位置。

### 8.2 实现脚本

创建 `scripts/collect_canonical_blocks.py`，接口固定为：

```bash
python scripts/collect_canonical_blocks.py \
  --target Qwen/Qwen3-4B \
  --draft z-lab/Qwen3-4B-DFlash-b16 \
  --manifest artifacts/manifests/prompts_v1.jsonl \
  --block-size 16 \
  --top-k 64 \
  --anchors-per-sample 32 \
  --output artifacts/canonical/qwen3_4b
```

### 8.3 每个 block 保存的字段

```text
sample_id                 string
domain                    math | code | chat
prompt_token_count        int
anchor_offset             int
anchor_token_id           int
gold_ids                  int32[L]
parallel_hidden           bf16[L, D]
base_topk_ids             int32[L, 64]
base_topk_logits          fp16[L, 64]
base_logsumexp            fp32[L]
target_topk_ids           int32[L, 64]       # 可选但推荐
target_topk_logits        fp16[L, 64]         # 可选但推荐
base_top1_match           bool[L]
metadata                  checkpoint/config revisions
```

数据按 1–2 GB shard 保存，先写临时文件，完成校验后原子 rename，避免中断产生半个 shard。

### 8.4 数据切分

- 按原始 prompt 切分 train/validation/test，不能按 anchor 随机切分；
- 建议 80%/10%/10%，但 test 必须有数百 prompts，不能只看比例；
- 同一 prompt 的所有 anchor 必须只属于一个 split；
- benchmark test prompts 不得进入 head training。

现有 `prompts_v1` 只有 96 prompts（75/9/12，约 600/72/96 blocks），并且
来源是 GSM8K/MATH500/HumanEval/MBPP/MT-Bench 的评测 prompts。它固定命名
为 `probe_v1` 语义，只能用于 Gate 1 与 development learnability probe。

正式训练按学习曲线逐级扩展：

1. 2k disjoint training prompts × 8 anchors，约 16k blocks；
2. 5k prompts，约 40k blocks；
3. 10k prompts，约 80k blocks；
4. 独立 validation/test，各至少数百 prompts，且正式 benchmark 从不训练。

只有学习曲线仍在上升时才扩大下一档，避免盲目采集。

### 8.5 Collector correctness tests

创建 `tests/test_canonical_collector.py`，验证：

- top-1 由保存的 top-$K$ logits 可重建；
- gold_ids 与 target greedy continuation 一致；
- base_logsumexp 不小于 top-$K$ logsumexp；
- 同一 anchor 重跑 hidden/logits 在容差内一致；
- 直接用 base top-1 计算的 EAL 与 DFlash offline 结果一致；
- 数据中没有 DDTree path/tree 字段。

---

## 9. Gate 1：候选可用性与同锚点可实现上限

在训练任何 head 之前创建 `scripts/analyze_candidate_ceiling.py`。

### 9.1 必算指标

对 $K=1,2,4,8,16,32,64$ 分别计算：

- 每位置 gold top-$K$ recall；
- 到位置 $i$ 为止 gold 全部在 lattice 中的 prefix coverage；
- oracle accepted length：每位置只要 gold 在候选中就由 oracle 选择；
- first-miss gold rank；
- candidate entropy、top1/top2 margin；
- math/code/chat 分域结果；
- 95% bootstrap confidence interval。

### 9.2 Gate 1a：候选可用性

`10022338` 已完成该部分：K16 oracle accepted-draft EAL 为 11.178，
相对 K1 增加 5.148，且 headroom 跨 math/code/chat 存在。因此 Gate 1a
通过。该结果只证明正确 token/path 存在于 DFlash candidate lattice，不证明
learned selector 能找到它，也不能与不同分布的在线 Domino EAL 直接比较。

### 9.3 Gate 1b：同锚点比较

必须在 `10022338` 的每一个相同 canonical prefix 上同时测量：

- pure DFlash top-1；
- DFlash K8/K16 oracle；
- Domino checkpoint 的 backbone top-1；
- Domino on-policy GRU proposal，统一截断到 15 个 draft positions。

重建的 anchor/gold 与现有 shard 任一 token 不一致时，作业必须失败，不能
跳过。主统计使用 prompt-cluster bootstrap。K16 oracle 相比 matched-horizon
Domino 的 point gain 必须至少达到 `max(0.5 token, 10%)`，且 95% CI 排除 0。

### 9.4 Gate 1 总判定

推荐的继续条件：

1. Gate 1a 与 Gate 1b 均通过；
2. $K=16$ oracle EAL 在相同 anchors/horizon 上至少比 Domino 高 0.5 token，或者高 10%（取更严格者）；
3. 主要 headroom 不只来自一个数据集；
4. 从 $K=16$ 增加到 $K=32$ 的收益与预估额外延迟仍有合理 Pareto；
5. 第 1–3 位的 candidate coverage 足够高，因为它们决定大部分 EAL。

若不满足：停止 SPH 主线。可记录失败结论，但不继续做 GPU kernel。不能通过使用 DDTree top-$K$ recall 替代该 gate。

输出：

- `outputs/analysis/candidate_ceiling.json`
- `outputs/analysis/candidate_ceiling.pdf`
- `docs/experiment_log.md` 中的明确 go/no-go 结论。

---

## 10. Phase 3：实现四个严格可比的 head/decoder

### 10.1 共同输入和参数预算

所有方法只读取：

- DFlash `parallel_hidden`；
- DFlash base logits/top-$K$；
- verified anchor token；
- frozen target embedding。

第一轮使用 $K=16$、rank 32。主 SPH 先使用 no-mixer 的最小低秩
pairwise scorer；DFlash hidden 已含 block 双向信息，因此额外 bidirectional
mixer 只作为容量消融。所有 learned baseline 的参数量控制在 SPH 的 ±10%，
并报告真实 FLOPs 和 latency。

### 10.2 Baseline A：Independent BiHead

目的：回答“普通轻量双向 head 是否已经足够”。

具体实现：

1. 将 DFlash hidden 和 top-$K$ soft token embedding 各自投影到 64 维；
2. 相加后进入一个长度 15 的单层 bidirectional mixer；
3. mixer 使用 4-head self-attention 或 matched-parameter BiGRU，两者先选延迟更低者；
4. 输出 candidate residual logits；
5. 每个位置独立 argmax，不做结构化 path search。

必须测量：相对 DFlash 的 EAL 增益、head latency、参数量。

### 10.3 Baseline B：Local Pairwise Head

使用当前 low-rank edge scorer，但从左到右 local greedy：选定前一 token 后，只选当前条件概率最大的候选。它代表没有全局 future value 的 Markov correction。

### 10.4 Baseline C：Absorbing-OTHER Global Prefix-CRF + MAP

使用与 proposed 完全相同的 edge scores、full-vocabulary outside mass 和
absorbing-OTHER global normalization，但在完整候选路径中执行 MAP。它用于
区分“global normalization”与“acceptance objective”。不含 outside mass 的
candidate-only CRF 另列为消融，不能充当 proposed。

### 10.5 Proposed：Absorbing-OTHER Global Prefix-CRF + Survival-DP

仍使用完全相同 edge scores 与全局诱导的 candidate sub-probabilities，但通过

```text
V_i(previous) = max_current p(current | previous)
                * (1 + V_{i+1}(current))
```

选择预测 expected accepted prefix 最大的唯一 path。

### 10.6 必须保留的退化测试

- residual scale 为 0 时恢复 DFlash candidate scores；
- residual scale 为 0 时 global log-partition 为 0，candidate+OTHER 条件概率逐状态严格恢复 DFlash；
- future weight $lambda=0$ 时退化为 local greedy；
- $lambda=1$ 时为完整 survival objective；
- $K=1$ 时所有 path decoder 输出 DFlash top-1；
- CRF log-partition 和 survival-DP 对小规模输入与 brute force 完全一致。

---

## 11. Phase 4：训练流程

### 11.1 Stage A：冻结特征训练

先直接读取 canonical shards，不加载 target/DFlash 模型。这样一张普通 GPU 即可训练 head，迭代速度最快。

默认配置：

```yaml
candidate_k: 16
rank: 32
block_length: 15
optimizer: adamw
learning_rate: 3.0e-4
weight_decay: 0.01
warmup_ratio: 0.03
epochs: 5
precision: bf16
grad_clip: 1.0
seeds: [0, 1, 2]
```

学习率额外 sweep `1e-4, 3e-4, 1e-3`，但只能用 validation set 选择。

### 11.2 Candidate outside 处理

主路径从开发阶段起就使用 absorbing `OTHER` state，保存 exact base outside
mass，并在 gold 首次离开 top-$K$ 时做 prefix-censored supervision。这样
residual=0 能严格恢复 DFlash，也避免开发/正式目标不一致。

gold injection 仅允许作为明确标注的 debugging ablation，不能用于 checkpoint
选择、Gate 2 或论文主结果。

### 11.3 Loss

总损失包含：

- chain-CRF NLL；
- acceptance/survival auxiliary loss；
- base-anchor regularization，防止 residual 过大破坏 DFlash；
- 可选 calibration loss。

初始权重：

```text
L = L_crf + 0.1 * L_survival + 0.01 * L_base_anchor
```

随后只做小规模权重 sweep。不能使用 test set 选 loss 权重。

### 11.4 Checkpoint 选择

不能只按 token CE 选择。主选择指标是 validation canonical blocks 上的真实 accepted length；同时要求：

- 第 1 位准确率不显著下降；
- predicted utility 与 realized accepted length 校准；
- math/code/chat 至少两类不退化。

每个 checkpoint 保存 config、git commit、seed、training curves 和 validation metrics。

### 11.5 Stage B：可选联合微调

只有冻结 DFlash 的 SPH 已通过 Gate 2 后，才允许解冻 DFlash 最后 1–2 层或 correction-related projection。联合训练必须作为后续增益，不能掩盖 head 本身是否有效。

---

## 12. Gate 2：离线可行性判定

在完全未参与训练的 canonical test blocks 上比较：

1. DFlash top-1；
2. Independent BiHead；
3. Local Pairwise；
4. CRF Viterbi；
5. CRF Survival；
6. Domino eager。

### 12.1 必报指标

- average accepted draft tokens；
- first-token accuracy；
- 每位置 marginal accuracy；
- reach probability；
- conditional hazard；
- full-block acceptance；
- predicted/realized EAL calibration；
- 每个 domain 的 bootstrap CI；
- 参数量、FLOPs 和 eager head latency。

### 12.2 Gate 2 继续条件

- Survival-DP 显著优于相同 edge scorer 的 local greedy 和 Viterbi；
- Survival-DP 显著优于 matched-parameter Independent BiHead；
- 第 1 位准确率没有超过 0.2 percentage point 的显著下降；
- EAL 增益至少在 math/code/chat 中两类成立；
- 三个随机种子方向一致。

若失败，最多允许两次有理论依据的修改：

1. 增加 order-2/skip transition；
2. 增加小型 latent mode state。

不能通过无边界扩大 head、增加多轮 refinement 或切换到 tree verification 来“救”结果。两次修改后仍失败则主线 no-go。

---

## 13. Phase 5：接入 DFlash eager 推理

### 13.1 不直接污染 third-party 源码

先在 `src/sph/integration.py` 复制并最小改写官方 `dflash_generate` 控制流，调用原 DFlash model；不直接在 vendor git 中长期堆修改。稳定后再生成一份可审阅 patch。

### 13.2 插入点

官方 DFlash 每轮完成：

1. parallel backbone 得到 hidden；
2. target LM head 一次得到 base logits；
3. base logits 直接采样/argmax；
4. target 验证最长匹配前缀。

SPH 只替换第 3 步：

```text
base logits
  -> top-K ids/logits
  -> gather projected token embeddings
  -> pairwise edge scores
  -> CRF backward messages
  -> survival-DP/backtrack
  -> one draft token sequence
```

target verification、KV crop 和 bonus token 逻辑保持原样。

### 13.3 Correctness

创建 `tests/test_integration_gpu.py`：

- residual=0/K=1 与 DFlash 输出一致；
- SPH 只改变 draft path，不改变 target greedy 最终输出；
- 连续生成、EOS、短剩余长度和 block 尾部正确；
- cache crop 后下一轮 hidden 正确；
- 100 个 prompts 上 target-only 与 SPH 最终 token 完全一致；
- 显存无逐轮增长。

### 13.4 Temperature 范围

第一篇实现先完成并验证 $T=0$。$T=1$ 只有在给出正确 proposal/verification 逻辑和分布一致性测试后进入主表，不能直接把 greedy path 当作普通 stochastic proposal 而声称 lossless。

---

## 14. Gate 3：端到端价值判定

在 Qwen3-4B、batch 1、相同 prompts 上测：

- DFlash；
- Domino eager 与官方优化版；
- Independent BiHead eager；
- SPH eager。

每项至少：

- 20 次 warmup；
- 100 个以上 timed decoding rounds；
- 3 次完整重复；
- CUDA event 分阶段计时；
- wall-clock tokens/s；
- 95% CI。

继续 GPU kernel 优化的条件：

1. SPH 的 EAL 已明显高于 Domino 或接近 Domino但 head 结构显著更易优化；
2. eager SPH 的额外延迟没有吞掉全部理论收益；
3. 根据 profile，top-$K$/edge/DP 存在明确 fusion 空间；
4. 估计优化后端到端 TPS 至少能超过 Domino 3–5%。

若 eager EAL 有收益但即使按 kernel 下界估算也无法提高 TPS，则停止系统实现，不以 EAL 单独包装成加速论文。

---

## 15. Phase 6：GPU 性能优化

按 profile 顺序优化，不预先写复杂 kernel。

### 15.1 预计算 token projection

训练结束后，将冻结 embedding 经过左右低秩投影，保存为两个 `[vocab, rank]` lookup tables。推理时只 gather top-$K$ 行，避免每轮执行 candidate embedding 的大矩阵乘。

### 15.2 固定 shape

固定 $L=15$、$K=16$、rank 32，所有 tensor 预分配。禁止每轮 Python list、动态 shape、CPU sync 和 `.item()`。

### 15.3 Kernel 顺序

1. PyTorch batched implementation；
2. `torch.compile` control；
3. fused gather + edge scoring Triton kernel；
4. fused CRF backward + survival-DP + backtrack kernel；
5. CUDA Graph capture 整个 head；
6. 若 top-$K$ 成为瓶颈，再考虑与 LM-head epilogue 融合。

### 15.4 性能正确性

每个 kernel 与 fp32 reference 比较：

- edge scores 数值误差；
- log-partition；
- selected path；
- predicted utility；
- 极端 logits、重复 candidate、短 block；
- 多 batch 静态 shape。

### 15.5 性能报告

分别报告：

- head microseconds；
- head 占总 draft latency 比例；
- 与 Domino GRU/correction head latency 比；
- peak memory；
- batch/concurrency 1、2、4、8、16；
- A100/H100 等实际可用硬件，不跨机器混报。

---

## 16. Phase 7：完整实验矩阵

### 16.1 模型

主实验：

- Qwen3-4B + DFlash-b16；
- Qwen3-8B + DFlash-b16。

资源允许时增加第三个不同 family/scale，避免结论只适用于 Qwen3。

### 16.2 数据集

Math：

- GSM8K；
- MATH-500；
- AIME24/25。

Code：

- HumanEval；
- MBPP；
- LiveCodeBench。

Chat：

- MT-Bench prompts；
- Alpaca held-out prompts。

### 16.3 Baselines

必须运行：

- vanilla target autoregressive；
- DFlash；
- Domino；
- Independent BiHead；
- Local Pairwise；
- CRF Viterbi；
- CRF Survival。

能获得官方实现时加入：

- DSpark Markov/RNN；
- DeLS-Spec；
- DiffuSpec-style causal path search 移植到相同 DFlash candidates。

后一个对照非常重要：它回答收益是否只是“把 DiffuSpec beam search 搬到 DFlash”。 proposed method 必须在相同 candidate lattice 上证明 learned energy、exact normalization 或 survival objective 的独立贡献。

Tree 方法不属于 proposed scope，但可以在 related results 中报告 EAGLE-3 等公开强基线；不能混入“matched single-chain”主表。

### 16.4 主指标

- accepted draft tokens/round；
- emitted tokens/target call；
- draft latency；
- verify latency；
- end-to-end tokens/s；
- speedup over vanilla；
- peak memory；
- 并发下 throughput/TPOT。

### 16.5 消融

- $K=4,8,16,32$；
- rank $=16,32,64$；
- future weight $lambda=0,0.25,0.5,0.75,1$；
- local vs global normalization；
- Viterbi vs survival objective；
- no hidden context / no token identity / shuffled suffix；
- pairwise first-order vs order-2；
- frozen DFlash vs optional joint fine-tuning；
- eager vs compiled vs Triton/CUDA Graph。

所有消融只改变一个变量，并使用相同 checkpoint/data split。

---

## 17. Phase 8：机制分析

### 17.1 Domino state pollution 实验

对同一 canonical block 同时运行：

- Domino on-policy GRU；
- Domino teacher-forced GRU。

画出首错前后 state distance 和 logit divergence，证明：

- state pollution 真实存在；
- 首错前两者相同；
- 首错后差异不会进一步改变该轮 accepted prefix。

### 17.2 SPH 为什么有效

分析 SPH 相对 local greedy 改变早期 token 的案例：

- 固定搭配；
- 代码括号/缩进/API 模式；
- 数学符号和公式模板；
- 标点与结束模式；
- 开放式多模态 continuation。

分别统计“改对”“改错”和“只改善 suffix 但未改善 prefix”。

### 17.3 校准

报告：

- predicted utility vs realized accepted length；
- prefix survival reliability diagram；
- ECE、Brier、NLL；
- temperature scaling 前后；
- $lambda$ 对首 token 风险/后续收益的权衡。

### 17.4 Failure cases

- gold 不在 top-$K$；
- pairwise consistency 偏好流畅但 target 不同的模式；
- 长距离依赖超出 first-order state；
- chat 多模态导致未来误导早期 token；
- head latency 在高并发下失去优势。

---

## 18. Phase 9：新颖性审计

每周至少一次更新 related-work matrix，记录：

- 方法使用何种 drafter；
- 是否候选 lattice/path search；
- 是否 learned transition；
- 是否 globally normalized；
- decoding objective 是 MAP、beam score 还是 EAL；
- 是否单链 target verification；
- drafting latency 和系统实现。

必须重点对照：DFlash、Domino、DSpark、DeLS-Spec、SpecFormer、DiffuSpec、D-PACE、VSD、PTP 和 speech Viterbi 工作。

论文允许的核心 claim 应保持为：

> 在 target-conditioned DFlash candidate lattice 上，使用并行全局能量模型和针对 longest-prefix utility 的精确单链 Bayes-risk decoding，在不进行 sequential neural rollout 或 tree verification 的情况下改善接受长度/吞吐。

若检索到完全相同的算法，立即缩小或调整 claim，不隐瞒最近工作。

---

## 19. Phase 10：论文生产

### 19.1 论文结构

1. Introduction：单链 parallel drafting 的 modal collision 与 locally greedy commitment；
2. Background：DFlash、Domino、longest-prefix acceptance；
3. Analysis：错误 suffix pollution 与 acceptance-relevant branch 的区分；
4. Method：candidate energy、global normalization、survival-DP；
5. Theory：DP 最优性、复杂度、single-chain/lossless 条件；
6. Systems：projection lookup、fused kernel、CUDA Graph；
7. Experiments：EAL、TPS、matched baselines；
8. Analysis/Ablation：coverage、calibration、failure modes；
9. Limitations：top-$K$ ceiling、first-order approximation、sampling支持。

### 19.2 必须有的图表

- 图 1：DFlash vs Domino vs SPH 数据流；
- 图 2：错误分支污染与 prefix survival 的区别；
- 图 3：candidate lattice 只在 draft 内部，target 仅验证一条链；
- 图 4：local greedy/Viterbi/survival 的两模式案例；
- 表 1：主 EAL/TPS；
- 表 2：并发与 latency breakdown；
- 表 3：matched head/decoder ablation；
- 表 4：$K$/rank/$\lambda$；
- 图 5：oracle ceiling 与 realized gain；
- 图 6：calibration/reach/hazard。

### 19.3 写作纪律

- 每个主张绑定一张表、一个图或一个 theorem；
- 不使用 DDTree trace 支撑单链结论；
- 不把 oracle coverage 表述为可实现收益；
- 不只报告接受长度而省略 latency；
- 不用“消除误差累积”这种过宽表述；
- 公开负面结果和适用边界。

### 19.4 复现包

最终 release 至少包含：

- 安装说明；
- checkpoint 下载说明和 revision；
- 训练/评测 configs；
- fixed prompt manifests；
- CPU/GPU tests；
- eager 和 optimized path；
- 原始 JSON 指标与作图脚本；
- 一条 4B smoke command 和一条完整 benchmark command。

---

## 20. 从当前状态开始的执行顺序

日历不能覆盖 gate。上一步未通过时，不因为“到了某一天”而启动后续昂贵工作。

### Step 1：Gate 0 数值与缓存等价性

- 固定同一 teacher-forced token sequence；
- 比较 full-prefix、cached-single、cached-block；
- 比较 eager 与 SDPA；
- 同 shape replay 必须 top-1 完全一致；
- 跨 shape/backend 的 top-1 分歧必须由实测 top-2 logit 误差包络解释。

### Step 2：Gate 1b same-anchor ceiling

- 重用 `10022338` immutable shards；
- 重建每个 prefix 并逐 token 校验 anchor/gold；
- 在统一 15-token horizon 比较 DFlash、K8/K16 oracle、Domino backbone、
  Domino on-policy GRU；
- prompt-cluster bootstrap 后作出 Gate 1 总 go/no-go。

### Step 3：development learnability probe

- 当前 75/9/12 prompt split；
- no-mixer、K16、rank32；
- local/absorbing-CRF 两种训练 normalization，seed 0/1/2；
- validation 选 epoch，test 只在选定 checkpoint 上评一次；
- 结果只能决定是否扩数据，不能进入论文主表。

### Step 4：正式数据与 Gate 2

- 建立去污染的 2k/5k/10k prompt 学习曲线；
- 同 scorer 比较 local greedy、local survival、global MAP、global survival；
- 加入 Domino head-off/on、DSpark/DeLS-style local head 与 DiffuSpec-style
  path search；
- 三 seed、prompt-cluster CI、分域、首 token、校准和 failure cases。

### Step 5：Eager 与系统 Gate 3

- 只在 Gate 2 通过后集成；
- 2048-token、独立 warmup、至少三次 counterbalanced repeats；
- component CUDA events、final/EOS censor、实际 emitted token；
- 先 matched eager，再 matched optimized backend；
- 吞吐 CI 通过后才写 Triton/CUDA Graph 和扩展 8B。

## 21. 6–8 周排期

- 第 1 周：candidate gate、head training、离线 gate；
- 第 2 周：eager integration、端到端 gate；
- 第 3 周：Triton/CUDA Graph、4B 完整结果；
- 第 4 周：8B 训练与主实验；
- 第 5 周：baseline、并发、消融；
- 第 6 周：机制分析、图表、初稿；
- 第 7 周：补实验、内部 review、related-work 更新；
- 第 8 周：复现包、附录、最终检查。

## 22. Definition of Done

只有同时满足下列条件，项目才达到投稿完成状态：

- proposed verifier 始终只接收一条 sequence；
- Qwen3-4B/8B 至少两个规模均完成；
- proposed 在主域平均 EAL 和端到端 TPS 上优于强单链 baseline；
- 增益在 matched 参数/latency 下超过普通 BiHead；
- survival objective 相对 Viterbi/local greedy 有独立增益；
- 所有结果有 CI、原始 JSON 和固定 manifest；
- eager 与 optimized 结果均可复现；
- target 输出正确性通过；
- 论文明确讨论 DiffuSpec 等最接近工作；
- README 能让另一位研究者从零跑通 4B smoke 和主表中的一项。

## 23. 立即下一步

立即执行三个有依赖关系的作业：

1. `scripts/slurm/gate0_target_equivalence.sbatch`；
2. `scripts/slurm/gate1b_same_anchor.sbatch`；
3. 仅在前两项成功后运行 `scripts/slurm/head_probe_factorial.sbatch`。

前两项关闭正确性和可比性缺口；第三项只测试“selector 是否显示可学习
信号”。在正式去污染训练数据建立以前，不启动 full head sweep、在线集成或
GPU kernel。
