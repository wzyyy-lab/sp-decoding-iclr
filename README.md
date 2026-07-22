# DFlash Survival Path Head

这是一个面向 DFlash 单链 speculative decoding 的研究工作区。当前方法在
DFlash top-$K$ candidate lattice 上建立带吸收式 `OTHER` 的全局 prefix-CRF，
并通过 acceptance-aligned dynamic programming 选择唯一一条 draft sequence。
Target 仍使用普通 longest-prefix verification，不做 tree verification。

## 从这里开始

- [完整执行计划](PLAN.md)：从环境、baseline、canonical 数据、训练、推理集成、GPU kernel、完整实验到 ICLR 稿件的逐步方案。
- [方法说明](docs/method.md)：为什么错误 suffix 污染不等于额外接受长度损失，以及 SPH/CRF/survival-DP 的推导。
- [结果注册表](docs/results_registry.json)：机器可读的 evidence tier、artifact hash 与允许支持的主张。
- [核心实现](src/sph/survival_path_head.py)：low-rank edge scorer、chain CRF、Viterbi 和 survival-DP。
- [单元测试](tests/test_survival_path_head.py)：包括 brute-force 最优性验证。

## 当前状态（2026-07-22）

当前只有两项研究实验可用于分析；其余运行都只是环境、数据管线或反向传播
smoke，禁止进入论文表格或支持方法有效性主张：

- `10022338`：96 prompts、768 canonical blocks 的候选上限实验。结果支持
  **Gate 1a（候选可用性）通过**，但还没有完成与 Domino 的同锚点 Gate 1b；
- `10022436`：96 prompts 的 DFlash/Domino eager 基线实验。它是真实基线结果，
  但由于 256-token、单次运行和数值等价性尚未闭环，仍是 development-grade，
  不是最终论文 benchmark；
- `10022278`、`10022310`、`10022330/43`、`10022412`、`10022468` 均标为
  `non-evidence smoke`；特别是 `10022468` 的 6/6-block 训练只验证管线；
- absorbing-OTHER prefix-CRF、局部 control、MAP 和 survival decoder 已统一到
  同一套 edge scores；20 个 CPU 单元测试通过；
- 当前还没有任何可声称有效的 trained-SPH EAL，也没有正式 latency 结果；
- Qwen3-4B/8B target 与 DFlash 权重均在本地，4B Domino 权重也已下载；
- Domino 源码与独立环境位于 `third_party/Domino/`；
- 每次实验结论和作业号记录在 [experiment log](docs/experiment_log.md)。

运行 CPU 测试：

```bash
module load anaconda3
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

## 目录

```text
.
├── PLAN.md
├── docs/
│   ├── method.md
│   └── experiment_log.md
├── papers/                 # 33 个 PDF，全部保留
├── scripts/                # manifest、采集、分析、baseline 与 Slurm 入口
├── src/sph/
├── tests/
└── third_party/
    ├── dflash/             # 官方 DFlash git 仓库
    └── Domino/             # 官方 Domino git 仓库与 .venv
```

训练数据、checkpoint、profile 和 benchmark 输出后续统一放入被 `.gitignore` 排除的 `artifacts/`、`checkpoints/` 和 `outputs/`。

## 研究边界

- proposed method 最终只验证一条 draft sequence；
- candidate lattice 只存在于低成本 draft-side reranking 内部；
- 不使用 DDTree trace 训练或证明当前方法；
- 第一阶段冻结 DFlash，只训练 SPH；
- 普通 BiHead 是必须比较的 baseline，不是被预先排除的方案；
- DiffuSpec-style path search 是必须比较的最接近 baseline；
- 只有 EAL 收益转化为 matched-hardware 的端到端 TPS 收益，才继续作为 ICLR 主线。

## 2026-07-21 清理记录

已删除被当前分析否定的 ReFlash/PlanDomino 文档与原型、DDTree trace probes 和 Python caches。所有论文 PDF 均未删除；`papers/` 中的 33 个 PDF 保持原文件名，DFlash/Domino 核心论文路径不变。

下一步严格按门控顺序执行：

1. `gate0_target_equivalence.sbatch`：固定 token sequence，诊断 full-prefix、
   cached-single、cached-block、eager/SDPA 的 target logit 等价性；
2. `gate1b_same_anchor.sbatch`：在 `10022338` 的完全相同 anchors 上比较
   DFlash top-1、K8/K16 oracle、Domino backbone 与 Domino on-policy GRU；
3. 前两项通过后，`head_probe_factorial.sbatch` 才运行三 seed 的小规模
   learnability probe。该 probe 仍明确标记为 development，不能代替正式数据。
