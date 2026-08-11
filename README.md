# PARC-16：DFlash 全局并行单链纠错

本仓库是面向 DFlash / Domino speculative decoding 的完整研究工作区。当前主线是
**PARC-16（Sealed-Heldout Fixed-Reference Parallel Correction）**：DFlash 一次并行
产生完整 16-token provisional block，轻量 head 在所有 16 个位置和每位置 Top-16
候选之间做一次全局、非因果混合，然后一次同时输出唯一一条 16-token 序列。

在线路径严格不包含 Domino 式 GRU、自回归 token feedback、串行 target seed、迭代
refinement、beam/tree/trie/forest 或多路径验证。Target model 只用于离线监督和普通
speculative verifier。

## 当前主线

- Head：D256 / H8 / L2 / FFN512，新增参数 **2,438,400**，约为 537.427M
  DFlash draft model 的 **0.454%**。
- 输入：完整 `H[B,16,2560]`、每位置 base Top-16 logits/IDs、candidate embedding、
  anchor embedding。
- Mixer：256 个 position-candidate action nodes 经过无 causal mask 的 full
  self-attention；每个位置在选 token 前能看到完整 16-position block。
- 输出：一次 `[B,16,16]` scores，加一次逐位置 argmax，得到唯一 `[B,16]` chain。
- 训练：90K prompts、独立 5K validation、封存约 5K held-out；180K joint
  DFlash+PARC updates。训练集 EAL 只作诊断，validation 只选 checkpoint。
- 硬目标：held-out fixed EAL、dynamic EAL 与最终 A40 SGLang TPS 均至少达到
  same-job released Domino 的 `1.15x`。

权威边界见 [用户约束合同](refine-logs/parallel-global-head-v1/USER_CONSTRAINT_CONTRACT.md)，
最终方法见 [FINAL_PROPOSAL](refine-logs/parallel-global-head-v4/FINAL_PROPOSAL.md)，
完整复现入口见 [实现与 trace 指南](docs/PARC16_IMPLEMENTATION_GUIDE.md)。

## 当前进度（2026-08-11）

- PARC 架构、fixed-reference gain/harm objective、full16 数据收集、正式 trainer、
  validation/断点恢复逻辑已经实现。
- 18 项 PARC focused tests、Python compilation 和两个 Slurm launcher syntax check
  均通过；独立 experiment-bridge review 对 M1/M2 均给出 GO。
- 正式 16-way A800 数据任务：Slurm `10169014`，当前 `PENDING (Priority)`。
- 唯一 180K-step 正式训练：Slurm `10169018`，以 `afterok:10169014` 依赖排队。
- 当前没有正式 validation 或 held-out 效果数字。旧容量集的 EAL 9.5254 不是验证
  证据，不能用于声称超过 Domino。

实时记录见
[FORMAL_RUN_STATUS](refine-logs/parallel-global-head-v4/FORMAL_RUN_STATUS.md)。

## 代码入口

| 内容 | 文件 |
|---|---|
| PARC-16 head、全局 action-node mixer、gain/harm loss | `src/sph/parc.py` |
| 256-node full noncausal attention primitive | `src/sph/parallel_global_candidate_fusion.py` |
| 正式数据 catalog、prompt/block sampler、验证指标 | `src/sph/parc_training.py` |
| 270K reserve 的 train/validation/heldout 预划分 | `scripts/build_parc16_split.py` |
| full16 target/DFlash/Domino trace 收集 | `scripts/collect_parc16_data.py` |
| 180K joint DFlash+PARC 正式训练与 validation | `scripts/train_parc16.py` |
| 16-way A800 trace/materialization launcher | `scripts/slurm/parc16_full_data.sbatch` |
| 正式训练/精确 resume launcher | `scripts/slurm/parc16_joint_train.sbatch` |
| PARC focused tests | `tests/test_parc.py`, `tests/test_collect_parc16_data.py`, `tests/test_parc_training.py`, `tests/test_build_parc16_split.py` |

`scripts/`、`src/sph/` 和 `tests/` 同时保留此前 PGCF、JAPD、PCLD、GFPR、PLC、
R048–R056 等路线的完整代码和负面结果诊断，便于追溯为什么最终收敛到 PARC-16；
这些历史路线不属于当前在线架构。

## 环境

当前验证环境：Python 3.11.15、PyTorch 2.9.1+cu128、Transformers 4.57.1、
Safetensors 0.7.0、NumPy 2.4.3。安装本仓库实验依赖：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[experiment]'
```

官方外部仓库没有复制进本仓库：

```bash
git clone https://github.com/z-lab/dflash.git third_party/dflash
git -C third_party/dflash checkout 94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756

git clone https://github.com/jianuo-huang/Domino third_party/Domino
git -C third_party/Domino checkout 930e5cd823f4bbbaa82ae150acad03928a3a859f
```

模型目录需提供 `Qwen3-4B`、`Qwen3-4B-DFlash-b16` 和
`Qwen3-4B-Domino-b16`。集群路径通过两个 Slurm 文件顶部的 `PROJECT`、`ASSETS`、
`PYTHON` 配置。

## 最小代码验证

这些检查只防止实现错误，不是效果实验：

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  tests/test_parc.py \
  tests/test_collect_parc16_data.py \
  tests/test_parc_training.py \
  tests/test_build_parc16_split.py

.venv/bin/python -m py_compile \
  src/sph/parc.py src/sph/parc_training.py \
  scripts/build_parc16_split.py scripts/collect_parc16_data.py \
  scripts/train_parc16.py
```

## 文档索引

- [完整实现、trace schema 与运行命令](docs/PARC16_IMPLEMENTATION_GUIDE.md)
- [用户不可变约束](refine-logs/parallel-global-head-v1/USER_CONSTRAINT_CONTRACT.md)
- [当前问题定义](refine-logs/parallel-global-head-v4/PROBLEM_ANCHOR.md)
- [最终方法设计](refine-logs/parallel-global-head-v4/FINAL_PROPOSAL.md)
- [正式实验计划](refine-logs/parallel-global-head-v4/EXPERIMENT_PLAN.md)
- [实验 tracker](refine-logs/parallel-global-head-v4/EXPERIMENT_TRACKER.md)
- [full16 几何与 reserve 修订](refine-logs/parallel-global-head-v4/FULL16_GEOMETRY_AND_RESERVE_AMENDMENT_20260810.md)
- [最新任务状态](refine-logs/parallel-global-head-v4/FORMAL_RUN_STATUS.md)
- [完整历史实验日志](docs/experiment_log.md)
- [机器可读结果登记](docs/results_registry.json)
- [研究产物清单](MANIFEST.md)

## 数据与权重

生成的 trace、训练数据、checkpoint、日志和模型权重可能达到数百 GB，均由
`.gitignore` 排除。仓库提交的是生成它们所需的完整代码、配置、schema、测试、设计
文档和进度记录，不提交不可移植的大型二进制产物或密钥。
