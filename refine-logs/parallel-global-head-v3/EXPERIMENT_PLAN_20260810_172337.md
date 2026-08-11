# Experiment Plan：PCLD-16R

**Problem**：在保留 DFlash full16 一次并行单链生成的前提下，用轻量 global noncausal head 把 fixed/dynamic accepted length 与最终同栈 A40 SGLang throughput 都提高到 released Domino 的至少 `1.15x`。  
**Method Thesis**：将离线 clean target 的 16 个 predictive hidden residual 蒸馏到 candidate-conditioned full-block mixer；线上只使用 ordinary DFlash full16 features，一次同时输出唯一 16-token chain。  
**Date**：2026-08-10

## Immutable execution contract

- 一次输入完整 `H[B,16,2560]` 与 pure-base candidate lattice；
- 每个输出 query 无 causal mask 地读取全部 `16×16=256` candidate nodes；
- 一次调用输出 `S[B,16,16]`，一次逐位置 argmax 得到唯一 `[B,16]` chain；
- Top16 只作为每位置 candidate axis；
- 禁止 GRU、selected-token feedback、serial target decode/seed、Jacobi/iteration、beam/tree/trie/forest/multipath；
- 离线 teacher 可以使用 target，生产 head 除 ordinary final verifier 外不能增加 target forward；
- 冻结主架构 `D256/H8/L2/FFN1024`，精确 `3,826,688` 参数；任何结构、width、loss、temperature 或 schedule 变更都必须开新 workstream，不能救援本路线的失败门。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1（primary mechanism）full-block global context 与 predictive latent supervision 能在无 sequential feedback 时安全修复一条并行 chain | 这是 accepted length 提升的核心，不是参数量或候选树 | disjoint P2 中三个 global seeds 均改善 base；global 相对 matched local/no-latent 各 `ΔEAL>=0.15` 且 paired 95% CI lower `>0`；recovery/J2/harm/domain 全过门 | B0, B1 |
| C2（primary system）该轻量单链 head 的 EAL 增益能转成 `>=1.15x` Domino 端到端吞吐 | 用户最终目标是实际加速，不是 surrogate accuracy | sealed fixed/dynamic EAL 均 `>=1.15x` same-job Domino；完整 cycle 满足 EAL+1 预算；同栈 A40 SGLang paired TPS ratio 95% CI lower `>=1.15` | B2, B3 |
| A1（anti-claim）收益不是额外 target inference、serial generation、树搜索、更多候选或不公平计时造成 | 防止方法漂移并保证可部署性 | dataflow/API fail-closed；global/local/no-latent matched；K16 one-chain 固定；complete eager 与同栈 SGLang 公平 profile | B0, B1, B3 |

## Paper Storyline

- **Main paper must prove**：P2 global/local/no-latent 机制隔离；P3 sealed fixed/dynamic 主结果；P4 同栈 TPS。
- **Appendix can support**：P0/P1 mechanics/capacity/cost；位置、margin、domain、first-error failure anatomy。
- **Experiments intentionally cut**：PCA hard gate、hidden RMSE threshold、D512/width sweep、loss/temperature grid、post-hoc schedule、serial target、GRU、iteration、beam/tree/multipath、与 EAL 无关的运行时哈希检查。
- **Frontier necessity**：predictive latent distillation 是核心监督形式，但线上刻意不引入额外 frontier model；用 matched no-latent 而非装饰性 LLM/RL 实验检验其必要性。

## Experiment Blocks

### Block 0：Mechanics、teacher authority、capacity 与公平 eager cost

- **Claim tested**：实现严格符合 full16 global one-chain；teacher rows/score authority 正确；3.827M head 可优化且成本可信。
- **Why this block exists**：先排除 off-by-one、near-tie 错标、zero-init 假梯度、target leakage 和不公平 latency，再判断方法本身。
- **Dataset / split / task**：
  - 32 个跨域 R047 train blocks 做 P0 sidecar/GPU mechanics；
  - `artifacts/manifests/japd16_r047_split_20260810.json` 的冻结 capacity group：512 prompts、每 prompt 一个 block，strict multi-repair denominator 已知为 411；
  - source rollout 固定为 `artifacts/canonical/r047_anchor_t4_train_10164718`。
- **Compared systems**：PCLD global；pure DFlash base；released Domino eager。
- **Decisive metrics**：
  - target batched row 与 16-step offline recurrence 的 row0/row15/token parity；
  - direct target candidate scores 与 FP32 residual cancellation tolerance；
  - zero-init 16-position selected-token mismatch `=0`；
  - zero-init remote `g/Jacobian` sensitivity、deterministic nonzero-`U` remote score sensitivity；
  - `grad(U)` step0 finite/nonzero、upstream step0 zero、one-update upstream finite/nonzero；
  - production mask `None`、forbidden scopes/fields fail closed、参数数 `3,826,688`；
  - P1 candidate agreement、oracle-gap recovery、harm、strict J2、prompt-balanced EAL；
  - complete eager p50/p90/mean、peak memory。
- **Setup details**：
  - new sidecar stores target hidden rows, authoritative gathered teacher scores, target full-vocab top1/top2 stability data and frozen `epsilon_num`; canonical rollout remains immutable；
  - model production `forward` accepts only ordinary DFlash hidden/base candidate tensors/LM-head candidate rows，拒绝 gold/target hidden/teacher score；
  - capacity recipe：seed0、batch8、8,000 updates、AdamW `3e-4`、WD `1e-2`、warmup200、cosine至 `3e-5`、clip1.0、eval250、internal same-set EAL strict-greater 选 best、tie 取早；
  - A40 BF16 batch1 complete profile 必须计入 base vocab GEMM、FP32 Top16/LSE、candidate/LM-head gathers、global head、residual dot、argmax/gather；Domino 使用同 hidden/weight 与 released eager correction。
- **Success criterion**：所有 mechanics receipts PASS；P1 candidate agreement `>=99%`、oracle recovery `>=95%`、harm `<=1%`、J2 `>=99%`；complete eager `<=1.20x` Domino。
- **Failure interpretation**：mechanics/parity/profile 失败只修已定位实现错误后原样复跑；capacity 科学门失败即关闭 frozen PCLD-16R，不允许 width/loss/schedule sweep。
- **Table / figure target**：Appendix implementation/capacity/cost table。
- **Priority**：MUST-RUN。

### Block 1：Disjoint matched mechanism isolation

- **Claim tested**：remote full-block context 与 latent residual supervision 分别带来可泛化 accepted-prefix 增益。
- **Why this block exists**：这是 25K/100K 前最高价值的主机制 falsifier。
- **Dataset / split / task**：冻结 R047 prompt split `1589 fit / 199 internal-select / 199 untouched diagnostic`；三组 prompt 严格不交。diagnostic 在三臂九个 checkpoint 全冻结后一次性共同打开。
- **Compared systems**：global PCLD；parameter-matched local-mask PCLD；matched global no-latent（`alpha=0`）；外部表同时报告 pure base 与 released Domino。
- **Metrics**：prompt-balanced EAL、base-to-gold-oracle recovery、strict J2、harm、chat/code/math EAL；global-local 与 global-no-latent prompt-paired bootstrap 10,000 draws，seed `20260810`。
- **Setup details**：每臂 seeds `0/1/2`；batch16；10,000 updates；AdamW/LR/WD/warmup/cosine/clip 与 B0 相同；每 250 updates 在 internal-select 评估；checkpoint 仅按 internal-select EAL strict-greater 选择、tie 取早；同 seed 的三臂共享 prompt/block batch order。local 只改变 attention mask，参数数完全相同；no-latent 只令 `alpha=0`。
- **Success criterion（全部满足）**：
  - 每个 global seed diagnostic EAL `>= same-job base + 0.30`；
  - 三 seed aggregate oracle-gap recovery `>=20%`，strict J2 `>=40%`；
  - 每 seed harm `<=1%`，chat/code/math 各 `>=base`；
  - 先对每 prompt 跨 seeds 求均值，再 bootstrap；global-local 与 global-no-latent 均 `ΔEAL>=0.15` 且 95% CI lower `>0`。
- **Failure interpretation**：任一门失败，C1 不成立并停止该冻结路线；same-set fit、更多参数或 train loss 不能替代 diagnostic。
- **Table / figure target**：Main Table 1；global/context 与 latent contribution paired plot。
- **Priority**：MUST-RUN，仅 B0 全过后。

### Block 2：Conditional scale 与 sealed fixed/dynamic acceptance

- **Claim tested**：已通过的机制随数据扩展，达到明显超过 Domino 的 accepted length，并转移到 dynamic rollout。
- **Dataset / split / task**：OPB 25K prompts；条件式 100K prompts；sealed fixed 与 fresh dynamic manifests 在读取 outcome 前排除 train/select/diagnostic IDs 与 normalized-text near duplicates。
- **Compared systems**：frozen global PCLD、released Domino、pure DFlash base；不再运行 local/no-latent 大规模臂。
- **Metrics**：prompt-balanced fixed/dynamic EAL、EAL ratio、domains、paired bootstrap、harm/J2、learning slope。
- **Setup details**：25K seed0，固定 recipe；若过门，100K seeds0/1/2 共享一个 offline label sidecar；deployment checkpoint/seed 只按 internal-select 预先冻结，final outcome 不参与选择。
- **Success criterion**：
  - 25K：held-out EAL `>=7.8`、相对 P2 slope 为正、P2 两个 matched-control CI lower 仍为正、三域不退化；若 EAL `<7.55` 直接停止；
  - 100K opening：每 seed internal fixed EAL `>=8.3254859086`，并优先达到设计目标 `9.0`；
  - sealed final：deployment seed 的 fixed 与 dynamic EAL 各 `>=1.15x` same-job Domino，三域不退化；三 seed 全报告，禁止 final seed shopping。
- **Failure interpretation**：25K 失败是 scale slope 不足；fixed 过而 dynamic 失败是 rollout shift 未解决；任何失败均不进入 SGLang claim，也不授权结构漂移。
- **Table / figure target**：Main Table 2 与 data-scaling curve。
- **Priority**：MUST-RUN，仅 B1 全过后。

### Block 3：Joint EAL–cycle feasibility 与 same-stack SGLang

- **Claim tested**：accepted-length 增益在真实服务路径中转化为至少 15% 吞吐优势。
- **Dataset / split / task**：与 sealed final 相同的固定 prompt/token manifest；不按 timing/output 删除样本。
- **Compared systems**：同 checkout、同 backend、同 batch1/workload/A40 的 frozen PCLD 与 released Domino；pure base 为 secondary reference。
- **Metrics**：完整 cycle p50/p90/p99、EAL、`(EAL+1)/cycle_time`、peak memory；paired prompt TPS ratio 与 95% CI；target-only greedy output parity。
- **Setup details**：先用 eager complete cycle 检查必要条件：

  `T_P/T_D <= (EAL_P+1)/(1.15*(EAL_D+1))`。

  EAL `9.0` 且 Domino `7.2395529640` 时 PCLD cycle 必须 `<=1.0553548490x`。必要门通过后才集成同一冻结 head 到 SGLang；两边相同 graph/Triton 优化等级，固定 warmup/order，ABBA 多次冷启动。
- **Success criterion**：sealed fixed/dynamic acceptance 仍过门；lossless/stable output parity；paired SGLang TPS ratio 95% CI lower `>=1.15`。
- **Failure interpretation**：EAL 通过但系统门失败，只允许优化同一冻结 head 的 kernel/static buffer；不得改变候选、增加 serial/iteration/multipath；优化后完整公平重测。
- **Table / figure target**：Main Table 3 与 EAL–TPS Pareto。
- **Priority**：MUST-RUN，仅 B2 全过后。

### Block 4：Failure anatomy（派生分析）

- **Claim tested**：解释最终方法仍遗漏哪些位置/域/margin，以及 controls 为什么失败或成功。
- **Dataset / split / task**：只读取已完成的 P2/P3 predictions，不新增训练、不参与 checkpoint 选择。
- **Compared systems**：global/local/no-latent/base/Domino 的冻结输出。
- **Metrics**：first-error position、target margin、gold rank、support horizon、first/second repair、domain buckets、harm examples。
- **Success criterion**：完整报告，不设主路线救援门。
- **Failure interpretation**：若某机制 control 等价，收窄论文 claim；禁止据此 post-hoc 调参。
- **Table / figure target**：Appendix heatmap/case table。
- **Priority**：NICE-TO-HAVE。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Estimated Cost | Main Risk |
|---|---|---|---|---:|---|
| M0 | CPU/data/model contract | PCLD000–002 | full16/one-chain/no-leakage、exact params、loss/support/math tests | CPU only | implementation drift |
| M1 | A40 mechanics/sidecar/profile/capacity | PCLD003–006 | P0 receipts、complete cost、P1 four science gates | `0.5–3 A40 h` | target row/numerical authority or underfit |
| M2 | Disjoint mechanism falsifier | PCLD020–023 | all nine arms complete；all B1 gates | `3–12 A40 h` | online lattice cannot infer clean residual |
| M3 | Conditional scale | PCLD030–045 | 25K slope then 100K opening then sealed fixed/dynamic | `15–60 A40 h` | transfer plateaus or rollout shift |
| M4 | E2E system | PCLD050–051 | joint EAL-cycle necessary gate；TPS CI lower `>=1.15` | `5–15 A40 h` | global head/kernel overhead |
| M5 | Paper support | PCLD060 | derived failure analysis only | `<0.2 A40 h` | claim narrowing only |

## Compute and Data Budget

- **Fail-fast budget through M1**：`<=3 A40 GPU-hours`；P1 失败立即停止。
- **Budget through decisive P2**：累计约 `3.5–15 A40 GPU-hours`。
- **Conditional total**：约 `24–90 A40 GPU-hours`；大部分只在前门成功后发生。
- **Data preparation**：P0/P1 复用 R047 canonical rollout 与冻结 manifest，但必须新建 PCLD target-hidden sidecar；25K/100K 只在授权后收集。
- **Storage**：canonical rollout 不改写；sidecar 版本化存储 offline target hidden/teacher/numerical support，production batch schema 不含这些字段。
- **Human evaluation**：不需要。
- **Biggest bottleneck**：在线 DFlash full16 lattice 是否包含足够的跨 prompt 信息来预测 clean `T-H`，而不是参数量或 Top16 coverage。

## Risks and Mitigations

- **Teacher headroom 被误当 student result**：teacher `10.5972` 只作 ceiling；P1/P2 分别验证 capacity 与 transfer。
- **Near-tie 数值错标**：train-only BF16/FP32 full-vocab top1 agreement 与 `margin>2epsilon_num`；所有 loss 共用连续 prefix support。
- **Zero-init 阻断 upstream**：P0 明确验证 step0 `grad(U)` 与 one-update 后 upstream gradient，不靠训练失败后猜测。
- **Easy positions 淹没主目标**：decisive metrics 使用 EAL、oracle-gap recovery、strict J2 与 harm，而非只看 position-micro accuracy。
- **不公平 latency**：complete path 计入 vocab GEMM/Top16/LSE/gathers/residual dot/argmax；最终同栈、同优化等级。
- **结果后漂移**：每阶段硬门与 STOP 预注册；禁止 D512、loss/schedule sweep、serial/iteration/tree 救援。

## Final Checklist

- [x] 主机制、主 accepted-length 和系统 claim 均有对应 block
- [x] global context 与 latent supervision 由 matched controls 分别隔离
- [x] simplicity 由精确 3.827M frozen architecture 和无 rescue rule 保护
- [x] frontier supervision 的必要性由 no-latent control 检验，不增加在线 frontier inference
- [x] nice-to-have 派生分析不阻塞主目标
- [x] P0/P1/P2/P3/P4 单调授权，前门失败不继续烧卡
- [x] fixed、dynamic、EAL+1 cycle 与最终 SGLang 均有独立硬门
