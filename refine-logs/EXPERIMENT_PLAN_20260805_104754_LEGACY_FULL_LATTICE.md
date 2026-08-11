# Experiment Plan

**Problem:** Frozen DFlash 的完整 K16 lattice 含有可利用的候选特异跨位置信息，但旧 GCLS 的 candidate prepooling 与 additive encoding 只带来约 `+0.232` calibrated EAL。  
**Re-frozen thesis:** 完整格点上的 candidate-specific global interaction，可能比 axial candidate-summary mixer 获得更高 raw greedy accepted length；训练固定使用已有 smoothed Candidate-D-PACE `alpha=.5`。  
**Date:** 2026-08-04

## Claim Map

| Claim | Minimum convincing evidence | Linked blocks |
|---|---|---|
| C1a：coupled flat full-lattice mixer 优于 axial baseline | matched flat-additive 优于 axial-additive；不把差异单因果归于 prepool | B2 |
| C1b：显式 compatibility 在 full lattice 上进一步有益 | matched flat-compatibility 优于 flat-additive，且优于 axial | B2 |
| C1c：收益需要非因果 global evidence | winner 的 global-local、global-causal prompt-cluster CI 排除 0，三 seed 同方向 | B3 |
| System claim：offline EAL 能转化为吞吐 | untouched test raw EAL/harm 通过，端到端 TPS CI 为正 | B3, B4 |
| Anti-claim：只是更多参数、校准或更大 K | axial 同 D/H/L 且参数更多；raw/calibrated 分列；所有条件 K16/verifier 相同 | B2-B4 |

已删除的 claim：ARR/reach-aligned loss 与 base-prefix training safety。绑定容量救援中 ARR 在延长预算后失败，而 smoothed Candidate-D-PACE 的 additive/compatibility 两格均通过；后续不得恢复该 claim。

## Experiment Blocks

### Block 0: Preflight and Capacity Diagnosis — COMPLETE

- CPU deterministic tests：loss/support、float32 arithmetic、epoch-zero identity、scope perturbation、sorted/witnessed candidate0、artifact aggregation。
- GPU smoke R001：job 10132214，A40，完成 0:0；finite metrics，peak reserved 1.043 GiB。
- Initial ARR capacity R010-R013：job 10132235，三格均科学阴性。
- Binding rescue R014-R017：job 10132304；A/ARR fail，B/compat-smoothed pass，C/additive-smoothed pass；聚合 decision 为 `delete_arr_claim_and_refreeze_smoothed_cdpace`。
- Interpretation：implementation/environment/expressivity 足以继续 representation screen；capacity result 不是 held-out method evidence。

### Block 2: Matched Representation Isolation — CURRENT

- Claim tested：coupled flat full-lattice mixer 是否优于 axial topology baseline，以及 flat 内 multiplicative compatibility 是否改善 held-out utility。
- Train：从 Open-PerfectBlend 100K canonical 按 trainer 完全相同的 `sha256(seed + NUL + sample_id)` rule 预物化的 deterministic 25K prompt subset；prompt-set SHA256 固定为 `a3d25eba926ea8dc474d59b8a4bf3eabef6953d198bf2630d525344c3236fa73`。物化只减少重复 I/O，不改变 25K records、顺序、目标或训练预算。
- Selection：Phase-3 `validation_select`；不读取 `validation_gate` 或 reserved test；仅 development evidence。
- Conditions（全部 global、D128/H8/L2、FF4、dropout0、K16）：
  1. R030 `axial_additive_cdpace05`：candidate-prepooled axial topology baseline；
  2. R031 `flat_additive_cdpace05`：coupled flat full-lattice mixer treatment；
  3. R032 `flat_compat_cdpace05`：在 flat 上只改变 node encoder。
- Objective：length-normalized `candidate_dpace`，`alpha=.5`，base safety weight 0。
- Optimizer：AdamW，lr `6e-4`，wd0，warmup .04，clip1；batch64，9 epochs，seed0。
- Primary metric：raw prompt-balanced EAL delta vs the identical DFlash base。
- Secondary：repair、harm、oracle-gap recovery、first-token delta、minimum-domain delta、parameters、peak CUDA memory、runtime。calibrated output 仅作单独 diagnostic，不用于结构选择。
- Integrity：aggregator 必须验证 train prompt hash、train/validation counts、target/data/source hashes、optimizer/objective/budget 一致，只允许 mixer/node encoder/parameter count 不同。命名确定性初始化保证 R031/R032 全部同名参数逐位相同，R030/R031 的共同输入编码参数逐位相同，且模型构造不改变 shuffle RNG。
- Frozen selection：
  - `flat_additive > axial_additive` 才支持 coupled flat-mixer claim；否则 scientific stop。该比较不支持“prepool 是唯一原因”。
  - 在 flat-mixer comparison 通过时，`flat_compat > flat_additive` 才选择 compatibility；否则选择 additive 并删除 compatibility。
  - winner 还必须 raw delta `> +0.285`，harm `<= axial harm`，first-token 不比 axial 低超过 .001，才进入 B3。
- Limitation：147-prompt validation 方差较大，本 block 只作 architecture development，不作 claim-grade CI。

### Block 3: Matched Scope and Seed Confirmation

- Trigger：B2 stage gate 通过。
- Train：冻结 winner；先用 OPB-100K 完整 train，之后冻结 prompt-disjoint calibration/test manifest。
- Conditions：local/causal/global × seeds0/1/2；architecture、init、data、steps、selection rule 完全匹配。
- Baselines：DFlash、historical GCLS-v1 axial global；same-anchor Domino 仅作外部强基线。
- Primary：raw prompt-balanced EAL；paired global-local/global-causal prompt bootstrap。
- Secondary：repair、harm、first-token、per-domain、oracle gap、seed spread。
- Gate：三 seed 同方向，global-local/global-causal CI 排除 0，raw gain 复现并显著超过现有 `+0.285`。
- Test contract：hyperparameters、checkpoint rule 和 calibration rule 冻结后，untouched test 只打开一次；test harm 单侧 95% cluster UCB `<=5%`。

### Block 4: System Value and Failure Analysis

- Trigger：B3 raw test gate 通过。
- Systems：DFlash、raw winner、separate KEEP_BASE diagnostic、same-protocol Domino。
- Metrics：head latency、draft/verify/total latency、TPS、accepted length、peak memory、break-even；按 gold rank、base gap、position、domain 分层 repair/harm。
- Gate：raw winner 的端到端 TPS prompt-cluster CI 为正。
- Failure interpretation：EAL 正而 TPS 负时，只保留 offline selector claim；KEEP_BASE 不替代 raw result。

## Run Order

| Milestone | Runs | State | Decision |
|---|---|---|---|
| M0 preflight | R000-R002 | complete | environment and implementation pass |
| M1 ARR diagnosis/rescue | R010-R017 | complete | ARR deleted; smoothed Candidate-D-PACE frozen |
| M2 representation | R030-R033 | current | choose flat compatibility/additive or stop |
| M3 scopes/seeds | R040-R048 | blocked by M2 | confirm non-causal global effect |
| M4 independent/system | R050-R053 | blocked by M3 | harm/TPS gates |

## Compute and Data Budget

- M2：先用 CPU 一次性物化精确 25K subset，逐 shard 在单次读取中验证 source SHA256，断言实际 record prompt set 等于 logs 的 99,356，并生成精确 25K manifest；materializer 异常统一 exit2。只有物化 job exit0 且 metadata/prompt hash 审计通过后，才手工提交 3 个 development GPU runs，不用 `afterok` 隐藏失败。GPU 使用已验证 A40 `debug`，单任务 30 min limit，数组最多并发2。
- M3：胜出后才提交，预计 15-25 GPU-h。
- Data：OPB-100K train canonical 64 GiB；Phase-3 development validation 1.5 GiB。
- 当前最大统计瓶颈：`validation_select` 只有 147 prompts，不能支持正式 harm/CI claim。

## Binding Stop Rules

- representation gate 失败后不追加 slot/CRF/GRU/RL/更大模型 rescue。
- compatibility 失败但 flat-additive 通过：简化为 additive，不寻找新的 interaction module。
- global scope 失败：删除 global-evidence claim。
- independent test 不复现：报告 negative/inconclusive，不重开 test 调参。
- latency 不通过：停止 system claim，不能用 oracle 或 calibrated variant 掩盖。

## Checklist

- [x] CPU and GPU preflight
- [x] Capacity failure causally diagnosed
- [x] ARR/safety claim deleted and objective re-frozen
- [x] Representation cells and deletion rules frozen
- [ ] R030-R032 artifacts complete and R033 aggregate reviewed
- [ ] Matched scopes and three seeds
- [ ] Prompt-disjoint calibration/test
- [ ] Positive end-to-end throughput
