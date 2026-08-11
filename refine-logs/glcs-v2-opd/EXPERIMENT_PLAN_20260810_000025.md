# GFPR Experiment Plan

**Problem:** Domino在exact B16 held-out上的接受长度只有7.23955，而候选oracle存在约3 token以上空间；既有固定anchor训练不能泛化。  
**Method Thesis:** 在真实policy产生的verification anchors上，只保护当前accepted prefix并修current first rejection，同时把position 0纳入同一Domino causal head。  
**Date:** 2026-08-10  
**Hard target:** fixed exact-runtime EAL ≥8.325，true dynamic-rollout EAL ≥1.15× released Domino，通过harm gates后再做SGLang。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1 Primary | policy-induced reachable-frontier replay能解决static训练不泛化 | 同head、同prompt、同budget下dynamic all-16显著胜过released与fixed control；最终fixed EAL ≥8.325且dynamic ratio ≥1.15× | B1, B2, B3 |
| C2 Supporting | 收益来自正确的状态/目标，而不是更大模型 | 50.8M现有Domino head加一个position-0 scalar即可获得主收益；position-0与dynamic-anchor删除实验能解释增益 | B1, B3 |
| Anti-claim | 收益只是candidate扩大、更多参数、验证泄漏或少数prompt大涨 | A–C均为同一full-vocab action space；prompt-disjoint splits；paired bootstrap和gained/lost/harmful-prompt统计通过 | B0, B1, B2 |

## Paper Storyline

- Main paper必须证明：真实anchor加current frontier加all-16在未见prompt上达到硬EAL目标；收益转化为真实rollout和端到端吞吐。
- Appendix支持：K16/K17 oracle、position-0域分解、margin/harm分解、collector语义测试。
- 暂时切掉：tree search、sampling、首错后suffix KL、独立safety gate、大型candidate attention、100K无条件网格。
- Stage D的adapter/LoRA不是当前paper claim；仅在GFPR已有显著信号但容量不足时打开。

## Fixed Assets and Splits

- Target: /hpc2hdd/home/zwang668/sp decoding/hf_assets/models/Qwen3-4B
- Released Domino: /hpc2hdd/home/zwang668/sp decoding/hf_assets/models/Qwen3-4B-Domino-b16
- Phase3 source: artifacts/canonical/qwen3_4b_phase3_tier1_10035436
- Historical full-16 runtime cache: artifacts/canonical/plc_runtime_phase3_10158769
- Phase3 train: 2,000 prompts，chat/code/math为666/667/667。
- validation_select: 150 manifest prompts，实际完整collection为147 prompts、1,175 blocks；只用于checkpoint selection。
- validation_gate: 150 prompts；路线和超参冻结前不查看其适配结果。
- Scale source: artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770，优先nested 16K，再到32K。
- Reserved test: artifacts/manifests/phase3_reserved_test_v3.jsonl，仅在最终方法和系统配置冻结后使用。

## Experiment Blocks

### Block B0: Gate A — Exact Semantics and Oracle

- Claim tested: full-16实现是exact released identity；position 0与Top-16确实提供足够headroom。
- Why: 若identity、bonus或oracle错，任何训练结果都不可解释。
- Dataset: phase3 validation_select，先32-prompt smoke，再全部147 prompts。
- Systems: released exact runtime；all-16 DFlash Top-16 oracle；K17 oracle；K16 oracle。
- Decisive metrics:
  - released token和accepted-length reproduction；
  - released EAL；
  - frozen-position-0 versus all-16 oracle EAL；
  - K16/K17 oracle；
  - position-0 gold-in-Top16与可恢复增量；
  - r+1、r=16 bonus、GRU reset invariant。
- Setup:
  - 新collector一次生成同一路径的full16 hiddens和scores；
  - 不把不同forward shape的旧cache拼在一起。现有preflight已观察到相邻hidden最大绝对差约0.75，拼接只能做粗诊断，不能作为exact Gate A；
  - full-vocabulary bf16/float32加法与argmax遵循released runtime contract。
- Success:
  - step-0逐token identity 100%；
  - released EAL复现7.23955，允许仅由明确metric split差异解释的数值差；
  - all-16 oracle >8.325；
  - 若未来K16部署，使用对应frozen scorer重算的K16 oracle ≥8.825。
- Failure: 修collector/runtime alignment；不开始训练。
- Paper target: Appendix oracle/semantic table；position-0 motivation figure。
- Priority: MUST-RUN。

### Block B1: Gate B — 2K Prompt Causal Screen

- Claim tested: dynamic anchors与all-16 correction是held-out增益的因果来源。
- Why: 这是主假设的最便宜有效检验。
- Dataset:
  - train: phase3全部2,000 train prompts；
  - selection: validation_select；
  - validation_gate保持sealed。
- Compared systems:
  1. Released Domino，不训练；
  2. Fixed-15：fixed canonical offsets，现有位置1–15 head；
  3. Dynamic-15：actual r+1 anchors，位置1–15 head；
  4. GFPR-16：actual r+1 anchors，同head覆盖位置0–15。
- Metrics:
  - fixed exact-runtime prompt-balanced EAL；
  - true dynamic rollout prompt-balanced EAL；
  - paired delta和10,000次prompt-cluster bootstrap 95% CI；
  - total gained/lost accepted tokens、harmful prompts；
  - first-token acceptance、frontier repair、full-block rate；
  - chat/code/math及context-length分解。
- Setup:
  - 只训练prefix_gru、embed_proj和GFPR的alpha0；
  - alpha0 zero-init，head从released权重开始；
  - current-frontier hinge，lambda_break=1.0、lambda_keep=0.1；
  - keep loss除以max(q,1)，每prompt所有cycles权重和为1；
  - 单seed筛选，固定optimizer budget；不做大超参网格；
  - 每个arm先做64–128 blocks same-set overfit，确认能连续推进frontier，再跑2K。
- Continue criterion:
  - GFPR-16 fixed EAL ≥7.55且相对released ≥+0.30；
  - paired 95% lower bound >0；
  - lost tokens ≤0.5× gained tokens；
  - harmful prompts ≤20%。
- Stop:
  - EAL ≤7.40；
  - alignment正确但训练持续降低held-out；
  - dynamic-15与fixed-15无可重复差异且all-16也无信号。
- Failure interpretation:
  - Dynamic-15胜fixed但GFPR-16不再增益：position-0 scorer/optimization问题；
  - GFPR-16 same-set有效、held-out无效：head representation不足，允许Stage D容量比较；
  - 三臂均无效：on-policy frontier thesis被当前数据否证，不扩数据。
- Table target: main causal ablation table。
- Priority: MUST-RUN。

### Block B2: Gate C — 16K Scale and One Policy Refresh

- Claim tested: Gate B信号能扩大到主目标，并适应修正后的anchor distribution。
- Dataset:
  - OpenPerfectBlend balanced 16K prompt nested subset，必要时32K；
  - phase3 validation_select用于selection；
  - validation_gate只在method freeze后一次。
- Systems: released Domino；GFPR v0；GFPR v0+v1 50/50 replay。
- Metrics: 与B1相同，外加v0/v1 anchor histogram、frontier position histogram和old/new-policy repair。
- Setup:
  - Gate B通过后才collect；
  - v0由released policy采；
  - 选定v0 checkpoint重采v1；
  - prompt-level 50/50 mix；
  - 先1 seed做16K go/no-go；达到7.8后跑3 seeds与32K或延长budget；
  - seed selection规则预先固定，不用validation_gate选配置。
- Success:
  - fixed EAL ≥8.325；
  - dynamic rollout ratio ≥1.15×；
  - paired lower bound >0；
  - B1 harm gates全部通过。
- Failure:
  - v0有益但refresh无益：检查anchor churn和prefix protection；
  - refresh后<7.8：打开matched Stage-D head-versus-LoRA容量试验；
  - 7.8–8.324：主问题仍未解决，可增加16K到32K或轻量容量，但不能开始SGLang。
- Table target: main result/scale table和anchor-distribution plot。
- Priority: CONDITIONAL MUST-RUN。

### Block B3: Novelty and Simplicity Isolation

- Claim tested: 增益来自policy-correct frontier与position 0，而非额外参数。
- Compared systems:
  - Fixed-15；
  - Dynamic-15；
  - GFPR-16；
  - 仅当C1已有正信号：轻量candidate residual或final-layer LoRA matched arm。
- Metrics: EAL、oracle-gap recovery、参数量、eager/graph head latency、harm。
- Setup:
  - 核心三臂复用B1结果，不重复大训练；
  - optional capacity arm使用同data、loss和steps；
  - LoRA/backbone arm在线重算或重采hiddens，禁止使用stale cached hidden。
- Success: GFPR-16以最小组件达到或接近最好结果；若更大arm无显著matched增益，从final method删除。
- Failure: 若LoRA显著必要，paper contribution改为GFPR supervision加最小representation adaptation，并重新核算latency。
- Table target: main ablation/complexity table。
- Priority: 核心三臂MUST；容量arm CONDITIONAL。

### Block B4: Gate E — SGLang End-to-End

- Claim tested: EAL增益转化为真实系统吞吐。
- Entry: B2硬成功门全部通过。
- Systems: released Domino；final GFPR；DFlash作为上下文baseline。
- Setup: 同A40或目标GPU、同SGLang commit、batch、prompt/length、CUDA Graph与warmup。
- Metrics: EAL、draft/head/verify latency、time per output token、tokens/s、显存。
- Success: 相对Domino tokens/s约+15%或更高，且输出lossless。
- Failure: 若EAL达标但吞吐未达标，profile第16次head与full-vocab projection；再评估candidate contraction，不能回头降低EAL成功线。
- Table target: main system table与latency breakdown。
- Priority: CONDITIONAL MUST-RUN。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Estimated Cost | Main Risk |
|---|---|---|---|---:|---|
| M0 | CPU语义与loss测试 | R000–R004 | 全部tests通过 | <1 CPU-hour | off-by-one/错误mask |
| M1 | GPU smoke与Gate A | R010–R013 | identity、oracle、bonus全过 | 0.5–1.5 GPU-h | full16 alignment |
| M2 | 2K v0 collection | R020–R022 | dynamic/fixed records完整 | 4–8 GPU-h total，可4卡并行 | collector吞吐/磁盘 |
| M3 | Gate B三臂 | R030–R033 | GFPR ≥7.55且+0.30、CI/harm过门 | 4–10 GPU-h | head不泛化 |
| M4 | 16K+refresh | R100–R105 | EAL ≥8.325与dynamic ≥1.15× | 30–80 GPU-h | data/anchor shift |
| M5 | 3 seeds/final gate | R110–R113 | validation_gate与reserved test通过 | 12–30 GPU-h | variance |
| M6 | SGLang | R200–R203 | throughput约+15% | 4–12 GPU-h | head kernel latency |

## First Implementation Runs

1. R000–R004：synthetic unit tests，包括current frontier、normalized keep、prompt weights、r+1和r=16 bonus。
2. R010：32-prompt collector/full16 oracle smoke，step-0 exact identity。
3. R011：validation_select全量Gate A oracle。
4. R020：phase3 train 2K v0 dynamic collection，按domain/shard并行。
5. R030：64–128 block same-set overfit；随后三臂Gate B。

## Compute and Data Budget

- Gate A+B预计约10–20 GPU-hours，总wall time在4张A800/A40上约半天到一天。
- Gate C只有Gate B过门后发生；16K v0+v1收集与训练预计30–80 GPU-hours。
- 数据存储：
  - full16 bf16 hidden约80 KiB/block；
  - 2K prompts、约25–30 cycles/prompt可能约4–6 GiB；
  - 16K prompts、128-token continuation约15–25 GiB。
- Collector必须支持shards、atomic completion marker和resume；不做无意义的hash gate。
- 最大瓶颈是大量小draft forwards与full-vocab head训练；优先按prompt多GPU切分，而不是先优化外围工程。

## Risks and Mitigations

- 不同forward shape产生hidden数值差：Gate A统一从同一collector路径生成full16，不拼旧cache。
- Position-0 alpha在零点使head梯度受限：alpha本身在非零released correction上有梯度；smoke中记录alpha和position-0 margin梯度。
- Full-vocab logits显存高：小batch加gradient accumulation，复用现有Domino full-head实现。
- Current-frontier停在同一错误：每N steps报告frontier histogram；same-set overfit必须看到frontier向后移动。
- 动态prompt产生不同cycle数：prompt-normalized sampler/loss，不能按block简单平均。
- Validation选择泄漏：validation_select选checkpoint，validation_gate只在冻结后一次。
- Scale数据只有128 continuation：主训练仍有效；true rollout comparison使用相同token budget并报告cycle count。
- Stage D训练backbone：cached hiddens作废，强制online forward或recollection。

## Final Checklist

- [x] 主claim和anti-claim已冻结
- [x] 核心实验不超过5 blocks
- [x] strongest baseline为released Domino与同head static control
- [x] novelty由dynamic/fixed与all-16/15删除实验隔离
- [x] 大架构被置于conditional gate
- [x] paired统计与harm门已绑定
- [x] EAL成功线没有被proof-of-signal替代
- [ ] Gate A代码与数据通过
- [ ] Gate B效果通过
- [ ] Gate C达到8.325
- [ ] SGLang吞吐通过

