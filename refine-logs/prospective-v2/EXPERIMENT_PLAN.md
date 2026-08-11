# Experiment Plan: prospective-v2 FBPF-DFlash

**Problem**：在保持 released DFlash 单链、一次 target verification、输出 exactness 与原部署拓扑的前提下，提高真实未见 prompt 的接受前缀，并验证其相对普通 D-PACE LoRA 的机制增量。  
**Method Thesis**：用 frozen accepted prefix 定义 exact verifier-induced feasible set，用 adapted current first miss 定义动态修复 frontier，只提交保持全部 protected positions 正 margin 的 D-PACE task step。  
**Date**：2026-08-06  
**Research-refine status**：8.77/10，REVISE，drift NONE，fatal issue NONE，proceed_to_experiment_plan=yes。  
**Planning status**：G0_V3_PENDING_REVIEW；本计划不自动授权 implementation、GPU、prospective data generation 或 falsifier opening。v3 execution corrections were frozen 2026-08-06 15:41 +08:00 without changing the method thesis or claims.

## Route Boundary

- prospective-v2 与已关闭 R083 完全隔离，绝不称为 retry、rescue、fix 或 downstream。
- 禁止读取或使用旧 validation、reserved、formal、R083 outcome；旧 100k producer-train 只允许作为 hash/ngram exclusion index 与 aggregate power receipt 的来源。
- 不允许 suffix censoring、tree/multipath、runtime RNN/expert、DFlare-style fusion、threshold sweep 或结果后 loss reweight。
- formal verdict 仍为 REVISE；第一授权目标只是证明 implementation 与 cost feasibility。工程 gate 失败即关闭，不解释为科学负结果。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1-EFFICACY：FBPF 提高 prospective prompt-balanced EAL 并保持经验安全 | 直接回答 frozen selector 已达上限后，representation adaptation 是否真正解决 bottleneck | D–released EAL point≥+.30 且 paired 95% CI lower>0；released-referenced harm-rate upper≤.05；mean-harm upper≤.10；first-token contrast lower≥-.005；各 domain point≥0 | B1, B4 |
| C1-SYSTEM/DEPLOYMENT：FBPF 可合并回 released graph 且 latency 等价 | 验证收益没有依赖新增 runtime module/forward | bf16-stored merge、wrapper removal、trace/output exactness；latency TOST ±2% | B3 |
| C2：dynamic frontier 与 prefix feasibility 都是增益所必需，而不是普通 LoRA/D-PACE | 决定方法是否有独立 mechanism contribution | D–A EAL point≥+.10 且 CI lower>0；D–B EAL point≥+.10 且 CI lower>0；D–C released-referenced harm difference≤-.02 且 CI upper<0，同时 EAL NI lower≥-.05 | B2, B4 |

### Anti-claims to rule out

1. gain 只来自 1.835M LoRA parameters 或 ordinary D-PACE：由 D–A 排除；
2. current first-break 动态 frontier 不必要：由 D–B 排除；
3. feasibility optimizer 只是额外算力，不能减 harm：由 D–C 排除；
4. 部署收益来自改变 inference graph 或增加 forward：由 merge/trace/latency audit 排除；
5. 新路线只是旧 PROS-Gate 的改名重试：由 source、split、artifact 与 tracker closure 排除。

## Paper Storyline

- Main paper must prove：
  - Table 1：Released、A DPACE、B STATIC-PF、C DYNAMIC-U、D FBPF 的同一 falsifier prompt-balanced metrics；
  - Table 2：D–A、D–B、D–C 的预注册 contrasts 与 CIs；
  - Table 3：merge trace、throughput gate 与 deployment latency TOST。
- Appendix can support：
  - per-seed、per-domain、base-first-miss strata、harm magnitude/CDF；
  - optimizer commit/skip/restoration、active-set switch、gradient-ratio diagnostics；
  - power、component leakage 与 source-closure receipts。
- Experiments intentionally cut：
  - 重跑 GCLS/FMAS/SAVS/CAMRS；
  - full finetune、额外 LoRA rank/layer sweep；
  - DFlare fusion、Domino/DeLS/DSpark sequential head、tree drafting；
  - post-hoc threshold、loss coefficient rescue、旧 downstream；
  - 无法改变 C1/C2 reviewer belief 的 benchmark padding。

## Frozen Global Protocol

### Systems and model scope

- target：/hpc2hdd/home/zwang668/sp decoding/hf_assets/models/Qwen3-4B，frozen；
- released draft：/hpc2hdd/home/zwang668/sp decoding/hf_assets/models/Qwen3-4B-DFlash-b16；
- D-PACE source pin：third_party/D-PACE commit f36bad6e6b0f9f5b59e1e6cf405c705b46d2b43f；
- training/evaluation Python：third_party/Domino/.venv/bin/python，PyTorch 2.9.1+cu128，Transformers 4.57.1；
- current environment has no peft package；implement one auditable native LoRALinear instead of installing an unpinned dependency；
- exact target paths：`layers.{3,4}.self_attn.{q,k,v,o}_proj` 与 `layers.{3,4}.mlp.{gate,up,down}_proj`；
- rank=16、alpha=16、dropout=0、scale=1；LoRA B zero-initialized，adapter output exactly zero；
- A/B master parameters and Adam moments are float32；A uses `kaiming_uniform_(a=sqrt(5))` over sorted module paths with an isolated CPU generator seeded `2026080600+training_seed`；B is exact zero；all four arms under a seed have bitwise-identical A/B and do not advance global RNG；
- LoRA branch computes in float32 and casts delta to base-output dtype before addition；per-module A/B hashes are receipted；
- configuration formula and live enumeration must both equal 1,835,008 trainable parameters；
- base weights frozen bf16；logits/margins float32；projection reductions float64；
- attention path is exactly `attn_implementation="sdpa"`；all arms identical；
- merge uses float32 accumulation only, `(base.weight.float()+scale*(B@A)).to(base.weight.dtype)`；stored/runtime weights remain bf16 and every wrapper is removed。

### Native LoRA and functional candidate checks

- src/sph/fbpf.py owns LoRA injection/disable/merge、D-PACE loss、first-miss masks、blockwise constraints、batched VJP、transactional optimizer；
- candidate forward uses torch.func.functional_call over LoRA parameters only；anchors、target hidden、noise embedding、attention mask and labels are sampled/materialized once per outer step and reused by every backtracking candidate；
- exact-tie mask and non-gold winner index detach before grad_outputs construction；
- one is_grads_batched=True autograd.grad call returns K≤4 rows；stable ties use block-row order 0–3；
- restoration starts at batch-start without moment/t_adam mutation；after success it recomputes logits/masks/task-grad/constraints/VJP and attempts the same minibatch task transaction；restore+skip retains restored parameters but not shadow moments/t_adam；failure rolls back and aborts；k_outer increments once per consumed minibatch。

### Training budget

- every outer minibatch contains `source_prompts=1` expanded to `tensor_batch_size=4` rows with `anchors_per_row=1` and `reduction_divisor=4`；N=4、L=15，so pinned D-PACE `/bsz` matches the proposal；
- frozen target alone generates the greedy continuation using float32 decision logits and lowest-ID ties；for `maximum=T-16`, offsets are `int(round(j*maximum/3))`, j=0..3 with Python rounding and four-distinct-complete assertion；anchor=`continuation[o]`, gold position i=`continuation[o+i]`，and target context/features end strictly before anchor；
- no draft/comparator defines or filters gold；each retained prompt has continuation length≥19 and exactly four complete anchors；
- fit prompts：8,000；checkpoint prompts：1,000；falsifier prompts：n_f=max(1500,n_power)；
- one pass over fit：8,000 outer steps；checkpoints fixed at 1k,2k,…,8k；T_final=8k；
- lr peak 1e-4、betas (.9,.95)、eps 1e-8、wd 0、clip 1.0、4% warmup/cosine；
- matched seeds 0,1,2；within each seed all A/B/C/D data order and every non-method setting are identical；
- checkpoint selection runs independently per arm/seed with the frozen safety-first rule；
- five system definitions instantiate 13 concrete instances：released + 12 arm/seed hashes；each trained instance is labeled `selected_feasible` or `diagnostic_T_final` before the single falsifier opening；
- primary arm estimand first averages the three matched-seed prompt metrics, then applies the frozen prompt-balanced estimator；all individual seeds are reported；
- D must have a feasible selected checkpoint for all three seeds；otherwise C1-EFFICACY fails and the falsifier is not opened。A/B/C diagnostic seeds automatically fail their linked C2 contrasts but do not block C1 when all D seeds are feasible。

### Data isolation and sequence generation

- source：local Open-PerfectBlend parquet closure recorded in the existing producer metadata；
- deterministic split seed：20260806；
- exclude exact normalized hashes and 8-gram Jaccard≥.5 against the frozen old-100k/hash-only prior index；
- normalize with Unicode NFKC、lowercase and whitespace collapse；tokenize with Unicode regex `\w+|[^\w\s]`；use contiguous 8-grams or the full tuple when shorter；
- build candidates with deterministic inverted postings，then sorted exact-Jaccard≥.5 edges and deterministic union-find；component ID is SHA256 of sorted normalized hashes joined by NUL；exclude cross-domain components before allocation；
- reserve count per domain/split is `max(64,ceil(.10*active_domain_quota))`；allocate whole components in fixed fit/checkpoint/falsifier × math/code/chat order using SHA256 split/component ranks until exact active+reserve row counts are available；rank materials use UTF-8, lowercase hex hashes, ASCII decimal seed/ordinal without leading zero, and one NUL between fields；ranked surplus rows remain discarded in that split；independent replay compares ownership/status/order exactly；no component can cross a split；
- split builder emits row counts、component counts、domain counts、hashes and source closure，and an independent implementation must replay them；
- fit/checkpoint target-greedy sequences may be generated after split audit；falsifier prompts/continuations remain sealed until all model/checkpoint identities freeze；
- generation：Qwen tokenizer chat template、enable_thinking=False、T=0、max_new_tokens=128；
- prompts with fewer than 19 generated continuation tokens permanently consume that attempt and use the next predeclared same-domain/same-split reserve row；reserve exhaustion is a terminal sequence-collection failure class；every retained prompt has exactly four blocks；falsifier reserves are consumed sealed inside the same single opening；no reassignment/manual replacement；
- inability to meet exact active/reserve quotas fails the data gate；component splitting is forbidden；
- no row-level falsifier prompt、label、score or checkpoint rank may be printed to logs。

### Power and frozen estimand

- one producer-train-only POWER_RECEIPT.json may include only hashes、counts、cluster-size aggregates、conservative paired prompt-level SD/ICC upper bounds；
- no receipt may contain mean、sign、row、sample ID、checkpoint rank or downstream statistic；
- define `H^M=1[a^M<aReleased]`、mean harm `(aReleased-aM)_+`、first-token contrast `1[aM≥1]-1[aReleased≥1]` and C2 harm `H^D-H^C`；checkpoint gates are per-arm/per-seed versus released；
- superiority uses two-sided alpha .05；non-inferiority uses one-sided alpha .05；all stated effects require ≥80% power；no variance reduction from three training seeds；
- `DE=1+(((1+cv_upper^2)*mean_cluster_size_upper)-1)*icc_upper`；two-sided sample sizes use `ceil(DE*((z.975+z.8)*sd_upper/effect)^2)` for effects .30/.10/.10/.02；one-sided NI uses `ceil(DE*((z.95+z.8)*sd_upper/distance)^2)` for .05/.005；
- add harm-rate upper-bound precision `ceil(DE*.25*(z.95/.015)^2)` and mean-harm upper-bound precision `ceil(DE*(z.95*sd_mean_harm_upper/.03)^2)`；n_power is the maximum registered requirement；
- point estimand：equal blocks within prompt，then equal prompts；never globally average raw blocks and never first average components；
- uncertainty：10,000 frozen-seed paired connected-component cluster bootstrap replicates，domain-stratified；a sampled component carries all constituent prompt metrics；percentile quantiles use linear interpolation，two-sided .025/.975 and one-sided .05/.95；
- arm-level primary prompt metric averages matched seeds before the prompt-balanced point estimate/bootstrap；
- released/Domino historical gap recovery is not computed unless both are re-estimated under this exact estimand；current Domino 5.93853 remains background only。

## Experiment Blocks

### Block B0: Implementation Correctness and Engineering Feasibility

- Claim tested：the proposed optimizer is executable and its complete constraint set fits the declared cost/memory envelope。
- Why this block exists：separates software/cost failure from scientific falsification。
- Dataset / split / task：
  - CPU analytic tensors and tiny mock DFlash modules；
  - deterministic synthetic GPU token sequences with producer-train length aggregates only；
  - after split authorization，128 fit prompts/512 blocks for capacity。
- Compared systems：A DPACE and D FBPF for throughput；A/B/C/D for 512-block capacity。
- Decisive metrics：
  - official/manual D-PACE scalar and flattened-gradient parity；
  - zero-adapter equality、LoRA parameter count、merge equality；
  - all-position↔block-max equivalence；
  - exact candidate rejection under ties/active switches；
  - restore+commit、restore+skip、failed-restore rollback and parameter/moment/counter invariants；
  - each of three clean A/D pairs：median(TD)/median(TA)≤4、Q95(TD)/Q95(TA)≤6；
  - D peak allocated≤60 GiB and D–A≤12 GiB；allocated/reserved both reported；
  - gradient-ratio post-warmup median in [.05,20]。
- Setup：
  - three counterbalanced pair orders A→D、D→A、A→D；
  - each arm is a fresh exclusive-A800 process，20 warmups + 200 timed steps；
  - every warmup/timed D row has nonempty protected prefix，assert K=4；receipt emits K/protected-count histograms；engineering labels may be constructed from frozen DFlash logits solely to force this path and are never scientific truth；
  - A performs only real D-PACE work；D includes frozen reference、task backward、four-row VJP、projection/restoration and every exact forward；
  - p95 ratio is ratio of empirical p95 step times，not p95 of per-step ratios；
  - all three pairs must pass；no truncation/sampling。
- Capacity success：
  - 32-block micro-overfit has finite decreasing task loss for every arm；
  - 512-block run has no nonfinite/OOM/projection/restoration failure；
  - every accepted constrained commit is exactly feasible；
  - counter/state receipts replay exactly；fixed gradient-ratio gate passes。
- Failure interpretation：implementation failure stops coding stage；throughput/memory failure closes engineering route；neither is a scientific result。
- Table / figure target：Appendix implementation-assurance table；cost row enters main deployment table only after C1。
- Priority：MUST-RUN。

### Block B1: Main Prospective Anchor Result

- Claim tested：C1。
- Why this block exists：直接证明 FBPF 是否跨过 frozen selector 的 +.08–.28 EAL ceiling。
- Dataset / split / task：one sealed n_f-prompt OPB falsifier，4 complete blocks/prompt，greedy T0。
- Compared systems：released DFlash、A、B、C、D；Domino only historical context。
- Decisive metrics：prompt-balanced EAL delta、harm rate、mean harm、first-token；domain points；10k component-cluster paired CIs。
- Secondary metrics：accepted-length distribution、per-seed values、frozen first-miss strata。
- Setup：released + 12 frozen hashes（each selected_feasible or diagnostic_T_final）evaluated in one process family and one common opening；no checkpoint/model-specific filtering。
- Success criterion：exact C1 thresholds in Claim Map；all three D seeds must have passed checkpoint safety selection。
- Failure interpretation：
  - EAL threshold/CI fail：representation route does not clear the practical effect bar；
  - safety fail：batch-local feasibility did not generalize；
  - domain fail：claim scope narrows and full C1 fails。
- Table / figure target：Main Table 1；accepted-length delta CDF in appendix。
- Priority：MUST-RUN。

### Block B2: Factorial Mechanism Isolation

- Claim tested：C2 and all three anti-claims about ordinary LoRA、static frontier、unconstrained dynamics。
- Why this block exists：8.77 review 的最大 venue risk 是 known primitives 的组合；只有 decisive factorial separation 能支持 novelty。
- Dataset / split / task：与 B1 完全相同的一次 falsifier opening。
- Compared systems：
  - A DPACE；
  - B STATIC-PF；
  - C DYNAMIC-U；
  - D FBPF。
- Metrics：D–A、D–B EAL；D–C harm-rate difference and EAL non-inferiority；同一 seed-averaged prompt estimand/bootstrap。
- Setup details：同一 LoRA scope、参数量、steps、seeds、data order、checkpoint budget；禁止 private rescue。
- Success criterion：exact C2 thresholds in Claim Map。
- Failure interpretation：
  - D≤A：ordinary D-PACE LoRA explains gain；
  - D≤B：dynamic frontier unsupported；
  - D not safer/noninferior than C：prefix feasibility unsupported；
  - 任一项失败都删除对应 mechanism claim，不增加新 arm。
- Table / figure target：Main Table 2；optimizer diagnostics appendix。
- Priority：MUST-RUN。

### Block B3: Deployment Simplicity and Exactness

- Claim tested：C1-SYSTEM/DEPLOYMENT；FBPF 不增加 runtime module/forward，并保持 target-exact emitted output 与 ±2% latency equivalence。
- Why this block exists：defends the paper’s simplicity claim and rules out hidden graph changes。
- Dataset / split / task：
  - checkpoint-set trace audit；
  - C1-EFFICACY PASS 且 D seed0 selected-feasible identity 已冻结后进入 G8；G8 先且仅先执行该 checkpoint 的 merge/wrapper-removal/trace/dtype/output audit，PASS 后才允许 latency；
  - small end-to-end target-greedy output exactness replay。
- Compared systems：released DFlash vs adapter-disabled、adapter-enabled、merged D seed0。
- Metrics：
  - zero-adapter exact logits/argmax/accepted length；
  - adapter vs merged atol=.02、rtol=.02 plus exact argmax/length；
  - ordered module/operator/kernel names、shapes/counts、dtype、attention path、block size、one draft/one target forward；
  - fixture 为 checkpoint manifest stable rank 前 50 prompts，end-to-end speculative generation `max_new_tokens=64`，target-exact output；
  - 20 process restarts，每次 cycling 200 warmups + 50 alternating pairs；measured value 为 seconds/emitted-token，pair statistic 为 log(merged/released)，每 restart 的 r_j 取 50 ratios 的 median；
  - final estimate=`mean(r_j)`，s=`std(r_j,ddof=1)`，90% CI=`estimate±t_(.95,19)*s/sqrt(20)`；TOST 只有 CI endpoints 严格位于 log(.98)…log(1.02) 内才 PASS。
- Success criterion：all trace/output invariants and latency TOST pass。
- Failure interpretation：对应 system/deployment claim 关闭；不影响已经观测的 offline EAL，但不能声称 zero-overhead deployment。
- Table / figure target：Main Table 3 and appendix trace receipt。
- Priority：MUST-RUN as ordered G8a merge/trace/output then G8b latency after C1-EFFICACY。

### Block B4: Frozen Failure Analysis

- Claim tested：no new positive claim；解释成功/失败发生在哪些 first-break/domain strata，检查是否由少数 prompts 驱动。
- Why this block exists：帮助判断 DFlash–Domino 背景差距是否与 first-break correction 一致，不扩展方法。
- Dataset / split / task：B1/B2 同一已开 falsifier outcomes；不得重跑模型。
- Compared systems：released、A、B、C、D only。
- Metrics：base m0 strata、accepted-length delta histogram、harm magnitude tail、domain、commit/skip/restoration diagnostics 与 training gradient ratio。
- Success criterion：descriptive only；所有图使用冻结 bins/seeds，不产生新 gate。
- Failure interpretation：若收益集中于单 domain/极小 stratum，只缩窄讨论；不得据此调模型。
- Table / figure target：Appendix Figure A1/A2；main text one concise diagnosis if C1/C2 pass。
- Priority：NICE-TO-HAVE，且只复用已有 outcomes。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Estimated Cost | Main Risk |
|---|---|---|---|---|---|
| M0 Protocol freeze | source closure、native-LoRA contract、metric/power schemas | PV2-001 | fresh review GO before implementation | <1 CPU-h | hidden ambiguity |
| M1 CPU implementation | pure math、LoRA、optimizer、statistics tests | PV2-002…004 | full tests + independent code review GO | 4–12 CPU-h | state mutation/tie bug |
| M2 Synthetic GPU gate | real model smoke and 3 clean A/D pairs | PV2-005…007 | every pair passes all cost/memory rules | 2–6 A800-h | batched VJP/backtracking cost |
| M3 Prospective split | power、component split、independent replay | PV2-008…010 | source/split audit GO | 2–6 CPU-h | insufficient eligible components |
| M4 Fit/checkpoint data | target-greedy sequence generation | PV2-011 | exact counts/hashes; falsifier remains sealed | 12–35 A800-h | short continuations/storage |
| M5 Capacity | 32-block overfit + 512-block A/B/C/D | PV2-012…014 | all correctness/capacity gates pass | 4–8 A800-h | feasibility oscillation |
| M6 Main training | 4 arms × 3 matched seeds × 8k steps | PV2-015 | all D seeds feasible; identities freeze | 40–120 A800-h | optimization variance |
| M7 One opening | generate/evaluate common falsifier、bootstrap、claim audit | PV2-016…018 | C1/C2 adjudication | 8–18 A800-h | sealed protocol failure |
| M8 Deployment | ordered merge/trace/output then latency after C1-EFFICACY | PV2-019…020 | both receipts for final C1-SYSTEM/DEPLOYMENT | 2–6 A800-h | kernel variance |

Stop/go is monotonic：任何 gate failure 只允许修复明显 implementation bug 后重新走 fresh review；不能在科学 outcome opening 后修改方法、threshold、split 或 arm。

## Execution Authorization Ladder

| Gate | Opens | Remains Closed Until Pass |
|---|---|---|
| G0 plan + contract review | local source implementation and CPU/mock tests only | all GPU and data generation |
| G1 unit/parity/code review | one synthetic GPU smoke | throughput pairs、prospective data |
| G2 smoke review | exactly three A/D cost pairs | prospective split/data |
| G3 cost adjudication | power + split materialization/audit | target generation、training |
| G4 split audit | fit/checkpoint sequence generation | capacity/full training/falsifier |
| G5 data audit + capacity wrapper review | 32/512-block capacity only | full 12-run training |
| G6 capacity result review | 4×3 main training array | falsifier generation/evaluation |
| G7 checkpoint selection + identity audit | single common falsifier opening、C1-EFFICACY/C2 adjudication | latency and any additional evaluation |
| G8 C1-EFFICACY PASS + frozen D-seed0 identity | first exactly one merge/wrapper-removal/trace/dtype/output audit；only its PASS opens fixed latency；both receipts finalize C1-SYSTEM/DEPLOYMENT | no extra science route |

## Compute and Data Budget

- Total estimated：68–193 A800 GPU-hours，plus 6–20 CPU-hours for manifests/tests/audits。
- Storage：token sequences/checkpoints/outcomes estimated 30–120 GiB；do not cache full target hidden unless a reviewed benchmark proves it necessary and provenance-safe。
- Data preparation：new component-disjoint OPB manifests、fit/checkpoint sequences、sealed falsifier candidate list、power/source/split receipts。
- Human evaluation：none。
- Biggest bottleneck：D exact candidate forwards under batch-local infeasibility；second is 12 matched full training runs。
- Scheduler：i64m1tga800u currently exposes 8 A800×8-GPU nodes；each planned task requests one exclusive GPU unless a reviewed wrapper says otherwise。

## Risks and Mitigations

- Batched VJP still too slow：
  - Mitigation：strict synthetic gate；close route rather than sample constraints。
- Native LoRA merge changes graph or numerics：
  - Mitigation：zero-output init、independent parameter enumeration、float32 accumulation then bf16 storage、wrapper deletion、trace and exact argmax tests。
- New batch often infeasible after prior updates：
  - Mitigation：measure restoration/skip rates in 512-block capacity；abort on restoration failure；no replay/EMA rescue。
- Dynamic gradient scale pathological：
  - Mitigation：fixed [.05,20] engineering gate；no coefficient tuning。
- Near-duplicate leakage：
  - Mitigation：exact inverted-postings candidates + exact Jaccard union-find；independent replay from source hashes。
- Training-seed instability：
  - Mitigation：three matched seeds，seed-averaged primary prompt metric，all seed results reported。
- Falsifier contamination or premature opening：
  - Mitigation：hash-only sealed candidate identities，defer continuation generation，single process family evaluates all frozen checkpoints。
- Result novelty insufficient：
  - Mitigation：do not add modules；C2 factorial gate determines whether paper claim survives。

## Final Checklist

- [x] Exactly two primary claims。
- [x] Main anchor、novelty isolation、simplicity/deployment and failure analysis covered。
- [x] Baselines limited to released DFlash and one matched LoRA factorial family；Domino is descriptive only。
- [x] Prompt-balanced estimand and component bootstrap are unambiguous。
- [x] Seeds、steps、checkpoints、losses、optimizer、thresholds and failure meanings are frozen。
- [x] Nice-to-have diagnostics cannot delay or modify must-run evidence。
- [x] Old R083/downstream and post-hoc rescue remain closed。
- [ ] G0 fresh implementation-plan review passes。
- [ ] No GPU or data job launches before its explicit gate opens。
