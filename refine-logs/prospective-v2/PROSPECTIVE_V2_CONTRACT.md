# prospective-v2 Execution Contract

**Route**：FBPF-DFlash prospective-v2  
**Version**：3  
**Frozen at**：2026-08-06 15:41 +08:00  
**Status**：G0_GO_LOCAL_IMPLEMENTATION_AND_CPU_MOCK_ONLY  
**Sources**：FINAL_PROPOSAL.md + EXPERIMENT_PLAN.md  
**Old contract**：idea-stage/docs/research_contract.md remains an immutable GCLS artifact and does not govern this route。

**G0 review**：`G0_CONTRACT_V3_REVIEW_20260806_154300.md`，verdict GO，SHA256 `2b1d40e9298a52b351cc4c093f65dad884abfb7be69c168dabf40a3ae3a6e04d`。The reviewed pre-authorization contract SHA256 is `6761fba7ad499b7a373205dc549f2056b9bee042367cb5282b19d7639ef48570`。Only local implementation and CPU/mock tests are open；every later gate remains closed。

## 1. Immutable Problem and Scope

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

Formal refinement verdict is REVISE at 8.77/10，drift NONE，fatal issue NONE。This contract authorizes only monotonic gated implementation；it does not relabel the proposal READY。

## 2. Route Isolation

1. R083 remains CLOSED_OPERATIONAL_FAILURE_EXIT1_OPENING_CONSUMED_NO_RETRY_JOB10141601。
2. No old validation、reserved、formal or R083 outcome may be read、reconstructed or scored。
3. The old 100k producer-train route may contribute only：
   - a frozen normalized-hash/8-gram exclusion index；
   - aggregate counts、cluster-size summaries、paired SD/ICC upper bounds in POWER_RECEIPT。
4. No artifact、job、metric or claim may describe prospective-v2 as retry、rescue、repair or downstream of R083。
5. No scientific data generation or GPU job is opened by this file。

## 3. Frozen Model and LoRA

- target path：/hpc2hdd/home/zwang668/sp decoding/hf_assets/models/Qwen3-4B；
- draft path：/hpc2hdd/home/zwang668/sp decoding/hf_assets/models/Qwen3-4B-DFlash-b16；
- block size 16；predicted positions L=15；target layers [1,9,17,25,33]；
- hidden 2560；intermediate 9728；q heads 32；kv heads 8；head dim 128；draft layers 5；
- frozen target and frozen released draft parameters；
- trainable native LoRA exact paths：`layers.{3,4}.self_attn.{q,k,v,o}_proj` and `layers.{3,4}.mlp.{gate,up,down}_proj`；
- rank 16、alpha 16、scale 1、dropout 0；
- A/B master parameters and Adam moments are float32；A uses `kaiming_uniform_(a=sqrt(5))` over sorted module paths with an isolated CPU generator seeded `2026080600+training_seed`；B initializes exactly zero；matched arms share bitwise-identical A/B without advancing global RNG；
- trainable count must equal：
  - per layer 917,504；
  - total 1,835,008；
- no peft dependency；native wrapper must expose disable、state_dict、partial `torch.func.functional_call(...,strict=False)` override and merge；record every module’s A/B hash；
- base/target weights bf16；objective logits/margins float32；projection dot/norm reductions float64；
- attention implementation is exactly `sdpa`；
- LoRA branch computes in float32 and casts its delta to base-output dtype before add；
- “float32 merge” means accumulation only：`Wmerged=(Wbase.float()+scale*(B@A)).to(Wbase.dtype)`；stored/runtime weights remain bf16，then every wrapper is removed；
- zero adapter must be bitwise-equal in logits、argmax and accepted length；adapter/merged tolerance atol=.02、rtol=.02 plus exact argmax/length。

## 4. Frozen Batch and Objective

- outer minibatch：`source_prompts_per_outer=1` expanded to `tensor_batch_size=4` rows，`anchors_per_row=1`、N=4、`reduction_divisor=4`；pinned D-PACE `/bsz` must match the complete scalar/gradient；`bsz=1,anchors=4` is forbidden；
- predicted positions：15，anchor position excluded from loss；
- all argmax ties choose lowest vocabulary ID；
- the frozen target alone autoregressively generates the continuation using float32 decision logits；for continuation length T，`maximum=T-16` and `offsets[j]=int(round(j*maximum/3))` for j=0..3 using Python rounding，with four-distinct-complete assertion；
- at offset o，anchor=`continuation[o]` and gold `y_i=continuation[o+i]` for i=1..15；target context/features contain tokens strictly before anchor；no DFlash、Domino or comparator defines/filters gold；
- non-gold winner index、m0、mθ、protected masks and exact-tie masks detach and refresh every forward；
- protected set Pn contains positions strictly before frozen first mismatch m0；
- D-PACE alpha=.5，all 15 positions，no suffix mask；
- CPU float64 D-PACE scalar/gradient parity atol=1e-10、rtol=1e-9；
- GPU float32 scalar parity atol=rtol=2e-5；
- flattened LoRA-gradient max abs≤5e-5 and cosine≥.999999；
- epsilon_tie=1e-4；tau_f=1e-5；protected gamma therefore≥9e-5；
- dynamic/static hinge coefficient fixed at 1；empty mean is zero。

Arms：

| Arm | Loss | Optimizer |
|---|---|---|
| A | full D-PACE | ordinary AdamW |
| B | D-PACE + static frontier at m0 | prefix-feasible transactional optimizer |
| C | D-PACE + dynamic frontier at mθ | ordinary AdamW |
| D | D-PACE + dynamic frontier at mθ | prefix-feasible transactional optimizer |

Every arm uses the same LoRA scope、data、anchor records、steps、schedule、seeds and checkpoint budget。

## 5. Exact Constraint and Transaction

\[
c_{n,i}=10^{-4}-\gamma_{n,i},\qquad
C_n=\max_{i\in P_n}c_{n,i}.
\]

C_n≤tau_f iff every protected per-position c≤tau_f。Empty Pn is vacuous。For exact float32 ties，Gn is the uniform average of all tied gradients。A single is_grads_batched=True autograd.grad call produces K≤4 rows；constraint sampling/truncation is forbidden。

Optimizer constants：

- lr peak 1e-4；betas (.9,.95)；eps 1e-8；weight decay 0；
- global task-gradient clip 1.0；warmup ratio .04；cosine；
- k_outer advances once per consumed minibatch；
- t_adam advances only once per accepted task commit；
- max projection sweeps 4；tau_linear=1e-7；
- candidate alpha sequence [1,1/2,1/4,1/8,1/16,1/32,1/64,1/128]；
- exact candidate acceptance checks every protected per-position c≤tau_f；
- max restoration cycles 8；tau_restore=1e-4；minimum exact violation decrease 1e-7；
- failed projection produces a task skip；failed restoration restores batch-start theta and aborts the run；
- task skip commits neither parameter、moment nor t_adam；
- restoration begins from batch-start theta and changes theta only，never task moments/t_adam；after successful exact feasibility it recomputes logits、all detached masks/indices、task gradient、constraints and VJP at restored theta，then attempts the same-minibatch projected task transaction；
- restore+task-skip retains restored theta but keeps batch-start moments/t_adam；failed restoration rolls back batch-start theta and aborts；
- stable-tie projection order is block rows 0、1、2、3；k_outer increments exactly once after the consumed minibatch；
- an accepted task step commits scaled/projected theta、the complete unscaled shadow moments and t_adam+1 exactly once。

## 6. Frozen Training and Checkpoint Protocol

- fit prompts 8,000；checkpoint prompts 1,000；
- target continuation generation：Qwen chat template、enable_thinking=False、greedy T=0、max_new_tokens=128；
- four complete anchors are deterministic evenly spaced offsets；
- one fit pass：8,000 outer steps；
- checkpoint steps：1000、2000、3000、4000、5000、6000、7000、8000；
- T_final=8000；
- matched training seeds [0,1,2]；
- within each seed，A/B/C/D prompt order is identical；
- checkpoint safety gate：
  - harm-rate one-sided 95% upper≤.05；
  - mean-harm one-sided 95% upper≤.10；
  - first-token contrast versus released one-sided 95% lower≥-.005；
- select maximum prompt-balanced EAL among feasible checkpoints；exact tie chooses earliest；
- no feasible checkpoint uses T_final only as diagnostic and linked claims fail；
- checkpoint safety first-token is the per-arm/per-seed contrast `1[aM≥1]-1[aReleased≥1]`；
- all three D seeds require feasible selected checkpoints；if any D seed is diagnostic，C1-EFFICACY fails and the falsifier does not open；
- before falsifier opening，freeze five system definitions and 13 concrete instances：released plus 12 arm/seed hashes，each trained hash labeled `selected_feasible` or `diagnostic_T_final`；never call diagnostic selected；
- an A/B/C diagnostic seed automatically fails each linked C2 contrast but does not block C1 when all D seeds are feasible。

## 7. Prospective Data and Leakage

- split seed 20260806；
- domains in deterministic remainder order [math,code,chat]；
- fit quotas [2667,2667,2666]；
- checkpoint quotas [334,333,333]；
- falsifier quota for n_f assigns floor(n_f/3) to every domain，then remainders to math、then code；
- n_f=max(1500,n_power)；
- normalization is Unicode NFKC、lowercase、collapsed whitespace；Unicode regex tokenization is `\w+|[^\w\s]`；use contiguous 8-grams，or the complete token tuple when fewer than eight；
- exact normalized-hash and exact 8-gram Jaccard≥.5 exclusion against frozen prior index；within-pool inverted postings generate candidates，sorted exact-Jaccard edges feed deterministic union-find；
- component ID is SHA256 of sorted normalized hashes joined with NUL；cross-domain components are excluded before allocation；
- reserve count per domain/split is `max(64,ceil(.10*active_domain_quota))`；bucket order is fit/checkpoint/falsifier × math/code/chat；rank whole components by `SHA256(split_seed||NUL||split||NUL||domain||NUL||component_id)`，then rows by normalized hash and frozen source ordinal；rank material uses UTF-8, lowercase hexadecimal hashes, ASCII decimal seed/ordinal without leading zero, and exactly one NUL between fields；assign complete unused components until exact active and reserve row quotas can be selected；surplus rows remain discarded in that split；independent replay compares ownership、active/reserve/discarded status and order exactly；
- a component cannot cross fit/checkpoint/falsifier；failure to meet exact active/reserve quotas aborts the data gate；no component split or manual replacement；
- minimum target continuation is 19 tokens and every retained prompt has exactly four complete offsets；a short attempted row is permanently consumed and replaced only by the next pre-frozen same-domain/same-split reserve row；reserve exhaustion is a terminal sequence-collection failure class；
- fit/checkpoint sequence generation occurs only after split audit；
- falsifier prompt content/continuation/outcomes remain sealed until checkpoint/source identities freeze；
- one common falsifier producer/evaluator opens and evaluates released plus all 12 frozen hashes；falsifier reserve attempts occur sealed within that same opening and can never enter another split；
- row-level falsifier data or model-specific score is never printed to stdout/stderr。

Allowed POWER_RECEIPT fields are restricted to protocol/source hashes、aggregate counts、cluster-size aggregates、conservative paired prompt-level SD/ICC upper bounds、power formula/version and derived n_power。Means、signs、rows、IDs、checkpoint ranks and downstream outcomes are forbidden。

## 8. Frozen Estimand and Inference

Per block，accepted draft count a∈[0,15]：

- EAL=1+a；
- harm indicator H^M=1[aM<aReleased]；
- harm magnitude=(aReleased-aM)+；
- first-token contrast=1[aM≥1]-1[aReleased≥1]；
- C2 harm contrast=H^D-H^C，not 1[aD<aC]。

Aggregation：

1. equal blocks inside each prompt；
2. for a trained arm，average the three matched-seed prompt metrics；
3. equal prompts for the point estimate；
4. never globally average raw blocks and never pre-average a component；
5. 10,000 paired connected-component cluster bootstrap replicates，domain-stratified，seed 2026080601；
6. a sampled component carries every constituent prompt instance；
7. percentile bootstrap uses linear quantile interpolation；two-sided 95% quantiles are .025/.975，one-sided lower/upper are .05/.95。

Power：

- superiority two-sided alpha .05；
- non-inferiority one-sided alpha .05；
- power≥.80；
- D–released effect +.30 EAL；
- D–A +.10 EAL；
- D–B +.10 EAL；
- D–C harm -.02；
- D–C EAL NI margin -.05；
- first-token NI margin -.005；
- harm-rate one-sided upper-bound distance≤.015；
- mean-harm one-sided upper-bound distance≤.03；
- power computation seed 2026080602。

POWER_RECEIPT supplies conservative paired prompt-level `sd_upper` for each registered contrast plus `icc_upper`、`cv_cluster_size_upper`、`mean_cluster_size_upper` and `sd_mean_harm_upper`。Freeze

`DE=1+(((1+cv_upper^2)*mean_cluster_size_upper)-1)*icc_upper`。

- every two-sided superiority requirement uses `ceil(DE*((z_.975+z_.8)*sd_upper/effect)^2)` for effects [.30,.10,.10,.02]；
- every one-sided NI requirement uses `ceil(DE*((z_.95+z_.8)*sd_upper/distance)^2)` for distances [.05,.005]；
- harm-rate precision uses `ceil(DE*.25*(z_.95/.015)^2)`；
- mean-harm precision uses `ceil(DE*(z_.95*sd_mean_harm_upper/.03)^2)`；
- n_power is the maximum registered integer requirement；three training seeds provide no variance reduction。

## 9. Scientific Claim Gates

C1-EFFICACY passes at G7 only if all hold：

- D–released EAL point≥+.30 and paired 95% CI lower>0；
- D harm-rate 95% upper≤.05；
- D mean-harm 95% upper≤.10；
- D first-token contrast one-sided 95% lower≥-.005；
- every domain D–released EAL point≥0；

C1-SYSTEM/DEPLOYMENT passes after G8 only if merge/runtime graph/dtype/output exactness gates and latency TOST all pass。C1-EFFICACY may be reported without a zero-overhead claim if deployment fails，but the combined system/deployment claim fails。

C2 passes only if all hold：

- D–A EAL point≥+.10 and 95% CI lower>0；
- D–B EAL point≥+.10 and 95% CI lower>0；
- D–C released-referenced harm difference H^D-H^C point≤-.02 and 95% CI upper<0；
- D–C EAL one-sided 95% lower≥-.05。

Current Domino parallel EAL 5.93853 is descriptive background only。No gap-recovery percentage is allowed unless released and Domino are both recomputed under this exact prompt-balanced estimand。

## 10. Engineering and Deployment Gates

Three clean-process A/D pairs use orders A→D、D→A、A→D。Every process runs 20 warmup + 200 timed outer steps on one exclusive A800，resetting peaks after warmup。Every warmup/timed D minibatch asserts four nonempty protected sets，K=4；each receipt emits K and protected-position-count histograms。The engineering-only fixture may derive labels from frozen DFlash logits to force a later mismatch while preserving a nonempty prefix；these labels are never scientific truth。

- A executes only work required by actual D-PACE training；
- D includes frozen reference、task backward、four-row batched VJP、projection/restoration and every exact candidate forward；
- pair median ratio=median(TD)/median(TA)；
- pair p95 ratio=Q95(TD)/Q95(TA)，not p95 of per-step ratios；
- every pair median ratio≤4；
- every pair p95 ratio≤6；
- every D peak allocated≤60 GiB；
- every pair D–A peak allocated≤12 GiB；
- allocated and reserved peaks both reported；
- any OOM、nonfinite、projection/restoration failure closes the route；
- no sampled/truncated fallback。

Capacity uses 32-block micro-overfit then 128 fit prompts/512 blocks。Every constrained commit must be exact feasible。Post-warmup median norm ratio ||grad Ldynamic||/||grad LDPACE|| must lie in [.05,20]；this is an engineering gate, not a theoretical balance claim。

Deployment latency：

- predesignated D seed0 merged checkpoint；
- fixture is the first 50 checkpoint prompts by stable manifest rank，end-to-end speculative generation with `max_new_tokens=64` and exact target output；
- 20 independent process restarts；
- each cycles 200 warmups then executes 50 alternating released/merged pairs；
- measured value is seconds per emitted token；pair statistic is log(merged/released)，and restart value r_j is the median of its 50 pair statistics；
- across-restart estimate is the arithmetic mean of 20 r_j；s is their sample standard deviation with ddof=1；90% CI is `estimate ± t_(0.95,19)*s/sqrt(20)`；
- TOST alpha .05；PASS iff both 90% CI endpoints are strictly inside [log(.98),log(1.02)]。

## 11. Authorization Ladder and Permanent Stops

| Gate | Required Receipt | Opens |
|---|---|---|
| G0 | contract + fresh review GO | local implementation + CPU/mock tests only |
| G1 | unit/parity full pass + fresh code review GO | one synthetic GPU smoke |
| G2 | smoke result review GO | exactly three A/D cost pairs |
| G3 | independent cost PASS | power + split materialization/audit |
| G4 | independent split audit PASS | fit/checkpoint sequence generation |
| G5 | data audit + reviewed capacity wrapper GO | 32/512-block capacity only |
| G6 | capacity result review PASS | 4×3 full training |
| G7 | checkpoint selection + source/identity audit PASS | one common falsifier opening + C1-EFFICACY/C2 adjudication |
| G8 | C1-EFFICACY PASS + frozen D-seed0 selected-feasible identity | first exactly one selected-checkpoint merge/wrapper-removal/trace/dtype/output audit；only its PASS opens the fixed latency fixture；both receipts finalize C1-SYSTEM/DEPLOYMENT |

At every stage，all later stages remain closed。A scientific opening permanently forbids method、threshold、arm、split、seed、checkpoint or estimand modification。Negative/inconclusive outcomes delete claims；they do not authorize rescue experiments。

## 12. Required Machine-Readable Receipts

- contract JSON and SHA256；
- source closure with every implementation/test/wrapper hash；
- unit/parity report；
- code review raw response and disposition；
- GPU smoke JSON；
- per-process cost JSON plus independent adjudication；
- power receipt；
- split manifest/component receipt plus independent replay；
- sequence collection metadata；
- capacity receipts；
- 12 training/checkpoint-selection receipts；
- frozen checkpoint/source identity closure；
- one common prompt-level falsifier bundle；
- bootstrap/claim adjudication；
- merge/trace/latency receipts if opened。

All receipts write atomically to a new run root and include schema version、route、source hashes、input hashes、command/config、hostname/job ID、start/end time、status and failure class。No stage overwrites a prior attempt。
