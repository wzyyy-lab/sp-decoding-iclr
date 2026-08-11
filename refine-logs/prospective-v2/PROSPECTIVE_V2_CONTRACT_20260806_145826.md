# prospective-v2 Execution Contract

**Route**：FBPF-DFlash prospective-v2  
**Version**：1  
**Frozen at**：2026-08-06 14:58 +08:00  
**Status**：FROZEN_PENDING_G0_REVIEW  
**Sources**：FINAL_PROPOSAL.md + EXPERIMENT_PLAN.md  
**Old contract**：idea-stage/docs/research_contract.md remains an immutable GCLS artifact and does not govern this route。

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
- trainable native LoRA only on draft layers 3 and 4, modules q/k/v/o/gate/up/down；
- rank 16、alpha 16、scale 1、dropout 0；
- A matrix uses deterministic seeded initialization；B matrix initializes exactly zero；
- trainable count must equal：
  - per layer 917,504；
  - total 1,835,008；
- no peft dependency；native wrapper must expose disable、state_dict、functional override and float32 merge；
- base/target weights bf16；objective logits/margins float32；projection dot/norm reductions float64；
- merge computes Wmerged=float32(Wbase)+B@A，then removes every wrapper；
- zero adapter must be bitwise-equal in logits、argmax and accepted length；adapter/merged tolerance atol=.02、rtol=.02 plus exact argmax/length。

## 4. Frozen Batch and Objective

- outer minibatch：one prompt、four deterministic complete anchors，N=4；
- predicted positions：15，anchor position excluded from loss；
- all argmax ties choose lowest vocabulary ID；
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
- restoration changes theta only and never commits task moments/t_adam；
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
  - first-token one-sided 95% lower≥-.005；
- select maximum prompt-balanced EAL among feasible checkpoints；exact tie chooses earliest；
- no feasible checkpoint uses T_final only as diagnostic and linked claims fail；
- all three D seeds require feasible selected checkpoints；
- before falsifier opening，freeze 12 selected trained checkpoint hashes plus released checkpoint fingerprint and one source closure。

## 7. Prospective Data and Leakage

- split seed 20260806；
- domains in deterministic remainder order [math,code,chat]；
- fit quotas [2667,2667,2666]；
- checkpoint quotas [334,333,333]；
- falsifier quota for n_f assigns floor(n_f/3) to every domain，then remainders to math、then code；
- n_f=max(1500,n_power)；
- exact normalized-hash exclusion and 8-gram Jaccard≥.5 exclusion against frozen prior index；
- exact within-pool 8-gram postings + exact Jaccard + union-find define connected components；
- a component cannot cross fit/checkpoint/falsifier；
- prompt with continuation shorter than 16 tokens is replaced only by pre-frozen same-domain reserve order；
- fit/checkpoint sequence generation occurs only after split audit；
- falsifier prompt content/continuation/outcomes remain sealed until checkpoint/source identities freeze；
- one common falsifier producer/evaluator opens and evaluates released plus all 12 checkpoints；
- row-level falsifier data or model-specific score is never printed to stdout/stderr。

Allowed POWER_RECEIPT fields are restricted to protocol/source hashes、aggregate counts、cluster-size aggregates、conservative paired prompt-level SD/ICC upper bounds、power formula/version and derived n_power。Means、signs、rows、IDs、checkpoint ranks and downstream outcomes are forbidden。

## 8. Frozen Estimand and Inference

Per block，accepted draft count a∈[0,15]：

- EAL=1+a；
- harm indicator=1[aM<aReleased]；
- harm magnitude=(aReleased-aM)+；
- first-token=1[a≥1]。

Aggregation：

1. equal blocks inside each prompt；
2. for a trained arm，average the three matched-seed prompt metrics；
3. equal prompts for the point estimate；
4. never globally average raw blocks and never pre-average a component；
5. 10,000 paired connected-component cluster bootstrap replicates，domain-stratified，seed 2026080601；
6. a sampled component carries every constituent prompt instance。

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
- harm-rate one-sided 95% half-width≤.015；
- mean-harm half-width≤.03；
- power computation seed 2026080602。

## 9. Scientific Claim Gates

C1 passes only if all hold：

- D–released EAL point≥+.30 and paired 95% CI lower>0；
- D harm-rate 95% upper≤.05；
- D mean-harm 95% upper≤.10；
- D first-token one-sided 95% lower≥-.005；
- every domain D–released EAL point≥0；
- merged trace/output gates pass；
- latency TOST passes if zero-overhead deployment is claimed。

C2 passes only if all hold：

- D–A EAL point≥+.10 and 95% CI lower>0；
- D–B EAL point≥+.10 and 95% CI lower>0；
- D–C harm-rate point≤-.02 and 95% CI upper<0；
- D–C EAL one-sided 95% lower≥-.05。

Current Domino parallel EAL 5.93853 is descriptive background only。No gap-recovery percentage is allowed unless released and Domino are both recomputed under this exact prompt-balanced estimand。

## 10. Engineering and Deployment Gates

Three clean-process A/D pairs use orders A→D、D→A、A→D。Every process runs 20 warmup + 200 timed outer steps on one exclusive A800，resetting peaks after warmup。

- A executes only work required by actual D-PACE training；
- D includes frozen reference、task backward、complete batched VJP、projection/restoration and every exact candidate forward；
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
- 20 independent process restarts；
- each 200 warmups + 50 alternating released/merged pairs；
- one median paired log ratio per restart；
- TOST alpha .05；90% CI wholly inside [log(.98),log(1.02)]。

## 11. Authorization Ladder and Permanent Stops

| Gate | Required Receipt | Opens |
|---|---|---|
| G0 | contract + fresh review GO | local implementation only |
| G1 | unit/parity full pass + fresh code review GO | one synthetic GPU smoke |
| G2 | smoke result review GO | exactly three A/D cost pairs |
| G3 | independent cost PASS | power/split materialization |
| G4 | independent split audit PASS | fit/checkpoint sequence generation |
| G5 | data audit + capacity PASS | 4×3 full training |
| G6 | checkpoint selection + source/identity audit PASS | one falsifier opening |
| G7 | C1 result-to-claim PASS | deployment latency |

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
