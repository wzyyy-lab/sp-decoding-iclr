# Round 4 Refinement

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Anchor Check

- Anchor 与 round 0 逐字相同。
- 唯一变化是把逐位置 projector rows 按 block 做精确等价的 `max` 聚合；科学问题、约束集合和所有 claim 不变。
- 名称从不准确的 “sign-cone” 改为 **First-Break Prefix-Feasible Adaptation (FBPF)**。

## Simplicity Check

- `max_i c_i <= tau_f` 当且仅当所有 `c_i <= tau_f`，因此不是 constraint approximation、sampling 或 top-K。
- projector/restoration 每个非空 block 只产生一行，`N=4` 时最多 4 行；exact candidate 仍检查全部最多 60 个位置。
- exact ties 用 block 内所有最大项梯度的均值（max 的确定性合法次梯度）形成一行；非线性全位置检查负责最终安全。
- 没有新增模块、arm、loss 或 hyperparameter。

## Changes Made

### 1. O(N) blockwise-max projection

- 定义 `C_n=max_{i in P_n} c_{n,i}`；constraint vector 从最多 60 行降到最多 4 行，feasible set 完全相同。
- 用 batched VJP 对每个 block 的 exact-max tie mask 求一行梯度；最多四个 vectorized VJPs，不 materialize 60-row Jacobian。
- exact backtracking/restoration 每次仍重新评估全部 per-position `c`，任何 active-position switch 或不利 tie 都不能漏过。

### 2. Benchmark 与推断最后收口

- A/D throughput 在独立 clean processes 中运行三个 counterbalanced pairs；reset peak memory，计入 base branch、VJP、projection、所有 feasibility forwards，并报告绝对时间与比值。
- 指标明确为 blocks→prompt→connected component，再以 component 为 cluster bootstrap 单位。
- superiority power 用 two-sided alpha .05；non-inferiority power 用 one-sided alpha .05，与最终检验一致。
- 固定记录 breaker/D-PACE gradient norm ratio；严重失衡是 capacity failure，不允许事后调权重。

## Revised Proposal

# Research Proposal: FBPF-DFlash——First-Break Prefix-Feasible Adaptation

> `prospective-v2` 是与已关闭 R083 完全隔离的前瞻路线，绝不表示为 R083 retry/rescue/downstream。

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Technical Gap

### Local evidence

- Released DFlash EAL is `5.1120019436`, while K=16 oracle is about `9.727`: available alternatives do not become a safe greedy prefix.
- Frozen-feature correction is exhausted: GCLS peaks near `+0.28499 EAL` with `7.32%` harm; FMAS/SAVS/CAMRS show identity collapse, 1,518× harmful-gradient imbalance or tail/calibration failure; a 27.5M D640 teacher reaches only `+0.07799 EAL` on 99,356 prompts and underperforms compact d64 by `0.15063` (CI `[-0.23652,-0.06803]`).
- PROS R082 is almost Direct (`174/174` beneficial APPLY, `5/101` harmful KEEP, `6%` harm, only `+0.003125 EAL` over Direct).
- Hard reachable-support censoring is already closed: full Candidate-D-PACE hard capacity accuracy `1.0`, censored variants `0.940639/0.949772`. FBPF keeps full D-PACE at all 15 positions.
- DFlash block size 16 means one known anchor plus `L=15` predicted positions. Its five draft layers share target fusion; FBPF changes no topology/fusion.
- Same-anchor Domino parallel EAL `5.93853` defines only a descriptive gap of `0.82653`; unequal training forbids causal attribution.

### Primary-source boundary

- [DFlash](https://arxiv.org/abs/2602.06036): reused parallel drafter.
- [D-PACE](https://arxiv.org/abs/2605.18810): existing acceptance-aware smooth weighting.
- [Domino](https://arxiv.org/abs/2605.29707): causal correction head, excluded here.
- [DFlare](https://arxiv.org/abs/2606.02091): per-layer fusion/KV change, excluded here.
- [DeLS-Spec](https://arxiv.org/abs/2607.07409), [DSpark](https://arxiv.org/abs/2607.05147): runtime sequential/local experts, excluded here.

The focused gap is to adapt the drafter’s output decisions without trading away the exact sign decisions that form its already accepted prefix.

## Method Thesis

- Thesis: derive a minibatch feasible region from positive gold margins on frozen-accepted positions, repair the adapted current first miss, and take uncensored D-PACE task steps only when exact all-position feasibility is preserved.
- Smallest intervention: one fixed mergeable LoRA plus one training-only prefix-feasible optimizer; no inference addition.
- Modern fit: parameter-efficient foundation-model adaptation merges into the released DFlash graph.

## Contribution Focus

- Dominant contribution: verifier-specific construction of (a) a frozen-prefix feasible set from `i<m_0` and (b) a current repair frontier from `m_theta`, with an exact fixed-block no-shortening implication.
- Optional contribution: NONE.
- Non-contributions: LoRA, D-PACE, positive margins, hinge, max aggregation, cyclic projection, AdamW and constants are known/design choices; no generic optimizer novelty, representation-causal claim or unseen safety theorem.

## Proposed Method

### Complexity Budget

- Frozen: Qwen3-4B target and all released DFlash parameters.
- Trainable: rank-16/alpha-16/dropout-0 LoRA on last-two-layer `q/k/v/o/gate/up/down`; exact `1,835,008` parameters.
- Deployment: LoRA merged in float32 and wrappers removed; original single draft forward + target verification.
- Excluded: selector, GRU/expert, fusion change, teacher KL, EMA/replay, tree, threshold, extra deployed forward.

### System Overview

```text
N=4 complete blocks x 15 draft labels
 -> frozen target features
 -> frozen DFlash: m_0 / P_n
 -> adapted DFlash: m_theta
 -> full D-PACE + current-breaker repair
 -> <=4 blockwise-max constraint VJP rows
 -> transactional projected AdamW + exact all-position check
 -> merged DFlash
```

### Core Definitions

Use complete `N=4`, `L=15` blocks only. Objective/logit decisions are float32; projection dot/norm reductions are float64. Frozen/adapted logits are `b,z`; all argmax ties select lowest vocabulary ID. The detached non-gold winner defines

`gamma_{n,i}(z)=z_{n,i}(y_{n,i})-max_{v!=y}z_{n,i}(v)`.

`m_0`/`m_theta` are first frozen/adapted mismatches, or `L+1`; all indices/masks detach and refresh each forward. `P_n={i:i<m_0(n)}`.

### Full D-PACE Coverage

For every position:

`q=softmax(z)[y]`, `s=.5q+.5`, `p_i=prod_{j<=i}s_j`, `w_i=stopgrad(sum_{k>=i}p_k)`,

`L_DPACE=(1/N)sum_n sum_i w_{n,i} CE(z_{n,i},y_{n,i})`.

No suffix mask exists. `DPACE_ONLY_PARITY` pins D-PACE commit `f36bad6e6b0f9f5b59e1e6cf405c705b46d2b43f` and the implementation hash. Tolerances: CPU float64 scalar/per-gradient `atol=1e-10,rtol=1e-9`; GPU float32 scalar `atol=rtol=2e-5`; flattened LoRA-gradient max absolute difference `<=5e-5`, cosine `>=.999999`.

### Prefix Feasibility and Frontier

Fix `epsilon_tie=1e-4`, `tau_f=1e-5`. Per protected position,

`c_{n,i}=epsilon_tie-gamma_{n,i}(z)`.

Implemented feasibility is `c_{n,i}<=tau_f`, implying `gamma>=9e-5>0`. For a feasible batch, `m_theta>=m_0`.

`L_dynamic=mean_[m_theta<=L] [epsilon_tie-gamma_{n,m_theta}]_+`,

`L_static=mean_[m_0<=L] [epsilon_tie-gamma_{n,m_0}]_+`,

with empty mean zero. Unit scaling is frozen engineering choice. Full FBPF minimizes `L_DPACE+L_dynamic` subject to all prefix constraints, feasibility first.

### Exact Blockwise Equivalence

For each non-vacuous block define

`C_n(theta)=max_{i in P_n} c_{n,i}(theta)`.

`C_n<=tau_f` iff every `c_{n,i}<=tau_f`; therefore the feasible set is identical to the per-position formulation. The projector vector has only `K=#nonvacuous blocks<=4` outputs.

At current theta, let `T_n={i:c_{n,i}=C_n}` use exact float32 equality. Its deterministic max subgradient is the uniform average of all exact-tied gradients:

`G_n=(1/|T_n|)sum_{i in T_n} grad c_{n,i}`.

Build all at-most-four rows in one `torch.autograd.grad(..., is_grads_batched=True)` call using block/tie weight masks; no constraint position is sampled. Exact nonlinear candidate checks always evaluate the full per-position vector, so an active-position switch or adverse tie cannot be accepted silently. Tie tests require that an unsafe chosen/averaged-subgradient proposal is rejected or restored, never accepted.

### Transactional Optimizer

All arms use `lr_peak=1e-4`, betas `(.9,.95)`, eps `1e-8`, weight decay 0, global task clip 1.0, 4% warmup/cosine. `k_outer` indexes consumed minibatches/LR and always advances once; `t_adam` indexes accepted task commits/bias correction only.

Shadow proposal:

`m*=.9m+.1g`, `v*=.95v+.05g^2`,

`d0=-lr(k_outer) [m*/(1-.9^(t_adam+1))]/[sqrt(v*/(1-.95^(t_adam+1)))+1e-8]`.

Only an accepted task step commits `(theta+alpha*d,m*,v*,t_adam+1)` once. Skip leaves task state unchanged.

For feasible non-vacuous blocks, cyclically project `d0` against at most four linear rows. Each of at most four sweeps sorts sweep-start residual `u=C+Gd`, then recomputes current `u_n` immediately before `d<-d-u_n G_n/(||G_n||^2+1e-12)` when `u_n>tau_l=1e-7`. Exhaustion is `PROJECTION_BUDGET_EXHAUSTED`, not infeasibility.

Exact candidates `alpha={1,1/2,...,1/128}` are accepted only if **all per-position** constraints recomputed at `theta+alpha*d` satisfy `<=tau_f`; else task skip.

If the current batch is infeasible, stateless restoration performs at most eight relinearized cycles with block residual `C_n+tau_r+G_nd`, `tau_r=1e-4`, current residual recomputed before each update. It accepts only an exact all-position candidate reducing `V=max(0,max_n C_n-tau_f)` by `>=1e-7`; failure aborts. Restoration changes theta only, never task moments/t_adam. Vacuous blocks take ordinary task step. Feasibility is batch-local, not persistent/unseen guarantee.

### Exact Arms

- A `DPACE`: `L_DPACE`, ordinary AdamW.
- B `STATIC-PF`: `L_DPACE+L_static`, prefix-feasible optimizer.
- C `DYNAMIC-U`: `L_DPACE+L_dynamic`, ordinary AdamW.
- D `FBPF`: `L_DPACE+L_dynamic`, prefix-feasible optimizer.
- Released DFlash is immutable fifth outcome.

Same data/order/outer steps/rank/layers/schedule/checkpoint budget; no private rescue.

### Throughput and Capacity Gates

Throughput uses three counterbalanced A/D pairs; each arm of each pair runs in a separate clean process on the same exclusive A800, with 20 warmup + 200 timed outer steps. Reset peak CUDA stats after warmup. Timings include target/base branch, task backward, batched VJP, projection/restoration and every exact feasibility forward. Report absolute median/p95 seconds, peak allocated/reserved, and paired ratios.

Pass requires complete `K<=4` rows, no truncation; D/A median ratio `<=4.0`, p95 `<=6.0`; peak allocated `<=60 GiB` and D-A `<=12 GiB`; no OOM/nonfinite/projection/restoration failure. Failure closes the engineering route without constraint sampling.

Then 512 fit blocks test all parity/index/tie/counter/state/merge invariants and arm memorization. Record `||grad L_dynamic||/||grad L_DPACE||` initially and every 50 outer steps. Nonfinite, warmup-post median `<.05` or `>20` is a capacity failure; coefficient remains 1 and is never rescued.

### Integration and Deployment Audit

- Zero adapter exact argmax/accepted-length equality.
- Float32 merge; wrapper deletion; adapter-vs-merged `atol=.02,rtol=.02` and exact argmax/length.
- Released-vs-merged identical ordered module/operator/kernel names, shapes/counts, dtype, attention path, block size and one-draft/one-target forward.
- Target verification preserves exact emitted output.

### Prospective Data and Power

One whitelisted producer-train-only `POWER_RECEIPT.json` may contain source/code hashes, counts and conservative paired prompt-level SD/ICC upper bounds only—no means/signs/rows/IDs/checkpoint ranks. Old validation/reserved/formal/R083 outcomes remain forbidden.

Set `n_f=max(1500,n_power)`. Use two-sided alpha `.05` power for superiority effects D-vs-released `+.30 EAL`, D-vs-A `+.10`, D-vs-B `+.10`, D-vs-C harm `-.02`; use one-sided alpha `.05` for D-vs-C EAL noninferiority `-.05` and first-token noninferiority `-.005`. Require 80% power at the stated true effects, plus one-sided 95% harm-rate half-width `<=.015` and mean-harm half-width `<=.03`. Power rejection probability and mandatory observed point thresholds are separate.

Freeze 8,000 fit/1,000 checkpoint/`n_f` falsifier OPB-remainder prompts, excluding exact and 8-gram-Jaccard `>=.5` overlap with old 100k/hash-only prior index. Assign entire full-conversation/document near-duplicate connected components to one split, stratified by domain; independently replay before target generation.

Checkpoint feasibility: harm-rate 95% upper `<=.05`, mean-harm 95% upper `<=.10`, first-token one-sided 95% lower `>=-.005`; maximize EAL among feasible, earliest tie. Predesignated `T_final` is diagnostic fallback if none and dependent claims fail. Freeze all five identities/source closure/receipts, then one common falsifier opening.

### Metrics and Hierarchical Inference

For accepted draft counts `a_j^M in [0,15]`: `EAL=1+a`; block harm `1[a^M<a^0]`; mean harm `(a^0-a^M)_+` averaged over all blocks; first-token `1[a>=1]`.

Aggregation is explicitly:

1. equal-weight blocks within each prompt → one prompt metric;
2. equal-weight prompts within each connected component → one component metric;
3. component-cluster resampling, stratified by domain, for 10,000 frozen-seed paired bootstrap replicates.

No prompt/component is weighted by its block count.

Latency uses 20 independent process restarts. Each has 200 warmups and 50 alternating-order released/merged pairs, emitting one median paired log ratio. TOST alpha `.05` on 20 restart values requires the 90% CI wholly inside `[log(.98),log(1.02)]`.

### Failure Modes

- parity/VJP/tie/counter/restoration tests fail: implementation stop;
- throughput/memory gate fails: engineering close, no scientific conclusion;
- safety gate fails: batch feasibility did not generalize;
- D not > A: ordinary D-PACE LoRA explains gain;
- D not > B: dynamic frontier unsupported;
- D not lower-harm/noninferior-EAL vs C: prefix-feasibility claim unsupported;
- domain/merge/trace/latency fails: corresponding scope/system claim blocked.

### Novelty and Elegance Argument

FBPF keeps D-PACE unchanged as uncensored coverage. Its single proposal is to derive an exact prefix-feasible set from the verifier’s frozen accepted positions and an active target from the adapted current first miss; blockwise max is only an exact computational representation of that set. LoRA, margin scale and projection are disclosed primitives. It adds neither DFlare fusion nor Domino/DeLS/DSpark runtime causality nor GCLS/PROS frozen-output selection. Paper value requires the cost gate and all sealed factorial contrasts; failure closes rather than expands the method.

## Claim-Driven Validation Sketch

### Claim 1: Same-graph prospective improvement with bounded empirical harm

- D-vs-released EAL point `>=+.30`, paired 95% CI lower `>0`.
- Harm-rate 95% upper `<=.05`; mean-harm 95% upper `<=.10`; first-token one-sided 95% lower `>=-.005`; each domain point `>=0`.
- Trace/forward equality and restart-level latency TOST within `±2%`.

### Claim 2: Dynamic frontier and prefix feasibility each matter

- D-vs-A EAL point `>=+.10`, 95% CI lower `>0`.
- D-vs-B EAL point `>=+.10`, 95% CI lower `>0`.
- D-vs-C prompt-harm point `<=-.02`, 95% CI upper `<0`, EAL one-sided 95% lower `>=-.05`.

All are pre-powered and jointly opened. Report `Delta EAL/(5.93853-5.11200)` only as descriptive Domino-parallel gap recovery.

## Experiment Handoff Inputs

- Must prove: throughput feasibility, C1, D>A, D>B, D lower-harm/noninferior vs C.
- Must test first: D-PACE parity; blockwise/full-position equivalence; exact ties/active switches; counters/restoration/skip; merge trace.
- Data: 8k/1k/`n_f`, component-disjoint complete blocks, one opening.
- Main risks: batched VJP/feasibility overhead, unit task-gradient imbalance, batch-local generalization, power-driven data size.

## Compute & Timeline Estimate

- Gate/capacity `6–14 A800 GPU-h`; collection `12–35`; four arms bounded by gate `40–120`; falsifier/profile `10–20`; total `68–189 A800 GPU-h` depending on `n_f`.
- No human labels; target T0 continuation only.
- 5–8 days; every engineering/scientific gate fails closed.

