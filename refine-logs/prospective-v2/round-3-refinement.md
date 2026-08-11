# Round 3 Refinement

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Anchor Check

- Problem Anchor 逐字保留；没有改变模型、部署、数据或成功边界。
- 方法 claim 从宽泛“representation repair”缩至 verifier-induced sign constraints + current-breaker repair；LoRA 只是实现载体。
- 训练可行性仍是 capacity 前的工程 gate，不能把 optimizer 失败误报为科学负结果。

## Simplicity Check

- 删除 `0.5 * frozen margin`；prefix 只要求足以保证 unique-correct 的最小正 margin。
- 不加 constraint sampling/top-K/replay/QP；固定 4-block optimizer minibatch，全部最多 60 条约束一次向量化求 Jacobian。
- cyclic sweeps 复用同一线性化 Jacobian；只在 exact restoration cycle 才重新求值/求导。
- 五个 sealed outcomes 不变；没有新增模型或实验维度。

## Changes Made

### 1. 最小 verifier-induced sign cone

- 对 frozen accepted prefix 只约束 `gamma >= epsilon_tie - tau_f > 0`；`epsilon_tie` 与 task scale 明示为固定工程选择，不声称由 first-mismatch 唯一推出。
- implemented feasible set 明确为 `c <= tau_f`，与数值检查完全一致。

### 2. 约束 Jacobian 成本可审计

- 固定 `N=4` 完整 blocks，最多 `K=60` 约束；用 pinned `torch.func.jacrev(..., chunk_size=4)` 得到完整 `K x 1,835,008` float32 Jacobian，不截断约束。
- 每个 sweep 开头只用 residual 排序；每条 constraint 真正更新前重新计算当前 `c_j + G_j d`。
- capacity 先做相对 D-PACE throughput/peak-memory gate：median `<=4x`、p95 `<=6x`、peak `<=60 GiB` 且不超过 baseline `+12 GiB`；失败关闭路线。

### 3. 完整 counters/restoration/power/latency

- `k_outer` 管数据与 LR schedule，每个消费的 minibatch 都前进；`t_adam` 只在 task step 原子提交时前进并用于 bias correction。
- restoration residual 明确为 `c + tau_r + Gd`；restoration/skip 不动 task moments。
- power 加入 first-token non-inferiority 和 absolute harm precision；说明 power 与点估计门是不同事件。
- latency 改为 20 个独立进程 restart 的 batch-level paired log ratios 做 TOST。
- D-PACE parity 的 commit、scalar/gradient tolerance 全部数值固定。

## Revised Proposal

# Research Proposal: FBAC-DFlash——Verifier-Induced Sign-Cone Adaptation

> 路线身份：`prospective-v2`。这是 R083 已关闭路线之外的一条全新、前瞻式路线；不得把本方案、数据或实验描述成 R083 的重试、修复或下游阶段。

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Technical Gap

### Evidence boundary

1. Released DFlash EAL is `5.1120019436`; K=16 oracle is about `9.727`. Candidate availability is large, but realized greedy prefix is short.
2. Frozen-output correction has saturated: best GCLS is about `+0.28499 EAL` with `7.32%` harm; FMAS/SAVS/CAMRS fail through identity collapse, harmful-gradient imbalance, or tail/calibration; a 27.5M frozen-feature teacher gives only `+0.07799 EAL` on 99,356 prompts and is significantly worse than compact d64 (`-0.15063`, CI `[-0.23652,-0.06803]`).
3. PROS R082 is almost Direct (`174/174` beneficial APPLY, `5/101` harmful KEEP, `6%` harm, only `+0.003125 EAL` over Direct), so a larger frozen gate does not solve the boundary.
4. Hard reachable-support censoring is already scientifically closed: full Candidate-D-PACE capacity hard accuracy `1.0`, hard/soft censoring `0.940639/0.949772`. FBAC retains full 15-position D-PACE gradients and cannot be described as a retry.
5. Released DFlash has five parallel draft layers; block size 16 contains one known anchor plus `L=15` predicted positions. Target layers `[1,9,17,25,33]` share one `fc` fusion. FBAC changes no fusion or topology.
6. Same-anchor Domino parallel backbone EAL is `5.93853`, making the descriptive DFlash-to-parallel gap `0.82653`; differing training scale forbids causal architecture attribution.

### Literature boundary

- [DFlash](https://arxiv.org/abs/2602.06036) supplies the reused parallel drafter.
- [D-PACE](https://arxiv.org/abs/2605.18810) already supplies smooth acceptance-aware position weights.
- [Domino](https://arxiv.org/abs/2605.29707) adds prefix-causal correction; FBAC does not.
- [DFlare](https://arxiv.org/abs/2606.02091) changes per-layer target fusion/KV; FBAC does not.
- [DeLS-Spec](https://arxiv.org/abs/2607.07409) and [DSpark](https://arxiv.org/abs/2607.05147) add sequential/local experts or runtime fusion; FBAC does not.

The missing intervention is narrowly defined: preserve every sign decision that creates frozen DFlash’s accepted prefix while repairing the current first non-positive gold margin, without removing ordinary D-PACE coverage or changing inference.

## Method Thesis

- One-sentence thesis: A parallel drafter can be adapted in a verifier-aligned way by mapping frozen accepted positions to positive-margin half-spaces and the current first miss to one repair frontier, then taking full-D-PACE task steps only inside that minibatch sign cone.
- Smallest adequate intervention: one fixed mergeable LoRA and one training-only constrained optimizer; no inference component or new teacher signal.
- Foundation-model fit: PEFT confines the update to 1.835M parameters and can be merged into the original foundation-model drafter graph.

## Contribution Focus

- Dominant contribution: **verifier-induced sign-cone adaptation** for a parallel speculative drafter: `i<m_0` constructs the feasible set and `m_theta` constructs the active repair frontier.
- Optional contribution: NONE.
- Non-contributions: D-PACE, LoRA, hinge, positive margin, cyclic projection, AdamW and the exact constants are known/design choices; no generic optimizer, representation-causality or unseen-safety claim.

## Proposed Method

### Complexity Budget

- Frozen: Qwen3-4B target; released DFlash embedding/LM head/shared `fc`/first three draft layers/norm/original weights.
- Trainable: rank-16, alpha-16, dropout-0 LoRA on `q/k/v/o/gate/up/down` of the last two draft layers; exact `P=1,835,008` parameters.
- Training-only: frozen reference branch, complete constraint Jacobian, shadow Adam state and exact feasibility forwards.
- Excluded: selector, RNN/expert, fusion change, teacher KL, EMA/replay, tree, threshold, second deployed forward.

### System Overview

```text
4 complete prospective blocks, each 15 draft labels
  -> frozen target features
  -> frozen DFlash: m_0 and prefix sign constraints
  -> LoRA DFlash: m_theta and current repair
  -> full uncensored D-PACE task
  -> vectorized complete constraint Jacobian
  -> transactional sign-cone projected step
  -> merge LoRA

deployment = original one parallel draft forward + one target verification
```

### Core Mechanism

#### Exact block and margin semantics

Each optimizer minibatch has exactly `N=4` complete blocks and every block has `L=15` valid T=0 target-greedy draft labels. All objective/model decision arithmetic is float32; projection dot products/norms accumulate in float64.

For frozen/adapted logits `b_{n,i}`/`z_{n,i}`, float32 `argmax` uses deterministic lowest vocabulary ID. The non-gold competitor `v^-_{n,i}=argmax_{v!=y} z_{n,i}(v)` is also deterministically selected and detached. Define

`gamma_{n,i}(z)=z_{n,i}(y_{n,i})-z_{n,i}(v^-_{n,i})`,

`m_0(n)=first i with argmax b_{n,i} != y_{n,i}`, and `m_theta(n)` analogously, with `L+1` if fully correct. Indices/masks are detached and recomputed every forward. Protected set is `P_n={i:i<m_0(n)}`.

#### Exact full D-PACE

For all 15 positions,

`q_{n,i}=softmax(z_{n,i})[y_{n,i}]`,

`s_{n,i}=0.5*q_{n,i}+0.5`,

`p_{n,i}=prod_{j<=i}s_{n,j}`,

`w_{n,i}=stopgrad(sum_{k=i}^L p_{n,k})`,

`L_DPACE=(1/N) sum_n sum_i w_{n,i} CE(z_{n,i},y_{n,i})`.

No suffix is masked. `DPACE_ONLY_PARITY` pins `third_party/D-PACE` commit `f36bad6e6b0f9f5b59e1e6cf405c705b46d2b43f` and the implemented source hash before capacity. Required parity is:

- float64 CPU synthetic scalar/each gradient: `atol=1e-10`, `rtol=1e-9`;
- float32 GPU real mini-batch scalar: `atol=2e-5`, `rtol=2e-5`;
- flattened LoRA gradient: max absolute difference `<=5e-5` and cosine `>=0.999999`.

#### Minimal sign constraints and repair

Fix `epsilon_tie=1e-4` and numerical feasibility tolerance `tau_f=1e-5`. For every `i in P_n`,

`c_{n,i}(theta)=epsilon_tie-gamma_{n,i}(z_theta) <= tau_f`.

Thus every accepted implementation point has `gamma>=epsilon_tie-tau_f=9e-5>0`, which is sufficient for unique gold argmax on the frozen prefix. `epsilon_tie` is a fixed numeric design choice, not claimed to be uniquely derived.

If the batch is feasible, `m_theta>=m_0`. For any `m_theta<=L`,

`L_dynamic = mean_over_breaker_blocks [epsilon_tie-gamma_{n,m_theta}(z)]_+`.

For the static control,

`L_static = mean_over_blocks_with_m0<=L [epsilon_tie-gamma_{n,m_0}(z)]_+`.

Empty means are exactly zero. The unit coefficient in `L_DPACE+L_dynamic/static` is fixed before data and is an implementation choice.

Full FBAC solves the lexicographic problem

`min L_DPACE + L_dynamic`, subject to all current-minibatch `c<=tau_f`,

where feasibility precedes task descent. Only construction of the sign cone/frontier from first-mismatch semantics is contribution-specific.

#### Counters and functional AdamW proposal

All arms consume the same ordered outer minibatches. `k_outer` indexes data and the 4%-warmup/cosine LR schedule and advances once after every consumed minibatch, including a skip. `t_adam` counts only atomically accepted task updates and supplies bias correction. Stateless restoration changes neither counter until its enclosing minibatch finishes; it never changes moments. Fixed optimizer values are `lr_peak=1e-4`, `beta=(.9,.95)`, `eps=1e-8`, `weight_decay=0`, task global-norm clip `1.0`.

Given clipped task gradient `g`, the uncommitted shadow state is

`m*=.9m+.1g`, `v*=.95v+.05g^2`,

`d0=-lr(k_outer) * [m*/(1-.9^(t_adam+1))] / [sqrt(v*/(1-.95^(t_adam+1)))+1e-8]`.

Only an accepted task step commits `(theta+alpha*d, m*, v*, t_adam+1)` once. Task skip leaves theta/m/v/t_adam unchanged but advances `k_outer`; ordinary arms accept each finite task proposal.

#### Complete vectorized Jacobian and cyclic projection

There are `K=sum_n |P_n|<=60` constraints. At current `theta`, materialize the complete vector `c in R^K` and Jacobian `G in R^(K x P)` in float32 with pinned `torch.func.jacrev(constraint_vector, chunk_size=4)`; no row is sampled or truncated. Worst-case storage is about 0.41 GiB (`60*1,835,008*4` bytes). The same `G` is reused for all four linear sweeps.

For task displacement `d=d0`, each sweep:

1. compute sweep-start `u=c+Gd`; stable-sort by descending `u`, ties by `(sample_id,anchor_offset,position)`;
2. for each row in that order, **recompute current** `u_j=c_j+G_j d` after all earlier updates;
3. if `u_j>tau_l=1e-7`, set `d <- d-u_j G_j/(||G_j||^2+1e-12)`;
4. after the sweep recompute all `u`; success is `max u<=tau_l`.

Four exhausted sweeps return `PROJECTION_BUDGET_EXHAUSTED`, not an infeasibility proof. The task step is skipped transactionally.

For linear success, exact nonlinear checks try `alpha in {1,1/2,...,1/128}` and accept the largest with every recomputed `c(theta+alpha*d)<=tau_f`; otherwise skip. At zero-slack, no positive alpha is assumed to exist.

#### Stateless restoration

If current exact `max c>tau_f`, before any task proposal run at most 8 restoration cycles. Each cycle recomputes full `c,G`, initializes `d=0`, and cyclically targets slack residual

`u_j^R=c_j+tau_r+G_j d`, with `tau_r=1e-4`.

Sweep ordering uses sweep-start `u^R`; the current `u_j^R` is recomputed immediately before every update `d<-d-u_j^R G_j/(||G_j||^2+1e-12)`. Exact candidates use the same alpha list and are accepted only if `V=max(0,max c)` decreases by at least `1e-7`. Success is `max c<=tau_f`; no decreasing candidate or 8 exhausted cycles aborts as `RESTORATION_FAILURE`. Restoration commits theta only and never moments/t_adam.

Vacuous `K=0` commits the ordinary functional task proposal without projection. Feasibility is batch-local and may not persist on earlier or unseen blocks.

#### Fixed-block property

Because implemented feasibility gives positive margin for all `i<m_0`, adapted greedy cannot fail before frozen greedy on that fixed block. Making the frozen/current breaker margin positive extends it by at least one. This elementary property scopes the constraints; it is not an unseen-data theorem.

### Exact Arms

- A `DPACE`: `L_DPACE`, ordinary AdamW, no constraints.
- B `STATIC-C`: `L_DPACE+L_static`, exact sign-cone optimizer.
- C `DYNAMIC-U`: `L_DPACE+L_dynamic` at its current breaker even if regressed, ordinary AdamW.
- D `FBAC`: `L_DPACE+L_dynamic`, exact sign-cone optimizer.
- Released DFlash: immutable fifth outcome.

All share data/order/outer steps/rank/layers/LR schedule/checkpoint budget. No private rescue.

### Projection Throughput Gate

Before any scientific capacity interpretation, benchmark A and D on identical 512 fit blocks, `N=4`, 20 untimed + 200 timed outer steps. D passes only if:

- complete `K<=60` Jacobian is used with no truncation;
- median per-outer-step walltime ratio D/A `<=4.0` and p95 ratio `<=6.0`;
- peak CUDA allocated memory `<=60 GiB` and D-minus-A `<=12 GiB`;
- no OOM/nonfinite, `PROJECTION_BUDGET_EXHAUSTED`, or restoration failure.

Failure closes projected FBAC as an engineering route; it cannot be rescued with constraint sampling/top-K.

### Optional Supporting Component

NONE.

### Integration and Deployment

1. Zero adapter reproduces released argmax/accepted length exactly.
2. Train with frozen target/base and full transactional traces.
3. Merge LoRA in float32, remove wrappers, save deployment dtype.
4. Adapter-vs-merged logits: `atol=.02,rtol=.02`, exact argmax/accepted length on audit set.
5. Released-vs-merged ordered operator/module/kernel names, shapes/counts, dtype, attention path, block size and one-draft/one-target forward must be identical; only weights/logits differ.
6. Ordinary target verification preserves exact target output.

### Data, Power and Training Plan

1. **Sole old-data exception:** one audited producer-train-only script emits `POWER_RECEIPT.json` with source/code hashes, counts, and conservative upper bounds on prompt-level paired SD/ICC for EAL, harm, mean harm and first-token; no means, signs, rows, IDs or checkpoint ranks. Old validation/reserved/formal/R083 outcomes remain forbidden.
2. **Falsifier size:** `n_f=max(1500,n_power)`. `n_power` is the maximum 80%-power requirement at two-sided alpha .05 for true effects: D-vs-released EAL `+.30`; D-vs-A `+.10`; D-vs-B `+.10`; D-vs-C prompt-harm `-.02`; D-vs-C EAL noninferiority margin `-.05`; first-token noninferiority margin `-.005`. It also ensures one-sided 95% harm-rate CI half-width `<=.015` and mean-harm CI half-width `<=.03`. Power means probability of rejecting the stated null under that true effect; point-estimate gates remain separately mandatory. `n_f` can only increase before generation.
3. **Prospective manifest:** 8,000 fit, 1,000 checkpoint and `n_f` falsifier prompts from OPB remainder; exact and 8-gram Jaccard `>=.5` exclusions against old 100k/hash-only prior index.
4. **Components:** connected components of normalized full-conversation/document fingerprints and 8-gram overlap are wholly assigned to one split, stratified by domain; independent replay before target generation.
5. **Capacity:** after throughput gate, 512 fit blocks test D-PACE parity, suffix gradients, all optimizer branches/counters/state invariance, multi/tied constraints, restoration, exact merge and arm-specific memorization. Optimizer failure is not scientific evidence.
6. **Selection:** checkpoint feasible iff harm-rate 95% upper `<=.05`, mean-harm 95% upper `<=.10`, and first-token one-sided 95% lower `>=-.005`. Maximize checkpoint prompt-balanced EAL among feasible, earliest on ties. `T_final` is frozen diagnostic fallback if none; the arm is labeled infeasible and dependent claims fail.
7. **One opening:** freeze source closure, checkpoints, selection/power receipts and all thresholds; materialize released+A/B/C/D outcomes together exactly once.

### Exact Metrics and Inference

For block accepted draft counts `a_j^M in [0,15]`:

- `EAL=1+a_j^M`;
- harm `h_j^M=1[a_j^M<a_j^0]`;
- mean harm magnitude `mean_j (a_j^0-a_j^M)_+` over all blocks;
- first-token accuracy `1[a_j^M>=1]`;
- D-vs-C paired harm contrast is per-prompt `mean_blocks(h^D-h^C)`.

First average blocks within each prompt connected-component. All 10,000-replicate paired CIs resample these components, stratified by domain, under frozen seeds. Blocks are never independent units.

Latency uses 20 independent process restarts. Each restart performs 200 warmups then 50 alternating-order released/merged timed pairs and emits one median paired log ratio. TOST at alpha .05 uses the 20 restart-level values; its 90% CI must lie inside `[log(.98),log(1.02)]` under fixed exclusive A800 clocks.

### Failure Modes

- parity/Jacobian/counter/restoration unit failure: implementation stop;
- throughput/memory gate failure: engineering close, no scientific inference;
- checkpoint/falsifier safety fail: sign constraints did not generalize;
- D not > A: ordinary D-PACE LoRA explains gain;
- D not > B: dynamic frontier unsupported;
- D not lower-harm/noninferior-EAL vs C: constraint claim unsupported;
- any domain point negative: universal/scale claim blocked;
- merge/trace/latency fail: same-graph systems claim fails.

### Novelty and Elegance Argument

FBAC keeps D-PACE exactly as full coverage and does not claim new weighting. Its only proposed contribution is constructing a sign-feasible update cone from the verifier’s already accepted prefix and an active repair target from its current first miss. `epsilon_tie`, the unit task coefficient, LoRA and projection are disclosed design/implementation choices. This differs from DFlare’s fusion change, Domino/DeLS/DSpark’s runtime causal experts, and GCLS/PROS frozen-output selection. Venue value is conditional on the vectorized optimizer passing its cost gate and the sealed factorial contrasts being decisive; otherwise the route closes without adding modules.

## Claim-Driven Validation Sketch

### Claim 1: Prospective EAL gain within empirical safety and identical deployment graph

- D-vs-released EAL point `>=+.30`, paired 95% CI lower `>0`.
- Harm-rate 95% upper `<=.05`; mean-harm 95% upper `<=.10`; first-token one-sided 95% lower `>=-.005`; all domain point estimates `>=0`.
- Identical trace/forward audit and restart-level latency TOST within `±2%`.

### Claim 2: Both dynamic frontier and sign cone add value beyond matched controls

- D-vs-A EAL point `>=+.10`, 95% CI lower `>0`.
- D-vs-B EAL point `>=+.10`, 95% CI lower `>0`.
- D-vs-C prompt-harm point `<=-.02`, 95% CI upper `<0`, and EAL one-sided 95% lower `>=-.05`.
- Every contrast is powered and opened together; failure deletes its exact claim.

### Domino accounting

Report `Delta EAL/(5.93853-5.11200)` only as descriptive parallel-gap recovery.

## Experiment Handoff Inputs

- Must prove: throughput viability; C1; D>A, D>B, D lower-harm/noninferior vs C.
- Must test first: pinned D-PACE parity; full KxP Jacobian; stale-residual/counters/restoration/skip invariants; zero/merge trace.
- Data: 8k/1k/`n_f`, component-disjoint, complete 15-position blocks, one opening.
- Highest risks: jacrev runtime/memory, cyclic projection skip rate, batch-local generalization, power-driven `n_f`.

## Compute & Timeline Estimate

- Gate/capacity: `8–16 A800 GPU-h`; collection `12–35`; four arms bounded by the `<=4x` gate `50–140`; falsifier/profile `10–20`; total `80–211 A800 GPU-h`, scaling with `n_f`.
- No human labels; only target T0 continuations and one scalar-only power receipt.
- 5–9 days: implementation/tests 2, gate/capacity 1, collection/training 2–5, sealed audit 1. Fail closed at every gate.

