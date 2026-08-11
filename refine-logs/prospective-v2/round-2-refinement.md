# Round 2 Refinement

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Anchor Check

- Problem Anchor 与前两轮逐字一致。
- 新版只关闭训练算法歧义，没有改变研究对象、部署预算、数据边界或成功标准。
- `safe` 被严格限定为训练 batch 上的 margin feasibility 与 held-out empirical safety；不声称跨 batch 永久保持或未见 prompt 保证。

## Simplicity Check

- 保留一个 lexicographic constrained objective；不再把 D-PACE、breaker hinge、projection 叙述成三个贡献。
- 只训练完整 15-position blocks，删除 validity mask 与 partial-block 分支。
- full FBAC 删除 regression-breaker task branch：不可行 batch 统一由无状态 restoration 处理。
- 不增加 replay buffer、QP solver、teacher KL 或额外网络；多个约束用确定性 sequential half-space projection。

## Changes Made

### 1. 完全闭合 projected optimizer

- 每个 protected position 都是一个 constraint；每轮 sequential projection 扫描所有线性化违反项，稳定排序并处理 ties，而非只看单一 worst constraint。
- 对 infeasible batch 使用独立、无状态、多轮 restoration；只要求每次 exact forward 严格降低最大 violation。restoration 不读取/修改 task Adam moments。
- task AdamW proposal 在 shadow state 中计算；只有 exact-feasible parameter step 被接受时，参数、moments 和 step counter 才事务式提交一次。skip/restoration 均不提交 task state。
- 明确 vacuous、feasible、infeasible 三条分支，以及 float64 dot/norm、sweeps、tolerances、backtracking 与 failure conditions。

### 2. 让四个 arm 可直接实现

- A=`DPACE`：`L_DPACE` + ordinary AdamW。
- B=`STATIC-C`：`L_DPACE + L_static(m_0)` + prefix-constrained optimizer。
- C=`DYNAMIC-U`：`L_DPACE + L_dynamic(m_theta)` + ordinary AdamW。
- D=`FBAC`：`L_DPACE + L_dynamic(m_theta)` + prefix-constrained optimizer。
- 全部使用完整 15-position block、同一 rank/layers/data/order/steps/optimizer hyperparameters；`DPACE_ONLY_PARITY` 是独立测试模式，不再出现未定义 `eta`。

### 3. 完善 sealed validation contract

- 明确定义 harm、mean harm、first-token、prompt clustering、paired harm contrast 和 missing-feasible-checkpoint rule。
- 只白名单一个 producer-train aggregate variance receipt；禁止读取旧 validation/reserved/formal 或任何 row-level旧 outcome。
- power 覆盖所有 Claim-2 confirmatory contrasts；falsifier 数量只能在生成前上调。
- latency 使用 paired log-ratio TOST/90% equivalence CI，不再只看 sample median。

## Revised Proposal

# Research Proposal: FBAC-DFlash——由 speculative prefix 语义导出的主动集约束适配

> 路线身份：`prospective-v2`。这是 R083 已关闭路线之外的一条全新、前瞻式路线；不得把本方案、数据或实验描述成 R083 的重试、修复或下游阶段。

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Technical Gap

### 本地证据

1. Released DFlash EAL `5.1120019436`，K=16 oracle 约 `9.727`；主缺口是把可用候选变成连续正确前缀。
2. Frozen-feature selector 的完整证据已经饱和：GCLS 最佳约 `+0.28499 EAL` 且 harm `7.32%`；FMAS/SAVS/CAMRS 分别出现 identity collapse、约 1,518 倍 harmful-gradient dominance、tail/calibration 失效；27.5M D640 teacher 在 99,356 prompts 上只有 `+0.07799 EAL`，并显著差于 compact d64 (`-0.15063`, CI `[-0.23652,-0.06803]`)。
3. PROS R082 几乎退化为 Direct：beneficial APPLY `174/174`、harmful KEEP `5/101`、harm `6%`，相对 Direct 仅 `+0.003125 EAL`。冻结输出上的门无法安全修复首错。
4. 旧 hard reachable-support route 已被 capacity gate 证伪：full Candidate-D-PACE control hard accuracy `1.0`，hard/soft censoring 只有 `0.940639/0.949772`。因此 FBAC **永远保留 full D-PACE 的 15 位置监督**，不是该关闭路线的重试。
5. released DFlash 有 5 个 parallel draft layers，block size 16 中 position 0 是已知 anchor，实际 draft positions `L=15`。target layers `[1,9,17,25,33]` 经共享 `fc` 融合；本路线不改这个接口。
6. 同锚点 Domino parallel backbone EAL `5.93853`，相对 DFlash 的描述性 gap 为 `0.82653`；训练量不匹配，不能作架构因果推断。

### 前沿边界

- [DFlash](https://arxiv.org/abs/2602.06036)：复用的并行 drafter。
- [D-PACE](https://arxiv.org/abs/2605.18810)：已覆盖 acceptance-aware smooth position weighting；不是本贡献。
- [Domino](https://arxiv.org/abs/2605.29707)：prefix-causal correction；本路线不加顺序 head。
- [DFlare](https://arxiv.org/abs/2606.02091)：逐层 target fusion/异构 KV/progressive loss；本路线不改 fusion。
- [DeLS-Spec](https://arxiv.org/abs/2607.07409)、[DSpark](https://arxiv.org/abs/2607.05147)：顺序/局部 expert 或运行时融合；本路线不加第二 expert/forward。

缺少的是一个由 hard speculative-prefix 语义直接导出的 lexicographic training problem：**先保持 frozen 已接受 prefix 的正 margin 可行性，再在该可行域内学习完整 D-PACE 与当前 breaker 的延长目标。**

## Method Thesis

- One-sentence thesis: 对 released DFlash 做 mergeable LoRA 时，把 frozen accepted prefix 映射为必须满足的 margin half-spaces，把 current hard breaker 映射为唯一增量修复 active set，并在可行域内保留 full D-PACE coverage，可在不改变部署图的情况下优化真实 greedy prefix。
- Why this is the smallest adequate intervention: 一个固定 LoRA 参数化、一个 lexicographic constrained optimizer；无新推理模块、无 suffix censor、无额外 teacher signal。
- Why timely: PEFT 允许对 foundation-model drafter 的 decision boundary 做训练时约束更新，合并后不增加运行时状态或 kernel。

## Contribution Focus

- Dominant contribution: **speculative-prefix-derived active-set constrained adaptation**：repair set 与 feasible set 都由 verification 的 first-mismatch 结构唯一给出。
- Optional supporting contribution: NONE。
- Explicit non-contributions: D-PACE、LoRA、hinge、sequential projection、AdamW、margin protection 都是已知 primitive；不声称 generic optimizer novelty、representation-level causal evidence 或 unseen-data guarantee。

## Proposed Method

### Complexity Budget

- Frozen: Qwen3-4B target；DFlash embedding/LM head/shared `fc`/前三层/norm/原权重。
- Trainable: 最后两层 `q/k/v/o/gate/up/down` 的 rank-16、alpha-16、dropout-0 LoRA，精确 `1,835,008` 参数。
- Optimizer-only: frozen reference branch、constraint gradients、shadow Adam state；部署全部删除。
- Excluded: selector、GRU/expert、layer-wise fusion、top-M KL、EMA/replay、tree/multipath、runtime threshold、第二 draft forward。

### System Overview

```text
complete prospective 15-position block
  -> frozen target context
  -> frozen DFlash logits b: m_0 and protected prefix constraints
  -> adapted DFlash logits z: m_theta and current repair frontier
  -> lexicographic update
       level 0: restore / preserve every batch prefix constraint
       level 1: full D-PACE + one current-breaker margin
  -> merge 1.835M LoRA parameters into released DFlash weights

deployment: unchanged one DFlash parallel forward + one target verification
```

### Core Mechanism

#### Complete-block semantics

Every minibatch contains exactly `N` complete blocks and every block has `L=15` valid target-greedy draft labels `y_{n,1:L}`. There is no partial-block validity mask. All logits, probabilities, margins, dot products and norm reductions used by the objective/projection are computed in float32, except projection dot/norm accumulation is float64.

Let frozen/adapted logits be `b_{n,i}`/`z_{n,i}`. PyTorch float32 `argmax` gives deterministic lowest-ID ties. Define

`m_0(n)=min{i:argmax b_{n,i} != y_{n,i}}`,

`m_theta(n)=min{i:argmax z_{n,i} != y_{n,i}}`,

with value `L+1` if fully correct. Both and all masks are detached and recomputed every forward. Let

`gamma_{n,i}(z)=z_{n,i}(y_{n,i})-max_{v!=y_{n,i}}z_{n,i}(v)`.

The frozen protected set is `P_n={i:i<m_0(n)}`; it is empty for a first-position miss and contains all 15 positions for a fully correct frozen block。

#### Full D-PACE coverage

For each complete block:

`q_{n,i}=softmax(z_{n,i})[y_{n,i}]`,

`s_{n,i}=0.5*q_{n,i}+0.5`,

`p_{n,i}=prod_{j=1}^i s_{n,j}`,

`w_{n,i}=stopgrad(sum_{k=i}^L p_{n,k})`。

The exact reduction is

`L_DPACE=(1/N) sum_{n=1}^N sum_{i=1}^L w_{n,i} CE(z_{n,i},y_{n,i})`。

`DPACE_ONLY_PARITY` disables every frontier term and constraint path；its scalar loss and every LoRA gradient must match the pinned official D-PACE implementation at alpha `.5` within frozen CPU/GPU tolerances。Every suffix position receives its ordinary D-PACE gradient。

#### Repair and feasible sets

Set `epsilon_tie=1e-4`。For every frozen-correct prefix position define

`r_{n,i}=max(epsilon_tie, 0.5*gamma_{n,i}(b))`,

`c_{n,i}(theta)=r_{n,i}-gamma_{n,i}(z_theta) <= 0` for `i in P_n`。

At zero-init，the constraints are feasible except a frozen exact-tie edge case，which enters restoration and is separately counted。

For a feasible batch，the local proposition implies `m_theta(n)>=m_0(n)`。If `m_theta(n)<=L`，the dynamic frontier loss is

`ell_dyn(n)=[epsilon_tie-gamma_{n,m_theta(n)}(z)]_+`；fully correct blocks contribute zero。

`L_dynamic=sum_n ell_dyn(n)/max(sum_n 1[m_theta(n)<=L],1)`。

For the static control，replace `m_theta` by frozen `m_0`：

`L_static=sum_n 1[m_0(n)<=L]*[epsilon_tie-gamma_{n,m_0(n)}(z)]_+ / max(sum_n 1[m_0(n)<=L],1)`。

#### One lexicographic problem

Full FBAC is

`min_theta L_DPACE(theta)+L_dynamic(theta)`

`subject to c_{n,i}(theta)<=0 for every i in P_n of the current minibatch`。

Feasibility has priority over task descent；the constraint is not a weighted regularizer。The application-specific contribution is the derivation of `P_n` and `m_theta` from speculative first-mismatch semantics；the generic projection machinery is not claimed novel。

#### Transactional multi-constraint projected AdamW

Fixed optimizer parameters for all arms are `lr=1e-4`, `beta1=.9`, `beta2=.95`, `eps_adam=1e-8`, `weight_decay=0`, global task-gradient clip `1.0`，4% linear warmup then cosine schedule。Projection constants fixed pre-data are：feasibility tolerance `tau_f=1e-5`，linear tolerance `tau_l=1e-7`，restoration slack `tau_r=1e-4`，at most 4 sequential-projection sweeps，8 restoration cycles，and exact candidate scales `{1,1/2,...,1/128}`。

Task state is `S_t=(t,m_t,v_t)`。A functional proposal never mutates it：

```text
g = clip_global_norm(grad(L_task), 1.0)
m* = .9*m_t + .1*g
v* = .95*v_t + .05*g^2
d0 = -lr_t * (m*/(1-.9^(t+1))) / (sqrt(v*/(1-.95^(t+1)))+1e-8)
S* = (t+1,m*,v*)
```

Because weight decay is exactly zero，`d0` has no decoupled-decay term。

`PROJECT(theta,d0,C)` initializes `d=d0`。For each of at most 4 sweeps：

1. For every current constraint `j=(sample_id,anchor_offset,position)`，compute `g_j=grad c_j(theta)` and linear residual `u_j=c_j(theta)+<g_j,d>`。
2. Sort constraints by descending `u_j`，ties by the stable tuple `j`。
3. Sequentially，for each `u_j>tau_l`，update

   `d <- d - (c_j+<g_j,d>)/(||g_j||_2^2+1e-12) * g_j`。

4. Recompute every linear residual after the sweep。Stop only if `max_j u_j<=tau_l`；after 4 failed sweeps return `NO_LINEAR_SOLUTION`。

All protected constraints—including ties and multiple simultaneous actives—are scanned；no top-K constraint truncation exists。

`EXACT_ACCEPT(theta,d,C)` evaluates the nonlinear constraints at `theta+alpha*d` for `alpha` in `{1,1/2,...,1/128}` and returns the first candidate with `max_j c_j<=tau_f`。Continuity does not guarantee a positive step at zero slack；if none passes，the task proposal is skipped。

The full update has three explicit branches：

```text
C = all protected constraints in this complete-block minibatch

if C is empty:                                      # vacuous
    build (d0,S*) from task gradient
    commit theta+d0 and S* exactly once

elif max(C(theta)) > tau_f:                         # infeasible
    RESTORE_STATELESS(theta,C)
    leave S_t and task step counter unchanged
    after success recompute batch and enter feasible branch;
    after 8 failed cycles abort the run

else:                                               # feasible
    build shadow (d0,S*) from task gradient
    d = PROJECT(theta,d0,C)
    if projection fails: leave theta,S_t unchanged and log skip
    else alpha = EXACT_ACCEPT(theta,d,C)
         if no alpha: leave theta,S_t unchanged and log skip
         else atomically commit theta+alpha*d and S* exactly once
```

`RESTORE_STATELESS` never reads or mutates `S_t`。At each cycle it sets `d=0` and runs the same deterministic sequential half-space sweeps，but targets `c_j+<g_j,d><=-tau_r` for every currently violated/near-active constraint，where near-active is exactly `c_j>=-tau_r`；far-inactive constraints are still included in the final exact feasibility check。It tries the same alpha sequence and accepts only a candidate whose exact maximum positive violation

`V(theta)=max(0,max_j c_j(theta))`

is strictly smaller by at least `1e-7`。After an accepted restoration displacement it recomputes all constraints and repeats；success is `V<=tau_f`。If no alpha decreases `V` or 8 cycles do not restore feasibility，the run aborts as an optimizer failure，not a hypothesis result。Restoration changes only theta；task moments and task step stay unchanged。

This feasibility is explicitly **current-minibatch local**。It may not persist for earlier blocks and carries no held-out guarantee；checkpoint/falsifier harm tests that generalization。

#### Exact fixed-block proposition

If `gamma_i(z)>0` for all `i<m_0` on a fixed block，then adapted greedy equals gold throughout the frozen accepted prefix，so `m_theta>=m_0` and accepted draft length cannot shorten。If also `m_0<=L` and `gamma_{m_0}(z)>0`，then at least one additional draft token is accepted。This is a direct first-mismatch property，not a novel theorem or an unseen-data claim。

### Exact Training Arms

All use identical LoRA、target/base weights、complete blocks、data order、steps、learning-rate schedule and checkpoint times。

- A `DPACE`：`L_task=L_DPACE`，ordinary transactional AdamW，no constraints/restoration。
- B `STATIC-C`：`L_task=L_DPACE+L_static`，constraints `c<=0`，the exact restoration/projected optimizer above。
- C `DYNAMIC-U`：`L_task=L_DPACE+L_dynamic` using its possibly regressed current breaker，ordinary transactional AdamW，no constraints/restoration。
- D `FBAC`：`L_task=L_DPACE+L_dynamic`，constraints `c<=0`，the exact restoration/projected optimizer above。

Released DFlash is the fifth immutable outcome。No arm-specific optimizer or hyperparameter rescue is allowed。

### Optional Supporting Component

NONE。

### Modern Primitive Usage

- Mergeable LoRA is only the low-dimensional parameter carrier；frozen target supplies the same context/gold used by DFlash/D-PACE training。
- After merge there is no adapter operator、external state、extra forward or deployment-time constraint check。

### Integration and Deployment Audit

1. Zero adapter must reproduce released argmax/accepted length exactly and logits within pinned tolerance。
2. Train with frozen target/base branches and transactional optimizer logs。
3. Merge in float32，remove wrappers，save deployment dtype。
4. Adapter-vs-merged audit requires `atol=.02,rtol=.02` plus exact argmax/accepted length on the audit set。
5. Released-vs-merged profiler traces must have the same ordered module/operator names、tensor shapes、operator counts、attention backend、kernel names、dtype、block size and exactly one draft/one target forward；only weight identities/logits may differ。
6. Target verification remains unchanged，so emitted sequence is target-exact。

### Training and Data Plan

1. **Whitelisted power receipt only.** Before selecting new prompts，one audited script may read old **producer-train only** aggregate/raw records and output `POWER_RECEIPT.json` containing only sample counts、paired prompt-level SD/ICC upper bounds for EAL/harm/first-token，source hashes and code hash；it emits no means、directions、row IDs、checkpoint ranks or rows。Old validation/reserved/formal and R083 outcomes remain forbidden。This receipt is the sole old-outcome exception and is frozen before all new manifests。
2. **Power-complete size.** Let `n_f=max(1500,n_power)`，where `n_power` is the maximum 80%-power requirement at two-sided alpha .05 for：FBAC-vs-released EAL MDE `+.30`；FBAC-vs-DPACE `+.10`；FBAC-vs-STATIC-C `+.10`；FBAC-vs-DYNAMIC-U prompt-harm-rate difference `-.02`；and FBAC-vs-DYNAMIC-U EAL non-inferiority margin `-.05`。Use only conservative SD/ICC upper bounds from the receipt。`n_f` may increase before generation and can never decrease。
3. **Prospective split.** Freeze 8,000 fit、1,000 checkpoint、`n_f` falsifier prompts from Open-PerfectBlend remainder。Exclude exact and 8-gram Jaccard `>=.5` overlap against old 100k and hash-only prior exclusion index；no old model score participates。
4. **Component isolation.** Build connected components from normalized full-conversation/document fingerprints and 8-gram overlap；assign entire components to one split，stratified by math/code/chat。Independent replay must pass before target generation。
5. **Capacity gate.** 512 fit blocks test：15-position indexing；D-PACE scalar/all-gradient parity；all suffix gradients nonzero；vacuous/single/multiple/tied constraint projection；nonlinear backtracking；infeasible restoration；task-state invariance on restoration/skip；repeated batch changes；zero-init/merge；all four arms can overfit their intended objective。Algorithm failure is reported separately and closes implementation。
6. **Feasible-first selection.** On checkpoint，an arm is feasible only if block harm `<=5%`，mean harm `<=.10`，and first-token one-sided 95% lower bound `>=-.005` vs released。Choose maximal prompt-balanced EAL among feasible checkpoints，then earliest step on ties。Before training，each arm also designates the final scheduled step `T_final` as diagnostic fallback；if it has no feasible checkpoint，`T_final` is still frozen/evaluated in the common opening，the arm is labeled infeasible，and any claim requiring it automatically fails。
7. **One opening.** Freeze code/source closure、all five model identities and selection receipts before a single falsifier read；materialize all outcomes together。No later arm/seed/checkpoint/threshold is allowed。

### Exact Statistics

For block `j`，let released/arm accepted draft counts be `a_j^0,a_j^M in {0,...,15}`。

- EAL: `1+a_j^M`，reported prompt-balanced（mean blocks within prompt，then mean prompts）。
- Block harm indicator: `h_j^M=1[a_j^M<a_j^0]`；harm rate is mean over all blocks。
- Mean harm magnitude: mean over all blocks of `(a_j^0-a_j^M)_+`，not conditional on harm。
- First-token accuracy: `1[a_j^M>=1]`。
- Per-prompt harm: mean `h_j^M` over that prompt’s blocks。
- Constraint contrast: per-prompt `Delta h_p=h_p^FBAC-h_p^DYNAMIC-U`；claim requires point `<=-.02` and two-sided 95% CI upper `<0`。

Every paired CI resamples **prompt connected-components** as clusters，stratified by domain，with a frozen seed and 10,000 replicates；blocks are never independently resampled。Domain estimates use the same prompt-balanced unit。

### Failure Modes and Diagnostics

- D-PACE parity/suffix-gradient failure：implementation failure，stop。
- Multi-constraint projection/restoration test failure：optimizer failure，stop before scientific capacity inference。
- `>1%` task skips after warmup、median accepted alpha `<1/4`、or any restoration abort：engineering gate failure，close projected route。
- Checkpoint/falsifier harm failure：batch-local feasibility did not generalize；safety claim fails。
- D does not beat A：gain is ordinary D-PACE LoRA；FBAC mechanism fails。
- D does not beat B：dynamic frontier claim fails。
- D does not reduce harm vs C while retaining EAL：constraint claim fails。
- Any domain point estimate negative：no universal claim and no scale。
- Merge/operator/latency equivalence fails：deployment claim fails。

### Novelty and Elegance Argument

FBAC is not “another D-PACE weighting”：D-PACE remains byte-for-byte/parity-equivalent as the uncensored coverage objective。Nor is it generic safe finetuning：the hard repair set `m_theta` and every protected half-space `i<m_0` are derived from speculative verification’s first-mismatch functional，and their conjunction gives the exact fixed-block no-shortening implication。The contribution is narrowly this speculative-prefix construction and its lexicographic use；sequential projection、LoRA and margin hinge are implementation primitives。与 DFlare 相比不改 fusion；与 Domino/DeLS/DSpark 相比不加 causal/runtime expert；与 GCLS/PROS 相比不在 frozen output 上选动作。若 factorial contrasts 不成立，不扩大叙事。

## Claim-Driven Validation Sketch

### Claim 1: Same-graph prospective improvement with bounded empirical harm

- Experiment: one common `n_f`-prompt falsifier for released + A/B/C/D。
- Primary pass: D-vs-released `Delta EAL>=+.30`，95% paired cluster CI lower `>0`。
- Safety: harm `<=5%`；mean harm `<=.10`；first-token one-sided 95% lower `>=-.005`；math/code/chat point estimates all `>=0`。
- Deployment: identical traces/forward counts；paired latency uses alternating model order across 1,000 timed pairs after 200 warmups under fixed exclusive-A800 clocks。Apply TOST at alpha .05 to per-pair log latency ratio；the 90% CI must lie wholly in `[log(.98),log(1.02)]`。

### Claim 2: Dynamic active set and prefix feasibility each contribute beyond matched D-PACE LoRA

- D vs A: paired EAL point `>=+.10` and 95% CI lower `>0`。
- D vs B: paired EAL point `>=+.10` and 95% CI lower `>0`，supporting dynamic rather than frozen frontier。
- D vs C: prompt-harm difference point `<=-.02` and 95% CI upper `<0`，while EAL one-sided 95% lower `>=-.05`，supporting constraint without material utility loss。
- All are confirmatory and included in `n_power`；failure deletes the corresponding claim。

### Descriptive Domino accounting

Report `Delta EAL/(5.93853-5.11200)` as descriptive parallel-gap recovery only；training differences prohibit causal architecture claims。

## Experiment Handoff Inputs

- Must-prove: C1 same-graph prospective gain/safety；C2 D>A、D>B、D lower-harm/noninferior-EAL vs C。
- Must-test first: optimizer unit matrix including multiple/tied/infeasible/rejected-state cases；official D-PACE parity；zero/merge equivalence。
- Critical data: 8k/1k/`n_f` component-disjoint prospective prompts；complete 15-position blocks；one sealed opening。
- Highest risks: sequential projection cost/stall；two-layer LoRA capacity；batch-local constraints not generalizing；power-driven `n_f`/compute expansion。

## Compute & Timeline Estimate

- GPU: optimizer/capacity `5–10 A800 GPU-h`；prospective collection约 `12–30`；four arms含 projection约 `40–100`；falsifier/profile `8–16`。总计约 `65–156 A800 GPU-h`，随 `n_f` 上调。
- Data: no human labels；one scalar-only producer-train power receipt and fully prospective target T0 continuations。
- Timeline: implementation/tests 1–2 days；capacity 0.5–1 day；collection/training 2–4 days；sealed evaluation/audit 1 day；total 5–8 days。任一 gate 失败即停。
