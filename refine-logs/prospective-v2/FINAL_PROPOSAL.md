# Research Proposal: FBPF-DFlash——First-Break Prefix-Feasible Adaptation

> Execution contract v3 correction frozen at 2026-08-06 15:41 +08:00. The method thesis is unchanged; the tensor layout, data allocator, transaction, inference, deployment substage, and authorization semantics below supersede the ambiguous earlier wording.

> prospective-v2 是与已关闭 R083 完全隔离的前瞻路线，绝不表示为 R083 retry、rescue 或 downstream。

## Problem Anchor

- Bottom-line problem: 在保持单链、一次 target verification 和显著低于顺序 GRU 的新增推理开销下，提高 released DFlash 在真实、未见 prompt 上的接受前缀长度，并解释/解决其相对 Domino 的主要差距。
- Must-solve bottleneck: frozen-feature external selectors saturate around +0.08–0.28 EAL while surrogate improves but realized EAL/harm worsens; representation doesn’t expose safe first-break correction.
- Non-goals: no tree/multipath; no old R083 retry; no old downstream data; no posthoc threshold rescue; not merely reproduce DFlare layer fusion or add Domino/DeLS RNN.
- Constraints: Qwen3-4B released DFlash, target frozen, single-chain greedy T0 first, user cluster, prospective data only, output exactness.
- Success condition: new clean falsifier shows raw EAL positive CI vs DFlash, safe harm/first-token/domains; matched D-PACE LoRA control shows boundary mechanism; merged adapter latency neutral; otherwise close.

## Technical Gap

本地证据把问题定位在 draft representation，而不是候选可用性：

- released DFlash EAL 为 5.1120019436，K=16 candidate oracle 约 9.727；
- frozen-feature GCLS 最好约 +0.28499 EAL，但 harm 7.32%；FMAS、SAVS、CAMRS 分别出现 identity collapse、1518 倍 harmful-gradient imbalance 或 tail/calibration failure；
- 27.5M 参数的 D640 frozen selector 在 99,356 prompts 上仅 +0.07799 EAL，且比 compact d64 低 0.15063，95% CI 为 [-0.23652,-0.06803]；
- PROS R082 几乎退化为 Direct：beneficial APPLY 174/174，harmful KEEP 5/101，harm 6%，相对 Direct 仅 +0.003125 EAL；
- full Candidate-D-PACE hard capacity accuracy 为 1.0，而 hard/soft reachable-support censoring 仅 0.940639/0.949772。因此新方法不得屏蔽 suffix supervision。

同一 anchor 下 Domino parallel EAL 5.93853 只用于描述背景差距。除非 released DFlash 与 Domino 都能在同一 prospective falsifier、同一 prompt-balanced estimand 下重算，否则不报告 gap-recovery 比例，也不作因果归因。

## Method Thesis and Contribution

FBPF 在 frozen DFlash 已正确的 prefix 上构造 verifier-induced feasible set，同时修复 adapted model 当前 first mismatch；uncensored D-PACE task step 只有在 exact all-position feasibility 保持时才能提交。

唯一主贡献是：

1. frozen accepted prefix 定义保护集合；
2. adapted current first miss 定义动态修复 frontier；
3. blockwise maximum 以严格等价形式把每个 block 的全部约束压为一行；
4. transactional projected AdamW 用 exact nonlinear check 决定提交或跳过。

LoRA、D-PACE、margin、hinge、maximum、cyclic projection 和 AdamW 都不是 novelty claim。部署时 LoRA 合并回原 DFlash，推理图不增加 selector、RNN、expert、fusion、tree 或额外 forward。

## Model and Trainable Scope

- target：frozen Qwen3-4B；
- drafter：released DFlash，除 LoRA 外全部 frozen；
- block：一个已知 anchor 加 L=15 个 predicted positions；
- outer minibatch：一个 source prompt 产生四个 tensor rows；每行恰有一个完整 anchor block，因此 tensor batch size=N=4、anchors/row=1；
- LoRA：精确路径 `layers.{3,4}.self_attn.{q,k,v,o}_proj` 与 `layers.{3,4}.mlp.{gate,up,down}_proj`，rank 16，alpha 16，dropout 0；
- trainable parameters：精确 1,835,008；
- LoRA A/B master parameter 与 Adam moments 均为 float32；按完整 module path 排序，用独立 CPU generator 和 `lora_init_seed=2026080600+training_seed` 执行 `A: kaiming_uniform_(a=sqrt(5))`，B exact zero；同 seed 四 arms bitwise 同初始化且不推进 global RNG；记录 per-module A/B hashes；
- LoRA branch 用 float32 计算并在相加前 cast 到 base-output dtype；attention implementation 精确固定为 `sdpa`；
- deployment：仅在 merge accumulation 使用 float32，即 `(W_base.float()+scale*(B@A)).to(W_base.dtype)`；stored/runtime weights 仍为 bf16，删除 wrapper，保留与 released 相同的 single draft forward + one target verification。

## Core Definitions

target 独立于任何 draft/comparator 自回归生成 greedy continuation；决策 logits 显式转 float32，所有 argmax tie 选择最低 vocabulary ID。对 continuation 长度 `T`，固定

```text
maximum = T - 16
offsets[j] = int(round(j * maximum / 3)), j=0,1,2,3
```

其中 `round` 是 Python rounding，并 assert 四个 offset 互异且都构成完整 b16 block。对 offset `o_n`，anchor 是 `continuation[o_n]`，`y_{n,i}=continuation[o_n+i]`，`i=1,...,15`；target context/features 只含 anchor 之前的 token。Released/adapted DFlash、Domino 或任何 comparator 都不得定义或筛选 gold。

训练 objective logit 与 margin 采用 float32；projection dot/norm reduction 采用 float64。

对 block n、位置 i，gold label 为 y，adapted logits 为 z，定义 detached non-gold winner index

\[
v^-_{n,i}=\operatorname{stopgrad}\arg\max_{v\ne y}z_{n,i}(v),
\qquad
\gamma_{n,i}=z_{n,i}(y)-z_{n,i}(v^-_{n,i}).
\]

frozen/adapted first mismatch 分别为 m0 与 mθ；若无 mismatch，则为 L+1。m0、mθ、non-gold winner 以及所有 mask/index 都 detach，并在每次 forward 后刷新。

\[
P_n=\{i:i<m_0(n)\}
\]

是 frozen DFlash 已接受的 protected prefix。

## Full D-PACE Objective

对四个 tensor rows 的全部 15 个位置，不使用 suffix mask：

\[
q_i=\operatorname{softmax}(z_i)_{y_i},\quad
s_i=0.5q_i+0.5,\quad
p_i=\prod_{j\le i}s_j,\quad
w_i=\operatorname{stopgrad}\sum_{k\ge i}p_k,
\]

\[
\mathcal L_{\mathrm{DPACE}}
=\frac1N\sum_n\sum_i w_{n,i}\operatorname{CE}(z_{n,i},y_{n,i}).
\]

这里 `source_prompts_per_outer=1`、`tensor_batch_size=4`、`anchors_per_row=1`、`reduction_divisor=N=4`。因此 8,000 prompts × 4 blocks / batch 4 恰为 8,000 outer steps，并与 pinned D-PACE 的 `/bsz` reduction 一致；禁止使用 `bsz=1, anchors=4`。

DPACE_ONLY_PARITY 固定 D-PACE commit f36bad6e6b0f9f5b59e1e6cf405c705b46d2b43f 以及本地实现 hash。CPU float64 scalar/gradient tolerance 为 atol 1e-10、rtol 1e-9；GPU float32 scalar tolerance 为 atol=rtol=2e-5；flattened LoRA gradient max absolute difference 不超过 5e-5，cosine 不低于 .999999。

## Dynamic Repair and Exact Prefix Feasibility

固定 epsilon_tie=1e-4、tau_f=1e-5：

\[
c_{n,i}=\epsilon_{\mathrm{tie}}-\gamma_{n,i}.
\]

对所有 i∈Pn 要求 c≤tau_f，因此 protected gold margin 至少 9e-5，且在当前完整 block 上有 mθ≥m0。

\[
\mathcal L_{\mathrm{dynamic}}
=\operatorname{mean}_{m_\theta\le L}
[\epsilon_{\mathrm{tie}}-\gamma_{n,m_\theta}]_+,
\]

\[
\mathcal L_{\mathrm{static}}
=\operatorname{mean}_{m_0\le L}
[\epsilon_{\mathrm{tie}}-\gamma_{n,m_0}]_+.
\]

空集合 mean 定义为 0。系数 1 是预先冻结的 engineering choice，不作理论 balance claim，不允许结果后调权。

每个非空 protected block 定义

\[
C_n(\theta)=\max_{i\in P_n}c_{n,i}(\theta).
\]

C_n≤tau_f 当且仅当该 block 全部 per-position constraints 均满足，因此 feasible set 与原 formulation 完全相同。K 为非空 block 数，K≤4。

exact maximum tie set

\[
T_n=\{i:c_{n,i}=C_n\}
\]

使用 float32 exact equality；tie mask 完全 detach。确定性 subgradient 为所有 tied gradients 的等权均值：

\[
G_n=\frac1{|T_n|}\sum_{i\in T_n}\nabla c_{n,i}.
\]

把所有 c 展平，并以 detached block/tie masks 构造 grad_outputs；一次 torch.autograd.grad 调用、is_grads_batched=True，返回至多四行 G。任何 candidate 都重新计算全部 per-position c，所以 adverse tie 或 active-position switch 只能触发 backtracking、skip 或 restoration failure，不能静默提交 unsafe update。

## Transactional Optimizer

所有 arms 使用 lr peak 1e-4、betas (.9,.95)、eps 1e-8、weight decay 0、global task-gradient clip 1.0、4% warmup + cosine。

- k_outer：每消费一个 minibatch 加一，控制 LR schedule；
- t_adam：仅 accepted task commit 加一，控制 bias correction；
- task skip：parameter、moment、t_adam 全部不变；
- restoration：只改 parameter，不改 task moment 与 t_adam。

shadow moments 与原始方向为

\[
m^\star=.9m+.1g,\quad v^\star=.95v+.05g^2,
\]

\[
d_0=-\eta(k_{\mathrm{outer}})
\frac{m^\star/(1-.9^{t_{\mathrm{adam}}+1})}
{\sqrt{v^\star/(1-.95^{t_{\mathrm{adam}}+1})}+10^{-8}}.
\]

当前 batch feasible 时，最多四个 cyclic sweeps；每个 sweep 按 sweep-start residual 排序，但每次更新前重新计算当前 u=C+Gd。若 u_n>tau_l=1e-7，则执行

\[
d\leftarrow d-u_nG_n/(\lVert G_n\rVert^2+10^{-12}).
\]

候选 alpha∈{1,1/2,…,1/128} 只有在 theta+alpha d 的全部 per-position constraints 精确重算后均不超过 tau_f 才提交。提交时一次性写入 parameter、完整 shadow moments 与 t_adam+1。projection budget 耗尽是 engineering failure，不解释为 infeasibility。

若当前 batch infeasible，从 batch-start parameters 开始最多八次 relinearized restoration cycles，目标 residual 为 C+tau_r+Gd，tau_r=1e-4；restoration 不改变 moments 或 t_adam。每个 intermediate restoration 只能在 exact violation

\[
V=\max(0,\max_nC_n-\tau_f)
\]

至少下降 1e-7 时暂存。八轮后必须达到 exact all-position feasibility；否则恢复 batch-start theta 并终止该训练 run。成功 restoration 后，在 restored parameters 上重新计算 logits、全部 detached masks/indices、task gradient、constraints 与 VJP，再在同一 minibatch 尝试普通 projected task transaction。若该 task step skip，保留 restored parameters，但 moments 与 t_adam 仍保持 batch-start 值；若 commit，则按正常规则一次性提交 shadow moments 与 t_adam+1。稳定并列的 projection row 顺序固定为 block row 0、1、2、3；k_outer 在整个 minibatch 结束时恰增加一次。测试必须覆盖 restore+commit、restore+skip 与失败 rollback。restoration 本身不是 accepted task commit。

## Exact Arms

| Arm | Task loss | Optimizer |
|---|---|---|
| A DPACE | full D-PACE | ordinary AdamW |
| B STATIC-PF | D-PACE + static frontier | prefix-feasible optimizer |
| C DYNAMIC-U | D-PACE + dynamic frontier | ordinary AdamW |
| D FBPF | D-PACE + dynamic frontier | prefix-feasible optimizer |
| Released | immutable released DFlash | none |

四个训练 arms 使用相同 prospective data、order、outer steps、LoRA scope、schedule 与 checkpoint budget，无 private rescue。D–A 检验完整机制，D–B 隔离 dynamic frontier，D–C 隔离 prefix feasibility。

## Pre-Science Implementation and Cost Gates

在任何正式 data generation 或科学训练前，必须通过：

1. D-PACE scalar/gradient parity；
2. blockwise-max 与 all-position feasibility 等价测试；
3. detached tie/non-gold masks、averaged tie subgradient、active-switch rejection 测试；
4. restoration、skip、accepted-step 的 parameter/moment/counter transaction 测试；
5. zero-adapter、merge、argmax、accepted-length 与 runtime-trace invariants。

然后在同一独占 A800 上运行三个 counterbalanced A/D process pairs。每个 arm 每个 pair 都在独立 clean process 中执行 20 warmup + 200 timed outer steps，warmup 后 reset CUDA peak stats。每个 warmup/timed D minibatch 必须有四个非空 protected sets，assert K=4，并在 receipt 输出 K 与 protected-position-count histograms。

A 只能执行真实 D-PACE training 所需工作，不允许为了缩小 ratio 人为加入 frozen reference branch。D 必须计入 frozen/reference forward、四行 batched VJP、projection/restoration 以及全部 exact feasibility forwards。这个纯 engineering fixture 可从 frozen released-DFlash logits 构造 labels，使 forced mismatch 之前与 base 匹配并保证 K=4；这些 labels 不是 scientific truth，不能进入 capacity、training 或 evaluation。

对每个 pair 分别计算：

- median ratio = median(T_D)/median(T_A)；
- p95 ratio = Q95(T_D)/Q95(T_A)，不是 per-step ratio 的 p95。

三个 pairs 都必须满足 median ratio≤4.0、p95 ratio≤6.0、D peak allocated≤60 GiB、D–A peak allocated≤12 GiB，且无 OOM、nonfinite、projection 或 restoration failure。报告全部 pair 值、三对汇总、peak allocated 与 peak reserved；reserved 仅诊断，不设未审查的 numeric threshold。失败即关闭 engineering route，不允许 constraint sampling/truncation。

在 512 fit blocks 上记录初始以及每 50 outer steps 的

\[
\lVert\nabla L_{\mathrm{dynamic}}\rVert/
\lVert\nabla L_{\mathrm{DPACE}}\rVert.
\]

post-warmup median 小于 .05 或大于 20 属 capacity failure；这是固定 engineering gate，不是理论平衡保证。

## Prospective Data and One Opening

唯一允许的 POWER_RECEIPT 来自 whitelisted producer-train aggregate，仅可含 source/code hashes、counts、conservative paired prompt-level SD/ICC upper bounds；不得包含 means、signs、rows、IDs 或 checkpoint ranks。旧 validation、reserved、formal 与 R083 outcomes 永久禁用。

\[
n_f=\max(1500,n_{\mathrm{power}}).
\]

冻结 8,000 fit、1,000 checkpoint、n_f falsifier active prompts，来自 OPB remainder，并同时预分配 reserves。规范化固定为 Unicode NFKC、lowercase、连续 whitespace collapse；tokenization 使用 Unicode regex `\w+|[^\w\s]`，形成连续 8-gram，少于 8 tokens 时使用完整 token tuple。倒排 postings 只生成候选对，再以 exact Jaccard≥.5 建边；候选边排序后 deterministic union-find。component ID 是其 sorted normalized hashes 以 NUL 连接后的 SHA256。排除与旧 100k/hash-only index 的 exact/Jaccard overlap，并在分配前排除跨 domain components。

每个 domain/split 的 reserve 数为 `max(64, ceil(0.10 * active_domain_quota))`。split/domain 顺序固定为 fit、checkpoint、falsifier × math、code、chat；component rank 为 `SHA256(split_seed || NUL || split || NUL || domain || NUL || component_id)`，row rank 再附加 normalized hash 与 frozen source ordinal。rank material 的 byte encoding 固定为 UTF-8：hash 使用 lowercase hexadecimal、seed/ordinal 使用无前导零 ASCII decimal、字段之间恰有一个 NUL byte。按 rank 把完整、尚未使用的 components 依次归属当前 bucket，直至其 rows 足以提供精确 active quota 与 reserve quota；按 row rank 取前 active quota、再取 reserve quota，其余 rows 只能标为该 split discarded。独立 replay 必须逐行核对 component ownership、active/reserve/discarded status 与 order。任何 component 只能归属一个 split；若不能得到 exact active/reserve quotas，则 data gate 失败，禁止拆 component 或人工替换。

target continuation 至少 19 tokens，保留 prompt 必须恰有四个完整 blocks。短 continuation 的 attempted row 永久 consumed，并自动取同 domain、同 split 的下一个 pre-frozen reserve；reserve exhaustion 是 sequence receipt 的 terminal failure class，不得重新分配。falsifier reserve 的生成与替补属于同一次 opening，所有尝试保持 sealed，且不得转入其他 split。

checkpoint gate（每个 arm/seed 相对 released 独立计算）：

- harm-rate one-sided 95% upper≤.05；
- mean-harm one-sided 95% upper≤.10；
- first-token contrast one-sided 95% lower≥-.005；
- 在 feasible checkpoint 中最大化 EAL，完全相同则取最早 checkpoint；
- 若无 feasible checkpoint，只能冻结预指定 T_final 并标注 `diagnostic_T_final`，不得称为 selected；dependent contrast 自动失败。

五个 system definitions（Released/A/B/C/D）实例化 13 个 concrete model instances：released 加 12 个 arm/seed hashes；每个训练实例明确标注 `selected_feasible` 或 `diagnostic_T_final`。若任一 D seed 为 diagnostic，C1-EFFICACY 立即失败且 falsifier 不开启；若 A/B/C 某 seed 为 diagnostic，只使涉及该 arm 的 C2 contrast 自动失败，只要三个 D seeds 均 feasible，C1 仍可继续。冻结全部 identities、source closure、split 与 receipts 后，仅允许一次 common falsifier opening。

## Frozen Estimand and Hierarchical Inference

对 model M、block j，accepted draft count a_j^M∈[0,15]：

- EAL block value：1+a；
- released-referenced harm indicator：H^M=1[a^M<a^Released]；
- mean-harm value：(a^Released-a^M)_+；
- first-token contrast：1[a^M≥1]-1[a^Released≥1]；
- C2 D–C harm contrast：H^D-H^C，而不是 1[a^D<a^C]。

primary estimand 固定为 prompt-balanced：

1. 每个 prompt 内对其所有完整 blocks 等权平均，得到一个 prompt metric；
2. point estimate 是全部 prompts 的等权平均；
3. 绝不对全体 raw blocks 直接平均，也不先把 component 压成一个 component mean。

paired uncertainty 使用 10,000 frozen-seed connected-component cluster bootstrap replicates，按 domain 分层。每次抽到一个 component 时，携带其全部 prompt metrics，replicate statistic 仍对被抽中的 prompt instances 等权。component 只定义不可拆分的 resampling cluster，不改变 point estimand。CI 使用 percentile bootstrap 与 `quantile(method="linear")`：two-sided 95% 为 .025/.975；one-sided lower/upper 分别为 .05/.95。

POWER_RECEIPT 为每个注册 contrast 提供 conservative paired prompt-level `sd_upper`，并提供 `icc_upper`、`cv_cluster_size_upper`、`mean_cluster_size_upper` 与 `sd_mean_harm_upper`。固定

\[
DE=1+(((1+cv_{upper}^2)\,\bar m_{upper})-1)\,ICC_{upper}.
\]

对 two-sided superiority contrasts D–released EAL .30、D–A EAL .10、D–B EAL .10、D–C released-referenced harm .02，逐项计算 `ceil(DE*((z_.975+z_.8)*sd_upper/effect)^2)`；对 one-sided NI contrasts D–C EAL distance .05 与 first-token distance .005，逐项计算 `ceil(DE*((z_.95+z_.8)*sd_upper/distance)^2)`。另计算 harm-rate upper-bound precision `ceil(DE*.25*(z_.95/.015)^2)` 与 mean-harm upper-bound precision `ceil(DE*(z_.95*sd_mean_harm_upper/.03)^2)`。n_power 是所有项与 1500 之外的最大整数需求，不利用三个 training seeds 做方差折减。harm-rate 与 mean-harm precision 分别定义为 one-sided upper CI bound 到 point 的距离≤.015 与≤.03。

## Claim Gates

### C1-EFFICACY: prospective improvement with bounded empirical harm

- D–released EAL point≥+.30 且 paired 95% CI lower>0；
- harm-rate 95% upper≤.05；
- mean-harm 95% upper≤.10；
- first-token contrast one-sided 95% lower≥-.005；
- 每个 domain 的 EAL point≥0；

### C1-SYSTEM/DEPLOYMENT: released-topology deployment

- float32-accumulated/bf16-stored merge 后 wrapper 全部删除；
- merged runtime graph、ordered operators、dtypes、one-draft/one-target forward 与 released 相同；
- output exactness 与 latency equivalence gate 均通过。

### C2: dynamic frontier and prefix feasibility both matter

- D–A EAL point≥+.10 且 95% CI lower>0；
- D–B EAL point≥+.10 且 95% CI lower>0；
- D–C released-referenced harm difference `H^D-H^C` point≤-.02 且 95% CI upper<0；
- D–C EAL one-sided 95% lower≥-.05。

### Deployment latency equivalence

固定 latency fixture 是 checkpoint manifest stable rank 最前 50 个 prompts，end-to-end speculative generation `max_new_tokens=64`，并要求 emitted output 与 target exact。运行 20 个独立 process restarts；每个 restart 先用这些 prompts 循环 200 warmups，再运行 50 个 alternating-order released/merged pairs。每个 pair 的 measured value 是 seconds per emitted token 与 `log(merged/released)`；restart value `r_j` 是其 50 个 log ratios 的 median。最终 estimate 是 20 个 r_j 的 arithmetic mean，s 是 `ddof=1` sample standard deviation；90% CI 为 `estimate ± t_(0.95,19)*s/sqrt(20)`。TOST alpha .05，只有两个 CI endpoints 都严格位于 `[log(.98),log(1.02)]` 内才 PASS。

## Failure Interpretation

- parity、tie、counter、restoration、merge tests 失败：implementation stop；
- cost/memory gate 失败：engineering close，不作科学结论；
- D 不优于 A：普通 D-PACE LoRA 已解释 gain；
- D 不优于 B：dynamic frontier claim 不支持；
- D 未相对 C 降 harm 且 EAL non-inferior：prefix-feasibility claim 不支持；
- safety/domain/trace/latency gate 失败：对应 claim 与 scope 关闭；
- 所有 failure 都不得通过新增 arm、改 threshold、重开数据或回到旧 R083 路线 rescue。

## Compute Envelope

预估 gate/capacity 6–14 A800 GPU-hours，prospective collection 12–35，四 arms 40–120，falsifier/deployment 10–24，总计约 68–193 A800 GPU-hours，具体取决于 power-derived n_f。授权顺序唯一固定为 G0 local CPU/mock → G1 one synthetic GPU smoke → G2 exactly three A/D cost pairs → G3 power/split materialization and audit → G4 fit/checkpoint sequence generation → G5 32/512-block capacity → G6 4×3 training → G7 one common falsifier opening/C1-EFFICACY/C2 → G8（以 C1-EFFICACY PASS 与 frozen D-seed0 identity 为前提，先且仅先执行 selected-checkpoint merge/wrapper-removal/trace/dtype/output audit；该 audit PASS 后才执行 fixed latency；两份 receipt 决定 final C1-SYSTEM/DEPLOYMENT）。只有前一阶段审查与 gate 通过，下一阶段才获得执行授权。
