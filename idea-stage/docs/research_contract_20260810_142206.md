# Research Contract：JAPD-16 Parallel Global One-Chain Head

**Frozen on:** 2026-08-10 after research-refine Round 3 READY (`9.33/10`)  
**Source:** `refine-logs/parallel-global-head-v2/FINAL_PROPOSAL.md`  
**Scope:** Qwen3-4B / released DFlash-Domino-b16 / greedy temperature 0 / B16 / pure-base K16

## Problem and Primary Claims

### C1 — Joint multi-repair mechanism

一个轻量full16 global non-causal head能在一次并行、唯一链输出中利用全块信息，
JAPD objective相对parameter-matched local-JAPD和global-Candidate-D-PACE显著提高
disjoint held-out EAL与strict two-frontier J2。

最低证据：fresh300上两组matched EAL增益各`>=0.15`且paired prompt-bootstrap CI lower
`>0`；global EAL`>=7.55`；J2至少Domino并胜controls各10pp；每域不低于base；
prefix harm不高于Domino。

### C2 — Acceptance and system outcome

100K训练后的冻结checkpoint在untouched final600上fixed/dynamic EAL均至少达到same-job
Domino的`1.15x`，并在同栈A40 SGLang上取得paired TPS ratio 95% CI lower`>=1.15`。

## Frozen Method

- online head：复用D64/H4/L1 axial `GlobalDirectCandidateSelector`，`433,852`参数；
- input：完整`[B,16,2560]` DFlash hidden、pure-base Top16 IDs/logits、candidate/anchor
  embeddings、同一DFlash vocab tensor的FP32 logsumexp；
- global arm：无causal mask，每个candidate读取全部16个position summaries；
- output：一次`[B,16,16]` scores，一次argmax得到唯一`[B,16]` proposal；
- clean horizon：首个gold不在K16或same-geometry target replay top1不等于gold的位置；
- soft teacher：`T=2`，`0.9 onehot + 0.1 candidate-conditional target distribution`；
- AP：固定`Z=136`的all-prefix weighted CE；
- J2：仅至少两处base错误block激活，`M=-logsumexp(-d_prefix)`，
  `L_J2=softplus(-M)`；总loss固定`L_AP+L_J2`，无mix权重；
- exact prompt objective：有效block按`1/|B_p+|`加权并使用固定全局缩放；
- default D64；只有same-set capacity与full-fit optimization两门同时失败，才在读取
  fresh300前将全部三臂统一切换D256。

## Immutable Anti-Claims / Prohibitions

不使用或声称：causal/autoregressive token feedback、Domino GRU rollout、serial target
seed/decode、Jacobi/iteration、beam/tree/trie/forest/multipath、Top16路径轴、额外online
target feature/forward、same-set或oracle替代held-out EAL。R050–R056均为off-spec历史证据。

## Data and Selection Contract

- R047 train按prompt/domain、label-independent拆为fit/select/diagnostic；
- fresh300与final600在采label前冻结，排除全部历史IDs与normalized-text近重复；
- 25K/100K也排除fresh300/final600；
- final600在architecture/loss/recipe/seeds/checkpoint/deployment seed冻结前不读outcome；
- target logits/tokens只作offline labels；loader不得把`target_anchor_early_feature`等字段
  传入head；ground truth始终是dataset canonical target IDs。

## Stop Rules

`mechanics/capacity/latency -> fresh300 -> 25K -> 100K -> final600 -> SGLang`。
任何阶段任一hard gate失败即关闭对应route，不用加参、改loss、换seed、serial/tree或
系统小修救援。D64 complete eager path含vocab GEMM/LSE/Top16/gather/head/argmax，开发
门为`<=1.20x`eager Domino。

## Claim Boundary

Round 3 READY只授权实现与gated experiments，不支持任何性能claim。C1/C2必须分别由
fresh matched evidence、untouched fixed/dynamic evidence和same-stack system evidence
实际通过后，才能交给result-to-claim审查。
