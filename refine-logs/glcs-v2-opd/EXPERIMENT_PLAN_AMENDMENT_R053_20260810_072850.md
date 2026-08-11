# R053 实验计划：One-Pass Target Tree Pareto Falsifier

**Problem:** R051 已把 clean fixed-B16 EAL 提高到 9.060131，但四次串行 target seed 使 eager throughput 只有 Domino 的 0.2963x。下一方法必须保留一次 target forward，同时解决 draft-only selector 无法识别正确分支的问题。  
**Method Thesis:** Fast-K64 只产生受保护的单链 trunk；冻结 Domino 在每个 prefix 上用 gathered Top15+trunk 的 K16 support 生成 W16 path pool，再由一次 target tree-attention forward 在线选择最长匹配分支。  
**Date:** 2026-08-10  
**Status:** Same-family ARIS provisional GO；只授权 bounded node-budget falsifier。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1 Primary | target 在线验证 candidate trie 能消除已反复失败的 draft-only branch identifiability | 当前 clean B16 上，固定 draft-only N-tree 的 EAL ≥8.325485909，且不是 hindsight/gold allocation | B1 |
| C2 Supporting | 一次 target forward 的树能把接受增益转化为吞吐 | 包含 base GEMM、Fast beam、N-row target、full-vocab logits/KV write 的 optimistic eager cycle ≥1.20x Domino | B2 |
| Anti-claim | 收益只是完整 path oracle、gold 剪枝或串行 target 调用 | 分开报告 full-pool oracle、N-node hindsight 与 deployable tree；allocator 不读 target/gold；target calls/cycle=1 | B0, B1 |

## Paper Storyline

- Main paper必须证明：单次 verifier 内的 target-guided branch traversal 同时提高 EAL 与 TPS。
- Appendix支持：完整 W16 oracle、unique-node分布、N-node hindsight、HF tree-mask数值诊断。
- 当前明确 cut：串行 exact seed、N>64、W64、learned tree pruning、OPB12K distillation、prefix选择后第二次 target verify。

## Experiment Blocks

### B0：语义与控制

- Claim tested：proposal、tree 与 clean authority 没有偷用 gold 或改变 Fast baseline。
- Dataset：`r047_anchor_t4_validation_select_10164718`，147 prompts / 1175 blocks，batch1。
- Controls：Fast trunk逐token等于现有 Fast-K64；每位置 support 前15行等于 DFlash Top15 且含 trunk；N17逐block等于 trunk；tree从不低于 trunk。
- Clean authority：先做一次 teacher-path target forward；只要任一位置不等于 stored gold，就从 clean prefix 完整 autoregressive 重生成 p0..p15+bonus。
- Success：上述 mismatch 全为0；否则禁止解释 EAL。
- Priority：MUST-RUN。

### B1：固定 Node-Budget Accuracy Pareto

- Claim tested：小的 prefix-closed tree 是否保留足够 target-compatible branch。
- Proposal：horizon16、Fast-K64 trunk、W16、branch support为DFlash Top15+trunk fixed K16；ordinary cumulative candidate-logprob beam pruning；无 per-beam vocab GEMM。
- Tree allocation：先保留 anchor+完整 trunk 共17 rows；其余节点按 descendant paths 中最大 `gamma=.75` draft score best-first加入；parent 未选时 child 不可选；tie按token ID和原始插入顺序。
- Fixed budgets：`N={17,24,32,48,64}`，N包含anchor。
- Metrics：
  1. full W16 pool oracle；
  2. gold-aware N-node hindsight structural upper bound；
  3. deployable draft-only N-tree EAL（唯一可过门指标）；
  4. 三域、paired gain/loss、full-accept leaf count、full-trie unique nodes p50/p95/max。
- Accuracy success：deployable EAL ≥8.325485909，且chat/code/math都不低于clean Domino。
- Failure：任何 N≤64 未过时关闭 target multipath；不得追加 learned pruning、W64或更大树。
- Priority：MUST-RUN。

### B2：一次 Target Forward 的 Optimistic Eager Profile

- Claim tested：B1增益是否有可能在真实系统中达到约15%吞吐提升。
- Representative setting：同一A40、batch1、validation_select median context；同job重测 Domino complete non-common cycle。
- Included：base vocab GEMM、Fast beam、一次N-row 4-D tree-mask target forward、full-vocab logits、临时KV写入和tree traversal。
- Explicit optimistic exclusion：data-dependent trie packing与paged-KV pointer commit按零成本处理；因此这里只能排除不可能路线，不能支持最终SGLang claim。
- Metric：

  `TPS_ratio = ((E_N+1)/T_N) / ((E_D+1)/T_D)`。

- System success：optimistic complete-cycle TPS ≥1.20x Domino，为最终真实SGLang ≥1.15x留余量。
- Joint success：B1 accuracy、三域和B2 system三门同时通过，选最小N。
- Priority：MUST-RUN。

### B3：Conditional SGLang Tree Verifier

- Entry：仅R053 joint gate通过。
- Required：一次target model call；paged tree KV只commit selected branch；非tie selected-branch target top1 parity100%；full-accept bonus parity100%；真实CUDA-graph/Triton端到端TPS 95% CI lower bound ≥1.15x Domino。
- Failure：保留R053 accuracy诊断但删除系统claim，不允许第二次target verify补救。
- Priority：CONDITIONAL。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---:|---|
| M0 | CPU语义 | focused tests、py_compile、bash -n | 全过 | <0.1 CPU-h | trie prefix/off-by-one |
| M1 | clean accuracy | R053 full fixed set | EAL、domain、identity | 0.2–0.4 A40-h | 小树剪掉低概率正确枝 |
| M2 | eager upper bound | 同job median context profile | optimistic TPS ≥1.20x | <0.1 A40-h | N-row LM head/MLP成本 |
| M3 | SGLang | 仅joint pass | lossless且TPS CI ≥1.15x | 4–12 GPU-h | paged KV/tree kernel |

## Compute and Data Budget

- R053单个debug A40 job，30分钟硬上限；无训练、无新collection。
- 最大计算来自1175个clean target authority forwards；teacher-path完全一致的block不做顺序重生成。
- 最大系统风险不是head参数，而是N-row target的MLP、attention、LM-head随节点增长。

## Risks and Mitigations

- Stored gold teacher mismatch导致suffix stale：任一mismatch整块autoregressive重生成。
- Full-pool oracle被误当可部署结果：三个accuracy数字在JSON中分开命名，gate只读deployable tree。
- Gold影响tree：allocator API只接paths与draft scores；target tokens只进入独立scorer。
- HF mask数值不等于SGLang：首轮只作诊断；最终必须做selected branch/bonus parity。
- Profile漏掉packing/commit：明确标为optimistic lower bound并使用更严格1.20x门；只有通过才实现真实系统。

## Final Checklist

- [x] 主claim与anti-claim冻结
- [x] N、W、K、gamma与allocator冻结
- [x] full-pool/hindsight/deployable隔离
- [x] accuracy authority为clean batch1 target continuation
- [x] 只允许一次target forward
- [ ] Fresh ARIS code review GO
- [ ] R053 joint Pareto gate通过
- [ ] Conditional SGLang lossless/throughput通过

