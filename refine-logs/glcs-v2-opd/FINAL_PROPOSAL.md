# FINAL PROPOSAL: GFPR

**Status:** ARIS research-refine READY (9.20/10)

GFPR 的正式、时间戳版本是 FINAL_PROPOSAL_20260809_235057.md。核心路线是：Stages A–C 只适配现有 full-vocabulary Domino causal head，在真实 policy anchors 上保护 accepted prefix、修 current first rejection，并把 position 0 纳入同一 head；固定 held-out EAL 必须达到 8.325，真实 dynamic rollout 必须达到 Domino 的 1.15×，通过 harm gates 后才进入 SGLang。

完整设计演化见 round-2-refinement.md，最终 READY 审查见 round-3-review.md。
