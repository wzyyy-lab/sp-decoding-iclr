# R053 Experiment Plan

最新完整计划见 `EXPERIMENT_PLAN_AMENDMENT_R053_20260810_072850.md`。核心门：Fast-K64 trunk + Top15/trunk K16 W16 beam，固定 `N={17,24,32,48,64}` 的draft-only prefix-closed tree；只有deployable clean EAL ≥8.325485909、三域无回退且optimistic eager TPS ≥1.20x Domino同时成立，才授权一次target-forward的SGLang tree verifier。任何N≤64失败则关闭target multipath，不扩W64/N>64/learned pruning。

