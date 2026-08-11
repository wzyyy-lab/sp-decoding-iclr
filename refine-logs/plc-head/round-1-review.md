# Round 1 Review: PLC-Head

## Verdict

- **Score:** 7.43 / 10
- **Decision:** REVISE
- **Dimension scores:** fidelity 9.5, specificity 6.5, contribution focus 7.5, frontier awareness 7.5, feasibility 6.5, validation quality 7.5, venue readiness 7.0.

## Critical Issues

1. The exact B16 deployment path must be stated unambiguously: one base-top-1 prefix token, a Top-16 lattice for the remaining 15 positions, 15 parallel correction codes, one batched full-vocabulary projection/argmax, concatenation into one chain, and one ordinary target verification. The student may not consume released Domino tokens at inference because obtaining them would already execute the GRU.
2. Teacher extraction must be on-policy released Domino, not clean-gold-prefix GRU states. For each real anchor and lattice, record the released path, `W_s s_i`, corrected logits, and teacher token. Clean-prefix/gold supervision is only a later improvement signal.
3. Four mode slots must remain distinct through the global block. A shared global route state should generate per-position slot weights rather than final mean pooling. Begin with one width-128 global block; add a second only if imitation fails and measured latency allows it.
4. Candidate-set KL alone does not match full-vocabulary deployment. Use full-vocabulary teacher/target CE, or at minimum teacher/gold candidates plus global hardest negatives. Code regression is auxiliary; teacher token/logit agreement and EAL are the imitation gates.
5. Use a simple two-stage objective: first on-policy deployment imitation; then target improvement which strongly preserves teacher-correct reachable tokens and replaces teacher-wrong tokens. Weight later positions by clean-prefix reach and continuation utility. Decay code regression after warm-up instead of tuning a large collection of unrelated losses.
6. The distinction from failed GCLS must be explicit. GCLS was an independent pure-DFlash candidate scorer and improved only about 0.285 EAL. PLC reuses Domino's trained backbone and lexical projection, distills its on-policy causal correction, and predicts the correction sufficient statistic rather than a path score. If the distilled model does not initialize near the released Domino EAL, stop and repair imitation rather than scale training.
7. Hard gates should be stronger: imitation EAL within 0.05--0.10 of Domino; primary adapted gain at least +0.5 EAL, preferably +1.0; optimized-head latency at most 0.8x released Domino; same-hardware end-to-end TPS at least 1.20x, preferably 1.25x. The system measurement must include Top-K/gather, encoder, full projection, and argmax.
8. The proposed roughly 40.55M active parameters are valid and about 20% below Domino's 50.82M correction head.
9. The FLOP explanation must account for Domino's optimized CUDA-graph runner, which precomputes the embedding-to-GRU input table. The expected speedup comes from removing the 15-step dependency, avoiding that approximately 0.93GB table, issuing one batched `W_out` GEMM/read, and reducing target verification rounds through higher EAL—not from an eager-only GRU MAC comparison.
10. Keep the first implementation minimal: one global block, no LoRA, two training stages, full-vocabulary correction. Do not make a candidate-only scoring variant part of the main method.

## Required Revision

Freeze a single deployable architecture and exact tensor path, specify the on-policy imitation data, simplify the loss and capacity schedule, and evaluate against the optimized graph-mode Domino baseline rather than only the eager implementation.
