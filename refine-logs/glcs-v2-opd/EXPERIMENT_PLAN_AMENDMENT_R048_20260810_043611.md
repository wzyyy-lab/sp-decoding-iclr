# Experiment Plan Amendment: R048 Fast-K32 Draft-Prefix Early Verifier

**Date:** 2026-08-10 04:36 CST  
**Parent:** `EXPERIMENT_PLAN.md`  
**Precondition:** R047 stopped at step 200; best full-B16 EAL was 7.300777454.

## R048-A: oracle and systems feasibility before training

- Recompute the DFlash vocabulary lattice at K32 from the stored full-B16
  parallel hidden states.
- Generate the candidate-only Domino proposal using one batched vocabulary GEMM
  and gathered frozen Domino correction scores.
- Candidate support is Top31 plus the current proposal token (K32).
- Report the exact fixed-B16 candidate proposal baseline, one-repair reachable
  oracle, two-repair oracle (headroom only), and three-domain values.
- Stop before training if the one-repair oracle is below 8.40.
- Profile the candidate-only proposal and a perfect-correction layer-split path.
  Stop if ideal end-to-end throughput is below 1.20x released Domino.

## R048-B: deployment-shaped capacity falsifier

- Use 64 train prompts for both fitting and capacity evaluation; mark the result
  capacity-only and suppress any held-out claim.
- For each block, generate the frozen Fast-K32 proposal first.
- With target prefix KV, run exactly
  `anchor + proposal[:15]` through target layers 0--3. Store the 16 prediction
  states with the state-before-token alignment.
- Run the exact target verifier for labels. Only accepted prefix and original
  first rejection are valid.
- Train only the 180,224-parameter L4 tuned residual.
- Use one earliest confidence-gated replacement; never modify a second token.
- Require at least 90% recovery of the K32 one-repair oracle gain by step 200.

## R048-C: held-out efficacy

Only after A and B pass:

- Prompt-disjoint train/internal-calibration/`validation_select`.
- Threshold is selected without `validation_select`.
- step50 fixed EAL >=7.55;
- step100 fixed EAL >=7.80 and protected harm <=0.05 EAL;
- maximum 300 steps;
- no scale unless best fixed EAL >=8.10;
- accuracy success >=8.325485909.

A promising final checkpoint must be rerun through the true partial-target
proposal path and then through dynamic rollout before any SGLang claim.

