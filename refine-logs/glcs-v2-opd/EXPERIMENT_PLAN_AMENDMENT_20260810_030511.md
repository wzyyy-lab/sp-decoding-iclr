# GFPR Experiment Plan Amendment: Full-Vocabulary Target Distillation

**Date:** 2026-08-10 03:05  
**Parent plan:** `EXPERIMENT_PLAN.md`  
**Reason:** R041--R045 exposed a structural gap in the Stage-D distillation
falsifier: the target loss is restricted to a live Top-16 union and gives no
gradient when the verifier token is outside that union. This amendment stays
inside Block B3/Stage D because it adapts only the existing DFlash and Domino
weights and does not add an inference module.

## R046: Full-vocabulary target-posterior adaptation

- **Claim tested:** the remaining transfer failure is caused by candidate-set
  truncation and frontier-only supervision rather than unavailable online
  information.
- **Trainable system:** existing 537M DFlash backbone plus existing 50.8M
  Domino causal head. No new inference parameters, operators, or latency.
- **Teacher:** live Qwen3-4B logits computed on the exact gold prefix. Teacher
  logits are labels only and are never deployment inputs.
- **Loss:**
  - accepted positions before the current first rejection: full-vocabulary
    gold-margin protection only;
  - current first rejection: full-vocabulary target KL, whether or not gold is
    in DFlash Top-16;
  - later positions: lower-weight full-vocabulary target KL under the exact
    gold prefix, so all positions that become reachable after a repair are
    trained in the same pass;
  - mask rows where the replayed target top-1 disagrees with canonical gold.
- **Data:** Phase3 train, 1,986 complete prompt groups, four evenly spaced
  blocks per prompt; prompt-disjoint `validation_select` for checkpoint
  selection. Full B16 is measured by the independent released-runtime
  evaluator.
- **Sanity gate:** step-0 identity; finite loss/gradients; no A40 OOM; gradients
  reach gold when it is outside current Top-16; prefix/suffix masks pass unit
  tests.
- **Continue gate:** full-B16 EAL at least 7.8, three-domain non-negative trend,
  and no severe gained/lost asymmetry.
- **Hard success:** fixed full-B16 EAL at least 8.325. Only then run true
  dynamic rollout and SGLang throughput.
- **Stop:** if the best full-B16 checkpoint remains below 7.8, do not sweep
  learning rates or head sizes. Under the current single-chain/no-extra-target
  constraints this closes the same-runtime representation route; the next
  legitimate scope change is a larger draft or an online target early-exit.

