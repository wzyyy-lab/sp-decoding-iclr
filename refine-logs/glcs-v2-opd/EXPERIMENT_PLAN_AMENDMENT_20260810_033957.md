# GFPR Experiment Plan Amendment: Current-Anchor Target Early Exit

**Date:** 2026-08-10 03:39 CST  
**Parent plan:** `EXPERIMENT_PLAN.md`  
**Precondition:** R046 failed at 7.243804665 full-B16 EAL. Full-model,
full-vocabulary, all-16 target distillation therefore did not remove the
held-out transfer failure. Merely increasing the same training data is not a
credible route to the 8.325 target.

## R047: causal current-anchor target early-exit correction

- **Claim tested:** the dominant limitation is missing online information about
  the newly emitted anchor, not selector capacity or target-label quality.
- **New deployment input:** after the target KV cache has produced the current
  anchor token, execute that one token through target layers 0--3 and expose
  its layer-4 hidden state `h_anchor_t4` (width 2560). This state is causal: it
  depends only on the already-realized context and anchor. Training extracts
  the same state from a full causal replay at `hidden_states[4][anchor]`.
- **Frozen base path:** retain the released Domino GRU, input projection, and
  vocabulary basis exactly. At every position use the exact K16 union of
  DFlash Top-15 plus the current-prefix released Domino action. This preserves
  the released Domino token path at zero residual, including BF16 tie behavior.
- **New head:**
  - local projection `Linear(2560+1024, 64)`;
  - anchor projection `Linear(2560, 64)` after parameter-free RMS scaling;
  - SiLU interaction followed by zero-initialized `Linear(64, 256)`;
  - candidate residual is a dot product with the frozen released Domino
    vocabulary-basis rows.
  The trainable parameter count is exactly 409,600 with no new full-vocabulary
  projection and no sequential search branch.
- **Objective:** verifier-aligned target KL at temperature 2 on only the
  released accepted prefix and original first rejection. Later stored suffixes
  are excluded because their Domino states were produced under a wrong prefix.
  Accepted positions have weight 1 and the first rejection weight 4. Canonical
  gold remains the acceptance contract. No dense D-PACE, hard frontier hinge,
  or target-advantage term is mixed into the primary arm.
- **Data:** Phase3 train prompt groups; prompt-disjoint `validation_select` for
  checkpoint selection; exact full-B16 labels and released baseline.

## Gates

1. **CPU/unit contract:** the stored feature is the requested layer and anchor
   position; zero initialization reproduces released Domino exactly; gradients
   reach all three new projections but no frozen base parameter; later stale
   suffix positions receive zero target loss.
2. **GPU sanity:** 32 train prompts and the held-out fixed evaluator must show
   step-0 token/length identity, finite gradients, nonzero early-exit feature,
   and no A40 OOM.
3. **KV/cache-shape alignment:** on representative blocks compare the cached
   full causal replay feature against a batch-1 incremental context-prefill +
   single-anchor layers-0--3 forward. Before any claim, evaluation must use the
   incremental feature path and report its token-path delta from cached-replay
   evaluation; a material delta requires runtime-aligned recollection/training.
4. **Cheap efficacy gate:** by optimizer step 200, full-B16 EAL must be at least
   7.50 and the best checkpoints must still trend upward. Otherwise stop this
   architecture rather than sweeping ranks or learning rates.
5. **Continue gate:** best full-B16 EAL at least 7.80, paired bootstrap lower
   bound above zero, loss/gain ratio at most 0.5, and no domain with a clear
   negative trend.
6. **Latency gate:** on the same A40 eager setup, target layers 0--3 for one
   anchor plus the residual/gather head add at most 1.25 ms over the released
   Domino head. CUDA-graph/Triton comparison follows only after efficacy.
7. **Hard result:** fixed full-B16 EAL at least 8.325485909 (exactly 15% above
   7.239552964). Only then recollect true
   dynamic rollouts and integrate/profile the split verifier in SGLang. The
   final systems target remains at least 15% throughput over released Domino.

## Stop interpretation

If R047 fails the step-200 7.50 gate, the result falsifies this specific
four-layer anchor feature and 409.6K interaction head. It does not justify
claiming that the Top-16 oracle is learnable from the old DFlash/Domino state;
R036--R046 already provide contrary held-out evidence. A subsequent design must
introduce materially richer online target computation or a larger draft model,
not another loss/rank/data sweep of the same representation.
