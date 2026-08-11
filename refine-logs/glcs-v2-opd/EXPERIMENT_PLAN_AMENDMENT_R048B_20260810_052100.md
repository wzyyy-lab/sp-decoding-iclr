# R048-B Capacity Amendment after Systems Sanity

**Date:** 2026-08-10 05:21 CST  
**Parent:** `EXPERIMENT_PLAN_AMENDMENT_R048K_20260810_045547.md`

Job 10164859 rejects the Hugging Face DynamicCache/SDPA layer-split path as a
deployment implementation: weighted split overhead is about 2.79 ms and one of
68 checked posterior positions changes argmax.  No lossless or throughput
claim may use that path.

This does not answer whether L4 proposal-prefix states contain the information
needed to identify the earliest repair.  R048-B is therefore authorized only
as a 64-prompt same-set capacity falsifier:

- Fast-K64 proposal, L4, rank64, 180,224 trainable parameters;
- exactly one earliest repair and an explicit KEEP fallback;
- decision rows are anchor then proposal tokens, state-before-token aligned;
- accepted prefix plus original first rejection only; suffix masked;
- early features use a disposable cache fork;
- labels and final EAL use a clean, unsplit full target verifier;
- maximum 200 optimizer steps;
- step0 must exactly reproduce Fast-K64;
- zero harmful accepted-prefix rewrites at the selected capacity checkpoint;
- required recovery is `(model - Fast)/(oracle - Fast) >= 0.90`.

Failure closes the early-state route without more data/depth/rank.  Passing
authorizes one prompt-disjoint held-out run but no deployment claim.  SGLang
paged-KV/layer-boundary integration remains deferred until held-out EAL reaches
`8.325485909`.
