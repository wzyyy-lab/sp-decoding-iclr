# JAPD-16 M0 Experiment Code Review

**Time:** 2026-08-10 14:55:17 CST  
**Scope:** J000 objective/contracts, J001 sidecar replay, and J002 split/dataflow only  
**Reviewer:** fresh independent GPT-5.6-Sol xhigh agent under `experiment-bridge`

## Verdict

**GO** for exactly one J001 A40 32-block sidecar smoke and for closing the
completed J002 audit.  This verdict does not authorize M1 until the real J001
GPU receipt exits zero and all replay parity fields pass.

## Blocking findings and closure

1. The first implementation checked Top-16 IDs/logits and LSE but did not check
   the five scalar channels or final selected-token parity.  The verifier now
   instantiates a deterministic nonzero-readout D64 global axial audit head and
   fail-closes on stored-vs-replayed scalar, complete score, or selected-token
   disagreement (`atol=1e-5`, `rtol=1e-6`).
2. The first collate contract could admit an `h=0` block if another block from
   the same prompt was effective.  Collate now reconstructs each block's strict
   clean horizon and rejects every such block from an effective training batch.

Both blockers have targeted regressions and are closed.

## Independent checks

- Active architecture is full16, global non-causal, one-call `[B,16,16]`, and
  produces one 16-token chain.  No GRU, causal selected-token feedback,
  serial/iterative decode, beam, tree, trie, forest, or multipath path exists.
- JAPD math matches the frozen contract: clean horizon, `T=2`, hard/soft
  `0.9/0.1`, `Z=136`, undiluted joint certificate, and inclusive J2.
- Prompt-balanced estimator is unbiased over uniformly shuffled effective
  blocks; an exhaustive small enumeration matches the exact prompt objective.
- Canonical gold defines loss/EAL/J2.  Target candidate logits and target top1
  are offline supervision/clean-geometry labels and do not enter model forward.
- Sidecar uses batch1/full16/BF16 `F.linear` with the same tied embedding/vocab
  weight geometry as the collector; canonical rollout remains read-only.
- Real J002 audit: 1,987 prompts and 15,886 blocks; fit/select/diagnostic prompt
  counts `1589/199/199`, strictly disjoint; forbidden target-online fields never
  enter the collated batch.
- Focused tests: `14 passed` at review snapshot.  Broader selector/JAPD suite:
  `62 passed, 3 subtests passed`.  J000 reconstructs `745/15/0/207` exactly.
- Slurm syntax, Python AST, and diff whitespace checks passed.

## Non-blocking hardening

- The launcher was subsequently pinned from generic `gpu:1` to `gpu:a40:1`.
- The trainer's offline Domino metric field was subsequently added to the
  explicit collate whitelist with a regression; it remains absent from the
  selector forward signature.

## Authorization boundary

J001 must still produce an actual A40 report with exact Top-16/logit parity and
passing LSE/scalar/score/token replay checks.  Until then, M1 stays blocked and
there is no acceptance-length or throughput claim.
