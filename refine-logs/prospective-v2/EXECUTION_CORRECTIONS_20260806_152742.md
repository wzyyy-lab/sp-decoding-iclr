# prospective-v2 G0 Execution Corrections

**Frozen at**: 2026-08-06 15:27:42 +08:00  
**Scope**: execution-only clarification; FBPF method thesis, arm family, and scientific effect thresholds are unchanged.  
**Authorization**: pending fresh G0 v2 review; no implementation, GPU, data, producer receipt, capacity, training, falsifier, or latency work is opened by this artifact.

## Superseded v1 identities

- `PROSPECTIVE_V2_CONTRACT.md`: `f37ba7d187468fe410a10a35f716735b291fc0476f1cc76cc7335b315309a61d`
- `PROSPECTIVE_V2_CONTRACT.json`: `c0df56b83c5d8b0a2a278f456ddc75922874adf323d56c0eb4fb06f0b6de6336`
- `FINAL_PROPOSAL.md`: `ff1a46e23b8e4152325f758cc85e5ebb91f7d4501b5be89a22458e23daeea6b2`
- `EXPERIMENT_PLAN.md`: `092f0a447e0a1bda59dc227243f30e882bf44b28a87be7330342f1befc48c69a`
- raw v1 review: `G0_CONTRACT_REVIEW_20260806_152112.md`

The timestamped v1 contract and proposal/plan artifacts remain immutable. The canonical files now carry the self-contained v2 wording.

## Blocking findings resolved in v2

1. D-PACE reduction: one source prompt expands to four tensor rows, one anchor per row, `/bsz=4`; full scalar and flattened-gradient parity is required.
2. Gold and anchors: frozen target alone defines the continuation and labels; float32 decisions, lowest-ID ties, exact Python-rounding offset formula, and context boundary are frozen.
3. Completeness: minimum continuation is 19 tokens, exactly four complete blocks; short attempts are permanently consumed from preassigned same-domain/same-split reserves.
4. Native LoRA: exact module paths, float32 A/B masters and moments, isolated seed mapping, sorted initialization, float32 branch, bf16 stored merge, wrapper removal, and per-module hashes are frozen.
5. Restoration: restore/recompute/task flow, restore+skip semantics, rollback, stable row order, and once-only counters are executable and testable.
6. Statistics: all released references and contrasts are explicit; percentile bootstrap quantiles/interpolation and a deterministic design-effect power formula are frozen.
7. Components/reserves: normalization, tokenization, postings/Jaccard/union-find, cross-domain exclusion, component IDs, whole-component allocator, reserves, and exact-quota failure are frozen.
8. Cost fixture: every D warmup/timed minibatch must assert K=4 and report protected histograms; fixture labels are engineering-only.
9. Identities: five definitions versus 13 instances are separated; every trained hash is selected-feasible or diagnostic; diagnostic consequences are frozen.
10. Authorization: one G0–G8 ladder is used; C1-EFFICACY is adjudicated at G7 and C1-SYSTEM/DEPLOYMENT after G8 latency.

## Non-blocking findings resolved

- `attn_implementation="sdpa"` is exact.
- The compute range is corrected to 68–193 A800-hours.
- Latency uses the first 50 checkpoint prompts by stable manifest rank, 64-token end-to-end speculative generation, seconds/emitted-token, and restart-level median paired log ratios.

## Re-review requirement

The same G0 reviewer must re-read the complete canonical `FINAL_PROPOSAL.md`, `EXPERIMENT_PLAN.md`, `PROSPECTIVE_V2_CONTRACT.md`, and `PROSPECTIVE_V2_CONTRACT.json`. Only an explicit `G0_VERDICT: GO` may open local implementation and CPU/mock tests. All GPU and data stages remain closed.
