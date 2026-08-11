# prospective-v2 G0 v2→v3 Corrections

**Frozen at**: 2026-08-06 15:41:37 +08:00  
**Review basis**: `G0_CONTRACT_V2_REVIEW_20260806_153000.md` (`42caf382cdd3f0b5f6b88cc9e069aacb3e1858175c705f0f49b5477bbe4daaf2`)  
**Authorization**: pending G0 v3 re-review; opens nothing.

## Superseded v2 canonical identities

- `FINAL_PROPOSAL.md`: `926af36dd79b3398264929de2d0838499eaf9aea260fd6dde1a349fbe154205c`
- `EXPERIMENT_PLAN.md`: `d192fc91960cee7fcf3c657da1d3d3a84d78ef533e5ad751585fa89d016cd640`
- `PROSPECTIVE_V2_CONTRACT.md`: `2f7c3b212db79f551b7098168a7a5da1a9448ef49413e9958b4516819fdea48c`
- `PROSPECTIVE_V2_CONTRACT.json`: `335106ae12c69c1bf2685cf0c90bb27b31726961be5222ac6cc7ce5ffb9c5d83`

## Blocking fixes

1. G8 is now an explicitly ordered deployment stage. Its prerequisite is C1-EFFICACY PASS plus the frozen selected-feasible D-seed0 identity. G8a authorizes exactly one real merge/wrapper-removal/trace/dtype/output audit. Only G8a PASS opens the frozen G8b latency fixture. Final C1-SYSTEM/DEPLOYMENT requires both receipts.
2. For restart j, `r_j` is the median of 50 paired log latency ratios. The final estimate is `mean(r_j)`; `s=std(r_j,ddof=1)`; the 90% Student-t CI is `estimate ± t_(0.95,19)*s/sqrt(20)`. Both endpoints must lie strictly inside `[log(.98),log(1.02)]`.

## Additional deterministic pins

- Rank/hash byte encodings and exact independent allocator replay fields are frozen.
- Reserve exhaustion has an explicit terminal sequence-collection failure class.
- Proposal-level LoRA initialization/RNG and exact SDPA wording now match Plan/Contract/JSON.

The method, scientific arms, thresholds, data quotas, and outcome-opening boundary are unchanged.
