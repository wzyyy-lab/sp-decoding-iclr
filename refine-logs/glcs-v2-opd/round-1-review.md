# Round 1 Research Review

<details>
<summary>Full raw fresh-agent review</summary>

# Phase-2 Review: OPAL — On-Policy Advantage Lookahead over Domino

CALIBRATION: anchored

Calibration uses the supplied local evidence:

- Released exact-runtime held-out Domino: `7.23955` EAL.
- Held-out Top-16 oracle: `10.25449`.
- GLCS-v1 same-set capacity: `10.46289`, matching its oracle.
- Static 15,886-block training: best checkpoint remains step 0; training decreases held-out EAL.
- Static 295,604-block training: both frozen 10.3M and joint 61.1M variants remain below initialization.
- OPB anchors (`0,16,...,112`) and phase3 anchors (`0,34,69,...,240`) are not the policy-dependent verification anchors reached by approximately `accepted length + bonus`.
- GLCS-v1 freezes the first parallel token; fixed-prefix reach is only about `0.89`, leaving roughly 11% of blocks unrecoverable by construction.

## Scores

| Dimension | Weight | Score | Weighted |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9 | 1.35 |
| Method Specificity | 25% | 5 | 1.25 |
| Contribution Quality | 25% | 6 | 1.50 |
| Frontier Leverage | 15% | 6 | 0.90 |
| Feasibility | 10% | 6 | 0.60 |
| Validation Focus | 5% | 7 | 0.35 |
| Venue Readiness | 5% | 6 | 0.30 |
| **Weighted composite** | **100%** |  | **6.25 / 10** |

**GAP:** The proposal is `2.75` points below the READY threshold. The gap is not a need for more modules or benchmarks. It is a method-definition gap: the current candidate-truncated mixed-KL objective is not the Draft-OPD objective and is not correctly aligned with deterministic `T=0` acceptance; training prefix construction and selected-token feedback remain underspecified; position 0 is still apparently uncorrected; and the proposal combines an unvalidated data/objective change with an unvalidated candidate-lookahead architecture. These issues must be resolved before the 12–30M adapter is worth implementing.

## 1. Problem Fidelity — 9/10

The proposal preserves the hard anchor: it explicitly rejects `+0.1/+0.3` as success, targets `≥8.325`, and delays SGLang work until EAL passes. The empirical diagnosis is also directionally correct: same-set memorization establishes capacity but says nothing about prompt generalization, while both 15.9K and 295.6K static-anchor runs fail on unseen prompts.

There is, however, an objective-level drift. The deployment problem is exact greedy prefix acceptance, while the proposed primary loss is distribution matching after renormalizing the target over a candidate subset. Those are not equivalent when the target greedy token is outside that subset or when lowering KL does not change the candidate argmax.

## 2. Method Specificity — 5/10

**Weakness:** The teacher-prefix semantics are not precise enough, and a wrong implementation would silently create another teacher-forced dataset. For draft block `d_(m,1:B)` from verified context `c_m`, the target logits for decision `k` must be `z^T_(m,k)=T(c_m,d_(m,<k))`, including the draft-selected tokens before `k`. Accepted positions use a clean prefix because those selected tokens equal target greedy tokens. The first rejected position also has a clean prefix. Only later rejected suffix positions are conditioned on a wrong prefix.

The proposal says “replay target once” but does not make the off-by-one alignment, logged block identity, or GRU-state construction operational. It also does not state whether training `S_i` consumes logged policy selections, teacher tokens, or current recomputed selections.

**Concrete fix:** Define the replay record per verification round as:

- exact verified context identifier;
- deployed draft-generated block `d_(1:B)`;
- accepted length `r`;
- all deployed candidate sets and released Domino scores;
- target full-vocabulary top-1 ID and logit;
- target logits on stored candidates;
- policy/version ID.

Require the invariant `d_k == target_top1_k` exactly for `k ≤ r`, with the first mismatch at `r+1` under `T=0`. Construct the frozen GRU state from `anchor + logged/current selected d_<k`, never from gold suffix tokens. Refresh the dataset after the policy changes materially.

**Priority:** CRITICAL.

**Weakness:** “DFlash Top-16 and Domino action fixed 16-slot union” is undefined. A union can have 17 entries, and zero residual does not preserve Domino unless the candidate set contains the released action and the base score is the released Domino score.

**Concrete fix:** Define `C_i = DFlashTop16_i` when released action `a_i^D` is already present, and otherwise `C_i = {a_i^D} ∪ DFlashTop15_i`; or simply use 17 candidates for the prototype. Initialize scores to exact released Domino logits restricted to `C_i`, not raw DFlash logits. Verify bit-identical released decisions before training.

**Priority:** CRITICAL.

**Weakness:** Position 0 remains ambiguous and appears inherited from GLCS-v1’s frozen-prefix design.

**Concrete fix:** Scoring all 16 DFlash positions should be mandatory. Select position 0 from its Top-K with a zero-initialized residual that preserves the DFlash base top-1. Then initialize the frozen Domino GRU from `anchor + selected_position0` and causally score positions 1–15. This is a small interface correction, not an extra contribution, and removes the observed 11% unrecoverable-block ceiling.

**Priority:** CRITICAL.

## 3. Contribution Quality — 6/10

**Weakness:** OPAL currently bundles two independent hypotheses:

1. policy-correct frontier supervision fixes generalization;
2. candidate-conditioned global lookahead is needed beyond the existing Domino head.

The existing evidence supports the first hypothesis strongly but does not support the second. GLCS-v1’s static-data failure cannot show that candidate-specific lookahead is necessary because its anchors were not deployment states and it could not repair position 0.

The proposed global modes are also computed once from the same static lattice for every candidate. A candidate-specific query provides a candidate-conditioned readout, but not a counterfactual future lattice conditioned on choosing that candidate. Calling this “candidate-specific lookahead advantage” risks overstating what the architecture computes.

**Concrete fix:** Make the dominant contribution “greedy-frontier on-policy correction for all positions of a block-parallel draft.” Treat candidate-conditioned global lattice readout as a measured escalation. First compare against applying exactly the same replay and loss to the released Domino GRU/low-rank head. If the new adapter does not add a material matched-data gain, delete it.

Use “candidate-conditioned lattice readout” unless there is evidence that its future-token interaction behaves like branch-conditioned lookahead.

**Priority:** CRITICAL.

## 4. Frontier Leverage — 6/10

**Weakness:** The proposal borrows Draft-OPD’s accepted-forward/rejected-reverse KL without preserving the conditions that justify it. Draft-OPD defines both teacher and student distributions over the full vocabulary. OPAL renormalizes them over a varying candidate set. If the target greedy token lies outside that set, the renormalized target still declares some wrong candidate best even though every available action has zero greedy acceptance reward.

Furthermore, at `T=0`, verification is an exact greedy boundary. Later tokens after the first rejection cannot contribute to that block’s accepted length. Full rejected-suffix reverse KL is therefore a weakly aligned auxiliary and may recreate the suffix-loss dilution already implicated locally.

**Concrete fix:** Use frontier-only supervision as the primary greedy objective:

- preserve accepted positions;
- strongly train the first rejected position when the target top-1 lies in the candidate set;
- ignore later wrong-prefix suffix positions initially;
- when the target top-1 is unavailable, apply identity/residual regularization rather than teaching the “least wrong” candidate.

A suitable block loss is accepted-prefix boundary preservation plus a more strongly weighted first-rejection boundary loss and identity residual regularization, with `lambda_break > lambda_keep`, prompt-balanced block weights, and either CE or a gold-versus-best-competitor margin. Candidate-conditional KL can remain a low-weight auxiliary only at these reachable frontier positions.

For sampling, the design must change: acceptance is based on the `p/q` acceptance rule, not target argmax equality, and faithful full-distribution Draft-OPD becomes relevant. Do not use the greedy formulation to claim sampling improvements.

**Priority:** CRITICAL.

## 5. Feasibility — 6/10

**Weakness:** The data budget is plausible, but the first-stage estimate understates the alignment and runtime risks. Draft-OPD’s local paper uses a diverse 16K prompt pool and adapts the full draft model for eight epochs. Its Qwen3-4B greedy gains are approximately `+0.45` to `+0.56` EAL, materially smaller than OPAL’s required `+1.085`. Thus OPD is credible evidence for a gain, but not evidence that the hard target is reachable with a frozen draft plus compact residual.

Candidate-specific cross-attention also sits inside a selected-token-dependent sequential loop. The arithmetic is small, but fifteen or sixteen small attention launches can be latency-dominant unless fused or captured. The current “15 scalar scoring steps” description omits this cost.

**Concrete fix:** Reuse target verification logits rather than performing a redundant teacher replay where possible. Store only target top-1 ID/logit, candidate logits, and any normalization statistic needed by the chosen auxiliary. Profile one all-16 candidate step and the complete sequential head before scaling data.

Do not expand to 100K prompts merely because loss decreases. If frontier supervision gives a real but insufficient gain, compare the frozen residual with direct Domino-head adaptation and then a final-backbone-layer/LoRA arm on identical replay data.

**Priority:** IMPORTANT.

## 6. Validation Focus — 7/10

The validation is reasonably claim-driven, but two controls are more important than the proposed shared-code versus candidate-query ablation:

- all-16 scoring versus frozen position 0;
- direct released-Domino-head adaptation versus a new OPAL adapter under identical on-policy frontier supervision.

The current oracle should also be recomputed for the exact deployed union candidate construction. The existing `10.25449` oracle is encouraging but does not certify the proposed union or position-0 path.

## 7. Venue Readiness — 6/10

**Weakness:** In its current form, the paper story can be read as “Draft-OPD plus a cross-attention re-ranker.” That is not yet a sufficiently sharp mechanism contribution, especially when the KL transfer is technically inexact and the new architecture has not been shown necessary.

**Concrete fix:** Center the paper on one result-backed thesis: exact greedy speculative decoding should train only on policy-induced reachable frontiers, across all block positions. The supporting architecture earns a place only if it recovers a substantial additional part of the Top-K oracle gap over direct head adaptation. A credible venue-level result would be `≥8.325` held-out EAL plus deployment throughput, not a long ablation list around a `7.5` model.

**Priority:** IMPORTANT.

## Technical Assessment of the Two Main Mechanisms

### On-policy error-position replay

This is the strongest part of the proposal. The new anchor diagnostic materially strengthens it: neither OPB nor phase3 anchor schedules reproduce actual speculative rollout states. The 295.6K-block failure therefore does not refute on-policy training; it confirms that more static blocks do not fix state-distribution mismatch.

For greedy deployment, however, “error-position replay” should mean verified block starts plus the reachable acceptance frontier, not every wrong-prefix suffix. Record policy-generated blocks at actual verification anchors, train accepted positions and the first rejection, and refresh after policy changes.

### Candidate-specific lookahead

Candidate-conditioned scoring is plausible, but it is not yet established as necessary or sufficient. It may help use future lattice correlations to distinguish semantically compatible candidates, but it cannot repair:

- a target top-1 absent from the candidate set;
- wrong teacher-prefix alignment;
- a frozen wrong position 0;
- an objective that rewards a higher-probability but still rejected token.

It should therefore come after, not before, the data/prefix/objective sanity gate.

## Simplification Opportunities

1. Apply on-policy frontier supervision directly to the released Domino GRU and low-rank correction head first. This adds no inference component and is the cleanest test of the central causal hypothesis.
2. Drop later rejected-suffix reverse KL for the greedy route. Keep accepted-prefix preservation plus first-rejection repair.
3. Extend scoring to position 0 using the same residual interface; do not introduce a separate position-0 module or gate.

## Modernization Opportunities

1. Use DAgger-style policy-versioned frontier aggregation: released-Domino data followed by one current-policy refresh, with explicit mixture ratios.
2. Replace candidate-truncated KL as the primary loss with target-top1 boundary distillation and target-logit margins at reachable states.
3. Reuse logits already produced by exact target verification, avoiding a second teacher forward when alignment permits.

## Drift Warning

The problem statement is preserved, but there is a significant objective-level drift: candidate-renormalized mixed KL and wrong-prefix suffix replay optimize distribution similarity rather than exact greedy accepted-prefix length. Correct this before implementation.

## Verdict

**REVISE**

The on-policy diagnosis is promising and empirically grounded. The current OPAL architecture and loss should not be implemented as written.

## Prioritized Implementation Decision

1. **Exact highest-value sanity experiment:** run a `2K-prompt exact-runtime greedy-frontier screen` using released Domino rollouts at their actual verification anchors. Score all 16 DFlash positions. Reuse the released Domino architecture and train only its existing GRU/low-rank head with accepted-prefix preservation plus first-rejection target-top1 boundary loss. Compare against an identical static-anchor control. Do not train later rejected suffixes. Before optimization require baseline reproduction, stored first-mismatch agreement, exact target-prefix alignment, and an all-16 candidate oracle above `8.325`.
2. **Minimum code/data change:** add one exact on-policy rollout collector recording actual anchors, policy-selected blocks, accepted lengths, all-16 candidate sets, released scores, target top-1 IDs/logits, and target candidate logits. Extend scoring to position 0, then initialize the frozen Domino GRU from `anchor + selected_position0`. Add a frontier mask selecting accepted positions plus the first rejection. Do not implement the 12–30M candidate cross-attention adapter yet.
3. **Abort/continue thresholds:** treat `≥7.55` and `≥+0.30` on untouched held-out prompts as evidence frontier replay works, not final success. Abort the current mechanism before a new adapter if the 2K screen stays `≤7.40` or lowers EAL. Scale to 16K–32K and one policy refresh only after the screen passes. If refreshed training remains below `7.8`, compare direct head adaptation with final-layer/LoRA adaptation. The only success gate is held-out `≥8.325`; SGLang begins afterward.
4. **Route choice:** revise OPAL. Keep actual on-policy anchors, all-16 correction, selected-token feedback, and frontier supervision. Replace suffix KL with a greedy-frontier objective. First adapt the existing Domino head; add candidate-conditioned global lookahead only if the simpler head has real on-policy gain but plateaus below target.

</details>

## Parsed verdict

- Problem Fidelity: 9
- Method Specificity: 5
- Contribution Quality: 6
- Frontier Leverage: 6
- Feasibility: 6
- Validation Focus: 7
- Venue Readiness: 6
- Weighted overall: 6.25 / 10
- Verdict: REVISE
- Drift: candidate-truncated suffix KL drifts from exact greedy frontier utility
- Calibration note: the raw reviewer marked `anchored` using numerical experiment baselines, but no human-curated good/bad proposal exemplars were supplied. Under the ARIS taste-calibration protocol this run is conservatively recorded as `CALIBRATION: none` for acceptance purposes.
