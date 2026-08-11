CALIBRATION: none

# Round 2 Review: GFPR — Greedy-Frontier Policy Replay for Domino

## Summary Verdict

GFPR is a substantial improvement over Round 1. It removes the speculative candidate-attention module from the critical path, correctly focuses training on policy-induced reachable states, masks wrong-prefix suffixes for greedy decoding, and makes position-0 repair mandatory. The proposal is now a focused, minimal research route rather than a bundle of data, loss, and architecture hypotheses.

The main remaining blocker is an inconsistent deployment contract: Stage B directly fine-tunes the full-vocabulary Domino head, while the candidate section describes a frozen released-score-plus-residual K17/K16 policy. Those are different policies with different latency, candidate availability, identity, and loss semantics. The proposal must choose one for each stage.

## Scores

| Dimension | Weight | Score | Weighted |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 10 | 1.50 |
| Method Specificity | 25% | 8 | 2.00 |
| Contribution Quality | 25% | 9 | 2.25 |
| Frontier Leverage | 15% | 9 | 1.35 |
| Feasibility | 10% | 8 | 0.80 |
| Validation Focus | 5% | 9 | 0.45 |
| Venue Readiness | 5% | 8 | 0.40 |
| **Weighted composite** | **100%** |  | **8.75 / 10** |

**GAP:** GFPR is `0.25` below the READY threshold. The gap is narrow and concrete: unify the Stage-B action space and score contract, normalize accepted-prefix preservation so already-good long blocks do not dominate first-error repair, and add explicit harm/statistical gates. No new model component is needed to close the proposal gap.

## Resolution of Round 1 Concerns

| Round 1 concern | Round 2 status | Assessment |
|---|---|---|
| Static anchors do not match deployment states | Resolved | Actual rollout uses `o_{m+1}=o_m+r_m+1`, with policy versions and one refresh. |
| Target logits may use the wrong prefix | Substantially resolved | `T(c_m,d_{m,<i})` and first-mismatch invariants are correctly stated. One notation cleanup remains. |
| GRU might consume teacher tokens | Resolved | The proposal requires anchor plus logged/deployed selected tokens. |
| Position 0 is uncorrectable | Resolved | The shared Domino head is extended to position 0 with a zero-initialized gate. |
| K16 union and identity are undefined | Partially resolved | K17/K16 construction is defined, but it conflicts with direct full-vocabulary head adaptation. |
| Candidate-truncated mixed KL is misaligned with T=0 | Resolved | The primary objective is now greedy-frontier boundary learning; suffix KL is removed. |
| OPAL architecture and data hypothesis are confounded | Resolved | The new adapter is deferred until direct head adaptation shows positive but insufficient signal. |
| Weak falsification gates | Mostly resolved | Gates A–E are staged and effect-driven; harm and uncertainty need explicit thresholds. |

## 1. Problem Fidelity — 10/10

The immutable anchor is fully preserved:

- `7.55` is only a proof-of-signal threshold.
- `8.325` remains the sole method-success threshold.
- SGLang work remains behind the EAL gate.
- The route directly attacks the two locally demonstrated failures: off-policy/static anchors and a frozen first position.
- It does not substitute capacity memorization, teacher loss, or additional data volume for held-out exact-runtime EAL.

There is no substantive drift.

## 2. Method Specificity — 8/10

### Rollout, `r+1`, and target-prefix semantics

The cycle semantics are essentially correct. Under greedy verification, a cycle accepts `r` draft tokens, emits the target correction/bonus token, and advances by `r+1`. For accepted positions and the first rejection, the target is evaluated on a clean reachable prefix.

The notation `y_m` as an independent “target greedy continuation” is slightly dangerous after the first mismatch. Beyond that mismatch, an independent target rollout and `T(c_m,d_{<i})` are different objects, even though GFPR masks those later positions.

**IMPORTANT revision:** Define directly

\[
g_{m,i}=\arg\max_v T(v\mid c_m,d_{m,<i}), \qquad
r_m=\min\{i:d_{m,i}\neq g_{m,i}\}.
\]

Do not define the stored comparison sequence as an independent target rollout. Also state explicitly that the Domino GRU state is reset at each block, consumes the anchor first, and then consumes selected draft tokens after each decision.

For `r=16`, state that the target bonus token at position 16 is appended before the next block and that the next anchor/context includes it.

### Candidate/deployment contract

The K17/K16 construction itself is now precise:

- K17 preserves both the entire DFlash Top-16 and the released action.
- K16 replaces base rank 16 only when needed.
- Candidate ordering and exact released scores are specified.
- Position 0 uses DFlash logits; later positions use DFlash plus released Domino correction.

However, this contract conflicts with Stage B’s direct fine-tuning of the Domino GRU and full-vocabulary correction head:

- If the adapted head performs a full-vocabulary argmax, target top-1 is always representable and the `y_r∈C_r` mask is unnecessary.
- If deployment restricts the adapted policy to K16/K17, the candidate set requires the released Domino action under the current selected prefix. Computing that action after the head has been modified requires a frozen reference GRU/head or an equivalent frozen-score path.
- If current adapted full-vocabulary scores define the candidate set, it is no longer a released-score-plus-residual policy, and the stated identity/ceiling interpretation changes.
- A frozen reference head used at inference must be included in the latency model.

**CRITICAL revision:** Choose an explicit stage-specific contract. The cleanest minimal route is:

1. **Stages A–C:** direct full-vocabulary Domino-head adaptation. The action is `argmax_v s_i^θ(v)`, target top-1 is never “unavailable,” and exact identity comes from initializing `θ=θ_released` plus `g_0=0`. K17/K16 are offline oracle and optional deployment-contraction diagnostics only.
2. **Stage D candidate residual, if opened:** freeze the released Domino scorer, define K17/K16 from that frozen scorer under the current selected prefix, add only a candidate residual, and include frozen-reference scoring in latency.

Alternatively, use the candidate-restricted policy from Stage B onward, but then retain a frozen released scorer explicitly. Do not mix the two definitions.

Gate A must report the oracle for the candidate contract that will actually be deployed. If final deployment is K16, it is the K16 all-position oracle—not only K17—that must clear `8.325` with useful headroom.

## 3. Contribution Quality — 9/10

The dominant contribution is now focused and defensible:

> Exact-greedy block-parallel drafts should be adapted on policy-induced reachable frontiers, including the previously frozen pure-parallel first position.

This is a coherent mechanism rather than three parallel contributions. Actual policy anchors, accepted-prefix preservation, first-rejection repair, and position-0 coverage are parts of the same causal intervention.

The lack of a new neural architecture is not a weakness. Reusing the Domino head is the scientifically stronger choice because it isolates whether the supervision/state-distribution correction actually solves the observed generalization failure. The conditional lattice adapter is correctly demoted to an evidence-triggered escalation.

Venue-level novelty will ultimately depend on the magnitude and consistency of the result, but the proposal itself is appropriately parsimonious.

## 4. Frontier Leverage — 9/10

The revision now uses on-policy distillation principles appropriately rather than copying Draft-OPD mechanically:

- policy-generated states replace fixed target-clean offsets;
- exact greedy reachability replaces sampling-oriented suffix KL;
- the first rejected decision is the primary repair frontier;
- one controlled policy refresh addresses policy shift;
- later wrong-prefix suffixes are excluded from the greedy claim.

This is a clean adaptation of a modern on-policy primitive to a different inference regime. No additional RL, full DAgger loop, or fashionable architecture is needed.

## 5. Feasibility — 8/10

The proposed progression is practical:

- Gate A is read-only and cheap.
- Gate B changes data collection and the existing head rather than adding a large model.
- A 2K screen can falsify the route before 16K–32K collection.
- Position 0 adds one shared correction call rather than a separate model.
- One policy refresh is sufficient for the first causal test.

Two implementation details remain.

**IMPORTANT revision:** If Stage D unfreezes a DFlash backbone layer or LoRA, cached `parallel_hiddens` are no longer valid training inputs. Retain raw contexts/masks and recompute hiddens online for that arm, or recollect them after every backbone checkpoint. The frozen-head and joint-backbone arms cannot silently share stale hidden tensors.

**IMPORTANT revision:** Report position-0 cost as one additional full correction-head application when using the prototype full-vocabulary path. Calling it merely a scalar gate understates the actual projection work. Candidate gather can be evaluated later but should not be assumed free.

## 6. Validation Focus — 9/10

The gates are compact and causally informative:

- Gate A proves exact identity and attainable candidate ceiling.
- Gate B separates static versus actual anchors and positions 1–15 versus all 16.
- Gate C tests data scale and one policy refresh.
- Gate D compares capacity routes only after signal.
- Gate E measures system benefit only after EAL success.

This is not experimental bloat; it is a falsification ladder.

Two gate definitions should be tightened.

**IMPORTANT revision:** The accepted-prefix term currently sums over all `i<r`. Even with `λ_keep=0.1`, a block with 15 accepted tokens can contribute more preservation loss than a repairable first rejection. Replace the sum with

\[
\frac{\lambda_{\rm keep}}{\max(r,1)}
\sum_{i<r}[m_{\rm keep}-\Delta_i]_+.
\]

Fully accepted blocks should have a capped preservation budget rather than dominating training. This keeps the first repair frontier primary.

**IMPORTANT revision:** Add explicit paired uncertainty and harm gates:

- Gate B’s `+0.30` must have a paired prompt-bootstrap lower confidence bound above zero.
- Report total gained versus lost accepted tokens and harmful-prompt fraction.
- Predefine a tolerable harm condition, such as lost accepted tokens being no more than half the gained tokens.
- Apply the same harm gate at `8.325`; net EAL alone does not satisfy “not from many harmful overrides.”

The static-anchor arm should be generated using the same deployed drafting code at fixed offsets, so the only intended difference is anchor distribution.

## 7. Venue Readiness — 8/10

If GFPR reaches `≥8.325` and translates that gain to throughput, it can support a sharp paper:

- it explains why hundreds of thousands of static blocks fail;
- it identifies reachable greedy frontiers as the correct unit of supervision;
- it removes an architectural blind spot at position 0;
- it provides a minimal, deployment-compatible fix.

The remaining risk is not missing architecture; it is whether the method yields the unusually large required improvement. A result near `7.6` would validate a training observation but not support the anchored paper claim. A result above `8.325`, especially with the anchor and position-0 decompositions, would materially strengthen venue readiness.

## Frontier-Only Objective Assessment

The objective now matches the support of deterministic `T=0` EAL: accepted positions and the first rejection are exactly the positions reachable before verification terminates. Masking later wrong-prefix suffixes is cleaner than full Draft-OPD suffix replay for this target.

It is still a surrogate rather than the exact non-differentiable EAL objective. A uniform first-rejection margin guarantees only an immediate one-token repair; the longer extension appears after the repaired policy is rolled out again. The proposed v1 refresh is therefore not optional bookkeeping—it is what exposes the new later frontier and makes the surrogate consistent with multi-token EAL improvement.

With accepted-loss normalization and the policy refresh retained, this is the correct minimal objective.

## All-16 Adaptation Assessment

All-16 correction should remain mandatory. It removes a measured 11% structural reachability failure while adding only a reuse of the existing head and a zero-initialized gate. Position 0 must be included in:

- the candidate/oracle computation;
- target alignment tests;
- beneficial/harmful reporting;
- latency measurement;
- current-policy refresh.

This is part of the minimal sufficient route, not optional architectural embellishment.

## Simplification Opportunities

1. Use direct full-vocabulary head adaptation for Stages A–C and move K17/K16 restriction to an explicitly optional deployment-contraction stage. This removes the frozen-reference-policy ambiguity.
2. Normalize the accepted-prefix preservation term per block; do not add a separate preservation gate or calibration model.
3. Keep the candidate-conditioned lattice adapter closed until a matched direct-head run has positive held-out signal and a documented capacity plateau.

## Modernization Opportunities

NONE. The one-refresh, policy-versioned frontier replay is already the appropriate modern mechanism. Additional RL, tree search, or iterative DAgger would add complexity without addressing a demonstrated need.

## Required Revisions

### CRITICAL

1. **Unify the action-space and score contract.** Specify whether Stages A–C use direct full-vocabulary adapted Domino logits or a frozen released-score-plus-candidate-residual policy. Make candidate availability, identity, training loss, inference computation, and latency consistent with that choice.

### IMPORTANT

1. Define target labels as draft-prefix-conditioned greedy IDs `g_i=argmax T(c,d_<i)` and explicitly cover GRU reset and the `r=16` bonus-token case.
2. Average/cap accepted-prefix preservation per block so long already-correct blocks cannot outweigh first-rejection repair.
3. Add paired confidence and explicit harmful-regression gates to Gates B, C, and final success.
4. Recompute parallel hiddens for any trainable-backbone/LoRA arm.
5. Require the final deployed candidate oracle—not merely a prototype K17 oracle—to clear the target with useful headroom.

## Drift Check

**NONE.** The proposal continues to target held-out exact-runtime EAL `≥8.325` and eventual SGLang throughput. The proof-of-signal gates do not replace the success condition.

## Verdict

**REVISE**

GFPR is now the correct route family and is close to implementation-ready. It should not pivot to a larger architecture. Resolve the single action-space/score-contract inconsistency and tighten the frontier-loss and harm gates; no additional mechanism is required to reach READY.
