CALIBRATION: none

# Round 3 Review: GFPR — Unified Full-Vocabulary Greedy-Frontier Policy Replay

## Summary Verdict

Round 2 的实质阻塞已经闭合。Stages A–C 现在是一条单一、明确的 direct full-vocabulary adapted Domino policy；K16/K17 已降为 oracle 与可选部署 contraction，不再混入主训练策略。Target prefix、GRU reset、`r=16` bonus、frontier loss、harm gate、LoRA stale-hidden 禁令和位置 0 的完整成本均已写清。

该方案已经是解决当前问题的最小充分研究计划。没有必要为了形式再增加新架构、损失或实验。

## Scores

| Dimension | Weight | Score | Weighted |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 10 | 1.50 |
| Method Specificity | 25% | 9 | 2.25 |
| Contribution Quality | 25% | 9 | 2.25 |
| Frontier Leverage | 15% | 9 | 1.35 |
| Feasibility | 10% | 9 | 0.90 |
| Validation Focus | 5% | 10 | 0.50 |
| Venue Readiness | 5% | 9 | 0.45 |
| **Weighted composite** | **100%** |  | **9.20 / 10** |

**GAP:** The proposal exceeds the READY threshold by `0.20`. There is no remaining method-design gap. The unresolved gap is empirical: GFPR still has to demonstrate that policy-correct frontier supervision plus position-0 adaptation can produce the unusually large `+1.085` EAL improvement. Gates A–E now falsify that assumption without weakening the success condition, so this empirical uncertainty is not a proposal blocker.

## Round 2 Requirements Re-check

| Requirement | Status | Assessment |
|---|---|---|
| A–C unified as direct full-vocabulary policy | Resolved | Actions are `argmax` over adapted full-vocabulary Domino scores; no frozen reference head or candidate mask is used. |
| K16 only oracle/optional contraction | Resolved | K16/K17 do not define A–C actions or training loss. |
| Draft-prefix-conditioned target ID | Resolved | `g_i = argmax T(v | c,d_<i)` eliminates independent-target-continuation ambiguity. |
| GRU reset and selected-token feedback | Resolved | State resets per block, consumes anchor first, then actual selected tokens. |
| `r=16` bonus semantics | Resolved | Position-16 bonus is appended and becomes the next block anchor. |
| Per-block normalized keep loss | Resolved | Accepted-prefix preservation is divided by `max(q,1)` and cannot dominate repair merely because the prefix is long. |
| Paired bootstrap and harm gates | Resolved | Both proof-of-signal and final success require paired uncertainty and explicit loss/gain constraints. |
| LoRA stale-hidden prohibition | Resolved | Trainable-backbone arms must recompute or recollect hiddens. |
| Position-0 full-head cost | Resolved | The proposal counts and profiles a sixteenth full-vocabulary correction application. |

## 1. Problem Fidelity — 10/10

The proposal remains fully anchored to the actual objective:

- Fixed held-out EAL must reach `8.325`.
- True dynamic-rollout improvement must also reach `1.15×`.
- Harm constraints must pass.
- `7.55` remains only a scaling permission.
- SGLang work remains gated behind acceptance-length success.

The method directly addresses the two demonstrated deficiencies: incorrect state distribution and an uncorrectable position 0. It does not substitute training loss, static capacity, or a small improvement for the requested outcome.

## 2. Method Specificity — 9/10

### Policy and score contract

Stages A–C are now unambiguous:

\[
d_i^\theta=\arg\max_{v\in V}s_i^\theta(v).
\]

Position 1–15 use the adapted full-vocabulary Domino head. Position 0 reuses that head with `α₀=0` initialization. There is no candidate availability mask, renormalized candidate distribution, or second frozen policy at inference.

Exact released identity is correctly defined by:

- released initialization for the GRU/head;
- `α₀=0`;
- token-by-token reproduction across all 16 positions.

### Target and state semantics

The definition

\[
g_{m,i}=\arg\max_v T(v\mid c_m,d_{m,<i})
\]

is the correct teacher label for deployed greedy verification. For every position before the current first mismatch, teacher-prefix GRU execution equals actual current-policy execution because all earlier predicted tokens equal gold. Thus the current-frontier forward is not teacher-forcing leakage.

The proposal also correctly distinguishes:

- within-block frontier movement, exposed immediately by recomputing `q_θ`;
- across-cycle anchor movement, exposed by the mandatory v1 rollout refresh.

### Minor non-blocking note

If Stage D opens candidate contraction, K16/K17 oracle and identity should be recomputed using the frozen Stage-C scorer that actually defines the new union, rather than relying only on the released-Domino Gate-A union. The Stage-D text already implies this; making it an explicit entry criterion would remove the last possible implementation ambiguity.

## 3. Contribution Quality — 9/10

The contribution is now singular and technically meaningful:

> Exact-greedy block-parallel drafts should be adapted at policy-induced reachable frontiers, including the pure-parallel first position.

Actual anchors, current-frontier repair, accepted-prefix protection, position-0 coverage, and one policy refresh are not separate contributions. They form one internally necessary correction to the training/deployment mismatch.

Reusing the released architecture improves the scientific design. It ensures that a positive result can be attributed to the state distribution and causal training target rather than an unexplained increase in model capacity. The candidate-conditioned residual is appropriately conditional on evidence and does not dilute the main claim.

## 4. Frontier Leverage — 9/10

GFPR now uses on-policy learning in a regime-appropriate form:

- target-assisted actual policy cycles provide stable contexts;
- only deterministic greedy reachable states are supervised;
- the first current rejection is recomputed as parameters change;
- a policy refresh exposes new cross-cycle states;
- sampling-oriented reverse KL and wrong-prefix suffix replay are excluded.

This is a more natural use of on-policy distillation than copying Draft-OPD’s full objective into a candidate-restricted greedy selector. No additional RL or repeated DAgger loop is justified at this stage.

## 5. Feasibility — 9/10

The implementation path is realistic and bounded:

- one dynamic/fixed-control collector;
- reuse of existing target passes and Domino infrastructure;
- a 2K falsification screen before data scaling;
- only the existing 50.8M causal head plus one scalar parameter is trained;
- one additional position-0 full-vocabulary correction call is explicitly profiled;
- one policy refresh rather than an open-ended loop;
- raw contexts are retained for any later trainable-backbone arm.

The required effect is ambitious, but the proposal does not confuse ambition with implementation infeasibility.

## 6. Validation Focus — 10/10

The validation ladder is minimal and decisive.

### Gate A

Establishes exact identity, correct rollout semantics, all-position oracle headroom, and position-0 cost before training.

### Gate B

Separates the two causal hypotheses:

1. static versus dynamic anchors;
2. positions 1–15 versus all 16.

The same drafting and labeling code makes the control meaningful.

### Gate C

Tests whether the signal survives realistic data scale and the required policy refresh. It retains `8.325` as the only method-success threshold.

### Gates D–E

Capacity changes are opened only after demonstrated signal, and system integration occurs only after EAL success.

The normalized keep loss is now appropriate:

\[
\frac{\lambda_{\rm keep}}{\max(q_\theta,1)}
\sum_{i<q_\theta}[m_{\rm keep}-\Delta_i]_+.
\]

It caps preservation pressure per block while keeping the first rejection primary. Paired prompt-cluster bootstrap and explicit gained/lost-token and harmful-prompt gates prevent a positive mean from hiding widespread regressions.

## 7. Venue Readiness — 9/10

If the hard gates pass, the resulting story is sharp enough for a top venue:

- a large static dataset fails because its anchors are not policy states;
- greedy acceptance is governed by the current reachable frontier, not wrong-prefix suffix imitation;
- the released interface leaves position 0 structurally unrepairable;
- a minimal on-policy adaptation closes a substantial oracle gap without introducing another large model;
- the EAL gain is verified under both historical fixed evaluation and true dynamic rollout, then translated to throughput.

The lack of a new architecture is not a venue-readiness defect. For this problem, an effect-backed correction of the training state distribution is more compelling than an unnecessary module.

## Frontier Objective Assessment

The objective is now correctly aligned with deterministic `T=0` EAL to the degree possible with a differentiable local surrogate.

- Accepted positions are protected.
- The current first rejection is repaired.
- Later unreachable suffixes are excluded.
- Repairing the frontier exposes later positions on the next forward.
- Changing cycle lengths and future anchors is handled by v1 recollection.

This is the correct minimal bridge between local boundary optimization and multi-token accepted-length improvement.

## All-16 Assessment

Position-0 correction remains mandatory and is now properly specified:

- GRU zero state is reset per block;
- anchor is consumed before the first decision;
- the existing correction head is reused;
- `α₀=0` preserves released identity;
- selected position 0 updates the state for position 1;
- its complete full-vocabulary cost is profiled;
- it participates in rollout refresh, harm reporting, and oracle analysis.

No separate position-0 network or gate model is needed.

## Critical Revisions

NONE.

## Important Revisions

NONE.

## Simplification Opportunities

NONE. The primary route is already appropriately minimal. Stage D must remain closed unless Stage C provides significant but insufficient held-out improvement.

## Modernization Opportunities

NONE. Policy-versioned greedy-frontier replay is already the appropriate modern primitive; additional RL, tree search, or iterative agentic optimization would be forced.

## Drift Check

**NONE.** The proposal still solves the original held-out EAL and throughput problem, and every intermediate gate is explicitly prevented from replacing the final target.

## Verdict

**READY**

GFPR is implementation-ready as a research proposal. Proceed with Gate A and Gate B exactly as specified. READY here certifies the method plan and falsification structure; it does not pre-claim that the unrun method has achieved `8.325`.
