# Round 2 Independent Re-review: PROS-Gate

**Verdict:** `REFINE`  
**Overall score:** **8.8/10**  
**Authorization:** none; three exact protocol revisions remain.

The core route passed conceptual review. The remaining blockers are the unique
Phase-3 split table, the complete capacity optimizer/selected-checkpoint
contract, and bit-level feature/schedule arithmetic.

## Raw reviewer response

<details>
<summary>Full verbatim response from the independent reviewer</summary>

# Round 2 Independent Re-review: PROS-Gate

`REFINE`

**Overall score: 8.8/10**

The refinement resolves the major conceptual failures: the new loss is utility-consistent, gate data are producer-out-of-sample, the one-shot falsifier is meaningful, and the 38,674-parameter sidecar is a strong simplification. However, three binding protocol ambiguities remain, so the current document is not yet implementation-authorizing.

`CALIBRATION: none`

## Scorecard

| Dimension | Score |
|---|---:|
| Problem fidelity | 9.6 |
| Mathematical correctness | 9.2 |
| Method specificity | 8.7 |
| Objective–deployment alignment | 9.3 |
| Contribution quality | 8.1 |
| Frontier leverage | 8.5 |
| Optimization feasibility | 8.6 |
| Validation focus | 9.4 |
| Data/checkpoint discipline | 8.5 |
| Claim discipline | 9.4 |
| Venue readiness | 8.0 |

## Round-1 blocker status

| Round-1 blocker | Status | Assessment |
|---|---|---|
| Utility-inconsistent hinge | **Resolved** | Gain-weighted unit-margin hinge has the correct conditional decision and regret bound. |
| OPB producer/train leakage | **Resolved in principle; exact split conflict remains** | Phase-3 train is producer-OOS, but the document’s per-domain counts differ from the intended canonical split. |
| Missing contextual falsifier | **Resolved** | Frozen one-shot 200-prompt falsifier, no refit, raw threshold, and scalar comparator are well designed. |
| Duplicate D64 backbone | **Resolved; minor specification gaps remain** | Producer-state reuse and parameter arithmetic are correct. |
| Capacity identities/denominators | **Mostly resolved** | Neutral utility/tie semantics are separated, but optimizer and selected-checkpoint passage are underspecified. |
| 37,221-step schedule/checkpointing | **Resolved conceptually; reproducibility details remain** | Clean fit uses complete passes under a 5,120-update cap. |
| Provenance/claims | **Resolved** | Boundaries are appropriately strict and development-only. |

## Mathematical verification

### Gain-weighted hinge

Let

\[
a=E[g_+\mid f],\qquad b=E[(-g)_+\mid f].
\]

For \(z\in[-1,1]\), the conditional risk is

\[
R(z)=a(1-z)+b(1+z)=a+b+(b-a)z.
\]

Therefore:

- if \(E[g\mid f]=a-b>0\), the minimizer has \(z=1\);
- if \(E[g\mid f]<0\), it has \(z=-1\);
- if \(E[g\mid f]=0\), either action is utility-optimal.

Thus the deployed sign agrees with expected accepted-token gain, unlike the Round-0 hinge.

The regret bound also holds:

- beneficial block decoded KEEP implies \(z\le0\), so  
  \(\ell=g\operatorname{ReLU}(1-z)\ge g\);
- harmful block decoded APPLY implies \(z>0\), so  
  \(\ell=(-g)\operatorname{ReLU}(1+z)>-g\);
- correct and neutral decisions have zero regret.

### Prompt weighting

With

\[
w_x=\frac{N_{\rm blocks}}{N_{\rm prompts}n_p},
\]

a mean over all weighted records equals

\[
\frac1{N_{\rm prompts}}\sum_p\frac1{n_p}\sum_{x\in p}\ell(x).
\]

This exactly matches the prompt-balanced estimand, provided each split recomputes its own \(N_{\rm blocks}\), \(N_{\rm prompts}\), and \(n_p\).

### Sidecar arithmetic

The stated dimensions and parameter count are correct:

- position encoder: 17,424 parameters;
- block encoder and final projection: 21,250;
- total: **38,674**;
- producer ratio: **8.9157%**, correctly rounded to 8.92%.

The 257-dimensional block vector is also correct: four 64-dimensional pools plus one normalized change count.

## Blocking revisions

### 1. Resolve the binding Phase-3 split discrepancy

The canonical collected counts are chat 655, code 665, and math 667. The intended exact split is:

| Domain | Fit | Checkpoint | Falsifier |
|---|---:|---:|---:|
| Chat | **523** | **66** | **66** |
| Code | **531** | **67** | **67** |
| Math | **533** | **67** | **67** |
| **Total** | **1,587** | **200** | **200** |

The refinement currently specifies `525/65/65`, `531/67/67`, and `531/68/68`. Both versions total 1,587/200/200, but they generate different immutable manifests and therefore cannot coexist.

**Exact fix:** replace the table with `523/66/66`, `531/67/67`, `533/67/67`; retain the frozen within-domain hash ordering and assign fit, checkpoint, then falsifier prefixes accordingly.

### 2. Fully specify the capacity optimizer and make selected-checkpoint passage binding

The capacity section specifies batch size and update cap but not the optimizer, LR, warmup, scheduler, clipping, dropout, or weight decay. It also does not unambiguously state that the minimum-loss selected checkpoint itself must pass every behavior gate.

**Exact fix:**

```text
batch size                 32
passes                     320
steps per pass             16
total updates              5,120
optimizer                  AdamW
betas / epsilon            (0.9, 0.999) / 1e-8
learning rate              6e-4
weight decay               0
dropout                    0
gradient clip              1.0
warmup steps               floor(0.04*5120 + 0.5) = 205
post-warmup schedule       cosine to zero
seed                       0
capacity example weight    1/512 per prompt-unique record
```

After all 320 passes, select the earliest exact minimum-loss checkpoint. That selected checkpoint must satisfy every conjunctive gate. A passing nonselected checkpoint and “first passage” cannot rescue selection failure.

### 3. Freeze the remaining feature and schedule arithmetic

Several phrases still permit implementation divergence:

- “smoothly bounded retained top-K log mass” has no formula;
- LayerNorm affine/bias/epsilon and Linear biases are implicit;
- normalized change count is not explicitly defined;
- warmup uses unspecified `round`;
- deterministic per-pass permutation lacks a version-independent construction.

**Exact fix:**

```text
entropy = -sum_k q_k log(q_k) / log(16)
retained_mass = tanh((logsumexp(topK_logits)-base_logsumexp)/2)
rank_feature = selected_rank / 15
position_feature = position / 14
change_count = number_changed / 15
LayerNorm = affine=True, eps=1e-5
all hidden Linear layers = bias=True
final Linear(64,1) = bias=False
warmup = floor(0.04*total_steps + 0.5)
```

Order each pass by sorting block keys on:

```text
SHA256("pros-fit-order-v1\0" || pass_index || "\0" ||
       block_key || "\0" || fit_manifest_sha256)
```

with the unhashed block key as the collision fallback. Freeze every pass-order hash before training.

Also require recovery denominators to be finite and strictly positive; fail if recovery is outside `[0, 1+1e-6]` rather than clipping it.

## Remaining nonblocking weaknesses

- The contribution remains a careful DFlash-specific stacking/router design, not a broadly new learning principle.
- The contextual sidecar has no separability evidence yet; the falsifier correctly treats this as the central kill test.
- Phase-3 contains only 1,587 fit prompts and correlated anchors.
- `validation_select` is already development-observed and selected the Direct producer; it cannot become fresh evidence.
- The validation oracle leaves only 0.033766 EAL slack.
- The sidecar adds roughly 274K dense multiply-adds per block plus pooling and LayerNorm. This is small, but only fused online profiling can determine whether it fits the 0.98%–1.52% headroom.
- Offline anchors may not match online verification-boundary states.
- Paper readiness still requires formal evaluation, multiple seeds, online rollout, and end-to-end latency.

## Simplification opportunities

1. Keep the scalar ridge model strictly as a falsifier baseline; do not turn it into a second deployable contribution.
2. Keep capacity as plumbing/memorization evidence only. Do not present its pass in the method claim.
3. Do not add probability calibration, learned thresholds, auxiliary losses, or another backbone.

## Modernization opportunities

**NONE.** Reusing globally contextual frozen producer states is the appropriate foundation-model-era primitive. RL, LLM critics, or calibration modules would add complexity without addressing the bottleneck more directly.

## Drift warning

**NONE.** The proposal still directly targets the anchored problem: safely recover the realizable accepted-length benefit of a fixed Direct path without target calls or autoregressive candidate selection.

The dominant contribution is substantially sharper and simpler than Round 0. The remaining issues are exact protocol and implementation-definition defects, not a need to change the core route.

## Authorization

No implementation or GPU execution is authorized by this `REFINE` review. After the three exact revisions above, one short re-review is warranted. The strongest route remains:

> Frozen Direct producer → detached globally contextual producer states → 38,674-parameter zero-initialized sidecar → prompt-balanced gain-weighted hinge on the producer-OOS Phase-3 fit split → checkpoint-only selection → one-shot falsifier → conditional development opening.

Any eventual positive review may authorize CPU implementation and semantic tests only.

</details>
