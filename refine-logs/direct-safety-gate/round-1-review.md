# Round 1 Independent Review: Binary Direct Safety Gate

## Verdict

`REFINE`

Overall score: **6.3/10**. The route is problem-faithful and the binary oracle
is real, but the current objective is not population-aligned with
accepted-token utility, OPB gate supervision is in-sample for the producer,
the capacity pass would be nearly tautological, and a duplicate D64 backbone
is difficult to justify within the available latency headroom.

## Scorecard

| Dimension | Score |
|---|---:|
| Problem fidelity | 8.6 |
| Mathematical correctness | 6.3 |
| Method specificity | 6.7 |
| Objective-deployment alignment | 5.1 |
| Contribution quality | 5.6 |
| Frontier leverage | 6.3 |
| Optimization feasibility | 6.0 |
| Validation focus | 6.6 |
| Data/checkpoint discipline | 5.3 |
| Claim discipline | 8.0 |
| Venue readiness | 4.9 |

## Independently reconstructed evidence

The reviewer reconstructed all 1,175 validation examples from the frozen
Direct-control artifact.

| Policy | Prompt-balanced EAL |
|---|---:|
| DFlash | 5.112001943634597 |
| Direct-native | 5.334669582118561 |
| Direct-one-edit | 5.212099125364432 |
| Exact KEEP/Direct oracle | 5.430758017492711 |
| Exact three-way oracle | 5.438411078717201 |

Direct versus DFlash contains exactly 141 beneficial blocks gaining 374
unweighted block-tokens, 972 neutral blocks, and 62 harmful blocks losing 113
unweighted block-tokens. The binary oracle has 1,041 first-token-correct
blocks, versus 1,036 for Direct and 1,029 for DFlash.

The strict target is 5.396991943634597. Consequently:

- oracle slack above target is 0.033766073858114;
- required DFlash-to-binary-oracle recovery is 89.4069%;
- target improvement over Direct is 0.062322361516036;
- approximately 64.9% of Direct's avoidable loss must be recovered.

Oracle-tuned thresholds on change count and the sum, mean, minimum, and maximum
Direct margins did not exceed EAL 5.3381 under the safety constraints. The
remaining opportunity therefore requires contextual separation, not a scalar
confidence threshold.

## Retained correct mechanics

- Strict `APPLY iff z > 0`, with KEEP on ties, gives exact epoch-zero identity.
- If the Direct and DFlash paths are equal, forcing `z = 0` is correct.
- The original piecewise expression is the declared two-action
  cost-augmented hinge and pointwise bounds decoded normalized regret.
- At zero output, beneficial and harmful cases create opposite output
  gradients while neutral cases have PyTorch's zero ReLU subgradient.
- Direct and gate inference can be gold-free.
- Removing the maximum over 225 edit scores directly addresses CAMRS's
  observed extreme-tail failure.

The KEEP-versus-frozen-Direct action framing should be retained.

## Blocking revisions

### 1. Use a utility-consistent gain-weighted loss

The original conditional hinge can be sign-count consistent rather than
accepted-token-utility consistent when deployable features collide. Replace it
with

```text
y = sign(g)
loss(x) = |g| * ReLU(1 - y*z),  if g != 0
loss(x) = 0,                    if g == 0
```

and optimize the prompt-balanced empirical risk

```text
(1 / number_of_prompts) *
sum_p [(1 / blocks_in_prompt_p) * sum_{x in p} loss(x)].
```

This retains zero-threshold deployment, gain-scaled opposing gradients,
catastrophic-loss emphasis, and a pointwise regret bound for wrong decoded
actions. Its conditional decision sign follows expected signed gain rather
than positive-versus-negative example counts.

### 2. Remove producer in-sample gate labels

Do not train the meta-gate from OPB outcomes of the OPB-trained Direct
producer. Use the existing Phase-3 `train` collection, which OPB construction
explicitly excluded and which contains 1,987 collected prompts / 15,886
blocks. Before computing outcomes, hash-split prompt IDs within domain into
fit/checkpoint/falsifier sets totaling 1,587/200/200 prompts. Verify exact zero
overlap with the OPB producer prompts and all validation/reserved splits.

- Train only on fit.
- Select checkpoints only on checkpoint.
- Freeze once, evaluate falsifier once, and never refit.
- Exclude the previously inspected `validation_gate` split.
- Open `validation_select` only after a falsifier pass.
- Keep formal test sealed.

OPB producer outcomes can be descriptive diagnostics, but not gate-training
labels.

### 3. Add a producer-out-of-sample contextual-separability gate

The 512-block same-subset test is a memorization/plumbing check, not
generalization evidence. Before `validation_select`, the frozen 200-prompt
falsifier must pass under raw `z > 0` deployment with at least:

- prompt-balanced binary-oracle gap recovery at least 0.90;
- harmed fraction at most 0.05;
- no more than one first-token-correct block below frozen Direct;
- positive EAL against both DFlash and Direct;
- zero regret-bound violations;
- clear improvement over frozen scalar-summary baselines;
- complete prompt-cluster and beneficial/neutral/harmful diagnostics.

Failure closes the contextual route before development evaluation.

### 4. Replace the duplicate backbone with a Direct-reusing sidecar

The original inherited selector interface neither explicitly supplied a Direct
path indicator nor Direct latent/score features. A second full 433,772-
parameter backbone is also inconsistent with the narrow 0.98%-1.52%
verification-advance headroom.

Reuse detached frozen Direct node states and scores. At each position expose
the chosen and rank-zero node states and their difference, Direct total-score
and residual margins, the DFlash log-probability difference, selected rank,
position, change indicator, entropy, and retained mass. Apply a small shared
MLP, pool across 15 positions with change count, and finish with a bias-free,
zero-initialized scalar projection. Force `z = 0` for identical paths. Native
Direct outputs must reproduce exactly before and after adding the sidecar.

### 5. Freeze exact capacity identities and denominators

Capacity examples must lie inside the producer-out-of-sample fit split and use
unique prompts plus unique keys `(sample_id, anchor_offset, context_length)`.
Rank within outcome strata using a frozen SHA256 of seed, block key, producer
hashes, and collection hash. Define beneficial as `g > 0`, harmful as `g < 0`,
and changed-neutral as `g = 0 and p_D != p_B`; fail if the requested
256/128/128 prompt-unique strata do not exist.

Report separately zero-regret utility optimality, KEEP-preferring tie-policy
agreement, benefit recall, harm avoidance, false APPLY, harmed fraction,
unclipped prompt-balanced oracle-gap recovery, and oracle regret.

### 6. Re-freeze optimization and checkpointing

Treat 5,120 capacity updates only as a generous same-subset memorization bound;
report first passage and select the earliest minimum-loss checkpoint. OPB's
37,221-step schedule is no longer relevant.

For Phase-3 fit, use at most 5,120 updates, 0.04 warmup, and evaluate at complete
deterministic passes. Select only on checkpoint by

```text
(prompt-balanced EAL, -harmed_count, -prompt-weighted_gain_hinge)
```

Epoch zero is eligible and exact ties keep the earliest checkpoint. Neither
falsifier nor `validation_select` may influence selection.

### 7. Tighten provenance and claims

Freeze the full runtime import closure, split manifest and prompt hashes,
outcome/path sidecar hashes, producer config/state/source hashes, canonical
shard and target-embedding hashes, exact native-Direct reproduction, start/end
source hashes, selected key, and all evaluated checkpoints.

The defensible novelty remains a narrow DFlash-specific, producer-reusing,
counterfactual full-path safety router. Top-venue claims require positive
producer-out-of-sample, formal-test, multi-seed, and end-to-end latency
evidence.

## Nonblocking risks

- Phase-3 fit has only 1,587 prompts and may lack enough diversity.
- The validation oracle is post hoc and `validation_select` also selected the
  Direct producer, so this remains development evidence.
- Offline canonical anchors may differ from online verification-boundary
  states.
- Neutral APPLY decisions are not EAL errors and must not be conflated with
  harm.
- Binary-oracle headroom leaves little robustness margin.
- Even positive offline EAL can lose wall-clock throughput unless the sidecar
  is fused and profiled.
- This is more plausibly a systems component than a standalone ICLR-level
  algorithm.

## Authorized next step

CPU implementation and semantic tests only are supportable for a
producer-reusing, producer-out-of-sample Direct Safety Gate trained with
prompt-balanced, gain-weighted unit-margin hinge supervision on Phase-3 fit,
selected on its checkpoint split, and gated by a one-shot contextual
falsifier.

No GPU job, outcome-manifest materialization job, or validation opening is
authorized by this review.

## Raw reviewer response

<details>
<summary>Full verbatim response from the independent reviewer</summary>

## Independent review

`REFINE`

Overall score: **6.3/10**. The route is problem-faithful and the binary oracle is real, but the current objective is not population-aligned with accepted-token utility, OPB gate supervision is in-sample for the producer, the capacity pass would be nearly tautological, and a duplicate D64 backbone is difficult to justify within the available latency headroom.

### Scorecard

| Dimension | Score |
|---|---:|
| Problem fidelity | 8.6 |
| Mathematical correctness | 6.3 |
| Method specificity | 6.7 |
| Objective–deployment alignment | 5.1 |
| Contribution quality | 5.6 |
| Frontier leverage | 6.3 |
| Optimization feasibility | 6.0 |
| Validation focus | 6.6 |
| Data/checkpoint discipline | 5.3 |
| Claim discipline | 8.0 |
| Venue readiness | 4.9 |

### Independently verified evidence

I reconstructed the validation examples from `direct_one_edit_10133585.json`:

| Policy | Prompt-balanced EAL |
|---|---:|
| DFlash | 5.112001943634597 |
| Direct-native | 5.334669582118561 |
| Direct-one-edit | 5.212099125364432 |
| Exact KEEP/Direct oracle | 5.430758017492711 |
| Exact three-way oracle | 5.438411078717201 |

Direct versus DFlash consists of exactly:

- 141 beneficial blocks, gaining 374 unweighted block-tokens;
- 972 neutral blocks;
- 62 harmful blocks, losing 113 unweighted block-tokens.

The binary oracle has 1,041 first-token-correct blocks, versus 1,036 for Direct and 1,029 for DFlash.

The strict target is 5.396991943634597. Therefore:

- oracle slack above target: **0.033766073858114**;
- required recovery of the DFlash-to-binary-oracle gap: **89.4069%**;
- target improvement over Direct: **0.062322361516036**;
- required recovery of Direct’s avoidable loss: approximately **64.9%**.

The supplied post-hoc scalar diagnostic strengthens the feasibility concern: even oracle-tuned thresholds on change count and sum/mean/min/max Direct margins did not exceed 5.3381 under the safety constraints. The required gain therefore depends on genuine contextual separability, not confidence thresholding.

All cited Direct/control hashes match the current artifacts and sources.

### What is correct

The finite-example mechanics are mostly sound:

- `APPLY iff z>0`, with KEEP on ties, gives exact epoch-zero identity.
- If `p_D=p_B`, forcing `z=0` is correct because both actions are identical.
- The proposed piecewise hinge is exactly the two-action cost-augmented hinge under the declared tie-broken oracle.
- With PyTorch’s zero ReLU subgradient, beneficial and harmful examples give opposite output-projection gradients at initialization, while neutral examples give zero gradient.
- The loss pointwise upper-bounds decoded normalized regret.
- The Direct path and gate forward path can be gold-free; target gold is only required for training labels.
- Removing the maximum over 225 edit scores directly addresses CAMRS’s observed extreme-tail failure.

These properties merit retaining the two-action framing.

## Blocking revisions

### 1. Replace the binary hinge with a gain-weighted, utility-consistent loss

The current loss is a pointwise regret bound but is not Fisher-consistent for the expected accepted-token decision when deployable features collide.

For example, suppose indistinguishable inputs have:

- `g=+1/15` with probability 0.9;
- `g=-1` with probability 0.1.

Then `E[g]=-0.04`, so KEEP is optimal. The proposed conditional hinge is minimized at positive `z`, because its slope between the negative and positive gains depends on sign counts, not token mass. It therefore chooses APPLY.

Neutral examples also receive a one-sided APPLY penalty despite having zero EAL regret, further changing the population target.

Use instead:

\[
y=\operatorname{sign}(g),\qquad
\ell(x)=
\begin{cases}
|g|\,\operatorname{ReLU}(1-yz), & g\ne0,\\
0, & g=0.
\end{cases}
\]

This preserves all desired properties:

- zero-threshold deployment;
- opposite gain-scaled gradients at initialization;
- pointwise regret upper bound whenever the decoded action is wrong;
- conditional decision sign determined by `E[g|features]`;
- catastrophic losses receive proportionally larger weight.

Use prompt-balanced example weights because the binding EAL estimand is prompt-balanced:

\[
L_{\text{train}}
=\frac1{|\mathcal P|}
\sum_{p\in\mathcal P}\frac1{n_p}\sum_{x\in p}\ell(x).
\]

Uniform-block training is not exactly aligned with the reported metric.

### 2. Do not train the meta-gate on OPB outcomes from an OPB-trained producer

The frozen Direct producer was trained on the same 793,989 OPB blocks proposed for gate supervision. That is classic stacked-learner in-sample leakage. The producer’s prompt-balanced gain is 0.29313 on its training diagnostic versus 0.22267 on validation—31.6% larger, although corpus shift prevents attributing the difference solely to overfitting.

A clean existing remedy is available. The Phase-3 train collection is producer-out-of-sample because OPB explicitly excluded the Phase-3 development manifest. It contains 1,987 collected train prompts:

- chat: 655 prompts / 5,230 blocks;
- code: 665 prompts / 5,320 blocks;
- math: 667 prompts / 5,336 blocks.

Before computing outcomes, hash-split those prompt IDs within domain into:

| Domain | Fit | Checkpoint | Falsifier |
|---|---:|---:|---:|
| Chat | 525 | 65 | 65 |
| Code | 531 | 67 | 67 |
| Math | 531 | 68 | 68 |
| **Total** | **1,587** | **200** | **200** |

Freeze the split from canonical prompt identity only. Verify zero overlap with OPB producer prompts and every validation/reserved split.

- Train only on `fit`.
- Select checkpoints only on `checkpoint`.
- Freeze once, evaluate `falsifier` once, and never refit.
- Exclude the already inspected `validation_gate`.
- Open `validation_select` only if the falsifier passes.
- Keep the formal test sealed.

OPB producer outcomes may remain descriptive diagnostics, but not gate-training labels.

### 3. Add a producer-out-of-sample contextual-separability gate

The 512-example same-subset gate cannot establish the required separability. With 433,772 parameters, 512 examples, and 5,120 updates—320 passes—it is predominantly a memorization/plumbing test.

Before `validation_select`, the frozen 200-prompt falsifier must pass with the raw `z>0` rule. At minimum require:

- prompt-balanced binary-oracle gap recovery ≥0.90;
- harmed fraction ≤0.05;
- no more than one first-token-correct block below Direct;
- positive EAL against both DFlash and Direct;
- zero regret-bound violations;
- clear improvement over frozen scalar-summary baselines;
- complete beneficial/neutral/harmful and prompt-cluster diagnostics.

A failure closes the contextual gate before the expensive development evaluation.

### 4. Replace the duplicate D64 gate backbone with a producer-reusing sidecar

The score family is not obviously incapable: because the axial node states are globally contextual, a gathered sum can represent rich functions. But the proposal says the gate “sees the frozen Direct path” without specifying a path indicator or Direct score/latent input; the inherited `GlobalDirectCandidateSelector` interface contains neither.

More importantly, the latency budget is extremely tight. Relative to Direct-native:

- reaching the strict EAL target improves verification advance by only about **0.98%**;
- even the perfect binary oracle improves it by only about **1.52%**.

A second complete 433,772-parameter axial backbone is therefore not authorized without profiling and is dominated by a smaller design.

Reuse detached frozen Direct node states and scores. For each position, construct features from:

- chosen and rank-zero frozen Direct node states and their difference;
- Direct total-score and residual margins;
- DFlash log-probability difference;
- selected rank, position, change indicator, entropy, and retained mass.

Apply a small shared MLP, pool across the 15 positions, include change count, and use a bias-free zero-initialized scalar projection. Force `z=0` when the paths are identical. Direct outputs must reproduce the frozen control exactly before and after adding the sidecar.

### 5. Make capacity construction and all denominators exact

Freeze the capacity subset only inside the producer-out-of-sample `fit` split. Use unique prompts as well as unique block keys.

Define the block key as `(sample_id, anchor_offset, context_length)` and rank within each outcome stratum by a frozen SHA256 of the seed, block key, producer hashes, and collection hash. Define:

- beneficial: `g>0`;
- harmful: `g<0`;
- changed-neutral: `g=0` and `p_D != p_B`.

Fail if the requested 256/128/128 prompt-unique strata are unavailable.

Separate these metrics:

- zero-regret utility optimality, for which either neutral action is optimal;
- tie-policy agreement, for which neutral must KEEP;
- benefit recall: applied beneficial / beneficial;
- harm avoidance: kept harmful / harmful;
- false APPLY: applied nonbeneficial / nonbeneficial;
- harmed fraction: applied harmful / all blocks;
- prompt-balanced recovery: `(EAL_gate-EAL_base)/(EAL_oracle-EAL_base)`, without clipping;
- oracle regret: `EAL_oracle-EAL_gate`.

The current phrase “utility-optimal binary decision” is ambiguous on neutral ties.

### 6. Re-freeze the optimization/checkpoint protocol

The existing budgets have different status:

- **5,120 capacity updates** equals 320 passes over 512 examples. It is defensible only as a generous upper-bound memorization test, not optimization or mechanism evidence. Report first passage and the selected earliest minimum-loss checkpoint.
- **37,221 development updates** equals `3 × ceil(793989/64)` and is arithmetically correct for OPB. Once OPB is correctly removed from gate training, it is no longer defensible.

For the Phase-3 fit route, preregister a step-based **5,120-update** maximum, warmup 0.04, and checkpoint evaluation at complete deterministic passes. Select only on the checkpoint split by:

```text
(prompt-balanced EAL, -harmed_count, -prompt-weighted weighted-hinge)
```

Epoch zero remains eligible and exact ties retain the earliest checkpoint. The falsifier and `validation_select` must never participate in checkpoint selection.

### 7. Tighten provenance and claim boundaries

The implementation package must freeze:

- full runtime import closure;
- gate split manifest and prompt hashes;
- outcome/path sidecar hashes;
- all producer config/state/source hashes;
- canonical shard and target-embedding hashes;
- exact native Direct reproduction;
- source hashes at start and end;
- selected-checkpoint key and all evaluated checkpoints.

The novelty language is appropriately cautious, but the contribution remains narrow. SpecDec++ already trains acceptance heads, Hybrid Verified Decoding predicts accepted payoff and routes between draft sources, SelectiveNet covers learned rejection, and BASTION combines accepted-length surrogates with hardware-aware control. [SpecDec++](https://arxiv.org/abs/2405.19715), [Hybrid Verified Decoding](https://openreview.net/forum?id=vr5iRoUn0I), [SelectiveNet](https://mlanthology.org/icml/2019/geifman2019icml-selectivenet/), [BASTION](https://arxiv.org/abs/2605.29727).

The defensible novelty is only a DFlash-specific, producer-reusing, counterfactual full-path safety router. Top-venue claims require positive producer-out-of-sample, formal-test, multi-seed, and end-to-end latency evidence.

## Nonblocking risks

- Phase-3 fit has only 1,587 prompts; the falsifier may reveal insufficient diversity.
- The validation oracle was discovered post hoc and `validation_select` also selected the Direct producer, so all resulting evidence remains development-only.
- Offline canonical anchors may differ from online verification-boundary states.
- A gate that applies many neutral paths is not an EAL error; report it separately rather than conflating it with harm.
- The binary oracle caps attainable improvement and leaves little robustness margin under domain shift.
- Even a successful EAL result may reduce wall-clock throughput unless the sidecar is fused and profiled.
- The contribution is likely an engineering/system component rather than a standalone ICLR-level algorithm.

## Strongest design authorized

I would authorize **CPU implementation and semantic tests only** for:

> A producer-reusing, producer-out-of-sample Direct Safety Gate trained with prompt-balanced, gain-weighted unit-margin hinge supervision on the frozen Phase-3 fit split, selected on its checkpoint split, and required to pass a one-shot contextual-separability falsifier before touching `validation_select`.

This retains the elegant KEEP-versus-frozen-Direct decision while fixing the objective, leakage, validation, and latency problems. It does **not** authorize any GPU job, manifest-generation job, or validation opening.

</details>
