# Research Proposal: Policy-Reach Supervision for Axial GCLS

## Problem Anchor

- **Bottom-line problem:** Improve DFlash's greedy single-chain accepted length with a fixed-depth, draft-side selector while ultimately closing meaningful ground against Domino, without tree verification or a sequential GRU.
- **Must-solve bottleneck:** The current axial GCLS sees useful bidirectional candidate-lattice signal, but repairs only 18.29% of first-miss opportunities and recovers 6.18% of the K16 oracle gap; its Candidate-D-PACE training support includes positions that the selector's own greedy path can no longer reach, and training loss can improve while accepted length collapses.
- **Non-goals:** Do not rescue the closed flat full-lattice C1a/C1b route; do not add CRF/Viterbi, recurrent decoding, target calls, tree verification, target-label leakage, a new representation module, or joint/LoRA training in this route.
- **Constraints:** Frozen released DFlash-b16 and Qwen3-4B target embedding; K16, 15 positions, greedy temperature 0; existing prompt-disjoint OPB train and development artifacts; one objective factor changed at a time; sealed test remains unopened; development compute should fit an A40 debug job.
- **Success condition:** On a matched axial D64/L1 seed-0 OPB-25K screen, a policy-reach objective must beat a newly rerun Candidate-D-PACE control in raw prompt-balanced EAL, not increase harm by more than 1 percentage point, and show that direct loss after the current first breaker is exactly zero. Only then may the objective be confirmed on full OPB-99,356 with three seeds and prompt-cluster intervals.

## Technical Gap

The best existing axial GCLS already establishes the useful structural fact: with 99,356 training prompts, global scope beats matched local and causal controls over three seeds. The failed flat screen further says that unstructured 240-node attention is not the missing ingredient. The remaining training failure is more specific.

Candidate-D-PACE censors only after the gold token leaves K16. If every gold token stays in K16 but the current selector chooses a wrong token at position 3, positions 4–15 still receive their own classification losses even though they cannot affect this block's greedy accepted length. The flat-compat run makes this mismatch observable: its training objective fell every epoch, but raw EAL peaked at epoch 4 and fell below DFlash by epoch 9 while harms more than doubled.

Naively increasing width, depth, or data is insufficient. D128/L2 does not reliably beat D64/L1; 50K→99K scaling is diminishing; and the just-closed flat route underperforms axial. The smallest adequate intervention is therefore to change only which per-position losses are active, leaving the model, inputs, logits, optimizer, inference, and checkpoint metric unchanged.

Two routes were considered:

- **Route A — policy-reach supervision:** supervise all positions up to and including the current selector's first wrong reachable candidate, and zero direct suffix loss after it.
- **Route B — 20M–50M frozen-feature teacher:** test whether much more capacity raises the information ceiling.

Route A is chosen first because it directly targets an observed train-metric/EAL divergence, costs no inference FLOPs, and is a cheaper falsifier. Route B remains the next diagnostic if Route A fails; combining both now would destroy attribution.

## Method Thesis

- **One-sentence thesis:** A residual candidate selector should be trained on the prefix reachable by its own current greedy policy, including the breaker but excluding independent losses after it, because only that support matches longest-prefix verification.
- **Why this is the smallest adequate intervention:** It changes one detached binary support mask in the existing loss and adds no parameter or inference operation.
- **Why this route is timely:** It is policy-conditioned supervision for a frozen foundation-model draft head: the large pretrained model and lattice representation are reused, while training credit is aligned to the actual deployment policy rather than to gold candidate availability.

## Contribution Focus

- **Dominant contribution:** Head-AUF, an acceptance-until-failure support rule for training an extra top-K residual selector under longest-prefix verification.
- **Optional supporting contribution:** A small fixed coverage CE coefficient is only a stabilization ablation, not a co-equal method claim.
- **Explicit non-contributions:** No new attention architecture, loss smoothing theory, candidate lattice, D-PACE variant claim, repair/protection module, distillation, LoRA, or decoding algorithm.

## Proposed Method

### Complexity Budget

- **Frozen/reused:** Current `GlobalDirectCandidateSelector`, axial D64/H4/L1 global mixer, frozen DFlash hidden/logits, target token embeddings, data loader, optimizer, raw greedy inference, checkpoint selection, and calibration diagnostics.
- **New trainable components:** None.
- **New training mechanism:** One detached on-policy reach mask and an optional scalar coverage-CE coefficient.
- **Intentionally excluded:** first-miss margin, base protection, dynamic temperature, compatibility encoder, latent slots, target distillation, and backbone adaptation.

### System Overview

```text
frozen DFlash lattice -> unchanged axial GCLS -> K16 scores -> greedy ranks
                                                |             |
gold candidate ranks ---------------------------+-------------+
                                                v
                                  detached reachable-prefix mask
                                                |
                         CE on reachable prefix + current breaker only
```

Inference is exactly the historical axial GCLS path; the new mask exists only during training/evaluation diagnostics.

### Core Mechanism

For block `b`, position `i`, candidate-support indicator `g[b,i]`, gold rank `r[b,i]`, and current greedy rank `r_hat[b,i]`, define:

```text
correct[b,i] = g[b,i] and (r_hat[b,i] == r[b,i])
reach[b,0]   = true
reach[b,i]   = product_{j<i} correct[b,j]
active[b,i]  = detach(reach[b,i]) and g[b,i]
```

The breaker remains active because reach checks only previous positions. If gold leaves K16 at the breaker, that position has no valid candidate label and both it and its suffix are inactive. The loss is:

```text
L_auf = (1/B) sum_{b,i} active[b,i] * CE(scores[b,i], r[b,i])
```

The optional stabilization cell adds:

```text
L = L_auf + 0.1 * L_coverage
```

where `L_coverage` is uniform CE on the existing gold-in-K prefix mask. It is predeclared as a separate cell. Pure Head-AUF remains the main mechanism, and the 0.1 term is retained only if it wins the same raw-EAL gate.

Future lattice nodes remain usable inputs: early reachable scores attend to all position summaries, so early loss gradients still update shared global pathways. What disappears is only a suffix position's independent output loss after the selector has already broken.

### Integration into the Base Pipeline

The CLI gains `loss_weighting=head_auf` and `coverage_aux_weight`. Metrics record policy-active positions, coverage-active positions, their ratio, and post-breaker coverage-only positions. The output schema keeps historical Candidate-D-PACE fields for the control. Epoch-zero score/path identity remains exact.

### Training Plan

1. **Unit and capacity gate:** synthetic masks prove breaker inclusion, zero post-breaker direct support, detached reach, and finite gradients. A fixed 128-block memorization probe compares Candidate-D-PACE and Head-AUF; every cell must meet the existing capacity thresholds without harm.
2. **Matched development screen:** axial D64/H4/L1, seed0, exact materialized OPB-25K, batch64, 12 epochs, lr `6e-4`, dropout/weight decay 0. Cells: Candidate-D-PACE; Head-AUF; Head-AUF + 0.1 coverage CE.
3. **Confirmation only after a win:** rerun the chosen objective and Candidate-D-PACE on OPB-99,356 with seeds0/1/2. Test remains sealed.

Checkpoint selection remains raw prompt-balanced EAL first, then first-token accuracy, minimum domain delta, and candidate accuracy. KEEP_BASE calibration is diagnostic and cannot choose the winner.

### Failure Modes and Diagnostics

- **Sparse self-paced support:** detect low policy/coverage active ratio and stalled repair; the fixed 0.1 coverage cell tests the minimal mitigation.
- **Objective gaming by conservative rank0:** raw EAL, first-miss repair, hard accuracy, and selected-rank counts expose it; identity alone cannot win because the control delta gate is strict.
- **High-variance moving support:** compare epoch curves and three seeds only after the seed-0 gate; do not add a soft-reach loss post hoc.
- **More repairs but more harms:** require harm within +1 percentage point of the matched control and first-token accuracy no worse by more than 0.001.
- **No material gain:** close the objective route and proceed to the separately frozen feature-ceiling diagnostic, not another objective sweep.

### Novelty and Elegance Argument

The novelty is not generic hard-example mining or prefix masking. Gold-coverage censoring asks whether a correct path remains representable; Head-AUF asks whether the deployed selector's own current greedy path remains verifiable. This distinction is specific and load-bearing for an extra residual selector whose actions can both repair and harm a strong base path. The route remains focused because it changes no inference mechanism. Whether it is sufficient for a paper-level claim depends on matched evidence; the immediate purpose is to remove a demonstrated optimization confound before judging frozen-feature selectability.

## Claim-Driven Validation Sketch

### Claim 1: Policy-reach support improves the existing axial selector

- **Minimal experiment:** three matched seed-0 OPB-25K cells.
- **Baselines/ablations:** Candidate-D-PACE, pure Head-AUF, Head-AUF+0.1 coverage.
- **Metric:** primary raw prompt-balanced EAL; secondary harm, first-token accuracy, repair, oracle-gap recovery, and prompt-active ratios.
- **Required direction:** best Head-AUF cell must exceed the rerun Candidate-D-PACE control, with harm ≤ control+0.01 and first-token drop ≤0.001.

### Claim 2: Any gain is not a moving-support single-seed artifact

- **Minimal experiment:** control versus frozen winner on full OPB-99,356, three seeds.
- **Metric:** paired prompt-cluster CI for raw winner-control and global-DFlash; harm one-sided analysis remains development-only until an untouched test is opened.
- **Required direction:** positive winner-control in every seed and CI lower bound >0.

## Experiment Handoff Inputs

- **Must prove:** exact mask semantics; no post-breaker direct loss; raw EAL improvement over a same-code control.
- **Must-run ablations:** pure Head-AUF versus 0.1 coverage; no other loss terms.
- **Critical data/metrics:** exact OPB-25K hash `a3d25eba...fa73`, 199,818 blocks, raw prompt-balanced EAL, harm, repair, rank buckets, active ratios.
- **Highest-risk assumptions:** detached moving support is stable enough; future evidence still trains through early queries; the historical misalignment is causal rather than merely correlated with overfitting.

## Compute & Timeline Estimate

- **Capacity:** less than 10 GPU-minutes total on A40 for three 128-block cells.
- **Development:** approximately 0.5–1.0 A40 GPU-hour total for three D64/L1 OPB-25K cells if run as an array.
- **Confirmation:** approximately 3–6 A40/A800 GPU-hours for six full-data runs, only after the development gate.
- **Implementation and tests:** less than one working session; no new data collection.
