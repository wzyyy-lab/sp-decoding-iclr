# Experiment Plan

**Problem:** released Domino's accepted draft length is 7.015792 on the local
same-anchor Qwen3-4B benchmark and must be exceeded.  
**Method thesis:** acceptance-frontier adaptation of Domino's causal correction,
followed only when necessary by target replay or added interaction capacity,
can recover more of the available single-chain path quality.  
**Date:** 2026-08-08

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1: the final method exceeds released Domino under the same exact-verification metric | This is the user's hard outcome | Same-job paired EAL gain on clean held-out data, point `>0`; target `>=7.5`, preferably CI lower `>0` | B1, B2, B4 |
| C2: the gain comes from frontier/state alignment rather than a static logit rescale or independent expert | Determines which mechanism solved the bottleneck | Final method beats released Domino, global scale, and fixed DeLS fusion; objective/gate deletion identifies the useful part | B0, B3 |

**Anti-claim to rule out:** a gain caused only by selecting a favorable hardware
rounding path or repeatedly tuning the confirmation split.

## Paper/engineering storyline

- Must prove: same-anchor EAL superiority and identify the mechanism responsible.
- Can support later: per-domain robustness, throughput, and scale to 100K data.
- Intentionally cut now: frozen selectors, hash/provenance closure, large
  baseline lists, and paper-format formalities before the performance gate.

## Experiment blocks

### B0: Released-head diagnostics

- **Claim tested:** static calibration or a fixed local expert is insufficient.
- **Data:** phase-3 validation-select, 147 prompts / 1,175 anchors.
- **Systems:** released Domino; correction scales 0--2; Domino+released DeLS.
- **Primary metric:** prompt-balanced accepted draft tokens over 15 positions.
- **Status/result:** completed. Best global scale gives `+0.018586`, CI crosses
  zero. Positive DeLS weights hurt; best fusion is Domino-only scale 0.75.
- **Interpretation:** proceed to learned adaptation.
- **Priority:** MUST, completed.

### B1: Cached correction-head adaptation

- **Claim tested:** local frontier-aligned head training can exceed the released
  correction without changing the parallel backbone.
- **Data:** train split for optimization; validation-select for model selection.
- **Compared systems:** released head, REACHABLE-BREAKER, DECAY-CE,
  DYNAMIC-FRONTIER; final-head AUF is diagnostic only.
- **Setup:** released GRU/rank-256 initialization; initially freeze the large
  vocabulary projection and train GRU/rank input with L2-SP; frozen target/backbone;
  3 seeds only after a variant clears `+0.10`; small LR/epoch screen first.
- **Metrics:** on-policy prompt-balanced EAL primary; conditional acceptance
  by position, first reachable breaker, teacher loss/accuracy, per-domain EAL,
  full-horizon rate, and harmful/beneficial block changes.
- **Success criterion:** best variant `>=7.25` and paired delta `>=+0.10` on
  validation-select; otherwise Stage 2 or B4 is triggered.
- **Failure interpretation:** the released head is not merely distributionally
  miscalibrated, or 15.9K cached blocks are insufficient.
- **Priority:** MUST.

### B2: Clean held-out confirmation

- **Claim tested:** C1.
- **Data:** validation-gate, untouched during B1 selection.
- **Systems:** freshly rerun released Domino and one frozen selected method.
- **Metrics:** paired prompt-balanced EAL and prompt-cluster bootstrap; per-domain
  deltas, first-token/full-horizon rates, draft latency.
- **Success criterion:** selected method strictly exceeds same-job Domino;
  target `>=7.5`; preferred CI lower `>0`.
- **Failure interpretation:** selection overfit; move to replay/capacity stage
  without retuning on validation-gate.
- **Priority:** MUST, only after a method is frozen.

### B3: Mechanism deletion

- **Claim tested:** C2.
- **Data:** validation-select; validation-gate only for the final deletion if C1
  has already passed.
- **Variants:** selected objective versus DECAY-CE; learned gate off if present;
  target replay off if present; global 0.9 scale and fixed DeLS controls.
- **Success criterion:** the claimed component accounts for a meaningful share
  of the paired gain, not just teacher loss.
- **Priority:** MUST after positive B1/B4; otherwise cut.

### B4: Conditional target replay / capacity escalation

- **Trigger:** B1 gains `<+0.20` or stays below 7.5.
- **Run order:** proposal-prefix target replay -> joint final-backbone-layer
  tuning -> learned adaptive gate -> 2/4/6-pass block refiner.
- **Rule:** evaluate one escalation at a time and retain only improvements in
  on-policy EAL.
- **Success criterion:** validation-select `>=7.5` and `>=+0.20` over same-run
  Domino before held-out confirmation.
- **Failure interpretation:** move to the next capacity level; do not return to
  frozen top-K selectors.
- **Priority:** MUST until the hard target is met.

## Run order and milestones

| Milestone | Goal | Runs | Decision gate | Cost | Main risk |
|---|---|---|---|---:|---|
| M0 | Metric and route sanity | R001--R003 | Same-run Domino reproduced; static routes characterized | completed, <0.2 GPU-h | hardware rounding |
| M1 | Cache Domino features | R004 | exact semantic replay; baseline EAL reproduced from cache | 0.2--0.5 GPU-h | context/shift-label alignment |
| M2 | Reachable-head objective screen | R005--R007 | one variant `>=+0.10`; otherwise expand capacity | 1--3 GPU-h | small-data overfit |
| M3 | Scale best head | R008 | `>=7.5` on select or launch B4 | 2--10 GPU-h | released checkpoint already near local optimum |
| M4 | Held-out confirmation | R009 | same-job EAL > Domino | <1 GPU-h | selection does not transfer |
| M5 | Conditional escalation | R010+ | target `>=7.5`, then confirm | 5--60 GPU-h | target cache/iterative latency |

## Compute and data budget

- Immediate cached-head stage: approximately 3--8 GPU-hours including seeds.
- Target-replay stage: approximately 5--15 GPU-hours on phase-3 data.
- Joint backbone or 100K scale: approximately 20--80 A800 GPU-hours.
- No human annotation.  Existing exact target-generated continuations are used.
- Biggest bottleneck: materializing richer target conditionals if head-only
  clean-prefix training saturates.

## Risks and mitigations

- **Small hardware-dependent baseline drift:** always compare paired systems in
  one job/device and rerun the released checkpoint.
- **Teacher metric improves while on-policy EAL falls:** select only by on-policy
  EAL; teacher accuracy is diagnostic.
- **Wrong-prefix label error:** target replay obtains target conditionals on the
  actual proposed prefix; original continuation labels are never reused there.
- **Select split overfit:** validation-gate remains unopened until one method is
  frozen; a failed confirmation does not authorize gate tuning.
- **Head capacity ceiling:** escalate to joint backbone or iterative refinement,
  both supported by current literature.

## Final checklist

- [x] Strongest local baseline reproduced and decomposed
- [x] Static scale and fixed expert controls tested
- [ ] Cached training/evaluation metric agrees with same-anchor evaluator
- [ ] Main method exceeds Domino on validation-select
- [ ] One frozen method exceeds Domino on clean held-out data
- [ ] Mechanism deletion explains the gain
- [ ] Throughput and domain behavior reported
