# Adaptive Amendment: Fixed-Step Prompt-Diverse Feature Probe

**Frozen:** 2026-08-04 23:51:43 CST, before any full-data D640 launch  
**Evidence tier:** adaptive development diagnostic  
**Route:** positive-only; never supports an information-ceiling conclusion

## Why the 10K gate is not a compute-saving gate

The preregistered OPB-10K probe uses 79,931 blocks, batch size 64, and 30
epochs, or 37,470 optimizer updates. The existing full OPB collection has
793,989 blocks; three epochs give 37,221 updates. Thus the 10K and full-data
cells have effectively the same optimizer-step and model-example-processing
budget; full-data I/O and final-train evaluation are larger. The 10K cell
instead makes 30 epoch-level passes over each prompt/block bundle, versus
three full-data passes. With about eight blocks per prompt, that is roughly
240 block presentations per 10K prompt versus 24 per full-data prompt: ten
times more repetition while discarding roughly 90% of the available prompt
diversity.

The already-completed compact D64 learning curve makes this confound large:

| prompts | epochs | updates | raw EAL delta vs DFlash |
|---:|---:|---:|---:|
| 10,000 | 30 | 37,470 | +0.00996 |
| 25,000 | 12 | 37,476 | +0.10775 |
| 50,000 | 6 | 37,464 | +0.21927 |
| 99,356 | 3 | 37,221 | +0.24247 |

This historical compact curve used the earlier `loss_weighting=dpace`; the
new probes use `candidate_dpace`. The curve is supportive evidence for the
fixed-step diversity confound, not an objective-matched predecessor or a
substitute for the new compact control.

Therefore a negative D640 result at 10K can be caused by prompt-diversity
starvation or repetition overfit even when the tested frozen features support
held-out prediction. This is especially acute for the 27.5M-parameter probe.
The original 10K result remains reportable, but it no longer vetoes the
fixed-step prompt-diverse diagnostic. This is an explicit adaptive protocol
amendment, not a threshold relaxation.

## Information visible when this amendment was frozen

The amendment was written after debug array `10132757` began, so the last
partial trajectory explicitly inspected while making the amendment is
disclosed:

- compact D64 had reached epoch 18 and was already below DFlash at that epoch;
- D640 had reached epoch 3, with raw validation delta about `+0.03681`, oracle
  gap recovery about `0.00798`, and 64 harmed validation blocks;
- neither cell had completed, no final checkpoint had been selected, and the
  fail-closed summary had not run.

The full-data success thresholds below are identical to the 10K thresholds
and were not chosen from a completed D640 result.

## Frozen matched cells

| Cell | Mixer / encoder | Width / heads / layers | LR | Parameters |
|---|---|---:|---:|---:|
| compact reference | axial / additive | D64 / H4 / L1 | 6e-4 | 433,772 |
| high-capacity probe | flat / compatibility | D640 / H10 / L4 | 3e-4 | 27,482,160 |

Common configuration:

- Candidate-D-PACE, `alpha=0.5`, post-break weight 1, safety weight 0;
- global scope, top-16, dropout 0, batch 64, three epochs, weight decay 0;
- warmup ratio 0.04, gradient clip 1.0, seed 0;
- all eight completed OPB parts, 99,356 prompts and 793,989 blocks;
- exact prompt-set SHA256
  `45471a62f93a488f3f7653c096bebcddb0ddae3773f6c99744bd070e348a9405`;
- exactly 37,221 optimizer updates;
- request the `i64m1tga800u` partition with a four-hour fail-safe walltime;
  runtime `nvidia-smi` records the device actually allocated, but GPU model
  is not a scientific gate. The walltime is not an optimization budget and
  the artifact must still report exactly 37,221 updates;
- evaluation only on the existing 147-prompt / 1,175-block
  `validation_select`; the sealed gate stays unopened.

The two cells may differ only in the preregistered architecture fields and
learning rate. Trainer/head hashes, target identity, validation metadata, all
eight external metadata identities, common configuration, budget, and prompt
set must match fail-closed.

## Unchanged positive-only decision rule

The D640 cell is a positive witness only if either:

1. raw prompt-balanced EAL improvement over DFlash is at least `+0.6`; or
2. oracle-gap recovery is at least `0.15`.

Calibration and bootstrap intervals are descriptive and cannot pass the
gate. A positive result says only that the tested frozen inputs plus the high
capacity function class are sufficient for a material held-out mapping and
authorizes a separate compression/distillation project. A negative result is
an engineering stop for this frozen selector family; it does not prove the
inputs contain no useful information.

The compact-vs-probe paired prompt bootstrap and domain breakdown are
reported regardless of the gate outcome. No hyperparameter rescue is allowed
inside this two-cell run.
