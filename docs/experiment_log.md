# Experiment log

This file records conclusions, not just commands. Smoke-scale results are kept
separate from evidence that may appear in a paper.

## Evidence registry (authoritative)

Only two completed runs are research experiments at this point:

| Job | Tier | What it can support |
|---:|---|---|
| `10022338` | real, Gate 1a | Candidate availability and oracle upper bound on canonical anchors |
| `10022436` | real, development baseline | DFlash/Domino eager EAL and timing diagnosis; not final paper timing |

Jobs `10022278`, `10022310`, `10022330`, `10022343`, `10022412`, and
`10022468` are **non-evidence environment/plumbing smoke runs**. Their records
remain below only for debugging provenance. They must not enter a paper table,
an abstract claim, a gate decision, or a comparison with a trained method.

## 2026-07-21 — Non-evidence: environment and model smoke

- Slurm job: `10022278`
- Partition/device: `debug`, NVIDIA A40 48 GB
- Environment: `third_party/Domino/.venv` overlaying the verified
  `/hpc2hdd/home/zwang668/.venvs/sglang-py311` packages
- Runtime: Python 3.11.15, PyTorch 2.9.1+cu128, Transformers 4.57.1
- Models: local Qwen3-4B target, Qwen3-4B-DFlash-b16, and
  Qwen3-4B-Domino-b16

Result: both drafts loaded and generated the same 32-token target continuation.
Domino returned raw verification advances `[17, 15]`. Its checkpoint uses
`shift_label=true`, so it drafts 16 tokens and has a maximum raw advance of 17;
pure DFlash drafts 15 tokens and has a maximum raw advance of 16. All subsequent
tables must report both accepted draft tokens and raw verification advance.

## 2026-07-21 — Non-evidence: pure-DFlash canonical collector smoke

- Slurm job: `10022310`
- Data: 6 prompts (2 math, 2 code, 2 chat), 2 anchors per prompt, 12 blocks
- Block/candidates: 15 draft positions, saved top-64
- Collection time: 13.64 seconds on A40
- Peak allocated memory: 8.54 GiB

| K | Mean accepted draft tokens | Mean verification advance | Full-block coverage |
|---:|---:|---:|---:|
| 1 | 4.583 | 5.583 | 0.083 |
| 2 | 7.167 | 8.167 | 0.167 |
| 4 | 8.750 | 9.750 | 0.250 |
| 8 | 10.333 | 11.333 | 0.500 |
| 16 | 10.750 | 11.750 | 0.500 |
| 32 | 11.583 | 12.583 | 0.500 |
| 64 | 12.917 | 13.917 | 0.583 |

Interpretation: the implementation is viable and the tiny sample has substantial
candidate headroom. This is not a Gate 1 conclusion: confidence intervals are
wide and the four chat blocks happen to have full top-16 coverage. Formal job
`10022338` uses 96 prompts and 8 anchors per prompt on the A800 queue.

## 2026-07-21 — Non-evidence: unified eager baseline smoke

- Slurm job with detailed mismatch records: `10022343`
- Same 6-prompt manifest, 64 requested new tokens, A40

| Method | Mean accepted draft tokens | Mean verification advance | Decode tok/s | Wall tok/s |
|---|---:|---:|---:|---:|
| DFlash | 3.671 | 4.671 | 113.0 | 105.7 |
| Domino eager | 5.338 | 6.338 | 134.5 | 123.5 |

These timings are smoke diagnostics, not final benchmark numbers: there are only
six prompts and one warmup. They do show that candidate headroom remains after
the Domino gain on this sample.

HF one-token `generate()` and block target verification were not always bitwise
token-identical under bf16 SDPA. DFlash and Domino diverged from HF greedy at the
same token on two samples (positions 21 and 58), which points to target numerical
shape effects rather than an unverified draft token. Formal reporting will keep
logical verification correctness separate from strict floating-point token
identity and will test eager/SDPA attention plus target-logit margins.

## 2026-07-21 — Non-evidence: Domino state-pollution plumbing diagnostic

- Slurm job: `10022412`
- Data: the same 12 canonical blocks as the pure-DFlash smoke
- Domino checkpoint: shift-label block with 16 drafted positions

Domino base top-1 accepted 4.167 draft tokens on average; the on-policy GRU
raised this to 4.750. The base top-16 oracle accepted 11.583 draft tokens
(verification advance 12.583), so the small sample still has large candidate
headroom after accounting for Domino's 16-position formulation.

Teacher forcing made later predictions much cleaner: among 124 positions after
the first rejection, 56 were teacher-forced-correct but on-policy-wrong. GRU
state distance grew from 1.23 at position 2 to roughly 10.58 at position 16.
Nevertheless, teacher-forced and on-policy accepted-prefix lengths were
identical for every one of the 12 blocks. This is the direct empirical version
of the causal argument: state pollution is real, but starts only once the
current verification round can no longer accept a longer prefix. A useful
replacement head must change the decision at or before the first mismatch,
not merely repair the already unreachable suffix.

The next architecture therefore uses a genuinely parallel bidirectional mixer
plus candidate-transition scores and survival-risk path decoding. It replaces
the GRU loop; it is not an adapter stacked after Domino.

## 2026-07-21 — Non-evidence: end-to-end head-training plumbing smoke

- Slurm job: `10022468`
- Data: 6 train and 6 test blocks from the Domino diagnostic smoke
- Head: top-16, rank 32, 64-dimensional one-layer bidirectional attention mixer
- Trainable parameters: 529,697
- Runtime: 30 epochs in 3.50 seconds on A40, including validation each epoch

The then-current local-normalization path worked: targeted safetensors embedding load, candidate gather,
exact outside-mass normalization, prefix-censored NLL, backward pass, local/
Viterbi/survival decoding, metrics, and checkpoint saving. The six-block test
set stayed at the base advance (6.5) initially and fell to 6.33 after overfitting;
this is expected for a deliberately tiny plumbing test and is **not feasibility
evidence and not an experiment result**. It also predates the absorbing-OTHER
global prefix-CRF now used by the proposed trainer.

## 2026-07-22 — Real 4B candidate ceiling and unified eager baseline

Both queued A800 jobs completed normally on `gpu1-20` (NVIDIA A800-SXM4-80GB).
The stderr files contain only checkpoint progress bars and a harmless
Transformers warning about sampling flags under greedy decoding.

### Candidate ceiling (`10022338`)

- Data: 96 prompts, balanced across math/code/chat, with 8 fixed target-path
  anchors per prompt (768 canonical blocks).
- Draft: pure Qwen3-4B-DFlash-b16, 15 predicted positions, top-64 retained.
- Collection: 734.27 seconds; peak allocated GPU memory 8.62 GiB.

| K | Mean accepted draft tokens | Mean verification advance | 95% cluster-bootstrap CI (advance) | Full-block coverage |
|---:|---:|---:|---:|---:|
| 1 | 6.030 | 7.030 | [6.508, 7.613] | 12.24% |
| 2 | 7.708 | 8.708 | [8.117, 9.301] | 19.79% |
| 4 | 9.076 | 10.076 | [9.426, 10.689] | 28.52% |
| 8 | 10.143 | 11.143 | [10.557, 11.751] | 38.80% |
| 16 | 11.178 | 12.178 | [11.609, 12.708] | 48.57% |
| 32 | 12.010 | 13.010 | [12.418, 13.521] | 57.68% |
| 64 | 12.747 | 13.747 | [13.189, 14.224] | 67.71% |

The paired K=16 minus K=1 gain is 5.148 accepted draft tokens, with a
prompt-clustered 95% bootstrap CI of [4.724, 5.521]. At the first top-1 miss,
the gold token is within top-2/4/8/16 for 55.34%/80.27%/89.76%/95.25% of
blocks, respectively; only 1.93% lie outside the saved top-64. K=16 retains
91.55% of full-vocabulary probability mass on average.

**Gate 1a (candidate availability) passes.** The DFlash proposal distribution usually contains the
correct alternative, including at the first position that determines accepted
prefix length. The bottleneck is therefore not merely candidate generation;
there is substantial room for a better single-path selector. K=16 is the
initial operating point, with K=8 as the lower-cost ablation.

This is a gold-aware candidate-availability ceiling, not an achievable model
result. It proves that useful paths exist but does not prove that suffix features
identify them. Also, canonical K=1 (6.030) must not be compared directly with
online DFlash EAL (4.047): canonical anchors are uniform fixed target offsets,
whereas online EAL weights contexts by verification rounds. Full-prefix/no-cache
collection and DynamicCache block execution may also differ numerically.
Therefore Gate 1b requires Domino and the oracle on exactly the same stored
anchors, with reconstruction mismatch treated as a hard failure.

### Unified online baseline (`10022436`)

- Data: the same 96 prompts, up to 256 newly generated tokens each.
- Backend: eager Python draft loops with bf16 SDPA target execution.

| Method | Draft positions | Mean accepted draft tokens | Mean verification advance | Decode tok/s | Wall tok/s |
|---|---:|---:|---:|---:|---:|
| DFlash | 15 | 4.047 | 5.047 | 120.70 | 118.30 |
| Domino | 16 | 5.052 | 6.052 | 131.67 | 128.57 |

On paired prompts, Domino gains 1.005 accepted draft tokens (24.83%); the 95%
cluster-bootstrap intervals are [0.806, 1.242] tokens and [20.42%, 29.70%].
After truncating Domino to the same 15-position horizon, its EAL is 4.958, so
the matched-horizon gain remains 0.911 tokens (22.50%). Its extra sixteenth
position contributes only 0.094 tokens per round and does not explain the main
gain. Wall throughput improves by 8.68% in this harness.

| Domain | DFlash EAL | Domino EAL | Absolute gain | Relative gain |
|---|---:|---:|---:|---:|
| Math | 5.295 | 7.483 | +2.188 | +41.33% |
| Code | 5.012 | 5.701 | +0.689 | +13.75% |
| Chat | 2.587 | 3.216 | +0.629 | +24.33% |

The baseline confirms that Domino is strong and that beating it requires more
than a small average calibration gain. The largest opportunity is math; chat
has the lowest absolute EAL and remains the hardest domain.

### Correctness caveat and next decision

Strict token identity against a separate one-token HF greedy run is not yet
satisfactory: 40/96 DFlash samples and 38/96 Domino samples are bitwise exact.
There are 31 samples exact for both and 49 inexact for both; 56/96 have the same
first mismatch index (including the jointly exact samples). This overlap is
consistent with bf16 SDPA shape-dependent near-tie effects, but consistency is
not proof of exact target equivalence. These numbers are valid diagnostics for
candidate headroom and eager performance, but the paper cannot claim a
lossless system until an eager-attention or fixed-shape reference run closes
this gap.

The next sequence is fixed: (1) target logit/cache/shape equivalence diagnosis,
(2) same-anchor Gate 1b, and only then (3) a development learnability probe.
The proposed model is the absorbing-OTHER global prefix-CRF; the old locally
normalized head remains a baseline. Because DFlash hidden states are already
bidirectional, causal-vs-bidirectional mixer is not a clean causality test and
is no longer the primary gate.

## 2026-07-22 — Method/protocol correction (not an experiment)

- Added an absorbing-OTHER variable-length prefix CRF whose zero residual
  exactly recovers DFlash candidate and outside probabilities.
- Kept the locally normalized Markov model and candidate-only CRF as controls.
- Updated the trainer to use validation-only checkpoint selection and one final
  test evaluation, with mandatory `evidence_tier` metadata.
- Added Gate 0 target-equivalence and Gate 1b same-anchor scripts. Both refuse
  output overwrite and record input/config/script hashes; anchor reconstruction
  mismatch is a hard failure.
- CPU tensor suite: 20 tests passed, including brute-force partition, censored
  NLL, finite gradients, normalization, and exact base recovery.

This entry records implementation state only. It adds no empirical SPH result.
