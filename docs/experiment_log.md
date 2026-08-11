# Experiment log

This file records conclusions, not just commands. Smoke-scale results are kept
separate from evidence that may appear in a paper.

## Evidence registry (authoritative)

The user's pre-existing experiment history contained exactly two real research
runs. Every other pre-existing job was environment/plumbing smoke:

| Job | Tier | What it can support |
|---:|---|---|
| `10022338` | real, Gate 1a | Candidate availability and oracle upper bound on canonical anchors |
| `10022436` | real, development baseline | DFlash/Domino eager EAL and timing diagnosis; not final paper timing |

Jobs `10022278`, `10022310`, `10022330`, `10022343`, `10022412`, and
`10022468` are **non-evidence environment/plumbing smoke runs**. Their records
remain below only for debugging provenance. They must not enter a paper table,
an abstract claim, a gate decision, or a comparison with a trained method.

After that evidence boundary was frozen, the following new, controlled
follow-up jobs were run. They are development diagnostics or gates, not formal
paper evidence:

| Job | Tier | Outcome |
|---:|---|---|
| `10034918` | development diagnostic | Gate 0 passed; all cross-shape top-1 differences were explained by measured bf16 error/ties |
| `10034919` | development gate | Same-anchor Gate 1b passed on all 768 stored anchors |
| `10035142` | development probe | NLL-only head probe failed its predeclared EAL gate |
| `10035188` | development probe | NLL + survival-auxiliary probe also failed |
| `10035245` | development diagnostic | Audited nine checkpoints across train/select/test; established suffix-heavy changes and poor generalization |

Jobs `10035297` and `10035299` are new protocol-v2 plumbing smoke runs only.
Canceled, failed, or provenance-superseded attempts are never substituted for
the authoritative jobs above.

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
- CPU tensor suite: 27 tests passed, including brute-force partition, censored
  NLL, finite gradients, normalization, and exact base recovery.

This entry records implementation state only. It adds no empirical SPH result.

## 2026-07-22 — Gate 0 target numerical equivalence (`10034918`)

- Artifact: `artifacts/diagnostics/target_equivalence_10034918.json`
- SHA256: `fc1f8dddd73cc0d9f6c175a78d04ad17c859f5e17eb14d741e99868c53d16553`
- Frozen project commit: `289785a39d24f7cd89462acd72b774e7bbdc4934`
- Data: 6 prompts, 32 teacher-forced tokens per prompt, 192 predictions per
  comparison.

Both eager and SDPA same-shape cached replays were exactly reproducible
(192/192 top-1 matches and zero logit error). Across cached-single,
cached-block, full-prefix, and eager/SDPA shapes there were ten aggregate top-1
differences; every one was explained by the measured top-2/logit error
envelope, with zero unexplained disagreements. Maximum full-logit absolute
error was 1.0 in bf16.

**Gate 0 passes for logical verifier correctness.** This does not justify a
claim that different bf16 kernel shapes are bitwise token-identical. The paper
must distinguish mathematical losslessness (accepted draft tokens are checked)
from numerical near-tie behavior across kernels.

## 2026-07-22 — Same-anchor Gate 1b (`10034919`)

- Artifact: `artifacts/analysis/gate1b_same_anchor_10034919.json`
- SHA256: `cdbe38c5a10793e43589d07ca865b923ad7863cc6acba7d58a0e657a9e25f767`
- Frozen project commit: `4383e2e7e4da09af4ea83e8ecb2dea80bf8fb3c5`
- Data: all 96 prompts and 768 anchors from `10022338`; every reconstructed
  anchor and gold block matched exactly.

| Same 15-position horizon | Mean accepted draft tokens | Full-horizon acceptance |
|---|---:|---:|
| Pure DFlash top-1 | 6.030 | 12.24% |
| Domino checkpoint backbone top-1 | 6.264 | 13.02% |
| Domino on-policy GRU | 7.323 | 23.05% |
| DFlash oracle K8 | 10.143 | 38.80% |
| DFlash oracle K16 | 11.178 | 48.57% |

The paired K16-oracle minus Domino gain was +3.855 tokens with a prompt-cluster
95% bootstrap CI of `[3.449, 4.272]`, far above the predeclared 0.732-token
threshold. The gains were positive in chat (+3.602), code (+4.836), and math
(+3.129). Domino itself gained +1.293 over pure DFlash, including +1.059 from
its on-policy GRU relative to its jointly trained backbone.

**Gate 1b and therefore Gate 1 pass.** A large learnable candidate-space gap
remains beyond a strong Domino proposal on identical contexts. This is still an
oracle availability result, not evidence that SPH can identify those paths.

## 2026-07-22 — Small-data SPH probes (`10035142`, `10035188`)

Both jobs used the old `probe_v1` collection: only 75/9/12 prompts in
train/validation/test, with benchmark prompts rather than a proper training
corpus. They used a 327,713-parameter no-mixer rank-32 head, K16, three seeds,
validation-only epoch selection, and a single final test evaluation.

### NLL-only (`10035142`)

- Summary SHA256: `4a93e7a4058abd653c3f9ef815f029f12ac571277b7b930c2897d4a0b943ea41`
- Base test EAL: 6.469.
- Absorbing-CRF global-survival mean: 6.420, or -0.0486 versus base.
- Global-survival minus global-MAP mean: -0.0208.

### NLL + 0.1 survival auxiliary (`10035188`)

- Summary SHA256: `6789d4b9a4c4e7fe3a8e8234dae24bc626355f457c095105a99e8d4e69819a7b`
- Global-survival mean: 6.427, or -0.0417 versus base.
- Global-survival minus global-MAP mean: -0.0035.
- Global survival changed the global-MAP path on only 1.39% of test blocks and
  never changed the first token.

Both predeclared development gates fail. The auxiliary objective does not
rescue the small probe. These are informative negative development results,
not a formal method rejection: the twelve-prompt test has now been observed and
must not be used for any further hyperparameter selection.

## 2026-07-22 — Selected-checkpoint audit (`10035245`)

- Artifact: `artifacts/analysis/head_checkpoint_audit_10035245.json`
- SHA256: `ca1780e3d21698f91e7090ba27edf795e24db6f9461265a47318152cc68ea096`
- Nine checkpoints: six NLL-only (local/global) and three survival-auxiliary.

For the absorbing-CRF NLL-only head, mean global-survival minus base was +0.108
on train, +0.079 on validation, and -0.049 on test. With the survival auxiliary
it was +0.123, +0.088, and -0.042. The selected heads changed 46%–68% of full
paths on held-out blocks but only about 1%–3% of first-token choices. Most
learned changes therefore occur late in the path and often after the realized
first mismatch, where they cannot increase accepted length.

Decision: stop tuning `probe_v1`. Do one preregistered scale-up on clean
training data, with separate checkpoint-selection and development-gate prompts.
If global survival still has no independent held-out gain, revise the scoring
representation/objective rather than expanding an unconstrained sweep.

## 2026-07-22 — Exact-context protocol v2 smoke (`10035297`, `10035299`)

The new collector stores exact token IDs before every anchor, hashes all target
and draft checkpoint files, hashes every atomic shard, and writes
`metadata.json` only after a complete collection. `10035297` collected 12
blocks with a clean commit; `10035299` verified every shard and used
`stored_exact_context` rather than regenerating target continuations. Both are
plumbing smoke, not research evidence.

This change was motivated by an earlier A40 smoke that correctly hard-failed
when regenerated target tokens differed from the A800 canonical path. Future
formal same-anchor comparisons must replay stored contexts and never depend on
cross-hardware regeneration.

## 2026-07-22 — Phase 3 protocol v3 and queued work

The frozen manifest protocol contains 2,000 training prompts, 150
`validation_select` prompts, 150 disjoint `validation_gate` prompts, and 600
reserved formal-test prompts. Splits are prompt-level and balanced by domain.
Training sources are GSM8K-train, CodeAlpaca-20K, and first-user ShareGPT turns;
the reserved test uses unseen GSM8K/MATH-500, HumanEval/MBPP, MT-Bench, and
disjoint ShareGPT heldout prompts. Exact normalized overlap and 8-gram Jaccard
overlap against the benchmark pool are removed.

- Development manifest SHA256:
  `e16374068e9c8904214fbf282b4adb6187a0b099db5c37e79660fc46a2801d01`
- Reserved-test manifest SHA256:
  `ae25467fbb52b7091c8d9a5f98776b11ccf76e87e781850b5638734548a53bb4`
- Metadata SHA256:
  `5f9d2057a201ee44b32c630bde5a4c3ea72f7e17e90c4abc820e65b4cd1df275`
- Manifest build commit: `88b069ab0afd453425d6c2d46bd4baddfe8acead`
- A800 canonical collection: `10035436` (queued at time of writing).
- Dependent three-seed training: `10035437`; it starts only after collection
  completes successfully.

The formal test is registered as `reserved_unobserved`; it is neither collected
nor evaluated during this development gate.

## 2026-07-24 — Phase 3 failure analysis and GCLS job chain

The completed canonical collection `10035436` contains 18,253 blocks from
1,987 training prompts.  The six `10035437` runs all used the 327,713-parameter
`no_mixer` rank-32 head; none passed candidate sets from other draft positions
to the neural scorer.  K16 oracle EAL on `validation_gate` was 9.916 versus a
5.117 DFlash baseline, but the best six-run learned mean was only 5.183.
Improvements came from math while chat and code regressed, and most changed
suffixes were already unreachable after DFlash's first error.  See
`docs/phase3_failure_analysis.md` for the full diagnosis and preregistered
stopping rules.

A new 10,813,957-parameter candidate-lattice selector now has strictly matched
`local` and `global` attention scopes.  It consumes every one of the 15 x 16
candidate nodes, uses candidate-only listwise supervision with detached
reach-times-continuation weights, predicts top-K coverage/base correctness, and
tunes a KEEP_BASE threshold only on `validation_select`.

The A40 smoke `10056892` completed successfully in 40 seconds.  It exercised
the real 1.5 GiB canonical collection, frozen Qwen3-4B embeddings, bfloat16
forward/backward, dynamic-programming decoders, and metrics serialization.
Its two epochs and 64 reused blocks are plumbing evidence only.

Submitted hard-gated chain:

- `10056893`: 512-block memorization for both matched local and global scopes;
  it exits nonzero unless candidate accuracy is at least 95%, non-top1
  accuracy and first-miss repair are at least 90%, and at least 60% of the
  K16 oracle gap is recovered.
- `10056894_[0-5%3]`: local/global x three seeds on the prompt-disjoint Phase 3
  training and validation splits; it is released only after `10056893`
  succeeds.
- `10056895`: CPU aggregation after all six array tasks.  The preregistered
  development gate requires global KEEP_BASE gains of at least +0.20 versus
  DFlash and +0.15 versus matched local, no more than 0.1 percentage-point
  first-token loss, and no domain worse than -0.05 accepted tokens.

The reserved formal test remains uncollected and unobserved.

### 2026-07-24 follow-up

After auditing the first implementation, jobs `10056894` and `10056895` were
cancelled before allocation (zero elapsed time).  The initial loss used a
teacher-forced pairwise row and only an approximation to D-PACE, so those six
full-data runs would not cleanly isolate the value of global candidate-lattice
information.  Capacity probe `10056893` remains queued as an implementation
diagnostic.  Full development training will be resubmitted only after the
unary K+1/OTHER objective, exact smoothed D-PACE coefficients, and
local/global/shuffled controls are frozen.

## 2026-07-30 — GCLS-v1 data-scaling curve (`10099618`)

This experiment used the selected 433,772-parameter axial-global direct
selector (dimension 64, one block, K16, exact D-PACE).  The four hash-ranked
training subsets are strictly nested.  Epoch counts were chosen so that every
run received approximately 5,000 optimizer updates; this separates prompt
diversity from simply training longer.  All four array tasks completed with
exit code zero, and the recorded trainer/head hashes were unchanged from start
to finish.  The sealed `validation_gate` was not evaluated.

The confidence rule is a single validation-selected margin for deciding
whether to use the head's alternative candidate or retain DFlash rank 1.  It
was constrained to have no negative validation-domain EAL delta and at most a
0.001 first-token accuracy drop.  Therefore the calibrated column is
development evidence, not a held-out estimate.

| Train prompts | Blocks | Updates | Selected epoch | Raw EAL delta | Calibrated EAL delta | Oracle gap recovered |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 2,000 | 5,040 | 0 | +0.0000 | +0.0000 | 0.00% |
| 500 | 3,997 | 5,000 | 2 | +0.0049 | +0.0094 | 0.20% |
| 1,000 | 7,997 | 5,000 | 2 | +0.0372 | +0.0491 | 1.06% |
| 1,987 | 15,886 | 4,970 | 5 | +0.0763 | +0.09925 | 2.15% |

At 1,987 prompts the calibrated block-mean deltas were positive in chat
(+0.0339), code (+0.0612), and math (+0.1975); first-token accuracy improved
by 0.00255.  The calibrated policy improved 67 blocks and harmed 23.  With
250 prompts the training objective continued falling while held-out EAL
collapsed, and checkpoint selection correctly retained the exact DFlash
identity at epoch zero.

The monotone curve is evidence that training-prompt diversity is a real
bottleneck and that extra epochs on a small set cannot replace it.  It is not
evidence of a large practical win: even the full development set recovers only
2.15% of the K16 oracle gap.  Before collecting substantially more data or
opening the sealed gate, the next required experiment is matched
local/causal/global training over three seeds, followed by prompt-cluster
paired analysis and a controlled other-position context replacement test.

## 2026-07-30 — GCLS-v1 matched-scope seeds (`10099636`, `10099647`)

Nine runs used identical data, D-PACE objective, 433,772-parameter axial head,
optimizer budget, checkpoint rule, and validation-only safety calibration.
Only the visibility of other draft positions changed.  All source, data, and
configuration hashes matched, all jobs exited zero, and the sealed
`validation_gate` remained unobserved.

| Scope | Seeds | Mean calibrated EAL delta versus DFlash |
|---|---:|---:|
| local | 3 | +0.05786 |
| causal | 3 | +0.06013 |
| global | 3 | +0.07317 |

The individual global deltas were +0.09925, +0.06353, and +0.05673.  The
paired global-minus-local mean was only +0.01531 tokens; its prompt-cluster
bootstrap 95% interval was `[-0.00425, +0.03600]`.  Global-minus-causal was
+0.01304 with interval `[-0.01672, +0.04223]`.  Thus reranking itself is
repeatably better than DFlash rank 1 on this development split, but the
incremental value of bidirectional cross-position information is not yet
statistically resolved and the preregistered +0.15 global-minus-local gate
fails.  This result does not reject the global-information idea because the
preceding learning curve is still rising sharply at the 1,987-prompt boundary.

Aggregated artifact:
`artifacts/analysis/gcls_v1_scope_seeds_10099636_10099647.json`.

## 2026-07-30 — Open-PerfectBlend 100k scale-up

Domino reports training its lightweight head on 1.42 million
Open-PerfectBlend samples.  Our 1,987-prompt Phase 3 training split is about
714 times smaller, so it is not an adequate basis for claiming that the
selector has learned its task.  The first controlled scale-up uses 100,000
target-on-policy prompts, balanced as 33,334 math, 33,333 code, and 33,333
chat prompts.  It will support nested 10k/25k/50k/100k learning curves before
deciding whether to collect 400k or the full 1.42M.

The accepted v2 manifest has 100,000 unique normalized prompts and is split
exactly into eight 12,500-prompt shards.  It excludes exact normalized overlap
with every existing development/reserved manifest and excludes 8-gram Jaccard
overlap at or above 0.5.  An independent post-serialization audit recomputed
all prompt hashes, source/domain labels, overlap, file hashes, and the exact
partition; it passed with maximum excluded-set Jaccard 0.495868.  The earlier
v1 manifest is retained only as a rejected provenance artifact because an
Open-PerfectBlend source-name mismatch initially assigned Orca-Math to chat;
no job consumed v1.

- Manifest SHA256:
  `b05087a56e8e717605415026421f7bae23092eb7cb9509361a36932f80260e3a`
- Metadata SHA256:
  `dd19a84aabfc5cf69a04321ea9432dd1be3cf7d68f275cd4eb0c6035af0556c5`
- Builder SHA256:
  `b744c10e32227d4ee26053b2e1cb00d317b32567740ac1b26b2f0ab47d309f05`
- Independent auditor SHA256:
  `06866529f652940121c305db667244d1884ed8cfb4d7febe556f1dd6fc33e1f4`
- Initial four-anchor collection `10099692_[0-7]` was cancelled before
  allocation (zero elapsed time) after the final scale audit found it below
  the preregistered 8–16 anchor range.
- Eight-anchor collection `10099734_[0-7]` was also cancelled before
  allocation (zero elapsed time) after the real-data smoke established that a
  24-hour request was safely sufficient and would backfill sooner.
- Active eight-A800, eight-anchor canonical collection: `10099770_[0-7]`.

Each collection task processes 12,500 prompts using eight anchors per prompt,
128 target continuation tokens, DFlash top-64 candidates, and exact stored
contexts.  Expected total scale is approximately 800,000 blocks and 12
million position-level labels.  No generated continuation from
Open-PerfectBlend is reused: continuations are regenerated by the frozen
Qwen3-4B target to match the deployed on-policy distribution.

Real-data A40 plumbing/throughput smoke `10099764` completed 32/32 prompts and
256/256 blocks with exit code zero.  Collection took 128.57 seconds
(4.02 seconds per prompt), peak memory was 8.85 GiB, and an independent loader
confirmed exact compatibility with the existing A800 validation collection.
The slower-A40 extrapolation is 13.95 hours per 12,500-prompt shard, leaving a
large margin under the formal job's 24-hour request.

Dependent development training is already queued but cannot run unless all
eight collection tasks exit successfully:

- `10099772_[0-7]`: d64 axial local/global pairs at nested 10k, 25k, 50k,
  and all-available prompt scales.  Batch 64 and 30/12/6/3 epochs keep every
  task at approximately 37,500 optimizer updates.
- `10099779_[0-1]`: full-data d128, two-layer local/global capacity pair.
  This guards against falsely blaming the hypothesis if the small d64 head
  becomes the bottleneck after data scaling.

All jobs use exact D-PACE as the unchanged primary objective and tune the
base-retention margin only on `validation_select`.  The sealed gate remains
skipped.  Loss alternatives will be compared only after this curve identifies
whether the limiting regime is data, capacity, or objective; changing all
three simultaneously would make a positive or negative result uninterpretable.

## 2026-08-03 — Open-PerfectBlend 100k result audit

All eight collection tasks and all ten preregistered training tasks completed
with exit code zero.  The collection contains 99,356 valid prompts and 793,989
blocks: 260,759 chat, 266,566 code, and 266,664 math blocks.  The remaining
644 manifest prompts ended before one complete 16-token draft block and were
therefore skipped by the canonical collector.  Every part has a complete
metadata/shard manifest and passed integrity verification.

The d64 axial runs used strictly nested prompt subsets and approximately equal
optimizer-step budgets.  Local and global members at every scale have exactly
matching train-prompt hashes.  Checkpoint selection and the single
base-retention margin used only the same 147-prompt `validation_select` set;
the sealed gate was not evaluated.

| Train prompts | Global raw delta | Local raw delta | Global calibrated delta | Local calibrated delta | Calibrated global-local |
|---:|---:|---:|---:|---:|---:|
| 10,000 | +0.0100 | +0.0034 | +0.0247 | +0.0221 | +0.0026 |
| 25,000 | +0.1078 | +0.0363 | +0.1078 | +0.0491 | +0.0587 |
| 50,000 | +0.2193 | +0.0304 | +0.2218 | +0.0306 | +0.1912 |
| 99,356 | +0.2425 | +0.0686 | +0.2561 | +0.0519 | +0.2042 |

At full scale, the prompt-cluster bootstrap estimate for **raw** global minus
local is +0.17383 with development 95% interval `[+0.10544, +0.24356]`.
This result does not depend on post-hoc base retention.  The corresponding
calibrated difference is +0.20420 with interval
`[+0.13448, +0.27405]`, exceeding the preregistered +0.15 development
threshold.  Raw global minus DFlash is +0.24247
`[+0.16764, +0.32070]`; calibrated global minus DFlash is +0.25607
`[+0.18379, +0.33005]`.

The full-data raw global-local effect is positive separately in chat
(+0.12091, interval `[+0.04427, +0.20164]`), code (+0.26020,
`[+0.13265, +0.39796]`), and math (+0.14000,
`[+0.00750, +0.26250]`).  After calibration, global improves 150 blocks and
harms 52, increases first-token accuracy by 0.00681, and improves EAL versus
DFlash by +0.14621 chat, +0.25255 code, and +0.36500 math.

Mechanistically, full-data global reaches 10.24% train and 10.00% validation
non-top1 candidate accuracy, versus 4.63%/4.13% for matched local.  Its
first-miss repair rate is 16.26% on validation, versus 7.11% for local.  The
small train-validation gap argues against memorization: cross-position
evidence roughly doubles hard-token identification on held-out prompts.

The 50k-to-full calibrated increment is only +0.03426 with interval
`[-0.02867, +0.09803]`; thus prompt scaling is showing diminishing returns on
this development set.  The much larger d128 two-layer head underfits under the
same 37k-step budget: global is only +0.03377, and is -0.06633 below its local
control.  More parameters alone are not the current solution; d64 is the
selected architecture pending seed replication.

Artifact:
`artifacts/analysis/gcls_v1_open_perfectblend_100k_summary.json`, SHA256
`c69c9cb5d787127bc2fb635c90a17da94f069475e8c790508a99ca3077c75286`.
These bootstrap intervals are development-only
and conditional on validation-selected checkpoints/margins, not formal-test
confidence intervals.

Submitted follow-ups:

- `10123109_[0-6]`: complete three full-data seeds for local/causal/global.
- `10123112`: keep the selected global checkpoint fixed and mask its inference
  attention to causal or local scope, using the already-frozen margin.  This
  tests whether its gain actually requires cross-position pathways.
- `10123118_[0-7]`: d64/d128 x 6/9 epochs x local/global optimization-budget
  screen.  Both full-data architectures selected their final available epoch,
  so this isolates under-optimization from architecture capacity before any
  loss or representation is changed.
- `10123133`: evaluate the released Qwen3-4B Domino checkpoint on the exact
  same 147 `validation_select` prompts and 1,175 stored anchors as the global
  selector.  This supplies the missing direct method-level baseline; the
  +0.24247 result above is only an incremental gain over pure DFlash and does
  not establish superiority to Domino.

## 2026-08-04 — Full-data scope replication and optimization follow-up

All seven missing scope/seed jobs in `10123109_[0-6]`, the fixed-checkpoint
scope ablation `10123112`, and all eight optimization jobs in
`10123118_[0-7]` completed with exit code zero.  Every training run used all
99,356 available prompts and 793,989 blocks, and no run evaluated the sealed
`validation_gate`.

The matched d64, one-layer, three-epoch scope comparison is now complete over
three seeds:

| Scope | Mean raw delta vs DFlash | Mean calibrated delta vs DFlash |
|---|---:|---:|
| local | +0.06240 | +0.06009 |
| causal | +0.12848 | +0.14440 |
| global | +0.22153 | +0.23174 |

Global exceeded local in every seed by a mean +0.17165 calibrated EAL; the
prompt-cluster 95% development interval is `[+0.11099, +0.23413]`.  Global
also exceeded separately trained causal heads in every seed by +0.08734,
interval `[+0.02839, +0.15140]`.  The global head improved calibrated EAL over
DFlash in every seed (+0.25607, +0.22765, +0.21149) and was positive in all
three domains.  Thus the full-data development gate passes, and the evidence
for bidirectional cross-position information is no longer a single-seed
effect.

The fixed seed-0 global checkpoint reproduced its original EAL exactly under
its normal mask.  Replacing only its inference mask with causal reduced raw
EAL from 5.35447 to 5.06754; replacing it with local reduced raw EAL to
4.30916.  This is mechanistic evidence that the trained model actually uses
other-position pathways, including later draft positions.  Because masking a
trained network is an out-of-distribution intervention, the separately
trained three-seed comparison above remains the primary causal evidence.

Longer optimization removed the earlier apparent d128 failure.  The best
single development run was d64 global with nine available epochs (epoch 7
selected): raw/calibrated deltas +0.28499/+0.28584.  D128 global with six
epochs reached +0.27393/+0.28073.  The d64 nine-minus-six calibrated
difference was only +0.02636 with interval `[-0.03997, +0.09269]`, and d128
did not improve from six to nine epochs.  D128-minus-d64 intervals also cross
zero.  Therefore capacity and extra epochs are not presently resolved as
reliable gains; d64 remains the efficient default, while global-minus-local is
large and positive for every tested capacity/budget pair.

Artifacts:

- `artifacts/analysis/gcls_v1_open_perfectblend_scope_seeds_summary.json`
  (SHA256 `7769cc13b9492e169708c397b61974fc809c5394987ce9ed657866f23e00c05d`)
- `artifacts/analysis/gcls_v1_opb_scope_ablation_10123112.json`
  (SHA256 `58c6706150c2d4175ab233e01c3912a0d541f09702659374f0ec809e94548cad`)
- `artifacts/analysis/gcls_v1_open_perfectblend_optimization_summary.json`
  (SHA256 `a339078904e3ceb8108974a313ecd1b6e9d0cff1e36884862344a4188da6b7c3`)

The first same-validation Domino job `10123133` failed before model loading
because the 150-entry `validation_select` manifest includes three prompts
that produced no complete canonical block.  The comparator intentionally
retains strict failure by default.  It now has an explicit
`--allow-missing-canonical-samples` mode that records those IDs and evaluates
only the manifest/canonical intersection: the same 147 prompts and 1,175
anchors used by every head result above.  The repaired Domino job is
`10129478`; it was pending at the time of this entry.

## 2026-08-04 — Same-validation Domino comparison completed

The queued A800/A40 copies `10129478` and `10129766` were cancelled before
allocation after a debug A40 became available.  Final job `10129790` completed
with exit code zero in 61 seconds.  It replayed the exact stored contexts for
the same 147 prompts and 1,175 anchors used by GCLS.  The report explicitly
records the three requested manifest samples that have no canonical block.
No sample with a stored block was dropped.

The final comparator also fixes a reporting-only inconsistency caught during
audit: paired point estimates are now prompt-balanced, matching the
prompt-cluster bootstrap estimand; round-weighted estimates remain recorded
separately.  This changes the Domino-minus-DFlash point estimate from
+1.90043 to +1.90379 and does not affect any conclusion.

| Method on identical validation anchors | Prompt-balanced EAL |
|---|---:|
| Pure DFlash top-1 | 5.11200 |
| Global d64, mean of three 3-epoch seeds, raw | 5.33354 |
| Global d64, mean of three 3-epoch seeds, calibrated | 5.34374 |
| Best development run: global d64, 9 epochs, raw | 5.39699 |
| Best development run: global d64, 9 epochs, calibrated | 5.39784 |
| Released Domino checkpoint, backbone top-1 only | 5.93853 |
| Released Domino checkpoint, on-policy GRU | 7.01579 |
| Stored DFlash top-16 oracle | 9.72668 |

The three-seed global raw mean is -1.68226 behind Domino, with prompt-cluster
95% interval `[-2.04138, -1.34042]`.  Even the post-hoc best d64 nine-epoch
run is -1.61880 raw, interval `[-1.98639, -1.27466]`; calibration changes the
gap by less than 0.001.  Therefore the current GCLS implementation does not
match Domino.

The end-to-end gap has two separable components.  Domino's checkpoint
backbone top-1 is already +0.82653 above the stored pure-DFlash top-1, and its
on-policy GRU adds another +1.07726 over that backbone (interval
`[+0.91217, +1.24490]`).  The best current global head adds only +0.28499 raw
over its DFlash backbone.  Algebraically, the resulting Domino-minus-GCLS gap
is `0.82653 + 1.07726 - 0.28499 = 1.61880`.  Because Domino was trained on
approximately 1.42M samples whereas GCLS used 99,356 valid prompts, this
comparison measures released methods/checkpoints, not a controlled
architecture-only effect.  Still, both the stronger Domino backbone and the
much larger correction-head gain must be closed before a competitiveness
claim is supportable.

Final artifact:
`artifacts/analysis/domino_phase3_validation_select_10129790.json`, SHA256
`bb6518591af551c74a65e2f33bff52abde30095ff0d5aa93bd4f2b96c8c5c96a`.

## 2026-08-04 — Objective-support pivot and positive-only feature probe

The frozen three-cell prediction-conditioned reachable-support capacity
matrix ran as job `10132646`; fail-closed aggregate job `10132649` produced
`artifacts/training/gcls_v3_reach_capacity_10132646/reach_capacity_summary.json`.
Candidate-D-PACE control (`lambda=1`) passed every capacity check at 1.0 with
zero harm. Hard censoring (`lambda=0`) failed candidate accuracy (0.989313),
hard-candidate accuracy (0.940639), and oracle-gap recovery (0.949495). The
soft 0.1 cell passed four checks but failed hard-candidate accuracy at
0.949772. The binding all-cell gate is therefore scientific negative. No
OPB-25K run, threshold relaxation, or lambda rescue is authorized for this
route. Fresh result-to-claim review records `claim_supported=no`.

The separately preregistered positive-only frozen-feature probe then entered
its 512-block capacity stage. Flat compatibility D640/H10/L4 has exactly
27,482,160 trainable parameters and uses the established Candidate-D-PACE
alpha-0.5 control objective, not reachable censoring. Job `10132680` completed
1,920 steps in 168 seconds with 2.58 GiB peak allocated CUDA memory. It first
passed all five checks at epoch 60 and the selected epoch 103 reached 1.0
candidate accuracy, hard accuracy, repair, and oracle-gap recovery with zero
harm. Fail-closed summary job `10132681` passed.

This is only a same-subset function-class capacity witness. The matched 10K
held-out array `10132737` (compact axial D64 control and D640 probe) and
dependent summary `10132739` were submitted next. Its frozen positive-only
gate is D640 raw improvement over DFlash of at least +0.6 EAL or at least 15%
oracle-gap recovery. A negative stops the frozen probe without supporting an
information-ceiling claim; a positive only authorizes the preregistered 100K
diagnostic.

## 2026-08-05 — 10K throughput probe exposes repetition-overfit confound

Exact-config debug array `10132757` was launched to avoid the long A800 queue.
The compact D64 task completed all 37,470 updates in 11:43. Its selected epoch
4 achieved only +0.01421 raw prompt-balanced EAL over DFlash, 0.00308 oracle
gap recovery, 5.39% first-miss repair, and 3.23% harmed blocks. By epoch 30,
the training objective had fallen from 0.36443 to 0.27575 while validation
delta collapsed to -0.38763 and harm rose to 22.64%.

The D640 task required roughly two minutes per epoch on the debug A40, so it
could not complete the frozen 30 epochs within that partition's hard
30-minute limit. It was intentionally cancelled after epoch 9 rather than
presenting a truncated artifact as a gate result. The best observed partial
epoch was epoch 3 at +0.03681 raw EAL, 0.00798 oracle-gap recovery, and 5.45%
harm. By epoch 9 the training objective had improved from 0.37091 to 0.31250,
but held-out delta had reversed to -0.30175 and harm reached 18.38%. This is a
diagnostic trajectory only; R062/R063 have no scientific verdict.

The run also exposed a design flaw in the staged probe. OPB-10K for 30 epochs
uses 37,470 updates, while all 99,356 prompts for three epochs use 37,221.
The smaller gate therefore saves almost no model compute but makes 30 passes
over each prompt/block bundle instead of three and discards 90% of prompt
diversity. An adaptive, explicitly development-only amendment was frozen in
`refine-logs/feature-probe/FIXED_STEP_PROMPT_DIVERSITY_AMENDMENT.md` before
any full-data D640 launch. It retains the +0.6/15% positive-only threshold and
adds a matched full-data D64 control; a negative remains only an engineering
stop, never evidence that the frozen inputs contain no information.

Fresh experiment-bridge review initially blocked the amendment on walltime,
unpinned same-across-cell provenance, and non-deterministic domain-key
iteration. Before launch, the implementation moved to a four-hour fail-safe,
pinned the exact reviewed trainer/head/validation/target identities and all
eight train metadata hashes, sorted bootstrap support, corrected the
30-versus-3 repetition accounting, disclosed the historical `dpace` versus
new `candidate_dpace` distinction, and made all domain comparisons
prompt-balanced. Final review returned GO; the full suite passed 152 tests
plus three parameterized subtests.

Formal full-data array `10132819_[0-1]` and dependent fail-closed summary
`10132820` were submitted. All still-pending 10K backups (`10132737`,
`10132744`, `10132746`, `10132773`) and their old summary `10132739` were
cancelled before allocation, so no duplicate GPU work remains. A separate
30-minute debug task `10132856_1` runs the exact D640 full-data command only
to obtain an early prompt-diverse trajectory while the formal array waits;
unless it completes all three epochs, it is explicitly inadmissible for the
positive-only gate.

The debug task reached exactly one of three epochs before Slurm ended it at
the declared time limit. At epoch 1, raw prompt-balanced EAL was `5.13472`
versus DFlash `5.11200` (`+0.02272`), first-miss repair was `5.79%`, oracle-gap
recovery was `0.00492`, and 27/1,175 blocks were harmed (`2.30%`). This is an
early learning-curve point only: it neither passes nor fails the unchanged
`+0.6` / `15%` formal positive gate, and no selected-checkpoint artifact is
constructed from the truncated job. Formal jobs `10132819`/`10132820` remain
the binding evidence.

## 2026-08-05: First-Miss Action Selection pivot

Post-hoc decoding of the best historical D64 selector separated safety from
performance. Restricting the learned direct scores to one or a few edits could
reduce harm, but did not beat the unconstrained `+0.28499` raw EAL gain. In
contrast, the gold-aware one-edit action oracle—KEEP DFlash or repair exactly
its first miss—reached prompt-balanced EAL `6.64407` from base `5.11200`, a
`+1.53207` availability bound. This motivated FMAS: one 226-way block action
over KEEP plus 15 positions × 15 non-base ranks, trained by canonical oracle
action imitation and decoded with at most one edit.

ARIS research-refine required three review rounds. The final external score was
9.1/10 READY after separating `validation_select` selection from claim-grade
evidence, adding the exact Direct-one-edit causal control, narrowing CE to
canonical action imitation rather than an exact EAL-risk surrogate, and
freezing seed/formal-rollout estimands. The existing `validation_gate` is not
treated as fresh because Phase-3 already inspected it; the 600-prompt reserved
formal test remains unobserved and inaccessible.

Gate-0 implementation is isolated in new FMAS files and leaves the pinned
feature-probe trainer/head unchanged. The full CPU suite passed 167 tests plus
three parameterized subtests. The exact 512-block capacity manifest has file
SHA256 `d60613a0...67f`, subset SHA256 `1c4f911c...c355`, and contains 256 edit,
156 full-correct KEEP, and 100 out-of-K KEEP targets. Fresh experiment-bridge
review initially returned NO-GO for an extra checkpoint tie-break and missing
gradient/accounting tests; both were corrected, and re-review returned GO.
D64 capacity job `10133018` is the only authorized launch. It uses 5,120
updates and must jointly pass action accuracy `>=.97`, repairable-action recall
`>=.95`, single-edit oracle-gap recovery `>=.95`, and harm `<=.01`; otherwise
the 99K development run is forbidden.

Job `10133018` completed all 5,120 steps in 2:23 and passed Gate 1. The
CE-selected epoch 297 reached action accuracy `1.0`, repairable-action recall
`1.0`, single-edit oracle-gap recovery `1.0`, and harm `0.0`; all four frozen
checks passed. The model has 433,772 parameters. This same-subset result is a
capacity/optimization witness only, so its EAL gain is not treated as held-out
evidence. Under the frozen routing rule it authorizes the single seed-0 full
OPB-99,356 development run, but not seeds1/2 or formal-test access.

Before that development launch, fresh experiment-bridge review rejected the
first package because the mixed canonical shards entered `validation_gate`
records into memory before logical filtering. A hash-verified materializer
therefore produced a physical `validation_select`-only collection: 147 prompts,
1,175 blocks, and chat/code/math counts 383/392/400. The FMAS trainer now
rejects any external-training validation collection containing another split,
and the canonical data helper is pinned and snapshotted at run start/end.

A second review constructed a self-consistent wrong Direct checkpoint that the
initial one-edit evaluator accepted. The remediated evaluator locks all Direct
configuration fields except dynamic output, exact D64 Candidate-D-PACE
architecture/objective/data/budget, all eight train-part hashes, validation
cardinality, and Direct trainer/head start/end identities. It must also exactly
reproduce the reported DFlash and Direct-native summaries on the isolated copy
before reporting Direct-one-edit. Adversarial wrong-architecture, inactive
default, budget, data, provenance, cardinality, and output-path mutations now
fail. The full suite passed 177 tests plus three parameterized subtests, and
fresh re-review returned GO only for seed-0 checkpoint production.

FMAS development job `10133114` was submitted on the exact 99,356-prompt,
793,989-block, three-epoch, 37,221-step contract. Exact Direct-one-edit job
`10133115` is dependency-gated on matched Direct task `10132819_0`. Gate 2
remains scientifically open until all three Direct-native/Direct-one-edit/FMAS
isolated-validation artifacts exist and satisfy the frozen comparisons.

Job `10133114` completed all 37,221 updates in 19:23 with matching source/data
hashes and empty stderr. The result is a binding negative. CE fell from 4.12094
at identity to 2.53864 and held-out action accuracy rose from 16.26% to 33.19%,
yet epochs 1/2/3 produced EAL deltas `-0.50486/-0.48567/-0.42396` and harmed
`34.98%/33.28%/31.66%` of blocks. The frozen raw-EAL checkpoint rule therefore
retained epoch 0 at exactly DFlash EAL and zero harm. Its zero gain fails the
absolute `>+0.28499` gate, so Direct controls cannot rescue the route; pending
one-edit evaluator `10133115` was cancelled before allocation.

Fresh result-to-claim review independently reconstructed the action geometry.
Of 264,375 possible edits, only 984 (`0.3722%`) improve EAL, 90,120 (`34.09%`)
harm it at an average cost of 5.304 tokens, and the rest are neutral. Flat CE
does not price those error costs: it improved its own held-out surrogate while
making realized utility sharply worse. This closes canonical flat action CE,
not first-miss intervention generally; the 512-block capacity pass forbids an
architecture-capacity or frozen-information ceiling claim. Seeds1/2 and formal
data remain closed. The next allowed route must train signed action value/risk
directly and restart at CPU semantics plus a separately frozen capacity gate.

## 2026-08-05 — Signed-value capacity gate fails despite low average RMSE

The new signed action-value selector (SAVS) was implemented with exact-zero
residual initialization, strict-positive edit deployment, and dense
one-edit-prefix advantages. CPU semantics and the full test suite passed, and
fresh experiment-bridge review authorized exactly one D64/H4/L1 same-subset
capacity job.

Job `10133339` completed its exact 320 epochs / 5,120 updates in 156.32 s on
an A40. The selected epoch 307 uniquely minimized action-uniform MSE. The job
then exited 1 by design because the conjunctive scientific gate failed; its
metrics/checkpoint were complete and finite, and stderr contained only a
nonfatal tensor-detach warning.

| Gate item | Observed | Required | Result |
|---|---:|---:|---|
| all-action RMSE | 0.006909 | <=0.02 | pass |
| beneficial sign recall | 0.78125 | >=0.99 | fail |
| harmful nonpositive recall | 1.0 | >=0.99 | pass |
| one-edit oracle-gap recovery | 0.44546 | >=0.95 | fail |
| harmed fraction | 0.0 | <=0.01 | pass |

No selection-rule accident explains the result: across all epochs, the best
beneficial recall was `0.79297` and best gap recovery `0.47441`. The model
raised same-subset EAL from `7.44118` to `7.84895` but chose only 83 beneficial
edits alongside 272 neutral edits.

Diagnostics expose why average error is misleading. The epoch-zero harmful
output-gradient norm is `0.430526`, versus `0.0002835` for beneficial actions
(`1,518.6x`), and its direction almost exactly equals the total gradient.
Beneficial errors nevertheless make up `85.06%` of endpoint SSE. Fresh
result-to-claim review therefore records a high-confidence FAIL-CLOSE and a
positive-gradient-starvation-consistent diagnosis, with the explicit caveat
that a single run does not prove unique causality.

Full-data SAVS, continuation, D640, post-hoc thresholds, class-weight rescue,
and extra seeds are prohibited. The next permissible experiment must be a
newly refined and preregistered mechanism aimed at the sparse-beneficial /
max-selection mismatch, with its own CPU semantics, capacity gate, and fresh
review.

## 2026-08-05 — Tie-safe CAMRS passes the capacity gate

The new CAMRS route uses a deterministic non-oracle cost-augmented ReLU hinge
whose pointwise value upper-bounds decoded one-edit regret. Independent method
review scored the refined proposal 9.2/10 READY, and an adversarial code review
first rejected then hardened the capacity gate against infinities, NaNs,
impossible counts, and cardinality drift before authorizing one job.

Job `10133549` completed all 5,120 updates in 3:28 on an A40 with Slurm exit
`0:0` and empty stderr. The frozen earliest-minimum rule selected epoch 98,
the first exact zero-hinge epoch. Its selected metrics are:

| Gate item | Observed | Required |
|---|---:|---:|
| mean block hinge | 0.0 | <=0.0030078125 |
| beneficial positive | 256/256 | >=254/256 |
| utility-optimal selected | 256/256 | >=244/256 |
| harmful nonpositive | 57,629/57,765 | >=0.99 |
| prompt-balanced oracle-gap recovery | 1.0 | >=0.95 |
| selected harmful | 0/512 | <=5/512 |
| no-benefit false edits | 0/256 | <=2/256 |

CAMRS prompt-balanced EAL equals the single-edit oracle at
`8.356572258533044` on this same subset, versus base `7.4411764705882355`.
Independent result-to-claim review replayed all examples and checkpoint
selection, matched all source and artifact hashes, and returned
`claim_supported=yes`, `PASS-ADVANCE`, confidence high. This is a capacity and
optimization witness only; it is not held-out evidence.

The binding next step is one seed-0, 37,221-update full-OPB development run on
the physically isolated 147-prompt / 1,175-block validation collection. Before
that launch, exact external Direct-native and Direct-one-edit artifacts must be
hash-frozen and a fresh development code review must return GO. To avoid the
formal array's long priority wait without changing its state, the already
reviewed exact D64 task was resubmitted to debug as job `10133585`, with
one-edit evaluator `10133586` dependency-gated after it.

## 2026-08-05 — CAMRS controls frozen and unique development run launched

The expedited matched Direct task `10133585_0` completed `0:0` in 22:13 and
selected epoch 2. On the physically isolated 1,175-block / 147-prompt
`validation_select` copy, its raw prompt-balanced EAL is `5.334669582118561`
versus DFlash `5.112001943634597`. Dependent one-edit evaluator `10133586`
completed `0:0` in 11 seconds and exactly reproduced both summaries before
reporting one-edit EAL `5.212099125364432`. The one-edit decoder changed 896
blocks but improved only 78 and harmed 29, independently confirming the
over-editing / weak-realized-repair failure targeted by CAMRS.

All producer, checkpoint, evaluator, validation-data, and source identities
were frozen before any CAMRS full-data outcome in
`refine-logs/first-miss-max-regret/PRELAUNCH_CONTROL_FREEZE.md`. Fresh
experiment-bridge review initially returned NO-GO because the inherited
signed-value head was absent from the declared runtime closure. The trainer,
wrapper, and AST import-closure regression were repaired; 228 tests plus three
subtests passed, and focused re-review returned a bounded GO.

Exactly one D64/H4/L1 seed-0 CAMRS development job, `10133649`, was then
submitted with the frozen three-epoch / 37,221-update protocol. It must beat
DFlash by strictly more than `0.28499`, beat both Direct controls by at least
`0.05`, harm at most 5% of blocks, and trail Direct-native first-token count by
at most one block. Metrics are written before a scientific-failure exit. No
repeat, extra seed, D640 run, calibration, threshold change, continuation, or
formal-test access is authorized.

## 2026-08-05 — CAMRS full-data route fails closed; binary abstention remains feasible

Job `10133649` completed the exact three-epoch / 37,221-update development
contract in 19:57 on an A40. It wrote complete metrics and checkpoint artifacts
with empty stderr, then exited `1:0` by design because the conjunctive
scientific gate failed. The frozen raw-EAL-first selector retained epoch 0,
which exactly reproduces DFlash at EAL `5.112001943634597` and zero harm. Its
deltas are `0` versus DFlash, `-0.22266763848396387` versus Direct-native, and
`-0.10009718172983462` versus Direct-one-edit; it also trails Direct-native by
seven first-token-correct blocks.

Every trained checkpoint is worse. Epochs 1/2/3 reduce validation hinge from
`0.4430` to `0.2012/0.2004/0.1976`, but produce EAL deltas
`-0.07483/-0.10884/-0.12804` and harm `5.70%/7.32%/9.19%`. Harmful positive
scores grow to `1326/90120` at epoch 3, while beneficial positive recall is
only `101/984`; the max over 225 edits turns this thin false-positive tail into
108 harmed blocks. Fresh result-to-claim review independently replayed all
1,175 examples, all source/control/data hashes, checkpoint selection, and
Slurm semantics, returning `claim_supported=no`, `FAIL-CLOSE/PIVOT`, high
confidence. CAMRS full-data, repeats, continuation, D640, thresholds, weights,
and formal-test evaluation are closed.

A read-only control postmortem found a narrower feasible pivot. Direct-native
improves 141 blocks, is neutral on 972, and harms 62. An oracle restricted to
two actions—KEEP DFlash or APPLY the frozen Direct-native path—has EAL
`5.430758017492711` and zero harm, exceeding the existing absolute target
`5.396991943634597`. A three-way oracle adding Direct-one-edit reaches only
`5.438411078717201`. This does not establish a learned method, but motivates a
new binary Direct safety gate with explicit abstention, which must restart at
research refinement and preregistration before code or GPU work.

## 2026-08-05 — PROS-Gate split job fails closed on a combined-manifest bug

The first reviewed identity-only PROS-Gate split job, `10135740`, passed every
source and input hash preflight and then exited `1:0` before publishing an
artifact. The materializer reported that all 1,987 Phase-3 prompts overlapped
validation. Diagnosis showed that `phase3_development_v3.jsonl` combines 2,000
`train`, 150 `validation_gate`, and 150 `validation_select` rows; the failed
loader incorrectly excluded the whole file even though the frozen proposal
names only the two validation splits.

The repaired producer and independent auditor now freeze and verify, before
use, each file's exact path/size/hash, selected row splits, and complete
per-split row census. Their provenance and semantic hashes include the filter
and census, and the downstream receipt verifier pins those enriched identities.
A real identity-only replay keeps all 1,987 Phase-3 prompts / 15,886 blocks and
finds zero overlap with the 100,000 producer-train, 300 selected-validation, or
600 reserved-test prompts. The full CPU suite passes 302 tests plus three
subtests. This repairs implementation of the existing protocol; it changes no
scientific split, model, objective, threshold, or evaluation rule. One retry
remains blocked until a fresh failure-rescue review returns GO.

## 2026-08-11 — PARC-16 formal full-data path queued

The active route is now PARC-16, a 2,438,400-parameter full-action global
noncausal head that consumes all 16 DFlash prediction positions in one call and
emits exactly one 16-token chain. It contains no GRU/token feedback, serial
target seed, iteration, beam, tree, trie, forest, or multipath verification.

The implementation now includes the raw17-to-full16 non-shift pure-DFlash
geometry, strict 270K reserve split, full16 trace collection, 180K-step joint
DFlash+PARC training, validation-only checkpoint selection, and exact resumable
state. Eighteen focused tests and static checks passed; fresh independent review
returned M1 GO and M2 GO without authorizing a GPU smoke or capacity stage.

Formal materialization array `10169014` contains 16 one-A800 tasks and must
produce exactly 90K train plus 5K validation prompts, with eight full16 anchors
per prompt. Formal training job `10169018` is dependency-gated on the entire
array and will run the single 180K-step main recipe. As of this update both jobs
remain pending for A800 priority/dependency. There is no formal validation or
held-out EAL yet; the older same-set capacity EAL 9.5254 is explicitly not
effect evidence.
