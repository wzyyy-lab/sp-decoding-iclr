# R079 Continuous-Numeric Diagnostic Contract

**Date:** 2026-08-05  
**Status:** implementation complete; fresh submission review pending  
**Authorization:** no CUDA job, outcome retry, capacity, training, falsifier,
validation, reserved, or formal evaluation is authorized by this document

## Why the outcomes stage remains stopped

Job `10136583_[0-1]` passed all source, receipt, model, and data preflights but
both tasks failed before atomic publication.  The first retained-mass feature
reconstructed on CPU differed from its CUDA value by
`1.9073486328125e-06`, beyond the inherited absolute `1e-6` check.  No bundle,
temporary directory, or downstream artifact was created.  A one-field
tolerance increase and another outcome retry were rejected.

## Label-blind boundary

Canonical shards are monolithic PyTorch pickles, so the trusted
`CanonicalBlockDataset` loader necessarily deserializes `gold_ids` into process
memory.  The diagnostic makes only the narrower, auditable noninterference
claim: its code never indexes, copies, stacks, hashes, compares, branches on,
passes onward, logs, or computes from that field.  An explicit extractor reads
only `sample_id`, hidden states, candidate IDs/logits, base logsumexp, and the
anchor ID, then hands an identifier-free typed batch to all numerical code.
Static AST tests bind that allowlist and prohibit outcome, accepted-length,
gain, capacity, audit, and evaluation calls.

The frozen split manifest is used only to select the already assigned `fit`
and `checkpoint` prompt identities.  The diagnostic expects exactly 12,686
fit blocks and 1,600 checkpoint blocks.  It never selects falsifier identities
and has no validation, reserved, formal, or artifact-output argument.

## Frozen numeric policy

- Discrete values and direct copies remain bitwise exact.
- Rank and position allow only the CPU float32 value or one adjacent interior
  float32; endpoints 0 and 1 remain exact and values remain in `[0,1]`.
- Add/subtract relations use an exact float64 reference over persisted float32
  operands, a source-scale ULP, two ULPs of half-width, and a `2^-14` cap.
- Entropy uses an independent float64 reconstruction, range `[0,1]`, and fixed
  absolute envelope `2^-17`.
- Retained mass uses `E_lse = 8*ulp32(scale) + 2^-20`, interval propagation
  through `tanh`, two outward float32 neighbors, and `2^-20` outer widening.
  Its analytic per-example cap is
  `H = E_lse/2 + 2^-20 + 4*2^-23`; the policy requires source LSE ULP
  `<=2^-16`, observed half-width `<=H`, and `H<1e-4`.
- A cap or envelope violation is failure, not a reason to tune a constant.
  Every operation class includes a first-float-outside and a `1e-4` mutation
  that must be rejected.

The first fixed-cap draft correctly failed its own CPU stress test: 120 of
12,750 retained comparisons had half-width above `2^-14`, with maximum
`6.246579553227481e-05`.  A fresh independent review identified an internal
cap/formula inconsistency, not empirical CUDA drift, and derived the analytic
cap above while keeping all LSE ULP and floor constants unchanged.  The
repaired full CPU synthetic scan passes.

For the subset-invariant boundary, `b_min` is not accepted merely because its
immediate predecessor fails locally.  The code first bounds every cap-eligible
candidate by `E_lse <= 8*2^-16 + 2^-20`, obtains a global lower bound, verifies
that the complete lower-bound-to-candidate interval occupies one source-ULP
bucket (so the predicate is monotone), and only then requires `b_min` to pass
and its predecessor to fail.

## Frozen execution identities

- Diagnostic source: `d7e3f0d763b35e997b50f532e2a57bc399df26c93c60a70af6b5e76a27f4083d`
- Diagnostic tests: `841e214ba39640e305dfe15f340c2319e4bef485bd27219c559e8563a58b3ab2`
- Slurm wrapper: `1bedcf8b3418ebff72378d0c02473b4fae9a2ba027e8fd42ad7939996b9fefcb`
- Diagnostic source closure (58 files):
  `34d6f0c37caabf6039675b437ab708f2efe515fdb400bcbecc2fe1604ecf3fc3`
- Frozen split manifest:
  `413264e4fa6473c3363b1f16a73f7e03eaa237dadbc4a3a8c07ad91a841d5d9c`

`bash -n`, closure replay, and 24 focused tests pass.  The wrapper fails if
CUDA is unavailable, disables bytecode writes, creates no experiment artifact,
and suppresses successful preflight output.  The diagnostic's only stdout is
one canonical aggregate JSON object containing counts, per-field maxima and
violation censuses, environment versions, and PASS/FAIL.  It contains no IDs,
paths, tensor values, labels, outcomes, gains, or maxima locations.

If a fresh code review returns GO, it may authorize exactly one submission of
the wrapper hash above.  Any exception, timeout, incomplete scan, unexpected
stdout, wrong fit/checkpoint census, negative-case acceptance, or non-PASS
status stops the workflow without retry.

The first submission review returned NO-GO because the implementation only
required the predecessor's conjunction `cap_ok && subset_ok` to be false.  It
did not prove that cap eligibility remained true and the subset invariant
alone flipped to false.  The repaired runtime now requires both conditions
separately, and the unit test binds them.  The hashes above supersede the
pre-review diagnostic identities; the old wrapper is intentionally invalid.
The final witness now also requires the first and recomputed candidates to be
identical and binds the global lower bound, final candidate, and predecessor
to the same ULP/envelope bucket.  A closed-form comparison census rejects any
missing, duplicated, or unexpected field before the aggregate report is
printed, and the wrapper disables the CUDA cache.

The first authorized diagnostic attempt, job `10137369`, consumed wrapper
`4b2178a4...` and failed before the scan because frozen storage retains 64
candidates while the extractor wrongly required an unsliced width of 16.
That failed identity is preserved in
`R079_NUMERIC_DIAGNOSTIC_INPUT_SHAPE_FAILURE.md` and cannot be resubmitted.
The shape-rescue extractor now requires raw ID/logit tensors with the same
shape `[15,K]`, `K>=16`, and copies only `[:, :16]`, exactly matching the
frozen Direct-native lattice.  Tests bind real K=64 selection, no aliasing,
K<16 and width-mismatch rejection, selected-prefix ordering, and the fact that
unselected tail disorder/nonfiniteness is irrelevant.  No numeric policy,
grid, census, split, or report field changed.  The hashes above identify this
new rescue candidate and require a new review.
