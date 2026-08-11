# PROS-Gate CPU Gate-0 Code Review

Reviewed on 2026-08-05 under the ARIS `experiment-bridge` boundary.  The
reviewer was a fresh same-family GPT-5.6-Sol xhigh agent operating read-only.
Assurance is provisional because an independent-model integrity reviewer was
not available.

## Authorized review scope

The review covers only the synthetic CPU implementation of the normative
PROS-Gate proposal:

- `src/sph/direct_safety_gate.py`;
- `src/sph/direct_safety_protocol.py`;
- `tests/test_direct_safety_gate.py`;
- `tests/test_direct_safety_protocol.py`;
- the already pinned Direct producer and accepted-prefix dependency.

No review action authorizes real dataset loading, split/outcome/capacity
artifact materialization, training, evaluation, GPU use, or job submission.

## First-pass verdict: BLOCKING / NO-GO

The reviewer independently matched the four submitted hashes, reran the then
20 focused tests, verified the pinned Direct identities, and confirmed that
the new modules had no filesystem, dataset, CUDA, checkpoint-loading, or
artifact-writing path.  Static checks passed for the proposed 200-dimensional
features, four pools, 38,674-parameter sidecar, strict-positive decoder,
frozen-producer capture, utility-weighted hinge, prompt weighting, token
outcomes, split counts/order, warmup, ridge algebra, and recovery semantics.

Six Gate-0 blockers remained:

1. `BlockKey` and `pass_index` accepted floats and booleans instead of only
   canonical nonnegative integers.
2. The capacity mass `1/512` was multiplied and then averaged, producing
   `sum(loss)/512^2` rather than `sum(loss)/512`.
3. Zero initialization masked tests of nonzero decoding, exact 257-vector
   pooling, empty-change behavior, dtype/defaults, and gradient isolation.
4. There was no independent saved-record evaluator and no explicit
   least-privilege stage/split allowlist.
5. The exact ordered 21-scalar comparator representation was not implemented.
6. The prompt-unique 256/128/128 capacity selector was absent, while capacity
   checkpoint selection trusted a caller-supplied `passed` flag.

The reviewer also noted nonblocking future hardening for canonical lowercase
hashes, the binding ridge coefficient, bootstrap conventions, hook cleanup,
strict token types, and later import-closure freezing.

## Remediation

The CPU-only remediation closed the six findings without touching the pinned
Direct trainer or producer:

- Strict `Integral`, non-boolean, nonnegative block/pass identities now use
  canonical decimal serialization; SHA256 strings must be lowercase.
- The hinge exposes explicit multiplier-mean and probability-mass-sum
  reductions.  The capacity wrapper requires exactly 512 records and sums
  exact `1/512` masses.
- Tests now exercise positive, zero, and negative scores under a nonzero final
  projection; force identical Direct paths to zero; compare the captured
  257-vector against manual all/changed/first/count pools including the empty
  changed set; validate all dimensions/defaults/dtypes; instantiate the exact
  D64/H4/L1 axial-additive L15/K16 producer; prove source/producer/input
  gradient isolation; and verify hook cleanup after an exception while
  preserving an existing hook.
- `reconstruct_saved_gate_evaluation` independently rebuilds strict actions,
  DFlash/Direct/method/oracle accepted lengths and prompt-balanced EALs,
  hinge/regret/slack/violations, unclipped recovery, accepted-token and
  first-token outcomes, action/outcome composition, and explicit numerators
  and denominators.  `assert_stage_splits` rejects every split outside a
  stage's allowlist, including `validation_gate`, `validation_select`, formal,
  and reserved surfaces.
- `scalar_comparator_features` emits the fixed float32 `[N,21]` order, with
  normalized change count, three changed-margin summary groups,
  entropy/retained summaries, first-position features, and exact zero summaries
  for no-change blocks.  Golden vectors and the default ridge `1e-3` numeric
  solution are tested.
- `select_capacity_records` derives strata from gain/path, applies the frozen
  `pros-capacity-v1` hash in harmful/changed-neutral/beneficial scarcity order,
  excludes prompts globally, requires exact 128/128/256 composition, and fails
  on shortage.  `capacity_gate_passes` independently checks all finite/loss,
  bound, count, recall, avoidance, utility, recovery, and harmful-APPLY
  conjuncts plus count consistency; checkpoint selection ignores any caller
  `passed` flag.
- Token reconstruction additionally requires int64 IDs/paths, valid path
  indices, shapes, and a common device.

## Remediated frozen identities and verification

```text
direct_safety_gate.py       e3bd6392f7430e60e0eef16217dc904eeb018313ae8d4f543bd089a1943739b6
direct_safety_protocol.py   4c5ed7bb181aae92b3d941f427359bc82051aba39cb94f5bdfd33d63c22f044e
test_direct_safety_gate.py  728ba80518a0fde92ccd2db1a6621eeb21ae86306c8ec4ed44e9da5b2dd81740
test_direct_safety_protocol.py
                            f7c001517e45aa5b2e62e173515f24cb838b6a07f24859ab05fa31e9bc83a596
Direct trainer (unchanged)  e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110
Direct producer (unchanged) f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06
```

- Focused verification: 32 passed.
- Full verification: 260 passed, 3 subtests passed.
- Python compilation and `git diff --check`: passed.
- No real data, experiment artifact, checkpoint, CUDA, or GPU path was added.

## Final focused re-review: NO-GO

The same reviewer performed the protocol's one permitted focused re-review.
It independently matched all submitted and pinned hashes, reran 32 focused
tests and the 260-test / 3-subtest full suite, passed `git diff --check`, and
again found no filesystem, dataset, CUDA, checkpoint, or artifact access.

Findings 1, 2, 3, and 5 were closed.  Findings 4 and 6 were only partial:

1. The saved-record evaluator divided false APPLY by `apply_count`; the frozen
   definition instead divides by every nonbeneficial block,
   `harmful_count + neutral_count`.  The existing hand case accidentally made
   those two denominators equal.
2. Redundant first-token witnesses were not checked against the defining
   accepted-prefix condition `length > 0`.
3. The capacity adjudicator still accepted caller-provided recovery without
   deriving its numerator and strictly positive denominator from
   base/method/oracle EAL.  A metrics row with a negative reported denominator
   could therefore pass.

The final reviewer verdict was **NO-GO**.  The reviewer explicitly authorized
only additional local CPU synthetic remediation and prohibited real data,
artifacts, training, evaluation, GPU, and launch actions.  The one permitted
re-review is exhausted; this reviewer cannot be asked again.

## Post-review local remediation (not externally accepted)

The exact final findings were repaired locally:

- false-APPLY numerator is now divided by `harmful_count + neutral_count`; a
  counterexample with six APPLY decisions but only two nonbeneficial blocks
  prevents regression;
- saved first-token booleans must exactly equal `accepted_length > 0`, with
  positive-length/false and zero-length/true corruption tests;
- capacity adjudication now requires base/method/oracle EAL, independently
  computes numerator, strictly positive denominator, and unclipped recovery,
  and cross-checks every reported recovery field to `1e-12` absolute
  tolerance;
- zero, negative, NaN, forged, and internally inconsistent recovery cases all
  fail.

Final local-only identities and checks are:

```text
direct_safety_gate.py       e3bd6392f7430e60e0eef16217dc904eeb018313ae8d4f543bd089a1943739b6
direct_safety_protocol.py   c92ba988f764515b575d5ee59d24db6aa208b95761afc1fc721384c1a7591a1d
test_direct_safety_gate.py  728ba80518a0fde92ccd2db1a6621eeb21ae86306c8ec4ed44e9da5b2dd81740
test_direct_safety_protocol.py
                            f8f4c5e4a11e6f03a8ae8d8fec87a767f824b699c0d22d6c07038a56f2c466cd
Direct trainer (unchanged)  e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110
Direct producer (unchanged) f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06
focused                    33 passed
full                       261 passed, 3 subtests passed
py_compile/diff-check      passed
```

These local checks resolve the concrete counterexamples but do **not** convert
the binding external verdict.  CPU Gate-0 remains externally unaccepted, so
the experiment bridge stops before any real-data or GPU boundary.
