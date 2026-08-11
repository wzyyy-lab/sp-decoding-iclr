# R050 Amendment: Exact Target-Seeded Fast-K64

**Date:** 2026-08-10  
**Parent:** `EXPERIMENT_PLAN_AMENDMENT_R049_20260810.md`

R049-A found that shallow target residuals do not expose the next target token
to a zero-parameter target-logit reader: at L12, reward/count recovery was only
about 2.0%/2.7%, and the deployable zero-harm policy recovered nothing.  The
pre-registered direct/low-rank early-target route is closed.  A post-hoc 6.56M
same-set affine probe is rejected because it is severely over-parameterized for
3,984 valid rows and repeats the historical capacity-without-transfer failure.

R050 removes learned correction entirely.  It reschedules one exact target
token that the verifier must compute anyway and uses it to seed the frozen
Fast-K64 suffix.

## Serving-shaped algorithm

Given target prefix KV ending immediately before the current anchor:

1. Run target on the single anchor token and take the authoritative greedy
   `p0`; this also extends every target-layer KV by the anchor row.
2. In parallel where possible, run the existing DFlash backbone/base vocabulary
   projection for all 16 draft positions.
3. Force Fast-K64 position 0 to `p0`; initialize the frozen Domino GRU with
   `[anchor, p0]`; generate only `p1..p15` with the unchanged gathered K64
   scoring rule.
4. Reuse the anchor-extended target KV and run target on `[p0..p15]`.  Row 0
   verifies `p1`, row 14 verifies `p15`, and row 15 provides the bonus token.

Target token-layer geometry changes from one unsplit 17-token call to a split
`1+16` call.  No target token is computed twice, no target KV is invalidated,
and no new trainable parameter is introduced.

## R050-A exact accuracy gate

Evaluate batch 1 on the complete fixed `validation_select` set using clean
target prefix caches.  Compare:

- historical/replayed released Domino;
- ordinary Fast-K64;
- target-seeded Fast-K64 under the serving-shaped split target path;
- an unsplit `[anchor + seeded proposal]` numerical control.

Report prompt-balanced overall and chat/code/math EAL, gained/lost blocks,
position-0 parity, full-accept bonus parity, and split-versus-unsplit parity
over all 17 output decisions.  The first 16 rows determine EAL; row 16 is the
bonus emitted after a full accept.  For numerical
control, estimate a centered-logit tolerance only from rows whose split and
unsplit argmax already agrees; a mismatching row may not enlarge its own
tolerance.  Rows below twice this tolerance are ambiguous.  All remaining
stable rows must have identical argmax.

Hard outcome gates:

- fixed EAL `< 8.325485909`: close R050;
- `8.325485909 <= EAL < 9.0`: accuracy passes, but do not pay for integration;
- EAL `>= 9.0`, no domain regression, and stable parity 100%: proceed to eager
  split timing and SGLang design.

## System gate

At achieved EAL `E`, the maximum target-seeded iteration-time ratio compatible
with 1.15x throughput over Domino is

```text
(E + 1) / (1.15 * (7.239552964 + 1)).
```

First measure on the same A40/B16 setup:

- unsplit target `[anchor+16]`;
- split target `[anchor] + [16]`, including launch and KV costs;
- forced Fast-K64 suffix head;
- serialized and ideal-overlap schedules for target-anchor versus DFlash.

Final claims still require lossless token parity and end-to-end throughput at
least 1.15x in SGLang paged-KV/CUDA-graph/Triton execution.
