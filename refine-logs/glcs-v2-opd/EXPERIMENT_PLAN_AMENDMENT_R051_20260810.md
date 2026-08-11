# R051 Amendment: Minimal Exact Prefix Seed

**Date:** 2026-08-10  
**Parent:** `EXPERIMENT_PLAN_AMENDMENT_R050_20260810.md`

R050 single-token seeding failed: its clean unsplit authority EAL was only
`7.617954`, below `8.325486`.  R051 tests the only authorized exact extension,
once, at seed lengths `s={2,3,4}`.  No learned head or seed length above four is
allowed.

For seed length `s`, target sequentially processes inputs
`[anchor,p0,...,p{s-2}]`, producing authoritative exact tokens
`p0,...,p{s-1}` and a cache ending at `p{s-2}`.  The frozen Fast-K64 GRU consumes
`[anchor,p0,...,p{s-1}]` and generates positions `s..15`.  A serving-shaped
final verifier starts with input `p{s-1}`, not `p_s`, and runs through `p15`.
Thus the split target still processes exactly 17 token rows:

```text
s sequential seed rows + (17-s) final verifier rows = 17.
```

This equality is a compute-count identity, not a latency claim: each sequential
decode rereads target weights and adds launch/synchronization overhead.

## Accuracy authority and gates

For every generated proposal, clean batch-1 **unsplit** `[anchor+proposal16]`
verification is the sole accuracy authority.  Split self-EAL and its full
17-row/bonus parity are deployment diagnostics only and cannot make a seed pass.

Evaluate the complete fixed `validation_select` set and report overall plus
chat/code/math EAL for s=2,3,4.  Select the smallest seed satisfying unsplit EAL
at least 9.0.

- if `max_s EAL < 8.325485909`: close the exact-seed family;
- if `8.325485909 <= max_s EAL < 9.0`: accuracy-only; continue only if an
  actual same-job timing already satisfies the throughput inequality;
- if a seed reaches 9.0: authorize system profiling for the smallest such seed.

At EAL `E`, the hard iteration-time bound for 1.15x throughput is

```text
T_seed / T_Domino <= (E + 1) / (1.15 * (7.239552964 + 1)).
```

Final SGLang deployment still requires non-tied split parity 100% and exact
full-accept bonus parity 100%.

