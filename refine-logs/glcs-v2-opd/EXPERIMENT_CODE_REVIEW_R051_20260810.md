# R051 Exact-Prefix Code Review

**Verdict:** GO for one full A40 `validation_select` run  
**Review independence:** same-family / provisional

The initial review found one result-contract blocker: system-profile selection
incorrectly required per-domain non-regression in addition to the preregistered
clean-unsplit EAL threshold.  The implementation was corrected so routing uses
only clean batch-1 unsplit EAL, selects the smallest seed reaching 9.0, and
keeps per-domain results as diagnostics.

The final review confirmed:

- sequential caches for `s=2,3,4` end at `p{s-2}`;
- the frozen GRU consumes `anchor+p0..p{s-1}` and resumes generation at `s`;
- the final split verifier starts from input `p{s-1}` and yields all 17 rows,
  including the bonus decision;
- clean batch-1 unsplit verification of the generated full proposal is the
  sole accuracy authority; split EAL and parity cannot affect the route;
- forced-prefix K64 support, BF16 scoring, ordinary Fast-K64, and the R050
  one-token API have no identified regression.

Focused verification passed: 21 tests, Python compilation, and Slurm shell
syntax.  No expected A40 memory or 30-minute runtime blocker was identified.

