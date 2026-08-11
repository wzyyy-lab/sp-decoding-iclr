# R079 Split-Filter Failure-Rescue Review

This immutable review is identical in substance to the binding latest review
in `R079_ARTIFACT_CODE_REVIEW.md` at 2026-08-05 13:28 +0800.

## Verdict

**GO**, with no blocking findings, authorizing exactly one split resubmission
and then its already-reviewed independent audit. It does not directly
authorize any outcome materialization or training.

## Independent reconstruction

- Exact materializer/auditor/receipt/source-manifest hashes matched.
- Old whole-file validation logic was independently reproduced: 1,987
  Phase-3 overlaps.
- Correct selected-row logic yielded 100,000/300/600 exclusions and zero
  Phase-3 overlap for producer/validation/reserved.
- The scientific split remained 1,587/200/200 prompts and
  12,686/1,600/1,600 blocks with the preregistered domain counts.
- Both independent implementations, enriched provenance, semantic hashes,
  receipt pins, 57-file closure, and five wrappers matched.
- Focused CPU checks passed 20 tests; no outcome/GPU/Slurm/job was touched.

## Non-blocking hardening notes

1. A same-descriptor parse or post-parse rehash could narrow a theoretical
   concurrent exclusion-file replacement window; the mandatory independent
   audit currently contains it before downstream authorization.
2. The complete first-party Python closure does not itself include `.sbatch`
   files, although all five wrapper identities/syntax/static pins were checked.
