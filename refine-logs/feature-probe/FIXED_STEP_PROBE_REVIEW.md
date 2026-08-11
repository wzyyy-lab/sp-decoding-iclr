# Fixed-Step Prompt-Diverse Probe Review

Date: 2026-08-05  
Assurance: fresh same-family Codex review; provisional  
Final verdict: **GO**

## Initial blockers

The independent reviewer initially withheld launch authorization because:

1. a one-hour walltime was not a defensible fail-safe for the D640 cell;
2. the summary compared provenance across cells but did not pin both cells to
   the exact trainer, head, validation collection, and target files reviewed;
3. per-domain bootstrap dictionaries were constructed from unordered set
   iteration.

The implementation now requests a four-hour fail-safe allocation, while the
artifact still requires exactly 37,221 optimizer updates. It pins the exact
reviewed trainer/head/validation/target SHA256 identities and all eight OPB
metadata hashes, and it sorts prompt keys before every seeded domain
bootstrap.

## Additional audit corrections

The reviewer also identified four interpretation/reporting issues that were
fixed before launch:

- prompt repetition is 30 versus three epoch-level passes (about 240 versus
  24 block presentations per prompt), not 3.75 prompt repetitions;
- the historical diversity curve used `dpace`, whereas the new matched cells
  use `candidate_dpace`, so it is supporting rationale rather than an exact
  control;
- a negative output is labeled `engineering_stop`, not
  `scientific_negative`;
- domain EAL and domain bootstrap now share the prompt-balanced estimand;
  block-balanced domain EAL is retained only as an explicitly labeled
  diagnostic.

Adversarial tests additionally reject a source substituted identically in
both cells, target-signature drift, invalid selected epochs, prompt/domain
inconsistency, and calibration attempts to pass the raw gate.

## Final verification and authorization

The reviewer independently confirmed:

- 99,356 prompts, 793,989 blocks, exact prompt hash
  `45471a62f93a488f3f7653c096bebcddb0ddae3773f6c99744bd070e348a9405`,
  and `3 * ceil(793989/64) = 37221` updates;
- raw example-level reconstruction, exact prompt support, paired prompt and
  domain bootstrap, and pinned provenance fail closed;
- Slurm paths and summary job-ID interface agree;
- no launch blocker remains.

The reviewer ran 25 relevant tests; the final author-side full suite passed
152 tests plus three parameterized subtests. The **GO** authorizes only this
adaptive development diagnostic. It does not authorize a method claim or an
information-ceiling conclusion.
