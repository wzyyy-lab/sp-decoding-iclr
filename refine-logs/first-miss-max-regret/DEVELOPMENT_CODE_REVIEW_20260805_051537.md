# CAMRS Development Code Review

Reviewed 2026-08-05 under the ARIS `experiment-bridge` gate. The reviewer
was a fresh same-family GPT-5.6-Sol xhigh agent operating read-only. Assurance
is provisional because no independent-model integrity audit was available.

## Scope

The review covered the normative CAMRS proposal and capacity verdict, the
prelaunch Direct-control freeze, the development trainer and tests, the Slurm
wrapper, the complete declared runtime dependency closure, the isolated
validation collection, the eight external training parts, and the actual
Direct/control/capacity artifacts. Authorization was restricted to exactly one
D64/H4/L1 seed-0 development run.

## First-pass verdict: NO-GO

The reviewer found one blocking provenance omission:
`src/sph/first_miss_max_regret_selector.py` directly imports
`src/sph/first_miss_value_selector.py`, but that dependency was absent from
both the development trainer's start/end source snapshots and the Slurm hash
preflight. All scientific semantics, isolation, external controls, selection,
gate arithmetic, and output-on-failure behavior otherwise passed review.

## Remediation

- Added `src/sph/first_miss_value_selector.py` to
  `development_source_paths()`.
- Pinned its SHA256
  `caa9c5a611e7ead6d880c876e0a002b544115fc57dd2e43d46987b64195a82d2`
  in the Slurm wrapper.
- Updated the wrapper's development-trainer pin to
  `4720a07b062b6e77607ca867c97e9d8b37e64944962c25ac03c1320edb8ccf0b`.
- Added an AST-based regression that checked 14 captured local Python files
  and all 31 local import edges.
- Final development test SHA256:
  `c9bf17c24750195f7be0bc222f5b880290040115a65dcf33ffdcb92cb2bae0b6`.
- Final Slurm wrapper SHA256:
  `49de64a487e7de60860d69d9ad23f6f207f400b3830daa09ea2d566d5e501494`.
- Focused verification: 40 tests passed.
- Full verification: 228 tests and 3 subtests passed.
- Python compilation, Slurm syntax, CPU Direct-control preflight, and
  `sbatch --test-only` passed.

## Focused re-review verdict

No blockers remain. The reviewer independently confirmed:

- the formerly missing dependency is captured and pinned;
- the trainer and signed-value hashes match the wrapper;
- no local import edge or captured source lacks a provenance check;
- the remediation changes provenance only;
- the proposal, capacity verdict, prelaunch freeze, Direct artifacts,
  isolated data, and capacity artifacts retain their verified identities;
- objective, decoder, physical/prompt isolation, 37,221-step budget,
  checkpoint rule, external controls, gate arithmetic, output-on-failure, and
  one-job scope remain unchanged.

> GO: authorize exactly one D64/H4/L1 seed-0 CAMRS development job, no other runs

## Nonblocking caveats

The 30-minute A40 debug allocation is adequate but not generous. Python,
PyTorch, and safetensors versions are recorded rather than shell-preflight
pinned. `docs/method.md` is behind the normative CAMRS refine documents.
Assurance remains same-family and provisional. None broadens the authorization:
no additional seed, D640 variant, repeat, continuation, calibration, threshold
change, formal-test access, or post-outcome rescue is permitted.

