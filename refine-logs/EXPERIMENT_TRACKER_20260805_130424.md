+# Experiment Tracker — PROS-Gate

> Active execution tracker for the PROS-Gate route. Earlier full-lattice, FMAS, SAVS, and CAMRS history is preserved in `EXPERIMENT_TRACKER_20260805_104754_LEGACY_ALL_ROUTES.md`.

| Run ID | Milestone | Purpose | System / Variant | Split | Primary Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R077 | M0 | first CPU Gate-0 cycle | PROS-Gate sidecar/protocol | synthetic only | semantic tests + external verdict | MUST | LOCAL_FIXED_EXTERNAL_NO_GO | first review/re-review exhausted; final local hashes pass 33 focused and 261+3 full tests, but no external acceptance |
| R078 | M0 | newly authorized fresh Gate-0 review | final local PROS hashes | synthetic only | six-finding closure, I/O isolation, focused/full tests | MUST | PASSED_GO | fresh review found exact ridge/capacity consistency blockers; both fixed; focused re-review GO with 34 focused and 262+3 local full tests |
| R079 | M1 | split/outcome artifact materialization + independent audit | frozen Direct job10133585 + PROS feature writer | Phase-3 train identities split three ways; materialize fit/checkpoint only | counts, overlap, native/state witness, hashes, record reconstruction | MUST | REVIEWED_GO_SPLIT_PENDING | artifact review GO plus fresh jq-free portability rescue GO; exact 57-file closure dccf6540; execute only split→audit→fit/checkpoint outcomes→audit→capacity artifact→audit; falsifier route absent |
| R080 | M1 | same-subset capacity plumbing | fresh seed0 38,674-param sidecar | 512 prompt-unique fit records | loss≤5% epoch0; benefit≥254/256; harm-avoid≥127/128; utility≥509/512; recovery≥.95; harmful APPLY≤1 | MUST | BLOCKED_R079 | exactly one reviewed capacity job; no nonselected rescue |
| R081 | M1 | capacity result-to-claim | independent saved-record replay | R080 artifacts | exact conjunctive PASS/FAIL | MUST | BLOCKED_R080 | FAIL closes route; capacity cannot support held-out claim |
| R082 | M2 | clean fit/checkpoint + comparator freeze | PROS + 21-scalar ridge + constants | fit updates; checkpoint selects | lexicographic selected bundle, recovery validity, hashes | MUST | BLOCKED_R081 | falsifier path absent from trainer CLI |
| R083 | M3 | one-shot contextual falsifier | frozen selected PROS/comparators | 200-prompt falsifier | recovery≥.90; beat DFlash/Direct; harm≤5%; first-token shortfall≤1; +.05 vs comparator | MUST | BLOCKED_R082 | one opening, no refit/calibration/threshold rescue |
| R084 | M3 | falsifier result-to-claim | fresh independent reviewer | R083 records/artifacts | reconstructed gate + claim boundary | MUST | BLOCKED_R083 | FAIL closes exact route |
| R085 | M4 | conditional development evaluation | frozen PROS vs DFlash/Direct controls | validation_select 147 prompts / 1,175 blocks | ΔDFlash>.28499; ≥+.05 vs both Direct controls; harm≤5%; first-token shortfall≤1 | MUST_IF_R084_PASS | BLOCKED_R084 | development-only, no checkpoint selection |
| R086 | M4 | system-cost adjudication | unfused producer-state reuse sidecar | same frozen evaluation prompts | head/total latency, memory, TPS estimate | MUST_IF_R085_PASS | BLOCKED_R085 | latency cannot rescue efficacy; fused implementation requires separate review |

