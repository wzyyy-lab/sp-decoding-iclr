# PGCF-16 Gate-2 Result-to-Claim

**Date:** 2026-08-10  
**Route:** exact PGCF-v1 training route  
**Verdict:** `claim_supported = no`  
**Confidence:** high  
**Assurance:** same-family independent Codex review, provisional  
**Scientific job:** `10167001` (`exit 2` is the evaluator's declared Gate-2 failure, not a missing artifact)

## Immutable method compliance

The evaluated head is the in-scope method: it consumes all 16 DFlash positions,
uses global non-causal mixing, makes all 16 candidate decisions in one call, and
emits one 16-token chain.  It uses no selected-token feedback, serial target
decode, iteration, beam, tree, forest, or extra online target inference.  This
negative result closes only the exact PGCF-v1 training route; it does not relax
or close the user's parallel-global-single-chain architecture contract.

## Binding evidence

The two matched seed-0 heads were trained on 1,987 prompts / 15,886 blocks and
selected only by the disjoint `validation_select` split (147 prompts / 1,175
blocks).  Gate-2 used 10,000 prompt-cluster bootstrap draws.

| Quantity | Result |
|---|---:|
| Pure DFlash base Top-1 EAL | 6.0685131195 |
| Matched local head EAL | 6.0889212828 |
| Global PGCF-v1 EAL | 6.1027696793 |
| Released Domino EAL | 7.2395529640 |
| Required 1.15x-Domino EAL | 8.3254859086 |
| Deployable pure-base Top-16 oracle EAL | 10.9092565598 |
| Global minus local | +0.0138483965 |
| 95% paired prompt-bootstrap CI | [-0.0603741497, +0.0869776482] |
| Global oracle-gap recovery from base | 0.7077% |
| Global complete eager p50 / Domino | 1.820672 ms / 4.224000 ms |

The global-local effect misses the preregistered `+0.15` gate, its confidence
interval crosses zero and has an upper endpoint below `+0.15`, and chat regresses
from base (`2.75484` versus `2.77493`).  The remote intervention changes global
outputs and lowers EAL to `6.07969`; the matched local negative control has
`0/18,800` token mismatches.  This proves remote-input sensitivity, not useful
held-out global information, because the primary global-local effect is tiny and
statistically unresolved.

## Diagnosis

Capacity and latency are not the immediate blockers.  The 2,438,400-parameter
global head exactly fits the frozen 512-block capacity subset (EAL `11.02148`,
candidate/hard accuracy `99.985%/99.969%`, zero harm), and its eager latency is
well below Domino.  On disjoint data, however, it transfers almost none of that
capacity.

The failure is specifically prefix-utility transfer:

- of 946 repairable first-rejection blocks, global repairs 46 (`4.86%`), versus
  Domino's 339 (`35.84%`);
- global has a `4.36%` prefix-harm rate, roughly Domino-like, but only `52%`
  precision when it changes the frontier;
- 2,487 of its 2,641 token changes occur after the base first rejection, and
  619 of 756 changed blocks are suffix-only changes with no accepted-length
  effect;
- a perfect one-frontier-only repair oracle reaches only `7.49854`, still below
  `8.32549`, so a successful method must learn several mutually consistent
  positions in the one parallel output, not merely gate or repair one token;
- late training improves a fixed train diagnostic to `7.37305` while validation
  falls to `5.98627`, which is direct overfit evidence.

## Claim boundary and routing

Supported: PGCF-v1 has same-subset capacity, uses remote information, and passes
the A40 eager head-cost gate.

Not supported: useful global-context generalization, superiority to Domino,
the 1.15x fixed/dynamic EAL target, formal-test performance, or SGLang
throughput.

The exact v1 curriculum route is therefore closed.  PGCF-009 through PGCF-019
are not authorized from this result; collecting 199.8K blocks, increasing v1
parameters, or extending v1 training would contradict its frozen Gate-2 stop.
The only allowed continuation is a newly named and preregistered in-spec v2
whose loss/mechanism directly optimizes multi-position clean-prefix utility.
It must still satisfy every immutable full16/global/non-causal/one-call/one-chain
constraint and must pass a new disjoint held-out gate before any system work.

Primary artifacts:

- `artifacts/results/pgcf16_gate2_10167001/report.json`
- `artifacts/models/pgcf16_r047_screen_10166898/global_seed0/report.json`
- `artifacts/models/pgcf16_r047_screen_10166898/local_seed0/report.json`
- `profile_output/pgcf16_eager_10166801.json`
- `.aris/traces/result-to-claim/2026-08-10_run02/001-pgcf-gate2.response.md`

