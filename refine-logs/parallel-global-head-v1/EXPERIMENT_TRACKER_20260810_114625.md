# PGCF-16 Experiment Tracker

| ID | Stage | Artifact/job | Status | Authorization / exit condition |
|---|---|---|---|---|
| PGCF-000 | Research refine | FINAL_PROPOSAL + READY9.3 | COMPLETE | experiment-plan authorized |
| PGCF-001 | G0 implementation/tests | `EXPERIMENT_CODE_REVIEW_20260810_121346.md` | COMPLETE | 22 focused tests + fresh review GO |
| PGCF-002 | G1 A40 mechanics | job `10166796` | COMPLETE | finite BF16 train/eval/checkpoint; no harm |
| PGCF-003 | G2 global capacity | jobs `10166838/10166815` | COMPLETE | dense CE four-way pass + teacher99.813%; curriculum failure retained |
| PGCF-004 | G2 local diagnostic | job `10166853` | COMPLETE | EAL 10.9434, 99.680%/99.476% accuracy; diagnostic PASS |
| PGCF-005 | G3 eager profile | job `10166801` | COMPLETE | complete ratio 0.4310x <=1.20x |
| PGCF-006 | G4 global screen | job `10166898_0` | COMPLETE/FAIL | best validation EAL 6.10277; train diagnostic overfits |
| PGCF-007 | G4 local screen | job `10166898_1` | COMPLETE | matched control EAL 6.08892 |
| PGCF-008 | G4 remote intervention | job `10167001` | COMPLETE/FAIL | delta +0.01385, CI crosses zero, domain gate fails |
| PGCF-009 | G5 full16 OPB collect | 4-task array | CLOSED-V1 | G4 C2 failed; no v1 collection |
| PGCF-010 | G6 Stage-A primary | OPB25K seed0 | CLOSED-V1 | Gate-2 stop |
| PGCF-011 | G6 no-teacher diagnostic | conditional | CLOSED-V1 | Gate-2 stop |
| PGCF-012 | G7 mixed adaptation | seed0 | CLOSED-V1 | Gate-2 stop |
| PGCF-013 | G7 mixed adaptation | seed1 | CLOSED-V1 | Gate-2 stop |
| PGCF-014 | G7 mixed adaptation | seed2 | CLOSED-V1 | Gate-2 stop |
| PGCF-015 | G8 formal receipt/collection | reserved formal | CLOSED-V1 | Gate-2 stop |
| PGCF-016 | G8 fixed formal | one-shot | CLOSED-V1 | Gate-2 stop |
| PGCF-017 | G8 dynamic formal | one-shot | CLOSED-V1 | Gate-2 stop |
| PGCF-018 | G9 SGLang integration | mechanics | CLOSED-V1 | efficacy absent |
| PGCF-019 | G9 paired performance | A40 | CLOSED-V1 | efficacy absent |

No tree/beam/serial-target/causal route is a fallback in this tracker.
The exact v1 route is closed; the immutable parallel-global-single-chain
workstream may continue only through a separately refined/preregistered v2.
