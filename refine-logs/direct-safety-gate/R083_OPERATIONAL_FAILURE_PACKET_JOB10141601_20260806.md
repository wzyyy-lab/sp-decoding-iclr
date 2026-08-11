# R083 Operational Failure Packet — Job 10141601

**Date:** 2026-08-06  
**Stage:** one-shot contextual falsifier  
**Outcome:** **operational failure; scientific evaluation not reached**  
**Retry:** forbidden; opening consumed

## Terminal record

- Slurm job: `10141601` (`pros-falsifier`, debug, `gpu3-9`).
- State: `FAILED`, exit `1:0`, elapsed `00:00:30`.
- Start/end: `2026-08-06T13:05:16` to `2026-08-06T13:05:46`.
- Authorized wrapper SHA-256:
  `b20fc9461daac0385b09fbd0840aa5d476dcf71fadd3b6ba3288938b9d124560`.
- Stdout: 4,915 bytes, SHA-256
  `55dc834f6011e5986b996ee30d743f57e07c8087bc56edb9a3091a5a5c15f68f`.
- Stderr: 811 bytes, SHA-256
  `189dfeb09c983dabf828e64fcef11efd09e90878ce99e8519201dbea8d1779e5`.

The exception was:

```text
RuntimeError: split manifest differs from reconstruction: ['provenance']
```

## Exact run identity

- Evaluator:
  `54d430a6d9d92118e2005e6c985c6c04f0424cfcc86afe60c5eecce5f39aa571`.
- Materializer:
  `90001f5b2f0224e79d8d205bdd781876f7dcbed89273fb55d9ca4c3f53d95b2b`.
- Artifact verifier:
  `852915273e330344c1afe472f9d0c2b4789b564e6636af897051486fbaa3a6d9`.
- Publication module:
  `141abdef88320173fb03c438c5c54907118a64bcf0a29932268be389fb4f5f1c`.
- Source manifest:
  `204c025305a9665803e714708dc0eab29394644d5905ad76f1715c7309020878`.
- Source entries: 63 files, digest
  `fb943ef5be7fd2597e92f8bb230eaef480a4b78f33a1693a105e0f73aadbe796`.
- Frozen split manifest:
  `7a572670867bbf6f811aad58c7ed1365f179083cda8187f5402221d554d1f1c0`.

The official `seed0/RESERVATION.json` is mode `0400`, has SHA-256
`d51adb853ee8a0d48159a32c584c260077c8f1c4a7e948df4c407bb73149abff`,
and exactly binds job, output, seed, purpose, evaluator, wrapper, and source
manifest. Its state is `UNCOMMITTED_UNTIL_READY_JSON`. The job source snapshot
independently replays as the exact 63-file closure.

## Root cause

The scientific split itself did not drift. Its historical `provenance` embeds
the source closure under which R079 created it, whereas R083 reconstruction
injected the evaluator's current rescue-v2 closure and required the entire
parsed manifest to be equal.

| Field | Frozen split provenance | R083 reconstruction |
|---|---|---|
| Source-manifest SHA | `2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4` | `204c025305a9665803e714708dc0eab29394644d5905ad76f1715c7309020878` |
| Source-file count | 59 | 63 |
| Entries digest | `cc4a9ecea4c1f9a32e0ceae0e3c5551759e51be141bc4a9b37b6a5d03b88d02a` | `fb943ef5be7fd2597e92f8bb230eaef480a4b78f33a1693a105e0f73aadbe796` |

Frozen provenance SHA-256 was
`e6aabf46791f1a787708893cddb48d53f035921fa1e11bd4a0d21938757e675b`;
runtime provenance SHA-256 was
`0e31a095cecf698205aa8b9dee75fdd57a8286203dc2733e984892b5dd07d4d0`.
The independently inferred reconstructed-manifest SHA-256 was
`401247edaef9e4556bed2a74c90240a5b0c213af06cccd45959af5e1ac2ae985`.
Every other top-level manifest field reconstructed identically. This is a
versioned-provenance contract bug, not scientific split/data drift.

## Exact opening boundary

Before the failure the program reserved the output, snapshotted and verified
the source closure and frozen identities, loaded the R082 sidecar/ridge weights,
read the split JSON, deserialized the canonical train collection, read the
exclusion manifests, and built identity records. This consumed the one-shot
opening.

It failed before split assignment extraction, frozen Direct/target loading,
outcome materialization, any PROS/comparator forward or score, metrics,
bootstrap, or gate adjudication. Therefore no falsifier outcome or scientific
score was produced or inspected.

The hidden READY directory is only the preregistered filesystem smoke with
`scientific_status = NOT_APPLICABLE_FILESYSTEM_SMOKE`; it is not an official
R083 result. Official `seed0` contains no READY, publication manifest, metrics,
gate receipt, outcome bundle, score/record file, or scientific-failure packet.

## Protocol consequence

The frozen rescue-v2 contract defines every non-0/2 exit as an operational
failure that consumes the opening. Consequently the exact route is closed and
cannot be retried or rescued by changing provenance verification, split, seed,
threshold, checkpoint, fit, or later data.

