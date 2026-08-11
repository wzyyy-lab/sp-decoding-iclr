# R082 Publication Rescue v1 — Code / Launch Review

**Date:** 2026-08-05  
**Reviewer:** fresh `gpt-5.6-sol`, `ultra`, read-only  
**Verdict:** **GO — zero blockers**

Exactly one submission is authorized, and only for wrapper SHA-256
`1bffe45017da30393f2dbda5bd33b1e14ff1e0ca1a60c0be4fc86f4ad1885c74`.

The reviewer independently matched trainer `9bc069a5...`, old trainer snapshot
`1e95a5dd...`, old/new wrappers, closure `f36291a9...`, and all named test
hashes. Its old/new AST and source diff found no change to data, seed, order,
model, objective, optimizer, schedule, ridge, checkpoint selection, threshold,
or any existing scientific constant.

It accepted the exclusive output reservation, read-only reservation identity,
exact symlink-free tree manifest, repeated tree scans, bottom-up fsync, parent
pending receipt, directory-fd READY hardlink, final output-directory fsync,
failure preservation, strict consumer, publication/scientific-status
separation, pre-data wrapper smoke, and GPFS race/crash evidence. It replayed a
93-test focused suite with two expected skips, closure count/digest, wrapper
syntax, retained smoke identities, job10138454 accounting, and an empty active
fit queue without opening `.pt` inputs or using the old stdout trajectory.

The reviewer classified Slurm `$0` as the executed spool copy. External launch
authorization binds the submitted project wrapper hash; result review must
require the runtime `$0` hash printed by the wrapper, smoke identity,
reservation, and publication manifest all to equal the authorized hash above.
Any mismatch rejects the result and still consumes authorization.

Any submission, abnormal exit, scientific FAIL, hash/binding mismatch, or
non-successful Slurm state permanently consumes this authorization. There is
no retry. After submission, only a hash-bound `result-to-claim` review is
authorized. R083, falsifier, validation, reserved/formal data, and all other
later-data stages remain closed.
