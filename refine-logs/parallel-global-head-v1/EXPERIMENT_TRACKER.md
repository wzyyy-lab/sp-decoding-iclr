# PGCF-16 Experiment Tracker

Canonical tracker: `EXPERIMENT_TRACKER_20260810_114625.md`

Current: exact `PGCF-v1 CLOSED` after `PGCF-008 COMPLETE/FAIL`.
Gate1 capacity is COMPLETE/PASS: capacity-only job `10166838` reaches exact
oracle EAL with 99.985%/99.969% candidate/hard accuracy and zero harm, while
teacher-only job `10166815` reaches 99.813% supported policy accuracy.  The
intended curriculum failure `10166814` remains explicitly recorded.  Matched
local diagnostic job `10166853` is also complete at EAL 10.9434 and
99.680%/99.476% candidate/hard accuracy.  G4 matched 20k job `10166898`
completed, but Gate-2 job `10167001` found global EAL `6.10277`, local
`6.08892`, global-local `+0.01385` with 95% CI `[-0.06037,+0.08698]`, versus
Domino `7.23955` and the required `8.32549`.  PGCF-009 through PGCF-019 are
closed for v1.  A newly preregistered, still fully parallel/global/one-chain
v2 refinement is the only authorized research continuation.
