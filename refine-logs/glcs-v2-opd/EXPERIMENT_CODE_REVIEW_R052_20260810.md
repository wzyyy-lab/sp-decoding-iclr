# R052 Eager Profile Code Review

**Verdict:** GO for A40  
**Review independence:** same-family / provisional

The reviewer found no blocking timing or causal-geometry error.  Both complete
callbacks include base vocabulary GEMM, proposal head, and target verification.
For R051 the sequential cache ends at `prefix+4` after input `p2`, and the final
verifier consumes `p3..p15` (13 rows).  CUDA-event placement, synchronization,
and cache reset are symmetric and include the dependent GPU work.

The implementation was additionally hardened with proposal/input/cache length
assertions and per-context throughput/common-path analysis.  The same-evaluator
output-advance ratio yields a maximum R051/ Dom­ino time ratio of
`1.0545181356` for the 1.15x target.  Focused tests, Python compilation, and
Slurm shell syntax pass.

Scope remains bounded to HF SDPA eager with materialized shared DFlash hidden.
The known `300/303` emitted-bonus parity prevents a lossless deployment claim.

