# R053 Fast-Beam CUDA-Graph Profile

## Outcome

**PASS-GRAPH-FEASIBILITY / NOT YET A COMPLETE-CYCLE PASS.**  On the same A40
class used by R053, CUDA Graph reduces the fixed Fast-K64 beam from about
`15.98 ms` to `3.58 ms` at W16 while preserving every returned tensor exactly.
This removes most of the observed eager implementation penalty and justifies
the R055 fixed-forest experiment.  It does not by itself establish target-tree
latency, end-to-end throughput, or lossless serving.

Raw artifact: `profile_output/r053_beam_graph_10165436.json` (job 10165436,
NVIDIA A40, median fixed context length 163, 25 warmups / 200 repeats).

## Beam latency

| Beam width | Eager p50 | CUDA-graph p50 | Speedup | Fixed shared-anchor forest rows |
|---:|---:|---:|---:|---:|
| 4 | 15.9360 ms | 3.4468 ms | 4.62x | 65 |
| 8 | 15.9662 ms | 3.4970 ms | 4.57x | 129 |
| 16 | 15.9831 ms | 3.5840 ms | 4.46x | 257 |

Graph/eager parity passes for path tokens, edge log-probabilities, MAP scores,
the protected Fast-K64 trunk, and the K16 candidate IDs.  Every width also
reproduces the independent batch-1 Fast-K64 trunk.

The small W4-to-W16 latency increase (`0.1372 ms`) shows that sequential kernel
launch/GRU depth, rather than beam batch width, dominates this implementation.

## Relation to the system budget

For R053 N64 EAL `8.483722`, the final 1.15x target permits a complete cycle of
at most `38.5705 ms`.  Holding R053's separately measured base GEMM, N64 target
forward and traversal fixed gives a component estimate of `40.0169 ms` after
the graph beam, or about `1.108x` throughput.  Thus graph capture alone does not
make N64 pass.

At the full-W16 structural ceiling `9.129130`, the output ratio rises to
`1.2225x`; under the deliberately optimistic N64 target/traversal cost, the
same component estimate would imply about `1.184x`.  R055 must replace this
mixed upper bound with actual W4/W8/W16 forest EAL and dependent complete-cycle
timing.

## Instrumentation changelog

| File | Change | Purpose |
|---|---|---|
| `src/sph/r053_tree.py` | Removed device-to-host protected-trunk branches and dynamic `nonzero` from the W16 beam | Make the exact existing beam CUDA-graph safe without changing scores or tokens. |
| `scripts/profile_r053_beam_graph.py` | Added W4/W8/W16 eager/graph timing, five-tensor parity and exact latency budgets | Quantify whether the R053 eager bottleneck is removable. |
| `scripts/slurm/r053_beam_graph_profile.sbatch` | Added A40-pinned reproducible launcher | Keep hardware and timing protocol fixed. |

The raw profile's diagnostic row label used `17W` while the selected R055
forest shares one anchor and uses `1+16W`; this label did not enter any timing or
gate.  The source script now reports the correct shared-anchor row counts.
