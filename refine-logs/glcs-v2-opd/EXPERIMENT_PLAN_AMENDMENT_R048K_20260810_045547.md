# R048K Amendment: Reachability Refinement

**Date:** 2026-08-10 04:55 CST  
**Parent:** `EXPERIMENT_PLAN_AMENDMENT_R048.md`

The valid batch-1 Fast-K32 result is:

- proposal EAL `7.258989310`;
- one-repair oracle `8.385447036`;
- unrestricted one-repair oracle `8.466472303`;
- frontier count coverage `821/900 = 91.22%`.

K32 exceeds the hard target but misses the preregistered `8.40` safety gate.
It would require `94.68%` recovery of its candidate-constrained oracle gain,
which leaves too little learned-error and harm budget.  K32 training is
therefore prohibited.

R048K evaluates exactly K64 and K128 under the same batch-1 proposal and fixed
B16 evaluation contract.  Candidate width changes neither the 180,224 tuned
lens parameters nor the single base-vocabulary GEMM.  For each K, report its
own candidate-only proposal, constrained one- and two-repair oracles, frontier
count coverage, and reward-weighted coverage against the unrestricted oracle.

Choose the smallest K satisfying all efficacy conditions:

- one-repair oracle EAL at least `8.325485908649174`;
- required oracle-gain recovery at most `90%`;
- frontier count coverage at least `95%`;
- oracle reward coverage at least `95%`.

K64 passing selects K64 and K128 remains diagnostic only.  If K64 fails and
K128 passes, select K128.  If K128 fails, close one-repair candidate expansion;
do not test K256.  Before any capacity training, the selected K must also add
at most `0.10 ms` candidate-path p50 versus K32 and the measured ideal
perfect-repair layer-split throughput must be at least `1.20x` Domino.
