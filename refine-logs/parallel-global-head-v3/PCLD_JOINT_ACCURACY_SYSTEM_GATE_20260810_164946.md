# PCLD-16R Joint Accuracy–System Gate

Accepted draft length and end-to-end throughput cannot be gated independently.
For one speculative cycle, useful output is proportional to EAL + 1. Therefore a
necessary same-workload condition for 1.15x throughput over Domino is

\[
\frac{T_{\mathrm{PCLD}}}{T_{\mathrm{Domino}}}
\le
\frac{\mathrm{EAL}_{\mathrm{PCLD}}+1}
{1.15(\mathrm{EAL}_{\mathrm{Domino}}+1)}.
\]

Using the current development Domino EAL 7.2395529640:

- The formal 1.15x accepted-length floor 8.3254859086 gives only a 1.131795 output ratio. It would require the entire PCLD cycle to be at least 1.55% faster than Domino.
- Equal cycle time requires PCLD EAL at least 8.4754859087.
- PCLD EAL 9.0 gives a 1.213658 output ratio and allows the total cycle to be at most 1.055355x Domino while still reaching 1.15x throughput.

Consequently:

1. 8.3254859086 remains the immutable minimum accepted-length claim gate.
2. 9.0 is the design target before SGLang integration, because it leaves realistic head-cost room.
3. Every complete-cycle profile must report the right-hand-side latency budget from its measured EAL; a head-only 1.20x guide cannot authorize a systems claim.
