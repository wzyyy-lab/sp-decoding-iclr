# Authoritative Deployment Constraint Correction

The user clarified the intended comparison after the initial method review.

## Correct interpretation

- During research prototyping, compare eager PLC with eager Domino. CUDA-graph/Triton numbers are useful engineering evidence but are not the sole method gate because the final method will be integrated into SGLang and both systems must then receive comparable serving optimization.
- “Lightweight” means the **new encoder is small relative to the complete DFlash draft model**, and its added per-round cost must not erase the benefit of higher acceptance. It does not require PLC's isolated head to be smaller or faster than Domino's isolated head under every prototype implementation.
- The final outcome gate is same-stack, same-hardware SGLang end-to-end throughput, not an isolated microbenchmark.

## Revised quantitative contract

- DFlash/Domino checkpoint total: 588.250M parameters.
- Released correction head: 50.823M; draft model excluding that head: 537.427M.
- Initial PLC encoder target: about 2% or less of the headless DFlash draft model (10.75M parameters). This is a starting budget, not a hard cap: if acceptance is capacity-limited, width/depth/modes may increase as long as added latency remains compatible with the final 1.15x TPS target. Reused `W_h/W_out` are still counted when reporting the complete active head.
- Prototype latency diagnostic: compare eager-to-eager and graph-to-graph separately. A complete PLC head no more than about 1.2x the corresponding Domino head is acceptable for development, provided projected end-to-end throughput remains positive.
- Acceptance target: at least 1.15x released Domino EAL on frozen evaluation, with the measured Top-16 oracle (`7.240 -> 10.254`, +3.015) defining the ambitious ceiling.
- Final system target: at least 1.15x released Domino end-to-end TPS after comparable SGLang integration; higher is preferred.

The former `PLC head <=0.8x Domino head` condition is retained only as an achieved optimization result for v1, not as the user's architectural requirement.
