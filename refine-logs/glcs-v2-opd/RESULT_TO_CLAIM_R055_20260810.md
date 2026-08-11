# R055 Result-to-Claim Review

**Verdict:** `claim_supported = yes` for the bounded development claim.  Freeze
W8/N129 and proceed to SGLang.  This is not yet a deployment or end-to-end
throughput claim.

## Supported result

- W8 actual clean EAL is `8.667395529640428`, versus same-job released Domino
  `7.285471331389699` (`+1.381924198250729`).
- Every fixed development domain improves: chat `+0.8337053571428572`, code
  `+1.5051020408163263`, and math `+1.7874999999999996`.
- Strict clean output advance is `9.634232264334305`, versus Domino
  `8.285471331389699`.
- On the same A40 median-context profile, the complete non-common cycle is
  `35.57017707824707 ms`, versus Domino `39.55148696899414 ms`; the resulting
  development projection is `1.2929349694313073x`.
- W4 misses the frozen accuracy threshold (`8.279640427599611`), while W16
  misses the latency threshold (`0.8966667256652587x`).  W8 is therefore the
  unique and smallest joint Pareto point.

## Claim boundary

The evidence supports implementing the fixed B16, batch-one, Qwen3-4B W8
forest in SGLang.  It does not establish lossless serving or final throughput.
The HF reference still reports 39/1,175 emitted-output mismatches, selected
unique-winner parity 533/551, stable-row parity 10,391/10,397, and full-accept
bonus parity 385/391.  These are diagnostic for R055 but become hard gates in
the serving implementation.

The final claim additionally requires an end-to-end A40 throughput ratio whose
prompt-bootstrap 95% confidence-interval lower bound is at least `1.15x`
against an in-engine released-Domino baseline.  A single median-context p50
projection cannot satisfy that gate.

## Authorized next action

Implement only W8/N129 with one target forest verification call, exact
stable-non-tie branch/output/bonus parity, and correct paged-KV commit.  Close
the route if serving parity or the final throughput confidence gate fails.

This review is same-family provisional.  No integrity-forensics result was
available; the review evaluated semantic and numerical claim support.
