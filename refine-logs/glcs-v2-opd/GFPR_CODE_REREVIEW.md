# Fresh GFPR Remediation Re-review

**Reviewer:** fresh GPT-5.6-Sol xhigh agent (same-family provisional)  
**Date:** 2026-08-10  
**Verdict:** **READY_FOR_GATE_B**

No claim-bearing code or execution blocker remains in the remediated implementation. The current Gate-B path measures disjoint held-out fixed-anchor EAL correctly and has completed on A40 without OOM or timeout.

## Confirmed resolutions

- Train/eval prompt overlap now aborts unless explicitly marked capacity-only; capacity-only reports suppress `passed`.
- Released and adapted dynamic policies traverse their own `r+1` chains and are paired only at prompt level, never by zipping blocks.
- Claim-bearing dynamic comparison rejects identical policy versions and requires an existing adapted checkpoint.
- Repeatable rollout inputs, source-specific prompt keys, equal source counts, and `--initial-adaptation` implement v0/v1 50/50 replay.
- Refreshed train and fixed-eval rollouts must come from the requested initial policy.
- Gate A combines semantic invariants, oracle headroom, historical EAL tolerance where applicable, and nonzero failure exit.
- Splits are explicit; `validation_gate` remains sealed without a deliberate final override.
- Rollout target/Domino paths and saved adaptation provenance are validated on claim-bearing loads.
- Prompt accumulation uses the actual number of source-prompts, including partial final groups.
- Position-zero accuracy, repairs, harms, changed decisions, alpha, and gradient diagnostics are reported.
- The unused FP32 base-logit return was removed from normal training; nonfinite loss/gradient norms fail closed.

## Behavioral evidence checked by the reviewer

- 11 focused tests passed.
- All seven GFPR Slurm scripts passed `bash -n`.
- Python artifacts passed syntax parsing.
- Disjoint GPU smoke 10164017 correctly reported insufficient efficacy rather than a false pass.
- Gate-B Fixed-15 job 10164027 completed 375 steps on A40, exactly reproduced released EAL 7.239552964 at step 0, and correctly failed with best EAL 7.243926.

## Operational caveat

Arms launched before the launcher was changed from 2,000 to 10,000 bootstrap draws retain 2,000 draws in their process arguments. Any selected checkpoint used for the formal Gate-B CI must be reevaluated with 10,000 draws or relaunched.
