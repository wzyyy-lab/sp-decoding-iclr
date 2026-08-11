# PGCF-16 Gate0 Code Review

**Review time:** 2026-08-10 12:13:46 +0800  
**Reviewer:** fresh secondary Codex reviewer, same-family provisional  
**Verdict:** **GO for PGCF-002 A40 mechanics smoke only**  
**Blocking findings:** none

## Frozen compliance result

- The primary head consumes exactly `H[B,16,2560]` plus the pure base
  `Top16` lattice and shared embeddings.
- All `16 x 16 = 256` candidate nodes use two layers of unmasked global
  self-attention.  The claim-bearing class cannot select a causal or local
  scope.
- One invocation returns `[B,16,16]`; one tensor argmax returns exactly one
  `[B,16]` proposal chain.
- The online forward signature has no gold, policy, target feature,
  previous-token, or selected-token input.  There is no recurrent rollout,
  sequence search, iterative refinement, extra target inference, or
  multi-proposal verification.
- Exact default trainable parameters: `2,438,400`.
- Zero-init output exactly reproduces base Top-1.
- Raw-embedding and derived preprojected-table paths match in BF16.

## Loss and evaluation result

- Prefix loss uses safe ranks before gather and censors the suffix after the
  first unsupported gold token.
- Target loss is `KL(p_target || q_head)` on the clean gold-supported prefix;
  empty masks return exact zero.
- Teacher imitation ignores unsupported actions without expanding the online
  candidate set.
- All 16 positions are retained by the trainer and L15 records fail closed.
- Independent canonical recomputation matched the frozen references:
  base `6.068513119533527`, released Domino `7.239552964042760`, and pure
  base-Top16 oracle `10.909256559766764`.

## Fair profile result

The complete PGCF callback includes the base vocabulary GEMM, FP32 Top-16,
projected-table gather, global head, and final argmax.  The incremental
callback also includes the projected-table gather.  Released Domino uses the
same hidden states, target embedding/output basis, anchor, BF16 precision,
batch size, and eager execution.  Cached Top-16 and Domino-policy replay are
fail-closed checks.

## Verification performed

- focused PGCF tests: `22 passed`;
- Python parse/compile checks: passed;
- four Slurm launchers: `bash -n` passed;
- BF16 forward/backward finite: passed;
- exact parameter and attention-index checks: passed.

The repository-wide suite has three unrelated legacy source-closure failures,
so this artifact does not claim the whole repository is green.

## Authorization boundary

This review authorizes only `scripts/slurm/pgcf16_gpu_smoke.sbatch`.
Capacity, profile, disjoint screen, large-data collection, and SGLang work
remain blocked until their preceding gates pass.  The remote-intervention
diagnostic is still required before a complete global-mechanism Gate2 claim.
