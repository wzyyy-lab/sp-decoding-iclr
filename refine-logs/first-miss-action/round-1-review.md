# Round 1 External Review

**Score:** 7.6/10  
**Verdict:** REFINE  
**Implementation authorization:** CPU semantics only; no training/GPU gate yet.

## Blocking findings

1. `validation_select` was proposed for checkpoint selection, seed-0 routing,
   and bootstrap confirmation.  Selection-aware intervals on the same prompts
   are not confirmatory.  The method must freeze before a genuinely untouched
   evaluation, and final harm must use a one-sided prompt-cluster UCB.
2. Ordinary action CE imitates one canonical optimal action but is not an exact
   expected-EAL risk surrogate: some actions tie in realized EAL, and wrong
   actions have unequal regret.  The claim language must be narrowed.
3. The training-objective claim lacks the causal decoder control.  The matched
   direct checkpoint must be evaluated both with its native unconstrained path
   and with FMAS's exact KEEP/max-margin one-edit decoder; FMAS training then
   uses that same decoder.
4. Offline canonical EAL does not establish on-policy throughput because a
   changed accepted length changes later anchors/contexts and the selector has
   latency.  A frozen end-to-end rollout is required before a deployable claim.
5. Acceptance-aware training and path selection have close prior work.  The
   plausible novelty must be narrowed to a DFlash-specific, base-preserving,
   one-edit intervention policy with exact identity initialization, conditional
   on beating the matched one-edit decoder control.

## Nonblocking requirements incorporated in the revision

- Pin the capacity subset identity and KEEP/edit composition before launch.
- Define the single-edit oracle denominator and prompt aggregation formally.
- Add exhaustive small-L/K property tests and tie cases.
- Split KEEP-full-correct from KEEP-out-of-K diagnostics.
- Add gain-weighted repair recall.
- State that inference uses frozen DFlash features and a frozen Qwen target
  embedding lookup, but no target forward or target label.

## Important repository-specific correction

The review suggested using `validation_gate` as the untouched confirmation
split.  Repository history shows that this split was already inspected in the
Phase-3 failure analysis (`docs/phase3_failure_analysis.md:363-365`), so it is
not fresh.  The revision therefore applies the review's principle more
strictly: neither `validation_select` nor the contaminated `validation_gate`
can carry confirmation.  The pre-existing 600-prompt reserved formal-test
manifest remains `reserved_unobserved` and is the only claim-grade holdout.

