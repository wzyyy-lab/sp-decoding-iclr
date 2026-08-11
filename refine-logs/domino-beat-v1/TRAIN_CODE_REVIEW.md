# Cached head implementation review

Fresh-agent review confirmed the released shift-label alignment, teacher and
on-policy GRU states, frozen vocabulary projection, BF16 live parameters with
FP32 optimizer masters, prompt-balanced selection, and fail-closed accepted-
length replay.  Margin top-2 competitor gradients were also checked.

Two substantive corrections were made after the completed objective screen:

- standard DECAY-CE and D-PACE controls no longer use the method-specific hard
  base-reachability mask; reachable masking remains on AUF/breaker objectives;
- L2-SP now uses sum-of-squares rather than a 12M-parameter mean, so coefficient
  `1e-3` has a real regularizing scale.

The already-completed 13-run screen is therefore interpreted as an effectively
unregularized head objective screen.  Its best result was DECAY-CE
`7.039845` (`+0.024052`, interval crossing zero), which closes head-only tuning
for the hard performance target.
