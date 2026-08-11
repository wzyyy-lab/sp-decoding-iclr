# PGCF-16 Gate1 Capacity Result and Diagnosis

## Final verdict: PASS

Full Gate1 is the conjunction of an independent architecture-capacity witness
and an independent supported-Domino-action witness.  Both have now passed.

## Teacher-only witness — PASS

Job `10166815`, same global d256/L2/H8 head and fixed 512 records:

- selected teacher checkpoint: step `2600`;
- supported Domino-action accuracy: `99.8132337%`;
- required threshold: `99%`;
- active teacher gate: passed.

The run used teacher CE only for every update.  Prefix and target-KL values in
the log were diagnostics with zero loss weight.

## Frozen curriculum target witness — intended-loss diagnostic / FAIL

Job `10166814`, 8,000 steps; checkpoint selected only by maximum
prompt-balanced EAL:

- selected step: `7600`;
- EAL: `10.912109375`;
- pure base16 oracle: `11.021484375`;
- oracle-gap recovery: `98.1239531%` — pass;
- harmed fraction: `0%` — pass;
- prefix-reachable candidate accuracy: `99.0785043%` — diagnostic pass;
- all-supported candidate accuracy: `97.8352463%` — below 99%;
- all-supported hard/non-Top1 accuracy: `95.8705701%` — below 97%.

The run is therefore a formal target-capacity failure despite near-oracle EAL.
It is not reinterpreted as Gate1 success.

## Why the remaining failure is not yet an architecture-capacity result

On the fixed 512 records:

- all gold-supported rows: `6,883`;
- prefix-supported rows receiving the main objective: `5,643`;
- hard supported rows: `3,245`;
- clean target-KL rows: `6,677`;
- supported rows receiving neither direct prefix nor clean-KL supervision:
  `103`, including `88` hard rows.

The selected checkpoint has about 149 all-supported errors and 134 hard
errors.  Passing the two gates requires removing about 81 and 37 errors,
respectively, both within the size of the deliberately unsupervised set.
This result remains recorded as a failure of the all-supported token gates
under the intended masked curriculum; it is not rewritten as success.

## Independent dense-gold architecture witness — PASS

Following fresh review, job `10166838` trained the same global d256/L2/H8 head
from zero on all supported gold rows using capacity-only safe dense CE.  It did
not use teacher CE, prefix loss, target KL, an online target feature, or a
different inference path.

The prompt-balanced maximum-EAL checkpoint was step `7600`.  All four target
capacity conditions hold on that one checkpoint:

- EAL: `11.021484375`, exactly the pure base16 oracle;
- candidate accuracy: `99.9854715%`;
- hard/non-Top1 accuracy: `99.9691834%`;
- oracle-gap recovery: `100%`;
- harmed fraction: `0%`.

Combined with the independent teacher-only accuracy `99.8132337%`, this makes
the complete Gate1 capacity verdict **PASS**.  The interpretation is limited:
d256/L2 has sufficient same-set representation capacity.  The claim-bearing
held-out recipe remains the frozen curriculum, and no held-out benefit is
implied by the dense-CE witness.
