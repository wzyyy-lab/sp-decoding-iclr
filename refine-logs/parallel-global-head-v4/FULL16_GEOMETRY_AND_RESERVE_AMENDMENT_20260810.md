# PARC-16 Full16 Geometry and Eligibility-Reserve Amendment

**Status:** authoritative implementation correction before any GPU submission.

This amendment closes two code-review blockers without changing the PARC head,
loss, optimizer, active train/validation/held-out sizes, or success gates.

1. The released pure `Qwen3-4B-DFlash-b16` checkpoint is non-shift-label. Its
   native raw16 path contains one anchor carrier plus fifteen prediction rows.
   PARC requires sixteen prediction positions, so every PARC pure-DFlash path
   uses the explicit extended geometry `[anchor] + 16 masks -> raw17 -> rows
   1..16`. The head still receives exactly sixteen prediction states and emits
   one `[B,16]` chain in one global non-causal invocation. Released Domino stays
   on its native shift-label raw16/all-16-rows geometry. Final timing must charge
   PARC for the extra pure-DFlash carrier row; the extension is not described as
   native released-DFlash-b16 execution.

2. A prompt is eligible only if ordinary pre-EOS target generation supplies all
   129 continuation tokens required by the frozen eight-anchor schedule. Target
   generation never continues past EOS and shorter continuations never receive
   densely packed substitute anchors.

3. To preserve exact active cardinalities, the same cleaned Open-PerfectBlend
   source and prior-development exclusions are used to preassign a deterministic
   270K candidate reserve pool before any eligibility check: 90K candidates per
   domain, partitioned into mutually disjoint train/validation/held-out reserve
   sequences. Collection promotes candidates only inside their preassigned
   split and domain until the active sets contain exactly 90K train and 5K
   validation prompts. The held-out candidate sequence remains unmaterialized
   until the validation-selected checkpoint is locked; the one formal held-out
   job consumes it until exactly 5K eligible prompts are obtained.

4. The 270K number is a pre-label candidate pool, not a training size and not an
   efficacy experiment. The scientific run remains exactly 90K train prompts,
   5K validation prompts, and 5K sealed held-out prompts.

5. Step 0 is parity-only and selection-ineligible. Only trained checkpoints at
   steps 10K through 180K can authorize held-out evaluation. Constraint
   infeasibility and completed runs are terminal and cannot be auto-resumed.

