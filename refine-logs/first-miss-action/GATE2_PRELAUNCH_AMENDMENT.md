# FMAS Gate-2 Prelaunch Amendment

Frozen before any FMAS full-data development job was submitted or observed.
The completed 512-block capacity result is the only FMAS GPU result available
at this point; this amendment is therefore not outcome-adaptive.

## Physically isolated selection data

The FMAS training process may access only a materialized
`validation_select` collection, never the mixed source collection containing
`validation_gate`.  The isolated artifact has exactly 147 prompts and 1,175
blocks, with split counts chat 383, code 392, and math 400.  Its identities are:

- metadata SHA256: `b63be7bbfd56651aadbee57a819bfe0afb39395b1601b5ea4fc1564cc9f933d7`;
- prompt-set SHA256: `278c27e266e50c6b81b94a88bd8dbf5dc2645563add738db7536f2489a01edaa`;
- selected-manifest SHA256: `1496caa3d71ce64de9cd3fc2c29e40be60e9b636a988c9b400a0712e3ee5e811`;
- source metadata SHA256: `0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320`.

The materializer verifies every source shard's byte count and SHA256, checks
record domain/split against the original manifest, and records all source and
output identities.  The FMAS trainer rejects external-training development if
its `--data` collection contains any split other than `validation_select`.

## Exact checkpoint rule

The development checkpoint key remains the rule already stated in the initial
proposal: lexicographically maximize

```text
(raw prompt-balanced FMAS EAL, -block harmed fraction, action accuracy)
```

The update condition is strict `>` over epochs visited in ascending order, so
an exact three-way tie retains the earliest epoch.  No calibration, repair
recall, oracle statistic, domain statistic, or Direct-control result enters
checkpoint selection.

## Control completion rule

The FMAS run produces a checkpoint but cannot by itself complete Gate 2.  The
gate remains closed until all three frozen paths are evaluated on the isolated
artifact:

1. Direct-native from the matched D64 Candidate-D-PACE checkpoint;
2. Direct-one-edit from that identical checkpoint, decoded as KEEP versus the
   single globally highest positive candidate-over-base margin;
3. FMAS decoded by the identical one-edit action rule.

`scripts/evaluate_direct_one_edit.py` reconstructs the frozen Direct model,
requires checkpoint/config/epoch/parameter identities to match its metrics,
requires the isolated artifact to descend from the Direct run's source data,
and fails unless recomputed DFlash and Direct-native summaries exactly match
the original Direct report.  Only after this identity check may the frozen
Gate-2 thresholds in `round-1-refinement.md` be applied.  The sealed
`validation_gate` and formal test remain unopened.
