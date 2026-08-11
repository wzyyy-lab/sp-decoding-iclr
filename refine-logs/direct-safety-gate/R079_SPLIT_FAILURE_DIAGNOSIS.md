# R079 Split Failure Diagnosis and Row-Split Rescue

**Date:** 2026-08-05  
**Failed job:** `10135740`  
**Stage:** identity-only Phase-3 split materialization  
**Disposition:** fail closed; no split artifact was published and no downstream stage was opened

## Observed failure

All source, canonical-metadata, and exclusion-file identity checks passed. The
materializer then stopped before its atomic output write with:

```text
ValueError: Phase-3 prompts overlap validation: [...]
```

Slurm recorded `FAILED`, exit `1:0`, after 23 seconds. The requested
`artifacts/pros_gate/r079/split_manifest.json` did not exist after failure.

## Root cause

`phase3_development_v3.jsonl` is a combined manifest, not a validation-only
file. Its frozen row census is:

| row `split` | rows |
|---|---:|
| `train` | 2,000 |
| `validation_gate` | 150 |
| `validation_select` | 150 |

The failed implementation used every `sample_id` in that file as a validation
exclusion. The canonical Phase-3 source's 1,987 prompts are drawn from its
`train` rows, so all 1,987 were incorrectly reported as validation overlap.
This contradicted the already frozen proposal and experiment plan, which name
only `validation_gate` and `validation_select` as validation exclusions.

## Binding repair

Exclusion identity now consists of all of the following:

1. resolved file path, byte size, and SHA-256, verified before JSONL parsing;
2. an exact, frozen `selected_splits` list;
3. the exact full-file `row_counts_by_split` census.

The filters are:

| role | selected rows | frozen full-file census |
|---|---|---|
| `producer_train` | `train` | `train=100000` |
| `validation` | `validation_gate`, `validation_select` | `train=2000`, `validation_gate=150`, `validation_select=150` |
| `reserved` | `test` | `test=600` |

The materializer and independent auditor implement this parsing separately.
Both reject missing/unknown row fields, empty selected subsets, duplicate
selected prompts, missing selected splits, census drift, file identity drift,
and cross-file duplication. Persisted provenance contains both the filter and
census. Semantic exclusion hashes and the stdlib-only downstream receipt
verifier bind those fields, so a receipt produced under the old whole-file
semantics cannot authorize a later stage.

## Read-only real-identity check

The repaired loader produced:

| quantity | observed |
|---|---:|
| Phase-3 prompts | 1,987 |
| Phase-3 blocks | 15,886 |
| producer exclusions | 100,000 |
| validation exclusions | 300 |
| reserved exclusions | 600 |
| overlap with producer | 0 |
| overlap with selected validation rows | 0 |
| overlap with reserved | 0 |

Semantic exclusion SHA-256 values are:

- producer: `dcd1decfa63d17b4f4ee180a2d30e774ffb87bc9eed96a956f045a117039b16d`
- validation: `fc336fa8672140facd82dc6f73be067c02d87e27bb6e276137a290a16cc7ab09`
- reserved: `94c3e4274af5f310042766f962fcc3bf57b854d9b1f3e99bec950ddb367b4885`

This is an implementation correction to the preregistered exclusion roles. It
does not change the canonical source, prompt-domain allocation algorithm,
per-domain fit/checkpoint/falsifier counts, split seed material, or any model,
objective, threshold, or evaluation rule.

## Verification and launch boundary

- Complete first-party source closure: 57 files, manifest SHA-256
  `513ad34d8a71cd4bb340eaeda2dd8132be311a38f075d2148af2dadf7ef05a53`.
- Focused repaired surface: 20 tests passed.
- Full CPU suite: 302 tests and 3 parameterized subtests passed.
- Python compilation and all five Slurm-wrapper syntax checks passed.
- A fresh failure-rescue reviewer must return GO before one split resubmission.
- That GO may authorize only the split resubmission. Fit/checkpoint outcomes,
  capacity materialization, and capacity training remain behind their existing
  independent artifact receipts. Falsifier, validation/formal evaluation,
  refits, extra seeds/widths, calibration, and clean training remain closed.
