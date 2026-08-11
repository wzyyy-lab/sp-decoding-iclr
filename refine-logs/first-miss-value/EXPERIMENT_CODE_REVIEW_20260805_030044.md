# SAVS Gate-1 Experiment Code Review

**Timestamp**: 2026-08-05T03:00:44+08:00  
**Review mode**: fresh GPT-5.6-Sol, xhigh, same-family provisional  
**Verdict**: **GO**

GO authorizes exactly one D64/H4/L1 axial-additive, K16, batch-32,
seed-0, 512-block capacity job. It does not authorize full data, D640,
additional seeds, `validation_gate`, formal evaluation, calibration, or
threshold tuning.

## Blocking Findings

None.

## Non-Blocking Caveat

The Gate-1 loader deserializes the mixed canonical root before selecting the
frozen train subset. The optimization and evaluation set is nevertheless
exactly the 512 `train` records in the frozen manifest, and no
`validation_gate` record enters the model or any metric. This accepted
exception is limited to the same-subset capacity probe. Gate 2 continues to
require the physically isolated `validation_select` collection.

## Independent Correctness Checks

- Exhaustively compared all `512*225=115,200` targets with token-ID
  brute-force one-edit decoding: exact equality, maximum error `0.0`.
- Reconstructed 256 beneficial, 57,179 neutral, and 57,765 harmful actions.
- Verified 512 blocks, 459 prompts, train-only selected records, prompt hash
  `1e2be08968b2356f71e9818a5be5b8f3ecdd12ee50299ba6212a035f8a4d2707`,
  and exactly 5,120 optimizer steps.
- Verified residual differences, exact epoch-zero DFlash identity,
  strict-positive KEEP, uniform 225-action MSE, first/second-backward
  propagation, and class-gradient normalization.
- Verified every frozen metric denominator, prompt-balanced EAL/oracle
  recovery, all six inclusive gates, exact 256-positive count, and earliest
  exact MSE-tie retention.
- Nonfinite loss/gradient and identity violations fail closed.
- CUDA/autocast, serialization, and memory behavior present no blocker.

## Validation Rerun

- Focused tests: `11 passed`.
- Full suite: `188 passed, 3 subtests passed`.
- Python AST compilation: passed.
- `bash -n scripts/slurm/savs_capacity.sbatch`: passed.
- Manifest reconstruction and every pinned hash: passed.

## Frozen SHA256 Identities

| File | SHA256 |
|---|---|
| `src/sph/first_miss_value_selector.py` | `caa9c5a611e7ead6d880c876e0a002b544115fc57dd2e43d46987b64195a82d2` |
| `scripts/train_first_miss_value_selector.py` | `4b3b02ffc3c413034bf445d3352e09b24e68ccb10fb16c9f63e07240818ad483` |
| `tests/test_first_miss_value_selector.py` | `ede44bc0001a936ad5c99577a224f342e08a685a6ee6b95344320e75d1c8ec90` |
| `tests/test_first_miss_value_training.py` | `691e2104bdec75e034034f05de3f57431f49aac66b59e8c7efb177fc470b1f7e` |
| `scripts/slurm/savs_capacity.sbatch` | `2f30791efdd32324de1fea283923a42d906289873b63cc4ba47133275506c9d3` |
| `src/sph/global_direct_selector.py` | `f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06` |
| `scripts/train_global_direct_selector.py` | `e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110` |
| `src/sph/first_miss_action_selector.py` | `332fac8948c61b2287cb861988b700c8a249a91cec6f808c0010d47fe7260cef` |
| `scripts/train_first_miss_action_selector.py` | `2e27b4500fdd6d440078e40378c2aa56c09f09227072d7ffdc48bfaafddbcd10` |
| `src/sph/first_miss_capacity.py` | `58a92a38572dfca983361e257194a5d76f0a14f7eb9a196b8d7e749f584c7238` |
| `src/sph/data.py` | `c811701d0ec097afa86e594946857290bcfff80e2cfc2e8638f0bcdaffcc0742` |
| capacity manifest | `d60613a00fc8557f4ff227ec302ced42de6a071d030b7ae7eb9eb5120bf5b67f` |

Reviewer changed files: no. Reviewer submitted jobs: no.
