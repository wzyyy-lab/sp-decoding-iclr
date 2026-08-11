# FMAS Gate-2 Experiment Code Review

**Final verdict:** GO, narrowly scoped to seed-0 FMAS development checkpoint
production.  No Gate-2 result claim is authorized until the exact
Direct-native, Direct-one-edit, and FMAS artifacts all exist and pass the
frozen comparisons.

**Review mode:** independent read-only GPT-5.6-Sol reviewer, with two
fail-closed remediation rounds.  No reviewer changed files or submitted jobs.

## Initial blockers and remediation

The first review found that the mixed canonical collection loaded
`validation_gate` records before logical filtering, and that `src/sph/data.py`
was not pinned despite controlling shard integrity, splits, and collation.
The remediation materialized a physically isolated 147-prompt / 1,175-block
`validation_select` artifact, made external-training FMAS reject every mixed
split, and added `data.py` to start/end source hashes and snapshots.

The second review constructed a self-consistent but wrong Direct run that the
first one-edit evaluator accepted.  The final evaluator now requires all 42
non-dynamic Direct parser fields, exact config output/run identity, D64 model
and Candidate-D-PACE settings, 99,356 prompts / 793,989 blocks / 37,221 steps,
the prompt hash, 147/1,175 validation cardinality, all eight ordered train-part
paths and hashes, source-data identity, and Direct trainer/head hashes both at
run start and end.  Wrong architecture, inactive defaults, budget,
cardinality, data, shard, source, and output-path mutations fail closed.

## Frozen reviewed identities

| artifact | SHA256 |
|---|---|
| FMAS trainer | `2e27b4500fdd6d440078e40378c2aa56c09f09227072d7ffdc48bfaafddbcd10` |
| FMAS head/wrapper | `332fac8948c61b2287cb861988b700c8a249a91cec6f808c0010d47fe7260cef` |
| capacity helper | `58a92a38572dfca983361e257194a5d76f0a14f7eb9a196b8d7e749f584c7238` |
| canonical data helper | `c811701d0ec097afa86e594946857290bcfff80e2cfc2e8638f0bcdaffcc0742` |
| FMAS Slurm job | `c5784eac60f1445c32b61056cc028f2b57574a4fa248415e68b19347db8f8942` |
| Direct-one-edit evaluator | `802abd7fd8715e67a6b2cab5f33056a9ce5e17fb9af34723b0dd080850c450fd` |
| evaluator Slurm job | `b6a1e13086abfb105f90dba684e0ac7644d0d23fd307b5ac0f46668ba1b569b9` |
| isolated metadata | `b63be7bbfd56651aadbee57a819bfe0afb39395b1601b5ea4fc1564cc9f933d7` |
| isolated manifest | `1496caa3d71ce64de9cd3fc2c29e40be60e9b636a988c9b400a0712e3ee5e811` |
| prelaunch amendment | `9cda71223ba32dd91c5769264d2e28d8cc2c879eef3390db75af141c799ce9dd` |
| pinned Direct trainer | `e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110` |
| pinned Direct head | `f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06` |

## Independent verification

- All 72 mixed-source shards were re-hashed and all 1,175 selected records
  matched the source recordwise, including tensors.
- The isolated collection has five regular, single-link shards; its only split
  is `validation_select`, with chat/code/math block counts 383/392/400.
- Full suite: 177 passed plus 3 parameterized subtests.
- Evaluator-focused suite: 6 passed, including adversarial identity cases.
- FMAS/evaluator Python compilation and both Slurm syntax checks passed.
- No full-data FMAS job or result existed before the checkpoint rule and
  isolation amendment were frozen.

## Authorization boundary

This GO authorizes only the frozen D64/H4/L1 axial-additive, K16, batch-64,
three-epoch, seed-0 FMAS run over the full OPB collection.  It does not
authorize seeds 1/2, formal-test access, threshold changes, post-hoc decoder
selection, or a Gate-2 success/failure claim before the exact artifact triplet
and evaluator identity checks are complete.
