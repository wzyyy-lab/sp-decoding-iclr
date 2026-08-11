# Experiment Tracker：JAPD-16

| Run ID | Milestone | Purpose | System / Variant | Split | Decisive metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| J000 | M0 | objective/metric/invariant CPU contracts | JAPD reference | synthetic + R047 validation | inclusive J2 parity；certificate inequality；full16 shape | MUST | DONE | 45 focused tests + 3 subtests；real audit exact `745/15/0/207`；433,852 params；global remote delta .00549/local 0；step0 identity PASS |
| J001 | M0 | materialize/replay LSE smoke | sidecar builder | R047 32 blocks | Top16 IDs/logits、LSE/scalar/token parity | MUST | DONE | job `10167503` A40 `COMPLETED 0:0`；32 records；LSE/scalar/score零误差；token mismatch=0 |
| J002 | M0 | dataflow/leakage/split audit | loader + manifests | R047 train | fit/select/diag disjoint；head batch字段whitelist | MUST | DONE | `1589/199/199` prompt严格不交；15,886 blocks；forbidden online字段零泄漏 |
| J010 | M1 | D64 same-set capacity | global-JAPD D64 | stratified 512 blocks | J2≥99%；oracle recovery≥95%；harm≤1% | MUST | DONE_FAIL | job `10167565`；J2 `100%` PASS，recovery `92.7966%` FAIL，harm `1.3672%` FAIL；双失败规则的一半 |
| J011 | M1 | full-fit optimization gate | global-JAPD D64 | disjoint 512 fit prompts | J2≥90% AND oracle recovery≥80% | MUST | DONE_FAIL | job `10167566`；J2 `17.4518%`、recovery `4.0101%`，均FAIL；与J010共同触发预注册D256 |
| J012 | M1 | eager fairness profile | D64 JAPD vs Domino | A40 batch1 | complete p50≤1.20x Domino；p90/memory | MUST | DONE_PASS | hardened job `10167573`；保守complete p50 ratio `0.67345`；same-call分段可加；A40 eager PASS |
| J010-D256 | M1 | conditional same-set capacity | global-JAPD D256 | same frozen 512 blocks | identical J010 gates | MUST | DONE_FAIL | job `10167607`；J2 `100%` PASS，recovery `94.2090%` FAIL，harm `1.1719%` FAIL；best step7000 |
| J011-D256 | M1 | conditional full-fit gate | global-JAPD D256 | same frozen 512 prompts | identical J011 gates | MUST | DONE_FAIL | job `10167609`；J2 `24.6136%`、recovery `5.5133%`，均FAIL；best step1500 |
| J012-D256 | M1 | conditional eager profile | D256 JAPD vs Domino | A40 batch1 | identical J012 gate | MUST | DONE_PASS | job `10167608`；保守complete p50 ratio `0.999271`；4,539,888 params；完整公平profile PASS |
| J020 | M2 | small method train | global-JAPD | R047 fit/select | internal EAL/J2/harm | MUST | BLOCKED | D64默认，seed0 |
| J021 | M2 | visibility control | local-JAPD | same | matched metrics | MUST | BLOCKED | 与J020参数/数据/order一致 |
| J022 | M2 | objective control | global-Candidate-D-PACE | same | matched metrics | MUST | BLOCKED | alpha=0.5 |
| J023 | M2 | decisive mechanism gate | three arms + baselines | fresh300 | two ΔEAL≥0.15且CI>0；EAL≥7.55；J2/domain/harm gates | MUST | BLOCKED | 仅一次评估；失败关闭route |
| J030 | M3 | scale fail-fast | global-JAPD seed0 | 25K internal | max(7.80,1.075x Domino)+domain+J2 | MUST | BLOCKED | J023全过才启动 |
| J040 | M4 | 100K feature/label collection | frozen collector | 100K | completeness/schema/split | MUST | BLOCKED | 排除fresh300/final600 |
| J041 | M4 | full scale seed0 | global-JAPD | 100K internal | opening EAL/domain | MUST | BLOCKED | J030全过 |
| J042 | M4 | full scale seed1 | global-JAPD | 100K internal | opening EAL/domain | MUST | BLOCKED | 同recipe |
| J043 | M4 | full scale seed2 | global-JAPD | 100K internal | opening EAL/domain | MUST | BLOCKED | 同recipe |
| J044 | M4 | freeze deployment seed | median internal EAL | no final outcome | recorded seed/checkpoint | MUST | BLOCKED | tie取小seed |
| J045 | M4 | untouched acceptance result | deployment + all seeds vs Domino | final600 fixed/dynamic | both≥1.15x；domains；bootstrap | MUST | BLOCKED | 三seed全过opening才打开 |
| J050 | M5 | SGLang compliance integration | deployment JAPD | mechanics prompts | one-chain/token parity | MUST | BLOCKED | 禁止method drift |
| J051 | M5 | paired E2E claim | JAPD vs Domino | final600 ABBA | TPS ratio CI lower≥1.15；latency/EAL/memory | MUST | BLOCKED | J045全过才启动 |
| J060 | M6 | AP deletion | AP-only | internal diagnostic | EAL/J2/error recovery | NICE | BLOCKED | B2成功后，不调权重 |
| J061 | M6 | J2 deletion | J2-only | internal diagnostic | EAL/J2/error recovery | NICE | BLOCKED | B2成功后 |
| J062 | M6 | failure anatomy | frozen final checkpoints | internal + final report | position/margin/domain buckets | NICE | BLOCKED | 不参与选择 |

## Transition rules

- `J000–J002 PASS -> J010–J012`。
- D64双FAIL触发的唯一D256分支也已科学FAIL；JAPD-v2冻结recipe在M1关闭，`J020–J023`保持BLOCKED，禁止D512/loss调权/schedule sweep救援。
- `J023全部hard gates PASS -> J030`；否则永久关闭exact JAPD route。
- `J030 PASS -> J040–J043 -> J044`。
- 三个100K seed全部opening PASS后才运行J045。
- J045 fixed/dynamic全部PASS后才运行J050/J051。
- J060–J062永远不能修改主route或解锁失败stage。
