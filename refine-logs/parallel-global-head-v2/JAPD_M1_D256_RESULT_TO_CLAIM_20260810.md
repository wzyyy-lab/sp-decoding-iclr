# JAPD-16 M1 D256 Result-to-Claim

**Verdict：`NO`。D256 rescue 不支持进入 M2，exact JAPD-v2 recipe 在 M1 关闭。**

Assurance：fresh GPT-5.6-Sol reviewer，same-family provisional；confidence high。确定性 evidence precheck unavailable，reviewer 已直接核查指定 JSON。

## 实际支持

- D256/H8/L2 head 有 `4,539,888` 参数，完整输入一次输出 `[1,16,16]`，所有架构/parity receipts 通过。
- 完整 A40 eager p50 为 released Domino 的 `0.99927x`，所以延迟不是本轮失败原因。
- 512-block same-set capacity 上 candidate accuracy `99.8627%`、hard accuracy `99.7033%`、J2 `411/411`、EAL `11.4121`，证明小集记忆容量很强。

## 为什么仍然失败

Capacity 的 binding gate 仍为 FAIL：oracle-gap recovery `94.2090% < 95%`，harm `6/512 = 1.171875% > 1%`。八个 support-position 错误、七个 hard-position 错误已足以留下 `0.40039` EAL oracle gap；position-micro accuracy 不能代替 prefix-level utility。

更关键的是 512-prompt broader full-fit 仍是同集失败：candidate accuracy `70.7282%`、hard accuracy `25.5247%`、J2 `24.6136%`、recovery `5.5133%`、harm `18.1641%`；EAL `6.3862` 低于 Domino `7.3306`。这不是 held-out transfer 结论，而是 frozen JAPD recipe 在更广同集上就无法稳定优化 joint repair。

## 允许与禁止的后续

- J020–J022/fresh300 保持关闭；不提交 M2。
- 不允许 D512、JAPD loss 调权、schedule sweep、延长训练或 post-hoc threshold rescue。
- 串行 target decode、GRU/causal/autoregressive、iteration/Jacobi、beam/tree/trie/forest/multipath 继续为硬禁止。
- 后续只能作为独立 v3 workstream 重新精炼，并严格保留 full16 global noncausal one-call one-chain online contract。首先必须区分 frozen-feature identifiability、tail-risk objective mismatch 与表示适配需求，不能把本结果粉饰成泛化证据。

Primary artifacts：

- `artifacts/models/japd16_capacity_d256_10167607/report.json`
- `artifacts/models/japd16_full_fit_d256_10167609/report.json`
- `profile_output/japd16_eager_d256_10167608.json`
- `.aris/traces/result-to-claim/2026-08-10_run04/`
