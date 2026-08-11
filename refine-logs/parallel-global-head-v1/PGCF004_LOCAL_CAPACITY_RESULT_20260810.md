# PGCF-004 Matched-Local Capacity Diagnostic

Job `10166853` completed the preregistered D0 first-512, seed-0, dense-gold-CE
diagnostic for the parameter-matched local control.  It used the same d256,
two-layer, 2,438,400-parameter implementation and optimization settings as the
global capacity witness; the only structural difference was the fixed
same-position attention mask.

The earliest best prompt-balanced EAL checkpoint was step 6,400:

| Metric | Result |
|---|---:|
| model EAL | 10.943359375 |
| pure base16 oracle EAL | 11.021484375 |
| oracle-gap recovery | 98.6599665% |
| all-supported candidate accuracy | 99.6803719% |
| supported non-Top1 accuracy | 99.4761171% |
| harmed fraction | 0.1953125% |

The run was finite and passed all four target-capacity diagnostics.  This
rules out a broken local training branch, mask, or optimizer as the explanation
for a future global-local difference.  It is deliberately not a second Gate1
hard witness and is not held-out evidence that global mixing helps.

Authoritative artifact:
`artifacts/models/pgcf16_capacity_local_gold_ce_10166853/report.json`.
