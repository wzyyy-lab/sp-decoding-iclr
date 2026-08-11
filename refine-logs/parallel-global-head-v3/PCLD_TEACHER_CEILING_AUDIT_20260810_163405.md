# PCLD-16R Teacher-Candidate Ceiling Audit

> Scope: read-only development evidence; not a deployable result and not held-out student evidence.

## Source and authority

- Source rollout: artifacts/canonical/r047_anchor_t4_validation_select_10164718
- Records/prompts: 1,175 blocks / 147 disjoint validation-select prompts
- Candidate set: actual pure-base full-vocabulary FP32 Top16 stored by all_position_onpolicy_decode
- Teacher scores: target LM-head logits at the canonical clean prefix, gathered at those same 16 candidates
- Acceptance: canonical 16-token prefix match, then prompt-balanced mean over blocks within each prompt

No target hidden is used online here. This audit only asks whether the proposed clean target teacher has enough candidate-space headroom to justify distillation.

## Result

| Policy | Prompt-balanced EAL |
|---|---:|
| Pure DFlash rank 0 | 6.0685131195 |
| Released Domino | 7.2395529640 |
| Clean target-candidate teacher | 10.5971817298 |
| Gold-aware pure-base Top16 oracle | 10.9092565598 |

- Teacher / Domino EAL ratio: 1.4637895161
- Teacher recovery of base-to-oracle gap: 93.5532%
- Teacher versus base blocks: 906 gain / 3 loss / 266 tie
- Harm fraction: 3 / 1175 = 0.2553%
- All three loss blocks contain at least one stored target-top1 versus canonical-gold mismatch.

Per-domain prompt-balanced EAL:

| Domain | Base | Domino | Teacher | Gold oracle |
|---|---:|---:|---:|---:|
| chat | 2.774926 | 3.624256 | 6.755952 | 6.977307 |
| code | 5.125000 | 6.558673 | 10.191327 | 10.400510 |
| math | 10.155000 | 11.377500 | 14.682500 | 15.182500 |

## Interpretation

This closes only the teacher-headroom question: clean target candidate scores have ample room above the 8.3254859086 development target and above released Domino in every domain.

It does **not** show that:

- the online DFlash lattice can predict T-H across prompts;
- a shared rank-256 residual subspace is expressive enough;
- hidden-residual supervision transfers better than matched candidate KL;
- the final head improves fixed/dynamic EAL or SGLang throughput.

Those remain the P1/P2 student falsifiers. A train-only PCA spectrum may be reported as a latent diagnostic, but it is not a candidate-space upper bound and cannot close the route. The three harmed blocks also require clean-authority masking; they cannot be silently trained as correct canonical rows.
