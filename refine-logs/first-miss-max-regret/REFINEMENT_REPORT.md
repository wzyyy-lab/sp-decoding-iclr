# CAMRS Refinement Report

The route starts from the failed SAVS capacity result, where action-average
MSE hid positive max-policy errors. The initial CAMRS idea replaced dense
regression with a per-block cost-augmented structured hinge that upper-bounds
deployed regret. Round-1 review validated the bound but found implementation
and evidence-contract defects. The revision made the zero-loss point
stationary, separated block and prompt estimands, froze the exact capacity
threshold, added joint-epoch and optimization-diversion evidence, and narrowed
the statistical/novelty claims. Round-2 review returned `9.2 READY`.

Artifacts:

- initial proposal: `round-0-initial-proposal.md`;
- Round-1 review/revision: `round-1-review.md`, `round-1-refinement.md`;
- Round-2 review: `round-2-review.md`;
- final contract: `FINAL_PROPOSAL.md`;
- review history: `score-history.md`.
