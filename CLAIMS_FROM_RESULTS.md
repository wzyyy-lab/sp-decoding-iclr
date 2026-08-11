# Claims from Results: Full-Lattice Representation Screen

**Verdict:** `claim_supported = no`  
**Confidence:** high  
**Routing:** pivot  
**Integrity status:** unavailable — no `EXPERIMENT_AUDIT.json`  
**Assurance:** same-family Codex review, provisional, `[pending external review]`

## Claim decisions

| Claim | Decision | Binding evidence |
|---|---|---|
| C1a: flat-additive improves axial-additive | No | Raw prompt-balanced EAL difference is `-0.040816327` (`5.199465500 - 5.240281827`). |
| C1b observation: compatibility improves flat-additive | Exploratory yes | Single-seed development point estimate is `+0.019193392`; prompt-cluster bootstrap CI is `[-0.05952, +0.09949]`. |
| C1b as a method claim | No | C1a is the frozen prerequisite and stop gate; the exploratory point estimate cannot revive the route. |
| Full-lattice development gate | No | Best full-lattice raw delta is `+0.106656948`, below the strict existing-best threshold `> +0.285`. |

The matched prompt bootstrap also gives flat-additive minus axial `-0.04082`, 95% development CI `[-0.13095, +0.04932]`. The point-estimate stop rule fails even though this small development interval crosses zero. No sealed test split was opened.

## Supported statement

The binding development screen supports only this negative statement: under the frozen OPB-25K, D128/H8/L2, nine-epoch Candidate-D-PACE protocol, the coupled flat full-lattice mixer did not improve over the axial topology. Compatibility produced a small positive within-flat point estimate, but not robust evidence or a retainable method claim.

## Routing consequence

The full-lattice C1a/C1b route is closed. Further same-route rescue modules are prohibited by the preregistered contract. A new evidence-gathering pivot may instead change the training objective on the already-supported axial selector, provided it is frozen as a separate route before new results are inspected.

Primary artifact: `artifacts/training/gcls_v2_representation_10132458/representation_summary.json`  
Reviewer trace: `.aris/traces/result-to-claim/2026-08-04_run01/`
# 2026-08-04 Reachable-Support Capacity Addendum

The prediction-conditioned reachable-support route is closed. Its frozen
three-cell 128-block capacity gate was negative: the Candidate-D-PACE control
passed all five checks, while `lambda=0` and `lambda=0.1` both missed the
fixed-coverage hard-candidate accuracy threshold. This is a capacity-probe
diagnosis, not development evidence, and authorizes neither OPB-25K training
nor a method-effect claim. Exact values and routing are recorded in
`refine-logs/objective-pivot/RESULT_TO_CLAIM.md`.

# 2026-08-05 Flat FMAS Action-CE Addendum

The 226-way canonical-action CE route is closed by full-data development job
`10133114`. Although validation CE and action accuracy improved across
training, all trained epochs reduced prompt-balanced EAL by `0.424–0.505` and
harmed `31.7–35.0%` of blocks. The frozen selector correctly retained epoch 0,
whose zero gain fails the absolute Gate-2 threshold. Dense action
reconstruction shows only `0.372%` of edits are beneficial, while `34.09%` are
harmful with mean cost `5.304` tokens. This supports a cost-insensitive
surrogate mismatch diagnosis, not an information-ceiling or capacity claim.
Seeds1/2 and formal evaluation remain closed. Exact evidence and forbidden
extrapolations are in
`refine-logs/first-miss-action/GATE2_RESULT_TO_CLAIM.md`.

# 2026-08-05 SAVS Action-Uniform-MSE Addendum

The exact D64/H4/L1 signed-action-value capacity run `10133339` is closed as a
scientific negative. It achieved all-action RMSE `0.006909`, harmful
nonpositive recall `1.0`, and zero selected harm, but beneficial-sign recall
was only `0.78125` and one-edit oracle-gap recovery only `0.44546`, against
frozen thresholds `0.99` and `0.95`. No epoch passed either behavior gate.

This supports the narrow claim that low action-average MSE does not control
the max selector on this frozen same-subset probe. An epoch-zero
harmful/beneficial output-gradient ratio of about `1,518.6x`, together with
`85.06%` of final SSE coming from beneficial actions, is consistent with rare
positive-gradient starvation but does not prove a unique cause. It does not
support held-out, information-ceiling, architecture, or generic
value-regression claims. Full-data SAVS, continuation, D640, threshold rescue,
class-weight rescue, and extra seeds are forbidden. Exact evidence and routing
are in `refine-logs/first-miss-value/CAPACITY_RESULT_TO_CLAIM.md`.

# 2026-08-05 CAMRS Max-Regret Capacity Addendum

The exact seed-0 D64/H4/L1 tie-safe CAMRS capacity run `10133549` passed every
preregistered same-subset criterion. The earliest exact minimum-hinge
checkpoint is epoch 98: mean hinge and decoded regret are zero, all 256
beneficial actions are strictly positive, all 256 repairable blocks select a
utility-optimal action, prompt-balanced oracle-gap recovery is `1.0`, and
there are zero selected harmful actions and zero no-benefit false edits.

This supports same-subset representational/optimization capacity for the exact
procedure and shows that the decision-aware objective operationally removes
SAVS's sparse-positive/max-selection failure on this subset. It does not
support held-out generalization, frozen-feature sufficiency, safety, seed
stability, superiority to external Direct controls, or a paper-level claim.
Independent review returned `claim_supported=yes`, `PASS-ADVANCE`, confidence
high, with same-family provisional assurance. Exact evidence and the binding
Direct-control precondition are in
`refine-logs/first-miss-max-regret/CAPACITY_RESULT_TO_CLAIM.md`.

# 2026-08-05 CAMRS Full-Data Development Addendum

The exact seed-0 D64/H4/L1 CAMRS development procedure is closed by job
`10133649`. It completed all 37,221 updates with complete finite artifacts and
empty stderr, then exited 1 by design because the scientific gate failed. The
frozen selector retained epoch 0 identity at EAL `5.112001943634597`: delta
versus DFlash is `0`, versus Direct-native `-0.22266763848396387`, and versus
Direct-one-edit `-0.10009718172983462`. It also trails Direct-native by seven
first-token-correct blocks. Thus four binding checks fail.

Trained epochs do not rescue the route. Validation hinge decreases from
`0.4430` to `0.1976`, but EAL decreases monotonically to `4.98397`, harmed
blocks rise to `108/1175`, and harmful positive-score actions rise to
`1326/90120`. Beneficial sign recall remains only `101/984` at epoch 3. This
supports a narrow full-data deployment-boundary/max-tail failure diagnosis:
the same-subset CAMRS capacity witness does not transfer under the frozen
development procedure. It does not support generic architectural incapacity,
feature insufficiency, unique causality, seed claims, or formal-test claims.

Fresh result-to-claim review returned `claim_supported=no`,
`FAIL-CLOSE/PIVOT`, confidence high, with same-family provisional assurance.
This exact route cannot be continued, thresholded, reweighted, widened, or
repeated. Only a newly refined and preregistered mechanism with fresh CPU
semantics, capacity gate, and code review may proceed. Exact evidence is in
`refine-logs/first-miss-max-regret/DEVELOPMENT_RESULT_TO_CLAIM.md`.

# 2026-08-05 PROS-Gate Capacity-Plumbing Addendum

Frozen job `10138104` supports a provisional, same-subset capacity-plumbing
claim. Its pass-70 checkpoint independently reconstructs loss `0`, beneficial
APPLY `256/256`, harmful KEEP `128/128`, utility-optimal decisions `512/512`,
oracle recovery `1.0`, and zero regret-bound violations. The original
machine-readable `FAIL` remains immutable; it was caused by an evaluator /
adjudicator schema mismatch in which the already computed harmful-KEEP count
was not emitted under the gate-required alias.

Versioned CPU-only receipt `17ac807e…a352` adds only that redundant alias in
memory, verifies its numerator/partition/denominator invariants, retains the
earliest exact minimum at pass 70, and proves all frozen input hashes unchanged.
This result does not support producer-OOS generalization, comparator
superiority, C1/C2, calibration, or neutral conservatism; neutral APPLY is
`90/128`. R081 is closed, while only R082 implementation/preparation and a
fresh launch-contract review are authorized.

# 2026-08-06 PROS-Gate Fit/Checkpoint Addendum

The sole R082 job `10141115` provisionally supports one narrow claim: the
frozen procedure produced a unique, non-identity, replayable, recovery-valid
pass-5/update-995 checkpoint and an internally consistent frozen comparator
bundle from disjoint fit/checkpoint prompts. Slurm completed `0:0`; the READY
publication tree, runtime identities, 61-file source closure, 26 checkpoint
replays, deterministic orders, selected artifacts, and fit-only ridge all
independently verify.

Checkpoint diagnostics are not held-out efficacy evidence. Selected PROS EAL
is `5.136875`, versus always-Direct `5.13375` and ridge `5.13`; oracle recovery
is `0.6795366795`, harm is `96/1600 = 6%`, and false APPLY is `1049/1426`.
Accordingly R082 supports neither C1/C2 nor any R083, validation, generalization,
or formal claim. Fresh same-family review returned `claim_supported=yes` for
the bundle-only wording, confidence high, publication admissible, integrity
unavailable, and `r083_authorized=yes`. Exactly one unchanged R083 opening may
follow, then mandatory R084 result review; no refit, threshold, seed, alternate
checkpoint, calibration, or manual rescue is allowed.

Trace: `.aris/traces/result-to-claim/2026-08-06_run08/`.

# 2026-08-10 R053 One-Pass Target-Tree Addendum

**Verdict:** `claim_supported = partial`  
**Confidence:** high  
**Routing:** preserve multipath mechanism; run one bounded fixed-forest/graph system Pareto  
**Integrity status:** unavailable — no `EXPERIMENT_AUDIT.json`  
**Assurance:** same-family Codex review, provisional, `[pending external review]`

R053 supports a development-split acceptance claim: actual clean-prefix N64 EAL
is `8.483722060252672`, versus same-geometry clean Domino `7.285471331389699`
and the fixed target `8.325485908649174`.  Chat, code and math all improve;
316 blocks gain, 13 lose, and the net accepted-token gain is 1,406.  The full
W16 structural pool reaches `9.12913022351798`, but this is an upper bound and
not deployable EAL.

R053 does not support the throughput or lossless claim.  The current eager N64
cycle is `52.4564 ms` versus Domino `38.7517 ms`, for projected throughput
`0.8455785683937287x`; stable SGLang selected-branch and emitted-bonus parity
have not been established.  The unfused beam accounts for `16.9257 ms`, so the
system failure does not yet prove the fixed-shape method structurally
impossible.

The only authorized continuation is the preregistered R055 W4/W8/W16 padded
forest and CUDA-graph Pareto.  It must select the smallest width that retains
EAL `>=8.325485909`, has no domain regression, and reaches projected
development throughput `>=1.20x`.  Only then may a frozen SGLang implementation
be tested.  Frozen selector/loss sweeps and R054 repair-comb remain closed.

Primary artifact: `artifacts/analysis/r053_tree_budget_pareto_10165201.json`  
Reviewer trace: `.aris/traces/result-to-claim/2026-08-10_run01/`

# 2026-08-10 PGCF-16 Gate-2 Addendum

**Verdict:** `claim_supported = no`  
**Confidence:** high  
**Routing:** close exact PGCF-v1; separately refine an in-spec v2  
**Integrity status:** unavailable  
**Assurance:** same-family Codex review, provisional, `[pending external review]`

PGCF-v1 supports only same-subset capacity, remote-input sensitivity, and a
favorable eager head-cost claim.  The 2,438,400-parameter global head reaches
same-subset EAL `11.02148` and has complete A40 eager p50 `1.820672 ms`, versus
eager Domino `4.224000 ms`.

It does not support held-out global-context usefulness or superiority to
Domino.  On the disjoint 147-prompt development split, global EAL is
`6.1027696793`, local is `6.0889212828`, Domino is `7.2395529640`, and the
required 1.15x target is `8.3254859086`.  The global-local difference is
`+0.0138483965` with paired prompt-bootstrap 95% CI
`[-0.0603741497,+0.0869776482]`; chat is below base.  Therefore fixed/dynamic
formal efficacy and SGLang throughput claims are unsupported.

The exact v1 curriculum route is closed by its preregistered Gate-2 stop.  This
does not authorize any serial-target, causal/autoregressive, iterative,
beam/tree/forest, or multi-path fallback.  Research may continue only with a
newly preregistered v2 that retains full16 global non-causal one-call one-chain
inference and first proves disjoint held-out accepted-length improvement.

Primary artifact: `artifacts/results/pgcf16_gate2_10167001/report.json`  
Detailed decision: `refine-logs/parallel-global-head-v1/PGCF_GATE2_RESULT_TO_CLAIM_20260810.md`  
Reviewer trace: `.aris/traces/result-to-claim/2026-08-10_run02/`

# 2026-08-10 PCLD-16R P1 Capacity Addendum

**Verdict:** `claim_supported = no`  
**Confidence:** high  
**Routing:** close frozen PCLD-16R; separately refine a new compliant method  
**Integrity status:** unavailable — no `EXPERIMENT_AUDIT.json`  
**Assurance:** same-family Codex review, provisional, `[pending external review]`

PCLD-16R supports only stable-subset interpolation, high same-set mean EAL,
and favorable complete eager head cost.  At step 6000 it reaches EAL
`9.525390625`, candidate agreement `99.9875746%`, and stable J2 `314/314`.

It fails the binding complete-prefix capacity claim: oracle-gap recovery is
`66.9209040%`, harm is `6.25%`, and legacy strict J2 is
`322/411 = 78.3454988%`.  It therefore provides no held-out, fixed/dynamic,
SGLang, or end-to-end throughput evidence.  The frozen method closes before
P2; no schedule, width, threshold, temperature, or loss-weight sweep is
authorized.

Primary artifact: `artifacts/models/pcld16_capacity_10168532/report.json`  
Detailed decision: `refine-logs/parallel-global-head-v3/PCLD_P1_RESULT_TO_CLAIM.md`  
Reviewer trace: `.aris/traces/result-to-claim/2026-08-10_run05/`

# 2026-08-10 JAPD-16 M1 D64 Capacity Addendum

**Verdict:** `claim_supported = no` for D64 sufficiency  
**Confidence:** high  
**Routing:** run the single preregistered uniform D256 branch; M2 remains blocked  
**Integrity status:** unavailable  
**Assurance:** same-family Codex review, provisional, `[pending external review]`

D64 shows strong narrow same-set memorization but does not pass M1. J010 reaches
J2 `100%` and EAL `11.314453`, but oracle recovery is `92.7966% < 95%` and harm
is `1.3672% > 1%`. J011 is a clear full-fit failure: EAL `6.312012` versus
Domino `7.330566`, J2 `17.4518%`, oracle recovery `4.0101%`, and harm `16.8945%`.

Both frozen D64 gates therefore fail and strictly trigger the planned uniform
D256/H8/L2 branch. This is not evidence against the unrun held-out JAPD claim,
nor evidence for Domino/E2E superiority. D256 must pass the unchanged J010,
J011, and J012 gates before M2; otherwise the exact JAPD route closes without a
D512, loss/schedule sweep, or any serial/tree fallback.

Detailed decision: `refine-logs/parallel-global-head-v2/JAPD_M1_D64_RESULT_TO_CLAIM_20260810.md`  
Reviewer trace: `.aris/traces/result-to-claim/2026-08-10_run03/`

# 2026-08-10 JAPD-16 M1 D256 Closure Addendum

**Verdict:** `claim_supported = no` for D256 M1 sufficiency  
**Confidence:** high  
**Routing:** close exact JAPD-v2 at M1; refine a genuinely separate in-spec v3  
**Integrity status:** deterministic precheck unavailable  
**Assurance:** same-family Codex review, provisional, `[pending external review]`

D256 preserves the required full16/global/noncausal/one-call/one-chain topology,
uses `4,539,888` parameters, and its complete A40 eager p50 is `0.99927x`
released Domino. It nearly memorizes the 512-block capacity set: candidate
accuracy is `99.8627%`, hard accuracy `99.7033%`, J2 `100%`, and EAL
`11.4121`.

The binding gates nevertheless fail. Capacity recovery is `94.2090% < 95%`
and harm is `1.171875% > 1%`. On the broader 512-prompt same-set full-fit,
recovery is only `5.5133%`, J2 `24.6136%`, harm `18.1641%`, and EAL `6.3862`
versus Domino `7.3306`. Thus neither M2, held-out accepted length, nor system
speedup is supported.

The exact JAPD-v2 recipe closes without D512, loss/schedule rescue, or any
serial/GRU/iterative/tree/multipath fallback. A new v3 may proceed only through
fresh refinement under the immutable parallel one-chain contract.

Detailed decision: `refine-logs/parallel-global-head-v2/JAPD_M1_D256_RESULT_TO_CLAIM_20260810.md`  
Reviewer trace: `.aris/traces/result-to-claim/2026-08-10_run04/`
