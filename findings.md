# Research Findings

## 2026-08-04 — Full-lattice GCLS route closed

### What was tested

A frozen three-cell representation matrix compared axial-additive, flat-additive, and flat-compatibility selectors on the exact same materialized 25,000-prompt / 199,818-block OPB subset, validation prompts, optimizer budget, initialization contract, and Candidate-D-PACE objective.

### Result-to-claim verdict

An independent `gpt-5.6-sol` ultra reviewer returned `claim_supported=no`, confidence high, routing `pivot`. Flat-additive was `-0.040816327` raw EAL below axial-additive. Flat compatibility was `+0.019193392` above flat additive, but this was one development seed with a prompt-cluster interval crossing zero and cannot survive the C1a stop gate. The best full-lattice raw delta (`+0.106656948`) was far below the required `>+0.285` threshold. There is no experiment-integrity audit, so the semantic verdict is provisional.

### Failure diagnosis

1. The unstructured 240-node flat attention is a worse inductive bias than explicit within-position competition followed by cross-position axial mixing at this budget.
2. More capacity is not the immediate cure: historical D128/L2 did not reliably beat D64/L1, and the current flat variants also underperformed the larger axial control.
3. The training surrogate is misaligned with greedy accepted length. For flat-compatibility, training objective improved every epoch, yet raw development delta peaked at epoch 4 (`+0.106657`) and fell to `-0.106778` by epoch 9 while harmful blocks rose from 90 to 198.
4. Prompt diversity matters strongly, but scaling is diminishing: historical axial-global raw delta rose from `+0.00996` at 10K to `+0.10775` at 25K, `+0.21927` at 50K, and `+0.24247` at 99,356 prompts; 50K to full was much smaller than 25K to 50K.
5. The practical gap is mostly identifiability/representation, not candidate availability: the best learned selector recovered only `6.18%` of the K16 oracle gap, while Domino's released checkpoint remained `1.61880` EAL ahead of the best GCLS development run.

### Constraints for future attempts

- Do not rerun or post-hoc rescue the flat full-lattice C1a/C1b route.
- Do not treat candidate classification accuracy, training NLL, or a single-seed positive point estimate as an accepted-length claim.
- Do not open the sealed test until a new route, checkpoint rule, calibration split, and seed protocol are frozen.
- Keep the axial model as the proven structural baseline; change one major causal factor at a time.

### Pivot

Freeze a new objective-only route on the axial selector: compare Candidate-D-PACE with a policy-reach Head-AUF objective, then add a small coverage auxiliary only if pure Head-AUF is too sparse. This directly removes independent suffix supervision after the current selector's first breaker while retaining future lattice nodes as attention inputs. Repair/protection margins remain a later, separately gated factor.

Trace: `.aris/traces/result-to-claim/2026-08-04_run01/`.

## 2026-08-05 — Dense uniform value fitting misses sparse positive decisions

The preregistered SAVS capacity falsifier completed all 5,120 updates on the
frozen 512-block subset. Aggregate value RMSE was very low (`0.006909`) and
the strict-positive decoder selected no harmful actions, but positive-sign
recall was `0.78125` and oracle-gap recovery `0.44546`; both behavior gates
failed, and no epoch came close to the frozen `0.99/0.95` requirements.

The failure is structurally informative. At zero initialization, harmful
actions dominate the output-head gradient by about `1,518.6x`; at the selected
checkpoint, beneficial actions still account for `85.06%` of residual SSE.
Consequently a tiny average error can coexist with 56 missed positive signs
and only 83 selected beneficial edits. This is consistent with sparse-positive
gradient starvation and action-average/max-policy mismatch, but one run does
not isolate a unique causal factor.

The exact action-uniform-MSE route is closed. Its negative cannot be rescued
with longer training, size, thresholds, weights, or extra seeds. A subsequent
route must introduce and preregister a genuinely decision-aware mechanism and
repeat capacity-first review.

## 2026-08-05 — Max-regret supervision resolves the same-subset optimization failure

CAMRS replaces dense action-average regression with a tie-safe cost-augmented
structured hinge that directly upper-bounds decoded one-edit regret. On the
same frozen 512-block capacity manifest where SAVS missed 56 positive signs,
the exact D64/H4/L1 CAMRS run reached zero hinge at the earliest minimum epoch
98, selected all 256 utility-optimal repairs, made no harmful or false edit,
and recovered `1.0` of the one-edit oracle gap.

The gradient mechanism changes materially: all 256 repairable blocks
contribute an oracle-upward signal at initialization, whose projection norm is
`0.20358`, rather than having the positive class diluted by 57,765 harmful
actions under an action-average loss. This is strong evidence that CAMRS fixes
the observed optimization/surrogate mismatch on the adaptive subset, but it
does not distinguish generalizable structure from memorization. Some harmful
actions remain positive yet never win, and loss-augmented competitors need not
match deployed actions.

The route therefore advances only to one physically isolated development
test. External matched Direct-native and Direct-one-edit artifacts must be
produced and hash-frozen first, followed by a fresh code-review GO. No formal
claim is available until held-out behavior is observed.

## 2026-08-05 — CAMRS capacity repair does not survive the max-action tail

Full-data job `10133649` completes the capacity-to-development falsification.
Across epochs 1-3, held-out hinge falls to approximately `0.20`, but raw EAL
falls below DFlash by `0.075`, `0.109`, and `0.128`; harm rises from `5.70%`
to `9.19%`. At epoch 3, `98.53%` harmful-nonpositive recall still leaves 1,326
harmful actions with positive scores among 90,120 harmful actions. Max
selection converts that thin tail into 108 harmed blocks, while only 53 of 984
blocks select their utility-optimal repair.

This explains why the same-subset result and held-out result coexist: CAMRS
can enforce every structured constraint on 512 memorized blocks, yet its
226-way decision boundary does not generalize at the required tail precision.
The result is not clean proof of representation or optimizer incapacity because
the artifact omits a final train endpoint diagnostic. The defensible finding is
a finite-schedule deployment-boundary/max-tail failure. CAMRS materially
mitigates flat CE's over-editing, but does not solve it.

### Read-only pivot feasibility diagnostic

The frozen Direct control exposes a smaller action space with real headroom.
Direct-native improves 141 blocks, is neutral on 972, and harms 62; it reaches
EAL `5.334669582118561`. An oracle that chooses only between DFlash and this
fixed Direct-native path reaches prompt-balanced EAL `5.430758017492711` with
zero harm, above the existing strict target `5.396991943634597`. The
base/Direct/one-edit three-way oracle is only slightly higher at
`5.438411078717201`, so the extra action is unlikely to justify another
large-action selector.

This is exploratory upper-bound evidence, not a learned-method result. It
motivates a genuinely new binary abstention mechanism: KEEP DFlash versus
APPLY frozen Direct-native, trained to predict signed block-level gain. Such a
route directly removes the max-over-225 false-positive amplification and must
be independently refined and preregistered before implementation or GPU use.

## 2026-08-05 — PROS-Gate passes same-subset capacity after a versioned adjudication repair

The sole 38,674-parameter PROS-Gate capacity run `10138104` completed all 5,120 updates and selected the earliest exact minimum-loss checkpoint at pass 70. Independent saved-record replay gives loss `0`, beneficial APPLY `256/256`, harmful KEEP `128/128`, utility-optimal behavior `512/512`, oracle-gap recovery `1.0`, and zero regret-bound violations.

Its original machine verdict remains `FAIL`: the evaluator serialized harmful KEEP as `harm_avoidance_numerator` but omitted the gate-required alias `harmful_keep_count`, so the guarded lookup returned false before evaluating behavior. A separately versioned, CPU-only adjudication added only the algebraically redundant alias in memory, enforced alias/partition/denominator invariants, and produced append-only receipt `17ac807e…a352`. All eight frozen run hashes and the 60-file repair closure were identical before and after replay.

This establishes only same-subset optimization/plumbing capacity and closes R081 provisionally. It does not support producer-OOS generalization, C1/C2, calibration, or neutral conservatism: the selected checkpoint applies Direct on `90/128` neutral blocks. The only authorized next step is R082 implementation/preparation followed by a fresh launch-contract review.

## 2026-08-06 — PROS-Gate freezes a valid checkpoint bundle, but checkpoint diagnostics are weak

The sole reviewed R082 job `10141115` completed `0:0` and atomically published
a READY 128-file bundle. Independent saved-record replay and all 26 checkpoint
reconstructions identify pass 5/update 995 as the unique maximum of the frozen
lexicographic selector. Its selected checkpoint is non-identity, finite, and
recovery-valid; the fit/checkpoint prompt sets are disjoint, all runtime/source
hashes match, and the ridge/constant comparator bundle was frozen before
checkpoint access.

The checkpoint behavior is diagnostically weak: PROS reaches EAL `5.136875`,
only `+0.003125` over always-Direct and `+0.006875` over the ridge comparator.
It recovers `0.6795367` of the binary-oracle gap, applies on `96/101` harmful
blocks, harms `6.0%` of all blocks, and false-applies on `1049/1426`
non-beneficial blocks. These miss the later R083 recovery/harm thresholds, but
those thresholds were not preregistered R082 stop criteria.

Fresh `result-to-claim` review therefore returns `claim_supported=yes` only for
the narrow checkpoint-bundle claim, with high confidence, same-family
provisional assurance, no integrity audit, and zero blockers. It authorizes
exactly one unchanged R083 falsifier opening followed by R084 review. R082 does
not support C1/C2, generalization, validation, formal performance, calibration,
or any efficacy claim; no tuning, alternate checkpoint, seed, or rescue is
permitted before or after the one-shot falsifier.

Trace: `.aris/traces/result-to-claim/2026-08-06_run08/`.

## 2026-08-10 — One-pass target multipath clears the acceptance target; the unfused eager head does not clear throughput

R053 is the first material main-target success in the current route.  On all
147 validation-select prompts / 1,175 blocks, the actual HF N64 tree scored
against an unconditional batch-1 `qlen=1` target continuation reaches clean
EAL `8.483722`, versus clean Domino `7.285471`.  It gains on 316 blocks, loses
on 13, adds 1,406 accepted tokens net, and improves chat/code/math by
`+0.670/+1.250/+1.655`.  This supports the hypothesis that online target
verification of multiple draft branches resolves the identification failure
that defeated frozen single-chain selectors.

The current implementation is not a serving solution.  Its A40 eager complete
cycle is `52.456 ms` versus Domino `38.752 ms`, so projected throughput is only
`0.8456x`.  The target-tree forward itself remains about `33.64 ms` from N24 to
N64; the dominant avoidable cost is the host-driven W16 beam (`16.93 ms`) plus
traversal (`1.60 ms`).  The full W16 structural EAL `9.129130` and median trie
size 99 justify exactly one fixed-shape graph experiment, but they are not
deployable results.

Fresh result-to-claim review is `partial`, high confidence, same-family
provisional.  R055 therefore tests W4/W8/W16 padded forests and CUDA graphs
under a `1.20x` development throughput gate.  If no fixed width jointly passes
acceptance and latency, the one-pass target-multipath family closes; no learned
selector, OPB12K or repair-comb rescue is allowed.

Trace: `.aris/traces/result-to-claim/2026-08-10_run01/`.

## 2026-08-10 — PGCF-v1 has full16 parallel capacity but fails held-out transfer

The active PGCF-v1 head obeys the user's architecture contract: one invocation
globally mixes all 16 positions non-causally and emits exactly one 16-token
chain.  It has 2,438,400 parameters and reaches the exact Top-16 oracle on a
512-block same-subset capacity test, while its complete A40 eager p50 is only
`1.820672 ms` versus eager Domino's `4.224000 ms`.

The binding disjoint Gate-2 result is negative.  On 147 prompts / 1,175 blocks,
global EAL is `6.1027696793`, local is `6.0889212828`, released Domino is
`7.2395529640`, and the required target is `8.3254859086`.  Global-local is
only `+0.0138483965`; its 10,000-draw paired prompt-bootstrap 95% interval is
`[-0.0603741497,+0.0869776482]`.  Chat also falls below base.  Remote
intervention changes the global output but does not establish useful global
information because the primary effect is tiny and unresolved.

The mechanism diagnosis is prefix-utility transfer failure, not head capacity
or latency.  Global repairs only `46/946` repairable first rejections, while
2,487 of its 2,641 edits occur after the base first rejection and 619/756
changed blocks are suffix-only.  A perfect one-frontier repair reaches only
`7.49854`, so solving the main target requires jointly correct multi-position
prefixes in the single parallel output.  Late train EAL rises to `7.37305`
while validation falls to `5.98627`, directly exposing overfit.

Fresh result-to-claim review returns `claim_supported=no`, high confidence,
same-family provisional.  The exact v1 curriculum is closed; it cannot be
rescued by longer training, more v1 parameters, or immediate 199.8K
collection.  The broader full16/global/non-causal/one-call/one-chain workstream
remains active only through a separately refined and preregistered v2 that
directly optimizes multi-position clean-prefix utility.  Off-spec serial,
autoregressive, iterative, tree, beam, forest, or multi-path routes remain
forbidden.

Trace: `.aris/traces/result-to-claim/2026-08-10_run02/`.

## 2026-08-10 — JAPD D64 memorizes the narrow capacity set but fails M1

The 433,852-parameter full16/global/noncausal/one-chain JAPD head reaches J2
`411/411` and EAL `11.314453` on the frozen 512-block capacity group. It still
misses the binding oracle-recovery threshold (`92.7966% < 95%`) and harm ceiling
(`1.3672% > 1%`). On the larger 512-prompt full-fit diagnostic it transfers
poorly even within the training distribution: EAL is `6.312012` versus Domino
`7.330566`, J2 is `17.4518%`, recovery is `4.0101%`, and harm is `16.8945%`.

The defensible diagnosis is D64 capacity/optimization insufficiency, not a
held-out verdict on JAPD. Both D64 gates fail, satisfying the preregistered AND
condition for one uniform D256/H8/L2 retry with unchanged data, loss, schedule,
and architecture semantics. M2 remains blocked until D256 capacity, full-fit,
and complete eager latency all pass. No further width, objective tuning,
serial/GRU/iterative decoding, or tree/multipath rescue is authorized after a
D256 scientific failure.

Trace: `.aris/traces/result-to-claim/2026-08-10_run03/`.

## 2026-08-10 — D256 removes the small-head excuse but does not rescue JAPD-v2

The 4.54M-parameter D256 global head remains fully parallel and costs roughly
the same as eager Domino, so its negative scientific result is not a latency
failure. It reaches `99.8627%` candidate accuracy on 512 capacity blocks, yet
six harmed blocks are enough to miss both prefix-recovery and harm gates. On
the broader 512-prompt training group it recovers only `5.5133%` of the oracle
gap, with `18.1641%` harm and EAL below Domino.

This closes the exact JAPD-v2 recipe before M2. The evidence distinguishes
small-set memorization from scalable prefix control: average per-position
candidate fit can be almost perfect while rare early mistakes still dominate
EAL, and widening the frozen-feature selector does not solve broader same-set
optimization. It does not establish an information-theoretic ceiling or a
held-out result. Any continuation must be a separately refined full16 global
noncausal one-call one-chain method; D512, same-loss tuning, serial decoding,
iteration, and multipath rescue are forbidden.

Trace: `.aris/traces/result-to-claim/2026-08-10_run04/`.

## 2026-08-10 — PCLD-16R fits stable support but fails complete-prefix capacity

The 3,826,688-parameter PCLD-16R head obeys the immutable architecture
contract and has favorable complete eager cost.  On its frozen 512-block
same-set capacity group, the selected step-6000 checkpoint reaches EAL
`9.525390625`, teacher candidate agreement `99.9876%`, and stable J2
`314/314`.

The binding capacity decision is nevertheless negative: oracle-gap recovery
is `66.9209%` versus `95%` required, harm is `6.25%` versus `1%` allowed, and
legacy strict J2 is `322/411 = 78.3455%` versus `99%` required.  The receipt
contains 313 shared legacy/stable cases, 98 legacy-only cases, and one
stable-only case.  This isolates a support and safe-trajectory objective gap:
perfect mastery of the filtered loss-aligned population does not control the
complete early-error population.

Fresh result-to-claim review returns `claim_supported=no`, high confidence,
same-family provisional.  The exact PCLD-16R recipe is closed before P2 and
cannot be rescued by schedule, width, threshold, or loss-weight sweeps.  A
continuation must be a separately refined method that preserves full16 global
non-causal one-call one-chain inference and directly addresses complete-prefix
survival and false-edit risk.

Trace: `.aris/traces/result-to-claim/2026-08-10_run05/`.
