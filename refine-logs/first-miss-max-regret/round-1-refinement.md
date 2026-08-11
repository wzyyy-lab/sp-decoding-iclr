# Round-1 Refinement: Tie-Safe CAMRS

This revision resolves every Round-1 blocker without inspecting a CAMRS
training result or adding an auxiliary objective.

## 1. Exact oracle, deployment, and loss

Use FP32 for target utilities, action scores entering the loss, loss values,
bound checks, and gate arithmetic. Define the oracle deterministically:

```text
if max_edit v <= 0: a* = KEEP
else:               a* = lowest-index argmax_a v(a)
```

The capacity geometry has exactly one beneficial action on each of its 256
repairable blocks. Lowest-index fallback exists only for generic fixtures or
future utility-equivalent ties. Deployment independently uses KEEP whenever
the maximum edit score is `<=0`; positive edit-score ties choose the lowest
action index.

For every `a != a*`, define

```text
m(a) = s(a) + [v(a*)-v(a)] - s(a*)
c    = lowest-index argmax_{a != a*} m(a)
H(x) = ReLU(m(c)).
```

`ReLU(0)` is required to have derivative zero. Excluding the oracle and using
the explicit zero branch makes `s=v` both zero-loss and zero-gradient. It is
value-equivalent to the original max-including-oracle formula, while removing
its implementation-dependent zero-loss subgradient.

For a deployed `a_hat != a*`, `a_hat` is in the non-oracle maximum and

```text
H >= ReLU(s(a_hat)+v(a*)-v(a_hat)-s(a*))
  >= v(a*)-v(a_hat),
```

because score maximization gives `s(a_hat)>=s(a*)`. If `a_hat=a*`, regret is
zero and nonnegativity suffices. The bound therefore covers deployed KEEP,
beneficial, neutral, and harmful actions, including KEEP-preferred ties.

This is explicitly **cost-sensitive hard-constraint selection**, not post-hoc
class weighting of SAVS MSE. Regret controls both the margin and which
constraint is active.

## 2. Gate-0 additions

CPU tests must establish:

1. zero loss and exactly zero action-score and residual-score gradient at
   `s=v`;
2. exact bound for deployed KEEP/oracle/neutral/harmful cases;
3. KEEP ties, multiple positive score ties, non-oracle cost-augmented ties,
   oracle ties, all-neutral blocks, and no-benefit blocks with harms;
4. equality in value between explicit-ReLU/non-oracle CAMRS and the original
   max-including-oracle expression;
5. randomized FP32 tensors have minimum `H-regret >= -1e-6`, with zero
   violations beyond tolerance;
6. residual-difference gradients have the expected positive oracle and
   negative competitor update directions after accounting for
   `s(i,r)=rho(i,r)-rho(i,0)`;
7. the repairable oracle score gradient is `-1/B` per active block before
   residual coupling, not divided by 225;
8. first-backward/second-backward and frozen-input semantics remain exact;
9. saved examples reconstruct all objectives, gates, tie choices, and bound
   checks.

## 3. Frozen aggregation and numeric gate

Training and capacity checkpoint selection use the uniform block mean

```text
H_block = (1/512) sum_x H(x).
```

The frozen capacity manifest has 462 total oracle-gain tokens, so its exact
block-weighted normalized oracle advantage is

```text
V*_block = 462 / (512*15) = 0.06015625.
```

The hinge threshold is frozen numerically at

```text
H_block <= 0.05 * V*_block = 0.0030078125.
```

The pointwise bound makes the corresponding block-weighted regret/gap check a
diagnostic. The existing `>=0.95` **prompt-balanced** oracle-gap recovery is a
separate, nonredundant behavior gate; no implication between the two
aggregation schemes is claimed.

## 4. Exact conjunctive capacity gate

The D64/H4/L1, batch32, 320-epoch, 5,120-step optimizer/data contract remains
unchanged. Select the earliest exact minimum-`H_block` checkpoint. The selected
checkpoint must jointly satisfy:

- zero FP32 bound violations beyond `1e-6` and minimum slack `>=-1e-6`;
- `H_block <=0.0030078125`;
- beneficial strict-positive recall at least `254/256`;
- utility-optimal action accuracy at least `244/256` on repairable blocks
  (`>=0.95`; utility-equivalent oracle ties count correct);
- harmful nonpositive recall `>=0.99` over all 57,765 harmful actions;
- prompt-balanced one-edit oracle-gap recovery `>=0.95`;
- selected harm at most `5/512` blocks (`<=0.01`);
- false edits on no-benefit blocks at most `2/256` (`<=0.01`);
- exactly 256 beneficial actions, finite gradients, and exact epoch-zero
  identity.

The integer forms are the binding checks. For each epoch, save every raw value
and boolean, a single `joint_gate_passed`, the selection key, bound minimum
slack/violation count, and `is_selected`. The report must list all jointly
passing epochs and separately say whether the selected epoch passed. A
nonselected passing epoch is diagnostic only and cannot rescue a checkpoint
rule failure.

## 5. Hardest-competitor diagnostics

At zero scores the competitor is the minimum-utility action, not usually the
deployed boundary. This is a finite-optimization risk, not a theorem failure.
Each epoch must report:

- competitor equals deployed action fraction;
- competitor raw-score rank before cost augmentation;
- competitor utility sign and regret histogram;
- fraction whose competitor wins only after cost augmentation;
- distinct competitor coverage and epoch-to-epoch churn;
- zero-loss block fraction;
- repair-oracle-upward and competitor-downward output-projection gradient
  norms, cosine, cancellation ratio, and total norm;
- unclipped total gradient norm and fraction of steps clipped.

No boundary auxiliary, curriculum, smoothing, or second objective may be
introduced. If finite updates are diverted to already noncompetitive harmful
actions and the gate fails, that closes this exact objective/model/optimizer/
schedule combination; it does not establish that all structured objectives
or frozen features fail.

## 6. Statistical and evidence scope

The hinge inequality is per realized example and therefore remains an upper
bound after expectation even when utilities are stochastic conditional on
visible features. This does **not** make it Fisher-consistent: indistinguishable
inputs with conflicting utilities may make zero population hinge impossible,
and `s=v` is only a pointwise construction. No calibrated-value,
identifiability, population optimization, held-out safety, or information-
ceiling claim is made from capacity memorization.

Capacity train/evaluation intentionally reuse the exact same adaptive
manifest and are engineering evidence only. The trainer must retain manifest,
prompt-set, source start/end, target/draft fingerprint, optimizer-step, and
example-reconstruction provenance. Development must use physical split
isolation.

Before any development authorization, an amendment must freeze exact
Direct-native and Direct-one-edit artifact paths, hashes, configuration, and
evaluation semantics after the pending matched Direct job exists. On the
fixed 1,175-block validation collection, the first-token `0.001` tolerance
permits at most one fewer correct block than Direct-native; a cardinality
change fails closed.

## 7. Novelty boundary

CAMRS is an application of established multiclass/structured max-margin
machinery. Neither the hinge, loss-augmented inference, cost sensitivity,
accepted-length payoff prediction, nor payoff-guided scheduling is claimed as
algorithmically novel. The candidate contribution is limited to integration:
exact counterfactual one-edit prefix regret on a frozen DFlash lattice,
KEEP-anchored exact-identity scores, and a deployment-aligned pointwise regret
bound. No “first” claim is allowed until a broader search covers
cost-sensitive imitation learning, learning-to-search, direct loss
minimization, and counterfactual policy learning.

## 8. Authorization

This refinement authorizes only independent Round-2 review. CPU
implementation still requires a no-blocker score of at least 9.0. GPU remains
closed until subsequent experiment-bridge review.
