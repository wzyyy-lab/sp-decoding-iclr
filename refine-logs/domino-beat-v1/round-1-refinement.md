# Round 1 Refinement

The Stage-1 head-only mask remains unchanged: a frozen wrong base token makes
the block unreachable by the head.  The Stage-2 joint objective is now
explicitly different.  Position zero always receives a base gold-versus-best-
competitor margin/CE; correction-controlled positions retain detached
clean-prefix reachability; the moving parallel backbone is anchored to its
released base logits/hidden states.  Thus a jointly trainable backbone can
repair first-token failures without sacrificing the released base branch.

The head screen also adds a gold-versus-best-competitor softplus margin variant
at the first reachable breaker.  This directly targets the greedy decision
boundary and is evaluated beside CE under an identical training/evaluation
path.  The released DFlash K16 oracle is labeled correctly as a candidate-path
diagnostic rather than a Domino same-candidate oracle.

For the D-PACE control, with detached gold probabilities `p_i` and smoothing
`q_i=(1-s)p_i+s`, suffix position `j>=1` receives a survival-derived weight
computed from products containing `q_0`; no loss is applied at position zero
in the head-only stage, and normalization uses the sum of effective suffix
weights.
