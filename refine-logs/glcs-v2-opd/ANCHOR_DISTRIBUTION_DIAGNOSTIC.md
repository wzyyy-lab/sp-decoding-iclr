# Anchor Distribution Diagnostic

**Date:** 2026-08-09  
**Purpose:** explain why 295,604 static training blocks can reduce the training objective without improving exact-runtime held-out EAL.

## Observation

Read-only inspection of the stored `anchor_offset` fields shows that the two main collections use deterministic, non-deployment anchor schedules:

- Open-PerfectBlend training prompts normally contain eight blocks at offsets `0,16,32,48,64,80,96,112`; among the first 4,096 inspected train blocks, consecutive offset difference `16` accounts for 3,358 transitions and offset modulo 16 is zero for 3,871 blocks.
- Phase-3 prompts normally contain eight blocks at offsets `0,34,69,103,137,171,206,240`; among the first 4,096 inspected train blocks, consecutive differences `34/35` account for 2,909 transitions.

Released Domino's accepted length is about 7.2 on these distributions, so real speculative rollout advances by a policy-dependent amount rather than a fixed 16 or 34/35 tokens. The large GLCS-v1 run therefore sees many uniformly spaced target-clean states but not the state distribution induced by its own accepted/rejected blocks.

## Consequence

This diagnostic does not alone prove unique causality, but it identifies a concrete train/deployment mismatch consistent with all observed results:

1. same-subset capacity reaches the Top-16 oracle;
2. static hard-label training loss decreases;
3. held-out EAL and first-error behavior do not improve;
4. adding more blocks from the same fixed-offset collection does not help.

The next main experiment must collect actual policy-induced anchors and target distributions at draft-generated prefixes. Repeating more epochs or adding more uniformly spaced static blocks is not an adequate test of the main hypothesis.

## GLCS-v1 position-zero blind spot

The exact runtime cache gives a second, independent architectural limitation. GLCS-v1 treats Domino's pure parallel prefix as fixed and only scores the following 15 correction positions. On `validation_select`:

- fixed-prefix accuracy is `0.881702`;
- `139 / 1,175 = 11.83%` of blocks therefore have accepted length exactly zero before GLCS is allowed to act;
- the zero-length fraction is `22.19%` for chat, `10.97%` for code, and `2.75%` for math.

Those blocks cannot be repaired by any amount of capacity in the existing 15-position GLCS head. The next architecture must score all 16 parallel positions: select position zero from its Top-K lattice using zero-initialized global evidence, feed that selected token into Domino's causal state, and then score positions 1--15. This keeps epoch-zero behavior identical while opening a high-value part of the candidate oracle that GLCS-v1 excluded by construction.
