# FMAS Gate-2 Result-to-Claim

**Verdict:** `claim_supported = no`  
**Binding decision:** Gate 2 FAIL-CLOSE; close flat 226-way canonical-action CE.

## Integrity and protocol

- Job `10133114` completed `0:0` in `00:19:23`; stderr is empty.
- Exact budget: `3 × 12,407 = 37,221` optimizer steps.
- Train identity: 99,356 prompts, 793,989 blocks, prompt-set SHA256
  `45471a62f93a488f3f7653c096bebcddb0ddae3773f6c99744bd070e348a9405`.
- Selection identity: physically isolated 147 prompts / 1,175 blocks;
  metadata SHA256
  `b63be7bbfd56651aadbee57a819bfe0afb39395b1601b5ea4fc1564cc9f933d7`.
- Metrics SHA256:
  `be31fbb731423ee701750d9dccf83259023f220cc0e7e250559454bcfffd658c`.
- All trainer/head/data source hashes matched at run start and end. The frozen
  checkpoint rule correctly retained epoch 0.

## Binding result

| epoch | validation CE | action accuracy | Δ EAL vs DFlash | harmed fraction |
|---:|---:|---:|---:|---:|
| 0 selected | 4.12094 | 16.255% | 0.00000 | 0.00% |
| 1 | 2.70167 | 32.255% | -0.50486 | 34.979% |
| 2 | 2.58212 | 33.532% | -0.48567 | 33.277% |
| 3 | 2.53864 | 33.191% | -0.42396 | 31.660% |

The selected identity checkpoint has prompt-balanced EAL `5.11200194`, so it
already fails the absolute frozen requirement
`EAL_FMAS - EAL_DFlash > 0.28499`. Exact Direct-native and Direct-one-edit
controls cannot reverse this failure; no matched Direct result is needed to
close the FMAS route.

## Mechanistic diagnosis

Independent reconstruction of all `1,175 × 225 = 264,375` edit actions gives:

- 984 beneficial actions (`0.3722%`), mean gain `+1.829` accepted tokens;
- 90,120 harmful actions (`34.0879%`), mean cost `-5.304` tokens;
- 173,271 neutral actions (`65.5399%`).

The canonical labels are 984 edit, 115 full-correct KEEP, and 76 out-of-K
KEEP. Flat CE names one canonical action but treats every other action as the
same kind of error. Under the observed imperfect classification, it therefore
does not distinguish a neutral wrong edit from an early edit costing more than
five tokens on average. The simultaneous CE/action-accuracy improvement and
EAL/harm degradation is direct evidence of a cost-insensitive
surrogate/utility mismatch.

This is not a classic train-versus-validation overfit diagnosis: the training
objective and its held-out counterpart improve together. Nor is it evidence
for an information ceiling or insufficient architecture capacity: the same
D64 class exactly fit the frozen 512-block action mapping, while that
same-subset witness cannot establish full-distribution generalization.

## Supported and unsupported claims

Supported:

- Under the frozen D64, full OPB, three-epoch protocol, flat canonical-action
  CE fails Gate 2 and is unsafe away from its identity checkpoint.
- Better 226-class accuracy is not a valid proxy for EAL or harm in this
  action space.

Unsupported:

- no superiority over DFlash, Direct-native, Direct-one-edit, or Domino;
- no broad claim that first-miss action selection is impossible;
- no frozen-feature information-ceiling, model-capacity, throughput, formal
  generalization, or paper-level method claim.

## Routing

Seeds 1/2, formal-test access, D640 reinterpretation, and every further
full-data flat-CE GPU run are forbidden. Pending evaluator job `10133115` was
cancelled before allocation because the absolute DFlash gate had already
failed. CPU postmortem and a new mechanism are allowed, but a new GPU route
must use a separately frozen protocol and pass CPU semantics, a signed-utility
capacity gate, fresh code review, physical split isolation, and frozen
checkpoint/decoder selection. Threshold relaxation or post-hoc calibration of
this failed route is not allowed.
