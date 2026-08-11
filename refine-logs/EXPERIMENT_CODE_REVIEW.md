# Experiment Code Review

**Reviewer:** GPT-5.6-Sol xhigh, same-family provisional  
**Scope:** GCLS-v2 selector/loss/data/trainer/tests and R001/R010–R023 Slurm entrypoints  
**Initial verdict:** R001 NO-GO pending one documentation blocker; R010–R012 additionally blocked on aggregate gate semantics  
**Re-review:** all blockers closed; R001 and R010–R013 GO

## Correctness findings

The reviewer found no defect in accepted-reach math, candidate-support censoring, safe gather, float32 survival, block-balanced safety, canonical gold-ground-truth evaluation, prompt-disjoint logic, frozen inputs, metric computation, seeds, result paths, atomic artifacts or epoch-zero runtime identity.

It independently verified targeted tests and compilation, Slurm syntax and partitions, the exact 1,235,808 compact parameter count, target fingerprint and assets, and a real legacy canonical batch of shape (16,15,16) with sorted candidates and no historical argmax witness.

## Initial blocking findings

| ID | Finding | Required resolution |
|---|---|---|
| B1 | Flat blocks contain trainable relative-position and same-position biases initialized at log L but the proposal did not state them | Document shape, initialization, trainability and matched-scope use, or remove |
| B2 | Each lambda task used a hard capacity exit, making an expected negative condition look like an array crash although the stage rule is “at least one passes” | Make tasks persist metrics; aggregate after all three and distinguish artifact error from scientific negative |
| B3 | Code required candidate accuracy ≥99% and hard accuracy ≥97%, but the written gate only named repair/gap/harm; selection could overwrite a passing epoch | Freeze all thresholds and retain a gate-passing epoch before EAL tie-breaking, or remove extra checks |

## Non-blocking findings

- Legacy canonical rank zero is the stored float32-topk action but has no independent base_greedy_ids witness; report none_legacy versus complete, reject mixed collections, and require witness in new data.
- Add successful and deliberately corrupted runtime identity tests.
- Add automated objective-screen lambda selection before interpreting M2.
- Monitor objective-screen shared-filesystem load and wall time.

## Fixes applied

- B1: documented both trainable biases, initialization, rationale and scope matching in the final proposal, experiment plan and research contract.
- B2: removed per-task hard exit; added summarize_gcls_v2_capacity.py plus a CPU summary job. It requires all three artifacts, returns exit 2 for missing/corrupt results and exit 1 for a complete scientific negative.
- B3: froze the complete five-metric gate; added capacity-aware checkpoint ordering that retains a passing epoch before EAL tie-break.
- Legacy provenance: the collection loader rejects mixed witness status, validates complete witnesses and exports base_greedy_witness_status into metrics provenance.
- Tests: added finite-difference reach, bf16-to-float32, successful/corrupted identity, mixed-witness and passing-epoch retention checks.

## Post-fix verification

- 43/43 targeted tests passed.
- All reviewed Python files compile.
- Four Slurm scripts pass bash syntax checks.
- Git whitespace checks pass.

## Re-review result

No remaining blocking issues. The reviewer verified B1–B3 closure, collection-level legacy provenance, positive/negative runtime identity, 46/46 tests, Python compilation, four shell syntax checks, Git whitespace checks and the R013 CPU partition.

- **R001 GPU smoke:** GO
- **R010–R012 capacity array:** GO
- **R013 capacity summary:** GO after the array with the capacity job ID exported

Review independence remains same-family and acceptance is provisional.

## Post-capacity representation-screen review

After the binding ARR failure, the method was re-frozen to smoothed Candidate-D-PACE and a three-cell representation screen. A fresh read-only review initially returned NO-GO with five blockers:

1. treatment-dependent module construction changed shared parameter initialization and the later shuffle RNG;
2. axial-vs-flat changed more than prepooling, so a pure prepool causal claim was invalid;
3. malformed top-level artifact fields could escape as exit 1 instead of artifact-error exit 2;
4. frozen development thresholds were still CLI-overridable;
5. the A40 debug 30-minute wall time had inadequate I/O/training margin.

Fixes:

- added deterministic named initialization derived from run seed, restored the process CPU RNG after construction, and tested bitwise-identical flat shared parameters plus axial/flat common input parameters;
- narrowed the claim everywhere to a coupled flat full-lattice mixer comparison against the axial topology baseline;
- normalized missing/malformed schema exceptions to exit 2 and added CLI exit-code tests;
- removed threshold overrides and fixed strict raw delta `> .285` and first-token tolerance `.001` in code;
- moved the screen to `i64m1tga40u` with a two-hour limit.

Post-fix targeted verification: 47 tests passed, Python compilation, both Slurm syntax checks and whitespace checks passed. The one-time re-review found no remaining blocker and issued **GO for R030-R033**.

### Queue-safe materialized-subset amendment

The standard A40 queue had 24/24 GPUs occupied and roughly 30 runnable jobs ahead. A first fallback review rejected naive pre-materialization because source SHA was not checked, actual record IDs were not proven equal to collection logs, metadata pointed at only one source manifest, and `afterok` could hide a materialization failure.

The amended materializer now:

- reads each source shard once, checks exact bytes and metadata SHA256, then decodes the same in-memory payload;
- requires the complete scanned record prompt set to equal the 99,356 IDs recovered from collection logs and checks every manifest domain;
- emits an exact cross-part 25K manifest in original order and binds its hash in metadata/provenance;
- emits JSON `artifact_error` and exit 2 on any materialization exception;
- requires explicit post-job audit before the GPU array is submitted, rather than an opaque `afterok` chain.

The reviewer verified unchanged selection, order, 199,818 blocks, 28,107 steps, split and budget; 30-minute debug A40 headroom was judged reasonable after 4x I/O reduction. Final verdict: **GO for materialize → audit → R030-R033**. The queued standard-partition jobs 10132361/10132364/10132365 were canceled before execution with no artifacts, avoiding duplicate model selection.
