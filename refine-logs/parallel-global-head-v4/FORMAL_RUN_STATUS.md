# PARC-16 Formal Run Status

Updated: 2026-08-11 Asia/Shanghai

## Frozen scientific path

- Architecture: one full16 input, global non-causal mixing, one parallel
  `[B,16,16]` score tensor, and one `[B,16]` chain.
- No autoregressive token feedback, target seeding, iteration, beam, tree,
  trie, forest, or multipath verification.
- Formal data: exactly 90,000 train prompts and 5,000 prompt-disjoint
  validation prompts, eight full16 anchors per retained prompt.
- Sealed held-out candidates remain label-free until a trained checkpoint is
  locked.
- Training: one 180,000-step joint DFlash + PARC run; validation every 10,000
  steps; training/audit metrics are never efficacy evidence.
- Comparator: released Domino on the same validation prompts and anchors.

## Implementation and review receipt

- Pure non-shift DFlash full16 extension is `[anchor] + 16 masks -> raw17 ->
  rows[1:17]`; released Domino remains native shift-label raw16.
- The 270K reserve pool is partitioned before label generation. Each
  split/domain group has a fixed candidate order and exact retained quota.
  Prompts that do not generate 129 pre-EOS target tokens are rejected; a part
  fails instead of publishing a reduced dataset.
- Step-0 parity uses only the frozen 5K train-audit set and cannot be selected.
  Validation is first read at step 10K for checkpoint selection.
- Scientific stop reasons are terminal and cannot be converted into resumable
  scheduler interruptions.
- Focused verification: 18 tests passed; Python compilation and both Slurm
  syntax checks passed.
- Fresh experiment-bridge review: M1 GO, M2 GO; no standalone GPU smoke.

## Active job

- Formal materialization array: Slurm `10169014`.
- Shape: 16 one-A800 tasks on `i64m1tga800u`.
- Output root:
  `artifacts/canonical/parc16_full16_opb270k_reserve_10169014`.
- Current scheduler state (2026-08-11): all tasks remain pending for A800
  priority; no task has allocated a node and no failure has occurred.
- Formal 180K-step training job: Slurm `10169018`, dependency
  `afterok:10169014`.
- Training output:
  `artifacts/models/parc16_joint_formal_10169014`.
- The dependency can launch only after all 16 collectors exit successfully;
  the trainer then independently requires exactly 90K train and 5K validation
  prompts before allocating the optimizer.

No accepted-length result exists yet. The previous same-set capacity result is
not evidence for this run and must not be compared as if it were validation.
