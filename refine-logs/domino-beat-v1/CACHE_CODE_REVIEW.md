# Same-anchor cache code review

Fresh-agent review found the anchor/context/gold definitions, first-token base
top-1, later causal GRU rollout, and released checkpoint alignment correct.

One blocking issue was found and fixed before GPU execution: `.eval()` had not
disabled autograd, so buffered cached features could retain target and Domino
GPU graphs until shard flush.  `main` now runs under inference mode, cached
tensors are explicitly detached, and the writer rejects tensors requiring
gradients.

Useful non-formal fixes were also landed:

- validate Domino projector, shift-label, pure-prefix, and B16/15-position
  alignment;
- report prompt-balanced as well as round-weighted released EAL;
- write to a job-specific incomplete directory and rename only after complete
  metadata is present;
- test the no-gradient writer contract and the 15/16 alignment.

No hash checks were added.  Compilation, focused tests, and Slurm syntax pass.
