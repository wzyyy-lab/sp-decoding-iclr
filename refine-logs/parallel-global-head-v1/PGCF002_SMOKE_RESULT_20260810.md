# PGCF-002 A40 Mechanics Smoke Result

**Slurm job:** `10166796`  
**Result:** **PASS**  
**Claim boundary:** mechanics only; not an efficacy or capacity result

The reviewed PGCF-16 snapshot completed 20 BF16 optimizer steps on one A40
using 32 canonical full16 records.  The run exercised initial identity
evaluation, all three curriculum regions, backward, gradient clipping,
periodic evaluation, checkpoint selection, reload, and final evaluation.

Hard mechanics evidence:

- parameter count: `2,438,400`;
- input/output geometry: full `16 x 16` candidate lattice and one chain;
- all reported losses and gradient norms: finite;
- best checkpoint: step `20`;
- base EAL on the mechanics subset: `4.625`;
- best EAL: `4.78125`;
- harmed fraction: `0.0`;
- non-Top1 hard accuracy changed from zero to a nonzero value;
- `best.pt` and the independently reloaded `last_selected.pt` were written.

The capacity checks in this report are intentionally false because this job
uses only 20 steps and four prompts.  They are not smoke exit conditions.
PGCF-003 capacity and PGCF-005 eager profile are now authorized; disjoint
training remains blocked.
