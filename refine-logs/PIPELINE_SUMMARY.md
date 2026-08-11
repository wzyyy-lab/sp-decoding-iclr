# Research-Refine Pipeline Summary

**Anchored problem:** recover more real accepted draft tokens from the already computed DFlash K16 lattice without tree verification, sequential correction, or target adaptation.

**Current selected route:** base-anchored global full-lattice reranking. Training is re-frozen to the project's established smoothed Candidate-D-PACE `alpha=.5`; the representation screen will decide additive versus explicit compatibility nodes.

**Why the old result may be weak:** remote candidate hypotheses are collapsed before cross-position interaction, and candidate/context interactions are mainly additive. These are hypotheses, not conclusions, until R030-R032.

**What the capacity rescue established:** the first proposed unsmoothed accepted-reach objective starves hard alternatives of gradient. A longer-budget ARR control failed, while both compatibility and additive encoders passed all capacity thresholds with smoothed Candidate-D-PACE. ARR and training-safety claims are permanently removed from the main method; the structure route remains testable.

**Artifacts:**

- Method: `refine-logs/FINAL_PROPOSAL.md`
- Claims/data contract: `idea-stage/docs/research_contract.md`
- Plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Capacity diagnosis: `refine-logs/CAPACITY_FAILURE_DIAGNOSIS.md`
- Review history: `refine-logs/round-1-review.md` through `round-3-review.md`
- Compute contract: `.aris/compute/env-spec.json`, `.aris/compute/slurm.md`

**Execution order:** preflight complete → capacity diagnosis/rescue complete → matched representation screen R030-R033 → only on positive evidence, matched scopes/seeds and independent confirmation.

**Current status:** implementation and environment validated; the method/contract have been re-frozen to match the binding negative evidence; R030-R032 tooling and review are next.
