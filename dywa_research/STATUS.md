# Current research state

Updated: 2026-11-12. Keep this file short and replace stale statements when evidence changes.
History: [PROGRESS.md](PROGRESS.md). Detailed audit: [REVIEW.md](REVIEW.md).

## Objective

Measure the object-size generalization frontier of a fixed single-arm teacher,
isolate environment and policy limitations, and test whether an interpretable
arm-count decision improves reliability or cost with comparable controllers.
Bimanual necessity and a physical scaling law are hypotheses, not established facts.

## Evidence currently available

| Evidence | Interpretation | Source |
|---|---|---|
| Fixed-table 50% crossover radius ≈0.167 m raw / 0.173 m under the legacy immediate-episode correction | A frontier for this teacher and setup; not universal physical impossibility | `results/session5/frontier_stats.json` |
| Three evaluation seeds, 64-object competence set with one recorded spawn/seed per object-scale case | Sparse coverage; repeated episodes are not independent object/pose trials | `results/phase_c_multiseed/multiseed.csv`, `results/phase_c_predictor/competence_dataset.csv` |
| Competence score ≥0.90 selects 45/192 cases, historical macro episode success 81.1% | Binary competence confidence is not per-episode reliability | `results/session5/trigger_routing_audit.csv` |
| Scripted bimanual macro success ≈14.2% in another simulator | Exploratory baseline; not a fair arm-count comparison | `results/session4/deliverable_E_summary.json` |

These are re-analyses of historical data, not new simulation results. All radius
figures are enclosing object radii in metres, not diameters or arm-independent constants.

## Blockers in priority order

1. Repaired delayed-reset diagnostics have passed CPU mock tests but not Isaac Gym
   validation. Check baseline reproduction, actual reset heights and base/clip coordinates.
2. The saved v2 table fails the conservative radius-0.30 full-support certificate.
   Actual shape/orientation support and base/clip effects need review before any geometry edit.
3. Pose replay, right-censored episodes and teacher training-range shift confound
   causal claims. The legacy “feasible” metric is a sensitivity calculation.
4. No matched trained-bimanual comparison or calibrated reliability trigger exists;
   the separate Isaac Gym bimanual branch is an incomplete scaffold.

## Next useful action

Prepare a bounded proposal for the existing four diagnostic cells in
`gpu_s5_diag.sh`. State the checkpoint and normalization files, exact overrides,
baseline comparisons, allocated GPU/time budget, expected outputs and stop rules.
Do not execute it until the owner approves that particular run. Do not chain a
sweep or fine-tune after it. No fresh GPU availability is established.

Independent CPU work can inspect diagnostic bookkeeping, geometry and historical
data. Avoid repeatedly rerunning unchanged analyses whose checks already pass.

## Constraints and verified engineering state

- All research changes and cleanup stay within `dywa_research/` unless the owner
  authorizes a specific broader edit. Do not rename core files or silently alter configurations.
- Explicit GPU approval is required, including direct Python runs and remote jobs.
  Approval to push to GitHub is not permission to use a GPU.
- No GPU experiments were launched in the audit or workflow update.
- Code audit commit `6c9a9f7`: 20 CPU regression tests pass; 242 tracked core Python
  files parse. YAML and historical CSVs were byte-identical during that audit.
- GitHub publishing works through the existing SSH key. HTTPS CLI authentication
  failed; do not repeatedly retry it or change credentials without user involvement.
- The owner authorized publishing these reviewed research changes to `main`.
  Check the live remote before further pushes; never force-push over concurrent work.
