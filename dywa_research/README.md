# DyWA research

Measure how object size affects manipulation, explain the failures, and test when
a second arm improves reliability or cost. Bimanual necessity remains a hypothesis.

[Progress & evidence](PROGRESS.md) · [Research plan](research_prompt_size_transition.md) ·
[Open critiques](CRITIQUES.md) · [Agent workflow](AGENTS.md)

## Current state

Updated **2026-11-12**. These are re-analyses of historical data, not new simulation results.

| Evidence | What it supports | Artifact |
|---|---|---|
| Fixed-table 50% crossover radius ≈0.167 m raw / 0.173 m under the legacy immediate-episode correction | A frontier for this teacher and setup, not physical impossibility | `results/session5/frontier_stats.json` |
| Competence score ≥0.90: 45/192 selected cases, 81.1% historical macro episode success | Binary competence confidence differs from episode reliability | `results/session5/trigger_routing_audit.csv` |
| Scripted bimanual macro success ≈14.2% in another simulator | Exploratory baseline, not a matched arm-count comparison | `results/session4/deliverable_E_summary.json` |

The competence data cover 64 objects with one recorded spawn/seed per object-scale
case; repeated episodes are not independent poses. Size means enclosing **radius in
metres**, not diameter. Detailed audit and source checks are in [the log](PROGRESS.md).

## Next decision

Prepare a bounded proposal for the **four diagnostic cells** in `gpu_s5_diag.sh`.
Specify checkpoint/normalizer, exact settings, historical comparisons, GPU/time
budget, expected evidence and stopping conditions. Wait for owner approval before
execution; do not chain a sweep or fine-tune after it.

| Priority | Unresolved issue |
|---|---|
| 1 | Delayed-reset diagnostics pass CPU mocks but still need Isaac Gym baseline validation, reset-height and base/clip checks. |
| 2 | The saved v2 table fails a conservative radius-0.30 support certificate. Review real footprints and placement before changing geometry. |
| 3 | Pose replay, incomplete episodes and teacher training-range shift confound causal claims. The legacy “feasible” metric is a sensitivity calculation. |
| 4 | No matched trained-bimanual comparison or calibrated reliability trigger exists. The separate dual-arm scaffold is incomplete. |

## Working rules

- Keep edits and cleanup inside `dywa_research/` unless the owner authorizes a
  specific exception. Preserve raw results, checkpoints and meaningful filenames.
- Never launch GPU evaluations, training or remote compute without explicit owner
  approval for that run and budget. A launch flag or GitHub push approval is not
  compute permission. Do not bypass scientific gates.
- Make configuration changes only for a stated hypothesis and control. Check the
  resolved values; avoid tuning against the final test set.
- Update this current-state section and prepend one short entry to `PROGRESS.md`.
  Do not create a separate cleanup, status or session report for routine changes.

## Files and checks

| Location | Purpose |
|---|---|
| `results/phase_*`, `results/bimanual` | Preserved historical measurements |
| `results/session4`, `results/session5` | CPU re-analysis and audit artifacts |
| `analysis/`, `tests/` | CPU tools and regression tests |
| `plans/` | Outstanding implementation reviews |

From repository root, these commands use CPU only:

```bash
python3 -m unittest discover -s dywa_research/tests -v
python3 dywa_research/analysis/s5_repo_audit.py --core dywa
python3 dywa_research/analysis/s5_trigger_audit.py
```

The audit code at `6c9a9f7` passed 20 CPU regression tests; all 242 tracked core
Python files parsed. Rerun relevant checks when behavior changes, not for every
prose edit. CPU tools need NumPy, Pandas, SciPy, scikit-learn, Matplotlib, PyYAML
and CPU PyTorch; Isaac Gym is not required. Use a writable `MPLCONFIGDIR` when
needed and set `DYWA_ANALYSIS_OUT` to preserve historical re-analysis outputs.

Publishing through the existing GitHub SSH key works; HTTPS CLI authentication
failed. The owner authorized normal pushes to `main`; inspect the live remote and
preserve concurrent changes. GPU availability is not established.
