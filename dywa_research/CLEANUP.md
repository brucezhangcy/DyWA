# Cleanup manifest — 2026-11-12

All removals are inside `dywa_research/`. No raw CSV, YAML/config, checkpoint,
training source or research note was deleted. No important filename was renamed.

| Removed | Reason / replacement |
|---|---|
| `plot_deliverable_a.py` and `results/phase_a/deliverable_A_scaling_law.png` | Superseded single-seed presentation; `plot_multiseed.py` and its figure retained; Phase-A raw CSV retained. |
| `plot_sizeinv.py` and `results/phase_d_sizeinv/deliverable_D_control_limit.png` | Labels the confounded v1 experiment a control-limit result. Raw `sizeinv.csv`, launcher, and critique retained; use the caveated Session-4/5 analysis. |
| `plot_deliverable_e.py` and `results/bimanual/deliverable_E_single_vs_double_SCRIPTED.png` | Superseded by `analysis/s4_deliverable_e.py` and `results/session4/deliverable_E_v2_scripted_ci.png`, which explicitly label the cross-simulator comparison. Half-extents remain in `analysis/s4_deliverable_e.py`. |

Generated Python caches, Matplotlib cache, an intermediate syntax-only audit,
and duplicate regenerated Session-4 outputs were also removed. Historical log
mentions of removed images are provenance, not current artifact links. Removed
tracked content remains recoverable from the original Git commit.
