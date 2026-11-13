# DyWA research attempts

Start with [REVIEW.md](REVIEW.md) for the current evidence, repairs, unresolved
issues and approval-gated next steps. [CLEANUP.md](CLEANUP.md) lists removals.
Historical logs remain historical; newer corrections take precedence over their
superseded conclusions.

- `results/phase_*` and `results/bimanual`: preserved raw historical measurements.
- `results/session4`: preserved previous CPU re-analysis.
- `results/session5`: current audit, corrected uncertainty and trigger diagnostics.
- `analysis/`: CPU analysis and validation tools.
- `tests/`: CPU regression tests with a delayed-reset test environment.
- `plans/`: outstanding bimanual scaffold review.

From repository root, these commands use CPU only:

```bash
python3 -m unittest discover -s dywa_research/tests -v
python3 dywa_research/analysis/s5_repo_audit.py --core dywa
python3 dywa_research/analysis/s5_trigger_audit.py
```

The CPU analyses need NumPy, Pandas, SciPy, scikit-learn, Matplotlib, PyYAML and
CPU PyTorch. They do not need Isaac Gym. For plots, point `MPLCONFIGDIR` to a
writable scratch directory if necessary. Set `DYWA_ANALYSIS_OUT` to a new results
folder when rerunning the S4 scripts to preserve historical outputs.

No GPU run is authorized by this README. All research shell launchers refuse to
run without an explicit human-supplied approval flag; direct Python simulator
entries still require owner permission. The current large-radius v2 geometry
also fails the new conservative support gate; settings were deliberately preserved.
