# DyWA research audit — 2026-11-12

The fixed-table size frontier remains a useful empirical result. The evidence does
not yet establish that two arms are physically necessary, or a trigger with 90%
per-episode reliability. This review improves the research tooling and corrects
several routes to misleading conclusions without changing the core simulator,
model, YAML files, experimental scale ranges, or raw measurements.

## Scope and provenance

- Repository: `brucezhangcy/DyWA`.
- Research base: open PR #1, branch `cloud-session-4`, commit
  `b234229f24308e1e93f30d962d43f1cbf92eeeb7`.
- `main` is older (`f9beeccf11b47ffde1f2e1f1e165a4b1c7073a9b`). Applying this patch
  directly to main is not equivalent to continuing the current research branch.
- Improvements are local on `research/session5-audit`; no remote branch was changed.
- Every tracked edit/addition/deletion is inside `dywa_research/`. No important
  filename was renamed. See [CLEANUP.md](CLEANUP.md) for the six obsolete files removed.
- No GPU experiment was launched. The one read-only SSH availability check timed
  out; no lab-machine file was changed. GitHub's stored CLI login is invalid.

## Findings and repairs

| Priority | Finding | Action and remaining limit |
|---|---|---|
| High | The pending diagnostic reads a terminal root/goal as the next episode's start. `EnvBase` resets at the beginning of the following step. | Replaced that assumption with an observational `reset_indexed` hook, after pose assignment and before physics. Tracks exact starts/goals, asynchronous resets, lengths, incomplete episodes and errors. CPU trajectory equivalence verified; Isaac Gym verification still pending. |
| High | A failed run can be followed by more cells, the necessity script can return success after training fails, and final queue eval failure can be hidden by `queue DONE`. | Propagate failures, validate NPZ against taxonomy counts, reject missing/corrupt/nonfinite outputs, skip only `OK` cells, and preserve prior attempts before retrying. |
| High | Copying the global `/tmp/per-cat-suc.pkl` can attach another run's or an earlier run's labels. | Diagnostic entry now writes a cell-local pickle using `PER_OBJECT_OUT`; the S5 helper no longer copies the shared file. Historical launchers remain historical protocols. |
| High | The v2 table is insufficient to guarantee full support of radius-0.30 objects over the baseline spawn region. | Added a conservative CPU geometry gate. Kept all table/margin/scale settings unchanged; default large-radius v2 and fine-tune are intentionally blocked pending geometry review. This is a failed support guarantee, not proof that every such shape falls. |
| High | Binary competence confidence is interpreted as per-episode success probability. | Added object-disjoint routing audit. At score ≥0.90: 45/192 object-scale cases (38 objects), historical macro episode success **81.1%**, binary competence precision **91.1%**. These are distinct quantities. Not a calibrated deployable trigger. |
| Medium | The original predictor used row CV and globally fitted preprocessing; object identity crossed folds. | Its CV now uses object-disjoint folds and fold-local scaling. The newer S4 grouped analysis was already object-disjoint and was rerun. |
| Medium | The S4 frontier bootstrap resampled seeds separately within every scale, breaking paired seed-curve dependence. | Resample complete seed curves together. Point estimates unchanged; updated intervals below. Three seeds still give weak uncertainty estimates. |
| Medium | A bimanual plotting interpolation silently clamps radii below its observed support. | Return missing outside either end of the observed range. No simulator or configuration changed. |
| Medium | The displayed ROC's AUC label referred to a different aggregation of predictions. | Label now uses the actual plotted ensemble out-of-fold predictions. Ensemble OOF metrics remain descriptive; training sets overlap across folds. |
| Medium | Radius-grid generation parses Python values with `eval`. | Use AST + `literal_eval`; optional `--output` allows non-destructive verification. Generated YAML is semantically identical to the committed file; YAML was not changed. |
| Medium | GPU queue automatically defaults to all long experiments and guesses the first checkpoint directory. | All 14 research shell entry points require a human-supplied approval flag; queue defaults to diagnostics only; final eval requires an exact `FINETUNED_CKPT`. Bimanual training requires an explicit output glob. |

The approval flag is a launch interlock, not proof that permission was given. An
agent must still obtain the owner's explicit approval for the particular GPU
experiment and budget before using it. Direct Python entries also remain subject
to that instruction. No approval flag was supplied to a real GPU command here.

## Geometry: what the CPU calculation actually establishes

The pending v2 script uses table dimensions `[0.821, 1.026]` m and baseline spawn
half-extents approximately `[0.190, 0.23744]` m. With an enclosing radius of 0.30 m,
worst-case full-support clearances are **−0.0795 m in x** and **−0.02444 m in y**.
A sufficient table size, preserving those centres, is at least
`[0.980, 1.07488]` m before any additional clearance.

About **47.8%** of uniformly sampled spawn centres fail this enclosing-sphere
support certificate; **30.9%** of goal centres fail it. These are analytical
certificate fractions, **not measured failure/fall rates**. Real object footprint
and orientation can be much smaller than its enclosing sphere. The numbers do
not authorize a configuration change: table size also affects root-tensor and
software-clip interactions, which need inspection in the approved diagnostic.

The valid-pose clone also occurs before stable-pose z is rescaled. The code path
can leave a stale reset height when scale differs from one. Its actual penetration
and outcome effects have not been established by simulation here.

## What the historical data support

Whole-seed-curve bootstrap of the same binomial-logistic fit:

| Metric | 50% crossover radius | Exploratory 95% bootstrap interval |
|---|---:|---:|
| Raw episode-weighted success | 0.16707 m | [0.16498, 0.16998] m |
| Legacy immediate-filter sensitivity | 0.17294 m | [0.17003, 0.17563] m |

The second metric historically removes all ≤3-step episodes from its denominator
but retains their successes in its numerator. It should not be called exact
physical feasibility. The new NPZ report filters both numerator and denominator
and reports immediate successes explicitly. Existing raw tables were preserved.

The 64-object/192-row competence dataset has one recorded spawn/seed per
object-scale case and lacks episode denominators. The 81.1% routing result is a
mean of historical per-object rates, not an independently measured deployment
rate or a confidence bound. It does not test bimanual routing success.

The scripted bimanual number remains **14.2% macro**, from a different simulator,
objects and criteria. The CPU re-analysis reproduces the existing result; it is
not a matched experiment, trained-bimanual result, or a necessity proof. A failed
finite-budget single-arm fine-tune would also not prove physical impossibility.

## Checks completed

- Static parse of **all 242 tracked core Python files**, and `bash -n` on six core
  shell files. No syntax errors; no core source edited.
- Parse and shell syntax checks for the research folder; see the generated
  `results/session5/repository_audit.json` for current counts.
- **20 CPU regression tests pass**: delayed and partial reset handling, exact
  start/goal association, unchanged mock rollout, immutable snapshots, error
  detection, cell consistency, missing/corrupt results, all 14 shell launch locks,
  queue failure propagation, support gate, and configuration-preserving radius grid.
- S4 frontier, grouped predictor, and scripted bimanual re-analyses run to
  completion. Original Session-4 outputs remain unchanged. Byte-identical
  regenerated outputs were removed instead of accumulating duplicate results.
- New held-out trigger audit runs on CPU; its plot was visually inspected.
- Historical Phase-A/B/C/D CSVs pass structural/range/status/key checks and have
  SHA-256 fingerprints in the audit report. This does not establish data authenticity
  or recover omitted per-episode information.
- The separate `bimanual-dywa` scaffold remains unready: duplicate actor lookup,
  overlapping bases, shared configuration, hardcoded DOF slices and target-buffer
  layout remain unresolved. Existing `plans/bimanual_scaffold_review.md` is still
  applicable; no core edits were made to disguise these as solved.

## Literature spot-check

Primary arXiv records confirm the identities of
[DyWA](https://arxiv.org/abs/2503.16806),
[BiNoMaP](https://arxiv.org/abs/2509.21256), and
[DART](https://arxiv.org/abs/2604.17833). BiNoMaP's current record describes RL-free,
geometry-parameterized bimanual primitives; DART concerns objects on a tray.
[RT-Affordance](https://arxiv.org/abs/2411.02704) describes affordance-plan-conditioned
policies; citing it as direct evidence of a calibrated arm-count success predictor
is too strong. This was a source-identity/abstract spot-check, not an exhaustive
novelty search. “First”, “provably better”, and “bimanual necessity” remain proposed
claims, not established results.

## Next work, requiring owner approval before any GPU execution

1. Approve only the four-cell diagnostic first. Check that the repaired recorder
   reproduces historical baseline outcomes and that raw camera/tensor readings
   agree about spawn heights and robot base/clip coordinates. No long queue yet.
2. Review actual object footprints, fixed spawn/goal regions, table clearance and
   base placement together. Agree on the geometry change before editing core
   configuration. Compare a small-object anchor in the proposed environment.
3. Use a fixed episode budget per object and multiple spawn poses/seeds, plus
   exact start and outcome records. Report right-censoring and immediate outcomes.
4. Only after those gates: approve the controlled scale sweep, small-DR versus
   large-DR training, and matched trained bimanual comparison as separate budgets.
5. Fit a per-episode or count-weighted capability model; calibrate and select its
   routing threshold on separate object groups, then evaluate on untouched groups.
   Define a fallback when neither controller reaches the reliability target.

Until then, use: “This fixed teacher loses generalization at increasing object
scale in the tested setup.” Keep the physical-cause and arm-necessity questions open.
