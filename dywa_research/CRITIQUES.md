# Anticipated Critiques — Single-Arm Object-Size Frontier (Deliverable A + Phase-B)

Rigorous self-review, written to pre-empt a demanding reviewer/mentor. Each item: the criticism,
its **status** (ADDRESSED / PARTIAL / OPEN), the evidence, and the experiment that would close it.
Ordered by how badly it threatens the headline claim ("single-arm non-prehensile manipulation has a
capability frontier that collapses at object scale s* ≈ 0.17 m").

Legend: 🔴 could invalidate the claim · 🟠 weakens/qualifies · 🟢 hygiene.

---

## 1. 🔴 Train–test scale gap — is this a PHYSICAL limit or just POLICY generalization? — **OPEN (biggest threat)**
The teacher trained on scale DR **[0.05, 0.10]**. Every collapse point (s ≥ 0.15) is **out of
distribution**. So the curve may measure "this teacher never saw big objects," not "one arm
physically cannot." The whole thesis (bimanual *necessity*) rests on this distinction.
- **Close it (necessity experiment):** train/fine-tune a teacher WITH large objects in the DR range
  (e.g., scale up to 0.30) and show it STILL collapses above s*. Only then is the limit physical.
  Until run, the honest claim is bounded: *"a teacher trained at [0.05,0.10] fails to generalize
  beyond s*≈0.17, and neither reach nor mass explains it."*
- Partial evidence it's not pure OOD confusion: the Phase-B battery shows a smooth, physically
  ordered mass response (heavier→worse), not random flailing — consistent with a real control limit.

## 2. 🔴→🟢 Threshold-as-fraction confound — **ADDRESSED (rejected)**
Success = pos_err ≤ goal_radius (∧ orn_err ≤ goal_angle), goal_radius set by `force_rad`. At s=0.20
the tolerance is a *smaller fraction* of a big object, so "harder when big" could be a tighter-goal
artifact.
- **Correction found:** Deliverable A ran at force_rad=0.10 (not the 0.05 I first logged);
  `force_rad` overrides `task.goal_radius` in `__post_init__`. Tolerance was a constant 0.10 m across
  all scales, so the SHAPE is valid; absolute level is at the looser 0.10.
- **Tested (proper knob = force_rad), s=0.20:** suc_rate at force_rad {0.05,0.10,0.15,0.20} =
  {0.157, 0.120, 0.106, 0.142} — **no recovery even at a 0.20 m goal larger than the object.** The
  collapse is threshold-ROBUST. (Non-monotonicity across these ≈ noise → reinforces the need for
  seeds, #4.)
- **Strict-tolerance anchor:** in-dist s=0.10 at the tight force_rad=0.05 still gives **90.8%**
  (balanced 86.6%) — the reproduction survives the strict criterion; it wasn't a loose-goal illusion.
- **Remaining nuance:** a size-RELATIVE tolerance (force_rad=k·s) would hold the goal a constant
  *fraction* of object size; worth one pass to re-derive s*, but the absolute-tolerance robustness
  already makes an artifact unlikely.

## 3. 🔴→🟠 Object feasibility at large scale — **ADDRESSED (confounded confirmed + corrected)**
Concern: at large scale, infeasible spawns/goals cause failures unrelated to *control*.
- **Measured (taxonomy sweep):** fraction of episodes ending in ≤3 steps ("immediate") is 0% for
  s ≤ 0.15, then spikes to **63.7% at s=0.20 and 89.2% at s=0.30** — a real contamination that
  appears exactly in the collapse region (workspace contracts by object radius ⇒ big objects spawn
  at/over the boundary and instantly go OOB).
- **Corrected frontier (feasible-only = success / (done − immediate)):** s=0.20 raw 12.0% →
  **33.0%**; s=0.30 raw 0.09% → 0.9%. So the raw curve OVERSTATED the collapse; s* shifts right
  (feasible 50%-crossover ≈ 0.18 vs raw 0.167). **The collapse SURVIVES the correction** (33%→0.9%),
  so it is real, just less extreme. Both curves are now plotted (deliverable_B_taxonomy.png).
- **Residual caveat:** "immediate ≤3 steps" is a heuristic; some instant OOB is genuine control loss
  (arm knocks a big object out), so feasible-only may slightly OVER-correct. Honest frontier lies
  between raw and feasible. A per-object bbox-vs-workspace admissibility filter would tighten this.

## 4. 🟠 Single seed, no error bars — **OPEN**
All runs are seed=0. Differences like reach 12.0→12.4% are within unknown noise; I cannot call them
distinct. Historical run-to-run noise ≈ 3 pp.
- **Close it:** ≥3 seeds per headline point; report mean ± std or bootstrap 95% CI. With ~900
  episodes/point, binomial CI on a 12% rate is ≈ ±2%, so seed variance likely dominates — must
  measure it, not assume.

## 5. 🟠 Metric ambiguity: episode-weighted vs object-balanced — **PARTIAL**
suc_rate (episode-weighted) and balanced_suc_rate (object-mean) diverge sharply (s=0.20: 0.120 vs
0.278). Episode-weighting over-counts objects that generate many short failing episodes, biasing the
number DOWN. Which is "the" frontier?
- **Reported both** in all tables. **Close it:** fix a *constant number of episodes per object*
  (evaluate N poses × M objects, not a step budget) so the two metrics converge and denominators are
  identical across conditions. Current variable episode counts (633–1168 at s=0.20) are a real flaw.

## 6. 🟠 Unphysical mass model — **PARTIAL**
Mass is decoupled from volume in the pipeline (sampled uniformly, independent of scale). Deliverable A
therefore holds big objects unrealistically light (0.1–0.5 kg). A reviewer will ask for volume-scaled
mass (m ∝ ρ·s³).
- Phase-B brackets mass 0.05–0.80 kg at fixed size (heavier→worse), so we know the direction.
- **Close it:** add a `target_mass ∝ s³` condition (needs a small env patch, since the pipeline
  can't couple them out of the box) and re-run the frontier; expect it to move s* LEFT (stricter).

## 7. 🟠 Single robot / task / simulator / teacher — **OPEN (external validity)**
One arm (Trossen WXAI), one task (6-DOF push), one sim (Isaac Gym), one checkpoint, teacher (not
student). s* may be arm-specific (reach, 6-DOF, gripper width) rather than a property of "single-arm
non-prehensile manipulation."
- **Close it:** replicate the frontier on the Franka (paper's main robot) and on the deployable
  student; if s* is qualitatively similar, the claim generalizes. Note real-robot transfer is a
  separate, larger question.

## 8. 🟠 Isotropic-scaling simplification — **OPEN**
Only a single scalar scale is swept (set_actor_scale takes one factor). Real "large objects" differ
in aspect ratio; a wide flat tray and a compact cube of equal scale fail via different mechanisms
(toppling vs reach vs graspless control). The 1-D scale axis collapses that.
- **Close it:** correlate per-object success with geometric features (max cross-section, height,
  aspect ratio, volume) at fixed scale; report which feature best predicts failure. (This is also the
  honest, paper-free version of a "capability predictor" — a regression on physical features, not a
  borrowed method.)

## 9. 🟢 Eval differs from the repo's canonical path — **ADDRESSED, documented**
Results come from a patched entry (`valid_ppo_arm_patched.py`), dependency shims, and a `urdf_fixed`
path override. Risk: results could be artifacts of these patches.
- The patch is a 1-line arg fix + tuple-unpack (behavior-neutral); shims implement standard attention;
  `urdf_fixed` rewrites only path *strings* (geometry byte-identical). **Anchor reproduces 94.5%**,
  matching the independent 93.1% training number — strong evidence the harness is faithful. All
  documented in the progress log for exact reproduction.

## 10. 🟢 Unexplained `*** Can't create empty tensor` at run end — **PARTIAL**
Prints once, AFTER the result dict, on every run. Appears to be a teardown artifact (results already
emitted), but a reviewer will flag any error string.
- **Close it:** trace its origin (likely gymtorch teardown with an empty per-object buffer) and
  confirm it never precedes/affects the stats.

## 11. 🟢 Small object set — **OPEN**
test_set.json ≈ 12 unique objects × poses. Per-object variance is high and a couple of objects can
swing balanced_suc_rate. **Close it:** larger held-out set; report per-object breakdown + variance.

## 12. 🟢 Silent config overrides (generic) — **ADDRESSED as practice**
The force_rad/goal_radius incident proves CLI overrides can be clobbered in `__post_init__`. **Adopted
rule:** every experiment verifies the RESOLVED config from the run log, not just the CLI string.

---

## Status summary
- **Directly established:** frontier exists & is steep (Del. A); reach rejected; inertia modulates
  but doesn't cause; anchor faithful (94.5%).
- **In progress:** threshold artifact (force_rad sweep), strict-tolerance recheck.
- **Highest-priority OPEN before any "necessity/bimanual" claim:** #1 (train–test gap) and #3
  (object feasibility). These two, plus #4 (seeds), are the minimum to make Deliverable A defensible.
