# Anticipated Critiques — Single-Arm Object-Size Frontier (Deliverable A + Phase-B)

Rigorous self-review, written to pre-empt a demanding reviewer/mentor. Each item: the criticism,
its **status** (ADDRESSED / PARTIAL / OPEN), the evidence, and the experiment that would close it.
Ordered by how badly it threatens the headline claim ("single-arm non-prehensile manipulation has a
capability frontier that collapses at object scale s* ≈ 0.17 m").

Legend: 🔴 could invalidate the claim · 🟠 weakens/qualifies · 🟢 hygiene.

---

## 0. 🔴 FIXED-TABLE geometry artifact — **RE-OPENED (Session 4): the "decisive" size-invariant test is itself confounded**
**SESSION-4 UPDATE (code reading + geometry, no new sim; details in #13/#14).** The v1 size-invariant table did NOT
hold the task constant. In `mode=valid` the spawn region is ±0.5·0.95·table_dims with NO radius contraction
(scene.py:1365-1377, `prevent_fall=False`), and the goal region is ±0.5·0.8·table_dims (push_task.py:684-692). So
scaling the table also scaled both regions. Mean spawn-to-goal distance went from 0.238 m to 0.419 m at s = 0.30.
At s = 0.30, 14% of goals and 22% of spawn CoMs are beyond the arm's 0.768 m physical reach. Meanwhile the robot
base stays welded at x = −0.5 (trossen.py:206-221 hardcodes table 0.4); see `results/session4/geometry_audit.csv`.
The collapse into timeouts in v1 is therefore *consistent with* longer and unreachable tasks, not only larger
objects. The fixed-table frontier still collapses under the feasible-only metric (s = 0.30: 0.6% ± 0.4%,
3 seeds), so "a collapse exists" stands. "It is a pure control limit" is NOT established. **Close it:**
`sizeinv_v2.sh` (one constant big table, spawn and goal regions held at the baseline, plus a distance
control).
*The original text below is kept for history. Its mechanism sentence ("placement region = 0.5·margin·table −
radius") describes TRAIN mode, not the valid-mode evals.*

### (superseded) Session-2 resolution text
UPDATE (size-invariant-table experiment): scaled the table WITH the object so the placement/OOB
region stays constant across scales. Result: immediate terminations → 0% (artifact removed), yet the
frontier STILL collapses — 93%→58%→12%→1%→0% at s=0.10/0.15/0.20/0.25/0.30, s* essentially unchanged
(~0.155). Crucially the failure mode FLIPS from OOB to **TIMEOUT** (85–98% at collapse): with ample
room and no boundary loss, the single arm simply cannot maneuver a large object to the goal pose in
time. This is the controllability limit (H2), cleanly isolated. **The table artifact was real but is
NOT the cause of the frontier.** The bimanual-necessity direction is re-validated, and #1 (necessity
fine-tuning) is now UNBLOCKED and scientifically valid to run on the size-invariant table.
Residual: timeout-dominated failure raises a fair question — would a longer episode timeout help?
(worth one control; but 0% at s=0.30 with room suggests not merely a time budget issue.)

### (historical) original concern — FIXED-TABLE geometry artifact
The DyWA table is a fixed small URDF (`table-with-wall-small.urdf`); `table_dims=[0.4,0.5]` sets
bounds only, not the physical table. Objects are scaled so bounding radius ≈ s (set_actor_scale).
Two fixed-table facts then bite exactly at the measured s*:
- Object placement/OOB region = 0.5·margin·table_dims − radius per xy side collapses to a POINT at
  **s≈0.19** (x-dim), forcing spawns to table center and immediate OOB (scene:660-677).
- Task OOB bound = ±0.5·margin·table_dims ≈ ±[0.16,0.20] (push_task:512), FIXED; but at s=0.20 the
  object radius (0.20) already equals the table half-width — the object is as big as the table.
Evidence it dominates: immediate-termination 0%→64%→89% across s=0.15/0.20/0.30; multi-seed variance
peaks only at s=0.20 (the cliff); s* (≈0.17) ≈ placement-collapse scale (0.19). **So the measured
"size frontier" is substantially a table-size artifact, not a demonstrated single-arm control limit.**
- **Close it (required before #1):** re-measure with a table that SCALES with object size (needs a
  larger table ASSET — not a pure config override, since the mesh is fixed). Only the object-to-table
  ratio held constant isolates control from table geometry. The `prevent_fall=False` test was
  inconclusive (objects use precomputed stable poses, so that knob doesn't move spawns).
- **Consequence:** #1 (necessity fine-tuning) is GATED on this — training for hours on objects that
  don't fit the table would be scientifically invalid. Deferring it is the rigorous call.

## 1. 🔴 Train–test scale gap — is this a PHYSICAL limit or just POLICY generalization? — **OPEN (runnable: `gpu_necessity_finetune.sh`, on the v2 env)**
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

## 3. 🔴→🟠 Object feasibility at large scale — **ADDRESSED for the metric; the cause is RE-ATTRIBUTED (Session 4)**
**SESSION-4 UPDATE.** The immediate (≤3-step) terminations are an *additive* artifact from a few envs that replay
one bad spawn, not a shift in how the other envs behave. Evidence (`results/session4/immediate_loop_check.csv`):
at s = 0.20 the non-immediate episode counts are 339 / 324 / 335 across seeds 0/1/2, while the immediate counts are
596 / 2 / 76. Valid mode draws ONE start pose per env for the whole eval (`num_valid_poses=1`, scene.py:256,
974-981), so a bad spawn re-terminates every few steps. Feasible-only is therefore the right fixed-table metric:
its between-seed over-dispersion is 1.5× binomial, against 16× for the raw metric. The *cause* is NOT the
"radius-contracted region collapses at s ≈ 0.19" mechanism: valid mode never applies that contraction. Two
candidates remain, both unverified: (a) the valid-mode spawn z is cloned in `create_assets` before
`scene.setup` rescales stable-pose z (scene.py:1367 vs 615-620; base.py:391-422 call order), so scaled-up
objects could spawn inside the table; (b) contact between the object and the arm at spawn. `gpu_s5_diag.sh`
dumps the data that decides between them.
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

## 4. 🟠 Single seed, no error bars — **PARTIAL (3 seeds at 4 scales; Session-4 stats)**
**SESSION-4 UPDATE** (`analysis/s4_frontier_stats.py`): 50% crossover from a binomial-logistic fit with a seed
bootstrap: raw s* = 0.167 [0.164, 0.171], feasible-only s* = 0.173 [0.171, 0.175]. Per-seed interpolation gives
0.169 ± 0.002 (raw) and 0.174 ± 0.003 (feasible). Three seeds make for a coarse bootstrap. Phase-A (seed 0) gives
s* = 0.164 from the logistic fit. At s = 0.20, binomial CIs understate the uncertainty about 16× in variance
(raw), so always report seed-level CIs. The earlier claim "binomial CI ≈ ±2%" was wrong at the cliff.
All runs are seed=0. Differences like reach 12.0→12.4% are within unknown noise; I cannot call them
distinct. Historical run-to-run noise ≈ 3 pp.
- **Close it:** ≥3 seeds per headline point; report mean ± std or bootstrap 95% CI. With ~900
  episodes/point, binomial CI on a 12% rate is ≈ ±2%, so seed variance likely dominates — must
  measure it, not assume.

## 5. 🟠 Metric ambiguity: episode-weighted vs object-balanced — **PARTIAL (mechanism now understood, #3/#13)**
Session 4: the divergence comes from looping envs (one fixed spawn per env) that emit hundreds of ≤3-step
episodes. The episode-weighted rate counts each of those episodes; the per-object rate counts each object once.
There is also a smaller bias: each env's episode still running at the end of `eval_step` is dropped, and those
are disproportionately long, i.e. timeouts. That bias makes success look slightly better in timeout-heavy cells.
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

## 11. 🟢 Small object set — **OPEN (corrected count)**
Session 4: the Phase-C per-object data contain **64 distinct object keys** (`competence_dataset.csv`),
one per env (num_env = 64). Each object is evaluated from a single spawn pose per run (#13), so per-object
success mixes object identity with that one pose.
test_set.json ≈ 12 unique objects × poses. Per-object variance is high and a couple of objects can
swing balanced_suc_rate. **Close it:** larger held-out set; report per-object breakdown + variance.

## 12. 🟢 Silent config overrides (generic) — **ADDRESSED as practice**
The force_rad/goal_radius incident proves CLI overrides can be clobbered in `__post_init__`. **Adopted
rule:** every experiment verifies the RESOLVED config from the run log, not just the CLI string.

## 13. 🔴 Valid-mode spawn mechanics — effective sample size and pose replay — **OPEN (diagnostic ready: `gpu_s5_diag.sh`)**
Found by reading the code in Session 4: `mode=valid` (used by every eval) samples ONE xy per env once
(`_valid_poses`, scene.py:1365-1377, `num_valid_poses=1`, `prevent_fall=False`) and replays it every episode. Only
the goal is resampled. Consequences: (i) the effective sample size at a scale is closer to 64 start
configurations than to the ~300–5000 episodes; (ii) a seed changes all 64 spawns at once, which explains
seed-to-seed swings at the cliff; (iii) per-object labels (Deliverable C) confound object with its spawn pose.
Possible spawn-z bug (hypothesis): `_valid_poses` is cloned before stable-pose z is rescaled.
**Close it:** run `gpu_s5_diag.sh` (it dumps per-env spawn z, xy and outcomes); for headline numbers raise
`env.single_object_scene.num_valid_poses` (for example to 8) or report results per spawn.

## 14. 🔴 Size-invariant table v1 confounds size with task distance and reach — **OPEN (fix ready: `sizeinv_v2.sh`)**
See #0. Numbers from `analysis/s4_geometry_audit.py`: at s = 0.30 the v1 table is [0.821, 0.921]. The spawn
half-x grows from 0.19 to 0.39 and the goal half-x from 0.16 to 0.33. Mean spawn-to-goal distance goes from
0.238 to 0.419 m. 13.9% of goals and 21.6% of spawn CoMs are beyond the 0.768 m EE reach (fixed table: 0%). The
robot is still welded at x = −0.5, so the table's near edge comes within 0.09 m of the base. The task timeout is
fixed at 300 steps (icra_base.yaml). There is also a possible root-tensor issue: `Trossen.reset` writes
x = −0.5·table_x − 0.3 into the root tensor (trossen.py:364-376). The EE clip box is centred on that tensor
(trossen.py:418-426), so with a non-default table the clip could sit up to 0.21 m away from the real base;
`gpu_s5_diag.sh` checks this. Also note the v1 drop at s = 0.15 (0.579 against 0.681 ± 0.012 on the fixed
table) happens at a size where the fixed table has 0% immediate terminations. That is direct evidence the v1
change made the task harder in its own right.

## 15. 🟠 Phase-B single-knob battery is within seed noise — **OPEN (re-run on v2 with ≥3 seeds)**
All battery cells are seed 0 at s = 0.20. Seed 0 has 596 looping immediate episodes, and the between-seed SD of
the raw rate is 0.079 (`results/session4/phaseB_vs_seed_noise.csv`). Every mass, reach and threshold delta is
≤ 0.69 seed-SD. So "heavier hurts monotonically" is not supported. What survives: no knob brings success
anywhere near the in-distribution ~0.93, even under the most favourable correction (success count / ~335
non-immediate episodes ≤ 0.33). The reach knob only tests the SOFTWARE clip. The byte-identical ws 0.9 and 1.2
results match the PHYSICAL reach of 0.768 m (FK of trossen_kin.py constants), which binds before either value.

## 16. 🟠 "Reach rejected (H1)" is too strong — **PARTIAL (reframe)**
On the fixed table the EE can reach world x ≤ +0.10 (clip) or ≤ +0.268 (physical). It can get behind a large
object, to the object's far side at CoM_x + s, for only 50% / 25% / 0% of spawns at s = 0.10 / 0.20 / 0.30
(`geometry_audit.csv`, `frac_spawn_far_side_within_clip`). Goals that need the object pulled toward the robot
therefore become geometrically hard for ONE fixed-base arm as s grows. That is a reach-*around* limit. It is
arguably part of the real single-arm limitation, and it is exactly what a second, opposite arm would fix. Treat
it as an explicit hypothesis (H1'), not as rejected. Test: on v2, condition success on goal direction (the diag
dump records start and goal).

## 17. 🟠 Deliverable-C predictor: leakage and yaw-only goal confound — **ADDRESSED (Session 4) with caveats**
The original `cross_val_score(cv=5)` used unshuffled row folds, with each object in both train and test. The
object-disjoint grouped CV (`analysis/s4_predictor_cv.py`) gives: scale-only AUC 0.68 (acc 0.69 against a
majority baseline of 0.667); scale + shortest scaled extent AUC 0.82 (acc 0.79); the original size + aspect/flatness
set AUC 0.82. Shortest extent adds +0.19 AUC over scale-only [0.11, 0.27] (object bootstrap). **Confound:** thin
objects get YAW-ONLY goals, an easier task (scene.py:68-70 `is_thin`, middle/smallest OBB extent ≥ 2.5,
`use_yaw_only_logic: true`). The AABB proxy for this flag alone gives AUC 0.78. Within the non-thin (full 6-D
goal) objects, bulkiness still predicts failure at every scale (Spearman −0.40 / −0.54 / −0.73). Adding it beyond
scale + yaw-only flag gives +0.06 AUC [0.006, 0.117]. Net: "shape beyond size" is real, but about two-thirds of it
is the goal-type rule rather than physics. Report both.

---

## Status summary (updated Session 4, 2026-10-17)
- **Holds:** a steep single-arm frontier exists on the fixed table under both raw and feasible-only metrics
  (s* = 0.167 [0.164, 0.171] raw, 0.173 [0.171, 0.175] feasible; 3 seeds). The threshold artifact is rejected
  (#2). The anchor is faithful (94.5%).
- **Re-opened:** "the frontier is a pure control limit" (#0/#14). The v1 size-invariant test also lengthened
  and partly removed reach from the task.
- **Highest-priority OPEN, all runnable now:** #13 diag → #14 v2 sweep → #1 necessity fine-tune
  (`gpu_s5_queue.sh`), then a trained bimanual number (`gpu_bimanual_residual.sh`).
- **Reframe:** reach-around (H1', #16) is a plausible single-arm-specific mechanism, and the one a second arm
  addresses.

### (superseded) Session-2 status summary

- **Directly established:** frontier exists & is steep (Del. A); reach rejected; inertia modulates
  but doesn't cause; anchor faithful (94.5%).
- **In progress:** threshold artifact (force_rad sweep), strict-tolerance recheck.
- **Highest-priority OPEN before any "necessity/bimanual" claim:** #1 (train–test gap) and #3
  (object feasibility). These two, plus #4 (seeds), are the minimum to make Deliverable A defensible.
