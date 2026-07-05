# Research Prompt — Object-Size Transition & Explicit Single-Arm vs. Bimanual Triggering for Non-Prehensile Manipulation

> **Role of this document.** This is a self-contained research prompt to hand to a capable
> coding + research agent operating inside the `DyWA` repo (with read access to the sibling
> `Aloha_real/sim_3d_bimanual-main` MuJoCo bimanual stack). It defines a publishable research
> program, not a single task. Work the phases in order; each phase has a falsifiable hypothesis,
> a concrete experiment grounded in this codebase, and a deliverable. Report negative results
> honestly — a clean characterization of *why* single-arm non-prehensile manipulation fails on
> large objects is itself a contribution.

---

## 0. One-sentence thesis

> *In non-prehensile manipulation, a single arm has a capability frontier that collapses as object
> size/mass grows; we (a) measure that frontier as a scaling law, (b) explain it mechanistically,
> and (c) break it with an **explicit, interpretable trigger** that decides — per object — whether
> to deploy one arm or two, plus a bimanual non-prehensile policy that extends the reachable object
> regime. The headline claim is a Pareto result: single-arm is sufficient and cheaper for small
> objects, bimanual is **necessary** above a measured crossover size, and an explicit trigger
> recovers the upper envelope of both.*

Why this is novel and publishable: DyWA (arXiv 2503.16806) and its predecessor CORN are
**single-arm** non-prehensile systems evaluated on objects of roughly fixed, small scale
(`scale ∈ [0.05, 0.15] m`). No prior work characterizes the single-arm size frontier for
non-prehensile pushing, and none provides an explicit, learned arm-count decision feeding a
bimanual non-prehensile controller. The "minimal-effectors" decision (use the fewest arms that
can do the job) is a clean, interpretable systems contribution on top of a measured physical law.

---

## 1. Grounding facts (already established in this repo — do not re-derive)

| Fact | Source |
|---|---|
| Single Trossen WXAI 6-DOF teacher (PPO) reaches **93.1% `cur_suc_rate` @ 500k** on the standard DGN object set | `dev_log.md:1664` |
| Decisive enabler was widening `ws_bound` so the EE reaches the table (not IK, not algorithm) | `dev_log.md:1529`, `1612` |
| Object scale set by `env.single_object_scene.{use_scale_dr,min_scale,max_scale}` = `{true, 0.05, 0.10}` (m) | `dywa/src/data/cfg/env/icra_base.yaml:29-34` |
| Scale applied as `set_actor_scale(scale = target_size / object_radius)`; propagates to radii, hulls, point clouds, bboxes | `dywa/src/env/common.py:74-79`, `scene/tabletop_with_object_scene.py:586-620` |
| **Success threshold `task.goal_radius = 0.05 m` is FIXED for all object sizes** (`sample_thresh=[1,1]`) — a confound | `dywa/src/data/cfg/env/icra_base.yaml:44`, `env/task/push_task.py:754` |
| Workspace bounds *contract* by object radius: `min_bound += radius; max_bound -= radius` | `scene/tabletop_with_object_scene.py:660-677` |
| Env is hardcoded single-arm: one robot actor per env (`robot = Trossen(cfg)`; one `create_actor`) | `dywa/src/env/push_env.py:133`, `robot/trossen.py:193-220` |
| Out-of-bounds (`oob`) terminates when the object leaves workspace bounds | `dev_log.md:1378` |
| Bimanual already exists for the **same arm** in MuJoCo: 2× Trossen WXAI, manager-based RL, `lift_cube` task | `Aloha_real/sim_3d_bimanual-main/.../trossen_ai_bimanual.xml`, `tasks/manipulation/lift_cube_env_cfg.py` |

**Critical methodological warning (act on this in Phase A):** because `goal_radius` is fixed at
0.05 m, "harder for big objects" partly reflects the threshold being a smaller *fraction* of a big
object, not a true capability limit. Every scaling result MUST be reported under at least two
threshold regimes: (i) **absolute** fixed `goal_radius=0.05`, and (ii) **size-relative**
`goal_radius = k · object_radius` (sweep `k`). If the frontier collapses under *both*, the limit is
physical, not an artifact. This control is part of the contribution.

---

## 2. Physical hypothesis (the "why")

A single non-prehensile contact applies a wrench through one patch. To drive a rigid object to a 6D
goal you must control translation **and** rotation. As object size/mass/inertia grow:

- **H1 (reach/workspace):** a single EE cannot stay in contact across the object's larger extent
  while keeping the object inside the contracting workspace → more `oob`, more lost contact.
- **H2 (controllability):** one contact under-actuates the planar wrench for large inertia — the
  arm can push translation but cannot simultaneously correct yaw without losing the object; error
  saturates above a size.
- **H3 (stability/toppling):** taller/larger objects topple under single-point pushing; two contacts
  provide a brace + push (force-closure-like control of the planar wrench without grasping).

The bimanual claim: **two contacts recover wrench controllability** — one arm braces/anchors while
the other drives, or both push antipodally — extending the frontier to objects a single arm cannot
control. Phases B–D test these explicitly.

---

## 3. Research phases

### Phase A — Measure the single-arm size frontier (scaling law)
**Hypothesis:** single-arm success rate is a monotonically decreasing, eventually-collapsing
function of object scale; the collapse persists under a size-relative success threshold.

**Do:**
1. Use the trained Trossen Stage-1 teacher (`output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt/last.ckpt`)
   and, as a cross-robot control, the paper Franka checkpoint.
2. Build an **evaluation-only scale sweep**: fix a held-out object set, evaluate at scales
   `s ∈ {0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, …}` extending well beyond the training range
   `[0.05, 0.10]`. (Note: training used DR up to 0.10; going beyond also probes generalization —
   keep the in-distribution vs out-of-distribution boundary explicit in every plot.)
3. For each scale, log: success rate, mean final pose error (pos + yaw), termination cause
   breakdown (`success` / `oob` / `timeout`), contact-loss frequency, and mean object displacement.
4. Repeat the entire sweep under both threshold regimes from §1 (absolute 0.05 m; size-relative
   `k·radius` for `k ∈ {0.3, 0.5, 0.75}`).
5. Decouple confounds with controlled ablations: (a) scale geometry only vs (b) scale geometry +
   mass (does mass scale with volume in the current asset pipeline? verify in `common.py` /
   asset loading and report), (c) widen `ws_bound`/table so reach is never the binding constraint —
   isolate controllability (H2) from reach (H1).

**Deliverable A:** a **scaling-law figure** (success vs scale, one curve per threshold regime and
per robot) with the **crossover/collapse size `s*` marked**, plus a termination-cause stacked-area
plot showing *how* failure shifts (oob-dominated vs timeout/pose-error-dominated) as size grows.
This figure alone is a paper result.

### Phase B — Diagnose the failure mode at large scale
**Hypothesis:** above `s*`, failures are dominated by loss of rotational control and/or toppling,
not merely reach.

**Do:** for objects just below and just above `s*`, record rollouts (raw GPU camera, not the
`NvdrCameraWrapper` "lying-tensor" path — see `progress_report.md §3`) and quantify: residual yaw
error vs residual position error, count toppling events, count contact-loss-then-oob sequences.
Map each failure to H1/H2/H3.

**Deliverable B:** a labeled failure taxonomy with per-mode frequency vs scale; qualitative
side-by-side clips of the same object succeeding small / failing large.

### Phase C — The EXPLICIT single-vs-bimanual trigger (core contribution)
**Requirement from the PI:** the arm-count decision must be **explicit and interpretable**, a
first-class module with measurable accuracy — NOT an emergent behavior hidden inside one
black-box policy.

Design it as a **capability predictor + minimal-effector rule**:
1. **Features (cheap, available before acting):** object bounding-box extents (x,y,z), max
   cross-section, volume, aspect ratio, estimated mass, planar inertia proxy, initial-vs-goal
   distance. All derivable from the same point cloud / asset metadata DyWA already computes
   (`scene/tabletop_with_object_scene.py` exposes `bboxes`, `radii`, `cloud`).
2. **Target:** `P(single-arm success)` regressed/classified from the Phase-A labeled sweep data
   (every (object, scale, outcome) tuple is a training example — you already generate thousands).
3. **Trigger rule:** choose **single arm** if `P̂(single success) ≥ τ`, else **bimanual**. τ tuned on
   a validation split to a target reliability (e.g. ≥90% of triggered-single cases succeed).
   Report the full ROC / reliability curve, the learned **decision boundary in (size, mass) space**,
   and a feature-importance / ablation showing *which* physical features drive the decision
   (this is the interpretable story).
4. **Baselines to beat:** (a) always-single, (b) always-bimanual, (c) a naive size-threshold rule
   `bimanual if scale > c`. Show the learned trigger dominates the naive threshold (it should use
   shape/mass, not size alone — e.g. a flat wide tray vs a compact heavy cube).

**Deliverable C:** trigger accuracy/ROC, the interpretable decision boundary plot, and the
crossover analysis — *"single arm suffices below the boundary, two arms are required above it."*

### Phase D — Build & train the bimanual non-prehensile policy
**Hypothesis:** two Trossen arms extend the success frontier to scales where single-arm collapses,
with the same DyWA training recipe.

**Chosen route — Isaac Gym (decided): extend DyWA's `push_env`.** This keeps the bimanual result
apples-to-apples with the single-arm 93.1% teacher — same simulator, same reward shaping, same
metrics, same object pipeline — which is what makes the Phase-E Pareto comparison rigorous rather
than cross-sim. Concretely:
1. Extend `dywa/src/env/push_env.py` to instantiate **two** Trossen actors per env; mirror
   `robot/trossen.py:create_actors` for a second arm.
2. Place each base with the create-time `gymapi.Transform()` fix from `progress_report.md §3`
   (the `fix_base_link` welded-base bug is already solved) — e.g. two arms behind opposite/adjacent
   table edges; make the second base pose a config knob and sweep it (arm separation matters for
   large objects).
3. Expand action space to **2×6D** and observation to include both arms' proprio + EE state;
   keep the object/goal observation shared. Update the Trossen `run` config `ctx_dim`/`dim_out`
   accordingly (cf. `trossen_icra_ours_abs_rel.yaml` which already documents the Trossen dims).
4. Define a bimanual push reward (shared goal-pose error; optionally a contact/anchor shaping term)
   and train a Stage-1 teacher with the **same recipe** as the single-arm run, reusing the
   widened-`ws_bound` lesson (`dev_log.md:1529`).

Train on objects spanning **and exceeding** `s*`. Log the same metrics as Phase A.

*Option to de-risk the bimanual policy (recommended — see §8):* instead of training a dual-arm
non-prehensile policy from scratch, **reuse BiNoMaP's bimanual non-prehensile primitives**
(arXiv 2509.21256; the `binomap/` code already lives in `Aloha_real/binomap/` and
`portfolio/BiNoMaP/`). BiNoMaP provides size-parameterized push/poke/pivot/wrap primitives for
two arms — wiring those in as the "bimanual controller" behind the trigger spends novelty budget on
the trigger + frontier (your actual contribution) rather than on re-deriving dual-arm RL. Treat the
Isaac dual-Trossen build as the apples-to-apples comparison and BiNoMaP primitives as the
fast/strong bimanual baseline.

*Fallback only if both stall:* `Aloha_real/sim_3d_bimanual-main` (mjlab, 2× Trossen already
modeled) can prototype a bimanual non-prehensile push by adapting `lift_cube_env_cfg.py` to
validate the physical claim quickly — but treat any cross-sim number as illustrative, not as part of
the headline Pareto comparison.

**Deliverable D:** bimanual scaling-law curve overlaid on the single-arm curve — must show the
bimanual curve staying high where single-arm collapses. Record emergent strategies (brace-and-push,
antipodal/pincer, hand-over-hand re-contact) — these are compelling qualitative results.

### Phase E — Unified system & the Pareto result
Compose Phases C+D: per object, the trigger picks arm count; evaluate the **combined** system across
the full size range against always-single and always-bimanual.

**Deliverable E (headline figure):** three curves — always-single, always-bimanual, trigger — where
the trigger tracks the **upper envelope**: matching single-arm efficiency for small objects (and its
lower coordination cost) and bimanual capability for large ones, with the crossover at `s*`. Add a
"cost" axis (arms used / energy / wall-clock) to make the minimal-effector argument quantitative.

### Phase F — Analysis, ablations, publication framing
- Ablate trigger features (size-only vs +mass vs +shape) → quantify the value of physical reasoning.
- Sensitivity to τ; failure analysis of mis-triggers (single chosen but failed, bimanual chosen but
  single would have sufficed).
- Confirm the scaling law is physical (holds under size-relative threshold and reach-unconstrained
  workspace) and not a goal_radius artifact.
- Sim-to-real discussion: same Trossen WXAI exists in both sims and in `lerobot_trossen` real
  drivers — articulate the path.

---

## 4. Concrete deliverables (publishable artifacts)
1. **Scaling-law figure**: single-arm success vs object size, crossover `s*` marked, under
   absolute + size-relative thresholds, Trossen + Franka.
2. **Failure taxonomy** (reach / controllability / toppling) with per-mode frequency vs size.
3. **Explicit trigger**: ROC + interpretable decision boundary in (size, mass, shape) space,
   beating size-only baseline.
4. **Bimanual policy (Isaac Gym, dual-Trossen `push_env`)** with a scaling-law curve extending the
   frontier past `s*` — same simulator/recipe as the single-arm teacher for a clean comparison.
5. **Unified Pareto figure** (the headline): trigger = upper envelope of single/bimanual + a cost axis.
6. **Emergent bimanual strategy** clips + analysis.

## 5. Suggested paper claims (target: a robotics/learning venue)
- *We characterize, for the first time, the object-size capability frontier of single-arm
  non-prehensile manipulation and show it collapses at a measurable crossover, for physical
  (controllability/stability) — not threshold-artifact — reasons.*
- *We introduce an explicit, interpretable single-vs-bimanual trigger from object geometry/mass that
  provably outperforms size-only heuristics.*
- *We train a bimanual non-prehensile policy that extends the manipulable object regime, and a
  unified trigger+policy system that Pareto-dominates either arm count alone.*

## 6. Guardrails
- Always separate in-distribution (`scale ≤ 0.10`) from extrapolation in plots.
- Never use `NvdrCameraWrapper` renders as physical ground truth — it reads the "lying" body tensor
  (`nvdr_camera_wrapper.py:520`) and paints meshes at written-but-PhysX-ignored poses; use a raw
  Isaac Gym GPU camera for any geometric/physical claim.
- Multi-seed every headline number; report run-to-run noise (≈3 pp observed historically).
- Report mass-scaling assumptions explicitly (does the asset pipeline scale mass with volume?).
- Apply the create-time `gymapi.Transform()` base-placement fix for any new (esp. second) arm.

## 7. First three concrete actions
1. Write an eval-only scale-sweep harness around `test_rma.py` / the Trossen teacher ckpt that
   loops over `scale` and `goal_radius` regimes and dumps per-(object,scale) outcomes to CSV.
2. Produce Deliverable A (scaling-law + termination-cause plots) and locate `s*`.
3. From that CSV, train the Phase-C capability predictor and plot the first decision boundary —
   this is the smallest end-to-end slice that already yields a publishable figure and de-risks the
   bimanual build.

---

## 8. Related work & novelty positioning (web-verified 2026-06-20)

A literature pass confirms the two core claims (size-frontier + explicit trigger) are open, with one
paper sitting close enough to require explicit differentiation. Verified arXiv IDs only.

### Must-cite, grouped
- **Non-prehensile base & baselines:** DyWA (2503.16806, ICCV 2025 — the system extended) ·
  CORN (2403.10760, ICLR 2024 — predecessor; supplies the reused point-cloud encoder) ·
  HACMan (2305.03942, CoRL 2023 — 89% sim-unseen / 50% real RL baseline) ·
  HACMan++ (2407.08585 — spatially-grounded primitives) ·
  Hierarchical & Modular Non-prehensile (2502.20843, 2025).
- **Bimanual non-prehensile / large-object:** **BiNoMaP (2509.21256, Zhou & Jia)** — dual-arm
  non-prehensile primitives (push/poke/pivot/wrap), size-parameterized, RL-free/demo-based ·
  DART (2604.17833, 2026 — dual-arm non-prehensile tray transport, MPC+RL, varies mass/geometry) ·
  Bimanual Manipulation Taxonomy (Krebs & Asfour, 2022 — when/why two hands) ·
  PerAct2 (2407.00278 — bimanual benchmark) · DA-VIL (2410.19712 — adaptive dual-arm RL).
- **Feasibility / affordance prediction (for the trigger):** RT-Affordance (2411.02704) plus the
  grasp-feasibility-prediction line — prior art for "predict success before acting", which the
  Phase-C capability predictor instantiates for arm-count choice.
- **Scaling (contrast, not precedent):** Data Scaling Laws in Imitation Learning (2410.18647,
  ICLR 2025) — performance vs *data quantity*, a different axis from capability vs *object scale*.

### The three closest threats and how this proposal stays distinct
1. **BiNoMaP (2509.21256) — closest.** Dual-arm non-prehensile, parameterized primarily by object
   **size**. *Differentiation:* BiNoMaP is **always bimanual**, demo/primitive-based, and never
   decides *whether* two arms are needed nor measures where single-arm fails. This proposal's
   contribution is exactly the missing piece — the **explicit trigger**, the **measured crossover
   `s*`**, and the **necessity** result (single-arm fails even when retrained, above `s*`).
   *Actionable:* cite prominently AND reuse BiNoMaP primitives as the Phase-D bimanual controller
   (code already in `Aloha_real/binomap/`) to de-risk the build.
2. **DART (2604.17833).** Always-dual MPC for tray transport; varies mass/geometry but makes no
   arm-count decision and studies no frontier. Different task and no trigger.
3. **Dual-hand *selection* for grasping.** Chooses left/right hand for a *grasp*; this proposal
   chooses *how many* arms for *non-prehensile* control, driven by a physical-capability predictor.

### Field-expected additions to honor in the experiments
- Report on the **standard DyWA/CORN/HACMan protocol** (1-view, unknown-state, unseen objects) so the
  single-arm numbers are directly comparable to published baselines.
- Include a **real-world or sim-to-real** discussion/result — DyWA reports 68% real; reviewers will
  expect at least a credible transfer argument (same Trossen WXAI exists in `lerobot_trossen`).
- Add a **"necessity" experiment**: retrain a single-arm policy *specifically* on objects above `s*`
  and show it still fails — this rebuts "you just undertrained the single arm / just always use two."
- Add **BiNoMaP-style size-parameterization as an explicit Phase-C baseline** the learned trigger
  must beat (size-only vs learned size+mass+shape).
