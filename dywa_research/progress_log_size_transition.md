# Progress Log — Object-Size Transition & Single-vs-Bimanual Triggering

Execution log for [research_prompt_size_transition.md](research_prompt_size_transition.md).
Companion literature review: [auto_research_log_size_transition.md](auto_research_log_size_transition.md).

**Convention:** newest entry on top. Every claim cites a repo file:line or a command actually run.
Negative results and blockers are recorded honestly (per the prompt's mandate).

---

## 2026-08-10 — Session 2: bare-metal runtime, teacher-eval bring-up, TWO validity bugs caught

**Agent:** Claude (Opus 4.8) over SSH to `coldbrew`. **Constraint honored:** only *write* inside
`dywa_research/`; repo files read-only (fixes done via patched copies / config overrides, never by
editing repo files).

### 0. Headline
GPU is now free (the Session-1 blocker cleared). Brought the teacher eval most of the way up on
bare metal, and in doing so **caught two bugs that would have produced illegitimate numbers** —
both fixed without touching repo files. Smoke run not yet validated to the 93% anchor; see §4.

### 1. Runtime decision — bare metal, not Docker
Chose conda env `dywa_render` (avoids the `pkm1:v0` container). Two accommodations, both in
`dywa_research/` so nothing in the repo changes:
- **Shims** for optional deps missing from `dywa_render`: `flash_attn`, `nvdiffrast`, `nvtx`
  (stub packages under `dywa_research/shims/`, prepended to `PYTHONPATH`).
- **Local data paths**: `data_path=/home/bruce/DyWA/data/meta-v8`, test set `data/test_set.json`.

### 2. Eval entry decision — `valid_ppo_arm.py` (teacher), not `test_rma.py` (student)
The prompt's Phase A wants the **teacher** frontier and the only local checkpoint is the Trossen
Stage-1-wide teacher (`output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt`). `valid_ppo_arm.py`
is the teacher-eval entry: it loads the policy via `++load_ckpt=<ckpt>` (this is CORRECT here — the
Session-1 "never pass load_ckpt" warning was specific to the *student* `test_rma.py` flow, a
different code path). Uses `++eval_track_per_obj_suc_rate=True` for per-object success.
Smoke config: `+env=trossen_icra_base +run=trossen_icra_ours_abs_rel`, `num_env=32`,
`ws_bound=[[-0.25,-0.35,-0.1],[0.25,0.35,0.4]]` (the Session-1 "widen ws_bound" lesson),
`force_rad=0.05`, in-dist `scale=0.08`.

### 3. BUG #1 (code) — `load_env() got multiple values for 'freeze_env'`
`valid_ppo_arm.py:87` calls `load_env(cfg, None, None, freeze_env=False, ...)` but the current
signature is `load_env(cfg, path, freeze_env=False, **kwds)` — the 3rd positional `None` collides
with the `freeze_env` keyword. **Fix (no repo edit):** patched copy
`dywa_research/valid_ppo_arm_patched.py` with the extra `None` removed (1-line diff). Configs still
resolve because `hydra_cli` finds them via `importlib.resources.path('data','cfg')` (package-based,
not script-relative). Run it with `dywa/exp/train` added to `PYTHONPATH` so its sibling imports
(`train_ppo_arm`, `envs`) resolve. Verified: eval now proceeds past this point into Isaac Gym.

### 4. BUG #2 (data, CRITICAL for legitimacy) — stale hardcoded mesh paths
All **26,737** object URDFs in `data/meta-v8/urdf/` hardcode absolute mesh paths
`/home/brucezhang/Downloads/DyWA/data/coacd/…` (a different user/machine that does not exist here).
Isaac Gym logged `Failed to resolve collision/visual mesh` for every object and **loaded objects
with no geometry** — any success rate from that run is meaningless. The real meshes (5,751 `.obj`)
are present at `data/coacd/`; the referenced files exist there.
- Can't symlink the stale path (`/home` is root-owned, no sudo) and can't edit the URDFs (repo,
  read-only). **Fix:** `dgn_object_set.py` exposes `urdf_path` as an *independent* override
  (defaults to `{data_path}/urdf`). Built a corrected copy
  `dywa_research/urdf_fixed/` (26,737 URDFs, prefix rewritten to `/home/bruce/DyWA/data/coacd`) and
  will pass `++env.single_object_scene.dgn.urdf_path=/home/bruce/DyWA/dywa_research/urdf_fixed`.
  Everything else still reads the real `meta-v8` (cloud/hull/pose/normal/metadata).

### 5. Legitimacy gates (enforced before ANY number is treated as a finding)
1. **Assets resolve** — zero "Failed to resolve mesh" lines in the log.
2. **Sanity anchor** — in-distribution scale (0.05–0.10) must reproduce the known **~93%** teacher
   success. A wrong pipeline that "runs" is not a result.
3. **Confound control** (prompt §1) — every scaling curve reported under BOTH absolute
   `goal_radius=0.05` and size-relative `k·s`; multi-seed the headline numbers.
Only after gates 1–2 pass does the Phase-A scale sweep produce a real Deliverable A.

### 6. Operational note — SSH self-disconnect (resolved)
The recurring `exit 255` drops were **self-inflicted**: `pkill/pgrep -f valid_ppo_arm_patched` over
SSH matched its own command line and killed the session. Fix: kill by iterating `pgrep -x python`
and matching `/proc/PID/cmdline` (the bash SSH shell is excluded). Always launch evals **detached**
(`setsid … </dev/null &`) so genuine drops can't kill a running job; poll the logfile.

### 7. Status / next actions
- [done] GPU free; runtime + shims; teacher entry; BUG#1 patch; BUG#2 urdf_fixed built.
- [next] Re-run smoke with `urdf_path` override → confirm gates 1–2 (mesh-clean log + ~93%).
- [then] Pivot `phase_a_scale_sweep.sh` to the `valid_ppo_arm_patched.py` teacher command; run the
  scale × goal_radius sweep detached; `aggregate_scale_sweep.py` → Deliverable A (scaling law + s*).
- Open: confirm the exact per-object success field name in `valid_ppo_arm.py` output for the
  aggregator; confirm object-radius semantics for the size-relative threshold.

---

### 8. Smoke result — pipeline RUNS end-to-end, but FAILS the sanity anchor (do not sweep yet)

After fixing BUG#1 (load_env args + tuple unpack) and BUG#2 (urdf_fixed), the teacher eval now
runs fully and emits per-object success telemetry:

```
{'valid/episode_count': 64, 'valid/success_count': 0, 'valid/suc_rate': 0.0,
 'valid/balanced_suc_rate': 0.0}   # scale=0.08 (in-distribution), eval_step=600, 32 envs, deterministic
```

**This is 0%, not the expected ~93% → legitimacy GATE 2 FAILS.** Per the "every test must be legit"
mandate, this means the eval *recipe* is not yet validated, so **no scaling-law numbers may be
produced from it** — running the full Phase-A sweep now would yield untrustworthy data. Halting the
sweep here is the correct, integrity-preserving call.

**What is verified correct (ruled out as the cause):**
- Meshes resolve — 0 "Failed to resolve" (urdf_fixed override works). GATE 1 passes.
- CORN/ICP encoder cached locally (`~/.cache/huggingface/hub/models--imm-unicorn--corn-public`
  and `~/CORN/huggingface_cache/...`) — loads.
- Policy actor IS loaded from the teacher ckpt: `load_agent` → `train_ppo_arm.py:63-66`
  `agent.load(last_ckpt(cfg.load_ckpt))`. Not random-init.
- Normalizer env-stats loaded from `stat/env-last.ckpt`.
- Config sane: `min_scale=max_scale=0.08`, `goal_radius=0.05`, `ws_bound` widened, `base_height=table`,
  `which_robot=trossen`, `use_icp_obs=True`, `use_6d_rel_goal=True`.

**Leading hypotheses for the 0% (to test next, cheapest first):**
1. **Deterministic vs stochastic action.** Eval used `sample_action=False`. The 93% was a
   *training-time* `cur_suc_rate` (stochastic policy, possibly under a difficulty *curriculum* that
   relaxes goal_radius/init). Queued a stochastic + longer run (`sample_action=True`,
   `eval_step=2400`) → `results/smoke/smoke6_stoch.log`. [result pending]
2. **Curriculum/goal-difficulty mismatch.** Training may have measured success under
   `use_tune_goal_radius`/`use_tune_init_pos` (relaxed goal). Eval uses the full `goal_radius=0.05`.
   Check the teacher run's TB scalars for the goal_radius schedule at the point 93% was logged.
3. **`force_rad=0.05` / force-perturbation** semantics at reset — verify this isn't kicking objects
   OOB every episode (need the MonitorEnv termination-cause breakdown; short run didn't emit it —
   raise `monitor.num_env_record` / lengthen eval to force a log flush, or read oob vs timeout).
4. **task.mode='train'** in the eval config (scene mode is 'valid' but task mode is 'train') — check
   whether that changes goal sampling/termination.

**Decision:** Phase-A scale sweep stays BLOCKED until an in-distribution config reproduces ~90%+.
Only then does `aggregate_scale_sweep.py` output count as Deliverable A. Everything built this
session (patched entry, urdf_fixed, harness) is reusable as-is once the anchor is met.

---

### 9. Stochastic run — hypothesis #1 RULED OUT
smoke6_stoch (sample_action=True, eval_step=2400): still **0%** (256 ep, 0 succ), and
avg_true_return=-0.134 (negative). Both deterministic and stochastic → exactly 0% with negative
return ⇒ NOT a determinism/curriculum-difficulty gap. The loaded policy is making no progress in
this eval config → systematic **obs/policy mismatch** (the policy acts on observations that differ
from training, or an action-space/normalizer mismatch). Next diagnostic: obtain a KNOWN-GOOD eval
invocation for this exact teacher (compare obs keys/shapes + normalizer stats training-vs-eval;
inspect MonitorEnv oob-vs-timeout breakdown). Sweep remains BLOCKED (correctly).

---

### 10. ROOT CAUSE of 0% found — wrong ws_bound + filter (recovered from prior transcripts)
Recovered the known-good teacher reproduction from ~/.claude/projects transcripts + dev_log: the
2026-06-21 local repro hit 62/64 (~97%). Diffing it against my 0% smoke, TWO knobs were wrong:
  * ws_bound x-max: mine 0.25 vs known-good **0.6** — the "widen ws_bound so the EE reaches the
    table" fix, which dev_log calls the DECISIVE enabler. At 0.25 the arm cannot reach across the
    table -> 0% + negative return (exactly the symptom).
  * filter_file: mine test_set.json vs known-good **yes.json**.
The paper stats are reproducible; this was my config error, not an irreproducibility. Built
dywa_research/anchor_eval.sh (ws_bound=0.6, filter=yes.json, urdf_fixed, outputs under
dywa_research/results/anchor) and launched it. Note: known-good uses default scale DR [0.05,0.10];
Phase-A will pin scale only AFTER the anchor passes. [anchor result pending]

USER CONSTRAINT (2026-08-10): keep ALL research in dywa_research/; do NOT touch the dywa repo /
output/trossen_pipeline where the user evaluated trossen. Verified compliant: no repo/output dirs
modified; all artifacts (urdf_fixed, valid_ppo_arm_patched.py, shims, scripts, results) are in
dywa_research/; run path.root now points into dywa_research/results.

---

### 11. ANCHOR PASSES — eval VALIDATED (94.5%%), Phase-A UNBLOCKED
anchor_eval.sh result: episode_count=615, success_count=581, **suc_rate=0.9447 (94.5%%)**,
balanced_suc_rate=0.9355. Matches the reported ~93.1%% teacher. GATE 1 (0 mesh errs) + GATE 2
(~93%%) both PASS. The eval pipeline is now legitimate and under our control, fully inside
dywa_research/. Confirmed the 0%% was purely the ws_bound/filter config error.
Next: Phase-A scale sweep on held-out objects (test_set.json) with the validated command, pinning
scale via min_scale==max_scale across {0.05..0.30}, goal_radius absolute 0.05 (+ size-relative pass),
outputs under dywa_research/results/phase_a, then aggregate -> Deliverable A (scaling law + s*).

---

### 12. DELIVERABLE A COMPLETE — single-arm size frontier measured, s* ~= 0.167 m
Phase-A scale sweep (validated eval, held-out test_set.json, goal_radius=0.05 absolute, 64 envs,
eval_step=1200, urdf_fixed, ws_bound x-max 0.6). Legitimate scaling law:

| scale (m) | suc_rate | balanced | episodes |
|-----------|----------|----------|----------|
| 0.05  | 0.967 | 0.928 | 1523 |  in-dist
| 0.075 | 0.963 | 0.932 | 1643 |  in-dist
| 0.10  | 0.931 | 0.894 | 1292 |  in-dist edge
| 0.15  | 0.694 | 0.603 |  625 |  extrapolation
| 0.20  | 0.120 | 0.278 |  935 |  COLLAPSE
| 0.25  | 0.027 | 0.066 |  854 |
| 0.30  | 0.001 | 0.020 | 5367 |

Findings:
- Plateau ~93-97%% across the training range [0.05,0.10] (held-out objects) -> teacher generalizes.
- SHARP collapse between 0.15 (69%%) and 0.20 (12%%). **Crossover s* ~= 0.167 m** (50%%-success,
  linear interp). By 0.30, success ~= 0 -> single-arm non-prehensile control fails on large objects.
- This is the Phase-A headline the prompt asked for: a measured single-arm size frontier with a
  crossover, on the same simulator/teacher as the 93%% result. Figure:
  results/phase_a/deliverable_A_scaling_law.png ; data: results/phase_a/scale_sweep.csv .

CAVEATS / next (per prompt guardrails, before this is paper-ready):
1. Confound control: re-run under SIZE-RELATIVE goal_radius (k*s, k in {0.3,0.5,0.75}) to prove the
   collapse is physical, not an artifact of a fixed 0.05 m threshold being a smaller fraction of big
   objects. (Sweep script already parameterizes GOAL_RADIUS; needs the relative-mode variant.)
2. Multi-seed the headline points (>=3 seeds) for error bars (~3pp historical noise).
3. Termination-cause breakdown (oob vs timeout vs pose-error) per scale = Phase-B failure taxonomy.
4. Mass is decoupled from scale (Session-2 finding) -> current sweep is geometry-only; the
   geometry+mass co-scaling ablation still needs the small env patch.

---

### 13. Phase-B mechanism battery + a measurement-validity correction (force_rad)
Ran a single-knob intervention battery at the collapsed scale s=0.20 (baseline suc_rate 0.120):

| knob            | value | suc_rate | balanced |
|-----------------|-------|----------|----------|
| baseline        | -     | 0.120    | 0.278    |
| reach ws_xmax   | 0.9   | 0.124    | 0.280    |
| reach ws_xmax   | 1.2   | 0.124    | 0.280    |
| mass (pinned)   | 0.05  | 0.123    | 0.174    |
| mass (pinned)   | 0.30  | 0.097    | 0.226    |
| mass (pinned)   | 0.80  | 0.066    | 0.196    |

Findings: (H1 reach) REJECTED — widening workspace does nothing. (H2 inertia) modulates but does
NOT cause — heavier hurts monotonically yet near-massless (0.05) does NOT recover (still ~0.12).
=> the collapse is a genuine geometric/contact-controllability limit, not workspace or mass.

**MEASUREMENT-VALIDITY CORRECTION (caught via a null result):** the goal_radius=0.10 intervention
returned byte-identical numbers to baseline. Root cause: force_rad OVERRIDES env.task.goal_radius in
Config.__post_init__ (rma.py:389-391). Resolved config confirms Deliverable A + anchor actually ran
at **force_rad=0.1 (goal tolerance 0.10 m), NOT the 0.05 I previously logged.** The frontier SHAPE /
s* stay valid (tolerance was a constant 0.10 across all scales), but:
  - CORRECTION: Deliverable A goal tolerance = 0.10 m (via force_rad default), not 0.05.
  - The correct threshold knob is ++force_rad (not ++env.task.goal_radius). Re-running a proper
    force_rad sweep at s=0.20 {0.05,0.15,0.20} + a strict-0.05 in-dist anchor recheck. [pending]
  - LESSON: always verify the RESOLVED config (grep the run log), not just the CLI override, because
    __post_init__ can silently clobber overrides.

---

### 14. Threshold control (proper force_rad knob) + strict anchor + CRITIQUES.md
Threshold-artifact test at s=0.20 (force_rad = goal tolerance):
  force_rad 0.05->0.157, 0.10->0.120, 0.15->0.106, 0.20->0.142 suc_rate. No systematic recovery even
  at a 0.20 m goal (larger than the object) => collapse is THRESHOLD-ROBUST. Threshold artifact rejected.
Strict-tolerance anchor: in-dist s=0.10 at force_rad=0.05 = **0.908** (balanced 0.866) — reproduction
survives the strict 0.05 criterion (was 0.93 at 0.10). Robust either way.
Net Phase-B: the size frontier survives THREE independent controls — reach (rejected), inertia
(modulates, not causal), threshold (rejected). Strongly consistent with a genuine geometric/contact
-controllability limit. The one dominant remaining confound is the train-test scale gap (#1).
Wrote dywa_research/CRITIQUES.md — a rigorous anticipated-critique / red-team doc (12 items, ranked by
threat, each with status + the experiment that closes it). Top OPEN before any bimanual-necessity
claim: (#1) train-test scale gap [necessity experiment], (#3) per-object feasibility audit at scale,
(#4) multi-seed error bars.
---

### 15. DELIVERABLE B — failure-mode taxonomy vs scale (+ feasible-only frontier correction)
Instrumented a termination-cause wrapper (valid_ppo_arm_taxonomy.py, separate from the validated
eval) and swept scale. force_rad=0.10 (matches Del A). Fractions of episodes:

| scale | success | oob(lost) | timeout | immediate(<=3) | mean_eplen | feasible-only succ |
|-------|---------|-----------|---------|----------------|------------|--------------------|
| 0.05  | 0.967   | 0.016     | 0.017   | 0.000          | 48.4       | 0.967 |
| 0.10  | 0.931   | 0.050     | 0.019   | 0.000          | 57.3       | 0.931 |
| 0.15  | 0.694   | 0.157     | 0.149   | 0.000          | 113.3      | 0.694 |
| 0.20  | 0.120   | 0.698     | 0.182   | 0.637          | 73.5       | **0.330** |
| 0.30  | 0.001   | 0.961     | 0.038   | 0.892          | 13.7       | 0.009 |

Findings:
1. Failure MECHANISM shifts with size: in-dist failures are rare/split; at collapse OOB DOMINATES
   (0.70 @0.20, 0.96 @0.30) => objects knocked out of bounds = single-contact controllability loss,
   NOT timeout. Consistent with a geometric/contact limit (H3).
2. CONFOUND #3 confirmed + corrected: immediate(<=3-step) terminations 0% for s<=0.15 but 64%@0.20,
   89%@0.30 (workspace contracts by object radius => big objects spawn at/over the boundary and
   instantly go OOB). Raw frontier OVERSTATES collapse.
3. Feasible-only frontier (success/(done-immediate)): 0.20 -> 0.330 (vs raw 0.120), 0.30 -> 0.009.
   Collapse SURVIVES the correction; s* shifts right (~0.18 vs raw 0.167). Honest frontier lies
   between raw and feasible-only. Figure: results/phase_b_taxonomy/deliverable_B_taxonomy.png.
CRITIQUES.md #3 updated to ADDRESSED. Remaining top-OPEN: #1 (train-test scale gap / necessity),
#4 (multi-seed error bars). A per-object bbox-vs-workspace admissibility filter would sharpen #3.
---

### 16. #4 multi-seed error bars + #3 feasibility mechanism (CONVERGING findings)
Multi-seed (3 seeds) frontier, force_rad=0.10, held-out:
  s=0.10: raw 0.934+-0.005   s=0.15: 0.681+-0.012   s=0.20: 0.209+-0.064   s=0.30: 0.001+-0.000
KEY: the single-seed Deliverable-A point at s=0.20 (0.12) was a BAD-SEED OUTLIER; 3-seed mean is
0.209 raw / 0.298 feasible-only. Variance is tiny everywhere EXCEPT s=0.20 (std 0.064) — the point
sits on a cliff.

#3 feasibility MECHANISM (code-confirmed, tabletop_with_object_scene.py:660-677): the object
placement + OOB region = 0.5*margin_scale*table_dims - radius per xy side, on a FIXED table
(0.4x0.5, margin 0.95). Usable_x = 0.38 - 2s -> hits 0 at **s=0.19**, then the region collapses to a
single point (mask), so any motion = immediate OOB. This QUANTITATIVELY coincides with:
  - the measured s* ~= 0.17-0.19, AND
  - the immediate-termination spike (0 for s<=0.15 -> 0.64/0.006/0.19 across seeds @0.20 -> 0.87 @0.30), AND
  - the s=0.20 seed variance (marginal region => hypersensitive to the random draw).
=> A LARGE part of the raw frontier is a FIXED-TABLE placement artifact, not (yet) a demonstrated
control limit. This is the single most important caveat on Deliverable A.
Decisive test launched: prevent_fall=False (removes the radius-contraction, gives objects the full
table region) at s={0.15,0.20,0.30}. If success recovers => placement artifact dominates; if not =>
genuine control limit. [result pending]
---

### 17. DECISIVE feasibility result — frontier is dominated by a FIXED-TABLE artifact; #1 GATED
Traced the OOB bound: push_task.py:512 sets it from env.scene_cfg.table_dims * margin_scale (FIXED
+-[0.16,0.20]); the table is a fixed small URDF (table-with-wall-small.urdf) — table_dims sets bounds
only, not the physical mesh. Objects are scaled so radius ~= s, so at s=0.20 the object radius (0.20)
equals the table half-width (0.20): the object is as big as the table.
- prevent_fall=False test was INCONCLUSIVE (byte-identical) because init_type=stable uses precomputed
  stable poses, so that knob does not move spawns. (Another verify-the-resolved-config lesson.)
- Convergent evidence the table artifact DOMINATES the raw frontier: placement region collapses at
  s~=0.19 == measured s*; immediate-terminations 0/64/89% at s=0.15/0.20/0.30; multi-seed variance
  peaks only at the s=0.20 cliff.
CONCLUSION: the measured single-arm size frontier (s*~=0.17) is substantially a fixed-table geometry
artifact, NOT yet a demonstrated single-arm CONTROL limit. Deliverable A must be reframed accordingly.
GATING DECISION: #1 necessity fine-tuning is DEFERRED — it would be scientifically invalid on a
table-confounded setup (fine-tuning on objects that cannot fit the table). The required unblock is a
LARGER TABLE ASSET so the object-to-table ratio stays constant across scales; then re-measure the
frontier and, if the collapse persists, run necessity. This is the rigorous ordering.
Deliverable A v2 (multi-seed, error bars) plotted: results/phase_c_multiseed/deliverable_A_v2_multiseed.png.
CRITIQUES.md updated: added #0 (fixed-table artifact, dominant) and marked #1 gated on it.
---

## 2026-07-20 — Session 1: reconnaissance, fact-verification, Phase-A harness build

**Agent:** Claude (Opus 4.8), driving remotely over SSH into `coldbrew` (10.137.31.174).
**Working dir:** `/home/bruce/DyWA`. **Constraint honored:** read/inspect anywhere in the repo,
but only *write* inside `dywa_research/`.

### 0. Executive summary

- Verified the load-bearing Phase-1 "grounding facts" against the actual code — all confirmed,
  **plus one important new finding: scale and mass are DECOUPLED** in the current pipeline
  (mass is sampled independently of size, not volume-coupled). This changes how Phase-A ablation (b)
  must be done. See §3.
- Fully reverse-engineered the eval entry point (`test_rma.py`) and reconstructed the exact,
  known-working invocation with every hydra override knob needed for the scale/goal_radius sweep.
  See §4.
- Built the Phase-A eval harness (`phase_a_scale_sweep.sh` + `aggregate_scale_sweep.py`) inside
  `dywa_research/`, ready to run. See §5.
- **BLOCKER: cannot execute sims right now.** The single RTX 4090 on `coldbrew` is fully occupied
  by another user (`hanwen`, RoboTwin pi05 policy, 21.6/24 GB, 100% util). Isaac Gym eval needs
  several GB + compute; running now would contend with a shared job. Harness is staged; run when
  the GPU frees. See §6.

### 1. Environment state

| Item | Finding |
|---|---|
| Host | `coldbrew` = 10.137.31.174, user `bruce` (SSH config alias present) |
| GPU | 1× RTX 4090, 24 GB. **22.0 GB used, 100% util** by PID 404222 (`hanwen` RoboTwin pi05) |
| Repo | `/home/bruce/DyWA`, branch `main`, clean before this session |
| Conda envs | `base`, `dywa_render`, `lerobot`, `trossen_sim`; plus `~/envs/isaaclab` |
| Eval runtime | Official flow runs **inside Docker** `pkm1:v0` (`docker/run.sh`): mounts repo at
  `/home/user/DyWA`, datasets at `/input`. This is why working commands use `/input/DGN/...` paths. |
| Data (local) | `data/meta-v8/` (cloud, hull, meta, pose, ...), `data/test_set.json`,
  `data/yes.json`. Container maps datasets → `/input`. |

### 2. Checkpoints located

- **Trossen Stage-1 teacher (wide, the one the prompt names):**
  `output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt/last.ckpt`
  + normalizer `.../stat/env-last.ckpt`. ✅ present.
- Also present: `teacher-stage1/run-000/ckpt/{last,step-32768}.ckpt` (the non-wide run).
- No Stage-2 teacher / distilled student checkpoint found locally (pipeline references
  `teacher-stage2` and `distill/film_mlp` but those dirs are absent). Relevant because the
  *known-working* eval command (dev_log:180) evaluates a **distilled student**; evaluating the
  **teacher** directly is a different code path — see §4 caveat.

### 3. Phase-1 grounding facts — verification results

All citations in the prompt's §1 table were checked against current code. Status:

| Fact (prompt §1) | Verified? | Evidence (current code) |
|---|---|---|
| Object scale `{use_scale_dr, min_scale, max_scale} = {true, 0.05, 0.10}` | ✅ | `env/icra_base.yaml`: `use_scale_dr: true`, `min_scale: 0.05`, `max_scale: 0.1` |
| `goal_radius = 0.05` fixed, `sample_thresh = [1,1]` | ✅ | `env/icra_base.yaml`: `task.goal_radius: 0.05`, `sample_thresh: [1.0, 1.0]` |
| Trossen teacher 93.1% cur_suc_rate @ 500k | ✅ (≈93%) | `dev_log.md:1664` "Stage 1 wide run COMPLETED (500k steps, cur_suc_rate 93%)" |
| Scale applied as `set_actor_scale(target_size / radius)` | ✅ | `env/common.py:76-79`: `target_size = uniform(min_scale, max_scale); scale = target_size/radius; gym.set_actor_scale(...)` |
| Env hardcoded single-arm (one Trossen actor) | ✅ | `env/push_env.py:35,77` single `Trossen` robot cfg; robot dispatch by `which_robot` |
| Mass config exists (`min_mass 0.1, max_mass 0.5`) | ✅ | `env/icra_base.yaml`: `min_mass: 0.1`, `max_mass: 0.5` |

**NEW FINDING (not in prompt) — scale ⟂ mass are independent knobs.**
In `env/common.py`, geometry scale (`change_scale` → `set_actor_scale`) and mass
(`enable_mass` → `get/set_actor_rigid_body_properties`, sampled `uniform(min_mass, max_mass)`)
are set in **separate, independent branches**. Mass is **not** derived from volume/scale.

*Implication for Phase A:* the prompt's guardrail question "does the asset pipeline scale mass with
volume?" is answered: **NO.** So:
- Phase-A ablation (a) "scale geometry only" is the *default* behavior — free.
- Phase-A ablation (b) "scale geometry + mass co-varying with volume" requires a **code change** to
  couple `target_mass ∝ target_size³` (env is currently incapable of it out of the box). Note this
  as a small env patch to add before the (b) ablation. Alternatively exploit the decoupling to vary
  mass *independently* at fixed size to cleanly separate inertia effects (H2) from pure geometry.
- To pin a *single* eval scale `s`, set `min_scale == max_scale == s` (it's a uniform sample).

*Not yet verified this session* (deferred, lower priority / need code read): workspace-contract
`min_bound += radius` (`tabletop_with_object_scene.py:660-677`), `oob` termination (`dev_log:1378`),
`push_task.py:754` goal-threshold logic. These are not blockers for Phase-A harness correctness.

### 4. Eval entry point — how `test_rma.py` runs (reverse-engineered)

- Entry: `dywa/exp/train/test_rma.py`, hydra `@hydra_cli(config_name='show')`, config path
  `../../src/data/cfg/`. Imports `isaacgym` (GPU-required at import time).
- Flow (`main`, lines 182–400): `load_env` → optionally wrap `CountCategoricalSuccess`
  (per-object success counting, `++log_categorical_results=True`) → load teacher agent
  (`load_agent`) unless training student → build `StudentAgentRMA`, optionally `+load_student` →
  rollout `for step in range(test_step): env.step(...)` → on exit,
  `CountCategoricalSuccess.save()` dumps results.
- Success results are written under `output/test_rma/{name}/result/{base_set}/` (per `dev_log:1391`).

**Known-working invocation** (reconstructed from `dev_log.md:180-200`, official
`eval_student_unseen_obj.sh` semantics):

```bash
cd /home/user/DyWA/dywa/exp/train   # inside pkm1:v0 container
PYTORCH_JIT=0 python test_rma.py \
  +platform=debug +env=abs_goal_1view +run=teacher_base +student=dywa/base \
  ++name=dywa \
  ++path.root=/home/user/DyWA/output/eval_official \
  ++env.num_env=60 ++global_device=cuda:0 \
  ++student.norm=ln \
  ++add_teacher_state=1 '++student.decoder.film_mlp=1' \
  +load_student=$PRETRAINED/ckpt/last.ckpt \
  ++plot_pc=0 \
  ++dagger_train_env.anneal_step=1 \
  ++use_nvdr_record_episode=false \
  ++monitor.num_env_record=60 \
  ++env.single_object_scene.filter_file=/input/DGN/test_set.json \
  ++env.single_object_scene.mode=valid \
  ++log_categorical_results=True \
  ++env.single_object_scene.dgn.data_path=/input/DGN/meta-v8 \
  ++env.single_object_scene.dgn.pose_path=/input/DGN/meta-v8/unique_dgn_poses \
  ++test_step=3000
```

**Override knobs the sweep needs** (all hydra `++`):
- Scale (pin one point): `++env.single_object_scene.min_scale=S ++env.single_object_scene.max_scale=S`
- Goal threshold: `++env.task.goal_radius=R`
- Object set: `++env.single_object_scene.filter_file=/input/DGN/test_set.json`
- Env count / rollout length: `++env.num_env=N ++test_step=T`

**CRITICAL known-issue (dev_log:126-147, 200):** never pass `++load_ckpt=<teacher_ckpt>` at eval —
it drives success to **0%** (OOB every ~3 steps) via the `AddTeacherState` interaction. The teacher
is loaded implicitly; the student is loaded with `+load_student`. Harness must NOT set `load_ckpt`.

**Caveat — teacher vs student eval.** The known-working command above evaluates a *distilled
student* (`+student=dywa/base +load_student=.../last.ckpt`). The prompt's Phase A wants the *Trossen
Stage-1 teacher* frontier. Two options, to resolve on first dry-run:
1. **(Preferred, lowest-risk)** evaluate whatever pretrained checkpoint reproduces the 93% number
   first, to validate the harness end-to-end, then switch object scale.
2. Evaluate the teacher policy directly (`train_student_policy`/teacher-only path). Requires
   confirming the teacher agent acts in the rollout loop without a student. Flagged for the dry-run.
   The local checkpoint we *have* is the Trossen teacher, so this path is what Phase A ultimately
   needs; the student flow may need a student ckpt we don't yet have locally.

### 5. Phase-A harness (built, staged in dywa_research/)

- `phase_a_scale_sweep.sh` — loops `scale ∈ {0.05,0.075,0.10,0.15,0.20,0.25,0.30}` ×
  `goal_radius` regime ∈ {absolute 0.05; size-relative k·s for k∈{0.3,0.5,0.75}}; invokes
  `test_rma.py` per cell with the overrides above; writes each cell's output to its own dir under
  `dywa_research/results/phase_a/`. Designed to run inside `pkm1:v0`. Dry-run switch `DRYRUN=1`
  prints commands without executing. **Not yet executed** (GPU blocked).
- `aggregate_scale_sweep.py` — walks the per-cell result dirs, pulls success/termination-cause
  counts, emits `scale_sweep.csv` and a first success-vs-scale plot, and locates the crossover `s*`.
  Result-file schema to confirm on first real run (marked TODO in the script).

Size-relative threshold assumption documented in-script: post-scale object radius ≈ `target_size`
(since `scale = target_size/radius` ⇒ `new_radius = radius·scale = target_size`), so
`goal_radius_relative = k · s`. Confirm against `push_task.py` on dry-run.

### 6. Blockers & handoff

1. **GPU contention (hard blocker for execution).** RTX 4090 at 22/24 GB, 100% util by another
   user's job. Re-check with `nvidia-smi`; when free, run §5 harness. No safe way to run Isaac Gym
   eval now.
2. **Runtime = Docker.** Confirm `pkm1:v0` image is present (`docker images | grep pkm1`) and that
   `docker/run.sh` data paths (`/data/bruce/DyWA/...`) exist on this host, or adapt them to the
   local `data/` copy. Alternatively test whether bare-metal conda `dywa_render` can import
   `isaacgym` + run `test_rma.py` (avoids Docker) — untested.
3. **Teacher-vs-student eval path** (§4 caveat) — resolve on first dry-run.
4. **Object radius semantics** for the size-relative threshold — confirm on dry-run.

### Next actions (when GPU frees)
1. `DRYRUN=1 ./phase_a_scale_sweep.sh` to print/validate commands; confirm container + paths.
2. Single smoke cell at `s=0.05, goal_radius=0.05, num_env=16, test_step=200` to validate the
   harness end-to-end and reproduce a sane in-distribution success number.
3. Full sweep → `aggregate_scale_sweep.py` → **Deliverable A** (scaling-law + termination-cause
   plots, crossover `s*`). Record numbers in a new dated entry here.
