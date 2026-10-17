# Progress Log — Object-Size Transition & Single-vs-Bimanual Triggering

Execution log for [research_prompt_size_transition.md](research_prompt_size_transition.md).
Companion literature review: [auto_research_log_size_transition.md](auto_research_log_size_transition.md).

**Convention:** newest entry on top. Every claim cites a repo file:line or a command actually run.
Negative results and blockers are recorded honestly (per the prompt's mandate).

---

## 2026-10-17 — Session 4 (cloud): CPU re-analysis + code audit; the "decisive" size-invariant test is CONFOUNDED; GPU queue ready

**Agent:** Claude (Opus 5.5) on a CPU-only machine (the user's MacBook, arm64, `nvidia-smi` not found). No coldbrew,
no Isaac Gym, no checkpoints, no Aloha_real. **No simulation was run.** Every number below comes from the
committed CSVs or a CPU computation run here (commands listed). Source files were read from git at `main`
@ f9beecc. `dywa/` is untouched. Branch `cloud-session-4`.

### 0. Headline (updated)
- **Still true:** on the fixed table, single-arm success collapses steeply with object size under both raw and
  feasible-only metrics. 50% crossover, binomial-logistic fit with a seed bootstrap: **s* = 0.167 [0.164, 0.171]
  raw, 0.173 [0.171, 0.175] feasible-only** (3 seeds; `results/session4/frontier_stats.json`).
- **No longer established:** "the frontier is a pure single-contact CONTROL limit" (Session 2 §19). The
  size-invariant table v1 also grew the spawn and goal regions and pushed targets out of the arm's reach (§2).
  CRITIQUES #0 is re-opened.
- **Strengthened:** Deliverable C's shape-beyond-size signal survives leakage-free CV and is larger than first
  reported. About two-thirds of it comes from the env's yaw-only-goal rule for thin objects, not from physics (§4).
- **Clarified:** Deliverable E (scripted) shows a FLAT two-arm success of about 0.14 with no size trend. It
  cannot support any necessity claim (§5).

### 1. Seed statistics and the "immediate termination" mechanism
`python3 dywa_research/analysis/s4_frontier_stats.py` → `results/session4/{frontier_seed_stats,immediate_loop_check,phaseB_vs_seed_noise}.csv`
- At s = 0.20, the raw rate is over-dispersed **16×** relative to binomial; feasible-only is 1.5×. Raw 0.209 ± 0.079
  (seed SD), t95 [0.01, 0.40]; feasible 0.298 ± 0.031, t95 [0.22, 0.38].
- **Immediate terminations are ADDITIVE.** Non-immediate episode counts at s = 0.20 are 339 / 324 / 335 for seeds
  0 / 1 / 2, while immediate counts are 596 / 2 / 76. A few envs replay one bad spawn and re-terminate every few
  steps. In `mode=valid` each env gets ONE start pose for the whole eval (`num_valid_poses=1`, scene.py:256,
  974-981, 1365-1377). Feasible-only is therefore the right fixed-table metric. CRITIQUES #3, #5, #13.
- **Phase-B battery (§13-14 of Session 2):** every mass, reach and threshold delta at s = 0.20 is ≤ 0.69 seed-SD
  (seed SD 0.079), and all cells are seed 0, the seed with 596 looping episodes. "Heavier hurts monotonically"
  is **not supported**. "No knob recovers" still holds: even k/335 ≤ 0.33. ws 0.9 and 1.2 being byte-identical
  matches the arm's physical reach of 0.768 m (FK), which binds before either clip value. CRITIQUES #15.

### 2. Code audit: CORRECTIONS to earlier mechanism claims (read-only, cited)
1. **Valid mode has no radius contraction.** `_valid_poses` uses `_get_xy(prevent_fall=False)`
   (scene.py:1369-1376). So the Session-2 mechanism "placement = 0.5·margin·table − radius collapses at s ≈ 0.19"
   (§16-17) describes TRAIN mode, not the evals. This also explains why `prevent_fall=False` was byte-identical
   (§17 blamed `init_type=stable`; that was wrong).
2. **The OOB bound was misidentified.** push_task.py:512 is the GOAL region (±0.5·0.8·table). OOB uses
   `ws_lo/ws_hi` = table ± (0.5·dims + `ws_margin` 0.3) in xy, with z ≥ table centre (push_task.py:325-329,
   684-698). In practice, OOB means the object fell off the table or went more than 0.3 m past an edge.
3. **Robot base is hardcoded.** trossen.py:206-221 welds the base at x = −0.5 from a hardcoded table of 0.4.
   `++table_dims` does not move it. `Trossen.reset` writes x = −0.5·table_x − 0.3 into the root tensor
   (trossen.py:364-376), and the EE clip is centred on that tensor (trossen.py:418-426). With a non-default table
   the clip may be off by up to 0.21 m. Unverified; `gpu_s5_diag.sh` checks it.
4. **Possible valid-mode spawn-z bug (hypothesis).** `_valid_poses` is cloned in `create_assets`
   (scene.py:1367) BEFORE `scene.setup` rescales stable-pose z (scene.py:615-620; call order base.py:391-422). Up-scaled
   objects could therefore spawn inside the table, which is a candidate cause of the looping immediate
   failures. Unverified.
5. **Thin objects get yaw-only goals.** scene.py:68-70 + `use_yaw_only_logic: true`; push_task.py:779-808.

**Geometry audit of the size-invariant table v1** (`python3 dywa_research/analysis/s4_geometry_audit.py` →
`results/session4/geometry_audit.csv`). At s = 0.30, v1 gives a mean spawn-to-goal distance of 0.419 m (fixed table:
0.238), and **13.9% of goals and 21.6% of spawn CoMs are beyond the 0.768 m EE reach** (fixed: 0%), with the
table edge 0.09 m from the base. The v1 timeout flip is therefore confounded. The v1 drop at s = 0.15 (0.579 against
0.681 ± 0.012 fixed, where there are 0 immediate terminations) shows the v1 change made the task harder by itself.
Also, on the fixed table the EE can get behind an object, to its far side, for only 50% / 25% / 0% of spawns at
s = 0.10 / 0.20 / 0.30. That reach-around limit (H1') is a candidate *single-arm-specific* mechanism, and the one a
second, opposite arm would fix (CRITIQUES #16).

### 3. Negative result recorded: the single-seed, single-table "control limit" conclusion was premature
Session 2 §19 wrote "#0 RESOLVED; genuine controllability limit". In light of §2 that should read
"collapse persists on a bigger table whose task is also longer and partly out of reach". The fix
(`sizeinv_v2.sh`) holds the table and both regions constant and varies only the object.

### 4. Deliverable C: leakage-free re-evaluation
`python3 dywa_research/analysis/s4_predictor_cv.py` → `results/session4/predictor_*.{csv,json}`, `deliverable_C_grouped_cv_roc.png`
- The original 5-fold `cross_val_score` used unshuffled row folds sorted by scale (scale-blocked) with each
  object in both train and test.
- Object-disjoint 5-fold, 50 reps (logistic): scale-only AUC **0.68** (acc 0.69 against a 0.667 majority baseline);
  scale + shortest scaled extent AUC **0.82** (acc 0.79); original size + aspect/flatness AUC 0.82. Shortest
  extent adds +0.19 AUC over scale-only [0.11, 0.27] (object bootstrap).
- Within a fixed scale, flatness and aspect correlate with success at |ρ| = 0.50–0.58 (p < 0.001) at all three scales.
- **Confound:** a yaw-only-goal proxy (AABB M/S ≥ 2.5; 23 of 64 objects) alone gives AUC 0.78. Thin objects succeed
  +0.16 to +0.23 more often. Within the non-thin objects (full 6-D goals), bulkiness still predicts failure
  (ρ = −0.40 / −0.54 / −0.73), and it adds +0.06 AUC [0.006, 0.117] beyond scale + the yaw-only flag. Per-object
  labels average about 8 episodes from ONE spawn pose (#13). Cross-scale rank consistency is ρ ≈ 0.65.

### 5. Deliverable E: what the scripted lower bound does and does not show
`python3 dywa_research/analysis/s4_deliverable_e.py` → `results/session4/deliverable_E_{summary.json,matched_radius.csv,binned.csv,v2_scripted_ci.png}`
- Macro 0.142, object-bootstrap 95% CI [0.109, 0.172]. Logistic slope versus radius is −3.7/m, 95% CI [−7.3, +1.1],
  so there is **no detectable size trend**. Within each box, scale 1.3 versus 0.7 goes up for 5 boxes and down
  for 5. Boxes differ (χ² p = 0.023; Granimals 0.02).
- At matched radius, scripted two-arm success is far below single-arm for r ≤ 0.20 (0.17 against 0.72–0.78 in
  0.10–0.15; 0.15 against 0.33–0.48 in 0.15–0.20). It is equal at 0.20–0.25 (0.157 against 0.07–0.23). Above 0.25 it is
  0.095 [0.062, 0.143] against 0.009–0.118. The point-estimate crossover of the fit is r ≈ 0.20–0.26 depending
  on the single-arm curve.
- **Reading:** the scripted primitive fails for reasons unrelated to size (it is flat), so it cannot test
  necessity. The fact that it does not decay with size is the only weakly encouraging signal. It is cross-sim,
  with different objects and success criteria (the mjlab success definition was not checked this session).

### 6. Bimanual-dywa scaffold review (read-only)
`dywa_research/plans/bimanual_scaffold_review.md`: 8 defects. The blocking ones: both arms resolve to the same
actor (`'robot'` name lookup, trossen.py:222-263); both are welded at the same hardcoded pose; DOF slicing
`[:, :N_ARM]` is hardcoded; the DOF-target buffer has the wrong shape for 2 robots. Only T1 is (partly) done.
The scaffold has not been shown to work.

### 7. GPU queue prepared (coldbrew GPU reported free by the user, 2026-10-17). All NOT yet run.
- `gpu_s5_queue.sh`: one command, stages 1–5:
  - 1: `gpu_s5_diag.sh` re-runs 4 existing cells through `valid_ppo_arm_diag.py`, a read-only per-episode
    dump generated by `patch_diag.py`. Its outputs must reproduce the existing CSVs (behaviour-neutral check).
    `analysis/s5_diag_report.py` then tests H-loop, H-z, H-dist and H-base.
  - 2: `sizeinv_v2.sh`: constant table T* = [0.821, 1.026], `margin_scale` 0.46285 / 0.38977 so the regions
    equal the baseline. 8 scales × 3 seeds, plus a distance control (s = 0.10 on the v1 0.30 table).
  - 3–4: `gpu_necessity_finetune.sh` (#1) first as a smoke run, then a 100k-iteration fine-tune with scale DR
    [0.05, 0.30] on the v2 env. A matching in-distribution-DR control is available (SMAX=0.10).
  - 5: evaluates the fine-tuned ckpt on v2.
- `gpu_bimanual_residual.sh`: searches for a surviving residual checkpoint, then prints the runbook's training
  section. It needs `TRAIN_CMD` from the runbook (not known here; the script does not guess). It then runs a
  paired scripted vs trained eval on `bimanual_scales.yaml` and on the new matched-radius
  `bimanual_scales_radius_grid.yaml` (`analysis/make_bimanual_radius_grid.py`).

### 8. Pruned (separate commit "prune obsolete ..."; git history keeps everything)
See the PR body: `bigregion.sh` + `results/phase_c_bigregion/` (duplicate of the taxonomy CSV; its null result is
now explained by §2.1), `gpu_wait_bimanual.sh` (obsolete watcher), and the stale Session-1 test_rma/Docker
harness detail in this log.

### NEXT
1. **[GPU]** `conda activate dywa_render && setsid bash dywa_research/gpu_s5_queue.sh > dywa_research/results/s5_queue.log 2>&1 < /dev/null &`
   Stage 1 first: check the diag cells reproduce `multiseed.csv` seed 0, then read `results/s5_diag/diag_report.txt`.
2. **[GPU]** If v2 still collapses at a similar s*: the control-limit claim is restored and the necessity
   result (stage 4) becomes interpretable. If v2 recovers: the frontier was largely task geometry. Either way,
   rewrite Deliverable D.
3. **[GPU]** Bimanual: put the runbook's training command in `TRAIN_CMD` and run `gpu_bimanual_residual.sh`.
4. **[CPU]** After 1–3: re-run `analysis/s4_*` on the new CSVs, condition success on goal direction (H1'), and
   raise `num_valid_poses` for headline numbers (#13).
5. **[GPU, longer]** In-DyWA bimanual: fix the Trossen actor_name / base-pose / dof_offset items in the scaffold review first.

---

## 2026-08-21 — Session 3: mjlab bimanual env RECONSTRUCTED; scripted bimanual eval running

### Resume + environment reconstruction (Deliverable E prep)
Resumed to run the cross-sim bimanual eval. GPU freed (hanwen's job done); the Aug-16 watcher had died
on reboot (~Aug-17) without running the eval. Ran `bimanual_eval.sh` directly and cleared a chain of blockers:
- **scipy** missing in mjlab_venv -> installed.
- **Lost asset prep** (runbook's uncommitted working-tree changes, gone on reboot): `TROSSEN_OBJECTS_DIR`
  (=`<trossen_wxai>/xmls/assets/meshes/objects`) didn't exist -> created symlink `objects -> objects_cat_box_obj`.
  `DEFAULT_OBJECT_NAME='fridge_top_door'` (asset absent) -> set to a box object `Crayola_Crayons_120_crayons`
  (1 line). Both are revertable Aloha_real changes matching the runbook's described prep.
- Eval needs `--checkpoint` OR `--scales_manifest` even for `--scripted_only` -> wrote
  `bimanual_scales.yaml` (10 box objects x scales 0.7/1.0/1.3 = 30 variants); rewrote `bimanual_eval.sh`
  to use it (drops `--object_names`).
- **ROOT version-rot**: mjlab was authored for mujoco-warp 3.5.0.2, but the earlier setup installed 3.11
  (the pinned mujoco `3.7.1.dev` nightly had 404'd). 3.11 broke on multiccd/margin/`ls_parallel`/`wp.context`.
  **RECONSTRUCTED the correct stack**: `mujoco==3.7.0` (stable PyPI, has `mjENBL_MULTICCD`) +
  `warp-lang==1.12.0` + `mujoco-warp 3.5.0.2` (git rev 5a86ec2). All version errors vanished; physics is
  authentic (no engine patches) -> trustworthy numbers.

### Bimanual eval RUNNING (results pending at handoff)
`bimanual_eval.sh` (scripted 2-arm push, 30 variants x 30 episodes) running on GPU. Confirmed a genuine
bimanual env: `joint_pos (16,)`=2x8dof, `left/right_ee_to_object`, `actions (12,)`. Results print at the
end -> `results/bimanual/eval.log`.
NEXT: parse per-object success; map each box bbox -> bounding radius; overlay bimanual vs the single-arm
frontier = Deliverable E (cross-sim, illustrative); then log + push.


---

### Deliverable E (SCRIPTED lower bound) — single vs double, cross-sim
Scripted bimanual eval finished: 30 box×scale variants, per-object success 0.00–0.30, **MACRO AVG 0.142**.
Mapped each box's half-extents (runbook §1) -> bounding radius (0.116–0.327 m); overlaid vs the single-arm
frontier -> results/bimanual/deliverable_E_single_vs_double_SCRIPTED.png (+ bimanual_results.csv).
FINDING: the SCRIPTED bimanual primitive is a WEAK baseline (~14%, low across all radii) and does NOT beat
single-arm — at small radius single-arm is far higher (0.68–0.93 vs ~0.14), at large radius both are low.
EXPECTED: scripted primitive alone is weak; the runbook trains a residual RL policy ON TOP to make bimanual
strong, and that checkpoint was wiped on reboot. So Deliverable E is currently a LOWER BOUND only; a fair
single-vs-double claim needs the trained residual policy (retrain via runbook ~hours, or locate model_3300.pt).
Cross-sim => illustrative regardless.

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

**Leading hypotheses for the 0%** (determinism, curriculum, force_rad, task.mode): condensed in Session 4. All were
superseded by §9 (stochastic still 0%) and §10 (root cause: ws_bound x-max 0.25 and the wrong filter_file).

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

### 11. ANCHOR PASSES — eval VALIDATED (94.5%), Phase-A UNBLOCKED
anchor_eval.sh result: episode_count=615, success_count=581, **suc_rate=0.9447 (94.5%)**,
balanced_suc_rate=0.9355. Matches the reported ~93.1% teacher. GATE 1 (0 mesh errs) + GATE 2
(~93%) both PASS. The eval pipeline is now legitimate and under our control, fully inside
dywa_research/. Confirmed the 0% was purely the ws_bound/filter config error.
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
- Plateau ~93-97% across the training range [0.05,0.10] (held-out objects) -> teacher generalizes.
- SHARP collapse between 0.15 (69%) and 0.20 (12%). **Crossover s* ~= 0.167 m** (50%-success,
  linear interp). By 0.30, success ~= 0 -> single-arm non-prehensile control fails on large objects.
- This is the Phase-A headline the prompt asked for: a measured single-arm size frontier with a
  crossover, on the same simulator/teacher as the 93% result. Figure:
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
> **[Session 4 correction]** The "usable_x = 0.38 − 2s" mechanism applies to TRAIN mode only. In the evals the immediate failures are an additive looping-env artifact (2026-10-17 entry §1–2).
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
> **[Session 4 correction]** push_task:512 is the GOAL region, not the OOB bound. The valid-mode spawn has no radius contraction, so `prevent_fall` is hardcoded False there; that, not `init_type=stable`, is why the test was byte-identical (2026-10-17 entry §2).
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

### 18. DELIVERABLE C (partial) - single-arm COMPETENCE predictor = the trigger's single-arm branch
Built the decision module 'can ONE arm do THIS object?' from per-object success (64 objects x 3
transition scales 0.125/0.15/0.175 = 192 points) vs object geometry (from clouds).
Results (interpretable): logistic & depth-3 tree both CV acc ~= 0.697.
  Pearson r with success: scaled_S=-0.565 (shortest extent, STRONGEST), scale=-0.509,
  flatness=-0.466, aspect=+0.296 (elongated -> more success), scaled_L=-0.279 (weakest).
  Tree importances: scaled_S 0.54, scale 0.22, scaled_M 0.17, aspect 0.075.
  SHAPE-beyond-SIZE: size-only acc 0.640 -> size+shape 0.687 (delta +0.047) => shape adds a real,
  modest signal beyond size.
INTERPRETATION: single-arm competence is best predicted by the object's SHORTEST scaled extent
(bulkiness), not overall size: one arm handles THIN/FLAT objects to larger sizes but fails on BULKY/
chunky ones. This is the explicit, interpretable single-arm branch of a single-vs-bimanual trigger.
CAVEATS: ~70% acc is first-pass (single seed, 64 objects, and the fixed-table confound (#0) adds
noise/leaks a size-fit signal); the +5% shape term is the part least table-confounded.
NOT YET DONE (answering the user directly): this is only the SINGLE-ARM branch. A full single/double
ADAPTIVE policy also needs (Phase D) a real bimanual 2-Trossen policy to compare against, which needs
a dual-arm ENV (repo modification, conflicts with the 'keep to dywa_research' rule) and is gated on
the table confound (#0). Deliverable_C_competence.png + competence_dataset.csv saved.
---

### 19. DECISIVE — size-invariant-table test: frontier is a REAL control limit (#0 RESOLVED)
> **[Session 4 correction]** This test is confounded. It also grew the spawn and goal regions and put 14–22% of targets out of reach (see the 2026-10-17 entry §2). #0 is re-opened.
Scaled the table WITH the object (table_dim=(usable_half_baseline+s)/0.475 per axis) so the
placement/OOB region stays constant = the s=0.10 baseline. Pure config override (table is procedural
gym.create_box; the URDF default is unused). Held-out, force_rad=0.10, seed0:

| scale | table[x,y]   | success | oob   | timeout | immediate | fixed-table succ |
|-------|--------------|---------|-------|---------|-----------|------------------|
| 0.10  | [0.40,0.50]  | 0.931   | 0.050 | 0.019   | 0.000     | 0.931 |
| 0.15  | [0.51,0.61]  | 0.579   | 0.067 | 0.354   | 0.000     | 0.694 |
| 0.20  | [0.61,0.71]  | 0.120   | 0.026 | 0.854   | 0.000     | 0.12-0.21 |
| 0.25  | [0.72,0.82]  | 0.012   | 0.012 | 0.977   | 0.000     | 0.027 |
| 0.30  | [0.82,0.92]  | 0.000   | 0.023 | 0.977   | 0.004     | 0.001 |

CONCLUSION: removing the fixed-table placement artifact drives immediate terminations to ~0, yet the
frontier STILL collapses (s* ~= 0.155, essentially unchanged). The failure mode FLIPS from OOB to
TIMEOUT (0.85-0.98 at collapse): with room and no boundary loss, a single arm cannot maneuver a large
object to the 6D goal in time. => a GENUINE single-contact controllability limit (H2), not a table
artifact. #0 RESOLVED; bimanual-necessity direction re-validated. Failure is now clean/interpretable.
UNBLOCKED: #1 necessity fine-tuning is now scientifically valid (run it on the size-invariant table
with extended scale DR). Residual control worth one check: does a longer episode timeout help?
Figure: results/phase_d_sizeinv/deliverable_D_control_limit.png. CRITIQUES #0 -> RESOLVED.
---

### 20. Bimanual mjlab env FIXED (dependency hell resolved); eval ready
Resumed the single-vs-double thread. The user has a dual-arm non-prehensile Trossen PUSH env (mjlab):
/home/bruce/Aloha_real/sim_3d_bimanual-main. Fixed the broken bring-up (their uv.lock pinned a 404'd
mujoco nightly + mujoco-warp git rev; driver is CUDA 12.9 so torch cu130 gave cuda=False):
  - torch -> 2.11.0+cu128 (cuda True, 1 device; driver 12.9 supports cu128 not cu130).
  - mujoco-warp -> 3.11.0 (git main) matching the only-available mujoco 3.11.1 (older pin removed the
    mjENBL_MULTICCD API; nightly index purged all 3.7.x builds).
  - venv at dywa_research/mjlab_venv; Aloha_real tree UNTOUCHED (uv.lock restored). mjlab imports clean.
Eval ready (dywa_research/bimanual_eval.sh): scripted bimanual push primitive on the 10 cat_box objects
(half-extents up to ~0.15 m => bounding radius ~0.15-0.23 m = the single-arm collapse regime),
--scripted_only, per-object success. BLOCKED only on GPU (another user's job at 100%/22GB); will run
when it frees. Comparison plan (Deliverable E): overlay bimanual box success vs single-arm frontier at
matched bounding-radius; cross-sim => illustrative, clearly labeled.
---

### 21. Headline conclusion (for the record) + generality note + how-we-differ
ONE-LINER: "For this single Trossen arm, objects beyond ~0.16 m bounding radius (~30 cm across)
become too large for one contact to steer to a target pose -- success collapses from ~93% to near
zero, dominated by timeouts."
0.16 m = object BOUNDING RADIUS (spans ~0.32 m across); it is the 50%-success crossover. Failure =
timeout (can't steer a large object's pose in time) = single-contact controllability limit, NOT
inability to nudge.
GENERALITY (honest): currently NARROW -- one arm (small Trossen WidowX), one sim, one task, teacher
policy; s*~=0.16 m is ARM-SPECIFIC, not universal. The general contribution is the METHOD (measure
frontier + isolate cause) + the mechanistic finding, which transfer even though the number does not.
For generality/publishability: add cross-robot (Franka), the necessity experiment, and the bimanual
comparison. Literature how-we-differ table added to auto_research_log (2026-08-16): prior work frames
large objects as GRASPABILITY; we measure single-arm CONTROL vs size + prove controllability.
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

### 4–6. Eval entry, Phase-A harness, blockers (condensed in Session 4; full text in git history before the S4 prune commit)
- The Docker/`test_rma.py` student-eval path was reverse-engineered here and then **abandoned** for the bare-metal
  teacher entry `valid_ppo_arm.py` (Session 2 §2). Its harness files (`phase_a_scale_sweep.sh`,
  `aggregate_scale_sweep.py`) no longer exist; Deliverable A was produced by `phase_a_sweep.sh`.
- Key fact kept: **never pass `++load_ckpt=<teacher>` when evaluating a distilled STUDENT with `test_rma.py`**
  (`dev_log.md:126-147`, drives success to 0%). This does not apply to the teacher entry `valid_ppo_arm.py`.
- Key fact kept: pin one eval scale with `min_scale == max_scale == s`; the goal tolerance knob is `++force_rad`
  (it overrides `env.task.goal_radius`, Session 2 §13).
- The Session-1 blockers (GPU contention, Docker runtime, teacher-vs-student path, radius semantics) were all
  resolved in Session 2.
