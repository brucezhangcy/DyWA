# RESUME — DyWA size-frontier → single-vs-double study

> NOTE (user instruction): after resuming, DELETE this file (git rm + push) — content is preserved in
> `progress_log_size_transition.md`.

**Start here.** Full history: `progress_log_size_transition.md` (newest on top; Session 3 = 2026-08-21).
Host `coldbrew`, repo `brycezhangcy/DyWA` branch `main` (latest ~`e4dd95ace`). Work in `dywa_research/`.

## Science state (single-arm COMPLETE, pushed)
Frontier s*≈0.16 m bounding radius (Deliverable A); timeout-dominated failure (B); **fixed-table
confound RESOLVED -> real single-contact controllability limit**; competence predictor (C). Headline:
"for this single Trossen arm, objects beyond ~0.16 m bounding radius (~30 cm across) are too large for
one contact to steer to a target pose — success ~93%→~0, timeout-dominated." Narrow/arm-specific (honest).

## ACTIVE: Deliverable E — single-vs-double (mjlab cross-sim)
**mjlab env RECONSTRUCTED & WORKING** in `dywa_research/mjlab_venv`: `mujoco 3.7.0` + `warp-lang 1.12.0`
+ `mujoco-warp 3.5.0.2` (git rev 5a86ec2) — the versions mjlab was authored for (the pinned mujoco
nightly had 404'd; earlier 3.11 install broke on API drift). No engine patches -> authentic physics.
Aloha_real prep (revertable, matches runbook): symlink `<trossen_wxai>/xmls/assets/meshes/objects ->
objects_cat_box_obj`; `env_cfgs.py` `DEFAULT_OBJECT_NAME='Crayola_Crayons_120_crayons'`.

**Bimanual eval:** `dywa_research/bimanual_eval.sh` (scripted 2-arm push, `--scripted_only`,
`--scales_manifest bimanual_scales.yaml` = 10 box objects x scales 0.7/1.0/1.3) -> per-object success in
`results/bimanual/eval.log`. First run: **scripted-bimanual MACRO AVG ≈ 0.14** (objects 0.07–0.27).

### CRITICAL caveat for Deliverable E
This is the **SCRIPTED primitive only** (no trained policy) — the runbook trains a **residual RL policy
ON TOP** to make it good; that checkpoint is NOT in `logs/`. So ~14% is a WEAK bimanual baseline, NOT a
fair "two arms" number, and does NOT yet beat single-arm at matched size. For a real single-vs-double
claim, need the **trained residual checkpoint** (retrain via runbook `docs/runbooks/
cat_box_subset_10_residual_push.md`, ~hours GPU) OR locate it. Report scripted-only only as a
lower-bound, clearly labeled.

### On resume — do this
1. `ssh coldbrew`; check the eval finished: `tail -60 dywa_research/results/bimanual/eval.log` for the
   full per-object table (30 variants) + MACRO AVERAGE. (Background waiter `bazcud3rx` was set to dump it.)
2. Parse per-object success; map each box's half-extents -> bounding radius (√(hx²+hy²+hz²); half-extents
   in runbook `cat_box_subset_10_residual_push.md` §1). Overlay **bimanual (scripted) vs single-arm
   frontier** vs radius = Deliverable E, LABELED "scripted-only lower bound, cross-sim/illustrative".
3. Decide with user: is scripted-only enough (as a lower bound), or retrain the residual policy for a
   strong bimanual number? Then log (new progress §) + push.
4. If eval died before finishing: re-run `bash dywa_research/bimanual_eval.sh > results/bimanual/eval.log 2>&1`
   (env is reconstructed; GPU-gated).

## Other threads
- #1 necessity (single-arm, main, GPU): fine-tune teacher WITH large objects on size-invariant table.
- Rigorous in-DyWA bimanual: branch `bimanual-dywa` (untested scaffold `trossen_bi.py` + plan). Only if a
  same-sim number is wanted; GPU-debug from plan T3.
- Cross-robot Franka (arm-specificity). Timeout-budget control.

## Ops
- Single-arm eval: conda `dywa_render` via `valid_ppo_arm_patched.py` (+ shims + urdf_fixed). Goal
  tolerance = `++force_rad` (overrides task.goal_radius; verify RESOLVED config).
- Commit `dywa_research/` paths from repo ROOT, `-F /tmp/f`. gitignored: urdf_fixed/, mjlab_venv/, *.log,
  per-cell dirs. SSH: never `pkill -f <pat>` matching the ssh cmdline; kill python by pid/proc-cmdline.
- Aloha_real tree: only the 2 revertable prep changes above; don't otherwise modify.
