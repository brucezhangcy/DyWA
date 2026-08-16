# RESUME — DyWA size-frontier → single-vs-double study

> NOTE (user instruction): after this file has been used to resume, DELETE it (git rm + push) — its
> content is preserved in `progress_log_size_transition.md`.

**Start here.** Full history: `progress_log_size_transition.md` (newest on top). Critiques: `CRITIQUES.md`.
Literature + how-we-differ: `auto_research_log_size_transition.md`. Host: `coldbrew`. Repo
`brucezhangcy/DyWA`, branch `main` (latest `89097ba2d`). All work in `dywa_research/`; `dywa/` source and
`output/trossen_pipeline` untouched on main; `Aloha_real/` tree untouched.

## Where the science stands (all pushed)
- **Single-arm study COMPLETE:** validated eval (~93% anchor); size frontier **s\*≈0.16 m bounding
  radius** (Deliverable A, multi-seed); failure taxonomy = timeout-dominated (B); **fixed-table confound
  RESOLVED** — size-invariant-table test shows the frontier survives ⇒ a real single-contact
  **controllability limit** (not reach/mass/threshold/table); competence predictor (C, ~0.70 acc,
  shape adds ~5%). Headline: *"for this single Trossen arm, objects beyond ~0.16 m bounding radius
  (~30 cm across) are too large for one contact to steer to a target pose — success collapses ~93%→~0,
  dominated by timeouts."*
- **Honest limitation:** narrow (one small arm, one sim, one task, teacher policy); 0.16 m is
  arm-specific. Generality needs cross-robot (Franka) + necessity + bimanual.

## ACTIVE TASK when paused: single-vs-double comparison (Deliverable E)
User decided **LEAN route first** (mjlab cross-sim), rigorous in-DyWA route optional/deferred.

**mjlab bimanual (cross-sim, illustrative) — ARMED, waiting on GPU:**
- Env FIXED at `dywa_research/mjlab_venv` (torch 2.11+cu128 cuda True, mujoco 3.11.1, mujoco-warp 3.11).
- `dywa_research/gpu_wait_bimanual.sh` (detached, PID may still be alive) auto-runs
  `dywa_research/bimanual_eval.sh` (`--scripted_only` on 10 cat_box objects) when GPU <8 GB used →
  writes `results/bimanual/eval.log` (+ `watcher.status`). BLOCKED on the shared RTX 4090 (hanwen's
  RoboTwin pi05 job at ~100%/22 GB).

### On resume — do this
1. `ssh coldbrew` — check GPU (`nvidia-smi`), the watcher (`cat dywa_research/results/bimanual/watcher.status`),
   and whether `results/bimanual/eval.log` exists (watcher may have run it).
2. **If eval.log exists:** parse per-object bimanual push success; map each cat_box object's half-extents
   → bounding radius (√(hx²+hy²+hz²), ~0.15–0.23 m); **overlay bimanual vs the single-arm frontier at
   matched radius = Deliverable E** (label cross-sim/illustrative). Plot + log (new progress §) + push.
3. **If not run & GPU free:** run `bash dywa_research/bimanual_eval.sh > results/bimanual/eval.log 2>&1`
   manually. **If GPU busy:** leave the watcher; or (if watcher died) re-arm `gpu_wait_bimanual.sh` detached.

## Other open threads (lower priority)
- **#1 necessity** (main, no repo edits, GPU-gated): fine-tune the single-arm teacher WITH large objects
  on the size-invariant table; if it STILL fails → control limit is physical, not OOD.
  `train_ppo_arm.py ++load_ckpt=<teacher> + extended scale DR`.
- **Rigorous in-DyWA bimanual** (branch `bimanual-dywa`): plan `dywa_research/plans/bimanual_build_plan.md`
  + UNTESTED scaffold `dywa/src/env/robot/trossen_bi.py`. Resume there ONLY if the lean result warrants a
  same-sim number; GPU-debug from plan task T3 (the key change: each Trossen needs a `dof_offset` since
  `apply_actions` hardcodes `env.tensors['dof'][:, :N_ARM]`; object actor index shifts with 2 robots).
- **Timeout-budget control:** does a longer episode timeout raise large-object success? (one cheap run).
- **Cross-robot (Franka):** replicate the frontier to test arm-specificity.

## Ops notes / gotchas
- Eval runtime: conda `dywa_render` via `valid_ppo_arm_patched.py` (+ `shims/` + `urdf_fixed/`).
  **Goal tolerance = `++force_rad`** (it OVERRIDES `task.goal_radius` in `__post_init__` — verify the
  RESOLVED config in run logs, not just the CLI).
- Commit only `dywa_research/` paths, from repo ROOT (`cd /home/bruce/DyWA`), message via `-F /tmp/f`.
  gitignored: `urdf_fixed/`, `mjlab_venv/`, `*.log`, per-cell result dirs.
- SSH self-disconnect: never `pkill -f <pat>` where `<pat>` matches the ssh command line; kill python by
  iterating `pgrep -x python` + `/proc/PID/cmdline`. Run GPU jobs detached (`setsid … </dev/null &`).
- Recent housekeeping: trimmed 4 superseded scripts + progress-log bloat (dup §21, abandoned test_rma
  block, `%%`).
