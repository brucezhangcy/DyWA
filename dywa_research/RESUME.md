> NOTE (user instruction): after this file has been used to resume, DELETE it (git rm + push) to save space — its content is preserved in progress_log_size_transition.md.

# RESUME — DyWA single-arm size frontier → single-vs-double (bimanual) study

**Single entry point for resuming.** Full history: [progress_log_size_transition.md](progress_log_size_transition.md)
(newest section on top). Anticipated critiques + status: [CRITIQUES.md](CRITIQUES.md).
Everything here lives in `dywa_research/`; the DyWA repo source and `output/trossen_pipeline`
(user's trossen eval) are **untouched**. Host: `coldbrew` (ssh alias). Last updated: 2026-08-10.

## Where the science stands (all pushed to origin/main)
- **Validated eval**: Trossen Stage-1-wide teacher reproduces ~93% in-dist (anchor 94.5%). Root-caused
  the earlier 0% to `ws_bound` x-max 0.25→**0.6** + filter `yes.json`. Runs bare-metal in conda
  `dywa_render` with shims (`shims/`) + patched entry `valid_ppo_arm_patched.py` + `urdf_fixed/`
  (corrects stale coacd mesh paths; gitignored, rebuild via urdf_fixed_build if missing).
- **Deliverable A** — single-arm size frontier (held-out): 93/58/12/1/0% at s=0.10/0.15/0.20/0.25/0.30
  (multi-seed mean; single-seed s=0.20 was an outlier). s*≈0.155–0.17.
- **Mechanism**: reach REJECTED, threshold REJECTED, inertia modulates-not-causal.
- **Fixed-table confound (CRITIQUES #0) RESOLVED**: size-invariant-table test (scale table with
  object; table is procedural `gym.create_box`, so pure config override) → immediate terminations→0
  but frontier STILL collapses; failure flips OOB→**TIMEOUT** ⇒ genuine single-contact
  controllability limit (H2). **This validates the bimanual-necessity direction.**
- **Deliverable C** — single-arm competence predictor (trigger's single-arm branch): CV acc ~0.70;
  strongest feature = scaled shortest extent (bulkiness); shape adds ~5% beyond size.
- Key metric knob learned: goal tolerance is set by `++force_rad` (it OVERRIDES `env.task.goal_radius`
  in __post_init__). Always verify the RESOLVED config in the run log, not just the CLI.
- Results/figures under `results/` (phase_a, phase_b_*, phase_c_*, phase_d_sizeinv).

## ACTIVE TASK when paused: bring up the mjlab BIMANUAL env and eval it (single-vs-double)
The user HAS a dual-arm non-prehensile **Trossen push** env (mjlab/MuJoCo):
`/home/bruce/Aloha_real/sim_3d_bimanual-main` (+ `binomap`, `pi0.5-bimanual`). Same robot+task family
as DyWA, different sim ⇒ cross-sim = **illustrative** comparison (honestly labeled). The eval script
`scripts/eval_residual_push_multi.py` supports `--scripted_only` (runs the scripted bimanual push
primitive with NO trained checkpoint) and reports per-object success.

### BLOCKER (dependency hell) — this is where to resume
- Their `uv.lock` pins **mujoco `3.7.1.dev904477783`** from `py.mujoco.org` → **404 (nightly removed)**,
  and **mujoco-warp** via git rev `5a86ec28aa07741eb2e000d158f4ca4068ec146e`
  (github.com/google-deepmind/mujoco_warp), torch **2.9.0+cu128**, `rsl-rl-lib==5.0.1`,
  `warp-lang>=1.12` (pypi.nvidia.com), python 3.11.
- My fresh `uv sync --upgrade` (venv at `dywa_research/mjlab_venv`, their lock RESTORED, tree intact)
  resolved WRONG versions: `mujoco 3.11.1` + a mujoco-warp that crashes on import
  (`AttributeError: mjtEnableBit has no attribute mjENBL_MULTICCD`), and `torch 2.13.0+cu130` with
  **cuda=False**. So the venv exists but is version-broken.

### Resume steps for mjlab
1. Rebuild the venv with the PINNED, compatible versions (do NOT `--upgrade`):
   - `torch==2.9.0+cu128` from `https://download.pytorch.org/whl/cu128` (fixes cuda=False; matches driver).
   - `mujoco-warp` from git rev `5a86ec28aa07741eb2e000d158f4ca4068ec146e`.
   - A `mujoco` version COMPATIBLE with that mujoco-warp rev (has `mjENBL_MULTICCD`). The exact pinned
     dev wheel 404s → find the nearest available nightly on `py.mujoco.org` (or a stable 3.7.x that
     still has that enum), OR ask the user for the working env/machine (training ran 2026-04; a working
     env may exist on another box). Consider `uv sync --frozen` if the wheels come back, or override
     just `mujoco` to an available pin while keeping the mujoco-warp git rev.
   - Keep the venv OUT of their repo: `UV_PROJECT_ENVIRONMENT=/home/bruce/DyWA/dywa_research/mjlab_venv`.
     Always back up + restore their `uv.lock` (`/tmp/uv.lock.bak`) so `Aloha_real` stays undisturbed
     (it has UNCOMMITTED changes — never modify tracked files there).
2. Verify: `PYTHONPATH=src mjlab_venv/bin/python -c "import torch,mujoco,mjlab,mujoco_warp; print(torch.cuda.is_available())"` → must be True and import clean.
3. Find the BOX object set: `eval_residual_push_multi.py` OBJECT_SETS only registers `cat_flat_6`,
   `cat_round_7`; the box set is `objects_cat_box_obj` (see `src/mjlab/tasks/manipulation/config/trossen/env_cfgs.py:73`
   and `scripts/object_list_generating/objects_cat_box_obj_subset_10.txt`). Either add a `cat_box`
   entry via `--object_names` or register it; box half-extents up to ~0.15 m (full ~0.30 m) = the
   large-object regime where the single arm collapses.
4. Run (from `sim_3d_bimanual-main`, PYTHONPATH=src):
   `mjlab_venv/bin/python scripts/eval_residual_push_multi.py --scripted_only --object_names <box list> --num_envs 32 --episodes_per_object 30`
   → per-object bimanual push success. (A trained residual checkpoint is NOT in `logs/`; runbook
   `docs/runbooks/cat_box_subset_10_residual_push.md` documents training if we want the RL policy.)
5. **Deliverable E**: overlay bimanual success on the DyWA single-arm frontier (Deliverable A / the
   size-invariant control-limit curve) at matched object sizes. Expected story: two arms hold the
   large-object regime where one arm TIMES OUT. Label clearly as cross-sim/illustrative.

## Other open threads (lower priority)
- **#1 necessity** (now UNBLOCKED, no repo edits): fine-tune the single-arm teacher WITH large objects
  on the size-invariant table; if it STILL fails, the control limit is confirmed physical (not OOD).
  Training entry `dywa/exp/train/train_ppo_arm.py` with `++load_ckpt=<teacher>` + extended scale DR.
- **Timeout-budget control**: does a longer episode timeout raise large-object success? (residual on
  the size-invariant result).
- **#4/#8 polish**: more seeds/objects; per-object shape features for the predictor.

## Git / ops notes
- Repo `github.com/brucezhangcy/DyWA`, branch `main`; user pushes to main to sync machines. Commit
  only `dywa_research/` paths, from the REPO ROOT (`cd /home/bruce/DyWA`), message via `-F /tmp/file`.
  `.gitignore` excludes `urdf_fixed/`, `*.log`, per-cell result dirs, `mjlab_venv/`.
- SSH self-disconnect gotcha: never `pkill -f <pattern>` where <pattern> matches the ssh command
  line — kill python procs by iterating `pgrep -x python` + `/proc/PID/cmdline`. Run evals detached
  (`setsid … </dev/null &`) and poll logs.
- Last commits: c7333e27f (size-invariant control-limit), 3e577d70c (Deliverable C).
