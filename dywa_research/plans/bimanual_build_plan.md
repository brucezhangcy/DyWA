# DyWA Bimanual (2× Trossen) Build Plan

**Goal:** Add a two-arm Trossen non-prehensile push mode to DyWA (Isaac Gym) so the bimanual policy is
trained/evaluated in the SAME simulator, reward, metrics, and object pipeline as the single-arm
frontier — a true apples-to-apples comparison (unlike the cross-sim mjlab route).

**Approach:** Additive, on branch `bimanual-dywa`. Prefer NEW files + a `which_robot='trossen_bi'`
switch over editing single-arm code paths, so the existing single-arm setup is untouched and the whole
thing is reversible (`git checkout main`). Build iteratively with runtime verification at `num_env=4`
(Isaac Gym bugs — especially tensor indexing — only surface at runtime; every task ends by RUNNING).

**Tech:** Isaac Gym Preview 4, conda `dywa_render`, PPO (rsl-rl-style, `train_ppo_arm.py`). GPU-gated.

---

## Architecture (verified by reading the code)
- `env/env/base.py:821-834` (`EnvBase`) creates per-env actors inside aggregate blocks: `scene` then
  `robot`. **This is the single integration point to add a 2nd robot actor.** Adding an actor shifts
  global actor/DOF/rigid-body indices → the object/table indices and all tensor slicing must account
  for it.
- `env/robot/trossen.py:196` (`create_actors`) welds ONE arm at a baked base `gymapi.Transform()`;
  `n_dofs`/`n_bodies`/Jacobian computed for one arm. Action apply + IK are per-arm.
- `env/task/push_with_arm_task.py` computes `hand_state`/`robot_state` obs and the reward from the
  (single) EE. `push_env.py` is thin: `apply_actions`→`robot.apply_actions`; obs assembled by wrappers/
  task, `compute_observations` returns `{}`.
- Action space = `robot.action_space` (6-DOF); config picks robot via `which_robot`.

## Key risks
1. **Tensor/actor global indexing** (highest). With 2 robots the per-env DOF count doubles and the
   object's actor index shifts; root-state / dof-state / jacobian / contact tensors are sliced by these
   indices across the whole env. Mis-indexing → silent wrong physics or crashes. Mitigate: verify body/
   actor counts and print index maps at `num_env=4` before any training.
2. **Base placement / reachability** — two arms must bracket the table within reach; reuse the
   create-time `gymapi.Transform()` welded-base fix (progress log / dev_log). Make arm-2 base pose a
   config knob and sweep separation.
3. **Reward shaping** for two contacts (shared goal-pose error; optional anchor/contact term).
4. **From-scratch training** (~3-4 days GPU per dev_log single-arm figure); single-arm weights do NOT
   transfer (12-DOF action, new obs).

---

## Tasks (bite-sized; each ends by RUNNING at num_env=4)

### T1 — Dual-Trossen robot wrapper (new file, no base edits yet)
- Create `dywa/src/env/robot/trossen_bi.py`: a `TrossenBi` robot that owns two `Trossen` instances with
  distinct base poses (`base_pose_left/right` config), and exposes `action_space=12`, `create_actors`
  (creates both), `apply_actions` (splits 12→2×6), and combined EE/`hand_state` (stacked).
- Reuse `Trossen` internals; parametrize its baked base `Transform` (currently hardcoded in
  `create_actors:196-224`) via a `base_pose` arg (small additive change to `Trossen.create_actors`
  signature with a default = current behavior, so single-arm is unchanged).
- **Verify:** import + instantiate cfg; assert `action_space==12`, two base poses distinct.

### T2 — Register the robot + config
- Add `which_robot='trossen_bi'` branch in `push_env.py:105-123` (additive elif) → `self.robot_cfg`.
- New run/env cfg `cfg/env/trossen_bi_icra_base.yaml` + `cfg/run/trossen_bi_ours.yaml` cloned from the
  trossen single-arm ones, with `which_robot: trossen_bi`, doubled action dims, arm base poses, widened
  `ws_bound` for two arms.
- **Verify:** hydra composes the cfg; `load_env` builds the env object (may fail at sim init — that's T3).

### T3 — Two actors in base.py (the core change) + index bookkeeping
- In `env/env/base.py:833`, after the robot aggregate block, add a second block:
  `actors['robot2'] = self.robot.create_actors2(...)` (or have `TrossenBi.create_actors` create both
  inside one block). Update `self.__counts` for the extra bodies/shapes.
- Audit every place that assumes 1 robot: object actor index, `refresh_tensors` slicing, dof-state
  reshape (`num_env, total_dofs`), root-state indexing, jacobian body indices, net-contact-force.
- **Verify (critical):** `num_env=4`, print `gym.get_actor_count(env)`, per-actor body counts, dof
  counts, and the object actor index; confirm the sim steps without NaN and object/table indices are
  correct. This is the make-or-break runtime gate.

### T4 — Control both arms + observations
- `TrossenBi.apply_actions`: split action, run each arm's IK/control (reuse `Trossen.apply_actions`).
- Extend `push_with_arm_task.py` obs: `hand_state` → both EE states (or add `hand_state_2`,
  `robot_state_2`); update shapes in the run cfg's `icp_emb`/state net (`ctx_dim`, `dim_out`).
- **Verify:** rollout random 12-DOF actions at num_env=4; both arms move, obs shapes match net cfg,
  no index errors; objects react to both contacts.

### T5 — Bimanual reward
- In `push_with_arm_task.py`, keep the shared goal-pose success/reward; add optional shaping: each arm's
  hand-object potential (mirror `hand_obj_pot_coef`), optional brace/anchor term. Success criterion
  unchanged (object pose vs goal) so it's comparable to single-arm.
- **Verify:** reward is finite, positive on scripted success, at num_env=4.

### T6 — Train (GPU-gated, ~days)
- `PYTORCH_JIT=0 python train_ppo_arm.py +platform=... +env=trossen_bi_icra_base +run=trossen_bi_ours
  ++env.num_env=4096 ...` (NO load_ckpt — from scratch), widened ws_bound, scale DR incl. large objects.
- Reuse the widened-ws_bound + create-time base-pose lessons. Checkpoint periodically; watch
  `cur_suc_rate`.
- **Verify:** loss/return healthy; smoke at num_env=512 first.

### T7 — Eval + compare (apples-to-apples)
- Adapt `valid_ppo_arm_patched.py` for the bimanual ckpt; run the SAME scale sweep + size-invariant
  table as the single-arm frontier. Overlay bimanual vs single-arm success vs bounding-radius.
- **Deliverable E (rigorous):** two-arm curve stays high where single-arm times out — same sim, so a
  controlled result (no cross-sim caveat).

---

## Status / gating
- T1-T2 (+ T3 code scaffolding) are writable now; T3 verify onward is **GPU-gated** (Isaac Gym init
  needs the GPU; currently 100% on another user's job). Build iteratively when it frees.
- Everything on branch `bimanual-dywa`; single-arm `main` untouched. Commit per task; push the branch.
