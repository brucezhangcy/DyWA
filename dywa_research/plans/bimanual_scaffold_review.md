# Read-only review: `bimanual-dywa` scaffold vs `plans/bimanual_build_plan.md`

Session 4 (cloud, 2026-10-17). Reviewed `origin/bimanual-dywa` @ c6f8872 (`dywa/src/env/robot/trossen_bi.py`, 107 lines)
against the plan on that branch and against `main` @ f9beecc (`trossen.py`, `franka_util.py`). **Nothing was run.**
This review does not show that the scaffold works. It lists defects you can find by reading the code, which
will stop it from working.

## Plan coverage
| Plan task | State on branch |
|---|---|
| T1 dual-arm wrapper | `TrossenBi` class exists. Its prerequisite edits to `Trossen` (`dof_offset`, base-pose override) are **not made**, so T1 is incomplete |
| T2 register robot + cfg | not started (there is no `which_robot='trossen_bi'` branch in `push_env.py` and no yaml) |
| T3 second actor + index audit | not started |
| T4 obs/action | partly written inside the wrapper (12-D split, stacked `hand_state`) |
| T5 reward, T6 train, T7 eval | not started |

## Defects found by reading the code (each one blocks a correct 2-arm sim)
1. **Both arms resolve to the same actor.** `Trossen.create_actors` names every arm actor `'robot'`
   (`trossen.py:222-223`), and `Trossen.setup` looks the actor up by that name with
   `find_actor_index(env, 'robot')` (`trossen.py:261-263`, `franka_util.py:555-558`). Both sub-arms
   therefore get the first arm's indices, EE body, base body and Jacobian. Fix: give `Trossen.Config` an
   `actor_name` (default `'robot'`) and use it in both places.
2. **Both arms are welded at the same pose.** The base pose is baked from hard-coded constants
   (`trossen.py:206-221`: x = -0.5, y = 0). `TrossenBi.Config.arm1_base_xy/arm1_base_yaw` are never read.
   On top of that, `c1 = cfg.arm` is the *same* config object as `c0`, so a per-arm override would leak to
   both arms. Fix: add a `base_xy/base_yaw` override to `Trossen.Config` (default None means the current
   behaviour) and give each arm a `copy.deepcopy` of the config.
3. **DOF slicing is hard-coded to the first robot.** `env.tensors['dof'][:, :N_ARM]` appears at
   `trossen.py:382,455,463`, and the `dof_offset` attributes that `TrossenBi` sets are never read. Arm 1 would
   read and overwrite arm 0's joints.
4. **The DOF-target write has the wrong shape for two robots.** Each arm owns a `(num_env, n_dofs)`
   `_control` buffer (`trossen.py:311-312`) and passes it to `set_dof_position_target_tensor_indexed`
   (`trossen.py:465-473`). With two robots per env, the simulator's DOF-target tensor is
   `(num_env, 2*n_dofs)`. The two arms need one shared buffer, each writing its own slice, followed by a
   single indexed call covering both actor indices.
5. **The workspace clip uses the root tensor.** `arm_base = env.tensors['root'][self.indices, :3]`
   (`trossen.py:418`), and `Trossen.reset` *writes* the table-derived single-arm x into that root tensor
   (`trossen.py:364-376`). The code's own comment says this write does not move a welded actor. For a
   second arm at +x, the clip box would be centred on the wrong base unless `reset` uses the per-arm
   override. (The same mechanism matters for single-arm runs with a non-default `table_dims`; see
   CRITIQUES #14.)
6. **The config is not a `ConfigBase`.** `RobotBase` (`env/robot/base.py`) has no `Config`, so
   `TrossenBi.Config` falls back to a plain dataclass over `object`. Every other robot config subclasses
   `util.config.ConfigBase`, for example `Trossen.Config(ConfigBase)` at `trossen.py:52-53`. The hydra and
   `recursive_replace_map` plumbing will probably not treat it like the other robot configs. Also,
   `Tuple[float, float] = (None, None)` gives `None` values to a field typed as `float`. Fix: subclass
   `ConfigBase` and use `Optional[...]`.
7. **The second arm's base position is not mirrored.** arm 1 is at x = +0.40, but the mirror of arm 0
   (−0.5) is +0.5. With a reach of 0.768 m (FK, `analysis/s4_geometry_audit.py`) both positions cover the
   table. The asymmetry should be deliberate or fixed.
8. **The net dims are not updated.** `trossen_icra_ours_abs_rel.yaml` has `ctx_dim: 36 = 9 + 6 + 16 + 5`.
   Two arms need `9 + 12 + 32 + 5 = 58` and `dim_out = 2048 + 58 = 2106`, assuming the obs simply stacks both
   arms (plan T4).

## Recommended order on coldbrew (each step ends by RUNNING at `num_env=4`)
1. Make the three small additive `Trossen` changes with defaults that preserve single-arm behaviour:
   `actor_name`, a base-pose override, and `dof_offset` with a shared control buffer. Then rerun the
   single-arm anchor (`anchor_eval.sh`, expected ≈ 94.5%) to show nothing regressed.
2. Do T2 and T3, then print the actor, body and DOF counts and the object actor index per env (the plan's
   T3 gate).
3. Only then attempt T4–T7.

The mjlab residual-policy route (`gpu_bimanual_residual.sh`) is the faster way to get a trained-bimanual
number. It stays cross-simulator.
