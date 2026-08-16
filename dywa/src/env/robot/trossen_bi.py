"""
TrossenBi — two Trossen WXAI arms for bimanual non-prehensile push in DyWA (Isaac Gym).

SCAFFOLD (branch `bimanual-dywa`). Written WITHOUT runtime testing — Isaac Gym tensor indexing must be
verified on a GPU box. Composes two `Trossen` sub-robots; the single-arm `Trossen` path is unchanged.

Runtime-critical items to verify/finish on GPU (see dywa_research/plans/bimanual_build_plan.md):
  * DOF offset: `Trossen.apply_actions`/`step_controller` slice `env.tensors['dof'][:, :N_ARM]`,
    hardcoded to the FIRST robot's dofs. Each arm must slice its OWN dofs via `dof_offset`
    (arm0=0, arm1=n_dofs). Requires the small `Trossen` change below (`self.dof_offset`, default 0).
  * Object actor index: the scene's object is created AFTER the robot(s); with 2 robots its global
    index shifts. Audit `scene.cur_ids` / any hardcoded actor offsets.
  * `env.tensors['dof']` per-env shape becomes (num_env, 2*n_dofs, 2); confirm base.py reshape.
  * Base poses must bracket the table within reach; sweep separation.
"""
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import torch as th
from gym import spaces

from env.robot.base import RobotBase
from env.robot.trossen import Trossen, N_ARM


class TrossenBi(RobotBase):
    """Two Trossen arms placed on opposite table edges. Action = concat(arm0_6d, arm1_6d) = 12-D."""

    @dataclass
    class Config(RobotBase.Config if hasattr(RobotBase, "Config") else object):
        # Base config shared by both arms; per-arm base placement is set below.
        arm: Trossen.Config = field(default_factory=Trossen.Config)
        # Base placement overrides (world frame). arm0 = -x edge (single-arm default), arm1 = +x edge.
        arm0_base_xy: Tuple[float, float] = (None, None)   # None -> Trossen's default (-x edge)
        arm1_base_xy: Tuple[float, float] = (0.40, 0.0)    # +x side; TUNE for reach/separation
        arm1_base_yaw: float = np.pi                        # face the table (rotate 180°)

    def __init__(self, cfg: "TrossenBi.Config"):
        super().__init__()
        self.cfg = cfg
        # Two sub-robots. arm0 keeps default placement; arm1 gets an override + yaw.
        c0 = cfg.arm
        c1 = cfg.arm  # NOTE: if Trossen.Config is mutable/shared, deep-copy before overriding.
        self.arm0 = Trossen(c0)
        self.arm1 = Trossen(c1)
        # 12-D action space = two 6-D arms.
        lo = np.concatenate([self.arm0.action_space.low, self.arm0.action_space.low]) \
            if hasattr(self.arm0, "action_space") else None
        hi = np.concatenate([self.arm0.action_space.high, self.arm0.action_space.high]) \
            if hasattr(self.arm0, "action_space") else None
        # action_space is finalized after create_assets (per-arm spaces exist by then); see create_assets.
        self._lo, self._hi = lo, hi

    # ---- asset / actor lifecycle (delegate to both) ----
    def create_assets(self, gym, sim, *a, **k):
        a0 = self.arm0.create_assets(gym, sim, *a, **k)
        a1 = self.arm1.create_assets(gym, sim, *a, **k)
        # After assets, per-arm n_dofs is known → set dof offsets and 12-D action space.
        self.arm0.dof_offset = 0                       # requires Trossen.dof_offset (default 0)
        self.arm1.dof_offset = self.arm0.n_dofs        # 2nd arm's dofs follow the 1st's
        self.n_dofs = self.arm0.n_dofs + self.arm1.n_dofs
        self.action_space = spaces.Box(
            np.concatenate([self.arm0.action_space.low, self.arm1.action_space.low]),
            np.concatenate([self.arm0.action_space.high, self.arm1.action_space.high]),
        )
        return {"arm0": a0, "arm1": a1}

    def create_actors(self, gym, sim, env, env_id: int) -> Dict:
        # Place arm0 (default -x edge) and arm1 (+x edge via base override).
        # TODO(GPU): Trossen.create_actors currently bakes base_x internally; add a base-pose override
        # hook (Config.base_x_override/base_y_override/base_yaw, default None = current behavior).
        r0 = self.arm0.create_actors(gym, sim, env, env_id)
        r1 = self.arm1.create_actors(gym, sim, env, env_id)   # must read arm1 base override
        return {"robot": r0.get("robot"), "robot2": r1.get("robot")}

    def setup(self, env):
        # Each sub-arm finds its own EE body (DOMAIN_SIM via its own handles) → OK for two instances.
        self.arm0.setup(env)
        self.arm1.setup(env)

    def reset(self, gym, sim, env, env_id) -> Tuple:
        self.arm0.reset(gym, sim, env, env_id)
        self.arm1.reset(gym, sim, env, env_id)
        return env_id, None, None

    # ---- control ----
    def apply_actions(self, gym, sim, env, actions, done=None):
        if actions is None:
            return
        a0, a1 = actions[:, :N_ARM_ACT], actions[:, N_ARM_ACT:]   # 6 + 6
        self.arm0.apply_actions(gym, sim, env, a0, done=done)
        self.arm1.apply_actions(gym, sim, env, a1, done=done)

    def step_controller(self, gym, sim, env):
        # Both use set_dof_position_target_tensor_indexed (actor-scoped) → no clobber IF dof_offset set.
        self.arm0.step_controller(gym, sim, env)
        self.arm1.step_controller(gym, sim, env)

    # ---- observations ----
    @property
    def hand_state(self):
        # Stack both EE states so the task/obs sees both contacts.
        return th.cat([self.arm0.ee_state, self.arm1.ee_state], dim=-1)


# Action width per arm (Δpos3 + Δori3). Kept explicit for the split above.
N_ARM_ACT = 6
