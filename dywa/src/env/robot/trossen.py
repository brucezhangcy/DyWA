#!/usr/bin/env python3
"""
Trossen WidowX AI (WXAI) robot class for IsaacGym.

Mirrors env/robot/franka.py structure. Supports cpos_n control mode:
relative Cartesian position + orientation deltas → numerical IK (PyTorch) →
joint torques via PD controller (effort mode).

DOF layout (from wxai_base.urdf, after fixed-joint handling):
  [0] joint_0              revolute arm  ±3.054 rad
  [1] joint_1              revolute arm  [0, π]
  [2] joint_2              revolute arm  [0, 2.356]
  [3] joint_3              revolute arm  ±π/2
  [4] joint_4              revolute arm  ±π/2
  [5] joint_5              revolute arm  ±π
  [6] right_carriage_joint prismatic     [0, 0.044] m  (driven same as left)
  [7] left_carriage_joint  prismatic     [0, 0.044] m  (gripper control)
"""

import pkg_resources
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import einops
import numpy as np
import nvtx
import torch as th
from gym import spaces
from isaacgym import gymapi, gymtorch

from env.robot.base import RobotBase
from env.robot.franka_util import find_actor_handles, find_actor_indices
from env.robot.trossen_kin import (
    HOME_Q, JOINT_LOWER, JOINT_UPPER, wxai_fk, wxai_ik,
)
from util.config import ConfigBase
from util.math_util import (
    matrix_from_quaternion,
    quat_from_axa,
    quat_multiply,
)
from util.torch_util import dcn

N_ARM = 6       # revolute arm joints
N_GRIPPER = 2   # right_carriage + left_carriage prismatic joints
# DOF indices match URDF order: joint_0..5, right_carriage, left_carriage
IDX_GRIPPER_RIGHT = 6
IDX_GRIPPER_LEFT = 7


class Trossen(RobotBase):

    @dataclass
    class Config(ConfigBase):
        asset_root: str = pkg_resources.resource_filename('data', 'assets')
        robot_file: str = 'trossen_arm/urdf/wxai_base.urdf'

        init_type: str = 'home'

        # Control mode: cpos_n (Cartesian position, numerical IK)
        ctrl_mode: str = 'cpos_n'
        target_type: str = 'rel'
        rot_type: str = 'axis_angle'

        # Cartesian action limits
        max_pos: float = 0.06    # m per step
        max_ori: float = 0.1     # rad per step
        lock_orn: bool = False

        # PD gains (joint space, effort mode)
        KP_joint: float = 300.0
        KD_joint: float = 20.0

        # For compatibility with push_with_arm_task.py energy tracking
        regularize: Optional[str] = None

        # IK parameters
        # n_iter=5 converges to sub-mm/sub-0.1° for the ±6cm/±0.1rad per-step
        # deltas the control loop actually sees; iterations 6-30 were waste.
        # See dev_log 2026-05-19 "How to speed up Trossen training".
        ik_n_iter: int = 5
        ik_damping: float = 0.05

        # Workspace clipping
        clip_bound: bool = True
        track_object: bool = True
        obj_margin: float = 0.15
        accumulate: bool = False

        # Workspace bounds [min, max] for x, y, z in base frame
        ws_bound: Optional[List[List[float]]] = field(default_factory=lambda: [
            [-0.25, -0.35, 0.35],
            [+0.25, +0.35, 0.75],
        ])

        # Robot base placement
        base_height: str = 'ground'
        keepout_radius: float = 0.3

        # Gripper position when "open"
        gripper_open: float = 0.04    # m

        # Friction
        default_hand_friction: float = 1.5
        default_body_friction: float = 0.1
        restitution: float = 0.5

        # Physics
        VISCOUS_FRICTION: float = 0.2
        use_effort: bool = True

        # Feature flags (for compatibility with existing task code)
        add_tip_sensor: bool = False
        disable_table_collision: bool = False

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self, cfg: 'Trossen.Config'):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = Trossen.Config(**cfg)
        self.cfg = cfg
        # These are populated in setup()
        self.num_env: int = 0
        self.device: th.device = th.device('cpu')
        self._first: bool = True
        self.env = None  # set in setup()

    # ── RobotBase interface ──────────────────────────────────────────────────

    def create_assets(self, gym, sim,
                      counts: Optional[Dict[str, int]] = None):
        cfg = self.cfg
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.enable_gyroscopic_forces = True
        asset_options.vhacd_enabled = False
        asset_options.convex_decomposition_from_submeshes = True

        robot_asset = gym.load_urdf(
            sim, cfg.asset_root, cfg.robot_file, asset_options)

        # Friction: high on fingers, low on arm body
        robot_props = gym.get_asset_rigid_shape_properties(robot_asset)
        finger_body_names = ['carriage_left', 'carriage_right',
                             'gripper_left', 'gripper_right']
        shape_indices = gym.get_asset_rigid_body_shape_indices(robot_asset)
        finger_shape_idx = []
        for name in finger_body_names:
            h = gym.find_asset_rigid_body_index(robot_asset, name)
            if h >= 0:
                si = shape_indices[h]
                finger_shape_idx.extend(range(si.start, si.start + si.count))
        self.__finger_shape_indices = finger_shape_idx

        for i, p in enumerate(robot_props):
            p.friction = (cfg.default_hand_friction
                          if i in finger_shape_idx
                          else cfg.default_body_friction)
            p.restitution = cfg.restitution
        gym.set_asset_rigid_shape_properties(robot_asset, robot_props)

        # DOF limits / effort limits
        self.n_bodies = gym.get_asset_rigid_body_count(robot_asset)
        self.n_dofs = gym.get_asset_dof_count(robot_asset)
        dof_props = gym.get_asset_dof_properties(robot_asset)
        dof_lo, dof_hi, eff_hi = [], [], []
        for i in range(self.n_dofs):
            dof_lo.append(dof_props['lower'][i])
            dof_hi.append(dof_props['upper'][i])
            eff_hi.append(dof_props['effort'][i])
        self.dof_limits = (np.asarray(dof_lo), np.asarray(dof_hi))
        self.eff_limits = np.asarray(eff_hi)

        # 6D Cartesian action space (Δpos + Δori axis-angle)
        if not cfg.lock_orn:
            lo = [-cfg.max_pos] * 3 + [-cfg.max_ori] * 3
            hi = [+cfg.max_pos] * 3 + [+cfg.max_ori] * 3
        else:
            lo = [-cfg.max_pos] * 3
            hi = [+cfg.max_pos] * 3
        self.action_space = spaces.Box(np.asarray(lo), np.asarray(hi))

        # Store asset body name → index dict for setup()
        self.link_dict = gym.get_asset_rigid_body_dict(robot_asset)

        if counts is not None:
            counts['body'] = gym.get_asset_rigid_body_count(robot_asset)
            counts['shape'] = gym.get_asset_rigid_shape_count(robot_asset)

        self.assets = {'robot': robot_asset}
        return dict(self.assets)

    def create_actors(self, gym, sim, env, env_id: int):
        cfg = self.cfg
        # `fix_base_link=True` welds the actor at the pose passed to create_actor.
        # The reset-time root tensor write does NOT move a welded actor in this
        # code path (empirically — the rendered position contradicted the written
        # value). So compute placement now and bake it into the Transform.
        # Hardcoded table values match tabletop_scene.py defaults
        # (table_pos=(0,0,0.2), table_dims=(0.4,1.0,0.4)); `env` here is the raw
        # Isaac Gym Env handle, not the EnvBase wrapper, so env.scene isn't
        # accessible.
        table_pos_x, table_pos_y = 0.0, 0.0
        table_dims_x, table_dims_z = 0.4, 0.4
        if cfg.base_height == 'origin':
            base_x, base_z = 0.0, 0.0
        else:
            base_x = table_pos_x - 0.5 * table_dims_x - cfg.keepout_radius
            if cfg.base_height == 'ground':
                base_z = 0.0
            elif cfg.base_height == 'table':
                base_z = table_dims_z
            else:
                base_z = float(cfg.base_height)
        initial_pose = gymapi.Transform()
        initial_pose.p.x = base_x
        initial_pose.p.y = table_pos_y
        initial_pose.p.z = base_z
        robot = gym.create_actor(
            env, self.assets['robot'], initial_pose, 'robot', env_id, 0b0100)

        dof_props = gym.get_asset_dof_properties(self.assets['robot'])
        for i in range(self.n_dofs):
            if i < N_ARM:
                dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][i] = cfg.KP_joint
                dof_props['damping'][i] = cfg.KD_joint
                dof_props['friction'][i] = 0.0
                dof_props['armature'][i] = 0.01
            else:
                dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][i] = 1e3
                dof_props['damping'][i] = 1e2
                dof_props['friction'][i] = 1e3
                dof_props['armature'][i] = 1e2
        gym.set_actor_dof_properties(env, robot, dof_props)

        # Color arm orange so it contrasts against the white table and grey floor
        n_bodies = gym.get_actor_rigid_body_count(env, robot)
        orange = gymapi.Vec3(0.85, 0.40, 0.05)
        for b in range(n_bodies):
            gym.set_rigid_body_color(env, robot, b, gymapi.MESH_VISUAL, orange)

        return {'robot': robot}

    def setup(self, env):
        """Called once after all actors are created (EnvBase.setup() phase)."""
        cfg = self.cfg
        gym = env.gym
        sim = env.sim
        self.env = env

        self.num_env = env.cfg.num_env
        self.device = th.device(env.cfg.th_device)
        self.robot_radius: float = 0.12

        # Actor handles and global sim indices
        self.handles = find_actor_handles(gym, env.envs, 'robot')
        self.indices = th.as_tensor(
            find_actor_indices(gym, env.envs, 'robot'),
            dtype=th.int32, device=self.device)

        # Body indices in DOMAIN_SIM for EE and base
        ee_body_indices = []
        tip_body_indices = []
        base_body_indices = []
        for i in range(self.num_env):
            ee_idx = gym.find_actor_rigid_body_index(
                env.envs[i], self.handles[i], 'ee_gripper_link', gymapi.DOMAIN_SIM)
            ee_body_indices.append(ee_idx)
            tip_body_indices.append(ee_idx)

            base_idx = gym.find_actor_rigid_body_index(
                env.envs[i], self.handles[i], 'base_link', gymapi.DOMAIN_SIM)
            base_body_indices.append(base_idx)

        self.hand_ids = th.as_tensor(ee_body_indices, dtype=th.long, device=self.device)
        # Aliases used by push_with_arm_task.py and arm_env.py
        self.ee_body_indices = self.hand_ids
        self.tip_body_indices = self.hand_ids
        self.base_body_indices = th.as_tensor(
            base_body_indices, dtype=th.int32, device=self.device)

        # link_body_indices for rendering wrappers (DOMAIN_ENV)
        _link_names = ['base_link', 'link_1', 'link_2', 'link_3',
                        'link_4', 'link_5', 'link_6', 'ee_gripper_link']
        self.link_body_indices = [
            gym.find_actor_rigid_body_index(
                env.envs[0], self.handles[0], name, gymapi.DOMAIN_ENV)
            for name in _link_names
        ]

        # Jacobian tensor: (N, n_bodies, 6, n_dofs)
        self._jacobian = gymtorch.wrap_tensor(
            gym.acquire_jacobian_tensor(sim, 'robot'))

        # Mass matrix tensor
        _mm = gym.acquire_mass_matrix_tensor(sim, 'robot')
        self.mm = gymtorch.wrap_tensor(_mm)

        # EE body asset index (0-based within actor, NOT DOMAIN_SIM)
        ee_asset_idx = self.link_dict['ee_gripper_link']
        # j_eef: (N, 6, 6) — Jacobian at EE, first 6 arm DOFs only
        self.j_eef = self._jacobian[:, ee_asset_idx - 1, :, :N_ARM]
        # Arm mass matrix slice
        self.mm = self.mm[:, :ee_asset_idx, :ee_asset_idx]

        # Control buffer (N, n_dofs)
        self._control = th.zeros(self.num_env, self.n_dofs,
                                 dtype=th.float, device=self.device)

        # Current hand friction per env (used by AddPhysParams obs wrapper —
        # parallels Franka.cur_hand_friction; Trossen doesn't randomize it
        # but the env wrapper still reads this attribute).
        self.cur_hand_friction = th.full(
            (self.num_env,), cfg.default_hand_friction,
            dtype=th.float, device=self.device)

        # Joint limits
        self.q_lo = JOINT_LOWER.to(device=self.device)
        self.q_hi = JOINT_UPPER.to(device=self.device)

        # Home config and initial targets
        self.q_home = HOME_Q.to(device=self.device)
        self._q_target = self.q_home.expand(self.num_env, N_ARM).clone()

        # Workspace bounds
        if cfg.ws_bound is not None:
            self.ws_lo = th.as_tensor(cfg.ws_bound[0], dtype=th.float, device=self.device)
            self.ws_hi = th.as_tensor(cfg.ws_bound[1], dtype=th.float, device=self.device)
        else:
            self.ws_lo = self.ws_hi = None

        # Accumulated EE target (7D pos+quat, x,y,z,qx,qy,qz,qw)
        self._ee_target = th.zeros(self.num_env, 7, dtype=th.float, device=self.device)
        with th.no_grad():
            T0 = wxai_fk(self.q_home.unsqueeze(0))
        self._ee_target[:, :3] = T0[0, :3, 3]
        self._ee_target[:, 3:7] = _mat_to_quat(T0[0, :3, :3])

        # Energy tracking (for compatibility with tasks that check regularize)
        if cfg.regularize is not None:
            self.energy = th.zeros(self.num_env, dtype=th.float, device=self.device)

        self._first = True

    def reset(self, gym, sim, env, env_id) -> Tuple:
        cfg = self.cfg
        if env_id is None:
            env_id = th.arange(self.num_env, dtype=th.int32, device=self.device)
        I = env_id.long()
        indices = self.indices[I]

        # First reset: place robot base. NOTE: empirically a no-op for fix_base_link
        # actors in this code path — the actual welded pose is set at create_actors
        # time via gymapi.Transform(). Kept here for parity with Franka and as a
        # defensive fallback covering all base_height modes (including 'origin').
        if self._first:
            iii = indices.long()
            root = env.tensors['root']
            if cfg.base_height == 'origin':
                root[iii, 0] = 0.0
                root[iii, 2] = 0.0
            else:
                root[iii, 0] = (env.scene.table_pos[..., 0]
                                - 0.5 * env.scene.table_dims[..., 0]
                                - cfg.keepout_radius)
                if cfg.base_height == 'ground':
                    root[iii, 2] = 0.0
                elif cfg.base_height == 'table':
                    root[iii, 2] = env.scene.table_dims[..., 2]
                else:
                    root[iii, 2] = float(cfg.base_height)
            root[iii, 6] = 1.0  # unit quaternion w=1
            self._first = False

        # Set DOF state
        dof = env.tensors['dof']
        dof[I, :N_ARM, 0] = self.q_home
        dof[I, IDX_GRIPPER_RIGHT, 0] = cfg.gripper_open
        dof[I, IDX_GRIPPER_LEFT, 0] = cfg.gripper_open
        dof[I, :, 1] = 0.0  # zero velocity

        # Reset control state
        self._q_target[I] = self.q_home
        self._control[I] = 0.0
        with th.no_grad():
            T0 = wxai_fk(self.q_home.unsqueeze(0))
        self._ee_target[I, :3] = T0[0, :3, 3]
        self._ee_target[I, 3:7] = _mat_to_quat(T0[0, :3, :3])

        if cfg.regularize is not None:
            self.energy[I] = 0.0

        return indices, None, None

    @nvtx.annotate("Trossen.apply_actions")
    def apply_actions(self, gym, sim, env, actions, done=None):
        """
        actions: (N, 6)  Δpos (3) + Δori axis-angle (3) in world frame.
        """
        if actions is None:
            return
        cfg = self.cfg
        N = self.num_env

        dpos = actions[:, :3].clamp(-cfg.max_pos, cfg.max_pos)
        daxa = (actions[:, 3:6].clamp(-cfg.max_ori, cfg.max_ori)
                if not cfg.lock_orn
                else th.zeros(N, 3, dtype=actions.dtype, device=self.device))

        # Current EE state (world frame) and arm base (world frame)
        ee_world = self.ee_state[..., :3]
        ee_quat = self.ee_state[..., 3:7]
        arm_base = env.tensors['root'][self.indices.long(), :3]

        # Target position: current EE + action delta, in world frame
        pos_world = ee_world + dpos

        # Workspace clipping (world frame)
        if cfg.clip_bound and self.ws_lo is not None:
            ws_lo_world = self.ws_lo + arm_base
            ws_hi_world = self.ws_hi + arm_base
            if cfg.track_object and hasattr(env, 'scene') and hasattr(env.scene, 'cur_ids'):
                obj_ids = env.scene.cur_ids.long()
                obj_pos = env.tensors['root'][obj_ids, :3]
                obj_rad = env.scene.cur_radii
                ws_lo_world = (obj_pos - obj_rad.unsqueeze(-1) - cfg.obj_margin).clamp(min=ws_lo_world)
                ws_hi_world = (obj_pos + obj_rad.unsqueeze(-1) + cfg.obj_margin).clamp(max=ws_hi_world)
            pos_world = pos_world.clamp(ws_lo_world, ws_hi_world)

        pos_world = pos_world.clamp(ee_world - cfg.max_pos, ee_world + cfg.max_pos)

        # Orientation: apply delta to current EE quat (world frame)
        dq = quat_from_axa(daxa)
        new_quat = quat_multiply(dq, ee_quat)

        self._ee_target[:, :3] = pos_world
        self._ee_target[:, 3:7] = new_quat

        # Convert target position to arm base frame for IK
        pos_base = pos_world - arm_base

        # Build 4×4 target transform for IK (base frame)
        R_target = matrix_from_quaternion(new_quat)
        T_target = (th.eye(4, dtype=th.float, device=self.device)
                    .unsqueeze(0).expand(N, 4, 4).clone())
        T_target[:, :3, :3] = R_target
        T_target[:, :3, 3] = pos_base

        # Numerical IK from current joint positions
        q_cur = env.tensors['dof'][:, :N_ARM, 0]
        self._q_target[:] = wxai_ik(
            T_target, q_init=q_cur,
            n_iter=cfg.ik_n_iter, damping=cfg.ik_damping)

    @nvtx.annotate("Trossen.step_controller")
    def step_controller(self, gym, sim, env):
        """Set position targets — PhysX's built-in PD does the rest."""
        q_cur = env.tensors['dof'][:, :N_ARM, 0]
        q_tgt = th.where(th.isnan(self._q_target), q_cur, self._q_target)
        self._control[:, :N_ARM] = q_tgt
        self._control[:, IDX_GRIPPER_RIGHT] = self.cfg.gripper_open
        self._control[:, IDX_GRIPPER_LEFT] = self.cfg.gripper_open

        gym.set_dof_position_target_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(self._control),
            gymtorch.unwrap_tensor(self.indices),
            len(self.indices),
        )

    @property
    def ee_state(self) -> th.Tensor:
        """EE body state (N, 13): pos + quat + lin_vel + ang_vel."""
        return self.env.tensors['body'][self.hand_ids]


# ── helpers ──────────────────────────────────────────────────────────────────

def _mat_to_quat(R: th.Tensor) -> th.Tensor:
    """Rotation matrix (..., 3, 3) → quaternion (..., 4) in (x, y, z, w) convention."""
    batch = R.shape[:-2]
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q = th.empty(*batch, 4, dtype=R.dtype, device=R.device)

    s = th.sqrt((trace + 1.0).clamp(min=1e-10)) * 2.0  # 4w
    q[..., 3] = 0.25 * s
    q[..., 0] = (R[..., 2, 1] - R[..., 1, 2]) / s.clamp(min=1e-8)
    q[..., 1] = (R[..., 0, 2] - R[..., 2, 0]) / s.clamp(min=1e-8)
    q[..., 2] = (R[..., 1, 0] - R[..., 0, 1]) / s.clamp(min=1e-8)

    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return q
