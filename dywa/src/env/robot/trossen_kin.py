#!/usr/bin/env python3
"""
Forward kinematics, geometric Jacobian, and damped-least-squares IK
for the Trossen WidowX AI (WXAI) 6-DOF arm.

Joint chain (from wxai_base.urdf):
  base_link -> joint_0(z) -> link_1
  link_1    -> joint_1(+y) -> link_2  origin=[0.02,  0,      0.04625]
  link_2    -> joint_2(-y) -> link_3  origin=[-0.264, 0,     0]
  link_3    -> joint_3(-y) -> link_4  origin=[0.245,  0,     0.06]
  link_4    -> joint_4(-z) -> link_5  origin=[0.06775,0,     0.0455]
  link_5    -> joint_5(+x) -> link_6  origin=[0.02895,0,    -0.0455]
  link_6    -> ee_gripper (fixed)     origin=[0.156062,0,    0]
"""

from typing import Optional

import torch as th


# ── URDF geometry constants ──────────────────────────────────────────────────

# Translation from parent frame origin to each joint, shape (6, 3)
_JOINT_ORIGINS = th.tensor([
    [0.0,     0.0,      0.05725],   # joint_0: base_link  → link_1
    [0.02,    0.0,      0.04625],   # joint_1: link_1     → link_2
    [-0.264,  0.0,      0.0],       # joint_2: link_2     → link_3
    [0.245,   0.0,      0.06],      # joint_3: link_3     → link_4
    [0.06775, 0.0,      0.0455],    # joint_4: link_4     → link_5
    [0.02895, 0.0,     -0.0455],    # joint_5: link_5     → link_6
], dtype=th.float32)

# Unit axis of rotation for each joint in its own parent frame, shape (6, 3)
_JOINT_AXES = th.tensor([
    [0.0,  0.0,  1.0],   # joint_0: +z
    [0.0,  1.0,  0.0],   # joint_1: +y
    [0.0, -1.0,  0.0],   # joint_2: -y
    [0.0, -1.0,  0.0],   # joint_3: -y
    [0.0,  0.0, -1.0],   # joint_4: -z
    [1.0,  0.0,  0.0],   # joint_5: +x
], dtype=th.float32)

# Fixed end-effector offset from link_6 origin (ee_gripper joint)
_EE_ORIGIN = th.tensor([0.156062, 0.0, 0.0], dtype=th.float32)

# Joint limits [lower, upper] in rad (arm) / m (gripper), shape (6,)
JOINT_LOWER = th.tensor([-3.0543, 0.0, 0.0, -1.5708, -1.5708, -3.1416],
                         dtype=th.float32)
JOINT_UPPER = th.tensor([3.0543, 3.1416, 2.3562, 1.5708, 1.5708, 3.1416],
                          dtype=th.float32)

# Home configuration: EE at world [0.18, -0.09, 0.51] — ~11cm above the table top
HOME_Q = th.tensor([0.0, 1.3, 1.5, 0.0, 0.5, 0.0], dtype=th.float32)


# ── helpers ──────────────────────────────────────────────────────────────────

def _axis_angle_rotation(axis: th.Tensor, angle: th.Tensor) -> th.Tensor:
    """
    Rodrigues rotation matrix for batched (axis, angle).
    axis:  (..., 3) unit vector
    angle: (...,)
    returns: (..., 3, 3)
    """
    c = th.cos(angle)
    s = th.sin(angle)
    t = 1.0 - c
    ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
    R = th.stack([
        t * ax * ax + c,      t * ax * ay - s * az, t * ax * az + s * ay,
        t * ax * ay + s * az, t * ay * ay + c,      t * ay * az - s * ax,
        t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c,
    ], dim=-1).reshape(*angle.shape, 3, 3)
    return R


def _make_T(R: th.Tensor, t: th.Tensor) -> th.Tensor:
    """Build 4×4 homogeneous transform from (R, t). Batched."""
    N = R.shape[:-2]
    T = th.zeros(*N, 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


# ── forward kinematics ───────────────────────────────────────────────────────

def wxai_fk(q: th.Tensor,
            return_all_frames: bool = False):
    """
    Forward kinematics for the WXAI arm.

    q: (..., 6)  joint angles in radians
    returns:
      T_ee: (..., 4, 4)  end-effector transform in base frame
      (if return_all_frames) list of 7 transforms: [T_j0, ..., T_j5, T_ee]
    """
    device = q.device
    dtype = q.dtype
    batch = q.shape[:-1]

    origins = _JOINT_ORIGINS.to(device=device, dtype=dtype)  # (6, 3)
    axes = _JOINT_AXES.to(device=device, dtype=dtype)        # (6, 3)
    ee_origin = _EE_ORIGIN.to(device=device, dtype=dtype)    # (3,)

    # Broadcast axes/origins to batch: (6, 3) -> (*batch, 6, 3)
    origins = origins.expand(*batch, 6, 3)
    axes = axes.expand(*batch, 6, 3)

    T_accum = th.eye(4, dtype=dtype, device=device).expand(*batch, 4, 4).clone()
    frames = []

    for i in range(6):
        # Translation to joint i origin in current accumulated frame
        t_i = origins[..., i, :]  # (..., 3)
        T_trans = _make_T(
            th.eye(3, dtype=dtype, device=device).expand(*batch, 3, 3),
            t_i
        )
        # Rotation about joint axis by q[..., i]
        R_i = _axis_angle_rotation(axes[..., i, :], q[..., i])
        T_rot = _make_T(R_i, th.zeros(*batch, 3, dtype=dtype, device=device))

        T_accum = T_accum @ T_trans @ T_rot
        frames.append(T_accum.clone())

    # Fixed EE offset
    T_ee_offset = _make_T(
        th.eye(3, dtype=dtype, device=device).expand(*batch, 3, 3),
        ee_origin.expand(*batch, 3)
    )
    T_ee = T_accum @ T_ee_offset
    frames.append(T_ee)

    if return_all_frames:
        return T_ee, frames
    return T_ee


# ── geometric Jacobian ───────────────────────────────────────────────────────

def wxai_jacobian(q: th.Tensor) -> th.Tensor:
    """
    Geometric Jacobian for the WXAI arm at joint configuration q.

    q: (..., 6)
    returns: (..., 6, 6)  [linear (3×6) stacked with angular (3×6)]
    """
    T_ee, frames = wxai_fk(q, return_all_frames=True)
    # frames[0..5] = transform up to and including joint i rotation
    # frames[6]    = T_ee

    p_ee = T_ee[..., :3, 3]  # (..., 3)

    device = q.device
    dtype = q.dtype
    batch = q.shape[:-1]
    axes = _JOINT_AXES.to(device=device, dtype=dtype)

    cols_lin = []
    cols_ang = []
    for i in range(6):
        # Joint axis in world frame: rotate original axis by cumulative rotation
        T_i = frames[i]                            # (..., 4, 4)
        # The axis in world frame is R_i @ axis_i_local, BUT frames[i] already
        # incorporates the rotation by q[i], so the z_i axis is the i-th column
        # of frames[i][:3,:3] @ local_axis. However for the geometric Jacobian
        # we want the axis direction AFTER the joint rotation has been applied.
        # Axes were defined in parent frame before rotation; after rotation by q[i]
        # the joint axis itself does not change direction — it is fixed in parent.
        # So z_i = frames[i-1][:3,:3] @ axis_i (parent frame axis), i.e. we need
        # the frame BEFORE the rotation of joint i (i.e. frames[i-1], or identity
        # for i=0). Use frames[i] rotation minus the last rotation contribution:
        # Simplest: extract the axis as R_{0..i-1} @ local_axis.
        if i == 0:
            R_parent = th.eye(3, dtype=dtype, device=device).expand(*batch, 3, 3)
        else:
            R_parent = frames[i - 1][..., :3, :3]  # rotation of parent frame

        local_axis = axes[i].expand(*batch, 3)  # (*batch, 3)
        z_i = (R_parent @ local_axis.unsqueeze(-1)).squeeze(-1)  # (..., 3)

        p_i = frames[i][..., :3, 3]  # joint origin in world frame after rotation
        # For geometric Jacobian: p_i should be the joint origin in world frame
        # BEFORE the rotation. frames[i] includes the rotation. We want the
        # position of the joint (origin point in world) = frames[i-1] @ t_i.
        # This is frames[i][..., :3, 3] minus the rotated contribution of local rot.
        # Actually frames[i] pos = T_accum_before_i @ t_i_world, which equals the
        # position of the joint center in world frame — correct for Jacobian.

        # Cross product for linear Jacobian: z_i × (p_ee - p_i)
        dp = p_ee - p_i                             # (..., 3)
        J_lin_i = th.linalg.cross(z_i, dp)         # (..., 3)
        cols_lin.append(J_lin_i)
        cols_ang.append(z_i)

    J_lin = th.stack(cols_lin, dim=-1)   # (..., 3, 6)
    J_ang = th.stack(cols_ang, dim=-1)   # (..., 3, 6)
    J = th.cat([J_lin, J_ang], dim=-2)   # (..., 6, 6)
    return J


# ── inverse kinematics ───────────────────────────────────────────────────────

def _rotation_error(R_target: th.Tensor, R_current: th.Tensor) -> th.Tensor:
    """Axis-angle error from current to target rotation. (..., 3)"""
    R_err = R_target @ R_current.transpose(-1, -2)
    # Extract axis-angle from rotation matrix
    trace = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    angle = th.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))  # (...,)
    # axis from skew-symmetric part; handle small-angle case
    s = th.sin(angle).unsqueeze(-1).clamp(min=1e-8)
    axis = th.stack([
        R_err[..., 2, 1] - R_err[..., 1, 2],
        R_err[..., 0, 2] - R_err[..., 2, 0],
        R_err[..., 1, 0] - R_err[..., 0, 1],
    ], dim=-1) / (2.0 * s)
    return axis * angle.unsqueeze(-1)


def wxai_ik(T_target: th.Tensor,
            q_init: Optional[th.Tensor] = None,
            n_iter: int = 30,
            damping: float = 0.05) -> th.Tensor:
    """
    Batched damped-least-squares IK for the WXAI arm.

    T_target: (..., 4, 4)  desired EE transform in base frame
    q_init:   (..., 6)     initial joint configuration (defaults to HOME_Q)
    n_iter:   int          number of IK iterations
    damping:  float        λ for damped least-squares (J J^T + λ²I)^{-1}

    returns: (..., 6) joint angles
    """
    device = T_target.device
    dtype = T_target.dtype
    batch = T_target.shape[:-2]

    q_lo = JOINT_LOWER.to(device=device, dtype=dtype)
    q_hi = JOINT_UPPER.to(device=device, dtype=dtype)

    if q_init is None:
        q = HOME_Q.to(device=device, dtype=dtype).expand(*batch, 6).clone()
    else:
        q = q_init.clone()

    lam2 = damping ** 2
    I6 = lam2 * th.eye(6, dtype=dtype, device=device)

    for _ in range(n_iter):
        T_cur = wxai_fk(q)                                     # (..., 4, 4)

        pos_err = T_target[..., :3, 3] - T_cur[..., :3, 3]    # (..., 3)
        ori_err = _rotation_error(T_target[..., :3, :3],
                                  T_cur[..., :3, :3])          # (..., 3)
        err = th.cat([pos_err, ori_err], dim=-1).unsqueeze(-1)  # (..., 6, 1)

        J = wxai_jacobian(q)                                   # (..., 6, 6)
        JJT = J @ J.transpose(-1, -2) + I6                    # (..., 6, 6)
        dq = (J.transpose(-1, -2) @
              th.linalg.solve(JJT, err)).squeeze(-1)           # (..., 6)

        q = (q + dq).clamp(q_lo, q_hi)

    return q
