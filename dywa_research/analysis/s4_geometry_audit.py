#!/usr/bin/env python3
"""Session 4 (CPU-only) geometry audit of the size-invariant-table experiment (Deliverable D / CRITIQUES #0).

Pure geometry from constants read in the source (no simulation). Code facts used (read 2026-10-17 @ main f9beecc):
  * valid mode spawn xy: scene.py:1365-1377 builds _valid_poses with _get_xy(prevent_fall=False)
    -> uniform in +-0.5*scene.margin_scale(0.95, scene.py:235)*table_dims  (NO radius contraction)
  * goal xy: push_task.py:684-692 goal_lo/hi = table_pos +- 0.5*task.margin_scale*table_dims;
    task.margin_scale = 0.8 (icra_base.yaml task block); min separation = goal_radius*1.1 (arm_div_base.yaml)
  * OOB: push_task.py:325-329 uses ws_lo/ws_hi = table +- (0.5*table_dims + ws_margin 0.3) in xy, z >= table centre
  * robot base: trossen.py:206-221 bakes base x = 0 - 0.5*0.4 - keepout 0.3 = -0.50 into the create-time Transform
    from HARDCODED table dims 0.4 -- it does NOT follow ++env.single_object_scene.table_dims
  * EE software clip: trossen.py:424-433 clamps EE target to ws_bound + arm_base; ws_bound x-max 0.6 -> world x <= +0.10
    (dev_log.md 2026-05-19 entry: "EE world x in [-0.75, +0.1]")
  * physical reach: FK of trossen_kin.py constants (ported below), sampled over joint limits.
Regimes: fixed table [0.4,0.5]; size-inv v1 as run (sizeinv_table.sh: table=(base_half+s)/0.475);
         size-inv v2 PROPOSED (v1 table + margin_scales shrunk so spawn & goal regions stay at the s=0.10 size).
Output: results/session4/geometry_audit.csv (+ printed table).
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "results", "session4")); os.makedirs(OUT, exist_ok=True)

# ---- WXAI FK (numpy port of trossen_kin.py:22-50 constants + wxai_fk chain)
ORG = np.array([[0, 0, 0.05725], [0.02, 0, 0.04625], [-0.264, 0, 0], [0.245, 0, 0.06],
                [0.06775, 0, 0.0455], [0.02895, 0, -0.0455]])
AX = np.array([[0, 0, 1], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
EE = np.array([0.156062, 0, 0])
LO = np.array([-3.0543, 0.0, 0.0, -1.5708, -1.5708, -3.1416]); HI = np.array([3.0543, 3.1416, 2.3562, 1.5708, 1.5708, 3.1416])


def rot(axis, ang):
    ax, ay, az = axis; c, s = np.cos(ang), np.sin(ang); t = 1 - c
    return np.stack([t*ax*ax+c, t*ax*ay-s*az, t*ax*az+s*ay, t*ax*ay+s*az, t*ay*ay+c, t*ay*az-s*ax,
                     t*ax*az-s*ay, t*ay*az+s*ax, t*az*az+c], -1).reshape(*ang.shape, 3, 3)


def fk(q):
    R = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy(); p = np.zeros((len(q), 3))
    for i in range(6):
        p = p + R @ ORG[i]; R = R @ rot(AX[i], q[:, i])
    return p + R @ EE


assert np.allclose(fk(np.array([[0.0, 1.3, 1.5, 0.0, 0.5, 0.0]]))[0], [0.4031, -0.0887, 0.5111], atol=2e-3)
rng = np.random.default_rng(0)
P = fk(LO + (HI - LO) * rng.random((1_000_000, 6)))
low = (P[:, 2] >= 0.0) & (P[:, 2] <= 0.15)        # EE within 15 cm above the table top (base_z = table top)
REACH = float(np.hypot(P[low, 0], P[low, 1]).max())
BASE = np.array([-0.5, 0.0]); CLIP_X = 0.10; FR = 0.10  # force_rad=0.10 -> goal_radius; min sep = 1.1*0.10


def regime_dims(s, regime):
    if regime == "fixed":
        return (0.4, 0.5), 0.95, 0.8
    tx, ty = round((0.09 + s) / 0.475, 3), round((0.1375 + s) / 0.475, 3)  # sizeinv_table.sh
    if regime == "sizeinv_v1":
        return (tx, ty), 0.95, 0.8
    return (tx, ty), 0.95 * 0.4 / tx, 0.8 * 0.4 / tx  # v2: x-extent of spawn/goal regions held at fixed-table value
    # (y uses the same factor -> y-regions shrink slightly below baseline; conservative)


rows = []
for s in (0.10, 0.15, 0.20, 0.25, 0.30):
    for regime in ("fixed", "sizeinv_v1", "sizeinv_v2_proposed"):
        (tx, ty), msc, mtask = regime_dims(s, regime)
        sp = (rng.random((200_000, 2)) - 0.5) * np.array([msc * tx, msc * ty])
        g = (rng.random((200_000, 2)) - 0.5) * np.array([mtask * tx, mtask * ty])
        ok = np.linalg.norm(g - sp, axis=1) >= 1.1 * FR   # approx. of sample_goal min-separation rejection
        sp, g = sp[ok], g[ok]
        dgoal = np.linalg.norm(g - sp, axis=1)
        rows.append(dict(scale=s, regime=regime, table_x=tx, table_y=ty,
                         spawn_half_x=0.5 * msc * tx, goal_half_x=0.5 * mtask * tx, table_near_edge_x=-0.5 * tx,
                         base_to_table_edge=-0.5 * tx - BASE[0],
                         mean_spawn_goal_dist=dgoal.mean(),
                         frac_goal_beyond_EE_clip=(g[:, 0] > CLIP_X).mean(),
                         frac_goal_beyond_phys_reach=(np.linalg.norm(g - BASE, axis=1) > REACH).mean(),
                         frac_spawn_com_beyond_phys_reach=(np.linalg.norm(sp - BASE, axis=1) > REACH).mean(),
                         # can the EE (clip x<=+0.10) get to the object's far side (CoM_x + s) to pull it back?
                         frac_spawn_far_side_within_clip=(sp[:, 0] + s <= CLIP_X).mean(),
                         frac_spawn_near_side_within_clip=(sp[:, 0] - s <= CLIP_X).mean()))
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/geometry_audit.csv", index=False)
pd.set_option("display.width", 250)
print(f"physical EE reach at table height (FK, 1M samples, z in [0,0.15] base frame): {REACH:.3f} m "
      f"-> world x_max = {BASE[0] + REACH:+.3f}; software clip world x_max = {CLIP_X:+.2f}")
print(df.round(3).to_string(index=False))
