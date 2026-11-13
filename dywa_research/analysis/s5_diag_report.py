#!/usr/bin/env python3
"""CPU report for the PerEnvDiag dumps written by valid_ppo_arm_diag.py (via gpu_s5_diag.sh).
Usage: python3 dywa_research/analysis/s5_diag_report.py results/s5_diag/*/diag.npz
Tests three Session-4 code-reading hypotheses (CRITIQUES #13-#14) using the data:
  H-loop  : immediate (<=3 step) terminations come from a FEW envs that replay one bad spawn pose
  H-z     : valid-mode spawn z is not rescaled with object scale (valid_pose_z != stable_pose0_z_rescaled)
  H-dist  : success depends on init->goal distance (so a table/region change that lengthens it confounds size)
  H-base  : robot root tensor x follows table_dims (-0.5*table_x-0.3) although the welded base stays at -0.5
"""
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) == 1:
    raise SystemExit('usage: s5_diag_report.py <diag.npz> [...]')

for path in sys.argv[1:]:
    if not Path(path).is_file():
        raise SystemExit('missing diagnostic: ' + path)
    d = np.load(path, allow_pickle=False)
    if 'schema_version' not in d or int(d['schema_version']) != 2:
        raise SystemExit('regenerate diagnostic with schema 2: ' + path)
    if d['errors'].size:
        raise SystemExit('diagnostic contains errors: ' + str(d['errors'].tolist()))
    print(f"\n===== {path}")
    if "static_table_dims" in d:
        print("table_dims", d["static_table_dims"].round(3).tolist(), " table_pos", d.get("static_table_pos", np.nan))
    if "static_robot_root_tensor_xyz" in d:
        r = d["static_robot_root_tensor_xyz"]
        print(f"[H-base] robot root tensor x: mean {r[:, 0].mean():+.3f} (welded create-time base = -0.500)")
    if "static_valid_pose_z" in d and "static_stable_pose0_z_rescaled" in d:
        vz, sz = d["static_valid_pose_z"], d["static_stable_pose0_z_rescaled"]
        dz = vz - sz
        print(f"[H-z] valid spawn z - rescaled stable z: mean {dz.mean():+.4f}  min {dz.min():+.4f}  max {dz.max():+.4f} "
              f"(negative = spawned LOWER than the scale-corrected stable height, i.e. possibly inside the table)")
    env, ln, suc, to = d["env"], d["len"], d["suc"].astype(bool), d["timeout"].astype(bool)
    if len(env) == 0:
        print("no episodes recorded"); continue
    imm = ln <= 3
    n_env = len(d['static_cur_radii']) if 'static_cur_radii' in d else int(env.max()) + 1
    if (env < 0).any() or (env >= n_env).any():
        raise SystemExit('invalid episode environment index')
    per_imm = np.bincount(env[imm], minlength=n_env); per_all = np.bincount(env, minlength=n_env)
    order = np.argsort(-per_imm)
    top = per_imm[order[:5]]
    print(f"episodes {len(env)}  success {suc.mean():.3f}  immediate {imm.mean():.3f}  timeout {to.mean():.3f}")
    if imm.any():
        print(f"[H-loop] envs with >=1 immediate ep: {(per_imm > 0).sum()}/{n_env}; top-5 envs hold "
              f"{top.sum()}/{imm.sum()} immediate eps ({top.sum() / imm.sum():.0%}); counts {top.tolist()}")
        if "static_valid_pose_z" in d and "static_stable_pose0_z_rescaled" in d:
            if 'static_env_asset_index' not in d:
                raise SystemExit('missing env-to-asset mapping')
            dz = dz[d['static_env_asset_index'].astype(int)]
            bad = per_imm > 0.5 * per_all
            print(f"[H-z x H-loop] mean dz for looping envs {dz[bad].mean() if bad.any() else np.nan:+.4f} vs "
                  f"others {dz[~bad].mean():+.4f}")
    ni = ~imm
    env_succ = np.array([suc[ni & (env == e)].mean() if (ni & (env == e)).any() else np.nan for e in range(n_env)])
    filtered = suc[ni].mean() if ni.any() else float('nan')
    balanced = np.nanmean(env_succ) if np.isfinite(env_succ).any() else float('nan')
    print(f"success excl. immediate {filtered:.3f}; env-balanced (excl. immediate) {balanced:.3f}")
    print(f"immediate successes: {int(suc[imm].sum())}; observed envs: {int((per_all > 0).sum())}/{n_env}")
    if 'pending_env' in d:
        print(f"right-censored episodes at budget end: {len(d['pending_env'])}; do not count these as completed failures or successes")
    if not np.isfinite(d['start']).all() or not np.isfinite(d['goal']).all():
        raise SystemExit('nonfinite episode geometry')
    dist = np.linalg.norm(d["goal"][:, :2] - d["start"][:, :2], axis=1)
    edges = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 9]
    print("[H-dist] success (excl. immediate) by init->goal xy distance:")
    for a, b in zip(edges, edges[1:]):
        m = ni & (dist >= a) & (dist < b)
        if m.sum():
            print(f"   [{a:.1f},{b:.1f}) n={m.sum():5d}  succ={suc[m].mean():.3f}  timeout={to[m].mean():.3f}")
