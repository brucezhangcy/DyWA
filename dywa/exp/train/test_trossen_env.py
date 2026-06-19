#!/usr/bin/env python3
"""
Visualization test: Trossen WidowX AI in the DyWA Isaac Gym environment.

Usage (headless, with GPU-camera recording, inside Docker):
    cd /home/user/DyWA/dywa
    PYTHONPATH=src python exp/train/test_trossen_env.py --headless --record

DGN dataset path defaults to /input/DGN (Docker mount). Override with the DGN_PATH env var
if running outside Docker.

Success criteria:
  - Console prints obs dict shapes and reward every 50 steps
  - 500 steps complete without error and "Done — test passed." prints
  - With --record: output/trossen_vis.mp4 is written
"""

import argparse
import os
import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float32

# isaacgym must be imported before torch
from isaacgym import gymapi, gymtorch

import torch as th
from pathlib import Path

from env.arm_env import ArmEnv, ArmEnvConfig
from env.task.push_with_arm_task import PushWithArmTask
from util.config import recursive_replace_map


# ── camera helpers ────────────────────────────────────────────────────────────

def _make_camera(gym, sim, envs, H=480, W=640,
                 eye=(0.3, -1.4, 0.9), at=(-0.4, 0.0, 0.55)):
    """3/4 view from front-side, looking back at arm. Arm base is at world (-0.5, 0, 0.4)
    — 0.3 m BEHIND the table back edge, at table-top height. Camera looks past the table
    front face toward the arm hovering behind it."""
    prop = gymapi.CameraProperties()
    prop.height = H
    prop.width = W
    prop.enable_tensors = True
    prop.use_collision_geometry = False
    prop.horizontal_fov = 68.0
    prop.near_plane = 0.01
    prop.far_plane = 100.0

    cam = gym.create_camera_sensor(envs[0], prop)
    gym.set_camera_location(
        cam, envs[0],
        gymapi.Vec3(*eye),
        gymapi.Vec3(*at),
    )

    desc = gym.get_camera_image_gpu_tensor(sim, envs[0], cam, gymapi.IMAGE_COLOR)
    color_t = gymtorch.wrap_tensor(desc)   # (H, W, 4) uint8 RGBA on GPU
    return cam, color_t


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true',
                        help='Run without viewer window (required when DISPLAY is unset)')
    parser.add_argument('--record', action='store_true',
                        help='Record frames via GPU camera → output/trossen_vis.mp4')
    parser.add_argument('--num_env', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of simulation steps to run')
    parser.add_argument('--out', type=str, default='output/trossen_vis.mp4',
                        help='Output video path (used with --record)')
    parser.add_argument('--base_height', type=str, default='table',
                        choices=['table', 'ground', 'origin'],
                        help='Where to weld the Trossen base. '
                             '"table" = production (-0.5, 0, 0.4); '
                             '"origin" = world (0, 0, 0), reproduces the pre-fix bug.')
    args = parser.parse_args()

    if args.record:
        args.headless = True   # camera recording always runs headless

    DGN = os.environ.get('DGN_PATH', '/input/DGN')
    cfg = ArmEnvConfig()
    cfg = recursive_replace_map(cfg, {
        'num_env': args.num_env,
        'use_viewer': not args.headless,
        'which_robot': 'trossen',
        'robot_state_type': 'pos_vel6',
        'task.timeout': 200,
        'task.nearest_induce': False,
        'trossen.init_type': 'home',
        'trossen.base_height': args.base_height,   # production: 'table' (z=0.4); 'origin' reproduces the pre-fix bug
        # keepout_radius left at default 0.3 → base at x=-0.5 (0.3 m behind table back edge x=-0.2)
        # use DGN objects (available locally) instead of default ACRONYM (Docker path)
        'single_object_scene.base_set': ('dgn',),
        'single_object_scene.dgn.data_path': f'{DGN}/meta-v8',
        'single_object_scene.dgn.pose_path': f'{DGN}/meta-v8/unique_dgn_poses',
        'single_object_scene.filter_file': f'{DGN}/yes.json',
        'single_object_scene.load_cloud': True,
        'single_object_scene.goal_type': 'stable',
        'single_object_scene.init_type': 'stable',
        'single_object_scene.randomize_init_pos': True,
        'single_object_scene.randomize_init_orn': True,
        'single_object_scene.num_object_types': 16,
    })

    env = ArmEnv(cfg, task_cls=PushWithArmTask)
    env.setup()
    env.gym.prepare_sim(env.sim)
    env.refresh_tensors()

    # ── optional recording setup ──────────────────────────────────────────────
    writer = None
    cam = None
    color_t = None
    if args.record:
        import cv2
        cam, color_t = _make_camera(env.gym, env.sim, env.envs)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix('.tmp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        H, W = color_t.shape[:2]
        fps = 30
        writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (W, H))
        print(f'Recording → {out_path}  ({W}×{H} @ {fps} fps)')

    # ── reset ─────────────────────────────────────────────────────────────────
    obs = env.reset()
    print('=== Initial observation shapes ===')
    for k, v in obs.items():
        print(f'  {k}: {tuple(v.shape)}')

    act_dim = env.action_space.shape[0]
    print(f'\nAction dim: {act_dim}  |  Running {args.steps} steps...\n')

    # ── diagnostic: print arm body positions after reset ──────────────────────
    env.refresh_tensors()
    body = env.tensors['body']
    robot = env.robot
    link_names = ['base_link', 'link_1', 'link_2', 'link_3',
                  'link_4', 'link_5', 'link_6', 'ee_gripper_link']
    print('=== Arm body world positions after reset ===')
    for name in link_names:
        from isaacgym import gymapi as _gymapi
        idx = env.gym.find_actor_rigid_body_index(
            env.envs[0], robot.handles[0], name, _gymapi.DOMAIN_SIM)
        pos = body[idx, :3].cpu().numpy()
        print(f'  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')
    print(f'  Table pos: {env.scene.cfg.table_pos}, dims: {env.scene.cfg.table_dims}')
    print()

    # ── step loop ─────────────────────────────────────────────────────────────
    for step in range(args.steps):
        # Large Y sweep (clearly visible), gentle X/Z oscillation. 2 full cycles.
        t = 2 * np.pi * step / args.steps * 2   # 2 cycles total → slower, larger reach
        actions = th.zeros(args.num_env, act_dim,
                           dtype=th.float32,
                           device=env.cfg.th_device)
        actions[:, 0] = float(0.12 * np.sin(t))          # sweep X (was 0.06)
        actions[:, 1] = float(0.20 * np.sin(t))          # sweep Y (was 0.10) — main motion
        actions[:, 2] = float(0.10 * np.sin(2 * t))      # Z oscillation (was 0.05)
        obs, rew, done, info = env.step(actions)

        if step % 50 == 0:
            print(f'step {step:4d} | reward mean: {rew.mean().item():.4f}')

        # capture frame
        if writer is not None:
            import cv2
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.start_access_image_tensors(env.sim)
            rgba = color_t.cpu().numpy()          # (H, W, 4) uint8
            bgr  = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(bgr)
            env.gym.end_access_image_tensors(env.sim)

        if not args.headless:
            env.gym.sync_frame_time(env.sim)

    if writer is not None:
        writer.release()
        import subprocess
        subprocess.run([
            'ffmpeg', '-y', '-i', str(tmp_path),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
            str(out_path)
        ], check=True, capture_output=True)
        tmp_path.unlink()
        print(f'\nVideo saved → {args.out}')

    print('\nDone — test passed.')


if __name__ == '__main__':
    main()
