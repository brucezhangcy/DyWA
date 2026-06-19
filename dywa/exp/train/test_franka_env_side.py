#!/usr/bin/env python3
"""
SIDE-VIEW visualization of Franka placement.

Companion to test_franka_env.py (which uses a top-down camera). The top-down
view makes the X-Y location obvious but doesn't show *vertical* burial — the
arm being at z=0 (floor) vs z=0.4 (table top). The side view here exposes
that: with the un-fixed code path, Franka is welded at world origin, so its
base is at z=0 (floor) and the lower body intersects the table block
(z ∈ [0, 0.4]). The arm appears partly inside the table, partly above it.

Used twice in the 2026-05-18 visualization pass: once with the un-fixed
franka.create_actors (identity gymapi.Transform — bugged) and once with the
symmetric create-time Transform fix applied to franka.py.

Usage (inside Docker):
    cd /home/user/DyWA/dywa
    PYTORCH_JIT=0 PYTHONPATH=src python exp/train/test_franka_env_side.py \
        --headless --record --out output/franka_sideview_bugged.mp4 --steps 150
"""

import argparse
import os
import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float32

from isaacgym import gymapi, gymtorch

import torch as th
from pathlib import Path

from env.arm_env import ArmEnv, ArmEnvConfig
from env.task.push_with_arm_task import PushWithArmTask
from util.config import recursive_replace_map


def _make_camera(gym, sim, envs, H=480, W=640,
                 eye=(1.6, -1.6, 0.9), at=(-0.25, 0.0, 0.3)):
    """SIDE/3-4 view. Eye is in front of and to the side of the workspace,
    slightly above table height, looking back toward the arm's intended
    location. With bugged Franka, arm appears at world origin (inside the
    table block, partly below table top). With fixed Franka, arm appears at
    (-0.5, 0, 0.4) — behind the table back edge, base at table-top height."""
    prop = gymapi.CameraProperties()
    prop.height = H
    prop.width = W
    prop.enable_tensors = True
    prop.use_collision_geometry = False
    prop.horizontal_fov = 60.0
    prop.near_plane = 0.01
    prop.far_plane = 100.0
    cam = gym.create_camera_sensor(envs[0], prop)
    gym.set_camera_location(
        cam, envs[0],
        gymapi.Vec3(*eye),
        gymapi.Vec3(*at),
    )
    desc = gym.get_camera_image_gpu_tensor(sim, envs[0], cam, gymapi.IMAGE_COLOR)
    color_t = gymtorch.wrap_tensor(desc)
    return cam, color_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--out', type=str, default='output/franka_sideview.mp4')
    parser.add_argument('--label', type=str, default='',
                        help='Tag printed for clarity (e.g. "bugged" or "fixed")')
    args = parser.parse_args()
    if args.record:
        args.headless = True

    DGN = os.environ.get('DGN_PATH', '/input/DGN')
    cfg = ArmEnvConfig()
    cfg = recursive_replace_map(cfg, {
        'num_env': args.num_env,
        'use_viewer': not args.headless,
        'which_robot': 'franka',
        'robot_state_type': 'pos_vel7',
        'task.timeout': 200,
        'task.nearest_induce': False,
        'franka.init_type': 'home',
        'franka.base_height': 'table',
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
        writer = cv2.VideoWriter(str(tmp_path), fourcc, 30, (W, H))

    obs = env.reset()
    if args.label:
        print(f'\n=== Franka SIDE-VIEW: {args.label} ===')

    env.refresh_tensors()
    body = env.tensors['body']
    robot = env.robot
    link_names = ['panda_link0', 'panda_link1', 'panda_link4', 'panda_link7',
                  'panda_hand', 'panda_leftfinger', 'panda_rightfinger']
    print('=== Franka body world positions after reset (read-back, may lie) ===')
    for name in link_names:
        try:
            idx = env.gym.find_actor_rigid_body_index(
                env.envs[0], robot.handles[0], name, gymapi.DOMAIN_SIM)
            pos = body[idx, :3].cpu().numpy()
            print(f'  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')
        except Exception as e:
            print(f'  {name}: <not found> ({e})')
    print(f'  Table pos: {env.scene.cfg.table_pos}, dims: {env.scene.cfg.table_dims}')

    act_dim = env.action_space.shape[0]
    for step in range(args.steps):
        actions = th.zeros(args.num_env, act_dim,
                           dtype=th.float32, device=env.cfg.th_device)
        obs, rew, done, info = env.step(actions)
        if writer is not None:
            import cv2
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.start_access_image_tensors(env.sim)
            rgba = color_t.cpu().numpy()
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(bgr)
            env.gym.end_access_image_tensors(env.sim)

    if writer is not None:
        writer.release()
        Path(tmp_path).rename(out_path)
        print(f'\nRecording -> {args.out} (mp4v fourcc; transcode to H.264 on host)')

    print('\nDone -- side-view diagnostic complete.')


if __name__ == '__main__':
    main()
