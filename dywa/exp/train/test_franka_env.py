#!/usr/bin/env python3
"""
Top-down debug test for the Franka arm placement.

Mirrors test_trossen_env.py but uses the Franka 7-DOF arm with base_height='table'
(matching icra_base.yaml). A top-down camera at z=3 looking straight down at world
origin shows where the arm actually renders in X-Y. If the arm appears at world
(-0.5, 0) — behind the table — the Franka placement is correct. If it appears at
world (0, 0) — center of the table — Franka has the same fix_base_link/reset-write
bug that Trossen had (see dev_log "Hidden bug discovered").

Usage (inside Docker):
    cd /home/user/DyWA/dywa
    PYTORCH_JIT=0 PYTHONPATH=src python exp/train/test_franka_env.py --headless --record
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


def _make_camera(gym, sim, envs, H=480, W=640,
                 eye=(0.001, 0.0, 3.0), at=(0.0, 0.0, 0.0)):
    """TOP-DOWN debug view. Eye 3 m above origin looking straight down.
    If Franka is correctly placed at world (-0.5, 0), it appears BELOW the table
    in the image (image-top = world +X). If at (0, 0) it overlaps with the table."""
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
    color_t = gymtorch.wrap_tensor(desc)
    return cam, color_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--num_env', type=int, default=4)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--out', type=str, default='output/franka_topdown.mp4')
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
        'franka.init_type': 'home',         # simplest init; avoid sampling code paths
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
    print('=== Initial observation shapes ===')
    for k, v in obs.items():
        print(f'  {k}: {tuple(v.shape)}')
    act_dim = env.action_space.shape[0]
    print(f'\nAction dim: {act_dim}  |  Running {args.steps} steps...\n')

    env.refresh_tensors()
    body = env.tensors['body']
    robot = env.robot
    # Franka has different link names than Trossen
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

    # Brief stepping — keep arm still, just confirm what's rendered.
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
        # No ffmpeg in container; rename .tmp.mp4 → .mp4 (mp4v codec, transcode on host).
        Path(tmp_path).rename(out_path)
        print(f'\nRecording → {args.out} (mp4v fourcc; transcode to H.264 on host)')

    print('\nDone — placement diagnostic complete.')


if __name__ == '__main__':
    main()
