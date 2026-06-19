#!/usr/bin/env python3
"""
DEPRECATED — do not use. Renders are broken: the Isaac Gym GPU camera
attached here cycles its output every ~60 frames (verified 2026-05-20:
frame_N == frame_N+60 md5-identical across a 162-frame success episode),
and the rendered scene also omits the spawned object on the table. Likely
a render-buffer / camera-sync bug specific to this approach.

Use show_trossen_nvdr_record.py instead — it wraps env with
NvdrRecordEpisode (URDF-mesh + nvdiffrast rendering), which is the proven
path that produced the working Franka rollouts.

Original docstring below for archival.
---
Trossen per-episode rollout recorder.

Same idea as show_trossen_record.py, but splits output into one MP4 per
episode (in `succ/` or `fail/` subdirs based on info['success']) so the
output matches the Franka NvdrRecordEpisode format.

Camera is attached to env 0; only env 0's episodes are recorded.
"""

import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float32

from isaacgym import gymapi, gymtorch

import torch as th
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
from icecream import ic
from gym import spaces
import cv2

from models.common import map_struct
from util.torch_util import dcn
from util.hydra_cli import hydra_cli
from env.util import set_seed

from train_ppo_arm import (
    Config as TrainConfig,
    setup as setup_logging,
    load_agent,
    load_env)


@dataclass
class Config(TrainConfig):
    sample_action: bool = False
    n_steps: int = 600
    cam_eye: tuple = (-0.25, -1.6, 0.7)
    cam_at: tuple = (-0.25, 0.0, 0.25)
    cam_h: int = 480
    cam_w: int = 640
    out_dir: str = 'output/trossen_pipeline/rollout_episodes'
    min_frames: int = 5  # skip ultra-short residual buffers right after warmup


def _make_camera(gym, sim, envs, H, W, eye, at):
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


def _flush(buffer, dest_dir, ep_idx, fps, size):
    if len(buffer) < 1:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f'episode_{ep_idx:04d}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for f in buffer:
        writer.write(f)
    writer.release()
    return path


@hydra_cli(config_name='show')
def main(cfg: Config):
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    set_seed(cfg.env.seed)
    cfg, env = load_env(cfg, path, freeze_env=True, check_viewer=False)

    obs_space = map_struct(
        env.observation_space,
        lambda src, _: src.shape,
        base_cls=spaces.Box,
        dict_cls=(Mapping, spaces.Dict))
    if cfg.state_net_blocklist is not None:
        for key in cfg.state_net_blocklist:
            obs_space.pop(key, None)
    dim_act = (
        env.action_space.shape if isinstance(env.action_space, spaces.Box)
        else env.action_space.n)
    cfg = replace(cfg, net=replace(cfg.net, obs_space=obs_space, act_space=dim_act))
    agent = load_agent(cfg, env, None, None)
    agent.eval()

    cam, color_t = _make_camera(env.gym, env.sim, env.envs,
                                cfg.cam_h, cfg.cam_w,
                                cfg.cam_eye, cfg.cam_at)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    succ_dir = out_dir / 'succ'
    fail_dir = out_dir / 'fail'
    H, W = color_t.shape[:2]
    print(f'Recording per-episode MP4s under {out_dir}  ({W}x{H} @ 30 fps)')

    buffer = []
    succ_idx = 0
    fail_idx = 0
    total_succ = 0
    total_fail = 0
    step_count = 0

    for (act, obs, rew, done, info) in agent.test(
            sample=cfg.sample_action, steps=cfg.n_steps):
        env.gym.render_all_camera_sensors(env.sim)
        env.gym.start_access_image_tensors(env.sim)
        rgba = color_t.cpu().numpy()
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        buffer.append(bgr.copy())
        env.gym.end_access_image_tensors(env.sim)

        if done is not None and bool(done[0].item()):
            success = bool(info['success'][0].item()) if (info and 'success' in info) else False
            if len(buffer) >= cfg.min_frames:
                if success:
                    p = _flush(buffer, succ_dir, succ_idx, 30, (W, H))
                    succ_idx += 1
                    total_succ += 1
                    tag = 'SUCC'
                else:
                    p = _flush(buffer, fail_dir, fail_idx, 30, (W, H))
                    fail_idx += 1
                    total_fail += 1
                    tag = 'FAIL'
                print(f'  step {step_count:4d} | episode {tag} ({len(buffer)} frames) → {p.name}')
            buffer = []

        if step_count % 50 == 0 and rew is not None:
            print(f'step {step_count:4d} | rew_mean: {rew.mean().item():.4f} '
                  f'(succ {total_succ} / fail {total_fail})')
        step_count += 1

    if buffer:
        _flush(buffer, out_dir / 'truncated', 0, 30, (W, H))
        print(f'  flushed {len(buffer)}-frame trailing buffer to truncated/')
    print(f'Done — env 0 produced {total_succ} successes, {total_fail} failures '
          f'in {step_count} sim steps.')


if __name__ == '__main__':
    main()
