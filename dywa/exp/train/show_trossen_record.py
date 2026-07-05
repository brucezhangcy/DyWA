#!/usr/bin/env python3
"""
Trossen playback recorder.

Loads a trained PPO checkpoint and rolls it out in the Trossen env, recording
frames via a directly-managed Isaac Gym GPU camera (same approach as
test_trossen_env.py). Avoids NvdrRecordEpisode/NvdrCameraWrapper which assume
the Franka URDF asset layout.

Usage (inside Docker):
    cd /home/user/DyWA/dywa/exp/train
    PYTORCH_JIT=0 python3 show_trossen_record.py \
        +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
        ++tag=trossen_show \
        ++env.num_env=4 \
        ++env.trossen.base_height=table \
        ++global_device=cuda:0 \
        ++path.root=/home/user/DyWA/output/trossen_pipeline/show \
        ++env.single_object_scene.dgn.data_path=/input/DGN/meta-v8 \
        ++env.single_object_scene.dgn.pose_path=/input/DGN/meta-v8/unique_dgn_poses \
        ++env.single_object_scene.filter_file=/input/DGN/yes.json \
        ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
        ++load_ckpt=<path/to/ckpt> \
        ++sample_action=false
"""

import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float32

from isaacgym import gymapi, gymtorch

import torch as th
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
from omegaconf import OmegaConf
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
    n_steps: int = 500
    # Warmup steps to re-accumulate env-normalizer stats online when the
    # trained stat/ ckpt is unavailable. Uses all parallel envs (each a distinct
    # object/scale/goal/physics sample) to converge mean/std quickly, then freezes.
    warmup: int = 0
    cam_eye: tuple = (-0.25, -1.6, 0.7)
    cam_at: tuple = (-0.25, 0.0, 0.25)
    cam_h: int = 480
    cam_w: int = 640
    out_path: str = 'output/trossen_pipeline/trossen_playback.mp4'
    cam_env: int = 0       # which env the camera films
    scout: bool = False    # if True: don't record; rank envs by successful push


def _make_camera(gym, sim, envs, H, W, eye, at, env_idx=0):
    prop = gymapi.CameraProperties()
    prop.height = H
    prop.width = W
    prop.enable_tensors = True
    prop.use_collision_geometry = False
    prop.horizontal_fov = 68.0
    prop.near_plane = 0.01
    prop.far_plane = 100.0
    e = envs[env_idx]
    # set_camera_location with an env handle is env-LOCAL — no origin offset.
    cam = gym.create_camera_sensor(e, prop)
    gym.set_camera_location(
        cam, e,
        gymapi.Vec3(*eye),
        gymapi.Vec3(*at),
    )
    desc = gym.get_camera_image_gpu_tensor(sim, e, cam, gymapi.IMAGE_COLOR)
    color_t = gymtorch.wrap_tensor(desc)
    return cam, color_t


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

    # SCOUT MODE: run the (deterministic) rollout once and rank envs by how
    # well they push their object to the goal, so we can film the best one.
    if cfg.scout:
        import numpy as _np
        n = cfg.env.num_env
        first_succ = _np.full(n, -1, dtype=int)   # step of first success per env
        min_dist = _np.full(n, _np.inf)           # closest obj->goal seen per env
        rms = env.normalizer.obs_rms
        for i, (act, obs, rew, done, info) in enumerate(agent.test(
                sample=cfg.sample_action, steps=cfg.n_steps)):
            suc = info.get('success') if isinstance(info, dict) else None
            if suc is not None:
                suc = suc.detach().cpu().numpy().astype(bool)
                newly = suc & (first_succ < 0)
                first_succ[newly] = i
            try:  # track obj->goal distance from unnormalized obs
                def _u(k):
                    t = obs[k]
                    r = rms[k]
                    return (t * th.sqrt(r.var + 1e-6) + r.mean
                            ).detach().cpu().numpy()
                d = _np.linalg.norm(_u('object_state')[:, :2]
                                    - _u('goal')[:, :2], axis=-1)
                min_dist = _np.minimum(min_dist, d)
            except Exception:
                pass
        succ_envs = [int(e) for e in _np.where(first_succ >= 0)[0]]
        ranked = sorted(succ_envs, key=lambda e: first_succ[e])
        print(f'SCOUT: {len(succ_envs)}/{n} envs succeeded.')
        print(f'SCOUT: earliest-success envs (env:step): '
              f'{[(e, int(first_succ[e])) for e in ranked[:10]]}')
        order = _np.argsort(min_dist)
        print('SCOUT: closest-approach envs (env:dist): '
              + str([(int(e), round(float(min_dist[e]), 3)) for e in order[:10]]))
        return

    # Camera + video writer
    cam, color_t = _make_camera(env.gym, env.sim, env.envs,
                                cfg.cam_h, cfg.cam_w,
                                cfg.cam_eye, cfg.cam_at,
                                env_idx=cfg.cam_env)
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix('.tmp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    H, W = color_t.shape[:2]
    writer = cv2.VideoWriter(str(tmp_path), fourcc, 30, (W, H))
    print(f'Recording → {out_path}  ({W}×{H} @ 30 fps)')

    if cfg.warmup > 0:
        env.normalizer.train()  # accumulate obs running mean/std online
        print(f'Warming up env-normalizer for {cfg.warmup} steps '
              f'across {cfg.env.num_env} parallel envs...')

    step_count = 0
    rec_count = 0
    for i, (act, obs, rew, done, info) in enumerate(agent.test(
            sample=cfg.sample_action, steps=cfg.warmup + cfg.n_steps)):
        if i < cfg.warmup:
            continue  # discard warmup frames; normalizer is still adapting
        if i == cfg.warmup and cfg.warmup > 0:
            env.normalizer.eval()  # freeze the reconstructed stats
            print('Normalizer frozen; recording...')
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        env.gym.start_access_image_tensors(env.sim)
        rgba = color_t.cpu().numpy()
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        writer.write(bgr)
        env.gym.end_access_image_tensors(env.sim)
        if rec_count % 50 == 0 and rew is not None:
            print(f'step {rec_count:4d} | rew_mean: {rew.mean().item():.4f}')
        rec_count += 1
        step_count += 1

    writer.release()
    Path(tmp_path).rename(out_path)
    print(f'Done — {step_count} frames written to {out_path}')


if __name__ == '__main__':
    main()
