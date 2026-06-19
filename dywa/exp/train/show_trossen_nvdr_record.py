#!/usr/bin/env python3
"""
Trossen per-episode rollout recorder using NvdrRecordEpisode (URDF-mesh +
nvdiffrast rendering). Mirrors the wrap order from test_rma.py (the Franka
path that produced working rollouts):

    load_env → load_agent (loads ckpt) → env.load (normalizer stats)
    → NvdrRecordEpisode wrap (AT THE END) → agent.test() loop

The native IG GPU camera approach in show_trossen_record_episodes.py is
broken (60-frame render cycle); use this script instead.
"""

import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float32

from isaacgym import gymapi, gymtorch  # noqa: F401

import torch as th
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
from gym import spaces

from env.env.wrap.nvdr_record_episode import NvdrRecordEpisode
from models.common import map_struct
from util.hydra_cli import hydra_cli
from env.util import set_seed
from train.ckpt import last_ckpt

from train_ppo_arm import (
    Config as TrainConfig,
    setup as setup_logging,
    load_agent,
    load_env,
)


@dataclass
class Config(TrainConfig):
    n_steps: int = 600
    use_nvdr_record_episode: bool = True


@hydra_cli(config_name='show')
def main(cfg: Config):
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    set_seed(cfg.env.seed)

    # Must be set before load_env so debug-line tensors are allocated.
    cfg.env.track_debug_lines = True

    cfg, env = load_env(cfg, path, freeze_env=True, check_viewer=False)

    # Build obs_space + load agent BEFORE wrapping with NvdrRecordEpisode.
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

    # Load env normalizer stats from the run's stat/ dir (same as test_rma.py).
    if cfg.load_ckpt is not None:
        ckpt_path = Path(cfg.load_ckpt)
        ckpt_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        stat_dir = ckpt_dir.parent / 'stat'
        if stat_dir.is_dir():
            stat_ckpt = last_ckpt(str(stat_dir))
            try:
                env.load(stat_ckpt, strict=False)
                print(f'Loaded env normalizer stats from {stat_ckpt}')
            except Exception as e:
                print(f'WARN: env.load failed ({e}); using initial normalizer.')
        else:
            print(f'WARN: stat dir not found at {stat_dir}; using initial normalizer.')

    # Wrap with NvdrRecordEpisode AT THE END — same order as test_rma.py
    nv_cfg = replace(cfg.nvdr_record_episode, episode_type='all')
    env = NvdrRecordEpisode(nv_cfg, env, hide_arm=False)
    # CRITICAL: redirect agent's stored env reference to the wrapped env so
    # agent.test() drives the recorder. Without this, agent.test() uses its
    # original (unwrapped) env captured at PPO.__init__ time and the recorder
    # never sees any step() calls — leading to zero exported episodes.
    agent.env = env
    print(f'Recording {cfg.n_steps} steps; videos → {cfg.nvdr_record_episode.record_dir}/env_*/')

    # Run the rollout — NvdrRecordEpisode intercepts every env.step() and writes
    # per-episode mp4 files split into succ/fail (or both, with episode_type='all').
    step_count = 0
    for (act, obs, rew, done, info) in agent.test(steps=cfg.n_steps):
        if step_count % 50 == 0 and rew is not None:
            print(f'step {step_count:4d} | rew_mean={rew.mean().item():+.4f}')
        step_count += 1
    print(f'Done — {step_count} steps recorded.')


if __name__ == '__main__':
    main()
