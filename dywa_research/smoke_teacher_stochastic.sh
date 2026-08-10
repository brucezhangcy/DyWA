#!/usr/bin/env bash
# Phase-A smoke: validate teacher eval end-to-end (bare-metal dywa_render).
set -euo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/bruce/DyWA/dywa_research/shims:/home/bruce/DyWA/dywa/exp/train:/home/bruce/DyWA/dywa/src:${PYTHONPATH:-}
cd /home/bruce/DyWA/dywa/exp/train
DATA=/home/bruce/DyWA/data
CKPT=/home/bruce/DyWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt
PYTORCH_JIT=0 python /home/bruce/DyWA/dywa_research/valid_ppo_arm_patched.py   +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel   ++name=phaseA_smoke   ++env.num_env=32 ++global_device=cuda:0   ++env.trossen.base_height=table   "++env.trossen.ws_bound=[[-0.25,-0.35,-0.1],[0.25,0.35,0.4]]"   ++env.single_object_scene.dgn.data_path=$DATA/meta-v8   ++env.single_object_scene.dgn.urdf_path=/home/bruce/DyWA/dywa_research/urdf_fixed   ++env.single_object_scene.dgn.pose_path=$DATA/meta-v8/unique_dgn_poses   ++env.single_object_scene.filter_file=$DATA/test_set.json   ++env.single_object_scene.mode=valid   ++env.single_object_scene.min_scale=0.08 ++env.single_object_scene.max_scale=0.08   ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920'   ++load_ckpt=$CKPT   ++force_rad=0.05   ++eval_track_per_obj_suc_rate=True   ++eval_step=2400 ++sample_action=True   ++use_nvdr_record_episode=false ++plot_pc=0
