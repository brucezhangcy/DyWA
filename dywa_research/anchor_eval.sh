#!/usr/bin/env bash
# Phase-A sanity ANCHOR: reproduce the teacher's in-distribution success (~90%+).
# Corrected knobs vs the 0% smoke: ws_bound x-max 0.6 (reach-the-table fix) + filter=yes.json.
# All outputs stay inside dywa_research/. Repo + output/trossen_pipeline untouched.
set -euo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/bruce/DyWA/dywa_research/shims:/home/bruce/DyWA/dywa/exp/train:/home/bruce/DyWA/dywa/src:${PYTHONPATH:-}
cd /home/bruce/DyWA/dywa/exp/train
PYTORCH_JIT=0 python /home/bruce/DyWA/dywa_research/valid_ppo_arm_patched.py \
  +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
  ++name=phaseA_anchor \
  ++path.root=/home/bruce/DyWA/dywa_research/results/anchor \
  ++env.num_env=64 ++global_device=cuda:0 \
  ++env.trossen.base_height=table \
  "++env.trossen.ws_bound=[[-0.25,-0.35,-0.1],[0.6,0.35,0.4]]" \
  ++env.single_object_scene.dgn.data_path=/home/bruce/DyWA/data/meta-v8 \
  ++env.single_object_scene.dgn.urdf_path=/home/bruce/DyWA/dywa_research/urdf_fixed \
  ++env.single_object_scene.dgn.pose_path=/home/bruce/DyWA/data/meta-v8/unique_dgn_poses \
  ++env.single_object_scene.filter_file=/home/bruce/DyWA/data/yes.json \
  ++env.single_object_scene.mode=valid \
  ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
  ++load_ckpt=/home/bruce/DyWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt \
  ++eval_track_per_obj_suc_rate=True \
  ++eval_step=600 \
  ++use_nvdr_record_episode=false ++plot_pc=0
