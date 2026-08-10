#!/usr/bin/env bash
# Phase-C data collection: per-object single-arm success at transition scales (where objects
# differentiate), to train an interpretable "can one arm do THIS object?" competence predictor.
# Captures /tmp/per-cat-suc.pkl (object -> success rate) after each run.
set -uo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/bruce/DyWA/dywa_research/shims:/home/bruce/DyWA/dywa/exp/train:/home/bruce/DyWA/dywa/src:${PYTHONPATH:-}
R=/home/bruce/DyWA/dywa_research; DATA=/home/bruce/DyWA/data
CKPT=/home/bruce/DyWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt
OUT=$R/results/phase_c_predictor; mkdir -p "$OUT"
cd /home/bruce/DyWA/dywa/exp/train
for scale in 0.125 0.15 0.175; do
  cell="$OUT/s${scale}"; mkdir -p "$cell"; rm -f /tmp/per-cat-suc.pkl
  echo "[collect] scale=$scale"
  PYTORCH_JIT=0 python "$R/valid_ppo_arm_taxonomy.py" \
    +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
    ++name="phaseCpred_s${scale}" ++path.root="$cell" \
    ++env.num_env=64 ++env.seed=0 ++global_device=cuda:0 ++env.trossen.base_height=table \
    "++env.trossen.ws_bound=[[-0.25,-0.35,-0.1],[0.6,0.35,0.4]]" \
    ++env.single_object_scene.dgn.data_path="$DATA/meta-v8" \
    ++env.single_object_scene.dgn.urdf_path="$R/urdf_fixed" \
    ++env.single_object_scene.dgn.pose_path="$DATA/meta-v8/unique_dgn_poses" \
    ++env.single_object_scene.filter_file="$DATA/test_set.json" \
    ++env.single_object_scene.mode=valid \
    ++env.single_object_scene.min_scale=$scale ++env.single_object_scene.max_scale=$scale \
    ++force_rad=0.10 \
    ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
    ++load_ckpt="$CKPT" ++eval_track_per_obj_suc_rate=True ++eval_step=1500 \
    ++use_nvdr_record_episode=false ++plot_pc=0 > "$cell/cell.log" 2>&1
  [ -f /tmp/per-cat-suc.pkl ] && cp /tmp/per-cat-suc.pkl "$cell/per_obj.pkl" && echo "[collect] saved per_obj.pkl for $scale" || echo "[collect] NO pkl for $scale"
done
echo "[collect] DONE"
