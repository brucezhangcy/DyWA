#!/usr/bin/env bash
# Phase-A scale sweep (VALIDATED eval; anchor reproduced 94.5% in-dist).
# Measures single-arm teacher success vs object scale on HELD-OUT objects (test_set.json),
# scales spanning in-distribution [0.05,0.10] and extrapolation up to 0.30.
# Uses the validated knobs: ws_bound x-max 0.6, urdf_fixed, teacher-stage1-wide ckpt.
# All outputs under dywa_research/results/phase_a/. Repo + output/ untouched.
set -uo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/bruce/DyWA/dywa_research/shims:/home/bruce/DyWA/dywa/exp/train:/home/bruce/DyWA/dywa/src:${PYTHONPATH:-}

RESEARCH=/home/bruce/DyWA/dywa_research
DATA=/home/bruce/DyWA/data
CKPT=/home/bruce/DyWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt
OUT=$RESEARCH/results/phase_a
FILTER="${FILTER:-$DATA/test_set.json}"      # held-out object set
GOAL_RADIUS="${GOAL_RADIUS:-0.05}"           # absolute regime (primary)
EVAL_STEP="${EVAL_STEP:-1200}"               # longer -> more episodes/point -> tighter estimate
NUM_ENV="${NUM_ENV:-64}"
SCALES="${SCALES:-0.05 0.075 0.10 0.15 0.20 0.25 0.30}"

mkdir -p "$OUT"
CSV="$OUT/scale_sweep.csv"
echo "scale,goal_radius,filter,episode_count,success_count,suc_rate,balanced_suc_rate,status" > "$CSV"
cd /home/bruce/DyWA/dywa/exp/train

for s in $SCALES; do
  tag="s${s}_gr${GOAL_RADIUS}"
  cell="$OUT/$tag"; mkdir -p "$cell"
  echo "[sweep] scale=$s goal_radius=$GOAL_RADIUS -> $cell"
  PYTORCH_JIT=0 python "$RESEARCH/valid_ppo_arm_patched.py" \
    +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
    ++name="phaseA_$tag" \
    ++path.root="$cell" \
    ++env.num_env="$NUM_ENV" ++global_device=cuda:0 \
    ++env.trossen.base_height=table \
    "++env.trossen.ws_bound=[[-0.25,-0.35,-0.1],[0.6,0.35,0.4]]" \
    ++env.single_object_scene.dgn.data_path="$DATA/meta-v8" \
    ++env.single_object_scene.dgn.urdf_path="$RESEARCH/urdf_fixed" \
    ++env.single_object_scene.dgn.pose_path="$DATA/meta-v8/unique_dgn_poses" \
    ++env.single_object_scene.filter_file="$FILTER" \
    ++env.single_object_scene.mode=valid \
    ++env.single_object_scene.min_scale="$s" ++env.single_object_scene.max_scale="$s" \
    ++env.task.goal_radius="$GOAL_RADIUS" \
    ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
    ++load_ckpt="$CKPT" \
    ++eval_track_per_obj_suc_rate=True \
    ++eval_step="$EVAL_STEP" \
    ++use_nvdr_record_episode=false ++plot_pc=0 \
    > "$cell/cell.log" 2>&1
  status=$?
  # parse the final result dict from the log
  line=$(grep -aoE "\{'scalar': \{'valid/episode_count':[^}]*\}\}" "$cell/cell.log" | tail -1)
  ep=$(echo "$line"    | grep -oE "episode_count': [0-9]+"        | grep -oE "[0-9]+" | tail -1)
  sc=$(echo "$line"    | grep -oE "success_count': [0-9]+"        | grep -oE "[0-9]+" | tail -1)
  sr=$(echo "$line"    | grep -oE "suc_rate': [0-9.]+"            | grep -oE "[0-9.]+" | head -1)
  bsr=$(echo "$line"   | grep -oE "balanced_suc_rate': [0-9.]+"   | grep -oE "[0-9.]+" | tail -1)
  st="OK"; [ $status -ne 0 ] && st="FAILED"
  echo "$s,$GOAL_RADIUS,$(basename $FILTER),${ep:-NA},${sc:-NA},${sr:-NA},${bsr:-NA},$st" >> "$CSV"
  echo "[sweep] scale=$s -> suc_rate=${sr:-NA} (ep=${ep:-NA})"
done

echo "[sweep] DONE. CSV:"; cat "$CSV"
