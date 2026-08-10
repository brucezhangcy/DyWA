#!/usr/bin/env bash
# Corrected threshold test: goal tolerance is set by force_rad (which overrides task.goal_radius
# in __post_init__). Deliverable A inadvertently used force_rad=0.1 (default). Here we (a) recheck
# the in-dist anchor at the stricter intended 0.05, and (b) sweep force_rad at the collapsed scale
# s=0.20 to test the threshold-artifact hypothesis PROPERLY.
set -uo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/bruce/DyWA/dywa_research/shims:/home/bruce/DyWA/dywa/exp/train:/home/bruce/DyWA/dywa/src:${PYTHONPATH:-}
RESEARCH=/home/bruce/DyWA/dywa_research
DATA=/home/bruce/DyWA/data
CKPT=/home/bruce/DyWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt
OUT=$RESEARCH/results/phase_b_threshold
mkdir -p "$OUT"
CSV="$OUT/threshold.csv"
echo "tag,scale,force_rad,episode_count,success_count,suc_rate,balanced_suc_rate,status" > "$CSV"
cd /home/bruce/DyWA/dywa/exp/train

# tag | scale | force_rad
RUNS=(
  "indist_fr0.05|0.10|0.05"
  "collapse_fr0.05|0.20|0.05"
  "collapse_fr0.15|0.20|0.15"
  "collapse_fr0.20|0.20|0.20"
)
for row in "${RUNS[@]}"; do
  IFS='|' read -r tag scale fr <<< "$row"
  cell="$OUT/$tag"; mkdir -p "$cell"
  echo "[thr] $tag scale=$scale force_rad=$fr"
  PYTORCH_JIT=0 python "$RESEARCH/valid_ppo_arm_patched.py" \
    +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
    ++name="phaseBthr_$tag" ++path.root="$cell" \
    ++env.num_env=64 ++global_device=cuda:0 ++env.trossen.base_height=table \
    "++env.trossen.ws_bound=[[-0.25,-0.35,-0.1],[0.6,0.35,0.4]]" \
    ++env.single_object_scene.dgn.data_path="$DATA/meta-v8" \
    ++env.single_object_scene.dgn.urdf_path="$RESEARCH/urdf_fixed" \
    ++env.single_object_scene.dgn.pose_path="$DATA/meta-v8/unique_dgn_poses" \
    ++env.single_object_scene.filter_file="$DATA/test_set.json" \
    ++env.single_object_scene.mode=valid \
    ++env.single_object_scene.min_scale=$scale ++env.single_object_scene.max_scale=$scale \
    ++force_rad=$fr \
    ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
    ++load_ckpt="$CKPT" ++eval_track_per_obj_suc_rate=True ++eval_step=1200 \
    ++use_nvdr_record_episode=false ++plot_pc=0 > "$cell/cell.log" 2>&1
  status=$?
  line=$(grep -aoE "\{'scalar': \{'valid/episode_count':[^}]*\}\}" "$cell/cell.log" | tail -1)
  ep=$(echo "$line"  | grep -oE "episode_count': [0-9]+"      | grep -oE "[0-9]+" | tail -1)
  sc=$(echo "$line"  | grep -oE "success_count': [0-9]+"      | grep -oE "[0-9]+" | tail -1)
  sr=$(echo "$line"  | grep -oE "'valid/suc_rate': [0-9.]+"   | grep -oE "[0-9.]+" | tail -1)
  bsr=$(echo "$line" | grep -oE "balanced_suc_rate': [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  st="OK"; [ $status -ne 0 ] && st="FAILED"
  echo "$tag,$scale,$fr,${ep:-NA},${sc:-NA},${sr:-NA},${bsr:-NA},$st" >> "$CSV"
  echo "[thr] $tag -> suc_rate=${sr:-NA}"
done
echo "[thr] DONE"; cat "$CSV"
