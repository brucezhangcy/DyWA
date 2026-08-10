#!/usr/bin/env bash
# Phase-B mechanism battery: WHAT binds the single-arm size frontier at the collapse point?
# At fixed collapsed scale s=0.20 (baseline ~12%), change exactly ONE physical knob and see which
# recovers success. Isolates: reach/workspace (H1) vs inertia/mass (H2) vs threshold artifact.
# Validated eval; all outputs under dywa_research/results/phase_b_mechanism/. Repo/output untouched.
set -uo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/bruce/DyWA/dywa_research/shims:/home/bruce/DyWA/dywa/exp/train:/home/bruce/DyWA/dywa/src:${PYTHONPATH:-}

RESEARCH=/home/bruce/DyWA/dywa_research
DATA=/home/bruce/DyWA/data
CKPT=/home/bruce/DyWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt
OUT=$RESEARCH/results/phase_b_mechanism
SCALE=0.20
NUM_ENV=64
EVAL_STEP=1200
mkdir -p "$OUT"
CSV="$OUT/mechanism_s020.csv"
echo "tag,knob,ws_xmax,mass,goal_radius,episode_count,success_count,suc_rate,balanced_suc_rate,status" > "$CSV"
cd /home/bruce/DyWA/dywa/exp/train

# intervention table: tag | knob | ws_xmax | mass(dr=default DR) | goal_radius
INTERVENTIONS=(
  "baseline|control|0.6|dr|0.05"
  "reach_ws0.9|reach|0.9|dr|0.05"
  "reach_ws1.2|reach|1.2|dr|0.05"
  "mass_0.05|inertia|0.6|0.05|0.05"
  "mass_0.30|inertia|0.6|0.30|0.05"
  "mass_0.80|inertia|0.6|0.80|0.05"
  "goal_0.10|threshold|0.6|dr|0.10"
)

run_one() {
  local tag="$1" ws="$2" mass="$3" goal="$4" cell="$OUT/$1"
  mkdir -p "$cell"
  local mass_args=""
  if [ "$mass" != "dr" ]; then
    mass_args="++env.single_object_scene.min_mass=$mass ++env.single_object_scene.max_mass=$mass"
  fi
  PYTORCH_JIT=0 python "$RESEARCH/valid_ppo_arm_patched.py" \
    +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
    ++name="phaseB_$tag" ++path.root="$cell" \
    ++env.num_env=$NUM_ENV ++global_device=cuda:0 ++env.trossen.base_height=table \
    "++env.trossen.ws_bound=[[-0.25,-0.35,-0.1],[$ws,0.35,0.4]]" \
    ++env.single_object_scene.dgn.data_path="$DATA/meta-v8" \
    ++env.single_object_scene.dgn.urdf_path="$RESEARCH/urdf_fixed" \
    ++env.single_object_scene.dgn.pose_path="$DATA/meta-v8/unique_dgn_poses" \
    ++env.single_object_scene.filter_file="$DATA/test_set.json" \
    ++env.single_object_scene.mode=valid \
    ++env.single_object_scene.min_scale=$SCALE ++env.single_object_scene.max_scale=$SCALE \
    $mass_args \
    ++env.task.goal_radius=$goal \
    ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
    ++load_ckpt="$CKPT" ++eval_track_per_obj_suc_rate=True ++eval_step=$EVAL_STEP \
    ++use_nvdr_record_episode=false ++plot_pc=0 > "$cell/cell.log" 2>&1
}

for row in "${INTERVENTIONS[@]}"; do
  IFS='|' read -r tag knob ws mass goal <<< "$row"
  echo "[battery] $tag (knob=$knob ws=$ws mass=$mass goal=$goal)"
  run_one "$tag" "$ws" "$mass" "$goal"; status=$?
  line=$(grep -aoE "\{'scalar': \{'valid/episode_count':[^}]*\}\}" "$OUT/$tag/cell.log" | tail -1)
  ep=$(echo "$line"  | grep -oE "episode_count': [0-9]+"      | grep -oE "[0-9]+" | tail -1)
  sc=$(echo "$line"  | grep -oE "success_count': [0-9]+"      | grep -oE "[0-9]+" | tail -1)
  sr=$(echo "$line"  | grep -oE "'valid/suc_rate': [0-9.]+"   | grep -oE "[0-9.]+" | tail -1)
  bsr=$(echo "$line" | grep -oE "balanced_suc_rate': [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  st="OK"; [ $status -ne 0 ] && st="FAILED"
  echo "$tag,$knob,$ws,$mass,$goal,${ep:-NA},${sc:-NA},${sr:-NA},${bsr:-NA},$st" >> "$CSV"
  echo "[battery] $tag -> suc_rate=${sr:-NA}"
done
echo "[battery] DONE"; cat "$CSV"
