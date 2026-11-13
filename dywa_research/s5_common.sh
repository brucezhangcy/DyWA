#!/usr/bin/env bash
# Shared helpers for the Session-5 GPU queue (written in Session 4 on a CPU-only machine: NOT yet run).
# The eval command line is copied verbatim from the validated multiseed_sweep.sh / sizeinv_table.sh; the only
# additions are optional extra hydra overrides ($EXTRA) and the diag entry (valid_ppo_arm_diag.py).
# Source me:  source /home/bruce/DyWA/dywa_research/s5_common.sh
set -uo pipefail
R=${R:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
DYWA=${DYWA:-$(cd "$R/.." && pwd)}
DATA=$DYWA/data
CKPT=${CKPT:-$DYWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt}
export LD_LIBRARY_PATH=${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$R/shims:$DYWA/dywa/exp/train:$DYWA/dywa/src:${PYTHONPATH:-}

s5_preflight() {
  source "$R/gpu_permission.sh" || return $?
  [ "${CONDA_DEFAULT_ENV:-}" = "dywa_render" ] || { echo "activate conda env dywa_render first"; return 1; }
  [ -d "$R/urdf_fixed" ] || { echo "missing $R/urdf_fixed (see progress log Session 2 BUG#2)"; return 1; }
  [ -f "$CKPT/last.ckpt" ] || { echo "missing teacher ckpt $CKPT/last.ckpt"; return 1; }
  python3 "$R/patch_diag.py" || return 1
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
}

# s5_eval_cell <cell_dir> <name> <scale> <seed> <eval_step> [extra hydra overrides...]
s5_eval_cell() {
  local cell=$1 name=$2 scale=$3 seed=$4 steps=$5; shift 5
  mkdir -p "$cell"
  # Preserve previous attempts; never let a stale NPZ or pickle validate a retry.
  local attempt
  attempt=$(mktemp -d "$cell/attempt.XXXXXX") || return 1
  local artifact
  for artifact in cell.log diag.npz per_obj.pkl; do
    [ ! -f "$cell/$artifact" ] || mv "$cell/$artifact" "$attempt/$artifact"
  done
  ( cd "$DYWA/dywa/exp/train" && DIAG_OUT="$cell/diag.npz" PER_OBJECT_OUT="$cell/per_obj.pkl" PYTORCH_JIT=0 python "$R/valid_ppo_arm_diag.py" \
    +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
    ++name="$name" ++path.root="$cell" \
    ++env.num_env=64 ++env.seed=$seed ++global_device=cuda:0 ++env.trossen.base_height=table \
    "++env.trossen.ws_bound=[[-0.25,-0.35,-0.1],[0.6,0.35,0.4]]" \
    ++env.single_object_scene.dgn.data_path="$DATA/meta-v8" \
    ++env.single_object_scene.dgn.urdf_path="$R/urdf_fixed" \
    ++env.single_object_scene.dgn.pose_path="$DATA/meta-v8/unique_dgn_poses" \
    ++env.single_object_scene.filter_file="$DATA/test_set.json" \
    ++env.single_object_scene.mode=valid \
    ++env.single_object_scene.min_scale=$scale ++env.single_object_scene.max_scale=$scale \
    ++force_rad=0.10 \
    ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
    ++load_ckpt="$CKPT" ++eval_track_per_obj_suc_rate=True ++eval_step=$steps \
    ++use_nvdr_record_episode=false ++plot_pc=0 "$@" > "$cell/cell.log" 2>&1 )
  local st=$?
  return $st
}

# s5_row <cell_log> <prefix-cols> <csv> : append TAXONOMY + feasible-only row (same schema as multiseed.csv tail)
s5_row() {
  python3 "$R/analysis/s5_validate_cell.py" "$1" "$2" "$3" "${4:-OK}"

}

# Presence check only: print resolved-key context. This does not prove expected values.
# Review the actual values against the approved protocol before interpreting a run.
s5_check_resolved() {
  local log=$1; shift
  for key in "$@"; do
    if grep -q "$key" "$log"; then grep -m3 "$key" "$log" | sed "s/^/[resolved] /"; else echo "[resolved] ERROR: '$key' not found in $log" >&2; return 1; fi
  done
}
