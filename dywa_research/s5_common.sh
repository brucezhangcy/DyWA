#!/usr/bin/env bash
# Shared helpers for the Session-5 GPU queue (written in Session 4 on a CPU-only machine: NOT yet run).
# The eval command line is copied verbatim from the validated multiseed_sweep.sh / sizeinv_table.sh; the only
# additions are optional extra hydra overrides ($EXTRA) and the diag entry (valid_ppo_arm_diag.py).
# Source me:  source /home/bruce/DyWA/dywa_research/s5_common.sh
set -uo pipefail
R=${R:-/home/bruce/DyWA/dywa_research}
DYWA=${DYWA:-/home/bruce/DyWA}
DATA=$DYWA/data
CKPT=${CKPT:-$DYWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt}
export LD_LIBRARY_PATH=${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$R/shims:$DYWA/dywa/exp/train:$DYWA/dywa/src:${PYTHONPATH:-}

s5_preflight() {
  [ "${CONDA_DEFAULT_ENV:-}" = "dywa_render" ] || { echo "activate conda env dywa_render first"; return 1; }
  [ -d "$R/urdf_fixed" ] || { echo "missing $R/urdf_fixed (see progress log Session 2 BUG#2)"; return 1; }
  [ -f "$CKPT/last.ckpt" ] || { echo "missing teacher ckpt $CKPT/last.ckpt"; return 1; }
  [ -f "$R/valid_ppo_arm_diag.py" ] || python3 "$R/patch_diag.py"
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
}

# s5_eval_cell <cell_dir> <name> <scale> <seed> <eval_step> [extra hydra overrides...]
s5_eval_cell() {
  local cell=$1 name=$2 scale=$3 seed=$4 steps=$5; shift 5
  mkdir -p "$cell"
  ( cd "$DYWA/dywa/exp/train" && DIAG_OUT="$cell/diag.npz" PYTORCH_JIT=0 python "$R/valid_ppo_arm_diag.py" \
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
  [ -f /tmp/per-cat-suc.pkl ] && cp /tmp/per-cat-suc.pkl "$cell/per_obj.pkl"
  return $st
}

# s5_row <cell_log> <prefix-cols> <csv> : append TAXONOMY + feasible-only row (same schema as multiseed.csv tail)
s5_row() {
  python3 - "$1" "$2" "$3" "${4:-OK}" <<'PY'
import sys, re, ast
log, prefix, csv, st = sys.argv[1:5]
txt = re.sub(r'\x1b\[[0-9;]*m', '', open(log, errors='ignore').read())
m = re.findall(r"TAXONOMY (\{.*\})", txt)
bal = re.findall(r"'valid/balanced_suc_rate': ([0-9.eE-]+)", txt)
if m:
    d = ast.literal_eval(m[-1]); n = d['term_done']; im = d['term_immediate_le3']; k = d['term_success']
    fe = k / (n - im) if n > im else float('nan')
    row = f"{prefix},{n},{d['frac_success']},{d['frac_oob']},{d['frac_timeout']},{d['frac_immediate']},{fe:.4f},{bal[-1] if bal else 'NA'},{st}\n"
else:
    row = f"{prefix},NA,NA,NA,NA,NA,NA,NA,FAILED\n"
open(csv, 'a').write(row); print(row.strip())
PY
}

# Rule #12 (CRITIQUES): verify the RESOLVED config, not the CLI string. Prints matching resolved keys.
s5_check_resolved() {
  local log=$1; shift
  for key in "$@"; do
    if grep -q "$key" "$log"; then grep -m3 "$key" "$log" | sed "s/^/[resolved] /"; else echo "[resolved] WARNING: '$key' not found in $log"; fi
  done
}
