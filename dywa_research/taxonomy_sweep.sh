#!/usr/bin/env bash
# Phase-B failure-mode taxonomy vs scale: HOW do episodes fail as objects grow?
# success / oob (lost object) / timeout (tried, couldn't) / immediate (<=3 steps ~ infeasible spawn).
# Distinguishes a genuine control limit (timeout/oob after real attempts) from infeasible setups (#3)
# and gives evidence on OOD-flailing vs trying (#1). force_rad=0.10 to match Deliverable A.
set -uo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/bruce/DyWA/dywa_research/shims:/home/bruce/DyWA/dywa/exp/train:/home/bruce/DyWA/dywa/src:${PYTHONPATH:-}
RESEARCH=/home/bruce/DyWA/dywa_research
DATA=/home/bruce/DyWA/data
CKPT=/home/bruce/DyWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt
OUT=$RESEARCH/results/phase_b_taxonomy
mkdir -p "$OUT"
CSV="$OUT/taxonomy.csv"
echo "scale,term_done,frac_success,frac_oob,frac_timeout,frac_immediate,mean_eplen,status" > "$CSV"
cd /home/bruce/DyWA/dywa/exp/train

for scale in 0.05 0.10 0.15 0.20 0.30; do
  cell="$OUT/s${scale}"; mkdir -p "$cell"
  echo "[tax] scale=$scale"
  PYTORCH_JIT=0 python "$RESEARCH/valid_ppo_arm_taxonomy.py" \
    +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
    ++name="phaseBtax_s${scale}" ++path.root="$cell" \
    ++env.num_env=64 ++global_device=cuda:0 ++env.trossen.base_height=table \
    "++env.trossen.ws_bound=[[-0.25,-0.35,-0.1],[0.6,0.35,0.4]]" \
    ++env.single_object_scene.dgn.data_path="$DATA/meta-v8" \
    ++env.single_object_scene.dgn.urdf_path="$RESEARCH/urdf_fixed" \
    ++env.single_object_scene.dgn.pose_path="$DATA/meta-v8/unique_dgn_poses" \
    ++env.single_object_scene.filter_file="$DATA/test_set.json" \
    ++env.single_object_scene.mode=valid \
    ++env.single_object_scene.min_scale=$scale ++env.single_object_scene.max_scale=$scale \
    ++force_rad=0.10 \
    ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
    ++load_ckpt="$CKPT" ++eval_track_per_obj_suc_rate=True ++eval_step=1200 \
    ++use_nvdr_record_episode=false ++plot_pc=0 > "$cell/cell.log" 2>&1
  status=$?; st="OK"; [ $status -ne 0 ] && st="FAILED"
  python3 - "$cell/cell.log" "$scale" "$st" "$CSV" <<'PY'
import sys, re, ast
log, scale, st, csv = sys.argv[1:5]
txt = open(log, errors='ignore').read()
txt = re.sub(r'\x1b\[[0-9;]*m', '', txt)
m = re.findall(r"TAXONOMY (\{.*\})", txt)
row = f"{scale},NA,NA,NA,NA,NA,NA,{st}\n"
if m:
    d = ast.literal_eval(m[-1])
    row = (f"{scale},{d['term_done']},{d['frac_success']},{d['frac_oob']},"
           f"{d['frac_timeout']},{d['frac_immediate']},{d['term_mean_eplen']},{st}\n")
open(csv, 'a').write(row)
print("[tax] scale", scale, "->", m[-1] if m else "NO TAXONOMY")
PY
done
echo "[tax] DONE"; cat "$CSV"
