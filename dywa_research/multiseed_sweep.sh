#!/usr/bin/env bash
# #4 Multi-seed frontier for error bars. Uses the taxonomy variant so each run yields raw success,
# termination causes, and feasible-only in one pass. Scales span the transition/collapse region.
set -uo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/bruce/DyWA/dywa_research/shims:/home/bruce/DyWA/dywa/exp/train:/home/bruce/DyWA/dywa/src:${PYTHONPATH:-}
RESEARCH=/home/bruce/DyWA/dywa_research
DATA=/home/bruce/DyWA/data
CKPT=/home/bruce/DyWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt
OUT=$RESEARCH/results/phase_c_multiseed
mkdir -p "$OUT"
CSV="$OUT/multiseed.csv"
echo "scale,seed,term_done,frac_success,frac_oob,frac_timeout,frac_immediate,feasible_succ,status" > "$CSV"
cd /home/bruce/DyWA/dywa/exp/train

for scale in 0.10 0.15 0.20 0.30; do
 for seed in 0 1 2; do
  cell="$OUT/s${scale}_seed${seed}"; mkdir -p "$cell"
  echo "[ms] scale=$scale seed=$seed"
  PYTORCH_JIT=0 python "$RESEARCH/valid_ppo_arm_taxonomy.py" \
    +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
    ++name="phaseC_s${scale}_seed${seed}" ++path.root="$cell" \
    ++env.num_env=64 ++env.seed=$seed ++global_device=cuda:0 ++env.trossen.base_height=table \
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
  python3 - "$cell/cell.log" "$scale" "$seed" "$st" "$CSV" <<'PY'
import sys, re, ast
log, scale, seed, st, csv = sys.argv[1:6]
txt = re.sub(r'\x1b\[[0-9;]*m', '', open(log, errors='ignore').read())
m = re.findall(r"TAXONOMY (\{.*\})", txt)
if m:
    d = ast.literal_eval(m[-1]); done = d['term_done']; im = round(d['frac_immediate']*done)
    sc = round(d['frac_success']*done); feas = sc/(done-im) if (done-im) > 0 else float('nan')
    row = f"{scale},{seed},{done},{d['frac_success']},{d['frac_oob']},{d['frac_timeout']},{d['frac_immediate']},{feas:.4f},{st}\n"
else:
    row = f"{scale},{seed},NA,NA,NA,NA,NA,NA,{st}\n"
open(csv, 'a').write(row); print("[ms]", scale, seed, "->", m[-1] if m else "NO TAX")
PY
 done
done
echo "[ms] DONE"; cat "$CSV"
