#!/usr/bin/env bash
# DECISIVE table-confound test: scale the table WITH the object so the usable placement/OOB region
# stays constant (= the s=0.10, table [0.4,0.5] baseline), isolating single-arm CONTROL from the
# fixed-table placement-shrink artifact. If the frontier still collapses -> real control limit.
# table_dim(s) = (usable_half_baseline + s)/(0.5*margin_scale=0.475), per axis. Pure config override.
set -uo pipefail
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/bruce/DyWA/dywa_research/shims:/home/bruce/DyWA/dywa/exp/train:/home/bruce/DyWA/dywa/src:${PYTHONPATH:-}
R=/home/bruce/DyWA/dywa_research; DATA=/home/bruce/DyWA/data
CKPT=/home/bruce/DyWA/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt
OUT=$R/results/phase_d_sizeinv; mkdir -p "$OUT"
CSV="$OUT/sizeinv.csv"; echo "scale,table_x,table_y,term_done,frac_success,frac_oob,frac_timeout,frac_immediate,feasible_succ,status" > "$CSV"
cd /home/bruce/DyWA/dywa/exp/train
for scale in 0.10 0.15 0.20 0.25 0.30; do
  # keep usable_half constant at baseline (x:0.09, y:0.1375); table = (base+s)/0.475
  tx=$(python3 -c "print(round((0.09+$scale)/0.475,3))")
  ty=$(python3 -c "print(round((0.1375+$scale)/0.475,3))")
  cell="$OUT/s${scale}"; mkdir -p "$cell"
  echo "[sizeinv] scale=$scale table=[$tx,$ty]"
  PYTORCH_JIT=0 python "$R/valid_ppo_arm_taxonomy.py" \
    +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
    ++name="phaseD_s${scale}" ++path.root="$cell" \
    ++env.num_env=64 ++env.seed=0 ++global_device=cuda:0 ++env.trossen.base_height=table \
    "++env.trossen.ws_bound=[[-0.25,-0.35,-0.1],[0.6,0.35,0.4]]" \
    "++env.single_object_scene.table_dims=[$tx,$ty,0.4]" \
    ++env.single_object_scene.dgn.data_path="$DATA/meta-v8" \
    ++env.single_object_scene.dgn.urdf_path="$R/urdf_fixed" \
    ++env.single_object_scene.dgn.pose_path="$DATA/meta-v8/unique_dgn_poses" \
    ++env.single_object_scene.filter_file="$DATA/test_set.json" \
    ++env.single_object_scene.mode=valid \
    ++env.single_object_scene.min_scale=$scale ++env.single_object_scene.max_scale=$scale \
    ++force_rad=0.10 \
    ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
    ++load_ckpt="$CKPT" ++eval_track_per_obj_suc_rate=True ++eval_step=1200 \
    ++use_nvdr_record_episode=false ++plot_pc=0 > "$cell/cell.log" 2>&1
  status=$?; st="OK"; [ $status -ne 0 ] && st="FAILED"
  python3 - "$cell/cell.log" "$scale" "$tx" "$ty" "$st" "$CSV" <<'PY'
import sys, re, ast
log, scale, tx, ty, st, csv = sys.argv[1:7]
txt = re.sub(r'\x1b\[[0-9;]*m','',open(log,errors='ignore').read())
m = re.findall(r"TAXONOMY (\{.*\})", txt)
if m:
    d=ast.literal_eval(m[-1]); done=d['term_done']; im=round(d['frac_immediate']*done)
    sc=round(d['frac_success']*done); fe=sc/(done-im) if (done-im)>0 else float('nan')
    open(csv,'a').write(f"{scale},{tx},{ty},{done},{d['frac_success']},{d['frac_oob']},{d['frac_timeout']},{d['frac_immediate']},{fe:.4f},{st}\n")
else:
    open(csv,'a').write(f"{scale},{tx},{ty},NA,NA,NA,NA,NA,NA,{st}\n")
print("[sizeinv]",scale,"->",m[-1] if m else "NO TAX")
PY
done
echo "[sizeinv] DONE"; cat "$CSV"
