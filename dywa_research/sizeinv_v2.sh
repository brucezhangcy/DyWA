#!/usr/bin/env bash
# S5 step 2 (GPU, hours): size-invariant table v2. Fixes the two confounds found in Session 4
# (CRITIQUES #14, analysis/s4_geometry_audit.py) in the v1 test (sizeinv_table.sh):
#   v1 scaled table_dims with s, which ALSO scaled the spawn region (0.475*table, valid mode has no radius
#   contraction: scene.py:1365-1377) and the goal region (0.4*table, push_task.py:684-692) -> mean
#   spawn-goal distance 0.238 -> 0.419 m and 14-22% of goals/spawns beyond the arm's 0.768 m reach at s=0.30,
#   while the robot base stays welded at x=-0.5 (trossen.py:206-221).
# v2: ONE large constant table T* for every scale, and margin_scales shrunk so spawn and goal regions
#   equal the fixed-table baseline (+-0.19 x +-0.2375 spawn, +-0.16 x +-0.20 goal). Only object size varies.
#   T* = [0.821, 1.026] keeps the 0.4:0.5 aspect so one factor (0.4/0.821) fixes both axes exactly.
# Controls: (a) s=0.10 on T* (table size alone should not matter -> expect ~0.93);
#           (b) s=0.10 on the v1 s=0.30 table with DEFAULT margins (grown regions) -> measures how much of
#               the v1 collapse is spawn/goal distance + reach rather than object size.
# Default: 8 scales x 3 seeds (control a = v2_s0.10) + control b x 3 seeds = 27 cells, eval_step=1200. Override SCALES / SEEDS.
# Run (detached):  conda activate dywa_render && setsid bash dywa_research/sizeinv_v2.sh </dev/null &
source "$(dirname "$0")/s5_common.sh"
s5_preflight || exit 1
OUT=${V2_OUT:-$R/results/phase_e_sizeinv_v2}; mkdir -p "$OUT"   # set V2_OUT (+CKPT) to eval another ckpt
CSV=$OUT/sizeinv_v2.csv
[ -f "$CSV" ] || echo "tag,scale,seed,term_done,frac_success,frac_oob,frac_timeout,frac_immediate,feasible_succ,balanced,status" > "$CSV"
TSTAR="[0.821,1.026,0.4]"
MS_SCENE=$(python3 -c "print(round(0.95*0.4/0.821,5))")   # 0.46285
MS_TASK=$(python3 -c "print(round(0.8*0.4/0.821,5))")     # 0.38977
V2="++env.single_object_scene.table_dims=$TSTAR ++env.single_object_scene.margin_scale=$MS_SCENE ++env.task.margin_scale=$MS_TASK"
SCALES=${SCALES:-"0.05 0.10 0.125 0.15 0.175 0.20 0.25 0.30"}
SEEDS=${SEEDS:-"0 1 2"}
run() {  # tag scale seed extra...
  local tag=$1 scale=$2 seed=$3; shift 3
  local cell="$OUT/${tag}_seed${seed}"
  if grep -q "^$tag,$scale,$seed,[0-9]" "$CSV"; then echo "[v2] skip done $tag seed$seed"; return; fi
  echo "[v2] $tag seed=$seed $(date -u +%T)"
  s5_eval_cell "$cell" "s5v2_${tag}_seed${seed}" "$scale" "$seed" 1200 "$@"; local st=$?
  s5_row "$cell/cell.log" "$tag,$scale,$seed" "$CSV" "$([ $st -eq 0 ] && echo OK || echo FAILED)"
  s5_check_resolved "$cell/cell.log" "table_dims" "margin_scale"
}
for seed in $SEEDS; do
  for s in $SCALES; do run "v2_s$s" "$s" "$seed" $V2; done
  run "ctrlB_s0.10_v1table0.30" 0.10 "$seed" "++env.single_object_scene.table_dims=[0.821,0.921,0.4]"
done
echo "[v2] DONE $(date -u +%T)"; cat "$CSV"
echo "next (CPU): python3 $R/analysis/s4_frontier_stats.py-style fit on $CSV; diag: python3 $R/analysis/s5_diag_report.py $OUT/*/diag.npz"
