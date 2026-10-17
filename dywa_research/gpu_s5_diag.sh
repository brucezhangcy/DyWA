#!/usr/bin/env bash
# S5 step 1 (GPU, short): test the Session-4 code-reading hypotheses (CRITIQUES #13/#14) before
# spending hours on new sweeps. Re-runs four EXISTING cells (seed 0, same args as multiseed/sizeinv) through
# valid_ppo_arm_diag.py, which adds per-episode start/goal/outcome + spawn-z / robot-root dumps.
# Sanity: the fixed-table numbers should reproduce results/phase_c_multiseed (seed 0) and
# results/phase_d_sizeinv; if they do not, the diag wrapper is NOT behaviour-neutral and must be fixed.
# Run (detached):  conda activate dywa_render && setsid bash dywa_research/gpu_s5_diag.sh </dev/null &
source "$(dirname "$0")/s5_common.sh"
s5_preflight || exit 1
OUT=$R/results/s5_diag; mkdir -p "$OUT"
CSV=$OUT/diag_cells.csv
echo "tag,scale,seed,term_done,frac_success,frac_oob,frac_timeout,frac_immediate,feasible_succ,balanced,status" > "$CSV"
# tag | scale | extra overrides
CELLS=(
  "fixed_s0.10|0.10|"
  "fixed_s0.20|0.20|"
  "fixed_s0.30|0.30|"
  "sizeinv_v1_s0.30|0.30|++env.single_object_scene.table_dims=[0.821,0.921,0.4]"
)
for c in "${CELLS[@]}"; do
  IFS='|' read -r tag scale extra <<< "$c"
  echo "[s5diag] $tag $(date -u +%T)"
  s5_eval_cell "$OUT/$tag" "s5diag_$tag" "$scale" 0 1200 $extra; st=$?
  s5_row "$OUT/$tag/cell.log" "$tag,$scale,0" "$CSV" "$([ $st -eq 0 ] && echo OK || echo FAILED)"
  s5_check_resolved "$OUT/$tag/cell.log" "table_dims" "goal_radius" "margin_scale"
done
python3 "$R/analysis/s5_diag_report.py" "$OUT"/*/diag.npz | tee "$OUT/diag_report.txt"
echo "[s5diag] DONE $(date -u +%T)"; cat "$CSV"
