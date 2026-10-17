#!/usr/bin/env bash
# ONE COMMAND for the Session-5 single-arm GPU queue on coldbrew (RTX 4090 confirmed free 2026-10-17).
# Written in Session 4 on a CPU-only machine: NOT yet run. Each stage stops the queue if it fails.
#   cd /home/bruce/DyWA && git fetch && git checkout cloud-session-4   (or main after the PR merges)
#   conda activate dywa_render && setsid bash dywa_research/gpu_s5_queue.sh > dywa_research/results/s5_queue.log 2>&1 < /dev/null &
#   tail -f dywa_research/results/s5_queue.log
# Stages: 1 diag (4 cells, ~1 h?)  2 size-inv v2 (27 cells)  3 necessity smoke  4 necessity fine-tune
#         (~8 h / 100k iters)  5 eval the fine-tuned ckpt on v2.  Skip stages with STAGES="1 2" etc.
# The bimanual residual retrain is separate (needs the runbook's train command): gpu_bimanual_residual.sh
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
STAGES=${STAGES:-"1 2 3 4 5"}
has() { [[ " $STAGES " == *" $1 "* ]]; }
stamp() { echo "===== [queue] $* $(date -u +%FT%TZ)"; }
has 1 && { stamp "stage 1 diag";      bash "$D/gpu_s5_diag.sh"            || { stamp "diag FAILED"; exit 1; }; }
has 2 && { stamp "stage 2 v2 sweep";  bash "$D/sizeinv_v2.sh"             || { stamp "v2 FAILED"; exit 1; }; }
has 3 && { stamp "stage 3 nec smoke"; SMOKE=1 bash "$D/gpu_necessity_finetune.sh" || { stamp "smoke FAILED"; exit 1; }; }
has 4 && { stamp "stage 4 nec train"; bash "$D/gpu_necessity_finetune.sh" || { stamp "train FAILED"; exit 1; }; }
if has 5; then
  stamp "stage 5 eval fine-tuned"
  NC=$(ls -d "$D"/results/necessity/largeDR_s0.30/run-*/ckpt "$D"/results/necessity/largeDR_s0.30/ckpt 2>/dev/null | head -1)
  [ -n "$NC" ] || { stamp "no fine-tuned ckpt dir found under results/necessity/largeDR_s0.30"; exit 1; }
  CKPT=$NC V2_OUT=$D/results/necessity/largeDR_s0.30/eval_v2 SCALES="0.10 0.15 0.175 0.20 0.25 0.30" bash "$D/sizeinv_v2.sh"
fi
stamp "queue DONE"
