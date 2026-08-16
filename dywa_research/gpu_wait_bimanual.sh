#!/usr/bin/env bash
# Wait for the GPU to free (another user's job at 100%/22GB), then auto-run the bimanual eval.
# Polls every 5 min; launches when <8 GB used; times out after 48h. Detached-friendly.
set -uo pipefail
OUT=/home/bruce/DyWA/dywa_research/results/bimanual
mkdir -p "$OUT"
STATUS="$OUT/watcher.status"
echo "armed $(date -u +%FT%TZ)" > "$STATUS"
for i in $(seq 1 576); do   # 576 * 5min = 48h
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$(date -u +%FT%TZ) used=${used}MiB" >> "$STATUS"
  if [ "${used:-99999}" -lt 8000 ]; then
    echo "GPU free (used=${used}) -> launching bimanual eval $(date -u +%FT%TZ)" >> "$STATUS"
    NUM_ENVS=32 EPPO=30 bash /home/bruce/DyWA/dywa_research/bimanual_eval.sh > "$OUT/eval.log" 2>&1
    echo "eval exit=$? $(date -u +%FT%TZ)" >> "$STATUS"
    exit 0
  fi
  sleep 300
done
echo "timed out after 48h $(date -u +%FT%TZ)" >> "$STATUS"
