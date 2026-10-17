#!/usr/bin/env bash
# S5 step 4 (GPU): get a TRAINED bimanual number for Deliverable E (mjlab, cross-sim). The scripted lower
# bound is 0.142 macro [0.109, 0.172] (results/session4/deliverable_E_summary.json).
# Written in Session 4 WITHOUT access to Aloha_real. The eval CLI below is the one that was run in Session 3
# (bimanual_eval.sh: scripts/eval_residual_push_multi.py --scales_manifest --num_envs
# --episodes_per_object [--scripted_only | --checkpoint]). The TRAINING command is NOT known here: it lives
# in Aloha_real docs/runbooks/cat_box_subset_10_residual_push.md. This script prints the runbook's training
# section and refuses to guess. Set TRAIN_CMD to the runbook's command (run from $REPO) to train.
#   bash gpu_bimanual_residual.sh                          # stage 0+1: look for a ckpt, print the runbook train cmd
#   TRAIN_CMD='...' setsid bash gpu_bimanual_residual.sh </dev/null &     # train (hours), then eval
#   BI_CKPT=/path/model_XXXX.pt bash gpu_bimanual_residual.sh            # eval an existing ckpt only
# Eval = the SAME 30-variant manifest as the scripted run (bimanual_scales.yaml: apples-to-apples with the
# lower bound) AND the matched-radius grid (bimanual_scales_radius_grid.yaml: r = 0.10..0.30 like the
# single-arm sweep), both scripted_only and checkpoint, so scripted vs trained is a paired comparison.
set -uo pipefail; shopt -s globstar nullglob
R=/home/bruce/DyWA/dywa_research
VP=$R/mjlab_venv
REPO=/home/bruce/Aloha_real/sim_3d_bimanual-main
RUNBOOK=${RUNBOOK:-$(ls /home/bruce/Aloha_real/docs/runbooks/cat_box_subset_10_residual_push.md "$REPO"/docs/runbooks/cat_box_subset_10_residual_push.md 2>/dev/null | head -1)}
OUT=$R/results/bimanual_trained; mkdir -p "$OUT"
NUM_ENVS=${NUM_ENVS:-32}; EPPO=${EPPO:-30}

# ---- stage 0: preflight (Session-3 env state; see progress log 2026-08-21)
"$VP/bin/python" -c "import mujoco, warp, mujoco_warp; print('mujoco', mujoco.__version__, 'warp', warp.__version__)" || exit 1
[ -e "${TROSSEN_OBJECTS_DIR:-/nonexistent}" ] || echo "[bi] NOTE: TROSSEN_OBJECTS_DIR unset; Session 3 used a symlink objects -> objects_cat_box_obj"

# ---- stage 1: find a checkpoint (the residual policy model_3300.pt was lost; maybe a copy survived)
if [ -z "${BI_CKPT:-}" ]; then
  echo "[bi] searching for residual checkpoints (model_*.pt) under /home/bruce ..."
  find /home/bruce -name 'model_*.pt' -path '*resid*' 2>/dev/null | head -20 | tee "$OUT/found_ckpts.txt"
  find /home/bruce -name 'model_3300.pt' 2>/dev/null | head -5 | tee -a "$OUT/found_ckpts.txt"
fi

# ---- stage 2: train (only with an explicit TRAIN_CMD copied from the runbook)
if [ -z "${BI_CKPT:-}" ] && [ -z "${TRAIN_CMD:-}" ]; then
  echo "[bi] No BI_CKPT and no TRAIN_CMD. Runbook: ${RUNBOOK:-NOT FOUND}"
  [ -n "${RUNBOOK:-}" ] && grep -n -i -E "train|rsl|ppo|residual|--task|num_envs|max_iter" "$RUNBOOK" | head -60
  echo "[bi] Copy the runbook's training command into TRAIN_CMD (and BI_CKPT_GLOB for its output), re-run."
  exit 2
fi
if [ -z "${BI_CKPT:-}" ]; then
  echo "[bi] TRAIN: $TRAIN_CMD  ($(date -u +%FT%TZ))" | tee "$OUT/train.info"
  ( cd "$REPO" && PYTHONPATH=src bash -c "$TRAIN_CMD" ) > "$OUT/train.log" 2>&1
  echo "[bi] train exit=$? $(date -u +%FT%TZ)" | tee -a "$OUT/train.info"
  BI_CKPT=$(ls -t ${BI_CKPT_GLOB:-$REPO/logs/**/model_*.pt} 2>/dev/null | head -1)
  [ -n "$BI_CKPT" ] || { echo "[bi] no checkpoint found after training; set BI_CKPT_GLOB"; exit 3; }
fi
echo "[bi] using checkpoint $BI_CKPT" | tee "$OUT/ckpt.info"

# ---- stage 3: paired eval (scripted vs trained) on both manifests
cd "$REPO"
for man in bimanual_scales bimanual_scales_radius_grid; do
  PYTHONPATH=src "$VP/bin/python" scripts/eval_residual_push_multi.py --scripted_only \
    --scales_manifest "$R/$man.yaml" --num_envs "$NUM_ENVS" --episodes_per_object "$EPPO" > "$OUT/${man}_scripted.log" 2>&1
  PYTHONPATH=src "$VP/bin/python" scripts/eval_residual_push_multi.py --checkpoint "$BI_CKPT" \
    --scales_manifest "$R/$man.yaml" --num_envs "$NUM_ENVS" --episodes_per_object "$EPPO" > "$OUT/${man}_trained.log" 2>&1
  echo "[bi] $man done: tail of trained log:"; tail -15 "$OUT/${man}_trained.log"
done
echo "[bi] NEXT (CPU): parse per-object success from the *_trained.log files into results/bimanual_trained/*.csv"
echo "      (same schema as results/bimanual/bimanual_results.csv) and re-run analysis/s4_deliverable_e.py on it."
