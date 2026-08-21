#!/usr/bin/env bash
# Bimanual (2-arm Trossen) non-prehensile PUSH eval — scripted primitive baseline on the box objects.
# Reports per-object success; compare vs the DyWA single-arm frontier (cross-sim, illustrative).
# Env: dywa_research/mjlab_venv. Objects/scales from bimanual_scales.yaml (10 boxes x scales 0.7/1.0/1.3).
set -uo pipefail
VP=/home/bruce/DyWA/dywa_research/mjlab_venv
REPO=/home/bruce/Aloha_real/sim_3d_bimanual-main
NUM_ENVS="${NUM_ENVS:-32}"
EPPO="${EPPO:-30}"
MANIFEST="${MANIFEST:-/home/bruce/DyWA/dywa_research/bimanual_scales.yaml}"
cd "$REPO"
PYTHONPATH=src "$VP/bin/python" scripts/eval_residual_push_multi.py \
  --scripted_only \
  --scales_manifest "$MANIFEST" \
  --num_envs "$NUM_ENVS" --episodes_per_object "$EPPO"
