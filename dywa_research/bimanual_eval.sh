#!/usr/bin/env bash
# Bimanual (2-arm Trossen) non-prehensile PUSH eval on the box objects — scripted primitive baseline.
# Reports per-object success. Compare vs the DyWA single-arm frontier (cross-sim, illustrative).
# Env: dywa_research/mjlab_venv (torch 2.11+cu128, mujoco 3.11.1, mujoco-warp 3.11.0).
set -uo pipefail
VP=/home/bruce/DyWA/dywa_research/mjlab_venv
REPO=/home/bruce/Aloha_real/sim_3d_bimanual-main
NUM_ENVS="${NUM_ENVS:-32}"; EPPO="${EPPO:-30}"
BOXES=$(sed 's#^#objects_cat_box_obj/#; s#^objects_cat_box_obj/objects_cat_box_obj/#objects_cat_box_obj/#' "$REPO/scripts/object_list_generating/objects_cat_box_obj_subset_10.txt" | tr '\n' ' ')
cd "$REPO"
PYTHONPATH=src "$VP/bin/python" scripts/eval_residual_push_multi.py \
  --scripted_only \
  --object_names $BOXES \
  --num_envs "$NUM_ENVS" --episodes_per_object "$EPPO"
