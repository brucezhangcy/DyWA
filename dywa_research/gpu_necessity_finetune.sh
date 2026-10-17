#!/usr/bin/env bash
# S5 step 3 (GPU, ~8 h per 100k PPO iters at the dev_log's ~3.7 it/s): the NECESSITY experiment
# (CRITIQUES #1). Fine-tunes the single-arm Trossen teacher WITH large objects in the scale DR on the
# v2 environment (constant big table T*, baseline spawn/goal regions; see sizeinv_v2.sh). If the
# fine-tuned teacher still collapses above s*, the limit is not just train/test scale shift.
# Training CLI is from dev_log.md (2026-05-17: train_ppo_arm.py +env=trossen_icra_base
# +run=trossen_icra_ours_abs_rel ++env.trossen.base_height=table ++agent.train.train_steps=N;
# 2026-05-19: widened ws_bound). ++load_ckpt initialises the agent (train_ppo_arm.py load_agent).
# Train mode honours scene.prevent_fall (scene.py:1009-1013); we set it false so spawn = the same
# +-0.19 x +-0.2375 region as valid mode (scene.py:1369-1376). UNTESTED: run SMOKE=1 first.
#   SMOKE=1 bash gpu_necessity_finetune.sh                  # 64 envs, 2k iters: does it start and log?
#   setsid bash gpu_necessity_finetune.sh </dev/null &      # SMAX=0.30 (large-object DR) main run
#   SMAX=0.10 TAG=ctrl_smallDR setsid bash gpu_necessity_finetune.sh </dev/null &   # same training, in-dist DR
#     (the control separates "more training on the v2 table" from "exposure to large objects")
# Then evaluate the fine-tuned ckpt on the SAME v2 env (the ckpt dir layout under path.root may be
#   <OUT>/run-000/ckpt -- check with ls):
#   CKPT=$R/results/necessity/$TAG/run-000/ckpt V2_OUT=$R/results/necessity/$TAG/eval_v2 bash sizeinv_v2.sh
source "$(dirname "$0")/s5_common.sh"
s5_preflight || exit 1
SMAX=${SMAX:-0.30}; TAG=${TAG:-largeDR_s${SMAX}}
STEPS=${STEPS:-100000}; NUM_ENV=${NUM_ENV:-1024}
if [ "${SMOKE:-0}" = "1" ]; then STEPS=2048; NUM_ENV=64; TAG=smoke_$TAG; fi
OUT=$R/results/necessity/$TAG; mkdir -p "$OUT"
TSTAR="[0.821,1.026,0.4]"
MS_SCENE=$(python3 -c "print(round(0.95*0.4/0.821,5))"); MS_TASK=$(python3 -c "print(round(0.8*0.4/0.821,5))")
echo "[nec] $TAG steps=$STEPS envs=$NUM_ENV scale DR [0.05,$SMAX] init=$CKPT $(date -u +%FT%TZ)" | tee "$OUT/run.info"
cd "$DYWA/dywa/exp/train"
PYTORCH_JIT=0 python train_ppo_arm.py \
  +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
  ++name="necessity_$TAG" ++path.root="$OUT" \
  ++env.num_env=$NUM_ENV ++global_device=cuda:0 ++env.trossen.base_height=table \
  "++env.trossen.ws_bound=[[-0.25,-0.35,-0.1],[0.6,0.35,0.4]]" \
  ++env.single_object_scene.dgn.data_path="$DATA/meta-v8" \
  ++env.single_object_scene.dgn.urdf_path="$R/urdf_fixed" \
  ++env.single_object_scene.dgn.pose_path="$DATA/meta-v8/unique_dgn_poses" \
  ++env.single_object_scene.filter_file="$DATA/yes.json" \
  ++env.single_object_scene.use_scale_dr=true \
  ++env.single_object_scene.min_scale=0.05 ++env.single_object_scene.max_scale=$SMAX \
  "++env.single_object_scene.table_dims=$TSTAR" \
  ++env.single_object_scene.margin_scale=$MS_SCENE ++env.task.margin_scale=$MS_TASK \
  ++env.single_object_scene.prevent_fall=false \
  ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
  ++load_ckpt="$CKPT" ++agent.train.train_steps=$STEPS > "$OUT/train.log" 2>&1
st=$?
echo "[nec] exit=$st $(date -u +%FT%TZ)" | tee -a "$OUT/run.info"
s5_check_resolved "$OUT/train.log" "max_scale" "table_dims" "margin_scale" "prevent_fall" | tee -a "$OUT/run.info"
grep -E "cur_suc_rate|suc_rate" "$OUT/train.log" | tail -3
