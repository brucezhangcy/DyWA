#!/bin/bash
# DyWA full training pipeline: Stage1 -> Stage2 -> Distill -> Eval
# Logs per-stage output to output/pipeline/stage{1,2,distill,eval}.log

set -e

CONDA_LIB=/home/brucezhang/anaconda3/envs/corn-deploy2/lib
TORCH_LIB=/home/brucezhang/anaconda3/envs/corn-deploy2/lib/python3.8/site-packages/torch/lib
DYWA_SRC=/home/brucezhang/Downloads/DyWA/dywa/src
DYWA_PKG=/home/brucezhang/Downloads/DyWA/dywa
DGN_BASE=/home/brucezhang/Downloads/DyWA/data
ROOT=/home/brucezhang/Downloads/DyWA/output/pipeline
TRAIN_DIR=/home/brucezhang/Downloads/DyWA/dywa/exp/train
ICP_CKPT="imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920"

export PYTORCH_JIT=0
export PYTHONPATH=$DYWA_SRC:$DYWA_PKG
export LD_LIBRARY_PATH=$CONDA_LIB:$TORCH_LIB

mkdir -p $ROOT

run() {
    # run <log_label> <python_args...>
    local label=$1; shift
    local log=$ROOT/${label}.log
    echo "[$(date '+%H:%M:%S')] Starting $label ..."
    cd $TRAIN_DIR
    conda run -n corn-deploy2 python "$@" 2>&1 | tee $log
    echo "[$(date '+%H:%M:%S')] $label DONE"
}

# ── Stage 1 ─────────────────────────────────────────────────────────────────
run stage1 train_ppo_arm.py \
  +platform=debug +env=icra_base +run=icra_ours_abs_rel \
  ++env.seed=56081 ++tag=stage1 ++global_device=cuda:0 \
  ++env.num_env=1024 \
  ++path.root=$ROOT/teacher-stage1 \
  ++env.single_object_scene.dgn.data_path=$DGN_BASE/meta-v8 \
  ++env.single_object_scene.dgn.pose_path=$DGN_BASE/meta-v8/unique_dgn_poses \
  ++env.single_object_scene.filter_file=$DGN_BASE/yes.json \
  ++icp_obs.icp.ckpt="$ICP_CKPT" \
  ++agent.train.train_steps=50000

S1_CKPT=$ROOT/teacher-stage1/run-000/ckpt
echo "[pipeline] Stage 1 checkpoint: $S1_CKPT"

# ── Stage 2 ─────────────────────────────────────────────────────────────────
run stage2 train_ppo_arm.py \
  +platform=debug +env=icra_base +run=icra_ours_abs_rel \
  ++env.seed=56081 ++tag=stage2 ++global_device=cuda:0 \
  ++env.num_env=2048 \
  ++path.root=$ROOT/teacher-stage2 \
  ++env.single_object_scene.dgn.data_path=$DGN_BASE/meta-v8 \
  ++env.single_object_scene.dgn.pose_path=$DGN_BASE/meta-v8/unique_dgn_poses \
  ++env.single_object_scene.filter_file=$DGN_BASE/yes.json \
  ++icp_obs.icp.ckpt="$ICP_CKPT" \
  ++is_phase2=true \
  ++phase2.min_reset_to_update=16384 \
  ++agent.train.lr=2e-6 \
  ++agent.train.alr.initial_scale=6.67e-3 \
  ++load_ckpt="$S1_CKPT" \
  ++agent.train.train_steps=20000

S2_CKPT=$ROOT/teacher-stage2/run-000/ckpt
echo "[pipeline] Stage 2 checkpoint: $S2_CKPT"

# ── Distillation (1-view student) ────────────────────────────────────────────
run distill train_rma.py \
  +platform=debug +env=abs_goal_1view +run=teacher_base +student=dywa/base \
  ++name=film_mlp \
  ++path.root=$ROOT/distill \
  ++env.num_env=256 ++global_device=cuda:0 \
  ++student.norm=ln \
  ++add_teacher_state=1 \
  ++student.decoder.film_mlp=1 \
  ++icp_obs.icp.ckpt="$ICP_CKPT" \
  ++env.single_object_scene.dgn.data_path=$DGN_BASE/meta-v8 \
  ++env.single_object_scene.dgn.pose_path=$DGN_BASE/meta-v8/unique_dgn_poses \
  ++env.single_object_scene.filter_file=$DGN_BASE/yes.json \
  ++load_ckpt="$S2_CKPT" \
  ++agent.train.train_steps=20000

STUDENT_CKPT=$ROOT/distill/film_mlp/ckpt/last.ckpt
echo "[pipeline] Student checkpoint: $STUDENT_CKPT"

# ── Evaluation on unseen objects ─────────────────────────────────────────────
mkdir -p $ROOT/eval
run eval test_rma.py \
  +platform=debug +env=abs_goal_1view +run=teacher_base +student=dywa/base \
  ++name=film_mlp \
  ++path.root=$ROOT/eval \
  ++env.num_env=60 ++global_device=cuda:0 \
  ++student.norm=ln \
  ++add_teacher_state=1 \
  ++student.decoder.film_mlp=1 \
  ++icp_obs.icp.ckpt="$ICP_CKPT" \
  ++env.single_object_scene.dgn.data_path=$DGN_BASE/meta-v8 \
  ++env.single_object_scene.dgn.pose_path=$DGN_BASE/meta-v8/unique_dgn_poses \
  ++env.single_object_scene.filter_file=$DGN_BASE/test_set.json \
  ++load_ckpt="$S2_CKPT" \
  +load_student="$STUDENT_CKPT" \
  ++dagger_train_env.anneal_step=1 \
  ++plot_pc=0 \
  ++monitor.num_env_record=60 \
  ++env.single_object_scene.mode=valid \
  ++log_categorical_results=True

echo "[pipeline] All stages complete. Results in $ROOT"
