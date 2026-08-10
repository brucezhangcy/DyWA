#!/usr/bin/env bash
# =============================================================================
# Phase A — single-arm object-size frontier: eval-only scale sweep
# Companion to dywa_research/research_prompt_size_transition.md (Phase A) and
# progress_log_size_transition.md §4-§5.
#
# WHAT IT DOES
#   For each object scale s and each goal-radius regime, runs test_rma.py once
#   (eval only, no training) and dumps per-object success/termination results to
#   a dedicated output dir. aggregate_scale_sweep.py then builds the scaling law.
#
# WHERE IT RUNS
#   Inside the pkm1:v0 Docker container (see docker/run.sh): repo at
#   /home/user/DyWA, datasets at /input. Launch the container, then run this
#   script from /home/user/DyWA/dywa_research/.
#
# SAFETY / STATUS
#   NOT yet validated end-to-end (GPU on coldbrew was occupied at build time).
#   Start with DRYRUN=1 to print commands, then SMOKE=1 for a tiny real run,
#   then the full sweep. See KNOWN ISSUES below.
#
# KNOWN ISSUES (from dev_log.md)
#   * NEVER pass ++load_ckpt=<teacher_ckpt> at eval -> success collapses to 0%.
#     The teacher loads implicitly; the student loads via +load_student.
#   * Hydra ++x=null is parsed as the string "null"; edit YAML defaults instead.
# =============================================================================
set -euo pipefail

# ---- knobs (override via env) ----------------------------------------------
DRYRUN="${DRYRUN:-0}"        # 1 = print commands only
SMOKE="${SMOKE:-0}"          # 1 = tiny single-cell run (s=0.05, few envs/steps)
NUM_ENV="${NUM_ENV:-60}"
TEST_STEP="${TEST_STEP:-3000}"
DEVICE="${DEVICE:-cuda:0}"

# Paths inside the container (edit if your mounts differ).
REPO="${REPO:-/home/user/DyWA}"
TRAIN_DIR="${REPO}/dywa/exp/train"
OUT_ROOT="${OUT_ROOT:-${REPO}/dywa_research/results/phase_a}"
DATA_ROOT="${DATA_ROOT:-/input/DGN}"
FILTER_FILE="${FILTER_FILE:-${DATA_ROOT}/test_set.json}"   # held-out object set

# The pretrained checkpoint to evaluate. See progress log §4 "teacher vs student".
# Default assumes the student-eval flow (known-working command). To evaluate the
# Trossen teacher directly, set EVAL_MODE=teacher (path resolved in build_cmd).
EVAL_MODE="${EVAL_MODE:-student}"
STUDENT_CKPT="${STUDENT_CKPT:-${PRETRAINED:-/home/user/DyWA/output/pretrained}/ckpt/last.ckpt}"
TEACHER_CKPT="${TEACHER_CKPT:-${REPO}/output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt/last.ckpt}"

# Sweep grid.
SCALES="${SCALES:-0.05 0.075 0.10 0.15 0.20 0.25 0.30}"
# Goal-radius regimes: "abs:0.05" fixed, or "rel:K" for goal_radius = K * scale.
REGIMES="${REGIMES:-abs:0.05 rel:0.3 rel:0.5 rel:0.75}"

if [[ "${SMOKE}" == "1" ]]; then
  SCALES="0.05"; REGIMES="abs:0.05"; NUM_ENV=16; TEST_STEP=200
fi

mkdir -p "${OUT_ROOT}"
echo "[phase_a] OUT_ROOT=${OUT_ROOT}  EVAL_MODE=${EVAL_MODE}  DRYRUN=${DRYRUN}  SMOKE=${SMOKE}"

# ---- goal_radius resolver ---------------------------------------------------
# Size-relative assumption: post-scale object radius ~= scale value s, because
# set_actor_scale uses scale=target_size/radius => new_radius = target_size = s
# (env/common.py:76-79). Confirm against push_task.py on first real run.
goal_radius_for() {  # args: regime, scale
  local regime="$1" s="$2"
  case "${regime}" in
    abs:*) echo "${regime#abs:}" ;;
    rel:*) awk -v k="${regime#rel:}" -v s="${s}" 'BEGIN{printf "%.5f", k*s}' ;;
    *) echo "ERROR bad regime ${regime}" >&2; return 1 ;;
  esac
}

# ---- command builder --------------------------------------------------------
build_cmd() {  # args: scale, goal_radius, run_name, out_dir
  local s="$1" gr="$2" name="$3" outdir="$4"
  local ckpt_args
  if [[ "${EVAL_MODE}" == "teacher" ]]; then
    # Teacher-only eval path (see progress log §4 caveat 2 - confirm on dry-run).
    ckpt_args="+load_student=${TEACHER_CKPT}"
  else
    ckpt_args="+load_student=${STUDENT_CKPT}"
  fi
  cat <<EOF
PYTORCH_JIT=0 python test_rma.py \
  +platform=debug +env=abs_goal_1view +run=teacher_base +student=dywa/base \
  ++name=${name} \
  ++path.root=${outdir} \
  ++env.num_env=${NUM_ENV} ++global_device=${DEVICE} \
  ++student.norm=ln ++add_teacher_state=1 '++student.decoder.film_mlp=1' \
  ${ckpt_args} \
  ++plot_pc=0 ++dagger_train_env.anneal_step=1 ++use_nvdr_record_episode=false \
  ++monitor.num_env_record=${NUM_ENV} \
  ++env.single_object_scene.filter_file=${FILTER_FILE} \
  ++env.single_object_scene.mode=valid \
  ++env.single_object_scene.min_scale=${s} \
  ++env.single_object_scene.max_scale=${s} \
  ++env.task.goal_radius=${gr} \
  ++log_categorical_results=True \
  ++env.single_object_scene.dgn.data_path=${DATA_ROOT}/meta-v8 \
  ++env.single_object_scene.dgn.pose_path=${DATA_ROOT}/meta-v8/unique_dgn_poses \
  ++test_step=${TEST_STEP}
EOF
}

# ---- sweep ------------------------------------------------------------------
manifest="${OUT_ROOT}/manifest.csv"
echo "scale,regime,goal_radius,run_name,out_dir,status" > "${manifest}"

for s in ${SCALES}; do
  for regime in ${REGIMES}; do
    gr="$(goal_radius_for "${regime}" "${s}")"
    tag="s${s}_$(echo "${regime}" | tr ':' '-')"
    name="phaseA_${tag}"
    outdir="${OUT_ROOT}/${tag}"
    mkdir -p "${outdir}"
    cmd="$(build_cmd "${s}" "${gr}" "${name}" "${outdir}")"

    echo "=============================================================="
    echo "[cell] scale=${s} regime=${regime} goal_radius=${gr} -> ${outdir}"
    if [[ "${DRYRUN}" == "1" ]]; then
      echo "${cmd}"
      echo "${s},${regime},${gr},${name},${outdir},DRYRUN" >> "${manifest}"
      continue
    fi
    status=OK
    ( cd "${TRAIN_DIR}" && eval "${cmd}" ) 2>&1 | tee "${outdir}/run.log" || status=FAILED
    echo "${s},${regime},${gr},${name},${outdir},${status}" >> "${manifest}"
  done
done

echo "[phase_a] done. manifest -> ${manifest}"
echo "[phase_a] next: python aggregate_scale_sweep.py --root ${OUT_ROOT}"
