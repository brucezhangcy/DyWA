#!/bin/bash
# Polls host load average, launches Trossen sanity training when load is acceptable.
# Logs to /data/bruce/wait_and_launch.log so the user can monitor.
# Designed to be run in the background; will exit after launching the training.

LOG=/data/bruce/wait_and_launch.log
FLAG=/data/bruce/.trossen_training_launched   # idempotency marker
THRESHOLD=25                                    # 1-min load avg threshold (host has 96 CPUs)
POLL_INTERVAL=300                                # 5 minutes

echo "[$(date '+%F %T')] wait_and_launch_trossen started. threshold=$THRESHOLD, poll=$POLL_INTERVAL s" >> "$LOG"

if [ -e "$FLAG" ]; then
  echo "[$(date '+%F %T')] flag $FLAG exists — already launched, exiting." >> "$LOG"
  exit 0
fi

while true; do
  LOAD=$(awk '{print $1}' /proc/loadavg)
  LOAD_INT=$(printf '%.0f' "$LOAD")
  echo "[$(date '+%F %T')] load_1m=$LOAD (threshold=$THRESHOLD)" >> "$LOG"
  if [ "$LOAD_INT" -lt "$THRESHOLD" ]; then
    echo "[$(date '+%F %T')] load below threshold — launching training" >> "$LOG"
    touch "$FLAG"
    # Launch the sanity training inside the existing dywa_1 container
    docker exec -d dywa_1 bash -c "cd /home/user/DyWA/dywa/exp/train && \
      PYTORCH_JIT=0 python3 -u train_ppo_arm.py \
        +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
        ++tag=trossen_sanity_100k \
        ++env.num_env=1024 \
        ++env.trossen.base_height=table \
        ++env.trossen.ws_bound='[[-0.25,-0.35,-0.1],[0.25,0.35,0.4]]' \
        ++global_device=cuda:0 \
        ++path.root=/home/user/DyWA/output/trossen_pipeline/sanity \
        ++env.single_object_scene.dgn.data_path=/input/DGN/meta-v8 \
        ++env.single_object_scene.dgn.pose_path=/input/DGN/meta-v8/unique_dgn_poses \
        ++env.single_object_scene.filter_file=/input/DGN/yes.json \
        ++icp_obs.icp.ckpt='imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920' \
        ++agent.train.train_steps=100000 \
        > /home/user/DyWA/output/trossen_pipeline/sanity.log 2>&1"
    echo "[$(date '+%F %T')] docker exec issued (-d). Training runs in dywa_1." >> "$LOG"
    echo "[$(date '+%F %T')] Container-side log: /home/user/DyWA/output/trossen_pipeline/sanity.log" >> "$LOG"
    echo "[$(date '+%F %T')] Host-side log:      /data/bruce/DyWA/output/trossen_pipeline/sanity.log" >> "$LOG"
    exit 0
  fi
  sleep "$POLL_INTERVAL"
done
