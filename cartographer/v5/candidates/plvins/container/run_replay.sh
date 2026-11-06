#!/usr/bin/env bash
# Invoke inside the isolated candidate container, with /data mounted writable.
set -euo pipefail
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
export ROS_HOME=/data/ros-home
config=${1:-/data/plvins.yaml}
bag=${2:-/data/sensors.bag}
result=${3:-/data/results}
rate=${4:-1.0}
if [[ -e "$result" ]]; then
  echo "Refusing to overwrite existing result directory: $result" >&2
  exit 2
fi
mkdir -p "$result"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  for attempt in $(seq 1 50); do
    live=0
    for pid in "${pids[@]}"; do if kill -0 "$pid" 2>/dev/null; then live=1; fi; done
    if [[ "$live" == 0 ]]; then break; fi
    sleep .1
  done
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep .2
  for pid in "${pids[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
roscore > "$result/roscore.log" 2>&1 & pids+=("$!")
for attempt in $(seq 1 50); do
  if rosparam list >/dev/null 2>&1; then break; fi
  sleep .1
done
rosparam set use_sim_time true
python${ROS_PYTHON_VERSION:-2} /opt/candidate-tools/ros1_observer.py --output "$result" > "$result/observer.log" 2>&1 & pids+=("$!")
roslaunch /opt/plvins_ws/replay.launch config:="$config" > "$result/plvins.log" 2>&1 & pids+=("$!")
sleep 2
rosbag play --clock --rate "$rate" --delay 1 "$bag" > "$result/playback.log" 2>&1
# Wall-clock drain only; no synthesized sensor data.
sleep 3
