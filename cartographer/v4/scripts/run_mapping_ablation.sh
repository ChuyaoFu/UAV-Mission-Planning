#!/bin/bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
source /work/cartographer/v4/carto_ws/install/setup.bash
export ROS_DOMAIN_ID=43
root=/work/cartographer/v4
source_label=${1:?};profile=${2:?};label=${3:?}
out=$root/results/$label
mkdir -p "$out"
python3 - <<'CHECK'
import os
from pathlib import Path
pids=[]
for p in Path('/proc').glob('[0-9]*/exe'):
 try:
  if Path(os.readlink(p)).name=='cartographer_node' and b'ROS_DOMAIN_ID=43' in (p.parent/'environ').read_bytes().split(b'\0'):pids.append(p.parent.name)
 except OSError:pass
assert not pids,('Existing replay mapper',pids)
CHECK
cp "$root/config/$profile.lua" "$out/config.lua"
"$root/carto_ws/install/cartographer_ros/lib/cartographer_ros/cartographer_node" -configuration_directory "$root/config" -configuration_basename "$profile.lua" --ros-args -p use_sim_time:=true -r points2:=/mapping/points -r imu:=/vio/imu -r odom:=/carto/vio_odom > "$out/slam.log" 2>&1 &
mapper_pid=$!
trap 'kill "$mapper_pid" 2>/dev/null || true; wait "$mapper_pid" 2>/dev/null || true' EXIT
python3 "$root/scripts/replay_recorded_mapping.py" "$root/data/$source_label" "$out" 4 > "$out/replay.log" 2>&1
python3 "$root/scripts/compare_slam.py" "$label" "$root/results/$source_label/evaluation-offline.json" > "$out/metrics.log" 2>&1
