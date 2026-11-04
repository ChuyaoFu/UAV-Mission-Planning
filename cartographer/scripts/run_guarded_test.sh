#!/usr/bin/env bash
# Run inside existing SITL container only. This short diagnostic may abort/land.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
set -u
label=${1:-guarded-04}
[[ "$label" =~ ^[a-z0-9-]+$ ]] || exit 2
[[ -f /.dockerenv && ${ROS_DOMAIN_ID:-} = 42 ]] || exit 2
px=/work/src/PX4-Autopilot/build/px4_sitl_default/bin
"$px/px4-listener" vehicle_status -n 1 | grep -q 'arming_state: 1' || exit 1
bag=/work/cartographer/data/$label
[[ ! -e "$bag" ]] || { echo 'Refuse to overwrite recording';exit 1; }
topics='/clock /vio/imu /vio/left/image_raw /vio/right/image_raw /vio/depth/image /carto/points /carto/depth_health /carto/localization_health /carto/down_range /tf /evaluation/poses /fmu/out/vehicle_status_v1 /fmu/out/vehicle_local_position_v1 /fmu/in/vehicle_visual_odometry /fmu/in/vehicle_command /fmu/in/trajectory_setpoint'
tmux new-session -d -s carto-record "source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; exec timeout --signal=INT --kill-after=8 65 ros2 bag record --storage mcap --storage-preset-profile zstd_fast --use-sim-time --disable-keyboard-controls --max-cache-size 134217728 -o $bag --topics $topics > /work/cartographer/logs/bag-$label.log 2>&1"
for _ in $(seq 1 15);do
 if find "$bag" -name '*.mcap' -size +0c 2>/dev/null | grep -q .;then break;fi
 sleep 1
done
find "$bag" -name '*.mcap' -size +0c | grep -q . || { echo 'Recorder not writing: did not arm';exit 1; }
tmux new-session -d -s carto-eval "source /opt/ros/jazzy/setup.bash; exec /usr/bin/python3 /work/cartographer/scripts/capture_localization.py --seconds 55 --output /work/cartographer/results/motion-$label.json > /work/cartographer/logs/eval-$label.log 2>&1"
tmux new-session -d -s carto-judge "source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; export GZ_PARTITION=visual_nav_sim; exec /usr/bin/python3 /work/cartographer/scripts/judge.py --record-seconds 55 --output /work/cartographer/results/judge-$label.json > /work/cartographer/logs/judge-$label.log 2>&1"
CARTO_FLIGHT_RESULT=/work/cartographer/results/flight-$label.json NAV_TEST_REQUIRE_VISION=1 NAV_TEST_BODY_FORWARD=1 /usr/bin/python3 /work/cartographer/scripts/flight_diagnostic.py > /work/cartographer/logs/flight-$label.log 2>&1 || true
"$px/px4-listener" vehicle_status -n 1 > /work/cartographer/results/status-$label.txt
grep -q 'arming_state: 1' /work/cartographer/results/status-$label.txt || { echo 'Still armed; do not reset';exit 1; }
/usr/bin/python3 - "$label" <<'PY'
import sys,shutil
from pathlib import Path
logs=list(Path('/work/src/PX4-Autopilot/build/px4_sitl_default/rootfs/log').glob('*/*.ulg'))
shutil.copy2(max(logs,key=lambda p:p.stat().st_mtime),'/work/cartographer/data/'+sys.argv[1]+'.ulg')
PY
cp /work/logs/px4-run.log /work/cartographer/logs/px4-$label.log
for _ in $(seq 1 75);do
 if ! tmux has-session -t carto-record 2>/dev/null && ! tmux has-session -t carto-eval 2>/dev/null && ! tmux has-session -t carto-judge 2>/dev/null;then break;fi
 sleep 1
done
ros2 bag info "$bag" > /work/cartographer/results/bag-info-$label.txt
/usr/bin/python3 /work/cartographer/scripts/judge.py --input /work/cartographer/results/judge-$label.json --output /work/cartographer/results/verdict-$label.json
/usr/bin/python3 /work/cartographer/scripts/audit_runtime.py
