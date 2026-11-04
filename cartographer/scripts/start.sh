#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
set -u
profile=${1:-source};sensor=${2:-ideal}
case "$profile" in source) world=dronehacks;;textured) world=dronehacks_textured;;*) exit 2;;esac
case "$sensor" in ideal) depth=/vio/depth/image;source_name=ideal_rendered_depth;;stereo) depth=/stereo/depth/image;source_name=opencv_passive_stereo;;*) exit 2;;esac
if tmux has-session -t nav-gz 2>/dev/null;then echo 'Stop existing disarmed simulation first';exit 1;fi
export NAV_WORLD="$world" NAV_ESTIMATOR=vision
bash /work/cartographer/scripts/start_sim.sh
for i in $(seq 1 30);do
 if grep -q 'Startup script returned successfully' /work/logs/px4-run.log && /work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-param show -q EKF2_EV_CTRL >/dev/null 2>&1;then break;fi
 sleep 1
done
/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-param set EKF2_EV_CTRL 11
# Same original measurement lever arm, position/yaw fusion only; no fabricated velocity.
tmux new-session -d -s nav-monitor 'exec /opt/px4-venv/bin/python3 /work/scripts/gcs-monitor.py > /work/cartographer/logs/gcs-monitor.log 2>&1'
tmux new-session -d -s carto-stereo 'source /opt/ros/jazzy/setup.bash; exec /usr/bin/python3 /work/cartographer/scripts/stereo_depth.py > /work/cartographer/logs/stereo.log 2>&1'
tmux new-session -d -s carto-depth "source /opt/ros/jazzy/setup.bash; DEPTH_TOPIC=$depth DEPTH_SOURCE=$source_name exec /usr/bin/python3 /work/cartographer/scripts/depth_cloud.py > /work/cartographer/logs/depth.log 2>&1"
tmux new-session -d -s carto-slam 'source /opt/ros/jazzy/setup.bash; exec ros2 run cartographer_ros cartographer_node -configuration_directory /work/cartographer/config -configuration_basename oak_3d.lua --ros-args -p use_sim_time:=true -r points2:=/carto/points -r imu:=/vio/imu > /work/cartographer/logs/slam.log 2>&1'
tmux new-session -d -s carto-ev 'source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; exec /usr/bin/python3 /work/cartographer/scripts/carto_to_px4.py > /work/cartographer/logs/ev.log 2>&1'
tmux new-session -d -s carto-range-bridge 'source /opt/ros/jazzy/setup.bash; exec ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/work/cartographer/config/range_bridge.yaml > /work/cartographer/logs/range-bridge.log 2>&1'
tmux new-session -d -s carto-range 'source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; exec /usr/bin/python3 /work/cartographer/scripts/down_range.py > /work/cartographer/logs/range.log 2>&1'
/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-param set EKF2_RNG_CTRL 2
/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-param set EKF2_HGT_REF 2
/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-param set EKF2_RNG_POS_Z 0.12
echo "Cartographer profile=$profile depth=$sensor. Disarmed. No GPS sensor; no truth localization input."
