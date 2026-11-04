#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
if [ -f /work/cartographer/v2/ros_ws/install/setup.bash ];then source /work/cartographer/v2/ros_ws/install/setup.bash;fi
root=/work/cartographer/v2
profile=${1:-source};sensor=${2:-stereo}
case "$profile" in source) world=dronehacks;;textured) world=dronehacks_textured;;*) exit 2;;esac
case "$sensor" in ideal) depth=/vio/depth/image;depth_source=ideal_rendered_depth;;stereo) depth=/stereo/depth/image;depth_source=opencv_passive_stereo;;*) exit 2;;esac
if tmux has-session -t nav-gz 2>/dev/null;then echo 'Existing simulation: stop after confirmed disarm';exit 1;fi
NAV_WORLD=$world NAV_ESTIMATOR=vision bash "$root/scripts/start_sim.sh"
for i in $(seq 1 40);do
 if grep -q 'Startup script returned successfully' "$root/logs/px4-run.log";then break;fi
 sleep 1
done
px=/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-param
"$px" set EKF2_EV_CTRL 13
"$px" set EKF2_HGT_REF 2
"$px" set EKF2_RNG_CTRL 2
"$px" set EKF2_RNG_POS_Z .025
"$px" set EKF2_EV_POS_X .12
"$px" set EKF2_EV_POS_Z -.055
for i in 0 1 2 3;do
 case $i in 0) x=.115;y=.115;km=.016;;1) x=-.115;y=-.115;km=.016;;2) x=.115;y=-.115;km=-.016;;3) x=-.115;y=.115;km=-.016;;esac
 "$px" set CA_ROTOR${i}_PX "$x"
 "$px" set CA_ROTOR${i}_PY "$y"
 "$px" set CA_ROTOR${i}_KM "$km"
done
"$px" set MPC_THR_HOVER .714
"$px" set MPC_USE_HTE 0
"$px" set MPC_XY_VEL_MAX .65
"$px" set MPC_Z_VEL_MAX_UP .5
"$px" set MPC_TILTMAX_AIR 20
run(){ local name=$1;shift;tmux new-session -d -s "$name" "source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; if [ -f /work/cartographer/v2/ros_ws/install/setup.bash ];then source /work/cartographer/v2/ros_ws/install/setup.bash;fi; exec $* > $root/logs/$name.log 2>&1"; }
run nav-monitor /opt/px4-venv/bin/python3 /work/scripts/gcs-monitor.py
run v2-vio ros2 run ov_msckf run_subscribe_msckf --ros-args -r __ns:=/ov_msckf -p config_path:=$root/config/gazebo_vio/estimator_config.yaml -p use_sim_time:=true -p processing_async:=false -p verbosity:=INFO
run v2-stereo /usr/bin/python3 "$root/scripts/stereo_depth.py"
run v2-depth env DEPTH_TOPIC=$depth DEPTH_SOURCE=$depth_source /usr/bin/python3 "$root/scripts/depth_cloud.py"
run v2-slam ros2 run cartographer_ros cartographer_node -configuration_directory "$root/config" -configuration_basename oak_3d.lua --ros-args -p use_sim_time:=true -r points2:=/carto/points -r imu:=/vio/imu -r odom:=/carto/vio_odom
run v2-fusion /usr/bin/python3 "$root/scripts/fusion.py"
run v2-range-bridge ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/work/cartographer/config/range_bridge.yaml
run v2-range /usr/bin/python3 "$root/scripts/down_range.py"
run v2-map ros2 run cartographer_ros cartographer_occupancy_grid_node -resolution .04 -publish_period_sec 1.0 --ros-args -p use_sim_time:=true
echo "Seven-inch approximation; profile=$profile sensor=$sensor, camera pitch recorded in config/model-manifest.json. Disarmed."
