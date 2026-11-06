#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
if [ -f /work/cartographer/v2/ros_ws/install/setup.bash ];then source /work/cartographer/v2/ros_ws/install/setup.bash;fi
source /work/cartographer/v3/carto_ws/install/setup.bash
source /work/cartographer/v5/ego_ws/install/setup.bash
root=/work/cartographer/v5
python3 "$root/scripts/self_mask.py"
mkdir -p "$root/logs" "$root/mission-output"
profile=${1:-source};sensor=stereo
case "$profile" in source) world=dronehacks;;unknown) world=cadfree_unknown;;*) exit 2;;esac
case "$sensor" in stereo) depth=/stereo/depth/image;depth_source=opencv_passive_stereo;;*) exit 2;;esac
if tmux has-session -t nav-gz 2>/dev/null;then echo 'Existing simulation: stop after confirmed disarm';exit 1;fi
NAV_WORLD=$world NAV_ESTIMATOR=vision bash "$root/scripts/start_sim.sh"
for i in $(seq 1 40);do
 if grep -q 'Startup script returned successfully' "$root/logs/px4-run.log";then break;fi
 if ! tmux has-session -t nav-px4 2>/dev/null;then echo "PX4 startup failed; no flight allowed";exit 1;fi
 sleep 1
done
grep -q "Startup script returned successfully" "$root/logs/px4-run.log" || { echo "PX4 startup timeout";exit 1; }
px=/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-param
"$px" set EKF2_EV_CTRL 13
"$px" set EKF2_HGT_REF 2
"$px" set EKF2_RNG_CTRL 2
read -r imu_x imu_y imu_z tof_x tof_y tof_z < <(python3 -c 'import json;d=json.load(open("/work/cartographer/v5/config/robot.json"));a=d["imu_in_body_m"];b=d["tof_in_body_m"];print(float(a[0]),-float(a[1]),-float(a[2]),float(b[0]),-float(b[1]),-float(b[2]))')
"$px" set EKF2_EV_POS_X 0
"$px" set EKF2_EV_POS_Y 0
"$px" set EKF2_EV_POS_Z 0
"$px" set EKF2_RNG_POS_Z "$tof_z"
for i in 0 1 2 3;do
 case $i in 0) x=.115;y=.115;km=.016;;1) x=-.115;y=-.115;km=.016;;2) x=.115;y=-.115;km=-.016;;3) x=-.115;y=.115;km=-.016;;esac
 "$px" set CA_ROTOR${i}_PX "$x"
 "$px" set CA_ROTOR${i}_PY "$y"
 "$px" set CA_ROTOR${i}_KM "$km"
done
"$px" set MPC_THR_HOVER .714
"$px" set MPC_USE_HTE 0
"$px" set MPC_XY_VEL_MAX .85
"$px" set MPC_Z_VEL_MAX_UP .6
"$px" set MPC_TILTMAX_AIR 20
"$px" set MPC_TKO_RAMP_T 1.5
"$px" set MPC_LAND_SPEED .35
run(){ local name=$1;shift;tmux new-session -d -s "$name" "source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; if [ -f /work/cartographer/v2/ros_ws/install/setup.bash ];then source /work/cartographer/v2/ros_ws/install/setup.bash;fi; source /work/cartographer/v3/carto_ws/install/setup.bash; source /work/cartographer/v5/ego_ws/install/setup.bash; export PYTHONPATH=$root/scripts:\$PYTHONPATH; exec /usr/bin/python3 $root/scripts/landlock_exec.py $* > $root/logs/$name.log 2>&1"; }
run nav-monitor /opt/px4-venv/bin/python3 $root/scripts/gcs-monitor.py
run v5-vio ros2 run ov_msckf run_subscribe_msckf --ros-args -r __ns:=/ov_msckf -p config_path:=$root/config/gazebo_vio/estimator_config.yaml -p use_sim_time:=true -p processing_async:=false -p verbosity:=INFO
run v5-stereo /usr/bin/python3 "$root/scripts/stereo_depth.py"
run v5-depth env DEPTH_TOPIC=/stereo/depth/image DEPTH_SOURCE=opencv_passive_stereo /usr/bin/python3 "$root/scripts/depth_cloud.py"
run v5-structure /usr/bin/python3 "$root/scripts/mapping_cloud.py"
run v5-slam ros2 run cartographer_ros cartographer_node -configuration_directory "$root/config" -configuration_basename oak_3d.lua -collect_metrics=true --ros-args -p use_sim_time:=true -r points2:=/mapping/points -r imu:=/vio/imu -r odom:=/carto/vio_odom
run v5-fusion /usr/bin/python3 "$root/scripts/fusion.py"
run v5-range-bridge ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=$root/config/range_bridge.yaml
run v5-range /usr/bin/python3 "$root/scripts/down_range.py"
run v5-frames /usr/bin/python3 "$root/scripts/sensor_frames.py"
run v5-cloud-view /usr/bin/python3 "$root/scripts/slam_cloud_view.py"
run v5-ego /usr/bin/python3 "$root/scripts/ego_launch.py"
echo "CAD-free measured-map stack started; no route goals loaded."
