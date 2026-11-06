#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
if [ -f /work/cartographer/v2/ros_ws/install/setup.bash ];then source /work/cartographer/v2/ros_ws/install/setup.bash;fi
source /work/cartographer/v3/carto_ws/install/setup.bash
source /work/cartographer/v5/ego_ws/install/setup.bash
root=/work/cartographer/v5
if tmux has-session -t nav-gz 2>/dev/null;then echo 'Existing simulation: stop after confirmed disarm';exit 1;fi
# Optional isolated estimator build. Keep the existing installation untouched.
vio_prefix=${VIO_INSTALL_PREFIX:-/work/cartographer/v2/ros_ws/install}
[[ "$vio_prefix" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Invalid estimator installation path';exit 2; }
vio_binary=$vio_prefix/ov_msckf/lib/ov_msckf/run_subscribe_msckf
[[ -x "$vio_binary" ]] || { echo 'Estimator binary unavailable; no simulation started';exit 2; }
vio_libraries=$vio_prefix/ov_core/lib:$vio_prefix/ov_init/lib:$vio_prefix/ov_msckf/lib
home_relative=${NAV_HOME_RELATIVE_ODOMETRY:-0}
[[ "$home_relative" = 0 || "$home_relative" = 1 ]] || { echo 'Invalid home-relative selector';exit 2; }
structural_backend=$(python3 -c 'import json;print(json.load(open("/work/cartographer/v5/config/navigation.json")).get("structural_evidence_backend","planar"))')
[[ "$structural_backend" = planar || "$structural_backend" = grabcut ]] || { echo 'Invalid evidence backend';exit 2; }
structural_variant=$(python3 -c 'import json;print(json.load(open("/work/cartographer/v5/config/navigation.json")).get("structural_evidence_variant","cold5_freeze1_seedvalidated_400p"))')
[[ "$structural_variant" =~ ^[A-Za-z0-9_]+$ ]] || exit 2
structural_floor=$(python3 -c 'import json;print(str(bool(json.load(open("/work/cartographer/v5/config/navigation.json")).get("structural_evidence_floor_completion",False))).lower())')
structural_package=$root/candidates/grabcut-ros2-node
if [ "$structural_backend" = grabcut ];then
 test -f "$structural_package/build/libboundary_lbd.so" || { echo 'Build optional structural helper first';exit 2; }
 python3 -c 'import json;c=json.load(open("/work/cartographer/v5/config/navigation.json"));assert c.get("geometric_free_rays",False) and c.get("inferred_geometry_input_max_age_s",.25)>=.5,"Explicit conditional geometry and compatible native input age required"'
fi
reference_flags=()
if [ "$home_relative" = 1 ];then reference_flags+=(--enable-home-relative);fi
python3 "$root/scripts/select_estimator_reference.py" "$root/config/gazebo_vio/estimator_config.yaml" "$root/config/robot.json" "${reference_flags[@]}"
python3 "$root/scripts/generate_robot_foot_geometry.py" --robot-model "$root/config/model.sdf" --output "$root/config/robot-foot-geometry.json" --point-error-m .002
python3 "$root/scripts/self_mask.py"
mkdir -p "$root/logs" "$root/mission-output"
python3 - "$vio_binary" "$vio_prefix" "$root/logs/estimator-binary.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
binary,prefix,out=map(Path,sys.argv[1:])
files=[binary,*[f for package in ('ov_core','ov_init','ov_msckf') for f in (prefix/package/'lib').glob('*.so')]]
out.write_text(json.dumps({'binary':str(binary),'install_prefix':str(prefix),'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2)+'\n')
PY
profile=${1:-source};sensor=stereo
case "$profile" in source) world=dronehacks;;unknown) world=cadfree_unknown;;*) exit 2;;esac
case "$sensor" in stereo) depth=/stereo/depth/image;depth_source=opencv_passive_stereo;;*) exit 2;;esac
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
"$px" set MPC_LAND_SPEED .20
run(){ local name=$1;shift;tmux new-session -d -s "$name" "source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; if [ -f /work/cartographer/v2/ros_ws/install/setup.bash ];then source /work/cartographer/v2/ros_ws/install/setup.bash;fi; source /work/cartographer/v3/carto_ws/install/setup.bash; source /work/cartographer/v5/ego_ws/install/setup.bash; export PYTHONPATH=$root/scripts:\$PYTHONPATH; exec /usr/bin/python3 $root/scripts/landlock_exec.py $* > $root/logs/$name.log 2>&1"; }
run nav-monitor /opt/px4-venv/bin/python3 $root/scripts/gcs-monitor.py
run v5-vio env LD_LIBRARY_PATH=$vio_libraries:${LD_LIBRARY_PATH:-} "$vio_binary" --ros-args -r __ns:=/ov_msckf -p config_path:=$root/config/gazebo_vio/estimator_config.yaml -p use_sim_time:=true -p processing_async:=false -p verbosity:=INFO
if [ "$home_relative" = 1 ];then
 run v5-home-relative env NATIVE_FAST_IMU_EPOCH_VERIFIED=${NATIVE_FAST_IMU_EPOCH_VERIFIED:-0} /usr/bin/python3 "$root/scripts/home_relative_odometry.py"
fi
run v5-stereo /usr/bin/python3 "$root/scripts/stereo_depth.py"
if python3 -c 'import json,sys;sys.exit(not json.load(open("/work/cartographer/v5/config/navigation.json")).get("geometric_free_rays",False))';then
 if [ "$structural_backend" = grabcut ];then
  run v5-structural /bin/bash "$structural_package/run_ros2.sh" "$root/config/gazebo_vio/kalibr_imucam_chain.yaml" "$root/config/robot.json" "$root/config/gazebo_vio" "$structural_package/build/libboundary_lbd.so" "$structural_variant" true ground_ipm -p include_floor_completion:=$structural_floor -p health_topic:=/nav/structural_health
 else
 run v5-planar /usr/bin/python3 "$root/scripts/planar_evidence_node.py" --ros-args -p enable_inferred_free:=true -p max_age_s:=0.25 -p compute_budget_s:=0.20 -p max_hz:=3.0 -r /nav/planar_evidence:=/nav/depth_evidence -r /nav/planar_evidence_health:=/nav/planar_health
 fi
fi
run v5-depth env DEPTH_TOPIC=/stereo/depth/image DEPTH_SOURCE=opencv_passive_stereo /usr/bin/python3 "$root/scripts/depth_cloud.py"
run v5-guideline /usr/bin/python3 "$root/scripts/guideline_node.py"
run v5-structure /usr/bin/python3 "$root/scripts/mapping_cloud.py"
run v5-slam ros2 run cartographer_ros cartographer_node -configuration_directory "$root/config" -configuration_basename oak_3d.lua -collect_metrics=true --ros-args -p use_sim_time:=true -r points2:=/mapping/points -r imu:=/vio/imu -r odom:=/carto/vio_odom
run v5-fusion env NAV_HOME_RELATIVE_ODOMETRY=$home_relative /usr/bin/python3 "$root/scripts/fusion.py"
run v5-range-bridge ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=$root/config/range_bridge.yaml
run v5-range /usr/bin/python3 "$root/scripts/down_range.py"
run v5-frames /usr/bin/python3 "$root/scripts/sensor_frames.py"
run v5-cloud-view /usr/bin/python3 "$root/scripts/slam_cloud_view.py"
run v5-ego /usr/bin/python3 "$root/scripts/ego_launch.py"
echo "CAD-free measured-map stack started; no route goals loaded."
