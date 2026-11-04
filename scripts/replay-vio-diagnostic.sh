#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
export ROS_DOMAIN_ID=43 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
cap=${1:?Use 20 or 25}
case "$cap" in 20|25) ;; *) exit 2;;esac
label=vio-replay-cap${cap}
out=/work/results/$label
mkdir "$out"
cp -r /work/results/vio-raw-diagnostic-01/config "$out/config"
sed -i "s/^track_frequency: .*/track_frequency: ${cap}.0 # upper bound for camera callback acceptance/" "$out/config/estimator_config.yaml"
vio_pid= rec_pid=
cleanup() {
    if [ -n "$rec_pid" ]; then kill -TERM "$rec_pid" 2>/dev/null || true; wait "$rec_pid" || true;fi
    if [ -n "$vio_pid" ]; then kill -TERM "$vio_pid" 2>/dev/null || true; wait "$vio_pid" || true;fi
}
trap cleanup EXIT
/work/ros_ws/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf --ros-args \
    -r __ns:=/ov_msckf -p config_path:="$out/config/estimator_config.yaml" \
    -p use_sim_time:=true -p processing_async:=false -p verbosity:=INFO \
    > "$out/openvins.log" 2>&1 &
vio_pid=$!
ros2 bag record -s mcap --storage-preset-profile zstd_fast --use-sim-time \
    --disable-keyboard-controls -o /work/data/$label \
    --topics /ov_msckf/poseimu /ov_msckf/odomimu /ov_msckf/points_msckf /ov_msckf/points_slam \
    > "$out/record.log" 2>&1 &
rec_pid=$!
sleep 2
ros2 bag play /work/data/vio-raw-diagnostic-01 --clock 250 --rate 1.0 \
    --disable-keyboard-controls --topics /vio/left/image_raw /vio/right/image_raw /vio/imu \
    > "$out/play.log" 2>&1
sleep 2
echo "DONE $label (offline ROS domain 43; no flight controller)"
