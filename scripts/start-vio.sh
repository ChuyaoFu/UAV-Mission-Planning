#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
if tmux has-session -t nav-vio 2>/dev/null; then
    echo 'nav-vio already exists' >&2; exit 1
fi
tmux new-session -d -s nav-vio 'source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; exec ros2 run ov_msckf run_subscribe_msckf --ros-args -r __ns:=/ov_msckf -p config_path:=/work/config/gazebo_vio/estimator_config.yaml -p use_sim_time:=true -p processing_async:=false -p verbosity:=INFO > /work/logs/openvins.log 2>&1'
