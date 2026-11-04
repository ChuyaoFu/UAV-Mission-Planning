#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u
# Sources are owned by the host account, while this isolated builder runs root.
git config --global --add safe.directory '*'
export CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6
cd /work/src/PX4-Autopilot
make px4_sitl_default -j6 > /work/logs/px4-build.log 2>&1
cd /work/src/Micro-XRCE-DDS-Agent
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /work/logs/agent-build.log 2>&1
cmake --build build -j6 >> /work/logs/agent-build.log 2>&1
cmake --install build >> /work/logs/agent-build.log 2>&1
ldconfig
cd /work/ros_ws
colcon build --symlink-install --packages-select px4_msgs --parallel-workers 1 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release > /work/logs/px4-msgs-build.log 2>&1
export CMAKE_BUILD_PARALLEL_LEVEL=4 MAKEFLAGS=-j4
colcon build --symlink-install --packages-up-to ov_msckf ov_eval --parallel-workers 2 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    > /work/logs/ros-build.log 2>&1
date -Is > /work/logs/stack-build.success
