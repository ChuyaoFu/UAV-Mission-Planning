#!/usr/bin/env bash
# Pinned upstream source, minimal ROS packages, no upstream simulated sensors.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
revision=23a8d5a191711dd65633df689bd00f55d4dea8f9
source /opt/ros/jazzy/setup.bash
mkdir -p "$root/upstream"
src="$root/upstream/ego-planner-swarm"
if [ ! -d "$src/.git" ];then git clone --no-checkout https://github.com/ZJU-FAST-Lab/ego-planner-swarm.git "$src";fi
if [ "$(git -C "$src" rev-parse HEAD 2>/dev/null || true)" != "$revision" ];then
 test -z "$(git -C "$src" status --porcelain)" || { echo 'Upstream checkout contains local edits';exit 1; }
 git -C "$src" checkout --detach "$revision"
fi
if git -C "$src" apply --reverse --check --ignore-whitespace "$root/patches/ego-swarm-ros2.patch" 2>/dev/null;then :
else git -C "$src" apply --check --ignore-whitespace "$root/patches/ego-swarm-ros2.patch";git -C "$src" apply --ignore-whitespace "$root/patches/ego-swarm-ros2.patch";fi
# Only packages required by the single-aircraft measured-sensor planner.
mkdir -p "$root/ego_ws/src"
for package in planner/plan_env planner/path_searching planner/bspline_opt planner/traj_utils planner/plan_manage uav_simulator/Utils/quadrotor_msgs;do
 mkdir -p "$root/ego_ws/src/$(dirname "$package")"
 cp -a "$src/src/$package" "$root/ego_ws/src/$package"
done
cd "$root/ego_ws"
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF --parallel-workers 2
