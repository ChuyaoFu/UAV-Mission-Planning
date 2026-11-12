#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
source /work/cartographer/v4/carto_ws/install/setup.bash
export XAUTHORITY=${XAUTHORITY:-/tmp/visual-nav-Xauthority}
export DISPLAY=${DISPLAY:?Desktop DISPLAY is required} GZ_PARTITION=visual_nav_sim
test -r "$XAUTHORITY"
# Honor the caller's display and existing cookie. Never weaken xhost controls.
tmux set-environment -g DISPLAY "$DISPLAY"
tmux set-environment -g XAUTHORITY "$XAUTHORITY"
root=/work/cartographer/v4
if ! tmux has-session -t nav-gui 2>/dev/null; then
 tmux new-session -d -s nav-gui "source /opt/ros/jazzy/setup.bash; exec env -u EGL_PLATFORM gz sim -g > $root/logs/gazebo-gui.log 2>&1"
fi
if ! tmux has-session -t v4-rviz 2>/dev/null; then
 tmux new-session -d -s v4-rviz "source /opt/ros/jazzy/setup.bash; source /work/cartographer/v4/carto_ws/install/setup.bash; exec env -u EGL_PLATFORM rviz2 -d $root/config/live-slam.rviz --ros-args -p use_sim_time:=true > $root/logs/rviz.log 2>&1"
fi
echo 'Gazebo and RViz requested on the authenticated Ubuntu desktop; aircraft remains disarmed.'
