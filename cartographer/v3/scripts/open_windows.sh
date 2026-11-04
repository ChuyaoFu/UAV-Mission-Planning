#!/usr/bin/env bash
set -euo pipefail
root=/work/cartographer/v3
if ! tmux has-session -t nav-gui 2>/dev/null;then
 tmux new-session -d -s nav-gui "source /opt/ros/jazzy/setup.bash; DISPLAY=:0 XAUTHORITY=/tmp/visual-nav-Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia GZ_PARTITION=visual_nav_sim exec gz sim -g > $root/logs/gazebo-gui.log 2>&1"
fi
if ! tmux has-session -t v3-rviz 2>/dev/null;then
 tmux new-session -d -s v3-rviz "source /opt/ros/jazzy/setup.bash; DISPLAY=:0 XAUTHORITY=/tmp/visual-nav-Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia exec rviz2 -d $root/config/live-slam.rviz --ros-args -p use_sim_time:=true > $root/logs/rviz.log 2>&1"
fi
if ! tmux has-session -t v3-view 2>/dev/null;then
 tmux new-session -d -s v3-view "source /opt/ros/jazzy/setup.bash; exec /usr/bin/python3 $root/scripts/slam_view.py > $root/logs/view.log 2>&1"
fi
export DISPLAY=:0 XAUTHORITY=/tmp/visual-nav-Xauthority GZ_PARTITION=visual_nav_sim
for i in $(seq 1 20);do
 if wmctrl -l | grep -q RViz && wmctrl -l | grep -q 'Gazebo Sim';then break;fi
 sleep 1
done
wmctrl -r 'Gazebo Sim' -b remove,maximized_vert,maximized_horz
wmctrl -r 'Gazebo Sim' -e 0,0,40,1250,1480
sleep 1
wmctrl -r 'Gazebo Sim' -e 0,0,40,1250,1480
wmctrl -r 'live-slam.rviz' -b remove,maximized_vert,maximized_horz
wmctrl -r 'live-slam.rviz' -e 0,1260,40,1280,1480
sleep 2
gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 2000 --req 'data: ""' || true
gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 2000 --req 'pose: {position: {x: 0, y: 0, z: 5.5}, orientation: {w: 0.70710678, y: 0.70710678}}' || true
wmctrl -lpG
