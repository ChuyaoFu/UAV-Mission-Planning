#!/usr/bin/env bash
set -euo pipefail
root=/work/cartographer/v5
if ! tmux has-session -t nav-gui 2>/dev/null;then
 tmux new-session -d -s nav-gui "source /opt/ros/jazzy/setup.bash; DISPLAY=:0 XAUTHORITY=/tmp/visual-nav-Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia GZ_PARTITION=visual_nav_sim exec gz sim -g > $root/logs/gazebo-gui.log 2>&1"
fi
if ! tmux has-session -t v5-rviz 2>/dev/null;then
 tmux new-session -d -s v5-rviz "source /opt/ros/jazzy/setup.bash; DISPLAY=:0 XAUTHORITY=/tmp/visual-nav-Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia exec rviz2 -d $root/config/live-slam.rviz --ros-args -p use_sim_time:=true > $root/logs/rviz.log 2>&1"
fi
if ! tmux has-session -t v5-view 2>/dev/null;then
 tmux new-session -d -s v5-view "source /opt/ros/jazzy/setup.bash; exec /usr/bin/python3 $root/scripts/online_view.py > $root/logs/view.log 2>&1"
fi
export DISPLAY=:0 XAUTHORITY=/tmp/visual-nav-Xauthority GZ_PARTITION=visual_nav_sim
for i in $(seq 1 20);do
 if wmctrl -l | grep -qi rviz && wmctrl -l | grep -q 'Gazebo Sim';then break;fi
 sleep 1
done
gazebo_id=$(wmctrl -l | awk '/Gazebo Sim/{print $1;exit}')
rviz_id=$(wmctrl -l | awk 'tolower($0) ~ /rviz/{print $1;exit}')
[[ -n "$gazebo_id" && -n "$rviz_id" ]] || { echo 'Both actual windows must be available';exit 1; }
read -r screen_width screen_height < <(xdpyinfo | awk '/dimensions:/{split($2,a,"x");print a[1],a[2];exit}')
[[ "$screen_width" =~ ^[0-9]+$ && "$screen_height" =~ ^[0-9]+$ ]] || exit 1
left_width=$((screen_width/2-8));right_width=$((screen_width-left_width-8));window_height=$((screen_height-100))
wmctrl -ir "$gazebo_id" -b remove,maximized_vert,maximized_horz
wmctrl -ir "$rviz_id" -b remove,maximized_vert,maximized_horz
wmctrl -ir "$rviz_id" -T 'RViz - Live SLAM'
wmctrl -ir "$gazebo_id" -e "0,0,40,$left_width,$window_height"
wmctrl -ir "$rviz_id" -e "0,$((left_width+8)),40,$right_width,$window_height"
sleep 2
gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 2000 --req 'data: ""' || true
gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 2000 --req 'pose: {position: {x: 0, y: 0, z: 5.5}, orientation: {w: 0.70710678, y: 0.70710678}}' || true
wmctrl -lpG
