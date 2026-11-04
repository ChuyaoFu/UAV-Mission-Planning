#!/usr/bin/env bash
set -euo pipefail
# Use the existing local desktop session; do not change xhost access controls.
auth_file=${XAUTHORITY:-}
if [ -z "$auth_file" ]; then
    auth_file=$(find "/run/user/$(id -u)" -maxdepth 1 -name '.mutter-Xwaylandauth.*' -print -quit)
fi
if [ -z "$auth_file" ] || [ ! -r "$auth_file" ]; then
    echo 'Run from the Ubuntu desktop terminal with DISPLAY and XAUTHORITY available.' >&2;exit 1
fi
display=${DISPLAY:-:0}
sudo docker cp "$auth_file" visual-nav-sim:/tmp/visual-nav-Xauthority >/dev/null
sudo docker exec visual-nav-sim tmux new-session -d -s nav-gui "source /opt/ros/jazzy/setup.bash; DISPLAY=$display XAUTHORITY=/tmp/visual-nav-Xauthority __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia GZ_PARTITION=visual_nav_sim exec gz sim -g > /work/logs/gui.log 2>&1"
echo 'Gazebo GUI requested on the Ubuntu desktop; see logs/gui.log.'
