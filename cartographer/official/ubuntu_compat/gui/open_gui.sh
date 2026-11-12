#!/usr/bin/env bash
# Live view on the Ubuntu desktop for a RUNNING official-compat stack (run inside official-compat-sim).
#   bash /compat/profile/ubuntu_compat/gui/open_gui.sh [--no-gz]
# Needs /compat/run/xauth (wildcard copy of the desktop's Xwayland cookie; see report) and DISPLAY :0.
# Windows: left half  "Gazebo Sim" GUI client on partition official_compat, camera following skybeta_0
#          right half "RViz - official-compat" (VIO vs truth paths in odom, stereo depth cloud, left + depth images)
# tmux sessions oc-gui-* (stop_compat.sh closes them too).
source "$(dirname "$0")/../env.sh"     # (no set -u: ROS setup.bash reads unset variables)
G=$UC/gui; NO_GZ=0; [ "${1:-}" = --no-gz ] && NO_GZ=1
export DISPLAY=:0 XAUTHORITY=$RUN/xauth QT_QPA_PLATFORM=xcb __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
xdpyinfo > /dev/null 2>&1 || { echo "no X access on :0 (refresh $RUN/xauth)"; exit 1; }
read -r SW SH < <(xdpyinfo | awk '/dimensions:/{split($2,a,"x");print a[1],a[2];exit}')
LW=$((SW / 2 - 8)); RW=$((SW - LW - 8)); WH=$((SH - 110))
# default Harmonic GUI, sized to half the screen, without the side panels (entity tree / inspector) -> big 3D view
/usr/bin/python3 - "$LW" "$WH" "$RUN/gz_gui.config" <<'PY'
import re, sys
s = open('/usr/share/gz/gz-sim8/gui/gui.config').read()
s = s.replace('<width>1000</width>', '<width>%s</width>' % sys.argv[1]).replace('<height>845</height>', '<height>%s</height>' % sys.argv[2])
for name in ('ComponentInspector', 'EntityTree'):
    s = re.sub(r'<plugin filename="%s".*?</plugin>\s*' % name, '', s, flags=re.S)
open(sys.argv[3], 'w').write(s)
PY
gui(){ local name=$1; shift; tmux kill-session -t "$name" 2>/dev/null; tmux new-session -d -s "$name" "source $UC/env.sh; export DISPLAY=:0 XAUTHORITY=$RUN/xauth QT_QPA_PLATFORM=xcb __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia; $* > $LOGS/$name.log 2>&1"; }
# close windows from an earlier open_gui (the gz GUI child process outlives its tmux session)
pkill -f "gz sim -g --gui-config" 2>/dev/null; pkill -f "gz_gui.config" 2>/dev/null; pkill -f "ubuntu_compat/gui/viz_paths.py" 2>/dev/null; pkill -f "official_compat.rviz" 2>/dev/null; sleep 2
gui oc-gui-paths /usr/bin/python3 $G/viz_paths.py --ros-args -p use_sim_time:=true
[ $NO_GZ = 1 ] || gui oc-gui-gz gz sim -g --gui-config $RUN/gz_gui.config
gui oc-gui-rviz rviz2 -d $G/official_compat.rviz --ros-args -p use_sim_time:=true
for i in $(seq 1 40); do
  wmctrl -l 2>/dev/null | grep -qi rviz && { [ $NO_GZ = 1 ] || wmctrl -l | grep -q 'Gazebo Sim'; } && break; sleep 1
done
R=$(wmctrl -l | awk 'tolower($0) ~ /rviz/{print $1; exit}'); Z=$(wmctrl -l | awk '/Gazebo Sim/{print $1; exit}')
if [ -n "$R" ]; then
  wmctrl -ir "$R" -b remove,maximized_vert,maximized_horz; wmctrl -ir "$R" -T 'RViz - official-compat (VIO green, truth magenta)'
  wmctrl -ir "$R" -e "0,$((LW + 8)),40,$RW,$WH"
fi
if [ -n "$Z" ]; then
  wmctrl -ir "$Z" -b remove,maximized_vert,maximized_horz; wmctrl -ir "$Z" -e "0,0,40,$LW,$WH"
  sleep 3
  gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 3000 --req 'data: "skybeta_0"' || true
  gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d --reptype gz.msgs.Boolean --timeout 3000 --req 'x: -0.6, y: 0.0, z: 1.9'   # world-frame offset: behind (south) and above, inside the cage || true
fi
wmctrl -lG
