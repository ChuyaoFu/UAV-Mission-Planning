#!/usr/bin/env bash
# Host-side entry. The repo is mounted at the same /work layout as the original.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root=$PWD
name=${UAV_CONTAINER:-uav-gazebo}
image=${UAV_IMAGE:-uav-gazebo:jazzy-harmonic}
if docker info >/dev/null 2>&1; then d=(docker); else d=(sudo docker); fi
inside() { "${d[@]}" exec "$name" bash -c 'source /opt/ros/jazzy/setup.bash; for setup in /work/ros_ws/install/setup.bash /work/cartographer/v2/ros_ws/install/setup.bash /work/cartographer/v4/carto_ws/install/setup.bash; do if [ -f "$setup" ]; then source "$setup"; fi; done; exec "$@"' bash "$@"; }
action=${1:-help};shift || true
case "$action" in
 build-image)
  "${d[@]}" build --platform linux/amd64 -f docker/Dockerfile.repro -t "$image" . "$@";;
 up)
  if "${d[@]}" container inspect "$name" >/dev/null 2>&1; then
   mount=$("${d[@]}" inspect "$name" --format '{{range .Mounts}}{{if eq .Destination "/work"}}{{.Source}}{{end}}{{end}}')
   [ "$mount" = "$root" ] || { echo 'Container belongs to another checkout; set a different UAV_CONTAINER.' >&2;exit 1; }
   "${d[@]}" start "$name" >/dev/null;exit 0
  fi
  extra=()
  mode=${UAV_GPU:-auto}
  if [ "$mode" = nvidia ] || { [ "$mode" = auto ] && command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; }; then
   extra+=(--gpus all)
   egl=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
   [ ! -f "$egl" ] || extra+=(-v "$egl:$egl:ro")
  else
   extra+=(-e LIBGL_ALWAYS_SOFTWARE=1 -e EGL_PLATFORM=surfaceless)
  fi
  [ ! -d /tmp/.X11-unix ] || extra+=(-v /tmp/.X11-unix:/tmp/.X11-unix:ro)
  "${d[@]}" run -d --name "$name" --label org.uav.handoff=true --platform linux/amd64 \
    --network host --ipc private --shm-size=2g --memory="${UAV_MEMORY:-12g}" \
    --memory-swap="${UAV_MEMORY_SWAP:-16g}" --init \
    -e ROS_DOMAIN_ID=42 -e GZ_PARTITION=visual_nav_sim \
    -v "$root:/work" "${extra[@]}" "$image";;
 build)
  inside env BUILD_JOBS="${BUILD_JOBS:-2}" bash /work/repro/build-workspaces.sh
  inside env BUILD_JOBS="${BUILD_JOBS:-2}" bash /work/repro/test-mapping.sh;;
 check) inside /usr/bin/python3 /work/repro/check-environment.py;;
 start)
  inside /usr/bin/python3 /work/repro/check-environment.py
  inside bash /work/cartographer/v4/scripts/start.sh source stereo;;
 smoke)
  inside /usr/bin/python3 /work/repro/check-environment.py
  inside bash /work/cartographer/v4/scripts/start.sh source stereo
  if inside /usr/bin/python3 /work/repro/smoke.py;then
   inside bash /work/cartographer/v4/scripts/stop.sh
  else
   echo 'Smoke failed; leaving disarmed diagnostics available. Use repro/ctl.sh stop.' >&2;exit 1
  fi;;
 gui)
  display=${DISPLAY:?Run gui in an Ubuntu desktop terminal with DISPLAY set}
  auth=${XAUTHORITY:-}
  if [ -z "$auth" ];then
   auth=$(find "/run/user/$(id -u)" -maxdepth 1 -name '.mutter-Xwaylandauth.*' -print -quit 2>/dev/null || true)
  fi
  [ -r "$auth" ] || { echo 'No readable desktop XAUTHORITY; run from the logged-in Ubuntu desktop.' >&2;exit 1; }
  "${d[@]}" cp "$auth" "$name:/tmp/visual-nav-Xauthority"
  "${d[@]}" exec -e DISPLAY="$display" -e XAUTHORITY=/tmp/visual-nav-Xauthority "$name" bash /work/repro/open-gui.sh;;
 trial)
  label=${1:?Provide a NEW label, for example trial-001}
  [[ "$label" =~ ^[[:alnum:]-]+$ ]] || { echo 'Label must be alphanumeric/hyphen.' >&2;exit 2; }
  shift
  "${d[@]}" exec -e DISPLAY="${DISPLAY:-:0}" -e XAUTHORITY=/tmp/visual-nav-Xauthority "$name" bash -c \
   'source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; source /work/cartographer/v4/carto_ws/install/setup.bash; exec /usr/bin/python3 /work/cartographer/v4/scripts/run_trial.py "$@"' bash "$label" --mission --route window --seconds 200 "$@";;
 finalize)
  label=${1:?Provide the completed trial label}
  [[ "$label" =~ ^[[:alnum:]-]+$ ]] || exit 2
  "${d[@]}" exec "$name" bash -c 'source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; source /work/cartographer/v4/carto_ws/install/setup.bash; exec /usr/bin/python3 /work/cartographer/v4/scripts/finalize_episode.py "$1"' bash "$label";;
 stop) inside bash /work/cartographer/v4/scripts/stop.sh;;
 shell) "${d[@]}" exec -it "$name" bash;;
 *) printf '%s\n' 'Usage: bash repro/ctl.sh build-image|up|build|check|smoke|start|gui|trial NEW-LABEL|finalize LABEL|stop|shell'
    printf '%s\n' 'start/gui are disarmed. trial explicitly commands a simulated flight.';;
esac
