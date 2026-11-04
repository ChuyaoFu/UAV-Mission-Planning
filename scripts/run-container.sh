#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if sudo docker container inspect visual-nav-sim >/dev/null 2>&1; then
    if [ "$(sudo docker inspect visual-nav-sim --format '{{.State.Running}}')" != true ]; then
        sudo docker start visual-nav-sim
    fi
    exit 0
fi
# Only /work is writable host storage; no host USB or autopilot ports passed.
sudo docker run -d --name visual-nav-sim --gpus all \
    --network host --ipc private --shm-size=2g --cpus=12 --memory=22g \
    --memory-swap=26g --init \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display \
    -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro \
    -v "$PWD:/work" visual-nav:jazzy-harmonic
