#!/usr/bin/env bash
# Extra sim containers for PARALLEL headless flights (run on the Ubuntu HOST; no sudo).
#   bash make_parallel_ct.sh [N]        # N = 2 -> official-compat-sim2; N = 3 -> also official-compat-sim3
# Same image / mounts as official-compat-sim (setup_host.sh), but: own network namespace (docker bridge, so DDS, gz
# transport, uXRCE and PX4 ports cannot meet the other stacks), ROS_DOMAIN_ID 44 / 45 and GZ_PARTITION
# official_compat2 / 3 (env.sh reads OC_DOMAIN / OC_PARTITION), and its own /compat/run (official-compat/run2, run3:
# logs, px4-rootfs, world, side bag). Flights still land in the shared profile/flights/. No X11 (headless only).
# Use: run_blind.sh TAG --parallel N   |   OC_CT=official-compat-sim2 bash fly_compat.sh NAME ...   |   docker rm -f to remove
set -eu
N=${1:-2}; H=/home/fcy/claude-work/official-compat
for k in $(seq 2 "$N"); do
  CT=official-compat-sim$k
  if docker inspect "$CT" > /dev/null 2>&1; then echo "$CT exists"; docker start "$CT" > /dev/null; continue; fi
  mkdir -p "$H/run$k"
  docker run -d --name "$CT" --gpus all --network bridge \
    -e NVIDIA_DRIVER_CAPABILITIES=all -e OC_DOMAIN=$((42 + k)) -e OC_PARTITION=official_compat$k \
    -v /home/fcy/visual_nav:/work:ro \
    -v "$H/PX4-Autopilot":/work/src/PX4-Autopilot \
    -v "$H":/compat \
    -v "$H/run$k":/compat/run \
    -v /home/fcy/claude-work/venue-replica/replica:/replica:ro \
    official-compat:snap sleep infinity > /dev/null
  echo "$CT started (domain $((42 + k)), partition official_compat$k, run dir $H/run$k)"
done
