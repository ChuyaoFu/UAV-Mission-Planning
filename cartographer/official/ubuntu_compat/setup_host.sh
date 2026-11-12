#!/usr/bin/env bash
# ONE-TIME setup of the official-compat profile on the Ubuntu host (what was run on 2026-11-06).
#   bash setup_host.sh            (host user fcy, docker group; no sudo)
# 1. snapshot image of visual-nav-sim (keeps the packages Codex installed in its writable layer; v5 untouched)
# 2. private PX4 copy with /fmu/out/distance_sensor added to the uXRCE-DDS topic list, rebuilt in place
#    (mounted over /work/src/PX4-Autopilot inside OUR container only; /work itself is mounted read-only)
# 3. container official-compat-sim (GPU, host network, ROS_DOMAIN_ID 43, GZ_PARTITION official_compat)
set -euo pipefail
H=/home/fcy/claude-work/official-compat
mkdir -p $H/logs $H/run $H/profile/flights
docker image inspect official-compat:snap >/dev/null 2>&1 || docker commit -m official-compat-snapshot visual-nav-sim official-compat:snap
if [ ! -d $H/PX4-Autopilot ]; then
  cp -a /home/fcy/visual_nav/src/PX4-Autopilot $H/PX4-Autopilot
  python3 - $H/PX4-Autopilot/src/modules/uxrce_dds_client/dds_topics.yaml <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
if '/fmu/out/distance_sensor' not in s:
    a = "  - topic: /fmu/out/battery_status\n"
    s = s.replace(a, "  - topic: /fmu/out/distance_sensor          # official-compat: official lab exposes the PX4 down range\n"
                     "    type: px4_msgs::msg::DistanceSensor\n    rate_limit: 50.\n\n" + a, 1)
    open(p, 'w').write(s)
PY
fi
docker rm -f official-compat-sim 2>/dev/null || true
docker run -d --name official-compat-sim --gpus all --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e ROS_DOMAIN_ID=43 -e GZ_PARTITION=official_compat \
  -v /home/fcy/visual_nav:/work:ro \
  -v $H/PX4-Autopilot:/work/src/PX4-Autopilot \
  -v $H:/compat \
  -v /home/fcy/claude-work/venue-replica/replica:/replica:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  official-compat:snap sleep infinity
docker exec official-compat-sim bash -lc "cd /work/src/PX4-Autopilot && make px4_sitl_default > /compat/logs/px4_build.log 2>&1; tail -n 2 /compat/logs/px4_build.log"
cp $H/profile/ubuntu_compat/fly_compat.sh $H/fly_compat.sh
echo "official-compat ready: bash $H/fly_compat.sh NAME -- [flight_run.py args]"
