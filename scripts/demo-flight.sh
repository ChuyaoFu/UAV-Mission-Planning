#!/usr/bin/env bash
# Desktop demonstration only: PX4 GPS SITL, independent of the VIO acceptance test.
set -euo pipefail
cd "$(dirname "$0")/.."
if sudo docker exec visual-nav-sim tmux has-session -t nav-px4 2>/dev/null; then
    status=$(sudo docker exec visual-nav-sim /work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-commander status)
    if ! [[ "$status" == *Disarmed* ]]; then
        echo 'Vehicle is not confirmed disarmed; finish the current flight first.' >&2
        exit 1
    fi
fi
sudo docker exec visual-nav-sim bash /work/scripts/stop-sim.sh
bash scripts/launch-environment.sh textured gps
bash scripts/open-gui.sh
echo 'GPS SITL demo ready. Starting in 12 seconds: take off 0.65 m, move forward 0.4 m, return, land.'
sleep 4
# Keep the model in view while it climbs. These are GUI camera services only.
sudo docker exec -e GZ_PARTITION=visual_nav_sim visual-nav-sim bash -c \
    'source /opt/ros/jazzy/setup.bash; gz service -s /gui/move_to --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 2000 --req '\''data: "x500_vio_0"'\''; gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 2000 --req '\''data: "x500_vio_0"'\''' || true
sleep 8
px4bin=/work/src/PX4-Autopilot/build/px4_sitl_default/bin
gps=$(sudo docker exec visual-nav-sim "$px4bin/px4-param" show -q EKF2_GPS_CTRL)
ev=$(sudo docker exec visual-nav-sim "$px4bin/px4-param" show -q EKF2_EV_CTRL)
if [ "$gps" != 7 ] || [ "$ev" != 0 ]; then
    echo "Unexpected estimator parameters: GPS=$gps EV=$ev; refusing this GPS demonstration." >&2
    exit 1
fi
label=demo-gps-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "results/$label"
sudo docker exec -e NAV_TEST_BODY_FORWARD=1 -e NAV_TEST_REQUIRE_VISION=0 visual-nav-sim bash -c \
    'source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; exec /usr/bin/python3 /work/scripts/flight-baseline.py' \
    2>&1 | tee "results/$label/flight.log"
cp results/baseline-flight.json "results/$label/baseline-flight.json"
echo "Demonstration completed and vehicle disarmed. Results: results/$label"
