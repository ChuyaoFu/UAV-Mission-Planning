#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile=${1:-source}
estimator=${2:-vision}
case "$profile" in
    source) world=dronehacks;;
    textured) world=dronehacks_textured;;
    reference) world=visual_nav;;
    *) echo 'Usage: bash scripts/launch-environment.sh source|textured|reference vision|gps' >&2; exit 2;;
esac
case "$estimator" in vision|gps) ;; *) echo 'Estimator must be vision or gps' >&2;exit 2;;esac
bash scripts/run-container.sh
if sudo docker exec visual-nav-sim tmux has-session -t nav-gz 2>/dev/null; then
    echo 'A simulation is already running. Stop it before changing profiles.' >&2;exit 1
fi
sudo docker exec -e NAV_WORLD="$world" -e NAV_ESTIMATOR="$estimator" visual-nav-sim bash /work/scripts/start-sim.sh
sudo docker exec visual-nav-sim bash /work/scripts/start-vio.sh
sudo docker exec visual-nav-sim tmux new-session -d -s nav-monitor 'exec /opt/px4-venv/bin/python3 /work/scripts/gcs-monitor.py > /work/logs/monitor.log 2>&1'
if [ "$estimator" = vision ]; then
    sudo docker exec visual-nav-sim tmux new-session -d -s nav-ev 'source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; exec /usr/bin/python3 /work/scripts/vio-to-px4.py > /work/logs/ev-adapter.log 2>&1'
fi
echo "Started profile=$profile estimator=$estimator. Vehicle remains disarmed; no test flight is launched."
