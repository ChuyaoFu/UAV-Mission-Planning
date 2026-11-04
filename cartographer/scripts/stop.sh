#!/usr/bin/env bash
set -euo pipefail
if tmux has-session -t nav-px4 2>/dev/null;then
 status=$(/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-listener vehicle_status -n 1)
 if ! [[ "$status" =~ arming_state:[[:space:]]1 ]];then echo "Refuse reset without confirmed disarm: $status";exit 1;fi
fi
for session in carto-flight carto-eval carto-contact carto-ev carto-slam carto-depth carto-stereo carto-range carto-range-bridge carto-judge;do
 if tmux has-session -t "$session" 2>/dev/null;then tmux send-keys -t "$session" C-c;fi
done
sleep 2
for session in carto-flight carto-eval carto-contact carto-ev carto-slam carto-depth carto-stereo carto-range carto-range-bridge carto-judge;do
 if tmux has-session -t "$session" 2>/dev/null;then tmux kill-session -t "$session";fi
done
bash /work/scripts/stop-sim.sh
