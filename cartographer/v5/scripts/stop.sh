#!/usr/bin/env bash
set -euo pipefail
if tmux has-session -t nav-px4 2>/dev/null;then
 status=$(/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-listener vehicle_status -n 1)
 [[ "$status" =~ arming_state:[[:space:]]1 ]] || { echo 'Refuse reset until confirmed disarm';exit 1; }
fi
for name in $(tmux list-sessions -F '#S' 2>/dev/null | grep '^v5-' | grep -v '^v5-build$' || true);do tmux send-keys -t "$name" C-c;done
sleep 2
for name in $(tmux list-sessions -F '#S' 2>/dev/null | grep '^v5-' | grep -v '^v5-build$' || true);do tmux kill-session -t "$name";done
bash /work/scripts/stop-sim.sh
pkill -TERM -f '^gz sim (-v 4 )?-r -s --headless-rendering /work/sim/worlds/cadfree_unknown.sdf$' || true
