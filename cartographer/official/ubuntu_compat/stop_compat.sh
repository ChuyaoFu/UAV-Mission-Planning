#!/usr/bin/env bash
# Stop the official-compat stack (sim only; every flight uses a fresh stack).
source "$(dirname "$0")/env.sh"
for s in $(tmux ls -F '#S' 2>/dev/null | grep '^oc-'); do tmux send-keys -t "$s" C-c 2>/dev/null; done
sleep 2
for s in $(tmux ls -F '#S' 2>/dev/null | grep '^oc-'); do tmux kill-session -t "$s" 2>/dev/null; done
pkill -INT -f 'gz sim .*replica_compat.sdf' 2>/dev/null; pkill -f 'px4 -w /compat/run/px4-rootfs' 2>/dev/null
pkill -f 'MicroXRCEAgent udp4 -p 8888' 2>/dev/null; pkill -f run_subscribe_msckf 2>/dev/null
pkill -f 'ubuntu_compat/(ev_bridge|oak_sim_driver).py' 2>/dev/null; pkill -f 'ubuntu_compat/bridge.yaml' 2>/dev/null
sleep 1; pkill -9 -f 'gz sim .*replica_compat.sdf' 2>/dev/null; pkill -9 -f 'px4 -w /compat/run/px4-rootfs' 2>/dev/null
echo "official-compat stack stopped"
exit 0
