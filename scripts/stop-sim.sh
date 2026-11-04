#!/usr/bin/env bash
set -euo pipefail
# Run inside visual-nav-sim only. Gazebo's launcher can survive a tmux HUP.
# Signal the foreground process before closing the sessions, then reap leftovers.
for session in nav-flight nav-ev nav-vio nav-evaluation nav-bridge nav-px4 nav-agent nav-gz nav-gui nav-monitor; do
    if tmux has-session -t "$session" 2>/dev/null; then
        tmux send-keys -t "$session" C-c
    fi
done
sleep 2
for session in nav-flight nav-ev nav-vio nav-evaluation nav-bridge nav-px4 nav-agent nav-gz nav-gui nav-monitor; do
    if tmux has-session -t "$session" 2>/dev/null; then
        tmux kill-session -t "$session"
    fi
done
pkill -TERM -f '^gz sim (-v 4 )?-r -s --headless-rendering /work/sim/worlds/(visual_nav|dronehacks|dronehacks_textured).sdf$' || true
