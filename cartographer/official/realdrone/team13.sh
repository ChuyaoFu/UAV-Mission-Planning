#!/usr/bin/env bash
# Team 13 real-drone launcher (run on the Mac, repo root). Wraps deploy, bench and staged flights on skybeta-05.
# The organizers' flight_interface owns PX4; the SAFETY PILOT arms (switch on Altitude, throttle low), then switches
# to Position within a few seconds -> Offboard -> our mission takes off. The pilot can take over at any time and
# lands the vehicle when the mission prints "PILOT: LAND NOW" (the interface has no land command, floor 0.3 m).
#
#   bash cartographer/official/realdrone/team13.sh status        # read-only health check
#   bash cartographer/official/realdrone/team13.sh deploy        # copy + build our code on the drone (no flight)
#   bash cartographer/official/realdrone/team13.sh bench         # props OFF: stereo depth + dry-run mission, no setpoints
#   bash cartographer/official/realdrone/team13.sh fly hover     # stage: hover | corridor | tunnel | gate | land | full
#   bash cartographer/official/realdrone/team13.sh use blar     # run the teammates' algorithm (src/team13_mission) instead
#   bash cartographer/official/realdrone/team13.sh which         # which algorithm is running now
#   bash cartographer/official/realdrone/team13.sh stop          # stop whichever algorithm runs in the 'algo' window
#
# Only ONE algorithm ever runs: both live in the tmux window 'algo' of the stack, and starting one kills the other
# (and stops the organizers' example in the 'competitor' window). drone.env is never edited.
#   bash cartographer/official/realdrone/team13.sh pull          # copy flight records to outputs/flights/realdrone/
#
# Env: PRIOR (default mission/course_prior.site.json), SPEED (v_cruise, default 0.4), IMAGES=1 to bag images.
set -euo pipefail
HOST="${DRONE_HOST:-dronehacks@skybeta-05.local}"
EXPECT="${DRONE_HOSTNAME:-skybeta-05}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
PRIOR="${PRIOR:-mission/course_prior.site.json}"
SPEED="${SPEED:-0.4}"
IMAGES="${IMAGES:-0}"
DX="docker exec skybeta-stack"

check_host() {
  local h; h="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${HOST}" hostname)"
  [[ "${h}" == "${EXPECT}" ]] || { echo "REFUSING: ${HOST} is '${h}', expected '${EXPECT}' (never touch another team's drone)" >&2; exit 1; }
}

stage_params() {
  local stop=""
  case "$1" in
    hover)    stop='"stop_after": "takeoff", "stop_hover_s": 10' ;;
    corridor) stop='"stop_after": "corridor"' ;;
    tunnel)   stop='"stop_after": "tunnel_traverse"' ;;
    gate)     stop='"stop_after": "window_traverse"' ;;
    land)     stop='"stop_after": "land_target"' ;;
    full)     stop='' ;;
    *) echo "unknown stage '$1' (hover|corridor|tunnel|gate|land|full)" >&2; exit 2 ;;
  esac
  echo "{\"pilot_mode\": true, \"pilot_pos_gain\": 0.95, \"v_cruise\": ${SPEED}${stop:+, ${stop}}}"
}

stop_all() {
  # never two algorithms / two flight_interfaces: stop the example and whatever runs in 'algo' (and the old 'team13')
  ssh "${HOST}" "${DX} tmux send-keys -t skybeta:competitor C-c 2>/dev/null; for w in algo team13; do ${DX} tmux send-keys -t skybeta:\$w C-c 2>/dev/null; done; sleep 4; for w in algo team13; do ${DX} tmux kill-window -t skybeta:\$w 2>/dev/null; done; pgrep -f 'flight_interface' >/dev/null && echo 'WARNING: a flight_interface is still running' || echo 'all algorithms stopped'" || true
}

cmd="${1:-status}"; shift || true
case "${cmd}" in
  status)
    check_host
    ssh "${HOST}" "python3 - <<'PY'
import json
s = json.load(open('/run/skybeta-status/ros_summary.json'))
t = s.get('topics', {})
for k in ('vehicle_status', 'vehicle_local_position', 'slam_odometry', 'cam_left_info', 'skybeta_state'):
    v = t.get(k, {}); print(f\"{k:24s} {v.get('rate_hz')} Hz, age {v.get('age_s')} s\")
PY
docker ps --format '{{.Names}} {{.Status}}'
ls ~/team_ws/team13_pkgs/team13_nav/official/${PRIOR} 2>/dev/null && echo 'prior: ok' || echo 'prior: MISSING (deploy first)'
${DX} tmux list-windows -t skybeta -F '#W' 2>/dev/null | tr '\n' ' '; echo"
    ;;
  deploy)
    bash "${HERE}/deploy.sh"
    ;;
  bench)
    check_host
    ssh "${HOST}" "${DX} bash /root/team_ws/team13_pkgs/team13_nav/official/realdrone/tools/bench_live.sh"
    ;;
  fly)
    stage="${1:?usage: fly hover|corridor|tunnel|gate|land|full}"
    check_host
    params="$(stage_params "${stage}")"
    name="${stage}_$(date +%H%M%S)"
    echo "[team13] stage=${stage} prior=${PRIOR} params=${params}"
    stop_all
    # 2) our launch in its own tmux window: flight_interface (auto_offboard_on_arm) + stereo depth + mission + record
    launch="source skybeta_env.sh; ros2 launch team13_nav team13.launch.py name:=${name} prior:=${PRIOR} params_json:='${params}' record_images:=$([[ ${IMAGES} == 1 ]] && echo true || echo false); exec bash"
    ssh "${HOST}" "${DX} tmux new-window -d -t skybeta -n algo \"bash -lc \\\"${launch}\\\"\""
    sleep 8
    ssh "${HOST}" "${DX} tmux capture-pane -p -t skybeta:algo | tail -15"
    cat <<EOF

[team13] READY when the log shows the mission waiting for offboard_active.
  Pilot: kill switch tested, props clear, switch on ALTITUDE, throttle LOW -> ARM -> within a few seconds switch to POSITION.
  The mission takes off by itself (must leave the ground within 10 s of arming).
  Take over any time: raise throttle to centre / move a stick (-> Position), or kill switch.
  When the log says "PILOT: LAND NOW", land manually. Then: bash $0 stop ; bash $0 pull
  Watch live: ssh ${HOST} -t '${DX} tmux attach -t skybeta'   (window 'algo'; Ctrl-b d to leave)
EOF
    ;;
  use)
    [[ "${1:-}" == "blar" ]] || { echo "usage: use blar   (ours = the 'fly' command)"; exit 2; }
    check_host
    stop_all
    launch="source skybeta_env.sh; ros2 launch team13_mission team13_mission.launch.py; exec bash"
    ssh "${HOST}" "${DX} tmux new-window -d -t skybeta -n algo \"bash -lc \\\"${launch}\\\"\""
    sleep 8
    ssh "${HOST}" "${DX} tmux capture-pane -p -t skybeta:algo | tail -15"
    echo "[team13] teammates' algorithm (team13_mission) is running in window 'algo'. Pilot procedure is the same."
    ;;
  which)
    check_host
    ssh "${HOST}" "pgrep -af 'team13_nav|team13_mission|competitor_example|flight_interface' | grep -v pgrep | cut -c1-150"
    ;;
  stop)
    check_host
    stop_all
    echo "Stopped. To bring back the organizers' example, restart the stack (drone on the ground, still): docker rm -f skybeta-stack"
    ;;
  pull)
    check_host
    mkdir -p "${REPO}/outputs/flights/realdrone"
    rsync -a "${HOST}:team_ws/team13_flights/" "${REPO}/outputs/flights/realdrone/"
    ls -t "${REPO}/outputs/flights/realdrone" | head
    ;;
  *) sed -n 2,16p "$0"; exit 2 ;;
esac
