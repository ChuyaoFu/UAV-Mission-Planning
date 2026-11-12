#!/usr/bin/env bash
# Team 13 launcher, run ON THE ORIN (ssh dronehacks@<drone>, then call this). Same commands as the Mac-side team13.sh.
#   bash ~/team_ws/team13_pkgs/team13_nav/official/realdrone/team13_onboard.sh status
#   ... which | use blar | fly hover|corridor|tunnel|gate|land|full | stop | bench | dryrun [seconds]
#   ... axis   (control check: x, y, z, yaw steps; AXIS_ARGS="--d 0.3" to change)
#   ... wall WALL_DIST WALL_H LAND_DIST   (over the wall, land behind, re-arm, back, land at start)
#   ... record [minutes] [rate]   (MANUAL RC flight: record stereo pairs + VIO/PX4 only, no team control)
# Only ONE algorithm runs: both live in tmux window 'algo' of the skybeta stack; starting one stops the other and the
# organizers' example. drone.env is never edited. Env: PRIOR, SPEED (default 0.4), IMAGES=1.
set -euo pipefail
[[ "$(hostname)" == skybeta-05 ]] || { echo "REFUSING: this is $(hostname), not skybeta-05" >&2; exit 1; }
PRIOR="${PRIOR:-mission/course_prior.site.json}"; SPEED="${SPEED:-0.4}"; IMAGES="${IMAGES:-0}"
# stereo noise filters validated on the real 2026-11-08 manual flights (realdrone-REPORT 5.8): on by default for fly;
# STEREO_ARGS="" turns them off
STEREO_ARGS="${STEREO_ARGS-noise_minsup:=150 noise_minsup_diff:=1.0 sgbm_speckle_window:=800 sgbm_speckle_range:=2}"
# the same filters as node params, for the teammate's run (her CBF needs our depth); BLAR_STEREO="" turns them off
BLAR_STEREO="${BLAR_STEREO--p noise_minsup_px:=150 -p noise_minsup_diff_px:=1.0 -p sgbm_speckle_window:=800 -p sgbm_speckle_range:=2}"
DX="docker exec skybeta-stack"

stop_all() {
  # close our standalone bag recorders cleanly first (index written), then the algorithms
  ${DX} pkill -INT -f 'record_manual.py' 2>/dev/null && sleep 3 || true
  ${DX} pkill -INT -f 'team13_stereo_depth' 2>/dev/null && sleep 2 || true
  ${DX} pkill -INT -f 'ros2 bag record -o /root/team_ws/team13_flights' 2>/dev/null && sleep 2 || true
  ${DX} tmux send-keys -t skybeta:competitor C-c 2>/dev/null || true
  for w in algo team13; do ${DX} tmux send-keys -t "skybeta:${w}" C-c 2>/dev/null || true; done
  sleep 4
  for w in algo team13; do ${DX} tmux kill-window -t "skybeta:${w}" 2>/dev/null || true; done
  if pgrep -f flight_interface >/dev/null; then echo 'WARNING: a flight_interface is still running'; else echo 'all algorithms stopped'; fi
}
preflight() {   # PX4 local vs VIO odom must agree (flight_interface flies setpoints in PX4's frame); NOCHECK=1 skips
  [[ "${NOCHECK:-0}" == 1 ]] && { echo '[preflight] SKIPPED (NOCHECK=1)'; return 0; }
  ${DX} bash -lc "source skybeta_env.sh >/dev/null 2>&1; python3 /root/team_ws/team13_pkgs/team13_nav/official/realdrone/tools/preflight_offset.py" \
    || { echo 'NOT STARTED (nothing was stopped or launched).'; exit 3; }
}
start_algo() {  # $1 = command line run inside the container
  ${DX} tmux new-window -d -t skybeta -n algo "bash -lc \"source skybeta_env.sh; $1; exec bash\""
  sleep 8; ${DX} tmux capture-pane -p -t skybeta:algo | tail -15
}
stage_params() {
  local stop=""
  case "$1" in
    hover) stop='"stop_after": "takeoff", "stop_hover_s": 10' ;;  corridor) stop='"stop_after": "corridor"' ;;
    tunnel) stop='"stop_after": "tunnel_traverse"' ;;  gate) stop='"stop_after": "window_traverse"' ;;
    land) stop='"stop_after": "land_target"' ;;  full) stop='' ;;
    *) echo "unknown stage '$1' (hover|corridor|tunnel|gate|land|full)" >&2; exit 2 ;;
  esac
  echo "{\"pilot_mode\": true, \"pilot_pos_gain\": 0.95, \"v_cruise\": ${SPEED}${stop:+, ${stop}}}"
}

cmd="${1:-status}"; shift || true
case "${cmd}" in
  status)
    python3 -c "
import json; t=json.load(open('/run/skybeta-status/ros_summary.json')).get('topics', {})
for k in ('vehicle_status','vehicle_local_position','slam_odometry','cam_left_info','skybeta_state'):
    v=t.get(k,{}); print(f\"{k:24s} {v.get('rate_hz')} Hz, age {v.get('age_s')} s\")"
    docker ps --format '{{.Names}} {{.Status}}'
    ls ~/team_ws/team13_pkgs/team13_nav/official/${PRIOR} >/dev/null 2>&1 && echo 'prior: ok' || echo 'prior: MISSING'
    ;;
  which) pgrep -af 'team13_nav|team13_mission|competitor_example|flight_interface' | grep -v -E 'pgrep|team13_onboard' | cut -c1-150 || echo 'nothing running' ;;
  use)
    [[ "${1:-}" == blar ]] || { echo 'usage: use blar   (ours = fly <stage>)'; exit 2; }
    # record everything needed to debug her run: bag (pose, VIO, PX4, setpoints, her /team13/status, /rosout),
    # the launch output, the preflight result and a snapshot of her code. IMAGES=1 adds the raw stereo pair.
    stamp=$(date +%Y%m%d-%H%M%S); HD=/home/dronehacks/team_ws/team13_flights/${stamp}_blar; CD=/root/team_ws/team13_flights/${stamp}_blar
    mkdir -p "${HD}"
    preflight 2>&1 | tee "${HD}/preflight.txt"; [[ ${PIPESTATUS[0]} == 0 ]] || exit 3
    stop_all
    # her CBF needs a depth topic and the drone has none: by default start OUR stereo depth and pass it to her.
    # OUR_DEPTH=0: don't start ours (she provides depth_topic:=... herself via BLAR_ARGS).
    if [[ "${OUR_DEPTH:-1}" == 1 ]]; then
      stereo_cmd="ros2 run team13_nav team13_stereo_depth --ros-args -p rate_hz:=10.0 ${BLAR_STEREO} > ${CD}/stereo.log 2>&1 & sleep 4;"
      depth_args='stereo_depth:=false depth_topic:=/camera/depth/image_raw depth_info_topic:=/camera/depth/camera_info'
    else stereo_cmd=''; depth_args=''; fi   # OUR_DEPTH=0: her launch starts its own stereo depth (stereo_depth:=true default)
    topics='/tf /tf_static /oak/imu/data /visual_slam/tracking/odometry /visual_slam/status /skybeta/state /skybeta/setpoint/pose /skybeta/setpoint/twist /fmu/out/vehicle_local_position /fmu/out/vehicle_attitude /fmu/out/vehicle_status_v1 /fmu/out/vehicle_land_detected /fmu/in/trajectory_setpoint /fmu/in/offboard_control_mode /fmu/in/vehicle_visual_odometry /oak/left/camera_info /oak/right/camera_info /team13/status /rosout /camera/depth/camera_info'
    [[ ${IMAGES} == 1 ]] && topics="${topics} /oak/left/image_raw /oak/right/image_raw /camera/depth/image_raw"
    start_algo "tar czf ${CD}/code.tar.gz --exclude=__pycache__ --exclude=.pytest_cache -C /root/team_ws/src team13_mission; ros2 bag record -o ${CD}/bag -s mcap ${topics} > ${CD}/bag.log 2>&1 & ${stereo_cmd} ros2 launch team13_mission team13_mission.launch.py ${depth_args} ${BLAR_ARGS:-} 2>&1 | tee ${CD}/launch.log"
    if ${DX} tmux capture-pane -p -t skybeta:algo -S -200 | grep -q -E 'Traceback|process has died'; then
      echo 'ERROR: her mission failed to start (see window algo) - stopping, do NOT arm'; stop_all; exit 1
    fi
    echo "[team13] teammates' algorithm + our stereo depth (for her CBF) running in window 'algo'. Recording to ${HD}/ (bag, launch.log, stereo.log, preflight.txt, code.tar.gz)" ;;
  fly)
    stage="${1:?usage: fly hover|corridor|tunnel|gate|land|full}"; params="$(stage_params "${stage}")"
    # start_algo wraps the command in bash -lc "..." (double quotes): escape the JSON's quotes or they are eaten
    params_q=${params//\"/\\\"}
    echo "[team13] params: ${params}"
    preflight
    stop_all
    start_algo "ros2 launch team13_nav team13.launch.py name:=${stage}_$(date +%H%M%S) prior:=${PRIOR} params_json:='${params_q}' record_images:=$([[ ${IMAGES} == 1 ]] && echo true || echo false) ${STEREO_ARGS}"
    if ${DX} tmux capture-pane -p -t skybeta:algo -S -200 | grep -q -E 'Traceback|JSONDecodeError|process has died'; then
      echo 'ERROR: our mission node failed to start (see window algo) - stopping, do NOT arm'; stop_all; exit 1
    fi
    echo "[team13] our algorithm (${stage}) running in window 'algo'. Pilot: ALTITUDE + throttle low -> ARM -> POSITION within seconds."
    echo "  Land manually when the log says 'PILOT: LAND NOW'. Watch: docker exec -it skybeta-stack tmux attach -t skybeta" ;;
  axis)
    # control check, no mission / no stereo: organizers' flight_interface + bag + tools/axis_test.py
    # (takeoff 0.8 m -> fwd 0.5 & back -> left 0.5 & back -> up 0.3 & back -> yaw +45/-45 & back -> 0.3 m floor)
    preflight
    stop_all
    start_algo "python3 /root/team_ws/team13_pkgs/team13_nav/official/realdrone/tools/axis_test.py ${AXIS_ARGS:-} & ros2 launch team13_nav team13.launch.py name:=axis_$(date +%H%M%S) mission:=false stereo:=false"
    if ${DX} tmux capture-pane -p -t skybeta:algo -S -200 | grep -q -E 'Traceback|process has died'; then
      echo 'ERROR: axis test failed to start (see window algo) - stopping, do NOT arm'; stop_all; exit 1
    fi
    echo "[team13] AXIS TEST running in window 'algo' (~110 s in Offboard, incl. 2 loops r 0.6 m). Pilot: ALTITUDE + throttle low -> ARM -> POSITION."
    echo "  Land manually at 'PILOT: LAND NOW'. Any stick/mode change = takeover, the test then stops for good." ;;
  wall)
    # over the wall, land behind it (pilot lands + disarms), re-arm, back over, land at the start (tools/wall_hop.py)
    # usage: wall WALL_DIST WALL_H LAND_DIST [extra wall_hop.py args]   e.g.  wall 1.2 1.73 2.4
    [[ $# -ge 3 ]] || { echo 'usage: wall WALL_DIST WALL_H LAND_DIST [--land-left L --margin M --v V]  (metres, tape-measured)'; exit 2; }
    wargs="--wall-dist $1 --wall-h $2 --land-dist $3 ${*:4}"
    ${DX} python3 /root/team_ws/team13_pkgs/team13_nav/official/realdrone/tools/wall_hop.py ${wargs} --check \
      || { echo 'ERROR: wall geometry rejected (see above) - nothing started'; exit 2; }
    preflight
    stop_all
    start_algo "python3 /root/team_ws/team13_pkgs/team13_nav/official/realdrone/tools/wall_hop.py ${wargs} & ros2 launch team13_nav team13.launch.py name:=wall_$(date +%H%M%S) mission:=false stereo:=false"
    if ${DX} tmux capture-pane -p -t skybeta:algo -S -200 | grep -q -E 'Traceback|process has died'; then
      echo 'ERROR: wall hop failed to start (see window algo) - stopping, do NOT arm'; stop_all; exit 1
    fi
    echo "[team13] WALL HOP running in window 'algo'. Drone FACING the wall, disarmed. Pilot: ALTITUDE, throttle low -> ARM -> POSITION, hands off."
    echo "  Leg 1 ends at 'PILOT: LAND NOW' behind the wall: land + DISARM. Then re-ARM -> POSITION for leg 2. Do NOT restart the stack in between." ;;
  record)
    # MANUAL flight, data only: the pilot flies on RC (Position). No team code runs, nothing is published, so
    # Offboard can never engage. Records stereo pairs (gray PNG, RATE Hz) + a small bag of VIO/PX4/IMU/TF.
    # usage: record [minutes=15] [rate_hz=10]
    mins="${1:-15}"; rate="${2:-10}"
    stamp=$(date +%Y%m%d-%H%M%S); HD=/home/dronehacks/team_ws/team13_flights/${stamp}_manual; CD=/root/team_ws/team13_flights/${stamp}_manual
    stop_all
    mkdir -p "${HD}"
    ${DX} bash -lc "source skybeta_env.sh >/dev/null 2>&1; python3 /root/team_ws/team13_pkgs/team13_nav/official/realdrone/tools/preflight_offset.py --max-m 99" > "${HD}/preflight.txt" 2>&1 || true
    topics='/tf /tf_static /oak/imu/data /visual_slam/tracking/odometry /visual_slam/status /oak/left/camera_info /oak/right/camera_info /fmu/out/vehicle_local_position /fmu/out/vehicle_attitude /fmu/out/vehicle_status_v1 /fmu/out/vehicle_land_detected /fmu/in/vehicle_visual_odometry /rosout'
    start_algo "ros2 bag record -o ${CD}/bag -s mcap ${topics} > ${CD}/bag.log 2>&1 & python3 /root/team_ws/team13_pkgs/team13_nav/official/realdrone/tools/record_manual.py --out ${CD} --rate ${rate} --minutes ${mins}; pkill -INT -f 'ros2 bag record -o ${CD}'"
    if ${DX} tmux capture-pane -p -t skybeta:algo -S -100 | grep -q -E 'Traceback'; then
      echo 'ERROR: recorder failed to start (see window algo)'; stop_all; exit 1
    fi
    echo "[team13] RECORDING (manual flight, no team control) to ${HD}/  - ${mins} min max, ${rate} Hz pairs."
    echo "  Pilot flies on RC as usual (Position). Stop: bash $0 stop   Progress: docker exec -it skybeta-stack tmux attach -t skybeta (window algo)" ;;
  stop) stop_all ;;
  bench) ${DX} bash -lc "PRIOR=${PRIOR} bash /root/team_ws/team13_pkgs/team13_nav/official/realdrone/tools/bench_live.sh" ;;
  dryrun)
    # Hand-carried calibration check: our stereo depth + mission in DRY-RUN (setpoints only on /team13/dry_run/...),
    # images + depth recorded. Nothing reaches /skybeta/setpoint or /fmu/in. Props OFF, drone NOT armed.
    secs="${1:-180}"
    stop_all
    start_algo "ros2 run team13_nav team13_stereo_depth --ros-args -p rate_hz:=10.0 & sleep 6; cd /root/team_ws/team13_pkgs/team13_nav/official && FLIGHTS_DIR=/root/team_ws/team13_flights python3 tools/flight_run.py --io skybeta --dry-run --name dryrun_calib --prior ${PRIOR} --params '{\\\"pilot_mode\\\": true}' --timeout ${secs} --notes 'hand-carried calibration dry run'; kill %1"
    echo "[team13] DRY RUN for ${secs} s (window 'algo'). Move the drone by hand now (protocol in the log/README). Records: ~/team_ws/team13_flights/*_dryrun_calib/" ;;
  *) sed -n 2,7p "$0"; exit 2 ;;
esac
