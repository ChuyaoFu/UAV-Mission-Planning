#!/usr/bin/env bash
# Live bench, props OFF, NO flight commands: stereo depth + mission (io:=skybeta, DRY-RUN setpoint topic) recorded by
# tools/flight_run.py (--light bag + log), plus bench_check.py (rates, TF, depth stats) and top. Nothing is published
# to /skybeta/setpoint/* or /fmu/in/*.
# Run inside the stack container:  bash /root/team_ws/team13_pkgs/team13_nav/official/realdrone/tools/bench_live.sh
# Env: RATE (stereo Hz, default 10), DUR (mission seconds, default 60), PRIOR (default mission/course_prior.example.json),
#      PRESET (stereo depth preset, default half320; see stereo_depth_node.OPTIONS), STEREO_ONLY=1 (no mission run),
#      NET_MASK=1 (stereo node with the cage-net phantom mask, net_mask.py), NET_MODEL=1 (model-based variant with the
#      fence of PRIOR and the VIO pose, net_model.py), DEPTH_SCAN=1 (planar scan /team13/depth_scan, depth_scan.py)
set -u
source skybeta_env.sh
OFF=/root/team_ws/team13_pkgs/team13_nav/official
LIB=/root/team_ws/install/team13_nav/lib/team13_nav
RATE="${RATE:-10.0}"; DUR="${DUR:-60}"; PRIOR="${PRIOR:-mission/course_prior.example.json}"; PRESET="${PRESET:-half320}"; NET_MASK="${NET_MASK:-0}"; NM=false; [ "${NET_MASK}" = 1 ] && NM=true
NET_MODEL="${NET_MODEL:-0}"; NMD=false; [ "${NET_MODEL}" = 1 ] && NMD=true
DEPTH_SCAN="${DEPTH_SCAN:-0}"; DSC=false; [ "${DEPTH_SCAN}" = 1 ] && DSC=true
OUT=/root/team_ws/team13_flights/bench_$(date +%Y%m%d-%H%M%S); mkdir -p "${OUT}"
echo "[bench] ${OUT}"
timeout -k 5 -s INT $((DUR + 45)) "${LIB}/team13_stereo_depth" --ros-args -p rate_hz:="${RATE}" -p preset:="${PRESET}" -p net_mask:="${NM}" -p net_model:="${NMD}" -p net_fence_json:="${PRIOR}" -p depth_scan:="${DSC}" > "${OUT}/stereo.log" 2>&1 &
SPID=$!
sleep 8
[ "${STEREO_ONLY:-0}" = 1 ] || ( cd "${OFF}" && FLIGHTS_DIR=/root/team_ws/team13_flights timeout -k 20 -s INT $((DUR + 40)) python3 tools/flight_run.py \
    --io skybeta --dry-run --light --name bench_dry --prior "${PRIOR}" \
    --params '{"pilot_mode": true, "pilot_pos_gain": 0.95}' --timeout "${DUR}" \
    --notes "bench, props off, dry-run setpoint topic" > "${OUT}/flight_run.log" 2>&1 ) &
FPID=$!
sleep 15
( top -b -d 5 -n 4 | grep -E "^%Cpu|load average|python3|component_c|MicroXRCE" > "${OUT}/top.txt" ) &
python3 "${OFF}/realdrone/tools/bench_check.py" 25 "${OUT}/bench_depth.npz" > "${OUT}/bench_check.json" 2> "${OUT}/bench_check.err"
wait "${FPID}"; kill -INT "${SPID}" 2>/dev/null; T0=$(date +%s.%N); wait "${SPID}"
echo "stereo exit after SIGINT: $(python3 -c "import time;print(round(time.time()-${T0},2))") s" | tee "${OUT}/stereo_exit.txt"
grep -h '"hz"' "${OUT}/stereo.log" | tail -4
chown -R "${HOST_UID:-1000}:${HOST_GID:-1000}" /root/team_ws/team13_flights 2>/dev/null || true
echo "[bench] done -> ${OUT}"
