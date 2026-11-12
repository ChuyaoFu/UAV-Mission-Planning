#!/usr/bin/env bash
# OFF-DRONE: fake skybeta camera (real captured pairs, 2 x 1024x640 @ 30 Hz + camera_info + TF) -> stereo depth node
# (PRESET) -> bench_check; then SIGINT the node and time its exit. Runs in the team13-replay container (domain 79).
# Usage: replay_stereo_only.sh PRESET [CPUS for the stereo node, e.g. 4,5] [cv_threads]
source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash 2>/dev/null
set -u
export ROS_DOMAIN_ID=79 PYTHONUNBUFFERED=1 HOME=/tmp
R=/r; OFF=$R/official; export PYTHONPATH=$R/skybeta/src/skybeta_flight_interface:${PYTHONPATH:-}
PRESET=$1; CPUS=${2:-4,5}; TH=${3:-2}; BUSY=${BUSY:-0}; EXTRA=${EXTRA:-}
OUT=$R/out/stereo_$(date +%H%M%S)_$PRESET; mkdir -p "$OUT"; echo "[stereo-only] $OUT busy=$BUSY extra=$EXTRA"
BUSY_PIDS=""
for i in $(seq 1 $BUSY); do taskset -c "$CPUS" python3 -c "while True: pass" & BUSY_PIDS="$BUSY_PIDS $!"; done
python3 $OFF/realdrone/tools/replay_fake_drone.py --npz $R/pair1.npz --duration 60 --no-switch > "$OUT/fake.log" 2>&1 &
FD=$!
sleep 2
taskset -c "$CPUS" python3 $OFF/realdrone/stereo_depth_node.py --ros-args -p rate_hz:=10.0 -p preset:="$PRESET" -p cv_threads:=$TH $EXTRA > "$OUT/stereo.log" 2>&1 &
ST=$!
sleep 8
python3 $OFF/realdrone/tools/bench_check.py 20 "$OUT/depth.npz" > "$OUT/check.json" 2> "$OUT/check.err"
T0=$(date +%s.%N); kill -INT $ST
for i in $(seq 1 50); do kill -0 $ST 2>/dev/null || break; sleep 0.1; done
T1=$(date +%s.%N)
if kill -0 $ST 2>/dev/null; then echo "exit_after_sigint: STILL RUNNING after 5 s" | tee "$OUT/exit.txt"; kill -9 $ST
else wait $ST; RC=$?; echo "exit_after_sigint_s: $(echo "$T1 - $T0" | bc) rc $RC" | tee "$OUT/exit.txt"; fi
[ -n "$BUSY_PIDS" ] && kill $BUSY_PIDS 2>/dev/null
kill -INT $FD 2>/dev/null; sleep 1; kill $FD 2>/dev/null; wait 2>/dev/null
grep -h '"hz"' "$OUT/stereo.log" | tail -3
python3 -c "import json;t=open('$OUT/check.json').read();d=json.loads(t[:t.rindex('}')+1]);print({k:d[k] for k in ('/camera/depth/image_raw','depth_left_same_stamp') if k in d})"
chown -R 1000:1000 "$OUT"
