#!/usr/bin/env bash
# OFF-DRONE integration run inside an isolated ROS 2 container (team13-replay on uav-ubuntu, --network none):
# fake drone (real captured stereo pairs + fake PX4/pilot) + the organizers' real flight_interface + our stereo depth
# node + mission (tools/flight_run.py --io skybeta) + bench_check. Usage: replay_run.sh NAME 'PARAMS_JSON' [fake args]
source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=77 PYTHONUNBUFFERED=1 HOME=/tmp
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/ros/jazzy/bin
R=/r; OFF=$R/official; export PYTHONPATH=$R/skybeta/src/skybeta_flight_interface:${PYTHONPATH:-}
NAME=$1; PARAMS=$2; shift 2
OUT=$R/out/$(date +%H%M%S)_$NAME; mkdir -p "$OUT"; echo "[replay] $OUT"
python3 $OFF/realdrone/tools/replay_fake_drone.py --npz $R/pair1.npz --duration 85 "$@" > "$OUT/fake.log" 2>&1 &
FD=$!
python3 -c "from skybeta_flight_interface.flight_interface import main; main()" --ros-args -p auto_offboard_on_arm:=true > "$OUT/flight_interface.log" 2>&1 &
FI=$!
python3 $OFF/realdrone/stereo_depth_node.py --ros-args -p rate_hz:=10.0 > "$OUT/stereo.log" 2>&1 &
ST=$!
sleep 3
( cd $OFF && FLIGHTS_DIR="$OUT/flights" python3 tools/flight_run.py --io skybeta --name "$NAME" --prior mission/course_prior.example.json \
    --params "$PARAMS" --timeout 70 > "$OUT/flight_run.log" 2>&1 ) &
FR=$!
sleep 4
python3 $OFF/realdrone/tools/bench_check.py 20 "$OUT/depth.npz" /skybeta/setpoint/pose > "$OUT/check.json" 2> "$OUT/check.err"
wait $FR; wait $FD
kill -INT $FI $ST 2>/dev/null; sleep 2; kill $FI $ST 2>/dev/null; wait 2>/dev/null
chown -R 1000:1000 "$OUT"
echo "[replay] done $OUT"
