#!/usr/bin/env bash
# ONE recorded flight of the UNCHANGED official node in the official-compat sim (run on the Ubuntu HOST).
#   bash fly_compat.sh NAME [--ev-source vio|gt] [--images gz|oak] [--depth ideal|stereo] [--reuse] [--keep] [--gui] [--profile DIR] [--replica DIR] [--px4-param NAME=VAL]... [-- flight_run.py args...]
#   --reuse: fly on the stack that is already running (e.g. with the live GUI attached) instead of a fresh one
#   --keep:  leave the stack (and any GUI windows) running after the flight
#   --gui:   fresh stack, then open the live GUI (gui/open_gui.sh), fly, and keep stack + windows (implies --keep)
#   --profile DIR: container path of the code tree to fly (default /compat/profile); e.g. a pinned snapshot made by
#            demo.sh (its flights/ and ubuntu_compat/ are symlinks to profile/, so the folder still lands in profile/flights)
#   --spawn F,L,YAW: vehicle start pose in the course start frame (per-stage tests; default the launch, north)
#   --replica DIR: container path of the venue replica to fly in (default /replica; warped copies: make_warp.py)
#   --px4-param NAME=VAL: set a PX4 parameter after the stack is up, before the flight (repeatable; e.g. a softer
#            MPC_XY_P / MPC_XY_VEL_P_ACC to mimic the official vehicle's damping); recorded in sim/px4_params*.txt
#   --reuse refuses to fly unless the vehicle rests at the spawn pose (at_spawn.py): the START frame is the pose at
#   arming, so a vehicle left where the last flight landed would shift every prior by that offset
# e.g. bash fly_compat.sh hover_land -- --prior ubuntu_compat/course_prior.replica.json --params '{"v_cruise":0.8}' --timeout 60
# Fresh stack per flight (re-arm after landing trips PX4 fd_esc_arming_failure; fresh params/dataman).
# Flight folder = profile/flights/<stamp>_NAME/ from tools/flight_run.py, plus sim/ (stack logs, PX4 ulog, side bag with Gazebo truth + raw VIO + the RIGHT stereo image/camera_info for offline SGBM,
# + params, check_compat before take-off, side bag with Gazebo truth + raw VIO, model/world/airframe as flown).
set -u
NAME=$1; shift
EV=vio; MODES=""; FR_ARGS=(); REUSE=0; KEEP=0; GUI=0; PROF=/compat/profile; PXP=()
while [ $# -gt 0 ]; do case "$1" in --ev-source) EV=$2; shift 2;; --images|--depth|--net-mask|--net-fence|--sgbm-speckle|--texture-min|--noise-minsup|--noise-temporal|--depth-scan|--depth-noise|--depth-dropout) MODES="$MODES $1 $2"; shift 2;; --net-model) MODES="$MODES $1 $2"; NETMODEL=$2; shift 2;;
  --reuse) REUSE=1; shift;; --keep) KEEP=1; shift;; --gui) GUI=1; KEEP=1; shift;; --profile) PROF=$2; shift 2;; --replica) MODES="$MODES --replica $2"; shift 2;; --spawn) MODES="$MODES --spawn $2"; shift 2;; --px4-param) PXP+=("$2"); shift 2;;
  --) shift; FR_ARGS=("$@"); break;; *) echo "bad arg $1"; exit 2;; esac; done
CT=${OC_CT:-official-compat-sim}; H=/home/fcy/claude-work/official-compat; UC=/compat/profile/ubuntu_compat   # OC_CT: parallel container
X(){ docker exec "$CT" bash -c "source $UC/env.sh; $1"; }
q(){ printf '%q ' "$@"; }
# --net-model true without --net-fence: use this flight's own --prior (path relative to the code profile)
if [ "${NETMODEL:-false}" = true ] && [[ "$MODES" != *--net-fence* ]]; then
  for ((i = 0; i < ${#FR_ARGS[@]}; i++)); do [ "${FR_ARGS[$i]}" = --prior ] && MODES="$MODES --net-fence $PROF/${FR_ARGS[$((i + 1))]}"; done
fi
echo "[fly_compat] $(date -Is) $NAME code=$PROF px4-param=[${PXP[*]:-}] ev=$EV modes:${MODES:- default(gz images, ideal depth)} args: ${FR_ARGS[*]:-}"
if [ $REUSE = 0 ]; then
X "bash $UC/stop_compat.sh" >/dev/null
X "bash $UC/start_compat.sh --ev-source $EV $MODES" || { echo '[fly_compat] start failed'; X "tail -n 30 \$LOGS/*.log"; X "bash $UC/stop_compat.sh"; exit 1; }
fi
X "python3 $UC/wait_ready.py --timeout 150" || { echo '[fly_compat] not ready'; X "tail -n 20 \$LOGS/oc-ev.log \$LOGS/oc-px4.log"; X "bash $UC/stop_compat.sh"; exit 1; }
if [ $REUSE = 1 ]; then X "python3 $UC/at_spawn.py" || { echo '[fly_compat] --reuse but the vehicle is not at the spawn pose: restart the stack (drop --reuse, or use --gui)'; exit 1; }; fi
if [ $GUI = 1 ] && [ $REUSE = 0 ]; then X "bash $UC/gui/open_gui.sh" 2>&1 | tail -n 2; sleep 5; fi
for kv in "${PXP[@]}"; do X "cd \$RUN/px4-rootfs && \$PX4_BUILD/bin/px4-param set ${kv%%=*} ${kv#*=}" | tail -n 1; done
X "cd \$PROFILE && python3 $UC/check_compat.py --secs 8 --out \$LOGS/check_compat.json > \$LOGS/check_compat.txt 2>&1"; X "tail -n 1 \$LOGS/check_compat.txt"
X "python3 $UC/wait_ready.py --timeout 60" >/dev/null || { echo '[fly_compat] lost readiness'; X "bash $UC/stop_compat.sh"; exit 1; }
X "rm -rf \$RUN/sidebag; tmux new-session -d -s oc-sidebag \"source $UC/env.sh; exec ros2 bag record -o \$RUN/sidebag -s mcap --storage-preset-profile zstd_fast --topics /gz/gt/odometry /ov_msckf/odomimu /ov_msckf/poseimu /gz/down_tof/scan /oak/imu /stereo/right/image /stereo/right/camera_info > \$LOGS/sidebag.log 2>&1\""
sleep 2
X "cd $PROF && python3 tools/flight_run.py --name $(q "$NAME") $(q "${FR_ARGS[@]}")" 2>&1 | grep --line-buffered -vE '^\{"t"'
RC=${PIPESTATUS[0]}
F=$(ls -td $H/profile/flights/*_"$NAME"/ 2>/dev/null | head -1)
X "tmux send-keys -t oc-sidebag C-c; sleep 3; tmux kill-session -t oc-sidebag 2>/dev/null"
if [ -n "$F" ]; then
  FC=/compat/profile/flights/$(basename "$F")
  X "mkdir -p $FC/sim && cd \$RUN/px4-rootfs && for k in \$(cat $UC/px4/params_of_interest.txt); do \$PX4_BUILD/bin/px4-param show \$k | grep ' '\$k' '; done > $FC/sim/px4_params.txt 2>&1;
     \$PX4_BUILD/bin/px4-param show > $FC/sim/px4_params_all.txt 2>&1;
     cp \$LOGS/*.log \$LOGS/*.json \$LOGS/*.txt $FC/sim/ 2>/dev/null; cp -r \$RUN/sidebag $FC/sim/sidebag 2>/dev/null;
     U=\$(ls -t \$RUN/px4-rootfs/log/*/*.ulg 2>/dev/null | head -1); [ -n \"\$U\" ] && cp \$U $FC/sim/px4.ulg;
     mkdir -p $FC/sim/asflown && cp \$RUN/models/skybeta_oak/model.sdf \$RUN/world/replica_compat.sdf \$RUN/world/layout.py $UC/px4/4999_gz_skybeta $UC/profile.json $FC/sim/asflown/; R_=\$(python3 -c \"import json; print(json.load(open('\$LOGS/stack.json')).get('replica', '/replica'))\"); [ -f \$R_/warp.json ] && cp \$R_/warp.json $FC/sim/asflown/;
     tar czhf $FC/sim/ubuntu_compat.tar.gz --exclude __pycache__ -C \$PROFILE ubuntu_compat"
fi
[ $KEEP = 1 ] || X "bash $UC/stop_compat.sh" >/dev/null
docker exec "$CT" chown -R 1000:1000 /compat/profile/flights
echo "[fly_compat] done rc=$RC folder=$F"
exit $RC
