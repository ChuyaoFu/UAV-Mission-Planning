#!/usr/bin/env bash
# Start ONE fresh official-compat stack inside container official-compat-sim (tmux sessions oc-*).
#   bash /compat/profile/ubuntu_compat/start_compat.sh [--ev-source vio|gt] [--no-vio] [--images gz|oak] [--depth ideal|stereo] [--replica DIR]
# --replica DIR: venue replica to fly in (layout.py + world/), default /replica; e.g. a warped copy from make_warp.py
# --depth stereo implies --images oak (replica OAK sensor model + SGBM in oak_sim_driver.py)
# --depth-scan on | H_MIN,H_MAX[,RANGE_MAX]: /team13/depth_scan LaserScan for the maze planner (oak_sim_driver depth_scan:=true;
#   band in metres above the camera, default -0.25,0.35,5), from the published depth after the net / noise filters
# Order: model+world -> gz (headless, NVIDIA EGL) -> ros_gz bridge -> OAK driver (+static TF)
#        -> MicroXRCEAgent -> PX4 SITL (airframe 4999 Sky Beta approx.) -> OpenVINS -> ev_bridge
set -eo pipefail
source "$(dirname "$0")/env.sh"
EV_SOURCE=vio; VIO=1; IMAGES=gz; DEPTH=ideal; REPLICA=/replica; SPAWN=""; NET_MASK=false; NET_MODEL=false; NET_FENCE=""; NOISE_ARGS=""; SCAN_ARGS=""
while [ $# -gt 0 ]; do case "$1" in --ev-source) EV_SOURCE=$2; shift 2;; --no-vio) VIO=0; shift;;
  --images) IMAGES=$2; shift 2;; --depth) DEPTH=$2; shift 2;; --replica) REPLICA=$2; shift 2;; --spawn) SPAWN=$2; shift 2;;
  --net-mask) NET_MASK=$2; shift 2;; --net-model) NET_MODEL=$2; shift 2;; --net-fence) NET_FENCE=$2; shift 2;;
  --sgbm-speckle) NOISE_ARGS="$NOISE_ARGS -p sgbm_speckle_window:=${2%,*} -p sgbm_speckle_range:=${2#*,}"; shift 2;;
  --texture-min) NOISE_ARGS="$NOISE_ARGS -p texture_min_dn:=$2"; shift 2;;
  --noise-minsup) NOISE_ARGS="$NOISE_ARGS -p noise_minsup_px:=${2%,*} -p noise_minsup_diff_px:=${2#*,}"; shift 2;;
  --depth-noise) NOISE_ARGS="$NOISE_ARGS -p depth_noise_k:=$2"; shift 2;; --depth-dropout) NOISE_ARGS="$NOISE_ARGS -p depth_dropout:=$2"; shift 2;;   # ideal depth only
  --noise-temporal) NOISE_ARGS="$NOISE_ARGS -p noise_temporal_tol:=${2%%,*} -p noise_temporal_frames:=$(echo $2 | cut -d, -f2)"; shift 2;;
  --depth-scan) case "$2" in on|true|1) SCAN_ARGS="-p depth_scan:=true";; off|false|0) SCAN_ARGS="";;   # on | H_MIN,H_MAX[,RANGE_MAX]
      *) IFS=, read -r S1 S2 S3 <<< "$2"; SCAN_ARGS="-p depth_scan:=true -p depth_scan_h_min_m:=$S1 -p depth_scan_h_max_m:=$S2${S3:+ -p depth_scan_range_max_m:=$S3}";; esac; shift 2;;
  *) echo "bad arg $1"; exit 2;; esac; done
[ -f "$REPLICA/layout.py" ] && [ -f "$REPLICA/world/venue_full.sdf" ] || { echo "no replica layout/world in $REPLICA"; exit 2; }
export GZ_SIM_RESOURCE_PATH=$RUN/models:$REPLICA/world
[ "$DEPTH" = stereo ] && IMAGES=oak
case "$IMAGES/$DEPTH" in gz/ideal|oak/ideal|oak/stereo) ;; *) echo "bad mode $IMAGES/$DEPTH"; exit 2;; esac
if tmux ls -F '#S' 2>/dev/null | grep -q '^oc-'; then echo 'official-compat stack already running: stop_compat.sh first'; exit 1; fi
mkdir -p "$RUN" "$LOGS"; rm -f "$LOGS"/*.log
echo "{\"ev_source\": \"$EV_SOURCE\", \"vio\": $VIO, \"images\": \"$IMAGES\", \"depth\": \"$DEPTH\", \"replica\": \"$REPLICA\", \"spawn\": \"$SPAWN\", \"net_mask\": \"$NET_MASK\", \"net_model\": \"$NET_MODEL\", \"net_fence\": \"$NET_FENCE\", \"noise_args\": \"$NOISE_ARGS\", \"depth_scan\": \"$SCAN_ARGS\", \"started\": \"$(date -Is)\"}" > "$LOGS/stack.json"

# 1. model + world (generated each start from profile.json / replica layout)
/usr/bin/python3 "$UC/gen_model.py" "$RUN/models" --image-model "$IMAGES" > /dev/null
# bridge: in oak image mode the clean gz images go to /gz/oak/*/image and oak_sim_driver publishes /stereo/*
if [ "$IMAGES" = oak ]; then
  sed -e 's#ros_topic_name: /stereo/left/image#ros_topic_name: /gz/oak/left/image#' \
      -e 's#ros_topic_name: /stereo/right/image#ros_topic_name: /gz/oak/right/image#' "$UC/bridge.yaml" > "$RUN/bridge.yaml"
else
  cp "$UC/bridge.yaml" "$RUN/bridge.yaml"
fi
/usr/bin/python3 "$UC/build_world.py" "$REPLICA" "$RUN/world/replica_compat.sdf" 0.0020833333 "$RUN/models/skybeta_oak" skybeta_0 "$SPAWN" > "$LOGS/world.log"
cp "$REPLICA/layout.py" "$RUN/world/layout.py"   # the layout actually flown (analysis tools read it from sim/asflown/)

# 2. PX4 rootfs: fresh params/dataman every flight (re-arm after landing trips fd_esc_arming_failure)
R=$RUN/px4-rootfs; rm -rf "$R"; mkdir -p "$R"
cp -a "$PX4_BUILD/etc" "$R/etc"; ln -s "$PX4/test_data" "$R/test_data"
cp "$UC/px4/4999_gz_skybeta" "$R/etc/init.d-posix/airframes/4999_gz_skybeta"

run(){ local name=$1; shift; tmux new-session -d -s "$name" "source $UC/env.sh; $* > $LOGS/$name.log 2>&1"; }

# 3. Gazebo (server only, headless rendering on the NVIDIA GPU)
run oc-gz gz sim -v 3 -r -s --headless-rendering "$RUN/world/replica_compat.sdf"
for i in $(seq 1 60); do
  if timeout 3 gz topic -l 2>/dev/null | grep -q '/world/replica/model/skybeta_0/link/base_link/sensor/imu_sensor/imu'; then break; fi
  sleep 1
done
timeout 3 gz topic -l | grep -q 'skybeta_0/link/base_link/sensor/imu_sensor/imu' || { echo 'gz world/model not ready, see oc-gz.log'; exit 1; }

# 4. bridge + OAK driver (camera_info, depth fix-up, static TF)
NF_ARG=""; [ -n "$NET_FENCE" ] && NF_ARG="-p net_fence_json:=$NET_FENCE"
run oc-bridge ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=$RUN/bridge.yaml -p use_sim_time:=true
run oc-oak /usr/bin/python3 $UC/oak_sim_driver.py --ros-args -p use_sim_time:=true -p image_model:=$IMAGES -p depth_mode:=$DEPTH -p net_mask:=$NET_MASK -p net_model:=$NET_MODEL $NF_ARG $NOISE_ARGS $SCAN_ARGS

# 5. uXRCE-DDS agent + PX4 SITL attached to the already-spawned model
run oc-agent MicroXRCEAgent udp4 -p 8888
run oc-px4 "cd $R && PX4_SYS_AUTOSTART=4999 PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=replica PX4_GZ_MODEL_NAME=skybeta_0 PX4_SIM_MODEL=gz_skybeta_oak PX4_UXRCE_DDS_PORT=8888 $PX4_BUILD/bin/px4 -w $R"
for i in $(seq 1 60); do
  grep -q 'Startup script returned successfully' "$LOGS/oc-px4.log" 2>/dev/null && break
  tmux has-session -t oc-px4 2>/dev/null || { echo 'PX4 exited, see oc-px4.log'; exit 1; }
  sleep 1
done
grep -q 'Startup script returned successfully' "$LOGS/oc-px4.log" || { echo 'PX4 startup timeout'; exit 1; }

# 6. estimator: OpenVINS stereo+IMU -> ev_bridge -> PX4 EV, /estimator/healthy, odom->base_link
if [ "$VIO" = 1 ] && [ "$EV_SOURCE" = vio ]; then
  run oc-vio env LD_LIBRARY_PATH=$OV_PREFIX/ov_core/lib:$OV_PREFIX/ov_init/lib:$OV_PREFIX/ov_msckf/lib:\${LD_LIBRARY_PATH:-} \
      $OV_PREFIX/ov_msckf/lib/ov_msckf/run_subscribe_msckf --ros-args -r __ns:=/ov_msckf \
      -p config_path:=$UC/openvins/estimator_config.yaml -p use_sim_time:=true -p processing_async:=false -p verbosity:=INFO
fi
run oc-ev /usr/bin/python3 $UC/ev_bridge.py --ros-args -p use_sim_time:=true -p source:=$EV_SOURCE
echo "official-compat stack started (ev_source=$EV_SOURCE images=$IMAGES depth=$DEPTH); logs in $LOGS"
