#!/usr/bin/env bash
# ONE command: Gazebo + RViz demo of our algorithm on the Ubuntu desktop. Fresh sim stack, Gazebo GUI (follow camera)
# + RViz (VIO path vs Gazebo truth, obstacle map, left image + depth), one RECORDED full-course flight (layout v4,
# official-map replica), windows stay open afterwards.
#
#   From the Mac (or on the Ubuntu host itself):
#     ssh uav-ubuntu-ts 'bash /home/fcy/claude-work/official-compat/profile/ubuntu_compat/demo.sh'
#
#   demo.sh [ideal|stereo] [PRIOR] [--latest] [--pin FLIGHT] [--force]      (run on the Ubuntu HOST)
#     ideal (default)  ideal depth camera;   stereo  OAK-like images + stereo (SGBM) depth
#     PRIOR            prior file relative to the code tree (default: pinned -> the prior that flight used,
#                      --latest -> ubuntu_compat/course_prior.replica.json = replica truth)
#     --latest         fly the CURRENT code in profile/ (may be mid-development / regressed)
#     --pin FLIGHT     pin the code of another recorded flight folder (default 20261107-201057_v4map_ideal_truth:
#                      full v4 course done, 0 contacts; mission/core.py 1b0a9305, perception/tunnel.py bbff5bd2)
#     --force          stop a flight that is already running (default: refuse; only one Gazebo at a time)
#   demo.sh --stop     close the sim and the GUI windows
#
# Pinned code = that flight's code.tar.gz unpacked once to official-compat/demo/<FLIGHT>/ (+ symlinks flights/ and
# ubuntu_compat/ -> profile/, so the sim stack is the current one and the flight folder lands in profile/flights/
# as <stamp>_demo_<mode>[_latest]). The flight runs detached (setsid): a dropped ssh session does not stop it; the
# log is official-compat/run/demo.log. Needs a logged-in GNOME session on the Ubuntu desktop (DISPLAY :0); the X
# cookie for the GUI (official-compat/run/xauth) is refreshed from that session on every start.
set -u
H=/home/fcy/claude-work/official-compat; CT=official-compat-sim; UCC=/compat/profile/ubuntu_compat
DEFAULT_PIN=20261107-201057_v4map_ideal_truth
LOG=$H/run/demo.log
X(){ docker exec "$CT" bash -c "source $UCC/env.sh; $1"; }

MODE=ideal; PRIOR=""; LATEST=0; PIN=$DEFAULT_PIN; FORCE=0; JOB=0; CODE=""; TAG=""
while [ $# -gt 0 ]; do case "$1" in
  ideal|stereo) MODE=$1; shift;; --latest) LATEST=1; shift;; --pin) PIN=$2; shift 2;; --force) FORCE=1; shift;;
  --stop) X "bash $UCC/stop_compat.sh" > /dev/null 2>&1; echo "[demo] sim and GUI closed"; exit 0;;
  --_job) JOB=1; shift;; --code) CODE=$2; shift 2;; --tag) TAG=$2; shift 2;;     # internal (detached job)
  -h|--help) sed -n '2,25p' "$0"; exit 0;;
  -*) echo "[demo] unknown option $1 (see --help)"; exit 2;;
  *) PRIOR=$1; shift;; esac; done

if [ $JOB = 0 ]; then
  # ---- front end: checks, X cookie, code tree; then start the detached job and follow its log
  docker inspect -f '{{.State.Running}}' "$CT" 2>/dev/null | grep -q true || { echo "[demo] container $CT is not running"; exit 1; }
  if X "pgrep -f '[t]ools/flight_run.py' > /dev/null"; then
    [ $FORCE = 1 ] || { echo "[demo] a flight is already running in $CT (only one Gazebo at a time); use --force to stop it"; exit 1; }
    X "pkill -INT -f '[t]ools/flight_run.py'; sleep 5"
  fi
  XA=$(ls -t /run/user/$(id -u)/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
  [ -n "$XA" ] || { echo "[demo] no GNOME/Xwayland session for $(whoami) on this host: log in on the Ubuntu desktop first"; exit 1; }
  xauth -f "$XA" nlist 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$H/run/xauth.new" nmerge - 2>/dev/null && mv -f "$H/run/xauth.new" "$H/run/xauth"
  X "DISPLAY=:0 XAUTHORITY=/compat/run/xauth xdpyinfo > /dev/null 2>&1" || { echo "[demo] cannot open DISPLAY :0 with the refreshed cookie ($XA)"; exit 1; }
  if [ $LATEST = 1 ]; then
    CODE=/compat/profile; TAG=demo_${MODE}_latest; DEF_PRIOR=ubuntu_compat/course_prior.replica.json; WHAT="CURRENT code in profile/"
  else
    D=$H/demo/$PIN
    if [ ! -f "$D/tools/flight_run.py" ]; then
      [ -f "$H/profile/flights/$PIN/code.tar.gz" ] || { echo "[demo] no code.tar.gz in profile/flights/$PIN"; exit 1; }
      mkdir -p "$D" && tar xzf "$H/profile/flights/$PIN/code.tar.gz" -C "$D" && cp "$H/profile/flights/$PIN/prior.json" "$D/demo_prior.json"
      ln -sfn ../../profile/flights "$D/flights"; ln -sfn ../../profile/ubuntu_compat "$D/ubuntu_compat"
      echo "[demo] unpacked the code of $PIN to $D"
    fi
    CODE=/compat/demo/$PIN; TAG=demo_${MODE}; DEF_PRIOR=demo_prior.json; WHAT="PINNED code of $PIN"
  fi
  PRIOR=${PRIOR:-$DEF_PRIOR}
  X "test -f $CODE/$PRIOR" || { echo "[demo] prior $PRIOR not found in $CODE"; exit 1; }
  echo "[demo] $(date '+%F %T') $WHAT, depth=$MODE, prior=$PRIOR -> flight <stamp>_$TAG in profile/flights/ (log $LOG)"
  echo "[demo] windows on the Ubuntu desktop: 'Gazebo Sim' (left) and 'RViz - official-compat' (right); about 3-4 min"
  setsid nohup bash "$0" --_job "$MODE" "$PRIOR" --code "$CODE" --tag "$TAG" > "$LOG" 2>&1 < /dev/null &
  sleep 2; tail -n +1 -f "$LOG" 2>/dev/null & TP=$!
  while pgrep -f "[d]emo.sh --_job" > /dev/null; do sleep 5; done
  sleep 1; kill $TP 2>/dev/null
  exit 0
fi

# ---- detached job: fly (fresh stack + GUI, kept open), then the sim-side analyses and one summary row
IMG=gz; [ "$MODE" = stereo ] && IMG=oak
bash "$H/fly_compat.sh" "$TAG" --gui --depth "$MODE" --images "$IMG" --profile "$CODE" \
  -- --prior "$PRIOR" --timeout 300 --notes "demo.sh: depth=$MODE, code=$CODE, prior=$PRIOR" 2>&1 \
  | grep --line-buffered -vE '^\{"t"|\[wait_ready\] [0-9]|^0x|RuntimeWarning|occ = occ' | stdbuf -oL cut -c1-200
F=$(ls -td "$H"/profile/flights/*_"$TAG"/ 2>/dev/null | head -1)
[ -n "$F" ] || { echo "[demo] no flight folder found"; exit 1; }
B=/compat/profile/flights/$(basename "$F")
X "cd $B && python3 $UCC/corner_check.py . > sim/corner_check.txt 2>&1; python3 $UCC/analyze_flight.py . > sim/analysis.txt 2>&1; python3 $UCC/vio_error.py . > sim/vio_error.txt 2>&1; true"
echo "[demo] result:"; X "python3 $UCC/v4_summary.py $B 2>/dev/null"
echo "[demo] flight saved: $F   (windows stay open; close with: bash $H/profile/ubuntu_compat/demo.sh --stop)"
