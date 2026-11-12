#!/usr/bin/env bash
# BLIND regression round on the Ubuntu HOST (main regime since 2026-11-07): stereo depth (OAK-like images + SGBM) +
# OpenVINS + blind priors (organiser map + per-element offsets, make_blind_prior.py) in WARPED worlds (make_warp.py),
# one fresh stack + live GUI per flight, then corner_check / analyze_flight / vio_error / analyze_obstacles and one
# summary with a per-level pass-rate table (v4_summary.py). Runs DETACHED (setsid): a dropped ssh does not stop it.
#
#   bash run_blind.sh TAG [--code latest|pinned|DIR] [--levels "map eyeball"] [--seeds "1 2 3 4 5"] [--warps "1 2 3"]
#        [--depth stereo|ideal] [--maze-only] [--parallel N] [--params JSON]
#   --maze-only   stop_after corridor (maze + first corridor only, ~2-3 min per flight), timeout 200 s
#   --set NAME    prior set: blind (default; no walls) or blind2 (+ the map walls)
#   --sgbm-speckle A,B  stereo noise filter (depth_filters.py / oak_sim_driver: SGBM speckleWindowSize A, speckleRange B)
#   --stack "FLAGS"  extra fly_compat stack flags, verbatim (e.g. "--noise-minsup 150,1 --sgbm-speckle 800,2")
#   --mazes "1 2 .." --kind blind|coarse   random-maze worlds (make_maze.py) instead of the warps: prior course_prior.maze<S>.<kind>.json
#   --custom FILE   flights listed in FILE, one per line: NAME REPLICA_DIR PRIOR_PATH [SPAWN|- [PARAMS_JSON ...]]
#                   (prior relative to the code profile; SPAWN = F,L,YAW_DEG vehicle start pose; PARAMS = this flight's
#                    params JSON, overrides --params, rest of the line)
#   --images gz|oak  image model override (default: oak with stereo depth, gz with ideal depth)
#   --net-filter  stereo net filter on (oak_sim_driver net_model + net_mask, fence from each flight's --prior)
#   --parallel N  N queues in containers official-compat-sim, -sim2 .. -sim6 (make_parallel_ct.sh N), all headless
#   the summary file official-compat/run/<TAG>.summary.md is rewritten after EVERY flight (partial tables)
#   e.g. bash run_blind.sh v4C                     # current code in profile/, 5 map + 5 eyeball seeds
#        bash run_blind.sh v4C_base --code pinned  # the demo's pinned code (baseline a new version must beat)
# Seed k flies in warp w_((k-1) mod #warps); flight names <TAG>_<level>_s<k>_w<warp>. Log: official-compat/run/<TAG>.log
set -u
H=/home/fcy/claude-work/official-compat; CT=official-compat-sim; UCC=/compat/profile/ubuntu_compat
X(){ docker exec "$CT" bash -c "source $UCC/env.sh; $1"; }
TAG=${1:?TAG}; shift
CODE=latest; LEVELS="map eyeball"; SEEDS="1 2 3 4 5"; WARPS="1 2 3"; DEPTH=stereo; JOB=0; MAZE=0; PAR=1; PARAMS=""; NETF=0; PSET=blind; SPECK=""; XSTACK=""; MAZES=""; KIND=blind; CUSTOM=""
while [ $# -gt 0 ]; do case "$1" in
  --code) CODE=$2; shift 2;; --levels) LEVELS=$2; shift 2;; --seeds) SEEDS=$2; shift 2;; --warps) WARPS=$2; shift 2;;
  --depth) DEPTH=$2; shift 2;; --_job) JOB=1; shift;; --maze-only) MAZE=1; shift;; --parallel) PAR=$2; shift 2;; --net-filter) NETF=1; shift;; --set) PSET=$2; shift 2;; --sgbm-speckle) SPECK=$2; shift 2;; --stack) XSTACK=$2; shift 2;; --mazes) MAZES=$2; shift 2;; --images) IMAGES=$2; shift 2;; --custom) CUSTOM=$2; shift 2;; --kind) KIND=$2; shift 2;;
  --params) PARAMS=$2; shift 2;; *) echo "bad arg $1"; exit 2;; esac; done
CTS=(official-compat-sim official-compat-sim2 official-compat-sim3 official-compat-sim4 official-compat-sim5 official-compat-sim6)
case "$CODE" in latest) PROF=/compat/profile;; pinned) PROF=/compat/demo/20261107-201057_v4map_ideal_truth;; *) PROF=$CODE;; esac
LOG=$H/run/$TAG.log
if [ $JOB = 0 ]; then
  for q in $(seq 0 $((PAR - 1))); do
    docker inspect -f "{{.State.Running}}" ${CTS[$q]} 2>/dev/null | grep -q true || { echo "[blind] container ${CTS[$q]} not running (make_parallel_ct.sh)"; exit 1; }
    docker exec ${CTS[$q]} pgrep -f "[t]ools/flight_run.py" > /dev/null && { echo "[blind] a flight is already running in ${CTS[$q]}; try later"; exit 1; }
  done
  for w in $WARPS; do X "test -f /compat/warp/w$w/replica/world/venue_full.sdf" || X "python3 $UCC/make_warp.py $w" || exit 1; done
  echo "[blind] $(date '+%F %T') $TAG code=$PROF depth=$DEPTH levels=[$LEVELS] seeds=[$SEEDS] warps=[$WARPS] maze_only=$MAZE parallel=$PAR net_filter=$NETF set=$PSET speckle=${SPECK:-off} stack=[${XSTACK}] -> log $LOG"
  setsid nohup bash "$0" "$TAG" --_job --code "$CODE" --levels "$LEVELS" --seeds "$SEEDS" --warps "$WARPS" --depth "$DEPTH" --parallel "$PAR" --params "$PARAMS" $([ $MAZE = 1 ] && echo --maze-only) $([ $NETF = 1 ] && echo --net-filter) --set "$PSET" ${SPECK:+--sgbm-speckle "$SPECK"} ${XSTACK:+--stack "$XSTACK"} ${MAZES:+--mazes "$MAZES" --kind "$KIND"} ${IMAGES:+--images "$IMAGES"} ${CUSTOM:+--custom "$CUSTOM"} > "$LOG" 2>&1 < /dev/null &
  # wait for THIS round's job by its command line (not $!: when the caller is itself a setsid session leader, setsid
  # forks and $! is a parent that exits at once -> callers running several rounds in sequence would start them all at once)
  sleep 2; tail -n +1 -f "$LOG" 2>/dev/null & TP=$!
  while pgrep -f "[r]un_blind.sh $TAG --_job" > /dev/null; do sleep 5; done
  sleep 1; kill $TP 2>/dev/null; exit 0
fi
IMG=gz; [ "$DEPTH" = stereo ] && IMG=oak; [ -n "${IMAGES:-}" ] && IMG=$IMAGES
read -ra WL <<< "$WARPS"; LIST=$H/run/$TAG.dirs; : > "$LIST"
TO=360; GUI=--gui; FP=${PARAMS:-}
if [ $MAZE = 1 ]; then TO=200; FP=${PARAMS:-'{"stop_after": "corridor"}'}; fi
[ $PAR -gt 1 ] && GUI=""                                     # parallel: all headless
NETARGS=""; [ $NETF = 1 ] && NETARGS="--net-model true --net-mask true"
[ -n "$SPECK" ] && NETARGS="$NETARGS --sgbm-speckle $SPECK"
[ -n "$XSTACK" ] && NETARGS="$NETARGS $XSTACK"
ITEMS=(); if [ -n "$CUSTOM" ]; then while read -r cn cr cp csp cpar; do [ -n "$cn" ] && [ "${cn#\#}" = "$cn" ] && ITEMS+=("custom:$cn|$cr|$cp|${csp:--}|$cpar"); done < "$CUSTOM"; elif [ -n "$MAZES" ]; then for S in $MAZES; do ITEMS+=("maze:$S"); done; else for LV in $LEVELS; do for S in $SEEDS; do ITEMS+=("$LV:$S"); done; done; fi
queue(){                                                     # queue q: items q, q+PAR, q+2*PAR, ... in container CTS[q]
  local q=$1 CT=${CTS[$1]} k LV S P W NAME F B
  for ((k = q; k < ${#ITEMS[@]}; k += PAR)); do
    LV=${ITEMS[$k]%%:*}; S=${ITEMS[$k]#*:}
    P=course_prior.$PSET.seed$S.json; [ "$LV" = eyeball ] && P=course_prior.$PSET.eyeball.seed$S.json
    W=1; [[ "$S" =~ ^[0-9]+$ ]] && W=${WL[$(( (S - 1) % ${#WL[@]} ))]}; NAME="${TAG}_${LV}_s${S}_w${W}"; REPL=/compat/warp/w$W/replica
    if [ "$LV" = maze ]; then P=course_prior.maze$S.$KIND.json; NAME="${TAG}_maze${S}_${KIND}"; REPL=/compat/maze/m$S/replica; fi
    ISP=""; IFP=""
    if [ "$LV" = custom ]; then IFS="|" read -r cn REPL P ISP IFP <<< "$S"; NAME="${TAG}_${cn}"; P="__ABS__$P"; [ "$ISP" = - ] && ISP=""; fi
    PARG=$FP; [ -n "$IFP" ] && PARG=$IFP
    echo "=== $(date '+%T') [$CT] $NAME prior $P world $REPL code $PROF"
    OC_CT=$CT bash $H/fly_compat.sh "$NAME" $GUI --depth "$DEPTH" --images "$IMG" --profile "$PROF" --replica "$REPL" $NETARGS ${ISP:+--spawn $ISP} \
      -- --prior "$( [ "${P#__ABS__}" != "$P" ] && echo "${P#__ABS__}" || echo "ubuntu_compat/$P")" --timeout $TO ${PARG:+--params "$PARG"} --notes "[blind $LV seed $S, warp w$W, depth=$DEPTH, maze_only=$MAZE, net_filter=$NETF, speckle=${SPECK:-off}, stack=${XSTACK// /_}, set=$PSET, $CT] code=$PROF" 2>&1 \
      | grep --line-buffered -E 'fly_compat\] (done|start)|PASS,|flight_run\] done|phase .* -> ' | stdbuf -oL sed "s/^/[$CT] /" | stdbuf -oL cut -c1-170
    F=$(ls -td $H/profile/flights/*_"$NAME"/ 2>/dev/null | head -1)
    [ -n "$F" ] || { echo "[$CT] no folder for $NAME"; continue; }
    B=/compat/profile/flights/$(basename "$F")
    docker exec "$CT" bash -c "source $UCC/env.sh; cd $B && python3 $UCC/corner_check.py . > sim/corner_check.txt 2>&1; python3 $UCC/analyze_flight.py . > sim/analysis.txt 2>&1; python3 $UCC/vio_error.py . > sim/vio_error.txt 2>&1; python3 $UCC/analyze_obstacles.py . > sim/obstacles.txt 2>&1; true"
    ( flock 9; echo "$B" >> "$LIST"
      docker exec "$CT" bash -c "source $UCC/env.sh; python3 $UCC/v4_summary.py $(tr '\n' ' ' < "$LIST") 2>/dev/null" > "$H/run/$TAG.summary.md"
      echo "[$CT] done $(wc -l < "$LIST")/${#ITEMS[@]}: $(docker exec "$CT" bash -c "source $UCC/env.sh; python3 $UCC/v4_summary.py $B 2>/dev/null | sed -n 3p" | cut -c1-200)" ) 9> "$LIST.lock"
  done
}
for q in $(seq 0 $((PAR - 1))); do queue $q & sleep 20; done
wait
echo "=== $(date '+%T') round $TAG summary"
cat "$H/run/$TAG.summary.md"
