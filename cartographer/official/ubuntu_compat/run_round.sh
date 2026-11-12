#!/usr/bin/env bash
# A round of recorded flights on the Ubuntu HOST, one fresh stack (+ live GUI, left running) per flight, then the
# sim-side analyses (corner_check, analyze_flight, vio_error) and one summary table (v4_summary.py).
#   bash run_round.sh TAG "MODES" "PRIOR_FILE ..." [NOTES]
#   e.g. bash run_round.sh v4A_stereo_tape "--depth stereo --images oak" "course_prior.replica.seed1.json course_prior.replica.seed2.json"
# Flight names: <TAG>_<prior stem without course_prior.replica.>; folders under profile/flights/.
set -u
TAG=$1; MODES=$2; PRIORS=$3; NOTES=${4:-}
H=/home/fcy/claude-work/official-compat; UC=/compat/profile/ubuntu_compat; CT=official-compat-sim
X(){ docker exec "$CT" bash -c "source $UC/env.sh; $1"; }
DIRS=()
for P in $PRIORS; do
  S=${P#course_prior.replica}; S=${S#.}; S=${S%json}; S=${S%.}; S=${S:-truth}; NAME="${TAG}_${S//./_}"
  echo "=== $(date -Is) $NAME prior $P modes $MODES"
  bash $H/fly_compat.sh "$NAME" --gui $MODES -- --prior "ubuntu_compat/$P" --timeout 300 --notes "[$MODES] [prior=$P] $NOTES" 2>&1 \
    | grep -E 'fly_compat\]|PASS,|flight_run\] done|phase .* -> (done|aborted|abort_land)' | cut -c1-220
  F=$(ls -td $H/profile/flights/*_"$NAME"/ 2>/dev/null | head -1)
  [ -n "$F" ] || { echo "no folder for $NAME"; continue; }
  B=/compat/profile/flights/$(basename "$F"); DIRS+=("$B")
  X "cd $B && python3 $UC/corner_check.py . > sim/corner_check.txt 2>&1; python3 $UC/analyze_flight.py . > sim/analysis.txt 2>&1; python3 $UC/vio_error.py . > sim/vio_error.txt 2>&1; true"
  X "python3 $UC/v4_summary.py $B 2>/dev/null | tail -1"
done
echo "=== round $TAG summary"
X "python3 $UC/v4_summary.py ${DIRS[*]} 2>/dev/null"
