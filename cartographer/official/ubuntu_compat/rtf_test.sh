#!/usr/bin/env bash
# PARALLELISM timing test on the Ubuntu HOST: N identical short flights started AT THE SAME TIME in containers
# official-compat-sim, -sim2 .. -simN (make_parallel_ct.sh N), then rtf_probe.py over the window all N were recording.
#   bash rtf_test.sh N TAG [EXTRA start_compat flags...]
# Flight: real-maze replica base, prior mission/course_prior.real_maze_20261108.json, nav_mode prior, maze -> tunnel exit
# (stop_after tunnel_traverse), rendered depth + mild noise + depth_scan (the per-stage harness load). nvidia-smi is
# sampled every 2 s -> run/TAG.gpu.csv (GPU memory / utilisation; per-process memory in run/TAG.gpu_apps.txt).
set -u
N=${1:?N}; TAG=${2:?TAG}; shift 2; H=/home/fcy/claude-work/official-compat; UCC=/compat/profile/ubuntu_compat
CTS=(official-compat-sim); for k in $(seq 2 "$N"); do CTS+=(official-compat-sim$k); done
for c in "${CTS[@]}"; do
  docker inspect -f "{{.State.Running}}" $c 2>/dev/null | grep -q true || { echo "[rtf] $c not running (make_parallel_ct.sh $N)"; exit 1; }
  docker exec $c pgrep -f "[t]ools/flight_run.py" > /dev/null && { echo "[rtf] a flight is running in $c: not starting"; exit 1; }
done
docker exec official-compat-sim test -f /compat/realmaze/base/replica/world/venue_full.sdf || docker exec official-compat-sim bash -c "source $UCC/env.sh; python3 \$UC/make_realmaze.py base" || exit 1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 2 > $H/run/$TAG.gpu.csv 2>&1 & SMI=$!
( for i in $(seq 1 60); do sleep 5; { date +%T; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; nvidia-smi pmon -c 1 -s um 2>/dev/null | tail -n +3; echo "MEM $(free -m | awk '/Mem:/{print $3" MiB used, "$7" MiB avail"}') LOAD $(cut -d' ' -f1 /proc/loadavg)"; } >> $H/run/$TAG.gpu_apps.txt; done ) & APPS=$!
PIDS=(); k=0
for c in "${CTS[@]}"; do
  k=$((k + 1))
  OC_CT=$c bash $H/fly_compat.sh ${TAG}_$k --depth ideal --images oak --replica /compat/realmaze/base/replica \
    --depth-noise 0.005 --depth-dropout 0.03 --depth-scan true "$@" \
    -- --prior mission/course_prior.real_maze_20261108.json --timeout 150 --params '{"stop_after": "tunnel_traverse", "nav_mode": "prior"}' \
    --notes "[timing test $TAG: $N flights at once, $c, real-maze base, prior mode, maze -> tunnel exit, ideal depth + noise + depth_scan $*]" \
    > $H/run/${TAG}_$k.log 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"; kill $SMI $APPS 2>/dev/null
DIRS=""; for k in $(seq 1 "$N"); do F=$(ls -td $H/profile/flights/*_${TAG}_$k/ 2>/dev/null | head -1); [ -n "$F" ] && DIRS="$DIRS /compat/profile/flights/$(basename $F)"; done
for d in $DIRS; do docker exec official-compat-sim bash -c "source $UCC/env.sh; cd $d && python3 \$UC/corner_check.py . > sim/corner_check.txt 2>&1; python3 \$UC/analyze_flight.py . > sim/analysis.txt 2>&1; true"; done
docker exec official-compat-sim bash -c "source $UCC/env.sh; python3 \$UC/rtf_probe.py $DIRS" > $H/run/$TAG.rtf.md 2> $H/run/$TAG.rtf.json
echo "GPU memory max $(cut -d, -f2 $H/run/$TAG.gpu.csv | sort -n | tail -1), util max $(cut -d, -f3 $H/run/$TAG.gpu.csv | sort -n | tail -1); RAM used max $(grep -o 'MEM [0-9]*' $H/run/$TAG.gpu_apps.txt | cut -c5- | sort -n | tail -1) MiB; load max $(grep -o 'LOAD [0-9.]*' $H/run/$TAG.gpu_apps.txt | cut -c6- | sort -n | tail -1)" >> $H/run/$TAG.rtf.md
for k in $(seq 1 "$N"); do echo "flight ${TAG}_$k: $(grep -o 'flight_run\] done: [^b]*' $H/run/${TAG}_$k.log | head -1)" >> $H/run/$TAG.rtf.md; done
cat $H/run/$TAG.rtf.md
