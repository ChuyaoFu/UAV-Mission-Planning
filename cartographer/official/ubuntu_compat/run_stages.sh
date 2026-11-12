#!/usr/bin/env bash
# PER-STAGE round on the Ubuntu HOST: every course stage started on its own (spawn at the stage's start pose, mission
# from that phase), stages in PARALLEL across the containers, rendered depth + mild noise, /team13/depth_scan on.
#   bash run_stages.sh TAG [--par N] [--stages tunnel,chicane,chwin,...] [--reps N] [--add LIST] [--world b|base|w591|v4] [--prior REL] [--nav-mode M]
#   default: real-maze replica b (make_realmaze.py b) + mission/course_prior.real_maze_20261108b.json, 6 stages x 3 flights, 5 parallel (lockstep: throughput; --par 3 for timing-sensitive checks; rtf_test.sh 2026-11-08: RTF 0.90 / 0.91 / 0.81 / 0.64 at 2 / 3 / 4 / 6 stacks; accept >= 0.85)
# Stage list from make_stage_list.py (spawn poses from the prior, truth clearance >= 0.35 m); flights via run_blind.sh --custom;
# summary run/<TAG>.stages.md (stage_summary.py). Launch detached: setsid nohup bash run_stages.sh TAG > run/TAG.stages.log 2>&1 &
set -u
TAG=${1:?TAG}; shift; H=/home/fcy/claude-work/official-compat; UCC=/compat/profile/ubuntu_compat
PAR=5; STAGES=tunnel,chicane,window,pad,wall,home; REPS=3; WORLD=b; PRIOR=mission/course_prior.real_maze_20261108b.json; NAV=""; XLIST=""
while [ $# -gt 0 ]; do case "$1" in --par) PAR=$2; shift 2;; --stages) STAGES=$2; shift 2;; --reps) REPS=$2; shift 2;;
  --world) WORLD=$2; shift 2;; --prior) PRIOR=$2; shift 2;; --nav-mode) NAV=$2; shift 2;; --add) XLIST=$2; shift 2;; *) echo "bad arg $1"; exit 2;; esac; done
case "$WORLD" in v4) REPL=/replica;; *) REPL=/compat/realmaze/$WORLD/replica
  docker exec official-compat-sim test -f $REPL/world/venue_full.sdf || docker exec official-compat-sim bash -c "source $UCC/env.sh; python3 \$UC/make_realmaze.py $WORLD" || exit 1;; esac
LIST=$H/run/$TAG.stages.list
docker exec official-compat-sim bash -c "source $UCC/env.sh; python3 \$UC/make_stage_list.py /compat/run/$TAG.stages.list --prior $PRIOR --replica $REPL --stages $STAGES --reps $REPS ${NAV:+--nav-mode $NAV}" || exit 1
[ -n "$XLIST" ] && cat "$XLIST" >> $LIST                  # --add FILE: extra run_blind --custom lines (e.g. full-mission flights) in the same batch
IDEAL="--depth-noise 0.005 --depth-dropout 0.03 --depth-scan true"
bash $H/profile/ubuntu_compat/run_blind.sh $TAG --parallel $PAR --depth ideal --images oak --stack "$IDEAL" --custom $LIST
# per-stage table (success / clean / contacts / < 0.28 m / stage time / stage metric) -> run/<TAG>.stages.md
docker exec official-compat-sim bash -c "source $UCC/env.sh; python3 \$UC/stage_summary.py $(tr '\n' ' ' < $H/run/$TAG.dirs) 2>/dev/null" > $H/run/$TAG.stages.md
echo "[stages] $TAG done: $H/run/$TAG.stages.md"; tail -12 $H/run/$TAG.stages.md
