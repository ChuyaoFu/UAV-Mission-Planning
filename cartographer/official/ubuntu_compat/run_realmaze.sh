#!/usr/bin/env bash
# REAL-maze replica round on the Ubuntu HOST (detached via setsid by the caller): maze -> tunnel exit
# (stop_after tunnel_traverse), rendered depth + mild noise, /team13/depth_scan on, 2 parallel, 3 flights per config
# in the worlds base / w591 / base (make_realmaze.py; w591 = + the uncertain wall_l-5.91):
#   <TAG>a  prior mode, mission/course_prior.real_maze_20261108.json
#   <TAG>b  map mode,   mission/course_prior.real_maze_20261108.json
#   <TAG>c  map mode,   mission/course_prior.site.json   (OLD prior: "we did not know the maze changed")
#   bash run_realmaze.sh TAG
set -u
TAG=${1:?TAG}; H=/home/fcy/claude-work/official-compat; R=$H/profile/ubuntu_compat/run_blind.sh
IDEAL="--depth-noise 0.005 --depth-dropout 0.03 --depth-scan true"
for v in base w591; do docker exec official-compat-sim bash -c "source /compat/profile/ubuntu_compat/env.sh; python3 \$UC/make_realmaze.py $v" || exit 1; done
mk(){ local f=$H/run/${TAG}$1.list p=$2; : > $f; i=1; for v in base w591 base; do echo "$1${i}_$v /compat/realmaze/$v/replica $p" >> $f; i=$((i + 1)); done; echo $f; }
RM=mission/course_prior.real_maze_20261108.json; SITE=mission/course_prior.site.json
bash $R ${TAG}a --parallel 2 --depth ideal --images oak --stack "$IDEAL" --params '{"stop_after": "tunnel_traverse", "nav_mode": "prior"}' --custom $(mk a $RM)
bash $R ${TAG}b --parallel 2 --depth ideal --images oak --stack "$IDEAL" --params '{"stop_after": "tunnel_traverse", "nav_mode": "map"}' --custom $(mk b $RM)
bash $R ${TAG}c --parallel 2 --depth ideal --images oak --stack "$IDEAL" --params '{"stop_after": "tunnel_traverse", "nav_mode": "map"}' --custom $(mk c $SITE)
