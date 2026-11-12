#!/usr/bin/env bash
# Batch b1 (2026-11-08, lead's request) on the Ubuntu HOST, par 5, rendered depth + mild noise, core 3c1599cb:
#   rm  x5  map mode on the real-maze replica b, start -> tunnel exit (stop_after tunnel_traverse)
#   m24 x3  map mode on random maze m24 (side first leg), blind prior, maze only (stop_after corridor)
#   tunnel x5  prior mode, spawn at the tunnel standoff, tunnel_align -> tunnel_traverse
#   chwin  x5  prior mode, spawn just past the tunnel exit, chicane -> window_traverse (the window stage)
#   setsid nohup bash batch_b1.sh TAG > run/TAG.batch.log 2>&1 &      -> run/TAG.stages.md
set -u
TAG=${1:-b1}; H=/home/fcy/claude-work/official-compat; UCC=/compat/profile/ubuntu_compat
docker exec official-compat-sim bash -c "source $UCC/env.sh >/dev/null; python3 \$UC/make_realmaze.py b" || exit 1     # newest prior b
X=$H/run/$TAG.extra.list; : > $X
for k in 1 2 3 4 5; do echo "rm$k /compat/realmaze/b/replica mission/course_prior.real_maze_20261108b.json - {\"stop_after\": \"tunnel_traverse\", \"nav_mode\": \"map\"}" >> $X; done
for k in 1 2 3; do echo "m24_$k /compat/maze/m24/replica ubuntu_compat/course_prior.maze24.blind.json - {\"stop_after\": \"corridor\", \"nav_mode\": \"map\"}" >> $X; done
bash $H/profile/ubuntu_compat/run_stages.sh $TAG --par 5 --stages tunnel,chwin --reps 5 --nav-mode prior --add $X
