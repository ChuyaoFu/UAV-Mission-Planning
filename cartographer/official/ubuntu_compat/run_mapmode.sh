#!/usr/bin/env bash
# The map-mode (maze planner) round, run on the Ubuntu HOST, detached, three sub-rounds in sequence (maze-only,
# 2 parallel, /team13/depth_scan on; OpenVINS on the OAK-model images in all):
#   DEPTH: A, A2, B = IDEAL rendered depth + mild noise (sigma 0.005 d^2, 3 % i.i.d. dropouts, no phantoms);
#          A3, C = stereo SGBM + net model/mask + noise filters (as v4K/v4R)
#   <TAG>A  10 random mazes, BLIND priors (offset walls + corridor; prior-seeded grid), nav_mode map
#   <TAG>A3  3 of the A mazes (3, 4, 24) on STEREO + filters, nav_mode map (sanity check)
#   <TAG>A2  3 mazes with an OUTDATED prior (another maze's walls: m5<-m3, m4<-m6, m24<-m1), nav_mode map
#   <TAG>B  v4 layout (warps w1-w3), blind2 EYEBALL priors, nav_mode map          (5 seeds)
#   <TAG>C  v4 layout (warps w1-w3), blind2 MAP-level priors, nav_mode prior       (5 seeds, regression)
#   bash run_mapmode.sh TAG         (normally started by go_mapmode.sh on the Mac)
set -u
TAG=${1:?TAG}; H=/home/fcy/claude-work/official-compat; R=$H/profile/ubuntu_compat/run_blind.sh
STACK="--noise-minsup 150,1.0 --sgbm-speckle 800,2 --depth-scan true"            # stereo (C, A3)
IDEAL="--depth-noise 0.005 --depth-dropout 0.03 --depth-scan true"             # rendered depth: sigma 0.005*d^2, 3 % dropouts (A, A2, B)
MAP='{"stop_after": "corridor", "nav_mode": "map"}'; PRI='{"stop_after": "corridor", "nav_mode": "prior"}'
bash $R ${TAG}A --maze-only --parallel 2 --depth ideal --images oak --stack "$IDEAL" --params "$MAP" --mazes "24 1 3 2 4 5 6 7 8 9" --kind blind
bash $R ${TAG}A2 --maze-only --parallel 2 --depth ideal --images oak --stack "$IDEAL" --params "$MAP" --mazes "5 4 24" --kind outdated
bash $R ${TAG}A3 --maze-only --parallel 2 --net-filter --stack "$STACK" --params "$MAP" --mazes "3 4 24" --kind blind
bash $R ${TAG}B --maze-only --parallel 2 --depth ideal --images oak --stack "$IDEAL" --params "$MAP" --set blind2 --levels eyeball --seeds "1 2 3 4 5"
bash $R ${TAG}C --maze-only --parallel 2 --net-filter --stack "$STACK" --params "$PRI" --set blind2 --levels map --seeds "1 2 3 4 5"
