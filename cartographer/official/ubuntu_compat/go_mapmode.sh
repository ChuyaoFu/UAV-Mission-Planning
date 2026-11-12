#!/usr/bin/env bash
# ONE command for the map-mode round, run on the MAC from cartographer/official:
#   bash ubuntu_compat/go_mapmode.sh SHA256_PREFIX [TAG]       e.g. bash ubuntu_compat/go_mapmode.sh 1a2b3c4d mapP2
# 1. refuses unless mission/core.py on the Mac has that sha256 prefix (mission-core's announced planner sha);
# 2. syncs common/ mission/ node/ perception/ tools/ ubuntu_compat/ to the Ubuntu profile, installs fly_compat.sh atomically;
# 3. loads every maze prior (blind / coarse / outdated) and every blind2 prior with the NEW core and checks Params(nav_mode='map') exists;
# 4. starts run_mapmode.sh TAG detached on the host (sub-rounds TAGA / TAGB / TAGC, see there).
set -eu
SHA=${1:?sha256 prefix}; TAG=${2:-map$(date +%H%M)}
cd "$(dirname "$0")/.."
GOT=$(shasum -a 256 mission/core.py | cut -c1-${#SHA})
[ "$GOT" = "$SHA" ] || { echo "[go_mapmode] mission/core.py is $(shasum -a 256 mission/core.py | cut -c1-16), not $SHA: not flying"; exit 1; }
rsync -a --exclude __pycache__ --exclude '*.pyc' common mission node perception tools ubuntu_compat uav-ubuntu-ts:/home/fcy/claude-work/official-compat/profile/
ssh uav-ubuntu-ts "H=/home/fcy/claude-work/official-compat; cp \$H/profile/ubuntu_compat/fly_compat.sh \$H/.fly_compat.sh.new && mv -f \$H/.fly_compat.sh.new \$H/fly_compat.sh
  echo \"[go_mapmode] Ubuntu core \$(sha256sum \$H/profile/mission/core.py | cut -c1-16)\"
  docker exec official-compat-sim bash -c 'source /compat/profile/ubuntu_compat/env.sh; cd /compat/profile && python3 -c \"
import sys, json, glob; sys.path.insert(0, \\\".\\\"); from mission.core import CoursePrior, Params
fs = sorted(glob.glob(\\\"ubuntu_compat/course_prior.maze*.json\\\")) + sorted(glob.glob(\\\"ubuntu_compat/course_prior.blind2*.json\\\"))
ps = [CoursePrior.from_dict(json.load(open(f))) for f in fs]; Params(nav_mode=\\\"map\\\")
print(\\\"[go_mapmode]\\\", len(fs), \\\"priors load; nav_mode map ok\\\")\"' || { echo '[go_mapmode] priors / params check FAILED: not flying'; exit 1; }
  setsid nohup bash \$H/profile/ubuntu_compat/run_mapmode.sh $TAG > \$H/run/$TAG.mapmode.log 2>&1 < /dev/null &
  echo \"[go_mapmode] started $TAG (A blind mazes map-mode, A2 outdated mazes map-mode, B v4 eyeball map-mode, C v4 map-level prior-mode) -> \$H/run/${TAG}A.log ...\""
