#!/usr/bin/env bash
# Copy team13_nav + the official stack onto the drone's team workspace and build it. Never touches ~/skybeta.
#
#   bash cartographer/official/realdrone/deploy.sh            # copy + build
#   NO_BUILD=1 bash .../deploy.sh                             # copy only (python edits need no rebuild: symlink-install)
#
# WHY NOT ~/team_ws/src: the stack's `competitor` window runs team_build.sh (colcon build over ALL of /root/team_ws,
# no --base-paths) whenever /root/team_ws/src holds a package.xml. ~/team_ws/oak-kalibr contains dozens of duplicate
# ROS1 catkin packages, so that build fails and the window drops to a shell: no flight_interface, no autonomy.
# So the package lives in ~/team_ws/team13_pkgs and is built with --base-paths team13_pkgs into the normal overlay
# (/root/team_ws/install, sourced by skybeta_env.sh): COMPETITOR_PKG=team13_nav still resolves.
set -euo pipefail
HOST="${DRONE_HOST:-dronehacks@skybeta-05.local}"
EXPECT="${DRONE_HOSTNAME:-skybeta-05}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL="$(cd "${HERE}/.." && pwd)"
DEST="team_ws/team13_pkgs/team13_nav"

h="$(ssh "${HOST}" hostname)"
[[ "${h}" == "${EXPECT}" ]] || { echo "[deploy] REFUSING: ${HOST} reports hostname '${h}', expected '${EXPECT}'" >&2; exit 1; }
ssh "${HOST}" "mkdir -p ${DEST}/official"
rsync -a --delete --exclude __pycache__ --exclude official "${HERE}/team13_nav/" "${HOST}:${DEST}/"
rsync -a --delete --exclude __pycache__ --exclude COLCON_IGNORE --exclude 'realdrone/team13_nav' \
      --exclude 'replica/world' --exclude 'replica/assets' --exclude docs --exclude flights --exclude '*.pdf' \
      --exclude '*.npz' --exclude '*.png' "${OFFICIAL}/" "${HOST}:${DEST}/official/"
ssh "${HOST}" "touch ${DEST}/official/COLCON_IGNORE; ls ${DEST} ${DEST}/official | tr '\n' ' '; echo"
if [[ -z "${NO_BUILD:-}" ]]; then
  ssh "${HOST}" "docker exec skybeta-stack bash -lc 'source skybeta_env.sh; team_build.sh --base-paths /root/team_ws/team13_pkgs --packages-select team13_nav'"
  ssh "${HOST}" "ls -ld ~/team_ws/install ~/team_ws/build ~/team_ws/log 2>/dev/null; ls ~/team_ws/install"
fi
echo "[deploy] ok -> ${HOST}:~/${DEST}"
