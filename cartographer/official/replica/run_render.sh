#!/usr/bin/env bash
# One-shot headless render of the venue replica on the Ubuntu GPU host.
# Run ON the host:  WORK=/home/fcy/claude-work/venue-replica OUT=out [PASSES="full depth"] bash run_render.sh [LIMIT] [IDS]
# Resumable: frames already in $WORK/$OUT/raw_<pass>/ are skipped, so just rerun after a stall.
# Expects $WORK/replica/ = a copy of cartographer/official/replica (with world/ built).
# Throw-away container only: no --network host, no ROS stack, no sudo, mounts nothing but
# $WORK. It runs as container-root because the NVIDIA EGL device is not reachable for a
# plain --user uid (gz then silently falls back to CPU rendering); outputs are chown'ed
# back to the calling host user at the end. NVIDIA EGL is forced via __EGL_VENDOR_LIBRARY_FILENAMES. Output: $WORK/$OUT/raw_{full,depth}/*.npz + logs.
set -euo pipefail
WORK=${WORK:-/home/fcy/claude-work/venue-replica}
OUT=${OUT:-out}
IMG=${IMG:-visual-nav:jazzy-harmonic}
LIMIT=${1:-0}
IDS=${2:-}
PASSES=${PASSES:-full depth}
mkdir -p "$WORK/$OUT"
docker run --rm --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e GZ_PARTITION=venue_replica -e GZ_IP=127.0.0.1 \
  -e LIMIT="$LIMIT" -e IDS="$IDS" -e OUT="$OUT" -e PASSES="$PASSES" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$WORK":/work -w /work/replica/world "$IMG" bash -lc '
trap "chown -R $HOST_UID:$HOST_GID /work/$OUT" EXIT
set -e
# the image ships only the Mesa EGL vendor file -> headless gz silently renders on the CPU
# (llvmpipe). Point glvnd at the NVIDIA EGL library injected by --gpus.
mkdir -p /tmp/egl
echo "{\"file_format_version\": \"1.0.0\", \"ICD\": {\"library_path\": \"libEGL_nvidia.so.0\"}}" > /tmp/egl/10_nvidia.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/tmp/egl/10_nvidia.json
for P in $PASSES; do
  gz sim -s -r --headless-rendering /work/replica/world/venue_${P}.sdf > /work/$OUT/gz_${P}.log 2>&1 &
  python3 /work/replica/render_gz.py --pass $P --poses /work/replica/world/poses.json --out /work/$OUT \
      --limit "$LIMIT" --ids "$IDS"
  pkill -f "venue_${P}.sdf" || true
  sleep 2
done
grep -iE "error|unable|not found|llvmpipe|software" /work/$OUT/gz_full.log | sort | uniq -c | head -20 || true
'
