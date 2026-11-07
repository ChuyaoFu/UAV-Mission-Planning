#!/usr/bin/env bash
# Optional CPU OpenCV helper. Does not enable inference or start flight.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
package=$root/candidates/grabcut-ros2-node
cmake -S "$package/native" -B "$package/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$package/build" -j2
echo 'Optional structural helper built. Flight backend selection remains unchanged.'
