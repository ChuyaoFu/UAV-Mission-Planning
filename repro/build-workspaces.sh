#!/usr/bin/env bash
# Runs inside the reproduction container. No host or physical-device setup.
set -eo pipefail
test -f /.dockerenv && test -f /work/repro/upstreams.lock.json
source /opt/ros/jazzy/setup.bash
jobs=${BUILD_JOBS:-2}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || exit 2
export CMAKE_BUILD_PARALLEL_LEVEL="$jobs" MAKEFLAGS="-j$jobs"
cd /work
git config --global --add safe.directory /work
mkdir -p logs data results
trap 'code=$?; if [ "$code" -ne 0 ]; then for file in /work/logs/repro-*-build.log; do if [ -f "$file" ]; then echo "Build diagnostic: $file"; tail -60 "$file"; fi; done; fi' EXIT
echo 'Fetching pinned sources and checking canonical patches'
python3 repro/fetch_sources.py
for project in src/PX4-Autopilot src/Micro-XRCE-DDS-Agent ros_ws/src/px4_msgs \
 cartographer/v2/ros_ws/src/open_vins cartographer/v3/carto_ws/src/cartographer \
 cartographer/v3/carto_ws/src/cartographer_ros; do
 git config --global --add safe.directory "/work/$project"
done
echo 'Compiling PX4 SITL (logs/repro-px4-build.log)'
(
 export PATH=/opt/px4-venv/bin:$PATH VIRTUAL_ENV=/opt/px4-venv PIP_CONSTRAINT=/opt/px4-constraints.txt
 cd /work/src/PX4-Autopilot
 make px4_sitl_default -j"$jobs"
) > logs/repro-px4-build.log 2>&1
echo 'Compiling Micro XRCE-DDS Agent (logs/repro-agent-build.log)'
cmake -S src/Micro-XRCE-DDS-Agent -B src/Micro-XRCE-DDS-Agent/build -DCMAKE_BUILD_TYPE=Release \
  > logs/repro-agent-build.log 2>&1
cmake --build src/Micro-XRCE-DDS-Agent/build -j"$jobs" >> logs/repro-agent-build.log 2>&1
cmake --install src/Micro-XRCE-DDS-Agent/build >> logs/repro-agent-build.log 2>&1
ldconfig
echo 'Compiling px4_msgs (logs/repro-msgs-build.log)'
(
 cd /work/ros_ws
 colcon build --symlink-install --packages-select px4_msgs --parallel-workers 1 \
   --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
) > logs/repro-msgs-build.log 2>&1
source /work/ros_ws/install/setup.bash
echo 'Compiling patched OpenVINS (logs/repro-vio-build.log)'
(
 cd /work/cartographer/v2/ros_ws
 colcon build --symlink-install --packages-up-to ov_msckf ov_eval --parallel-workers 1 \
   --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
) > logs/repro-vio-build.log 2>&1
echo 'Compiling patched Cartographer and ROS integration (logs/repro-cartographer-build.log)'
(
 cd /work/cartographer/v3/carto_ws
 colcon build --symlink-install --packages-up-to cartographer_ros --parallel-workers 1 \
   --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3 -DBUILD_GRPC=OFF
) > logs/repro-cartographer-build.log 2>&1
if [ ! -e cartographer/v4/carto_ws ]; then ln -s ../v3/carto_ws cartographer/v4/carto_ws; fi
test "$(readlink -f cartographer/v4/carto_ws)" = /work/cartographer/v3/carto_ws
for version in '' /v2 /v3 /v4; do mkdir -p "cartographer$version/"{logs,data,results}; done
python3 scripts/generate_sim.py
python3 scripts/generate_nogps_model.py
python3 scripts/import_course.py
# Include the optional historical scene, but the handoff defaults to source/stereo.
python3 scripts/add_regression_texture.py
python3 scripts/import_course.py --textured
python3 repro/record-build.py
echo 'Build complete. Next: bash repro/ctl.sh check, then bash repro/ctl.sh smoke.'
