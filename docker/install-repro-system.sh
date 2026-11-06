#!/usr/bin/env bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive RUNS_IN_DOCKER=true
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git python3-venv
python3 -m venv --system-site-packages /opt/px4-venv
printf 'numpy<2\npandas<3\n' > /opt/px4-constraints.txt
export PATH=/opt/px4-venv/bin:$PATH VIRTUAL_ENV=/opt/px4-venv PIP_CONSTRAINT=/opt/px4-constraints.txt
python3 - <<'PY'
import hashlib,json,urllib.request
from pathlib import Path
lock=json.loads(Path('/opt/uav-repro/upstreams.lock.json').read_text())
revision=lock['repositories']['px4']['revision']
out=Path('/tmp/px4-setup');out.mkdir(exist_ok=True)
for name,digest in lock['px4_setup_files'].items():
    data=urllib.request.urlopen(f'https://raw.githubusercontent.com/PX4/PX4-Autopilot/{revision}/Tools/setup/{name}',timeout=60).read()
    assert hashlib.sha256(data).hexdigest()==digest,name
    (out/name).write_bytes(data)
PY
bash /tmp/px4-setup/ubuntu.sh --no-nuttx
apt-get update
apt-get install -y --no-install-recommends \
  ros-jazzy-ros-gz ros-jazzy-rviz2 ros-jazzy-cv-bridge ros-jazzy-image-transport \
  ros-jazzy-tf2-geometry-msgs ros-jazzy-tf2-eigen ros-jazzy-pcl-conversions \
  ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-rosbag2-storage-mcap \
  ros-jazzy-robot-state-publisher ros-jazzy-urdf \
  python3-colcon-common-extensions python3-rosdep python3-scipy \
  python3-matplotlib python3-opencv python3-psutil python3-yaml \
  python3-gz-transport13 python3-gz-msgs10 python3-sphinx \
  libboost-all-dev libceres-dev libopencv-contrib-dev libabsl-dev \
  libgflags-dev libgoogle-glog-dev liblua5.2-dev libcairo2-dev \
  libprotobuf-dev protobuf-compiler libpcl-dev libgtest-dev libgmock-dev \
  libasio-dev libtinyxml2-dev libssl-dev \
  mesa-utils libegl1 libgl1 libglx0 libgl1-mesa-dri libglx-mesa0 \
  xvfb xauth x11-utils tmux ffmpeg imagemagick wmctrl
/opt/px4-venv/bin/python3 -m pip install pyulog pymavlink
rm -rf /var/lib/apt/lists/* /tmp/px4-setup
