#!/usr/bin/env bash
set -e
source /opt/ros/${ROS_DISTRO}/setup.bash
source /opt/plvins_ws/devel/setup.bash
exec "$@"
