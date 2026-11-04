#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
export ROS_DOMAIN_ID=42
for name in /carto_course_mission /carto_vio_fusion /depth_cloud /passive_stereo /downward_tof /cartographer_node;do
 ros2 node info "$name"
done
ros2 topic info /vio/depth/image --verbose
px=/work/src/PX4-Autopilot/build/px4_sitl_default/bin
"$px/px4-param" show -q EKF2_GPS_CTRL
"$px/px4-param" show -q EKF2_EV_CTRL
"$px/px4-listener" sensor_gps -n 1
