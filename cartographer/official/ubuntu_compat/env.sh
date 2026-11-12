# official-compat profile environment (source inside container official-compat-sim)
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
[ -f /work/cartographer/v2/ros_ws/install/setup.bash ] && source /work/cartographer/v2/ros_ws/install/setup.bash
# parallel stacks: a second container sets OC_DOMAIN / OC_PARTITION (docker run -e ...), own /compat/run mount
export ROS_DOMAIN_ID=${OC_DOMAIN:-43} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export GZ_PARTITION=${OC_PARTITION:-official_compat} GZ_IP=127.0.0.1
export COMPAT=/compat PROFILE=/compat/profile RUN=/compat/run
export UC=$PROFILE/ubuntu_compat LOGS=$RUN/logs
export PX4=/work/src/PX4-Autopilot PX4_BUILD=/work/src/PX4-Autopilot/build/px4_sitl_default
export OV_PREFIX=/work/cartographer/v2/ros_ws/install
export GZ_SIM_RESOURCE_PATH=$RUN/models:/replica/world
export GZ_SIM_SYSTEM_PLUGIN_PATH=$PX4_BUILD/src/modules/simulation/gz_plugins
# the image ships only the Mesa EGL vendor file: force NVIDIA EGL or headless gz renders on the CPU
mkdir -p /tmp/egl && echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' > /tmp/egl/10_nvidia.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/tmp/egl/10_nvidia.json
# system python (rclpy); the px4 venv first in PATH has no ROS
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/ros/jazzy/bin
export PYTHONUNBUFFERED=1
