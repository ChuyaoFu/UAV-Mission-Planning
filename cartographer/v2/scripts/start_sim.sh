#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
# Gazebo and the bridge do not depend on the still-building VIO workspace.
set -u
PX4=/work/src/PX4-Autopilot
export GZ_SIM_RESOURCE_PATH=/work/sim/models:$PX4/Tools/simulation/gz/models
export GZ_SIM_SYSTEM_PLUGIN_PATH=$PX4/build/px4_sitl_default/src/modules/simulation/gz_plugins
export GZ_SIM_SERVER_CONFIG_PATH=/work/sim/server.config
export GZ_PARTITION=visual_nav_sim ROS_DOMAIN_ID=42
export PX4_UXRCE_DDS_PORT=8889
world_file=/work/sim/worlds/${NAV_WORLD:-dronehacks}.sdf
estimator=${NAV_ESTIMATOR:-gps}
model_file=/work/sim/models/x500_vio/model.sdf
px4_params='PX4_PARAM_EKF2_GPS_CTRL=7 PX4_PARAM_EKF2_EV_CTRL=0 PX4_PARAM_EKF2_MAG_TYPE=0 PX4_PARAM_EKF2_HGT_REF=1'
if [ "$estimator" = vision ]; then
    model_file=/work/cartographer/v2/config/model.sdf
    px4_params='PX4_PARAM_EKF2_GPS_CTRL=0 PX4_PARAM_EKF2_EV_CTRL=15 PX4_PARAM_EKF2_HGT_REF=3 PX4_PARAM_EKF2_MAG_TYPE=5 PX4_PARAM_EKF2_EV_POS_X=0.12 PX4_PARAM_EKF2_EV_POS_Y=0 PX4_PARAM_EKF2_EV_POS_Z=-0.055'
fi
spawn='sdf_filename: "/work/sim/models/x500_vio/model.sdf", name: "x500_vio_0", pose: {position: {x: -2.5146, y: 0.9144, z: 0.14}, orientation: {w: 0.7071067812, z: 0.7071067812}}'
if [ "${NAV_WORLD:-dronehacks}" = 'visual_nav' ]; then
    spawn='sdf_filename: "/work/sim/models/x500_vio/model.sdf", name: "x500_vio_0", pose: {position: {z: 0.14}}'
fi
spawn=${spawn/\/work\/sim\/models\/x500_vio\/model.sdf/$model_file}
if ! tmux has-session -t nav-gz 2>/dev/null; then
    tmux new-session -d -s nav-gz "source /opt/ros/jazzy/setup.bash; GZ_CONFIG_PATH=/usr/share/gz GZ_PARTITION=visual_nav_sim exec /usr/bin/gz sim -v 4 -r -s --headless-rendering $world_file > /work/cartographer/v2/logs/gazebo.log 2>&1"
    for i in $(seq 1 30); do
        if timeout 3 gz service -l | grep -q /world/visual_nav/create; then break; fi
        sleep 1
    done
    if ! timeout 3 gz service -l | grep -q /world/visual_nav/create; then
        echo 'Gazebo world did not become ready; see /work/cartographer/v2/logs/gazebo.log' >&2
        exit 1
    fi
    gz service -s /world/visual_nav/create --reqtype gz.msgs.EntityFactory \
        --reptype gz.msgs.Boolean --timeout 5000 \
        --req "$spawn"
fi
if ! tmux has-session -t nav-bridge 2>/dev/null; then
    tmux new-session -d -s nav-bridge 'source /opt/ros/jazzy/setup.bash; ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/work/config/bridge.yaml > /work/cartographer/v2/logs/bridge.log 2>&1'
fi
if [ "${1:-}" = '--sensors-only' ]; then exit 0; fi
tmux new-session -d -s nav-agent 'MicroXRCEAgent udp4 -p 8889 > /work/cartographer/v2/logs/agent.log 2>&1'
tmux new-session -d -s nav-px4 "cd /work/cartographer/v2/px4-rootfs && $px4_params PX4_PARAM_MPC_LAND_SPEED=0.2 PX4_PARAM_UXRCE_DDS_SYNCT=0 PX4_SYS_AUTOSTART=4001 PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=visual_nav PX4_GZ_MODEL_NAME=x500_vio_0 PX4_SIM_MODEL=gz_x500_vio $PX4/build/px4_sitl_default/bin/px4 -w /work/cartographer/v2/px4-rootfs > /work/cartographer/v2/logs/px4-run.log 2>&1"
echo 'Simulation started in nav-* tmux sessions. Domain 42; Agent UDP 8889.'
