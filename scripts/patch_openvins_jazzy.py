#!/usr/bin/env python3
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'ros_ws/src/open_vins'
for rel in ['ov_msckf/src/ros/ROS2Visualizer.h','ov_msckf/src/ros/ROSVisualizerHelper.h']:
    p=root/rel;s=p.read_text()
    old='#include <tf2_geometry_msgs/tf2_geometry_msgs.h>'
    if '#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)' not in s:
        s=s.replace(old,'#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)\n#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>\n#else\n'+old+'\n#endif')
    old='#include <image_transport/image_transport.h>'
    if '#if __has_include(<image_transport/image_transport.hpp>)' not in s:
        s=s.replace(old,'#if __has_include(<image_transport/image_transport.hpp>)\n#include <image_transport/image_transport.hpp>\n#else\n'+old+'\n#endif')
    old='#include <cv_bridge/cv_bridge.h>'
    if '#if __has_include(<cv_bridge/cv_bridge.hpp>)' not in s:
        s=s.replace(old,'#if __has_include(<cv_bridge/cv_bridge.hpp>)\n#include <cv_bridge/cv_bridge.hpp>\n#else\n'+old+'\n#endif')
    p.write_text(s)
# Publish the valid stationary initial state before the first moving MSCKF
# update. Keep upstream initialized() semantics unchanged for other consumers.
p=root/'ov_msckf/src/core/VioManager.h';s=p.read_text()
anchor='  bool initialized() { return is_initialized_vio && timelastupdate != -1; }'
if 'bool has_initial_state()' not in s:
    assert anchor in s
    s=s.replace(anchor,anchor+'\n\n  /// A valid initialized inertial state, including stationary initialization.\n  bool has_initial_state() { return is_initialized_vio; }')
    p.write_text(s)
p=root/'ov_msckf/src/ros/ROS2Visualizer.cpp';s=p.read_text()
s=s.replace('last_visualization_timestamp == _app->get_state()->_timestamp && _app->initialized()', 'last_visualization_timestamp == _app->get_state()->_timestamp && _app->has_initial_state()')
s=s.replace('if (!_app->initialized())\n    return;', 'if (!_app->has_initial_state())\n    return;')
s=s.replace('if (!_app->initialized() || (timestamp - _app->initialized_time()) < 1)', 'if (!_app->has_initial_state() || (timestamp - _app->initialized_time()) < 1)')
# Retain upstream default behavior, but expose a synchronous ROS2 mode for this
# benchmark so prediction and camera updates cannot access filter state together.
s=s.replace('  if (!thread_update_running)\n    visualize_odometry(message.timestamp);','  visualize_odometry(message.timestamp);')
s=s.replace('rclcpp::SensorDataQoS(),','rclcpp::SensorDataQoS().keep_last(200),')
p.write_text(s)
p=root/'ov_msckf/src/run_subscribe_msckf.cpp';s=p.read_text()
anchor='  params.use_multi_threading_subs = true;'
if 'get_parameter_or<bool>("processing_async"' not in s:
    assert anchor in s
    s=s.replace(anchor,anchor+'\n#if ROS_AVAILABLE == 2\n  node->get_parameter_or<bool>("processing_async", params.use_multi_threading_subs, true);\n#endif')
    p.write_text(s)
print('Applied ROS2 compatibility, stationary-state publication, and optional synchronous processing patches')
