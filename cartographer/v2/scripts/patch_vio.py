#!/usr/bin/python3
"""Isolated v2 detector fallback for plain CAD edges; no truth initialization."""
from pathlib import Path
root=Path('/work/cartographer/v2/ros_ws/src/open_vins')
p=root/'ov_core/src/track/Grider_GRID.h';s=p.read_text();needle='cv::FAST(img(img_roi), pts_new, threshold, nonmaxSuppression);'
replacement=needle+'''
                      // Plain CAD line corners can be missed by FAST. Use real
                      // Shi-Tomasi image corners in sparse cells, with the same
                      // masks, stereo verification and MSCKF outlier rejection.
                      if (pts_new.size() < static_cast<size_t>(num_features_grid)) {
                        std::vector<cv::Point2f> corners;
                        cv::goodFeaturesToTrack(img(img_roi), corners, num_features_grid, 0.005, 8);
                        for (const auto &corner : corners) {
                          pts_new.emplace_back(corner, 7.0f, -1.0f, 1.0f);
                        }
                      }'''
if 'Plain CAD line corners' not in s:
 assert needle in s;p.write_text(s.replace(needle,replacement))
p=root/'ov_init/src/init/InertialInitializer.cpp';s=p.read_text();s=s.replace('int feat_thresh = 15;','int feat_thresh = 8; // v2: eight actual tracked corners; stationary IMU checks remain')
p.write_text(s)
# An independent measured range change disables stationary-only ZUPT permanently.
# The message carries a boolean motion indication, never a position or truth pose.
p=root/'ov_msckf/src/core/VioManager.h';s=p.read_text()
if 'notify_external_motion' not in s:s=s.replace('double initialized_time() { return startup_time; }','double initialized_time() { return startup_time; }\n  void notify_external_motion() { has_moved_since_zupt = true; }')
p.write_text(s)
p=root/'ov_msckf/src/ros/ROS2Visualizer.h';s=p.read_text()
if 'external_motion_seen' not in s:
 s='#include <std_msgs/msg/bool.hpp>\n#include <atomic>\n'+s
 s=s.replace('rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;','rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;\n  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_motion;\n  std::atomic<bool> external_motion_seen{false};')
p.write_text(s)
p=root/'ov_msckf/src/ros/ROS2Visualizer.cpp';s=p.read_text()
if 'external_motion_seen.store' not in s:
 s=s.replace('void ROS2Visualizer::setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser) {','void ROS2Visualizer::setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser) {\n  sub_motion = _node->create_subscription<std_msgs::msg::Bool>("/vio/in_motion", 10, [this](std_msgs::msg::Bool::SharedPtr m) { if(m->data) external_motion_seen.store(true); });')
 s=s.replace('_app->feed_measurement_camera(camera_queue.at(0));','if (external_motion_seen.load()) _app->notify_external_motion();\n        _app->feed_measurement_camera(camera_queue.at(0));')
p.write_text(s)
# Count accepted MSCKF and persistent-landmark updates in the real update loop.
p=root/'ov_msckf/src/core/VioManager.h';s=p.read_text()
if 'visual_update_count()' not in s:s=s.replace('double initialized_time() { return startup_time; }','double initialized_time() { return startup_time; }\n  int visual_update_count() const { return accepted_visual_updates; }\n  double visual_update_time() const { return accepted_visual_time; }').replace('double startup_time = -1;','double startup_time = -1;\n  int accepted_visual_updates = 0;\n  double accepted_visual_time = -1;')
p.write_text(s)
p=root/'ov_msckf/src/core/VioManager.cpp';s=p.read_text()
if 'accepted_visual_updates = 0' not in s:
 s=s.replace('updaterMSCKF->update(state, featsup_MSCKF);','accepted_visual_updates = 0;\n  updaterMSCKF->update(state, featsup_MSCKF);\n  accepted_visual_updates += featsup_MSCKF.size();')
 s=s.replace('updaterSLAM->update(state, featsup_TEMP);','updaterSLAM->update(state, featsup_TEMP);\n    accepted_visual_updates += featsup_TEMP.size();')
 s=s.replace('updaterSLAM->delayed_init(state, feats_slam_DELAYED);','updaterSLAM->delayed_init(state, feats_slam_DELAYED);\n  if (accepted_visual_updates > 0) accepted_visual_time = message.timestamp;')
p.write_text(s)
p=root/'ov_msckf/src/ros/ROS2Visualizer.h';s=p.read_text()
if 'pub_visual_health' not in s:s='#include <std_msgs/msg/float64_multi_array.hpp>\n'+s;s=s.replace('rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_points_msckf,','rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_visual_health;\n  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_points_msckf,')
p.write_text(s)
p=root/'ov_msckf/src/ros/ROS2Visualizer.cpp';s=p.read_text()
if 'pub_visual_health->publish' not in s:
 s=s.replace('pub_points_msckf = node->create_publisher', 'pub_visual_health = node->create_publisher<std_msgs::msg::Float64MultiArray>("visual_update_health", 10);\n  pub_points_msckf = node->create_publisher')
 s=s.replace('void ROS2Visualizer::publish_features() {','void ROS2Visualizer::publish_features() {\n  std_msgs::msg::Float64MultiArray h;\n  h.data = {_app->get_state()->_timestamp, _app->visual_update_time(), static_cast<double>(_app->visual_update_count())};\n  pub_visual_health->publish(h);')
p.write_text(s)
