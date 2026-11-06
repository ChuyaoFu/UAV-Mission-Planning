#!/usr/bin/python3
"""Read-only odometry benchmark recorder. No true pose input."""
import json,time,sys,rclpy,signal
from rclpy.signals import SignalHandlerOptions
from pathlib import Path
from nav_msgs.msg import Odometry
from rtabmap_msgs.msg import OdomInfo
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
stop=[False];signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True));rclpy.init(signal_handler_options=SignalHandlerOptions.NO);n=Node('rtab_odometry_benchmark',parameter_overrides=[Parameter('use_sim_time',value=True)]);poses=[];health=[]
def odom(m):
 p=m.pose.pose.position;q=m.pose.pose.orientation;poses.append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,p.x,p.y,p.z,q.x,q.y,q.z,q.w,max(m.pose.covariance[::7])])
def info(m):health.append({'stamp':m.header.stamp.sec+m.header.stamp.nanosec*1e-9,'inliers':m.inliers,'lost':m.lost,'features':m.features,'time_estimation':m.time_estimation})
subs=[n.create_subscription(Odometry,'/rtab/odom',odom,qos_profile_sensor_data),n.create_subscription(OdomInfo,'/rtab/odom_info',info,qos_profile_sensor_data)];end=time.monotonic()+float(sys.argv[2])
try:
 while time.monotonic()<end and not stop[0]:
  rclpy.spin_once(n,timeout_sec=.05)
  if poses and len(poses)%20==0:Path(sys.argv[1]).write_text(json.dumps({'poses':poses,'health':health}))
except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
Path(sys.argv[1]).write_text(json.dumps({'poses':poses,'health':health}));print('pose/info counts',len(poses),len(health));n.destroy_node();rclpy.try_shutdown()
