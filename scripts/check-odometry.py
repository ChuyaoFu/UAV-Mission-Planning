#!/usr/bin/python3
import json
import time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from px4_msgs.msg import VehicleLocalPosition
rclpy.init();node=Node('odometry_check');counts={};samples={}
def cb(k,m):
    counts[k]=counts.get(k,0)+1
    if k=='px4':samples[k]={'xyz':[m.x,m.y,m.z],'valid':[m.xy_valid,m.z_valid],'timestamp':m.timestamp}
    else:
        p=m.pose.pose.position;q=m.pose.pose.orientation
        samples[k]={'xyz':[p.x,p.y,p.z],'q_xyzw':[q.x,q.y,q.z,q.w],'stamp':m.header.stamp.sec+m.header.stamp.nanosec*1e-9}
subs=[node.create_subscription(Odometry,'/ov_msckf/odomimu',lambda m:cb('vio_odom',m),10),node.create_subscription(PoseWithCovarianceStamped,'/ov_msckf/poseimu',lambda m:cb('vio_pose',m),10),node.create_subscription(VehicleLocalPosition,'/fmu/out/vehicle_local_position_v1',lambda m:cb('px4',m),qos_profile_sensor_data)]
t=time.monotonic()
while time.monotonic()-t<8:rclpy.spin_once(node,timeout_sec=.05)
r={'counts':counts,'samples':samples};print(json.dumps(r,indent=2));Path('/work/results/odometry-check.json').write_text(json.dumps(r,indent=2))
node.destroy_node();rclpy.shutdown()
