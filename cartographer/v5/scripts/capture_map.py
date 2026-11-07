#!/usr/bin/python3
"""Read-only export of actual online occupancy and available optimized scans."""
import json,time,sys
from pathlib import Path
from collections import deque
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Int8MultiArray,Float64MultiArray
from cartographer_ros_msgs.srv import TrajectoryQuery
rclpy.init();n=Node('measured_map_export');q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL);cloud=deque(maxlen=1);frames=[];metadata=deque(maxlen=1);knowledge=deque(maxlen=1)
subs=[n.create_subscription(PointCloud2,'/mapping/global_cloud',lambda m:cloud.append(m),q),n.create_subscription(TFMessage,'/tf_static',lambda m:frames.extend(m.transforms),q),n.create_subscription(Float64MultiArray,'/grid_map/metadata',lambda m:metadata.append(list(m.data)),q),n.create_subscription(Int8MultiArray,'/grid_map/knowledge',lambda m:knowledge.append(np.asarray(m.data,dtype=np.int8)),q)];client=n.create_client(TrajectoryQuery,'/trajectory_query');end=time.monotonic()+8
while time.monotonic()<end and (not metadata or not knowledge or not frames or not cloud):rclpy.spin_once(n,timeout_sec=.05)
out=Path(sys.argv[1]);out.mkdir(exist_ok=True);summary={'online_map_available':bool(metadata and knowledge),'optimized_structure_available':bool(cloud),'reference_map_loaded':False}
if metadata and knowledge:np.savez_compressed(out/'online-occupancy.npz',metadata=metadata[-1],knowledge=knowledge[-1])
if cloud:
 m=cloud[-1];points=np.frombuffer(m.data,dtype='<f4').reshape(-1,3);np.save(out/'actual-optimized-structure.npy',points);summary['optimized_structure_points']=len(points)
v=next((t for t in frames if t.header.frame_id=='world' and t.child_frame_id=='map'),None)
if v:
 p=v.transform.translation;q=v.transform.rotation;(out/'measured-map-gauge.json').write_text(json.dumps({'frame':'map','world_from_map_translation':[p.x,p.y,p.z],'world_from_map_quaternion_xyzw':[q.x,q.y,q.z,q.w],'alignment':'first measured VIO/SLAM gauge only; no true world alignment'},indent=2))
if client.wait_for_service(timeout_sec=2.):
 req=TrajectoryQuery.Request();req.trajectory_id=0;future=client.call_async(req);rclpy.spin_until_future_complete(n,future,timeout_sec=5.)
 if future.done():
  poses=future.result().trajectory;(out/'optimized-trajectory.json').write_text(json.dumps([[p.header.stamp.sec+p.header.stamp.nanosec*1e-9,p.pose.position.x,p.pose.position.y,p.pose.position.z,p.pose.orientation.x,p.pose.orientation.y,p.pose.orientation.z,p.pose.orientation.w] for p in poses]))
(out/'map-export.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));n.destroy_node();rclpy.shutdown()
