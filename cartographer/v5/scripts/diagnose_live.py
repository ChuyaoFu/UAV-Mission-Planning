#!/usr/bin/python3
"""Read-only measured-state and full-body clearance snapshot; no truth inputs."""
import json,time
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8MultiArray,Float64MultiArray,String
from online_map import OnlineMap
rclpy.init();n=Node('measured_navigation_diagnostic');data={};q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
subs=[n.create_subscription(Float64MultiArray,'/grid_map/metadata',lambda m:data.update(meta=list(m.data)),q),n.create_subscription(Int8MultiArray,'/grid_map/knowledge',lambda m:data.update(state=np.asarray(m.data,dtype=np.int8)),q),n.create_subscription(Odometry,'/nav/body_odom',lambda m:data.update(odom=m),5),n.create_subscription(String,'/carto/localization_health',lambda m:data.update(health=json.loads(m.data)),5)]
end=time.monotonic()+3
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
if all(k in data for k in ['meta','state','odom']):
 m=OnlineMap(data['meta'],data['state']);o=data['odom'];p=o.pose.pose.position;p=np.array([p.x,p.y,p.z]);lo=m.idx(p-m.extent);hi=m.idx(p+m.extent);indices=np.indices(tuple(hi-lo+1)).reshape(3,-1).T+lo;bad=[i for i in indices if not m.inside(i) or not m.free[tuple(i)]]
 print(json.dumps({'measured_body':p.tolist(),'measured_velocity_message':[o.twist.twist.linear.x,o.twist.twist.linear.y,o.twist.twist.linear.z],'full_body_clear':m.traversable(p),'component':m.component(p),'safe_cells':int(m.safe.sum()),'blocking_cells':[{'offset_body_world_m':(m.xyz(i)-p).tolist(),'state':int(m.state[tuple(i)]) if m.inside(i) else 'outside'} for i in bad[:20]],'blocking_cell_count':len(bad),'health':data.get('health')},indent=2))
else:raise RuntimeError('Measured map/odometry unavailable')
n.destroy_node();rclpy.shutdown()
